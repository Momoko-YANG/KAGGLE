# Biomass Inference

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import r2_score
from common_defs import *


# --- Dataset with Cleaning & TTA Support ---
class OctaTileTestDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, global_size=224, title_size=224, do_flip=False):
        self.df = df
        self.img_dir = img_dir
        self.do_flip = do_flip
        self.cleaner = ImageCleaner()  # Instantiate cleaner
        self.g_res = A.Compose([A.Resize(global_size, global_size), A.Normalize(), ToTensorV2()])
        self.t_res = A.Compose([A.Resize(title_size, title_size), A.Normalize(), ToTensorV2()])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = os.path.basename(str(row['image_path']))
        path = self.img_dir / fname
        if not path.exists():
            base, _ = os.path.splitext(fname)
            for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
                c = self.img_dir / (base + ext)
                if c.exists(): path = c; break
        
        img = cv2.imread(str(path))
        if img is None: img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Cleaning ---
        img = self.cleaner.process(img)

        # TTA: Horizontal Flip
        if self.do_flip: img = cv2.flip(img, 1)

        h, w, c = img.shape
        cx, cy = w//2, h//2

        # Global view
        g_ten = self.g_res(image=img)['image']

        # 8-Tile split logic
        img_L, img_R = img[:, :cx], img[:, cx:]
        sub_cy, sub_cx = cy, cx // 2
        tiles_raw = [
            img_L[0:sub_cy, 0:sub_cx], img_L[0:sub_cy, sub_cx:],
            img_L[sub_cy:, 0:sub_cx], img_L[sub_cy:, sub_cx:],
            img_R[0:sub_cy, 0:sub_cx], img_R[0:sub_cy, sub_cx:],
            img_R[sub_cy:, 0:sub_cx], img_R[sub_cy:, sub_cx:]
        ]

        t_tens = []
        for t in tiles_raw:
            if t.size == 0: t = np.zeros((224, 224, 3), dtype=np.uint8)
            t_tens.append(self.t_res(image=t)['image'])
        return g_ten, torch.stack(t_tens)
    
def run_phase2():
    print("[Step 2] Starting Biomass Prediction (Octa-Tile + State + Cleaning + TTA)...")
    INPUT_CSV = 'interim_meta.csv'
    if not os.path.exists(INPUT_CSV): raise FileNotFoundError(f"{INPUT_CSV} not found")
    meta_df = pd.read_csv(INPUT_CSV)

    # Prepare Metadata Features
    m_sin = np.sin(2 * np.pi * meta_df['pred_month'].values / 12)
    m_cos = np.cos(2 * np.pi * meta_df['pred_month'].values / 12)
    heights = meta_df['pred_height'].values * 0.01
    ndvis = meta_df['pred_ndvi'].values
    sp_cols = [c for c in meta_df.columns if 'pred_species' in c]
    st_cols = [c for c in meta_df.columns if 'pred_state' in c]

    if not st_cols: st_probs = np.full((len(meta_df), 6), 1/6)
    else: st_probs = meta_df[st_cols].values

    meta_features = np.column_stack([m_sin, m_cos, heights, ndvis, meta_df[sp_cols].values, st_probs]).astype(np.float32)
    n_meta = meta_features.shape[1]

    # Load Normalization Stats
    try:
        t_mean = np.load(PHASE2_WEIGHTS_DIR / 'total_mean.npy')
        t_std = np.load(PHASE2_WEIGHTS_DIR / 'total_std.npy')
    except:
        print("Warning: normalization stats not found. Using defaults.")
        t_mean = 3.65; t_std = 0.64

    tta_options = [False, True]
    tta_results = []

    # TTA Loop
    for do_flip in tta_options:
        print(f"Inference TTA (Flip={do_flip})...")
        ds = OctaTileTestDataset(meta_df, TEST_IMG_DIR, global_size=224, title_size=224, do_flip=do_flip)
        dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

        f_res_tot, f_res_grn, f_res_clv, f_res_ded = [], [], [], []

        # 5-Fold Ensemble
        for fold in range(5):
            path = PHASE2_WEIGHTS_DIR / f"fold{fold}.pth"
            if not path.exists():
                continue

            model = BiomassOctaTileModel(n_meta_features=n_meta, pretrained=False).to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()

            p_tot, p_grn, p_clv, p_dev = [], [], [], []
            with torch.no_grad():
                for i, (g_imgs, t_imgs) in enumerate(tqdm(dl, desc=f"Fold {fold}")):
                    g_imgs = g_imgs.to(DEVICE); t_imgs = t_imgs.to(DEVICE)
                    idx_start = i * dl.batch_size
                    batch_meta = torch.tensor(meta_features[idx_start: idx_start + g_imgs.size(0)]).to(DEVICE)

                    with torch.amp.autocast('cuda'):
                        p_total_raw, p_ratios = model(g_imgs, t_imgs, batch_meta)
                    
                    p_total_np = p_total_raw.float().cpu().numpy()
                    p_ratios_np = p_ratios.float().cpu().numpy()

                    # De-normalize Total
                    real_total = np.expm1((p_total_np * t_std) + t_mean).clip(0, None)

                    # Calculate Components
                    green = real_total * p_ratios_np[:, 0]
                    clover = real_total * p_ratios_np[:, 1]
                    dead = np.maximum(0, real_total - green - clover)
                    p_tot.append(np.atleast_1d(real_total))
                    p_grn.append(np.atleast_1d(green))
                    p_clv.append(np.atleast_1d(clover))
                    p_ded.append(np.atleast_1d(dead))
            
            f_res_tot.append(np.concatenate(p_tot))
            f_res_grn.append(np.concatenate(p_grn))
            f_res_clv.append(np.concatenate(p_clv))
            f_res_ded.append(np.concatenate(p_ded))

            del model; gc.collect(); torch.cuda.empty_cache()

        if f_res_tot:
            # Average across folds for this TTA
            tta_results.append((
                np.mean(f_res_tot, axis=0), np.median(f_res_grn, axis = 0)),
                (np.median(f_res_clv, axis=0), np.mean(f_res_ded, axis = 0)
            ))

    # Average across TTA
    if tta_results:
        final_total = np.mean([r[0] for r in tta_results], axis=0)
        final_green = np.mean([r[1] for r in tta_results], axis=0)
        final_clover = np.mean([r[2] for r in tta_results], axis=0)
        final_dead = np.mean([r[3] for r in tta_results], axis=0)

        # Consistency Check: Re-scale components to match Total
        sum_comp = final_green + final_clover + final_dead
        scale = final_total / (sum_comp + 1e-6)
        final_green *= scale; final_clover *= scale; final_dead *= scale
    
    else:
        final_total = np.zeros(len(meta_df))
        final_green = np.zeros(len(meta_df))
        final_clover = np.zeros(len(meta_df))
        final_dead = np.zeros(len(meta_df))
    
    output_df = meta_df.copy()
    output_df['pred_Total'] = final_total
    output_df['pred_Green'] = final_green
    output_df['pred_Dead'] = final_dead
    output_df['pred_Clover'] = final_clover

    output_df.to_csv('interim_biomass.csv', index=False)
    print(f"[Step 2] Finished.")

if __name__ == '__main__':
    run_phase2()

