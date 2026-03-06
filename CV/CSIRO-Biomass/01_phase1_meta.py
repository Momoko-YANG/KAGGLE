import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm 
import cv2
from tqdm import tqdm
from sklearn.metrics import r2_score
from common_defs import *

# --- Model Definition (Meta Split + State) ---
class MetaPredictorSplitModel(nn.Module):
    def __init__(self, model_name='convnextv2_base.fcmae_ft_in22k_in1k', 
                 num_species_classes=3, num_state_classes=6):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='avg')
        n_features = self.backbone.num_features

        # Heads for each metadata target
        self.head_month = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
        self.head_height = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.head_ndvi = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_species_classes)
        )

        self.head_species = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_species_classes)
        )

        self.head_state = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_state_classes)
        )
    
    def forward(self, x):
        # Process split images (Batch, Splits, C, H, W)
        B, N, C, H, W = x.shape
        x_flat = x.view(B * N, C, H, W)
        feat = self.backbone(x_flat)
        feat = feat.view(B, N, -1)
        feat = torch.mean(feat, dim=1) # Average features from both splits

        return (self.head_month(feat), self.head_height(feat), self.head_ndvi(feat),
                self.head_species(feat), self.head_state(feat))

# --- Split Dataset with Cleaning ---
class MetaSplitTestDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, img_size=512):
        self.df = df
        self.img_dir = img_dir
        self.cleaner = ImageCleaner() # Instantiate cleaner
        self.resize = A.Compose([A.Resize(img_size, img_size)])
        self.to_tensor = ToTensorV2()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = os.path.basename(str(row['img_path']))

        # Handle potential file extension mismatches
        path = self.img_dir / fname
        if not path.exists():
            base, _ = os.path.splitext(fname)
            for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
                c = self.img_dir / (base + ext)
                if c.exists(): path = c; break

        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # --- Cleaning ---
        img = self.cleaner.process(img)

        # Split Logic (Process Left and Right halves separately)
        h, w, c = img.shape
        cx = w // 2
        img_L = img[:, :cx]
        img_R = img[:, cx:]

        t_L = self.resize(image=img_L)['image']
        t_R = self.resize(image=img_R)['image']

        def _norm(im):
            im = self.to_tensor(image=im)['image']
            im = im.float() / 255.0
            return (im - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        return torch.stack([_norm(t_L), _norm(t_R)])
    
def run_phase1():
    print("[Step 1] Starting Meta Prediction (Split + State + Cleaning)...")
    if not TEST_CSV.exists(): raise FileNotFoundError("Input CSV not found")
    test_df = pd.read_csv(TEST_CSV)

    # Standardize image paths
    if 'image_path' not in test_df.columns:
        if 'sample_id' in test_df.columns:
            test_df['img_id_temp'] = test_df['sample_id'].astype(str).apply(lambda x: x.split('__')[0])
            test_df['image_path'] = test_df['img_id_temp'].apply(lambda x: f"{x}.jpg" if not x.lower().endswith(('.jpg', '.png', '.jpeg')) else x)
        
    test_df['base_id'] = test_df['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    cols_to_keep = ['image_path', 'base_id']
    unique_img_df = test_df[cols_to_keep].drop_duplicates(subset=['image_path']).reset_index(drop=True)

    ds = MetaSplitTestDataset(unique_img_df, TEST_IMG_DIR, IMG_SIZE)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Accumulators for predictions
    p1_months, p1_heights, p1_ndvis, p1_species, p1_states = [], [], [], [], []

    # 5-Fold Inference
    for fold in range(5):
        path = PHASE1_WEIGHTS_DIR / f"meta_model_fold{fold}.pth"
        if not path.exists(): continue

        model = MetaPredictorSplitModel().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()

        fm, fh, fn, fsp, fst = [], [], [], [], []

        with torch.no_grad():
            for imgs in tqdm(dl, desc=f"P1 Fold {fold}"):
                imgs = imgs.to(DEVICE)
                pm, ph, pn, psp, pst = model(imgs)

                fm.append(pm.cpu().numpy())
                fh.append(ph.cpu().numpy())
                fn.append(pn.cpu().numpy())
                fsp.append(torch.softmax(psp, dim=1).cpu().numpy())
                fst.append(torch.softmax(pst, dim=1).cpu().numpy())
        
        p1_months.append(np.concatenate(fm))
        p1_heights.append(np.concatenate(fh))
        p1_ndvis.append(np.concatenate(fn))
        p1_species.append(np.concatenate(fsp))
        p1_states.append(np.concatenate(fst))
        del model; gc.collect(); torch.cuda.empty_cache()
    
    if not p1_months: raise RuntimeError("Phase 1 Failed: No weights found.")

    # Average predictions
    avg_m_vec = np.mean(p1_months, axis=0)
    pred_h = np.mean(p1_heights, axis=0).flatten()
    pred_n = np.mean(p1_ndvis, axis=0).flatten()
    avg_s = np.mean(p1_species, axis=0)
    avg_st = np.mean(p1_states, axis=0)

    # Convert month vector back to integer
    angles = np.arctan2(avg_m_vec[:, 0], avg_m_vec[:, 1])
    angles = np.where(angles < 0, angles + 2*np.pi, angles)
    pred_months = np.round(angles / (2*np.pi)*12).astype(int)
    pred_months = np.where(pred_months == 0, 12, pred_months)

    # Compile results
    output_dict = {
    'base_id': unique_img_df['base_id'],
    'image_path': unique_img_df['image_path'],
    'pred_month': pred_months,
    'pred_height': pred_h,
    'pred_ndvi': pred_n,
    'pred_species_0': avg_s[:, 0], 'pred_species_1': avg_s[:, 1], 'pred_species_2': avg_s[:, 2]
    }

    for i in range(avg_st.shape[1]): output_dict[f'pred_state_{i}'] = avg_st[:, i]

    output_df = pd.DataFrame(output_dict)

    if DEBUG_MODE:
        gt_df = pd.read_csv(TEST_CSV)
        if 'Height_Ave_cm' in gt_df.columns:
            merged = pd.merge(output_df, gt_df[['image_path', 'Height_Ave_cm', 'NDVI']].drop_duplicates(),
                                on = 'image_path', how='left')
            print("\n [Phase 1 Validation Scores]")
            print(f"  - Height R2: {r2_score(merged['Height_Ave_cm'], merged['pred_height']):.4f}")
    
    output_df.to_csv('interim_meta.csv', index=False)
    print(f"[Step 1] Finished.")


if __name__ == '__main__':
    run_phase1()






        


