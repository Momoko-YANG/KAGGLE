import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from pathlib import Path

# === Configuration ===
DEBUG_MODE = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512
BATCH_SIZE = 2

# === Paths ===
if DEBUG_MODE:
    TEST_IMG_DIR = Path('/kaggle/input/csiro-biomass/train')
    TEST_CSV = Path('/kaggle/input/csiro-biomass/train.csv')
    print("RUNNING IN DEBUG MODE (Using train.csv)")
else:
    TEST_IMG_DIR = Path('/kaggle/input/csiro-biomass/test')
    TEST_CSV = Path('/kaggle/input/csiro-biomass/test.csv')

# Pre-trained Weights Paths(Adjust these to your dataset paths)
PHASE1_WEIGHTS_DIR = Path('/kaggle/input/csiro-meta-predictor-exp051')
PHASE2_WEIGHTS_DIR = Path('/kaggle/input/csiro-convnextv2-large-exp052')

# === Image Cleaner Class ===
class ImageCleaner:
    def __init__(self, crop_margin=0.02, bottom_crop=0.10):
        self.crop_margin = crop_margin
        self.bottom_crop = bottom_crop
    
    def process(self, img_rgb):
        # 1. Edge Erase & Dust Correction
        img = self._safe_crop(img_rgb)
        # 2. Remove the date (orange text)
        img = self._remove_date_stamp(img)
        # 3. Shadow and uneven lighting correction
        img = self._apply_clahe(img)
        return img
    
    def _safe_crop(self, img):
        h, w = img.shape[:2]
        y_start = int(h * self.crop_margin)
        y_end = int(h * (1.0 - self.bottom_crop))
        x_start = int(w * self.crop_margin)
        x_end = int(w * (1.0 - self.crop_margin))
        if y_end <= y_start or x_end <= x_start: return img
        return img[y_start:y_end, x_start:x_end]
    
    def _remove_date_stamp(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        kernel = np.ones((3, 3), np.unit8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        if np.sum(mask) > 0:
            img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return img
    
    def _apply_clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final
    
# --- Shared Model Definitions ---
# FiLM Generator for modulating features
class FiLMGenerator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        # Initialize gamma to 1 (identity modulation)
        self.fc.bias.data[:out_features] = 1.0
    
    def forward(self, x):
        params = self.fc(x)
        gamma, beta = torch.chunk(params, 2, dim = -1)
        return gamma, beta
    
# Phase 2 Model: Biomass Regression with Octa-Tile & FiLM
class BiomassOctaTileModel(nn.Module):
    def __init__(self, model_name='convnextv2_large.fcmae_ft_in22k_in1k', n_meta_features=13, 
                 pretrained= False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                                          global_pool='')
        in_chans = self.backbone.num_features

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.film_gen = FiLMGenerator(in_chans, in_chans)

        self.meta_embed = nn.Sequential(
            nn.Linear(n_meta_features, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, in_chans)
            )
        
        self.attn = nn.MultiheadAttention(embed_dim=in_chans, num_heads=8, batch_first=True)

        # Heads
        self.head_total = nn.Sequential(
            nn.Linear(in_chans, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        self.head_ratio = nn.Sequential(
            nn.Linear(in_chans, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, g_img, tiles, meta):
        # Global Branch
        g_feat = self.global_pool(self.backbone(g_img)).flatten(1)
        gamma, beta = self.film_gen(g_feat)

        # Local Tile Branch
        B, N, C, H, W = tiles.shape
        t_feat = self.global_pool(self.backbone(tiles.view(B*N, C, H, W)).flatten(1))
        t_feat = t_feat.view(B, N, -1)

        # FiLM Modulation
        mod_feat = t_feat * gamma.unsqueeze(1) + beta.unsqueeze(1)

        # Metadata Injection & Attention
        m_emb = self.meta_embed(meta).unsqueeze(1)
        tokens = torch.cat([m_emb, mod_feat], dim=1)

        out = self.attn(tokens, tokens, tokens)[0].mean(dim=1)

        return self.head_total(out).squeeze(), torch.sigmoid(self.head_ratio(out))

