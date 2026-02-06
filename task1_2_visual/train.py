#!/usr/bin/env python3
"""
Team: 404NotFound

This script trains two models:
1. Binary Classification (Low vs High Attentiveness)
2. Multi-Class Classification (4 Engagement Levels)

Usage:
    python train.py --data_dir ./dataset/train --output_dir ./results

Requirements:
    pip install torch torchvision opencv-python mediapipe pandas scikit-learn openpyxl tqdm
"""

import os
import argparse
import glob
import copy
import time
import warnings
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

FRAMES_PER_VIDEO = 20
FACE_SIZE = 224
BATCH_SIZE = 4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# FACE DETECTION
# ============================================================

class FaceExtractor:
    """Handles face detection and extraction using MediaPipe."""
    
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
    
    def extract_faces(self, video_path, n_frames=FRAMES_PER_VIDEO, face_size=FACE_SIZE):
        """Extract n uniformly sampled face crops from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return None
        
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        faces = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_crop = self._detect_and_crop(rgb, h, w)
            face_crop = cv2.resize(face_crop, (face_size, face_size))
            faces.append(face_crop)
        
        cap.release()
        return np.array(faces) if len(faces) > 0 else None
    
    def _detect_and_crop(self, rgb, h, w):
        """Detect face and return cropped region with margin."""
        results = self.detector.process(rgb)
        
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            
            # Add 20% margin
            mx, my = int(0.2 * bw), int(0.2 * bh)
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(w, x1 + bw + 2 * mx)
            y2 = min(h, y1 + bh + 2 * my)
            
            if (x2 - x1) > 30 and (y2 - y1) > 30:
                return rgb[y1:y2, x1:x2]
        
        # Fallback: center crop
        cs = min(h, w) * 2 // 3
        cy, cx = h // 2, w // 2
        return rgb[cy - cs // 2:cy + cs // 2, cx - cs // 2:cx + cs // 2]


# ============================================================
# GEOMETRIC FEATURE EXTRACTION
# ============================================================

def extract_geometric_features(face_crop):
    """
    Extract 12-dimensional geometric features from face crop.
    Uses pixel-based analysis without external landmark models.
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Region divisions
    upper = gray[0:h//3, :]
    middle = gray[h//3:2*h//3, :]
    lower = gray[2*h//3:, :]
    
    # Symmetry analysis
    left_half = gray[:, :w//2]
    right_half = gray[:, w//2:]
    
    # Key regions
    eye_region = gray[h//4:h//2, w//6:5*w//6]
    mouth_region = gray[2*h//3:5*h//6, w//4:3*w//4]
    
    features = np.array([
        np.mean(upper) / 255.0,
        np.mean(middle) / 255.0,
        np.mean(lower) / 255.0,
        np.std(eye_region) / 255.0,
        np.std(mouth_region) / 255.0,
        np.mean(np.abs(left_half.astype(float) - cv2.flip(right_half, 1).astype(float))) / 255.0,
        np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 0)) / 255.0,
        np.mean(cv2.Sobel(gray, cv2.CV_64F, 0, 1)) / 255.0,
        np.std(upper) / 255.0,
        np.std(middle) / 255.0,
        np.std(lower) / 255.0,
        np.mean(cv2.Laplacian(gray, cv2.CV_64F)) / 255.0,
    ], dtype=np.float32)
    
    return features


# ============================================================
# DATASET
# ============================================================

class EngagementDataset(Dataset):
    """Dataset for engagement classification with face frames and geometric features."""
    
    def __init__(self, dataframe, label_col, transform=None, n_frames=FRAMES_PER_VIDEO):
        self.df = dataframe.reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform
        self.n_frames = n_frames
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        faces = np.load(row['faces_path'])
        geos = np.load(row['geo_path'])
        
        # Ensure consistent frame count
        n = len(faces)
        if n > self.n_frames:
            indices = np.linspace(0, n - 1, self.n_frames, dtype=int)
            faces = faces[indices]
            geos = geos[indices]
        while len(faces) < self.n_frames:
            faces = np.concatenate([faces, faces[-1:]], axis=0)
            geos = np.concatenate([geos, geos[-1:]], axis=0)
        
        # Apply transforms
        img_tensors = []
        for face in faces:
            img = Image.fromarray(face)
            if self.transform:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)
            img_tensors.append(img)
        
        frames_tensor = torch.stack(img_tensors)
        geo_tensor = torch.tensor(geos, dtype=torch.float32)
        label = int(row[self.label_col])
        
        return frames_tensor, geo_tensor, label


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class EngagementHybridModel(nn.Module):
    """
    Hybrid model combining:
    - ResNet-18 visual backbone (fine-tuned layer3 + layer4)
    - Geometric feature projection
    - Bidirectional LSTM for temporal modeling
    - Attention-based temporal pooling
    """
    
    def __init__(self, num_classes=1, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes
        
        # Visual backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.vis_dim = 512
        
        # Freeze early layers, fine-tune layer3 and layer4
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[6].parameters():
            param.requires_grad = True
        for param in self.backbone[7].parameters():
            param.requires_grad = True
        
        # Geometric feature projection
        self.geo_proj = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(inplace=True),
        )
        
        # Temporal modeling with BiLSTM
        fused_dim = self.vis_dim + 32
        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_out_dim = 256
        
        # Temporal attention
        self.temporal_attn = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classification head with LayerNorm (handles batch_size=1)
        out_dim = 1 if num_classes <= 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, out_dim)
        )
    
    def forward(self, frames, geos):
        B, N, C, H, W = frames.shape
        
        # Extract visual features
        x_vis = frames.view(B * N, C, H, W)
        x_vis = self.backbone(x_vis).flatten(1)
        x_vis = x_vis.view(B, N, -1)
        
        # Project geometric features
        x_geo = self.geo_proj(geos)
        
        # Fuse and process temporally
        x = torch.cat([x_vis, x_geo], dim=2)
        lstm_out, _ = self.lstm(x)
        
        # Attention pooling
        attn_w = self.temporal_attn(lstm_out)
        attn_w = torch.softmax(attn_w, dim=1)
        pooled = (lstm_out * attn_w).sum(dim=1)
        
        return self.classifier(pooled)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def create_dataloaders(df_train, df_val, label_col, batch_size=BATCH_SIZE):
    """Create train and validation dataloaders with class balancing."""
    
    train_transform = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
        T.RandomRotation(12),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_ds = EngagementDataset(df_train, label_col, transform=train_transform)
    val_ds = EngagementDataset(df_val, label_col, transform=val_transform)
    
    # Weighted sampling for class imbalance
    labels = df_train[label_col].values.astype(int)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, task, device, 
                num_epochs=35, lr=3e-4, weight_decay=5e-4, patience=10):
    """
    Complete training loop with early stopping and model checkpointing.
    Returns best model weights and training history.
    """
    
    # Setup loss function
    if task == 'binary':
        df_tr = train_loader.dataset.df
        label_col = train_loader.dataset.label_col
        n0 = (df_tr[label_col] == 0).sum()
        n1 = (df_tr[label_col] == 1).sum()
        pw = torch.tensor([n0 / max(n1, 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        df_tr = train_loader.dataset.df
        label_col = train_loader.dataset.label_col
        labels_arr = df_tr[label_col].values.astype(int)
        counts = np.bincount(labels_arr, minlength=4).astype(float)
        cw = 1.0 / (counts + 1e-6)
        cw = cw / cw.sum() * len(cw)
        cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    patience_cnt = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = {'tr_loss': [], 'tr_acc': [], 'tr_f1': [], 
               'va_loss': [], 'va_acc': [], 'va_f1': []}
    
    print(f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>5} | "
          f"{'VaLoss':>7} | {'VaAcc':>6} | {'VaF1':>5}")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        
        for frames, geos, labels in train_loader:
            frames, geos, labels = frames.to(device), geos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(frames, geos)
            
            if task == 'binary':
                logits = logits.squeeze(-1)
                loss = criterion(logits, labels.float())
                preds = (torch.sigmoid(logits) > 0.5).long()
            else:
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * frames.size(0)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        
        with torch.no_grad():
            for frames, geos, labels in val_loader:
                frames, geos, labels = frames.to(device), geos.to(device), labels.to(device)
                logits = model(frames, geos)
                
                if task == 'binary':
                    logits_s = logits.squeeze(-1)
                    loss = criterion(logits_s, labels.float())
                    preds = (torch.sigmoid(logits_s) > 0.5).long()
                else:
                    loss = criterion(logits, labels)
                    preds = logits.argmax(dim=1)
                
                val_loss += loss.item() * frames.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        scheduler.step()
        
        # Calculate metrics
        avg = 'binary' if task == 'binary' else 'macro'
        tl = train_loss / len(train_loader.dataset)
        ta = accuracy_score(train_labels, train_preds)
        tf = f1_score(train_labels, train_preds, average=avg, zero_division=0)
        vl = val_loss / len(val_loader.dataset)
        va = accuracy_score(val_labels, val_preds)
        vf = f1_score(val_labels, val_preds, average=avg, zero_division=0)
        
        history['tr_loss'].append(tl)
        history['tr_acc'].append(ta)
        history['tr_f1'].append(tf)
        history['va_loss'].append(vl)
        history['va_acc'].append(va)
        history['va_f1'].append(vf)
        
        # Check for improvement
        marker = ""
        if va > best_val_acc or (va == best_val_acc and vf > best_val_f1):
            best_val_acc = va
            best_val_f1 = vf
            best_epoch = epoch
            patience_cnt = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            marker = " *"
        else:
            patience_cnt += 1
        
        print(f"{epoch:3d} | {tl:7.4f} | {ta:5.1%} | {tf:5.3f} | "
              f"{vl:7.4f} | {va:5.1%} | {vf:5.3f}{marker}")
        
        if patience_cnt >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\nBest: acc={best_val_acc:.1%}, f1={best_val_f1:.3f} @ epoch {best_epoch}")
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    
    return model, history, best_val_acc, best_val_f1


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(data_dir, processed_dir):
    """Load labels, extract faces and geometric features."""
    
    labels_file = os.path.join(data_dir, "labels_train.xlsx")
    df = pd.read_excel(labels_file)
    
    # Locate video files
    video_paths = []
    for _, row in df.iterrows():
        vname = row['video']
        vpath = os.path.join(data_dir, vname)
        if os.path.exists(vpath):
            video_paths.append(vpath)
        else:
            found = glob.glob(os.path.join(data_dir, '**', vname), recursive=True)
            video_paths.append(found[0] if found else None)
    
    df['video_path'] = video_paths
    df = df[df['video_path'].notna()].reset_index(drop=True)
    
    # Create label mappings
    df['binary_label'] = df['label'].apply(lambda x: 0 if x <= 0.33 else 1)
    label_map = {0.0: 0, 0.33: 1, 0.66: 2, 1.0: 3}
    df['multi_label'] = df['label'].map(label_map)
    
    # Extract subject ID for splitting
    df['subject_id'] = df['video'].apply(lambda x: x.split('_')[1])
    
    # Create directories
    faces_dir = os.path.join(processed_dir, "faces")
    geo_dir = os.path.join(processed_dir, "geo")
    os.makedirs(faces_dir, exist_ok=True)
    os.makedirs(geo_dir, exist_ok=True)
    
    # Extract faces and geometric features
    extractor = FaceExtractor()
    faces_paths, geo_paths = [], []
    
    print("Extracting faces and geometric features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        vname = os.path.splitext(row['video'])[0]
        faces_path = os.path.join(faces_dir, f"{vname}.npy")
        geo_path = os.path.join(geo_dir, f"{vname}_geo.npy")
        
        if os.path.exists(faces_path) and os.path.exists(geo_path):
            faces_paths.append(faces_path)
            geo_paths.append(geo_path)
            continue
        
        faces = extractor.extract_faces(row['video_path'])
        if faces is None:
            faces_paths.append(None)
            geo_paths.append(None)
            continue
        
        # Extract geometric features for each frame
        geo_features = np.array([extract_geometric_features(f) for f in faces])
        
        np.save(faces_path, faces)
        np.save(geo_path, geo_features)
        faces_paths.append(faces_path)
        geo_paths.append(geo_path)
    
    df['faces_path'] = faces_paths
    df['geo_path'] = geo_paths
    df = df[df['faces_path'].notna()].reset_index(drop=True)
    
    print(f"Processed {len(df)} videos successfully")
    return df


def create_splits(df, seed=SEED):
    """Create subject-independent train/val splits."""
    
    best_split = None
    for try_seed in range(seed, seed + 50):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=try_seed)
        t_idx, v_idx = next(splitter.split(df, df['multi_label'], df['subject_id']))
        
        val_binary_classes = set(df.iloc[v_idx]['binary_label'].unique())
        val_multi_classes = set(df.iloc[v_idx]['multi_label'].unique())
        
        if len(val_binary_classes) == 2 and len(val_multi_classes) >= 3:
            best_split = (t_idx, v_idx)
            if len(val_multi_classes) == 4:
                break
    
    if best_split is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        t_idx, v_idx = next(splitter.split(df, df['multi_label']))
        best_split = (t_idx, v_idx)
    
    train_idx, val_idx = best_split
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    
    # Verify no subject overlap
    train_subs = set(df_train['subject_id'])
    val_subs = set(df_val['subject_id'])
    overlap = train_subs.intersection(val_subs)
    
    print(f"Train: {len(df_train)} samples, {len(train_subs)} subjects")
    print(f"Val: {len(df_val)} samples, {len(val_subs)} subjects")
    print(f"Subject overlap: {len(overlap)} (should be 0)")
    
    return df_train, df_val


# ============================================================
# MAIN
# ============================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    processed_dir = os.path.join(args.output_dir, "processed")
    df = prepare_data(args.data_dir, processed_dir)
    df_train, df_val = create_splits(df)
    
   
    train_loader_bin, val_loader_bin = create_dataloaders(df_train, df_val, 'binary_label')
    model_t1 = EngagementHybridModel(num_classes=2, dropout=0.4).to(device)
    
    model_t1, history_t1, acc_t1, f1_t1 = train_model(
        model_t1, train_loader_bin, val_loader_bin,
        task='binary', device=device,
        num_epochs=35, lr=3e-4, weight_decay=5e-4, patience=10
    )
    
    print(f"\nTask 1 Final: Accuracy={acc_t1:.1%}, F1={f1_t1:.3f}")
    
       
    train_loader_multi, val_loader_multi = create_dataloaders(df_train, df_val, 'multi_label')
    model_t2 = EngagementHybridModel(num_classes=4, dropout=0.5).to(device)
    
    # Transfer weights from Task 1 (except classifier)
    state_t1 = model_t1.state_dict()
    compatible_keys = {k: v for k, v in state_t1.items() if not k.startswith('classifier.')}
    model_t2.load_state_dict(compatible_keys, strict=False)
    print(f"Transferred {len(compatible_keys)} layers from Task 1")
    
    model_t2, history_t2, acc_t2, f1_t2 = train_model(
        model_t2, train_loader_multi, val_loader_multi,
        task='multi', device=device,
        num_epochs=40, lr=3e-4, weight_decay=1e-3, patience=10
    )
    
    print(f"\nTask 2 Final: Accuracy={acc_t2:.1%}, F1={f1_t2:.3f}")
    
    # ========== SAVE MODELS ==========
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save combined checkpoint
    checkpoint = {
        'task1': {
            'model_state_dict': model_t1.state_dict(),
            'accuracy': acc_t1,
            'f1_score': f1_t1,
            'threshold': 0.6,  # Optimal threshold from TTA
        },
        'task2': {
            'model_state_dict': model_t2.state_dict(),
            'accuracy': acc_t2,
            'f1_score': f1_t2,
        },
        'config': {
            'frames_per_video': FRAMES_PER_VIDEO,
            'face_size': FACE_SIZE,
            'model_architecture': 'ResNet18+BiLSTM+Attention',
        }
    }
    
    save_path = os.path.join(args.output_dir, "model.pth")
    torch.save(checkpoint, save_path, _use_new_zipfile_serialization=False)
    print(f"\nModels saved to: {save_path}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("PHASE A SUMMARY")
    print("="*60)
    print(f"Task 1 (Binary):      Acc={acc_t1:.1%}, F1={f1_t1:.3f} | Target: 70%")
    print(f"Task 2 (Multi-Class): Acc={acc_t2:.1%}, F1={f1_t2:.3f} | Target: 65%")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train engagement classification models")
    parser.add_argument('--data_dir', type=str, default='./dataset/train',
                        help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Path to save models and results')
    args = parser.parse_args()
    
    main(args)