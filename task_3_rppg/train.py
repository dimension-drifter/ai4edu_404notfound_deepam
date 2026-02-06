"""
Team: 404NotFound
Phase A: ResNet18 + BiLSTM + Attention for engagement classification
  - Task 1: Binary (Low/High engagement)
  - Task 2: 4-class (Distracted/Disengaged/Nominal/Highly Engaged)

Phase B: rPPG signal extraction using 3 algorithms
  - POS (Plane-Orthogonal-to-Skin)
  - TS-CAN (Temporal Shift Convolutional Attention Network)
  - EfficientPhys

Usage:
    python train.py --data_dir ./dataset/train --output_dir ./output
"""

import os
import sys
import glob
import copy
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import scipy.signal

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from PIL import Image
from tqdm import tqdm

# CONFIGURATION

FRAMES_PER_VIDEO = 20
FACE_SIZE = 224
RPPG_FACE_SIZE = 72
BATCH_SIZE = 4
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# FACE DETECTION

def get_face_detector():
    return mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

# PHASE A: FACE EXTRACTION FOR CLASSIFICATION

def extract_face_frames(video_path, face_detector, n_frames=FRAMES_PER_VIDEO, face_size=FACE_SIZE):
    """Extract n uniformly sampled frames, detect and crop faces."""
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
        face_crop = None

        results = face_detector.process(rgb)
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            mx, my = int(0.2 * bw), int(0.2 * bh)
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(w, x1 + bw + 2 * mx)
            y2 = min(h, y1 + bh + 2 * my)
            if (x2 - x1) > 30 and (y2 - y1) > 30:
                face_crop = rgb[y1:y2, x1:x2]

        if face_crop is None:
            cs = min(h, w) * 2 // 3
            cy, cx = h // 2, w // 2
            face_crop = rgb[cy - cs // 2:cy + cs // 2, cx - cs // 2:cx + cs // 2]

        face_crop = cv2.resize(face_crop, (face_size, face_size))
        faces.append(face_crop)

    cap.release()
    return np.array(faces) if len(faces) > 0 else None


# 
# GEOMETRIC FEATURES
# 

def extract_geometric_features(face_crop):
    """Extract 12-dim geometric features from a face crop."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    upper = gray[0:h // 3, :]
    middle = gray[h // 3:2 * h // 3, :]
    lower = gray[2 * h // 3:, :]
    left_half = gray[:, :w // 2]
    right_half = gray[:, w // 2:]
    eye_region = gray[h // 4:h // 2, w // 6:5 * w // 6]
    mouth_region = gray[2 * h // 3:5 * h // 6, w // 4:3 * w // 4]

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


# DATASET

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


class EngagementDataset(Dataset):
    def __init__(self, face_arrays, geo_arrays, labels, transform=None, n_frames=FRAMES_PER_VIDEO):
        self.face_arrays = face_arrays
        self.geo_arrays = geo_arrays
        self.labels = labels
        self.transform = transform
        self.n_frames = n_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        faces = self.face_arrays[idx]
        geos = self.geo_arrays[idx]

        n = len(faces)
        if n > self.n_frames:
            indices = np.linspace(0, n - 1, self.n_frames, dtype=int)
            faces = faces[indices]
            geos = geos[indices]
        while len(faces) < self.n_frames:
            faces = np.concatenate([faces, faces[-1:]], axis=0)
            geos = np.concatenate([geos, geos[-1:]], axis=0)

        img_tensors = []
        for face in faces:
            img = Image.fromarray(face)
            img = self.transform(img) if self.transform else T.ToTensor()(img)
            img_tensors.append(img)

        frames_tensor = torch.stack(img_tensors)
        geo_tensor = torch.tensor(geos, dtype=torch.float32)
        label = int(self.labels[idx])
        return frames_tensor, geo_tensor, label


# MODEL: ResNet18 + BiLSTM + Attention

class EngagementHybridModel(nn.Module):
    def __init__(self, num_classes=1, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.vis_dim = 512

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[6].parameters():
            param.requires_grad = True
        for param in self.backbone[7].parameters():
            param.requires_grad = True

        self.geo_proj = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(inplace=True),
        )

        fused_dim = self.vis_dim + 32
        self.lstm = nn.LSTM(
            input_size=fused_dim, hidden_size=128,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.lstm_out_dim = 256

        self.temporal_attn = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

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
        x_vis = frames.view(B * N, C, H, W)
        x_vis = self.backbone(x_vis).flatten(1)
        x_vis = x_vis.view(B, N, -1)

        x_geo = self.geo_proj(geos)
        x = torch.cat([x_vis, x_geo], dim=2)

        lstm_out, _ = self.lstm(x)
        attn_w = self.temporal_attn(lstm_out)
        attn_w = torch.softmax(attn_w, dim=1)
        pooled = (lstm_out * attn_w).sum(dim=1)
        return self.classifier(pooled)


# 
# TRAINING ENGINE
# 

def train_one_epoch(model, loader, criterion, optimizer, device, task='binary'):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for frames, geos, labels in loader:
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

        total_loss += loss.item() * frames.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg = 'binary' if task == 'binary' else 'macro'
    return (total_loss / n,
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average=avg, zero_division=0))


@torch.no_grad()
def evaluate(model, loader, criterion, device, task='binary'):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for frames, geos, labels in loader:
        frames, geos, labels = frames.to(device), geos.to(device), labels.to(device)
        logits = model(frames, geos)

        if task == 'binary':
            logits_s = logits.squeeze(-1)
            loss = criterion(logits_s, labels.float())
            preds = (torch.sigmoid(logits_s) > 0.5).long()
        else:
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

        total_loss += loss.item() * frames.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg = 'binary' if task == 'binary' else 'macro'
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=avg, zero_division=0)
    return total_loss / n, acc, f1


def train_model(model, train_loader, val_loader, task, device,
                num_epochs=35, lr=3e-4, weight_decay=5e-4, patience=10):
    """Full training loop with early stopping. Returns best model weights."""

    if task == 'binary':
        labels_arr = np.array([s[2] for s in train_loader.dataset])
        n0 = np.sum(labels_arr == 0)
        n1 = np.sum(labels_arr == 1)
        pw = torch.tensor([n0 / max(n1, 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        labels_arr = np.array([s[2] for s in train_loader.dataset])
        counts = np.bincount(labels_arr, minlength=4).astype(float)
        cw = 1.0 / (counts + 1e-6)
        cw = cw / cw.sum() * len(cw)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

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

    print(f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>5} | "
          f"{'VaLoss':>7} | {'VaAcc':>6} | {'VaF1':>5}")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        tl, ta, tf = train_one_epoch(model, train_loader, criterion, optimizer, device, task)
        vl, va, vf = evaluate(model, val_loader, criterion, device, task)
        scheduler.step()

        marker = ""
        if va > best_val_acc or (va == best_val_acc and vf > best_val_f1):
            best_val_acc, best_val_f1, best_epoch = va, vf, epoch
            patience_cnt = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            marker = " *"
        else:
            patience_cnt += 1

        print(f"{epoch:3d} | {tl:7.4f} | {ta:5.1%} | {tf:5.3f} | "
              f"{vl:7.4f} | {va:5.1%} | {vf:5.3f}{marker}")

        if patience_cnt >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"\n  Best: epoch={best_epoch}, acc={best_val_acc:.1%}, f1={best_val_f1:.3f}")
    model.load_state_dict(best_model_wts)
    return model, best_val_acc, best_val_f1


# 
# PHASE B: rPPG SIGNAL PROCESSING UTILITIES
# 

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.01)
    high = min(highcut / nyq, 0.99)
    if low >= high:
        return None, None
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(signal, lowcut=0.7, highcut=3.5, fs=30.0, order=2):
    if len(signal) < 15:
        return signal
    if fs <= 2 * lowcut:
        return signal - np.mean(signal)
    effective_highcut = min(highcut, fs * 0.49)
    if effective_highcut <= lowcut:
        return signal - np.mean(signal)
    b, a = butter_bandpass(lowcut, effective_highcut, fs, order=order)
    if b is None:
        return signal - np.mean(signal)
    try:
        return scipy.signal.filtfilt(b, a, signal)
    except Exception:
        try:
            return scipy.signal.lfilter(b, a, signal)
        except Exception:
            return signal - np.mean(signal)


def detrend_signal(signal, lambda_val=50):
    T = len(signal)
    if T < 5:
        return signal
    from scipy.signal import savgol_filter
    window = min(T // 2 * 2 + 1, 31)
    if window < 5:
        return signal - np.mean(signal)
    smoothed = savgol_filter(signal, window_length=window, polyorder=2)
    return signal - smoothed


def compute_bpm(bvp_signal, fs=30.0, lowcut=0.7, highcut=3.5):
    if len(bvp_signal) < 30:
        return 0.0
    windowed = bvp_signal * np.hanning(len(bvp_signal))
    n_fft = max(2048, 2 ** int(np.ceil(np.log2(len(windowed) * 4))))
    fft_vals = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mask = (freqs >= lowcut) & (freqs <= highcut)
    if not np.any(mask):
        return 0.0
    peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
    return peak_freq * 60.0


def compute_sqi(bvp_signal, fs=30.0):
    if len(bvp_signal) < 30:
        return 0.0
    windowed = bvp_signal * np.hanning(len(bvp_signal))
    n_fft = max(1024, 2 ** int(np.ceil(np.log2(len(windowed) * 2))))
    fft_vals = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    hr_power = np.sum(fft_vals[(freqs >= 0.7) & (freqs <= 3.5)])
    total_power = np.sum(fft_vals[(freqs >= 0.1) & (freqs <= 5.0)])
    if total_power < 1e-10:
        return 0.0
    return float(np.clip(hr_power / total_power, 0, 1))


# 
# PHASE B: rPPG ALGORITHM 1 — POS
# 

def rppg_pos(rgb_traces, fs=30.0, window_sec=1.6):
    """POS algorithm. Wang et al., IEEE TBME 2017."""
    T = len(rgb_traces)
    win_len = int(window_sec * fs)
    if win_len < 2:
        win_len = max(int(fs), 2)
    if win_len > T:
        win_len = T

    bvp = np.zeros(T)
    for t in range(0, T - win_len + 1):
        window = rgb_traces[t:t + win_len]
        mean_c = np.mean(window, axis=0)
        mean_c[mean_c < 1e-6] = 1.0
        Cn = window / mean_c

        S1 = Cn[:, 1] - Cn[:, 2]
        S2 = -2.0 * Cn[:, 0] + Cn[:, 1] + Cn[:, 2]

        alpha = np.std(S1) / (np.std(S2) + 1e-8)
        P = S1 + alpha * S2
        bvp[t:t + win_len] += P - np.mean(P)

    bvp = bvp / (np.max(np.abs(bvp)) + 1e-8)
    bvp = bandpass_filter(bvp, lowcut=0.7, highcut=3.5, fs=fs)
    return bvp


# 
# PHASE B: rPPG ALGORITHM 2 — TS-CAN
# 

class TemporalShift(nn.Module):
    def __init__(self, n_div=8):
        super().__init__()
        self.n_div = n_div

    def forward(self, x):
        B, C, T, H, W = x.shape
        chunk = C // self.n_div
        out = x.clone()
        out[:, :chunk, 1:, :, :] = x[:, :chunk, :-1, :, :]
        out[:, chunk:2 * chunk, :-1, :, :] = x[:, chunk:2 * chunk, 1:, :, :]
        return out


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.attn(x)


class TSCAN(nn.Module):
    """Temporal Shift Convolutional Attention Network. Liu et al., NeurIPS 2020."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.appearance = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AvgPool2d(2),
        )
        self.attention = AttentionBlock(64)
        self.motion_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d(2),
        )
        self.motion_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AvgPool2d(2),
        )
        self.temporal_shift = TemporalShift(n_div=8)
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        diff_frames = torch.zeros_like(frames)
        diff_frames[:, 1:] = frames[:, 1:] - frames[:, :-1]

        outputs = []
        for t in range(T):
            app_feat = self.appearance(frames[:, t])
            attn_feat = self.attention(app_feat)
            mot_feat = self.motion_conv1(diff_frames[:, t])
            mot_feat = self.motion_conv2(mot_feat)
            mot_feat = mot_feat * self.attention.attn(attn_feat)
            outputs.append(self.regressor(mot_feat))

        return torch.cat(outputs, dim=1)


# 
# PHASE B: rPPG ALGORITHM 3 — EfficientPhys
# 

class EfficientPhysModel(nn.Module):
    """EfficientPhys-inspired model. Liu et al., WACV 2023."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.Conv2d(16, 32, 1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AvgPool2d(2),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, diff_frames):
        B, T, C, H, W = diff_frames.shape
        x = diff_frames.view(B * T, C, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = x.view(B, T, 64).permute(0, 2, 1)
        return self.temporal_conv(x).squeeze(1)


# 
# PHASE B: VIDEO READER + rPPG EXTRACTION
# 

def extract_all_from_video(video_path, face_detector, face_size=72, detect_interval=15):
    """Single-pass: extracts RGB traces (for POS) and face crops (for DL models)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = 30.0

    rgb_means = []
    face_frames = []
    roi_box = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_count % detect_interval == 0:
            results = face_detector.process(rgb)
            if results.detections:
                det = results.detections[0]
                bbox = det.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                roi_box = (x1, y1, min(w, x1 + bw), min(h, y1 + bh))

        if roi_box is not None:
            fx1, fy1, fx2, fy2 = roi_box
            cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
            rw = max(1, (fx2 - fx1) * 3 // 10)
            rh = max(1, (fy2 - fy1) * 3 // 10)
            skin_roi = rgb[max(0, cy - rh):min(h, cy + rh), max(0, cx - rw):min(w, cx + rw)]
            rgb_means.append(np.mean(skin_roi, axis=(0, 1)) if skin_roi.size > 0 else np.zeros(3))

            face = rgb[fy1:fy2, fx1:fx2]
            if face.size > 0:
                face_frames.append(cv2.resize(face, (face_size, face_size)).astype(np.float32) / 255.0)
            else:
                face_frames.append(np.zeros((face_size, face_size, 3), dtype=np.float32))
        else:
            rgb_means.append(np.zeros(3))
            face_frames.append(np.zeros((face_size, face_size, 3), dtype=np.float32))

        frame_count += 1

    cap.release()
    if len(rgb_means) < 30:
        return None, None, fps
    return np.array(rgb_means, dtype=np.float64), np.array(face_frames, dtype=np.float32), fps


def run_dl_rppg(face_frames_arr, fps, device, model_class, use_diff=False, chunk_size=300):
    """Run a DL-based rPPG model on pre-extracted face frames."""
    T = len(face_frames_arr)
    if T < 30:
        return None

    if use_diff:
        input_arr = np.zeros_like(face_frames_arr)
        for i in range(1, T):
            mean_prev = np.mean(face_frames_arr[i - 1])
            if mean_prev > 1e-6:
                input_arr[i] = (face_frames_arr[i] - face_frames_arr[i - 1]) / mean_prev
            else:
                input_arr[i] = face_frames_arr[i] - face_frames_arr[i - 1]
    else:
        input_arr = face_frames_arr

    tensor = torch.from_numpy(input_arr).permute(0, 3, 1, 2).float().unsqueeze(0)

    model = model_class(in_channels=3).to(device)
    model.eval()
    bvp_chunks = []

    with torch.no_grad():
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk = tensor[:, start:end].to(device)
            bvp_chunks.append(model(chunk).cpu().numpy().flatten())
            del chunk
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    del model
    bvp = np.concatenate(bvp_chunks)
    bvp = detrend_signal(bvp)
    bvp = bandpass_filter(bvp, lowcut=0.7, highcut=3.5, fs=fps)
    bvp = bvp / (np.max(np.abs(bvp)) + 1e-8)
    return bvp


def extract_rppg_signals(video_path, face_detector, device):
    """Extract rPPG signals using all 3 algorithms. Returns dict of results."""
    rgb_traces, face_frames_arr, fps = extract_all_from_video(
        video_path, face_detector, face_size=RPPG_FACE_SIZE
    )

    results = {'fps': fps}

    # POS
    if rgb_traces is not None:
        try:
            bvp = rppg_pos(rgb_traces, fs=fps)
            results['pos'] = {'bvp': bvp, 'bpm': compute_bpm(bvp, fs=fps), 'sqi': compute_sqi(bvp, fs=fps)}
        except Exception:
            results['pos'] = None
    else:
        results['pos'] = None

    # TS-CAN
    if face_frames_arr is not None:
        try:
            bvp = run_dl_rppg(face_frames_arr, fps, device, TSCAN, use_diff=False)
            if bvp is not None:
                results['tscan'] = {'bvp': bvp, 'bpm': compute_bpm(bvp, fs=fps), 'sqi': compute_sqi(bvp, fs=fps)}
            else:
                results['tscan'] = None
        except Exception:
            results['tscan'] = None
    else:
        results['tscan'] = None

    # EfficientPhys
    if face_frames_arr is not None:
        try:
            bvp = run_dl_rppg(face_frames_arr, fps, device, EfficientPhysModel, use_diff=True)
            if bvp is not None:
                results['ephys'] = {'bvp': bvp, 'bpm': compute_bpm(bvp, fs=fps), 'sqi': compute_sqi(bvp, fs=fps)}
            else:
                results['ephys'] = None
        except Exception:
            results['ephys'] = None
    else:
        results['ephys'] = None

    return results


# 
# MAIN TRAINING PIPELINE
# 

def main():
    parser = argparse.ArgumentParser(description='Train engagement model + extract rPPG')
    parser.add_argument('--data_dir', type=str, default='./dataset/train', help='Training data directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--epochs_t1', type=int, default=35, help='Epochs for Task 1')
    parser.add_argument('--epochs_t2', type=int, default=40, help='Epochs for Task 2')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--skip_phase_a', action='store_true', help='Skip visual training')
    parser.add_argument('--skip_phase_b', action='store_true', help='Skip rPPG extraction')
    args = parser.parse_args()

    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)

    labels_file = os.path.join(args.data_dir, 'labels_train.xlsx')
    if not os.path.exists(labels_file):
        labels_file = os.path.join(args.data_dir, 'labels_train.csv')

    if os.path.exists(labels_file):
        if labels_file.endswith('.xlsx'):
            df = pd.read_excel(labels_file)
        else:
            df = pd.read_csv(labels_file)
    else:
        print(f"ERROR: Labels file not found in {args.data_dir}")
        sys.exit(1)

    print(f"Loaded {len(df)} entries from {labels_file}")

    # Locate videos
    video_paths = []
    for _, row in df.iterrows():
        vname = row['video']
        vpath = os.path.join(args.data_dir, vname)
        if os.path.exists(vpath):
            video_paths.append(vpath)
        else:
            found = glob.glob(os.path.join(args.data_dir, '**', vname), recursive=True)
            video_paths.append(found[0] if found else None)

    df['video_path'] = video_paths
    df = df[df['video_path'].notna()].reset_index(drop=True)

    # Labels
    df['binary_label'] = df['label'].apply(lambda x: 0 if x <= 0.33 else 1)
    label_map = {0.0: 0, 0.33: 1, 0.66: 2, 1.0: 3}
    df['multi_label'] = df['label'].map(label_map)
    df['subject_id'] = df['video'].apply(lambda x: x.split('_')[1])

    face_detector = get_face_detector()

    # 
    # PHASE A: Visual Classification Training
    # 
    if not args.skip_phase_a:
        
        print("PHASE A: EXTRACTING FACES AND FEATURES")
        

        all_faces = []
        all_geos = []
        valid_mask = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Face extraction"):
            faces = extract_face_frames(row['video_path'], face_detector)
            if faces is not None:
                geos = np.array([extract_geometric_features(f) for f in faces])
                all_faces.append(faces)
                all_geos.append(geos)
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        df = df[valid_mask].reset_index(drop=True)

        # Subject-independent split
        best_split = None
        for try_seed in range(SEED, SEED + 50):
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=try_seed)
            t_idx, v_idx = next(splitter.split(df, df['multi_label'], df['subject_id']))
            val_binary_classes = set(df.iloc[v_idx]['binary_label'].unique())
            val_classes = set(df.iloc[v_idx]['multi_label'].unique())
            if len(val_binary_classes) == 2 and len(val_classes) >= 3:
                best_split = (t_idx, v_idx)
                if len(val_classes) == 4:
                    break

        if best_split is None:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
            train_idx, val_idx = next(splitter.split(df, df['multi_label']))
        else:
            train_idx, val_idx = best_split

        # Build datasets
        def make_loaders(label_col, batch_size):
            train_faces = [all_faces[i] for i in train_idx]
            train_geos = [all_geos[i] for i in train_idx]
            train_labels = df.iloc[train_idx][label_col].values.astype(int)

            val_faces = [all_faces[i] for i in val_idx]
            val_geos = [all_geos[i] for i in val_idx]
            val_labels = df.iloc[val_idx][label_col].values.astype(int)

            train_ds = EngagementDataset(train_faces, train_geos, train_labels, train_transform)
            val_ds = EngagementDataset(val_faces, val_geos, val_labels, val_transform)

            class_counts = np.bincount(train_labels)
            sample_weights = (1.0 / (class_counts + 1e-6))[train_labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                      num_workers=0, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True)
            return train_loader, val_loader

        # Task 1: Binary
        
        print("TASK 1: BINARY CLASSIFICATION")
        
        t1_train, t1_val = make_loaders('binary_label', args.batch_size)
        model_t1 = EngagementHybridModel(num_classes=2, dropout=0.4).to(DEVICE)
        model_t1, acc_t1, f1_t1 = train_model(
            model_t1, t1_train, t1_val, 'binary', DEVICE,
            num_epochs=args.epochs_t1, lr=args.lr, patience=10
        )

        # Task 2: Multi-class (transfer from Task 1)
        
        print("TASK 2: MULTI-CLASS CLASSIFICATION")
        
        t2_train, t2_val = make_loaders('multi_label', args.batch_size)
        model_t2 = EngagementHybridModel(num_classes=4, dropout=0.5).to(DEVICE)

        # Transfer backbone + LSTM weights
        state_t1 = model_t1.state_dict()
        compatible = {k: v for k, v in state_t1.items() if not k.startswith('classifier.')}
        model_t2.load_state_dict(compatible, strict=False)
        print(f"Transferred {len(compatible)} layers from Task 1")

        model_t2, acc_t2, f1_t2 = train_model(
            model_t2, t2_train, t2_val, 'multi', DEVICE,
            num_epochs=args.epochs_t2, lr=args.lr, weight_decay=1e-3, patience=10
        )

    # 
    # PHASE B: rPPG Signal Extraction
    # 
    if not args.skip_phase_b:
        
        print("PHASE B: rPPG SIGNAL EXTRACTION")
        

        rppg_dir = os.path.join(args.output_dir, 'rppg_signals')
        os.makedirs(rppg_dir, exist_ok=True)

        rppg_results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="rPPG Extraction"):
            vname = os.path.splitext(row['video'])[0]
            sig_results = extract_rppg_signals(row['video_path'], face_detector, DEVICE)

            entry = {'video': row['video'], 'fps': sig_results['fps']}
            for algo in ['pos', 'tscan', 'ephys']:
                if sig_results[algo] is not None:
                    bvp = sig_results[algo]['bvp']
                    sig_path = os.path.join(rppg_dir, f"{vname}_{algo}.csv")
                    np.savetxt(sig_path, bvp, delimiter=',', header=f'bvp_{algo}', comments='')
                    entry[f'bpm_{algo}'] = round(sig_results[algo]['bpm'], 2)
                    entry[f'sqi_{algo}'] = round(sig_results[algo]['sqi'], 4)
                else:
                    entry[f'bpm_{algo}'] = 0.0
                    entry[f'sqi_{algo}'] = 0.0

            rppg_results.append(entry)

        df_rppg = pd.DataFrame(rppg_results)
        df_rppg.to_csv(os.path.join(rppg_dir, 'rppg_results.csv'), index=False)
        print(f"rPPG results saved to {rppg_dir}/")

    # 
    # SAVE MODEL CHECKPOINT
    # 
    checkpoint = {'config': {
        'frames_per_video': FRAMES_PER_VIDEO,
        'face_size': FACE_SIZE,
        'rppg_face_size': RPPG_FACE_SIZE,
        'model_architecture': 'ResNet18+BiLSTM+Attention',
        'rppg_algorithms': ['POS', 'TS-CAN', 'EfficientPhys'],
    }}

    if not args.skip_phase_a:
        checkpoint['task1'] = {
            'model_state_dict': model_t1.state_dict(),
            'accuracy': acc_t1,
            'f1_score': f1_t1,
            'threshold': 0.60,
        }
        checkpoint['task2'] = {
            'model_state_dict': model_t2.state_dict(),
            'accuracy': acc_t2,
            'f1_score': f1_t2,
        }

    save_path = os.path.join(args.output_dir, 'model.pth')
    torch.save(checkpoint, save_path, _use_new_zipfile_serialization=False)
    print(f"\nCheckpoint saved: {save_path} ({os.path.getsize(save_path) / 1e6:.1f} MB)")
    print("Training complete.")


if __name__ == '__main__':
    main()