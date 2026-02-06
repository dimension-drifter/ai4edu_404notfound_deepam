"""
    python inference.py --video_dir ./test_videos --model_path model.pth --rppg_only --rppg_output rppg_signals.csv
"""

import os
import sys
import glob
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
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

# CONFIGURATION

FRAMES_PER_VIDEO = 20
FACE_SIZE = 224
RPPG_FACE_SIZE = 72
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# MODEL DEFINITION (must match train.py exactly)

class EngagementHybridModel(nn.Module):
    def __init__(self, num_classes=1, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.vis_dim = 512

        self.geo_proj = nn.Sequential(nn.Linear(12, 32), nn.ReLU(inplace=True))

        fused_dim = self.vis_dim + 32
        self.lstm = nn.LSTM(
            input_size=fused_dim, hidden_size=128,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.lstm_out_dim = 256
        self.temporal_attn = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        out_dim = 1 if num_classes <= 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 128), nn.LayerNorm(128),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5), nn.Linear(64, out_dim)
        )

    def forward(self, frames, geos):
        B, N, C, H, W = frames.shape
        x_vis = self.backbone(frames.view(B * N, C, H, W)).flatten(1).view(B, N, -1)
        x_geo = self.geo_proj(geos)
        x = torch.cat([x_vis, x_geo], dim=2)
        lstm_out, _ = self.lstm(x)
        attn_w = torch.softmax(self.temporal_attn(lstm_out), dim=1)
        pooled = (lstm_out * attn_w).sum(dim=1)
        return self.classifier(pooled)


# rPPG MODELS (must match train.py exactly)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.attn(x)


class TSCAN(nn.Module):
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


class EfficientPhysModel(nn.Module):
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
        x = self.features(diff_frames.view(B * T, C, H, W))
        x = self.pool(x).flatten(1).view(B, T, 64).permute(0, 2, 1)
        return self.temporal_conv(x).squeeze(1)


# SIGNAL PROCESSING UTILITIES

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


def detrend_signal(signal):
    T = len(signal)
    if T < 5:
        return signal
    from scipy.signal import savgol_filter
    window = min(T // 2 * 2 + 1, 31)
    if window < 5:
        return signal - np.mean(signal)
    return signal - savgol_filter(signal, window_length=window, polyorder=2)


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
    return freqs[mask][np.argmax(fft_vals[mask])] * 60.0


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


def rppg_pos(rgb_traces, fs=30.0, window_sec=1.6):
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


# INFERENCE ENGINE

class EngagementPredictor:
    """Full inference pipeline for Phase A + Phase B."""

    def __init__(self, model_path, device=None):
        self.device = device or DEVICE
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = self.checkpoint.get('config', {})
        self.threshold = 0.60

        # Load Task 1 model
        self.model_t1 = None
        if 'task1' in self.checkpoint:
            self.model_t1 = EngagementHybridModel(num_classes=2).to(self.device)
            self.model_t1.load_state_dict(self.checkpoint['task1']['model_state_dict'])
            self.model_t1.eval()
            self.threshold = self.checkpoint['task1'].get('threshold', 0.60)

        # Load Task 2 model
        self.model_t2 = None
        if 'task2' in self.checkpoint:
            self.model_t2 = EngagementHybridModel(num_classes=4).to(self.device)
            self.model_t2.load_state_dict(self.checkpoint['task2']['model_state_dict'])
            self.model_t2.eval()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _extract_faces(self, video_path, n_frames=FRAMES_PER_VIDEO, face_size=FACE_SIZE):
        """Extract face frames for classification."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return None, None

        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        faces, geos = [], []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_crop = None

            results = self.face_detector.process(rgb)
            if results.detections:
                det = results.detections[0]
                bbox = det.location_data.relative_bounding_box
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                bw, bh = int(bbox.width * w), int(bbox.height * h)
                mx, my = int(0.2 * bw), int(0.2 * bh)
                x1, y1 = max(0, x1 - mx), max(0, y1 - my)
                x2, y2 = min(w, x1 + bw + 2 * mx), min(h, y1 + bh + 2 * my)
                if (x2 - x1) > 30 and (y2 - y1) > 30:
                    face_crop = rgb[y1:y2, x1:x2]

            if face_crop is None:
                cs = min(h, w) * 2 // 3
                cy, cx = h // 2, w // 2
                face_crop = rgb[cy - cs // 2:cy + cs // 2, cx - cs // 2:cx + cs // 2]

            face_crop = cv2.resize(face_crop, (face_size, face_size))
            faces.append(face_crop)

            # Geometric features
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            fh, fw = gray.shape
            upper = gray[0:fh // 3, :]
            middle = gray[fh // 3:2 * fh // 3, :]
            lower = gray[2 * fh // 3:, :]
            left_half = gray[:, :fw // 2]
            right_half = gray[:, fw // 2:]
            eye_region = gray[fh // 4:fh // 2, fw // 6:5 * fw // 6]
            mouth_region = gray[2 * fh // 3:5 * fh // 6, fw // 4:3 * fw // 4]
            geo = np.array([
                np.mean(upper) / 255.0, np.mean(middle) / 255.0, np.mean(lower) / 255.0,
                np.std(eye_region) / 255.0, np.std(mouth_region) / 255.0,
                np.mean(np.abs(left_half.astype(float) - cv2.flip(right_half, 1).astype(float))) / 255.0,
                np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 0)) / 255.0,
                np.mean(cv2.Sobel(gray, cv2.CV_64F, 0, 1)) / 255.0,
                np.std(upper) / 255.0, np.std(middle) / 255.0, np.std(lower) / 255.0,
                np.mean(cv2.Laplacian(gray, cv2.CV_64F)) / 255.0,
            ], dtype=np.float32)
            geos.append(geo)

        cap.release()
        if not faces:
            return None, None
        return np.array(faces), np.array(geos)

    def _extract_rppg_data(self, video_path, face_size=72, detect_interval=15):
        """Single-pass extraction for rPPG: RGB traces + face frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, 0.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1:
            fps = 30.0

        rgb_means, face_frames = [], []
        roi_box = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_count % detect_interval == 0:
                results = self.face_detector.process(rgb)
                if results.detections:
                    det = results.detections[0]
                    bbox = det.location_data.relative_bounding_box
                    x1 = max(0, int(bbox.xmin * w))
                    y1 = max(0, int(bbox.ymin * h))
                    bw, bh = int(bbox.width * w), int(bbox.height * h)
                    roi_box = (x1, y1, min(w, x1 + bw), min(h, y1 + bh))

            if roi_box is not None:
                fx1, fy1, fx2, fy2 = roi_box
                cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                rw = max(1, (fx2 - fx1) * 3 // 10)
                rh = max(1, (fy2 - fy1) * 3 // 10)
                skin = rgb[max(0, cy - rh):min(h, cy + rh), max(0, cx - rw):min(w, cx + rw)]
                rgb_means.append(np.mean(skin, axis=(0, 1)) if skin.size > 0 else np.zeros(3))

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
        return np.array(rgb_means, np.float64), np.array(face_frames, np.float32), fps

    def _run_dl_rppg(self, face_arr, fps, model_class, use_diff=False, chunk_size=300):
        """Run a DL rPPG model on face frames array."""
        T = len(face_arr)
        if T < 30:
            return None

        if use_diff:
            inp = np.zeros_like(face_arr)
            for i in range(1, T):
                m = np.mean(face_arr[i - 1])
                inp[i] = (face_arr[i] - face_arr[i - 1]) / m if m > 1e-6 else face_arr[i] - face_arr[i - 1]
        else:
            inp = face_arr

        tensor = torch.from_numpy(inp).permute(0, 3, 1, 2).float().unsqueeze(0)
        model = model_class(in_channels=3).to(self.device)
        model.eval()
        chunks = []

        with torch.no_grad():
            for s in range(0, T, chunk_size):
                e = min(s + chunk_size, T)
                chunks.append(model(tensor[:, s:e].to(self.device)).cpu().numpy().flatten())
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        del model
        bvp = np.concatenate(chunks)
        bvp = detrend_signal(bvp)
        bvp = bandpass_filter(bvp, lowcut=0.7, highcut=3.5, fs=fps)
        bvp = bvp / (np.max(np.abs(bvp)) + 1e-8)
        return bvp

    @torch.no_grad()
    def predict_engagement(self, video_path):
        """Run Phase A classification on a single video."""
        faces, geos = self._extract_faces(video_path)
        if faces is None:
            return {'video': os.path.basename(video_path), 'error': 'face_extraction_failed'}

        # Pad/trim to FRAMES_PER_VIDEO
        while len(faces) < FRAMES_PER_VIDEO:
            faces = np.concatenate([faces, faces[-1:]])
            geos = np.concatenate([geos, geos[-1:]])
        if len(faces) > FRAMES_PER_VIDEO:
            idx = np.linspace(0, len(faces) - 1, FRAMES_PER_VIDEO, dtype=int)
            faces, geos = faces[idx], geos[idx]

        imgs = torch.stack([self.transform(Image.fromarray(f)) for f in faces]).unsqueeze(0).to(self.device)
        geo_t = torch.tensor(geos, dtype=torch.float32).unsqueeze(0).to(self.device)

        result = {'video': os.path.basename(video_path)}

        if self.model_t1 is not None:
            logits = self.model_t1(imgs, geo_t).squeeze(-1)
            prob = torch.sigmoid(logits).item()
            result['binary_pred'] = int(prob >= self.threshold)
            result['binary_prob'] = round(prob, 4)
            result['binary_label'] = 'High' if result['binary_pred'] == 1 else 'Low'

        if self.model_t2 is not None:
            logits = self.model_t2(imgs, geo_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            labels_map = {0: 'Distracted', 1: 'Disengaged', 2: 'Nominal', 3: 'Highly_Engaged'}
            result['multi_pred'] = pred
            result['multi_label'] = labels_map[pred]
            result['multi_probs'] = {labels_map[i]: round(float(probs[i]), 4) for i in range(4)}

        return result

    def predict_rppg(self, video_path):
        """Run Phase B rPPG extraction on a single video."""
        rgb_traces, face_arr, fps = self._extract_rppg_data(video_path)

        result = {
            'video': os.path.basename(video_path),
            'fps': round(fps, 2),
            'n_frames': len(rgb_traces) if rgb_traces is not None else 0,
        }

        algorithms = {}

        # POS
        if rgb_traces is not None:
            try:
                bvp = rppg_pos(rgb_traces, fs=fps)
                algorithms['POS'] = {
                    'bvp': bvp.tolist(),
                    'bpm': round(compute_bpm(bvp, fs=fps), 2),
                    'sqi': round(compute_sqi(bvp, fs=fps), 4),
                }
            except Exception as e:
                algorithms['POS'] = {'error': str(e)}

        # TS-CAN
        if face_arr is not None:
            try:
                bvp = self._run_dl_rppg(face_arr, fps, TSCAN, use_diff=False)
                if bvp is not None:
                    algorithms['TS-CAN'] = {
                        'bvp': bvp.tolist(),
                        'bpm': round(compute_bpm(bvp, fs=fps), 2),
                        'sqi': round(compute_sqi(bvp, fs=fps), 4),
                    }
            except Exception as e:
                algorithms['TS-CAN'] = {'error': str(e)}

        # EfficientPhys
        if face_arr is not None:
            try:
                bvp = self._run_dl_rppg(face_arr, fps, EfficientPhysModel, use_diff=True)
                if bvp is not None:
                    algorithms['EfficientPhys'] = {
                        'bvp': bvp.tolist(),
                        'bpm': round(compute_bpm(bvp, fs=fps), 2),
                        'sqi': round(compute_sqi(bvp, fs=fps), 4),
                    }
            except Exception as e:
                algorithms['EfficientPhys'] = {'error': str(e)}

        result['algorithms'] = algorithms
        return result

    def predict_all(self, video_path):
        """Run both Phase A and Phase B on a single video."""
        engagement = self.predict_engagement(video_path)
        rppg = self.predict_rppg(video_path)
        engagement['rppg'] = rppg
        return engagement


# MAIN

def find_videos(path):
    """Find all video files in a path (file or directory)."""
    exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    if os.path.isfile(path):
        return [path]
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(path, f'*{ext}')))
        videos.extend(glob.glob(os.path.join(path, '**', f'*{ext}'), recursive=True))
    return sorted(set(videos))


def main():
    parser = argparse.ArgumentParser(description='Engagement + rPPG Inference')
    parser.add_argument('--video_path', type=str, help='Single video file')
    parser.add_argument('--video_dir', type=str, help='Directory of videos')
    parser.add_argument('--model_path', type=str, default='model.pth')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV for engagement')
    parser.add_argument('--rppg_output', type=str, default='rppg_signals.csv', help='Output CSV for rPPG')
    parser.add_argument('--rppg_json', type=str, default='rppg_signals.json', help='Output JSON with raw signals')
    parser.add_argument('--rppg_only', action='store_true', help='Only run rPPG extraction')
    parser.add_argument('--engagement_only', action='store_true', help='Only run engagement')
    args = parser.parse_args()

    if not args.video_path and not args.video_dir:
        print("ERROR: Provide --video_path or --video_dir")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        sys.exit(1)

    predictor = EngagementPredictor(args.model_path)
    print(f"Model loaded from {args.model_path}")
    print(f"Device: {predictor.device}")

    # Collect videos
    if args.video_path:
        videos = find_videos(args.video_path)
    else:
        videos = find_videos(args.video_dir)

    if not videos:
        print("ERROR: No videos found")
        sys.exit(1)

    print(f"Processing {len(videos)} videos...")

    engagement_results = []
    rppg_summary_results = []
    rppg_raw_signals = {}

    for vpath in tqdm(videos, desc="Inference"):
        vname = os.path.basename(vpath)

        # Phase A: Engagement
        if not args.rppg_only:
            try:
                eng = predictor.predict_engagement(vpath)
                engagement_results.append(eng)
            except Exception as e:
                engagement_results.append({'video': vname, 'error': str(e)})

        # Phase B: rPPG
        if not args.engagement_only:
            try:
                rppg = predictor.predict_rppg(vpath)

                # Summary row
                summary = {'video': vname, 'fps': rppg['fps'], 'n_frames': rppg['n_frames']}
                for algo_name, algo_data in rppg.get('algorithms', {}).items():
                    key = algo_name.lower().replace('-', '')
                    if 'error' not in algo_data:
                        summary[f'bpm_{key}'] = algo_data['bpm']
                        summary[f'sqi_{key}'] = algo_data['sqi']
                    else:
                        summary[f'bpm_{key}'] = 0.0
                        summary[f'sqi_{key}'] = 0.0
                rppg_summary_results.append(summary)

                # Raw signals for JSON
                raw_entry = {'video': vname, 'fps': rppg['fps']}
                for algo_name, algo_data in rppg.get('algorithms', {}).items():
                    if 'bvp' in algo_data:
                        raw_entry[algo_name] = {
                            'bvp': algo_data['bvp'],
                            'bpm': algo_data['bpm'],
                            'sqi': algo_data['sqi'],
                        }
                rppg_raw_signals[vname] = raw_entry

            except Exception as e:
                rppg_summary_results.append({'video': vname, 'error': str(e)})

    # Save engagement results
    if engagement_results:
        df_eng = pd.DataFrame(engagement_results)
        df_eng.to_csv(args.output, index=False)
        print(f"\nEngagement predictions saved to: {args.output}")
        print(df_eng.to_string(index=False))

    # Save rPPG summary CSV
    if rppg_summary_results:
        df_rppg = pd.DataFrame(rppg_summary_results)
        df_rppg.to_csv(args.rppg_output, index=False)
        print(f"\nrPPG summary saved to: {args.rppg_output}")

    # Save raw rPPG signals as JSON
    if rppg_raw_signals:
        with open(args.rppg_json, 'w') as f:
            json.dump(rppg_raw_signals, f, indent=2)
        print(f"rPPG raw signals saved to: {args.rppg_json}")

    print("\nInference complete.")


if __name__ == '__main__':
    main()