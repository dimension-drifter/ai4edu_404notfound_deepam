#!/usr/bin/env python3
"""
Phase A Inference Script - Student Engagement Classification
Team: 404NotFound

This script runs inference on test videos using trained models.

Usage:
    python inference.py --video_path test.mp4 --model_path model.pth
    python inference.py --video_dir ./test_videos --model_path model.pth --output results.csv

Requirements:
    pip install torch torchvision opencv-python mediapipe pandas numpy
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

FRAMES_PER_VIDEO = 20
FACE_SIZE = 224
BINARY_THRESHOLD = 0.6  # Optimal threshold from training


# ============================================================
# MODEL ARCHITECTURE (Must match training)
# ============================================================

class EngagementHybridModel(nn.Module):
    """Hybrid model: ResNet-18 + Geometric Features + BiLSTM + Attention."""
    
    def __init__(self, num_classes=1, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes
        
        resnet = models.resnet18(weights=None)
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
            input_size=fused_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
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


# ============================================================
# PREPROCESSING
# ============================================================

class VideoProcessor:
    """Handles face detection, extraction, and preprocessing for inference."""
    
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def process_video(self, video_path, n_frames=FRAMES_PER_VIDEO, face_size=FACE_SIZE):
        """
        Extract faces and geometric features from video.
        Returns: (frames_tensor, geo_tensor) or (None, None) if failed
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None, None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return None, None
        
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        faces, geos = [], []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_crop = self._detect_and_crop(rgb, h, w)
            face_crop = cv2.resize(face_crop, (face_size, face_size))
            
            geo_feat = self._extract_geometric(face_crop)
            
            faces.append(face_crop)
            geos.append(geo_feat)
        
        cap.release()
        
        if len(faces) == 0:
            return None, None
        
        # Pad or truncate to fixed length
        while len(faces) < n_frames:
            faces.append(faces[-1])
            geos.append(geos[-1])
        
        faces = faces[:n_frames]
        geos = geos[:n_frames]
        
        # Convert to tensors
        img_tensors = [self.transform(Image.fromarray(f)) for f in faces]
        frames_tensor = torch.stack(img_tensors).unsqueeze(0)
        geo_tensor = torch.tensor(np.array(geos), dtype=torch.float32).unsqueeze(0)
        
        return frames_tensor, geo_tensor
    
    def _detect_and_crop(self, rgb, h, w):
        """Detect face and return cropped region."""
        results = self.detector.process(rgb)
        
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
                return rgb[y1:y2, x1:x2]
        
        cs = min(h, w) * 2 // 3
        cy, cx = h // 2, w // 2
        return rgb[cy - cs // 2:cy + cs // 2, cx - cs // 2:cx + cs // 2]
    
    def _extract_geometric(self, face_crop):
        """Extract 12-dim geometric features from face crop."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        upper = gray[0:h//3, :]
        middle = gray[h//3:2*h//3, :]
        lower = gray[2*h//3:, :]
        
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
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
# INFERENCE ENGINE
# ============================================================

class EngagementPredictor:
    """Runs inference using trained models."""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = VideoProcessor()
        
        # Load checkpoint
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load Task 1 model (Binary)
        self.model_binary = EngagementHybridModel(num_classes=2, dropout=0.4).to(self.device)
        self.model_binary.load_state_dict(checkpoint['task1']['model_state_dict'])
        self.model_binary.eval()
        self.binary_threshold = checkpoint['task1'].get('threshold', BINARY_THRESHOLD)
        
        # Load Task 2 model (Multi-class)
        self.model_multi = EngagementHybridModel(num_classes=4, dropout=0.5).to(self.device)
        self.model_multi.load_state_dict(checkpoint['task2']['model_state_dict'])
        self.model_multi.eval()
        
        # Label mappings
        self.binary_labels = ['Low Attentiveness', 'High Attentiveness']
        self.multi_labels = ['Distracted', 'Disengaged', 'Nominally Engaged', 'Highly Engaged']
        
        print(f"Models loaded successfully on {self.device}")
    
    @torch.no_grad()
    def predict_single(self, video_path):
        """
        Run inference on a single video.
        Returns dict with binary and multi-class predictions.
        """
        frames, geos = self.processor.process_video(video_path)
        
        if frames is None:
            return {
                'video': os.path.basename(video_path),
                'binary_class': -1,
                'binary_label': 'Error',
                'binary_confidence': 0.0,
                'multi_class': -1,
                'multi_label': 'Error',
                'multi_confidence': 0.0,
            }
        
        frames = frames.to(self.device)
        geos = geos.to(self.device)
        
        # Binary prediction
        logits_bin = self.model_binary(frames, geos).squeeze()
        prob_bin = torch.sigmoid(logits_bin).item()
        pred_bin = 1 if prob_bin >= self.binary_threshold else 0
        conf_bin = prob_bin if pred_bin == 1 else (1 - prob_bin)
        
        # Multi-class prediction
        logits_multi = self.model_multi(frames, geos)
        probs_multi = torch.softmax(logits_multi, dim=1).squeeze()
        pred_multi = probs_multi.argmax().item()
        conf_multi = probs_multi[pred_multi].item()
        
        return {
            'video': os.path.basename(video_path),
            'binary_class': pred_bin,
            'binary_label': self.binary_labels[pred_bin],
            'binary_confidence': round(conf_bin, 4),
            'multi_class': pred_multi,
            'multi_label': self.multi_labels[pred_multi],
            'multi_confidence': round(conf_multi, 4),
        }
    
    def predict_batch(self, video_paths, show_progress=True):
        """Run inference on multiple videos."""
        results = []
        iterator = tqdm(video_paths, desc="Inference") if show_progress else video_paths
        
        for vpath in iterator:
            result = self.predict_single(vpath)
            results.append(result)
        
        return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main(args):
    # Initialize predictor
    predictor = EngagementPredictor(args.model_path)
    
    # Collect video paths
    if args.video_path:
        video_paths = [args.video_path]
    elif args.video_dir:
        video_paths = glob.glob(os.path.join(args.video_dir, '*.mp4'))
        video_paths += glob.glob(os.path.join(args.video_dir, '*.avi'))
        video_paths += glob.glob(os.path.join(args.video_dir, '*.mov'))
        print(f"Found {len(video_paths)} videos in {args.video_dir}")
    else:
        print("Error: Provide --video_path or --video_dir")
        return
    
    if len(video_paths) == 0:
        print("No videos found")
        return
    
    # Run inference
    results_df = predictor.predict_batch(video_paths)
    
    # Display results
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total videos processed: {len(results_df)}")
    print(f"\nBinary Classification:")
    print(results_df['binary_label'].value_counts())
    print(f"\nMulti-class Classification:")
    print(results_df['multi_label'].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run engagement classification inference")
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to a single video file')
    parser.add_argument('--video_dir', type=str, default=None,
                        help='Path to directory containing videos')
    parser.add_argument('--model_path', type=str, default='./model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results CSV')
    args = parser.parse_args()
    
    main(args)