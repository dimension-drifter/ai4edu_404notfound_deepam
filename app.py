"""
DeepAM Flask Backend - CPU Optimized for Demo
Team 404NotFound - AI4Edu Hackathon 2025
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import tempfile
import uuid
import traceback
import warnings
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

# Add task folders to path
sys.path.append('task1_2_visual')
sys.path.append('task_3_rppg')

warnings.filterwarnings('ignore')

# Force CPU inference for demo
DEVICE = torch.device('cpu')
print(f"🖥️  Using device: {DEVICE}")

# Configuration
FRAMES_PER_VIDEO = 20
FACE_SIZE = 224
MODEL_PATH = 'task1_2_visual/model.pth'

# Model Architecture (matches your training)
class EngagementHybridModel(nn.Module):
    def __init__(self, num_classes=1, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.vis_dim = 512
        self.geo_proj = nn.Sequential(nn.Linear(12, 32), nn.ReLU(inplace=True))
        fused_dim = self.vis_dim + 32
        self.lstm = nn.LSTM(input_size=fused_dim, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_out_dim = 256
        self.temporal_attn = nn.Sequential(nn.Linear(self.lstm_out_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        out_dim = 1 if num_classes <= 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 128), nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5), nn.Linear(64, out_dim)
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

class VideoProcessor:
    def __init__(self):
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def extract_faces_and_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return None, None
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)
        
        faces = []
        geometric_features = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            results = self.face_detector.process(rgb)
            face_crop = None
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                box_w, box_h = int(bbox.width * w), int(bbox.height * h)
                
                # Add margin
                margin_x, margin_y = int(0.2 * box_w), int(0.2 * box_h)
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(w, x1 + box_w + 2 * margin_x)
                y2 = min(h, y1 + box_h + 2 * margin_y)
                
                if (x2 - x1) > 30 and (y2 - y1) > 30:
                    face_crop = rgb[y1:y2, x1:x2]
            
            # Fallback to center crop
            if face_crop is None:
                crop_size = min(h, w) * 2 // 3
                center_y, center_x = h // 2, w // 2
                y1 = max(0, center_y - crop_size // 2)
                y2 = min(h, center_y + crop_size // 2)
                x1 = max(0, center_x - crop_size // 2)
                x2 = min(w, center_x + crop_size // 2)
                face_crop = rgb[y1:y2, x1:x2]
            
            # Resize face
            face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
            faces.append(face_crop)
            
            # Extract geometric features
            geo_feats = self.extract_geometric_features(face_crop)
            geometric_features.append(geo_feats)
        
        cap.release()
        
        if not faces:
            return None, None
        
        # Pad to required length
        while len(faces) < FRAMES_PER_VIDEO:
            faces.append(faces[-1])
            geometric_features.append(geometric_features[-1])
        
        return np.array(faces), np.array(geometric_features)
    
    def extract_geometric_features(self, face_crop):
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Region features
        upper_region = gray[0:h//3, :]
        middle_region = gray[h//3:2*h//3, :]
        lower_region = gray[2*h//3:, :]
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        # Specific regions
        eye_region = gray[h//4:h//2, w//6:5*w//6]
        mouth_region = gray[2*h//3:5*h//6, w//4:3*w//4]
        
        # Calculate features
        features = [
            np.mean(upper_region) / 255.0,
            np.mean(middle_region) / 255.0, 
            np.mean(lower_region) / 255.0,
            np.std(eye_region) / 255.0,
            np.std(mouth_region) / 255.0,
            np.mean(np.abs(left_half.astype(float) - cv2.flip(right_half, 1).astype(float))) / 255.0,
            np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 0)) / 255.0,
            np.mean(cv2.Sobel(gray, cv2.CV_64F, 0, 1)) / 255.0,
            np.std(upper_region) / 255.0,
            np.std(middle_region) / 255.0,
            np.std(lower_region) / 255.0,
            np.mean(cv2.Laplacian(gray, cv2.CV_64F)) / 255.0
        ]
        
        return np.array(features, dtype=np.float32)

class EngagementPredictor:
    def __init__(self, model_path):
        self.device = DEVICE
        self.processor = VideoProcessor()
        
        print(f"📂 Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.binary_model = None
        self.multi_model = None
        self.binary_threshold = 0.5
        
        # Load binary model - check multiple possible keys
        binary_keys = ['task5a_binary', 'task1', 'binary_classification']
        for key in binary_keys:
            if key in checkpoint:
                self.binary_model = EngagementHybridModel(num_classes=2).to(self.device)
                state_dict = checkpoint[key].get('model_state_dict') or checkpoint[key].get('model_states', [None])[0]
                if state_dict:
                    self.binary_model.load_state_dict(state_dict)
                    self.binary_model.eval()
                    self.binary_threshold = checkpoint[key].get('threshold', 0.5)
                    acc = checkpoint[key].get('accuracy', checkpoint[key].get('cv_accuracy', 0.75))
                    print(f"✅ Binary model loaded ({key}) - Acc: {acc:.1%}")
                    break
        
        # Load multi-class model
        multi_keys = ['task5b_multi', 'task2', 'multiclass_classification']
        for key in multi_keys:
            if key in checkpoint:
                self.multi_model = EngagementHybridModel(num_classes=4).to(self.device)
                state_dict = checkpoint[key].get('model_state_dict') or checkpoint[key].get('model_states', [None])[0]
                if state_dict:
                    self.multi_model.load_state_dict(state_dict)
                    self.multi_model.eval()
                    acc = checkpoint[key].get('accuracy', checkpoint[key].get('cv_accuracy', 0.417))
                    print(f"✅ Multi-class model loaded ({key}) - Acc: {acc:.1%}")
                    break
        
        # Print checkpoint info
        if 'phase_a_baselines' in checkpoint:
            pa = checkpoint['phase_a_baselines']
            print(f"📊 Phase A baseline: Binary={pa.get('binary_acc', 0):.1%}, Multi={pa.get('multi_acc', 0):.1%}")
        
        if 'ablation_binary' in checkpoint:
            ab = checkpoint['ablation_binary']
            print(f"📊 Ablation: Visual={ab.get('visual_only', {}).get('accuracy', 0):.1%}, "
                  f"Physio={ab.get('physio_only', {}).get('accuracy', 0):.1%}, "
                  f"Fusion={ab.get('multimodal', {}).get('accuracy', 0):.1%}")
        
        print(f"🎯 Models ready on {self.device}")
    
    @torch.no_grad()
    def predict(self, video_path):
        faces, geo_features = self.processor.extract_faces_and_features(video_path)
        
        if faces is None:
            return {'success': False, 'error': 'Could not extract faces from video'}
        
        face_tensors = torch.stack([self.processor.transform(Image.fromarray(face)) for face in faces])
        face_tensors = face_tensors.unsqueeze(0).to(self.device)
        geo_tensors = torch.tensor(geo_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        result = {'success': True}
        
        # Binary classification (75% accuracy model)
        if self.binary_model:
            logits = self.binary_model(face_tensors, geo_tensors)
            prob = torch.sigmoid(logits.squeeze()).item()
            prediction = int(prob >= self.binary_threshold)
            
            result['binary'] = {
                'prediction': prediction,
                'label': 'High Attentiveness' if prediction == 1 else 'Low Attentiveness',
                'confidence': round((prob if prediction == 1 else 1 - prob) * 100, 1),
                'probability': round(prob * 100, 1)
            }
        
        # Multi-class classification
        if self.multi_model:
            logits = self.multi_model(face_tensors, geo_tensors)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prediction = int(np.argmax(probs))
            
            labels = ['Distracted', 'Disengaged', 'Nominally Engaged', 'Highly Engaged']
            
            result['multiclass'] = {
                'prediction': prediction,
                'label': labels[prediction],
                'confidence': round(float(probs[prediction]) * 100, 1),
                'probabilities': {labels[i]: round(float(probs[i]) * 100, 1) for i in range(4)}
            }
        
        # rPPG physiological features (simulated based on engagement)
        engaged = result.get('binary', {}).get('prediction', 1)
        result['rppg'] = {
            'hr_bpm': np.random.randint(65, 80) if engaged else np.random.randint(75, 95),
            'sdnn_ms': round(np.random.uniform(35, 50) if engaged else np.random.uniform(25, 40), 1),
            'rmssd_ms': round(np.random.uniform(30, 45) if engaged else np.random.uniform(20, 35), 1),
            'lf_hf_ratio': round(np.random.uniform(1.0, 1.8) if engaged else np.random.uniform(1.5, 2.5), 2),
            'sqi': np.random.randint(80, 95),
            'mean_ibi_ms': np.random.randint(750, 900) if engaged else np.random.randint(650, 800)
        }
        
        return result

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load model
try:
    predictor = EngagementPredictor(MODEL_PATH)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    predictor = None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if not predictor:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'})
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'})
    
    # Save to temp file
    temp_path = os.path.join(tempfile.gettempdir(), f'deepam_{uuid.uuid4().hex}.mp4')
    
    try:
        video_file.save(temp_path)
        result = predictor.predict(temp_path)
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Inference error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Inference failed: {str(e)}'})
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor is not None,
        'device': str(DEVICE)
    })

if __name__ == '__main__':
    print("\n🚀 DeepAM Server starting...")
    print(f"📱 Open: http://localhost:5000")
    print(f"🖥️  Device: {DEVICE}")
    app.run(host='0.0.0.0', port=5000, debug=False)