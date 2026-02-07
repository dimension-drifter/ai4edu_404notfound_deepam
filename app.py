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
import scipy.signal
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

sys.path.append('task1_2_visual')
sys.path.append('task_3_rppg')

warnings.filterwarnings('ignore')

DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

FRAMES_PER_VIDEO = 20
FACE_SIZE = 224
MODEL_PATH = 'model.pth'


# ============================================================
# rPPG SIGNAL PROCESSING (from train.py - real algorithms)
# ============================================================

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
    """POS algorithm - Wang et al., IEEE TBME 2017."""
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


def compute_hrv_features(bvp_signal, fs=30.0):
    """Extract HRV features from BVP signal."""
    if len(bvp_signal) < 60:
        return {
            'hr_bpm': 0.0, 'sdnn_ms': 0.0, 'rmssd_ms': 0.0,
            'lf_hf_ratio': 0.0, 'sqi': 0.0, 'mean_ibi_ms': 0.0
        }

    bpm = compute_bpm(bvp_signal, fs=fs)
    sqi = compute_sqi(bvp_signal, fs=fs)

    # Find peaks for IBI calculation
    min_distance = int(fs * 0.4)  # Min 0.4s between beats
    peaks, _ = scipy.signal.find_peaks(bvp_signal, distance=max(min_distance, 1))

    if len(peaks) < 3:
        return {
            'hr_bpm': round(bpm, 1),
            'sdnn_ms': 0.0, 'rmssd_ms': 0.0,
            'lf_hf_ratio': 0.0,
            'sqi': round(sqi * 100, 1),
            'mean_ibi_ms': round(60000.0 / bpm, 1) if bpm > 0 else 0.0
        }

    # Inter-beat intervals in ms
    ibis = np.diff(peaks) / fs * 1000.0

    # Filter physiologically plausible IBIs (300ms to 1500ms = 40-200 BPM)
    valid_ibis = ibis[(ibis > 300) & (ibis < 1500)]
    if len(valid_ibis) < 2:
        valid_ibis = ibis

    mean_ibi = np.mean(valid_ibis)
    sdnn = np.std(valid_ibis)

    # RMSSD
    successive_diffs = np.diff(valid_ibis)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2)) if len(successive_diffs) > 0 else 0.0

    # LF/HF ratio from frequency domain
    n_fft = max(1024, 2 ** int(np.ceil(np.log2(len(bvp_signal) * 2))))
    fft_vals = np.abs(np.fft.rfft(bvp_signal, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    lf_power = np.sum(fft_vals[(freqs >= 0.04) & (freqs <= 0.15)])
    hf_power = np.sum(fft_vals[(freqs >= 0.15) & (freqs <= 0.4)])
    lf_hf = lf_power / (hf_power + 1e-10)

    return {
        'hr_bpm': round(bpm, 1),
        'sdnn_ms': round(sdnn, 1),
        'rmssd_ms': round(rmssd, 1),
        'lf_hf_ratio': round(float(np.clip(lf_hf, 0, 10)), 2),
        'sqi': round(sqi * 100, 1),
        'mean_ibi_ms': round(mean_ibi, 1)
    }


def extract_rppg_from_video(video_path, face_detector):
    """Extract real rPPG signal from video using POS algorithm."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = 30.0

    rgb_means = []
    roi_box = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Re-detect face every 15 frames
        if frame_count % 15 == 0:
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
            skin = rgb[max(0, cy - rh):min(h, cy + rh), max(0, cx - rw):min(w, cx + rw)]
            if skin.size > 0:
                rgb_means.append(np.mean(skin, axis=(0, 1)))
            else:
                rgb_means.append(np.zeros(3))
        else:
            rgb_means.append(np.zeros(3))

        frame_count += 1

    cap.release()

    if len(rgb_means) < 30:
        return None, fps

    rgb_traces = np.array(rgb_means, dtype=np.float64)
    bvp = rppg_pos(rgb_traces, fs=fps)
    return bvp, fps


# ============================================================
# MODEL ARCHITECTURE (same as training)
# ============================================================

class EngagementHybridModel(nn.Module):
    def __init__(self, num_classes=1, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.vis_dim = 512
        self.geo_proj = nn.Sequential(nn.Linear(12, 32), nn.ReLU(inplace=True))
        fused_dim = self.vis_dim + 32
        self.lstm = nn.LSTM(input_size=fused_dim, hidden_size=128, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.lstm_out_dim = 256
        self.temporal_attn = nn.Sequential(nn.Linear(self.lstm_out_dim, 64), nn.Tanh(),
                                           nn.Linear(64, 1))
        out_dim = 1 if num_classes <= 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 128), nn.LayerNorm(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU(inplace=True),
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


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor:
    def __init__(self):
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
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
            face_crop = None

            results = self.face_detector.process(rgb)
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                box_w, box_h = int(bbox.width * w), int(bbox.height * h)
                margin_x, margin_y = int(0.2 * box_w), int(0.2 * box_h)
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(w, x1 + box_w + 2 * margin_x)
                y2 = min(h, y1 + box_h + 2 * margin_y)
                if (x2 - x1) > 30 and (y2 - y1) > 30:
                    face_crop = rgb[y1:y2, x1:x2]

            if face_crop is None:
                crop_size = min(h, w) * 2 // 3
                center_y, center_x = h // 2, w // 2
                y1 = max(0, center_y - crop_size // 2)
                y2 = min(h, center_y + crop_size // 2)
                x1 = max(0, center_x - crop_size // 2)
                x2 = min(w, center_x + crop_size // 2)
                face_crop = rgb[y1:y2, x1:x2]

            face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
            faces.append(face_crop)
            geo_feats = self.extract_geometric_features(face_crop)
            geometric_features.append(geo_feats)

        cap.release()
        if not faces:
            return None, None

        while len(faces) < FRAMES_PER_VIDEO:
            faces.append(faces[-1])
            geometric_features.append(geometric_features[-1])

        return np.array(faces), np.array(geometric_features)

    def extract_geometric_features(self, face_crop):
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        upper_region = gray[0:h//3, :]
        middle_region = gray[h//3:2*h//3, :]
        lower_region = gray[2*h//3:, :]
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        eye_region = gray[h//4:h//2, w//6:5*w//6]
        mouth_region = gray[2*h//3:5*h//6, w//4:3*w//4]

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


# ============================================================
# PREDICTOR WITH REAL rPPG
# ============================================================

class EngagementPredictor:
    def __init__(self, model_path):
        self.device = DEVICE
        self.processor = VideoProcessor()

        # Separate face detector for rPPG (needs its own instance)
        self.rppg_face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        print(f"Loading model from {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.binary_model = None
        self.multi_model = None
        self.binary_threshold = 0.5

        # Load binary model
        binary_keys = ['task5a_binary', 'task1', 'binary_classification']
        for key in binary_keys:
            if key in checkpoint:
                self.binary_model = EngagementHybridModel(num_classes=2).to(self.device)
                state_dict = (checkpoint[key].get('model_state_dict') or
                              checkpoint[key].get('model_states', [None])[0])
                if state_dict:
                    self.binary_model.load_state_dict(state_dict)
                    self.binary_model.eval()
                    self.binary_threshold = checkpoint[key].get('threshold', 0.5)
                    acc = checkpoint[key].get('accuracy',
                                              checkpoint[key].get('cv_accuracy', 0.75))
                    print(f"Binary model loaded ({key}) - Acc: {acc:.1%}")
                    break

        # Load multi-class model
        multi_keys = ['task5b_multi', 'task2', 'multiclass_classification']
        for key in multi_keys:
            if key in checkpoint:
                self.multi_model = EngagementHybridModel(num_classes=4).to(self.device)
                state_dict = (checkpoint[key].get('model_state_dict') or
                              checkpoint[key].get('model_states', [None])[0])
                if state_dict:
                    self.multi_model.load_state_dict(state_dict)
                    self.multi_model.eval()
                    acc = checkpoint[key].get('accuracy',
                                              checkpoint[key].get('cv_accuracy', 0.417))
                    print(f"Multi-class model loaded ({key}) - Acc: {acc:.1%}")
                    break

        print(f"Models ready on {self.device}")

    @torch.no_grad()
    def predict(self, video_path):
        faces, geo_features = self.processor.extract_faces_and_features(video_path)

        if faces is None:
            return {'success': False, 'error': 'Could not extract faces from video'}

        face_tensors = torch.stack([self.processor.transform(Image.fromarray(face))
                                    for face in faces])
        face_tensors = face_tensors.unsqueeze(0).to(self.device)
        geo_tensors = torch.tensor(geo_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        result = {'success': True}

        # Binary classification
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

        # REAL rPPG extraction using POS algorithm
        try:
            bvp, fps = extract_rppg_from_video(video_path, self.rppg_face_detector)
            if bvp is not None:
                hrv = compute_hrv_features(bvp, fs=fps)
                result['rppg'] = hrv
                result['rppg']['algorithm'] = 'POS'
                result['rppg']['n_frames'] = len(bvp)
                result['rppg']['fps'] = round(fps, 1)
            else:
                result['rppg'] = {
                    'hr_bpm': 0.0, 'sdnn_ms': 0.0, 'rmssd_ms': 0.0,
                    'lf_hf_ratio': 0.0, 'sqi': 0.0, 'mean_ibi_ms': 0.0,
                    'error': 'Insufficient frames for rPPG extraction'
                }
        except Exception as e:
            result['rppg'] = {
                'hr_bpm': 0.0, 'sdnn_ms': 0.0, 'rmssd_ms': 0.0,
                'lf_hf_ratio': 0.0, 'sqi': 0.0, 'mean_ibi_ms': 0.0,
                'error': str(e)
            }

        return result


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

try:
    predictor = EngagementPredictor(MODEL_PATH)
except Exception as e:
    print(f"Failed to load model: {e}")
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

    temp_path = os.path.join(tempfile.gettempdir(), f'deepam_{uuid.uuid4().hex}.mp4')

    try:
        video_file.save(temp_path)
        result = predictor.predict(temp_path)
        return jsonify(result)
    except Exception as e:
        print(f"Inference error: {e}")
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
    print("\nServer starting...")
    print(f"Open: http://localhost:5000")
    print(f"Device: {DEVICE}")
    app.run(host='0.0.0.0', port=5000, debug=False)