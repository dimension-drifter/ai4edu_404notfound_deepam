# Student Engagement Recognition using Visual and rPPG Information
## Team 404NotFound
---
## Project Overview

Multimodal student engagement detection system that combines visual features extracted from facial video with physiological signals derived from remote photoplethysmography (rPPG). The system classifies student engagement into binary (Engaged vs Not Engaged) and multi-class (4 engagement levels) categories.

### Key Achievements

| Task | Metric | Score | Target | Status |
|------|--------|-------|--------|--------|
| Binary Classification (Task 5a) | Accuracy | 75.0% | 70.0% | 
| Binary Classification (Task 5a) | F1 Score | 0.800 | - | - |
| Multi-class Classification (Task 5b) | Accuracy | 41.7% | 65.0% | 
| Multi-class Classification (Task 5b) | Macro F1 | 0.343 | - | - |

---

## Resources & Demonstrations

Access our trained models and demonstration videos via the links below:

**Model Downloads**
- [Download Final Model (model.pth) - Google Drive](https://drive.google.com/file/d/1V5ySX1FGnHNLe3tSIlk1ZqqCT1YwsYJi/view?usp=sharing)

**Video Demonstrations**
- [Real-time Inference Demo 1 - Google Drive](https://drive.google.com/file/d/19oG1kzLUuqMczGR1KOOqMQ3I_SIjR-PP/view?usp=sharing)
- [Real-time Inference Demo 2- Google Drive](https://drive.google.com/file/d/1xM3VWLMTTvpjpmmKSrkLryz9F89nqjzP/view?usp=sharing)

---

## Problem Statement

With the rapid growth of online learning platforms, understanding and measuring student attentiveness has become a critical research challenge. This project addresses:

1. **Task 1/5a**: Binary classification of engagement (Engaged vs Not Engaged)
2. **Task 2/5b**: Multi-class classification into 4 levels (Distracted, Disengaged, Nominally Engaged, Highly Engaged)
3. **Task 3**: Remote photoplethysmography (rPPG) signal extraction
4. **Task 4/5**: Multimodal fusion of visual and physiological features

### Label Mapping

**Binary Classification:**
- Class 0 (Low Engagement): Labels 0.00, 0.33
- Class 1 (High Engagement): Labels 0.66, 1.00

**Multi-class Classification:**
- Class 0: Distracted (Label 0.00)
- Class 1: Disengaged (Label 0.33)
- Class 2: Nominally Engaged (Label 0.66)
- Class 3: Highly Engaged (Label 1.00)

---

## Solution Architecture

```
Input Video
    |
    v
+-------------------+
| Face Detection    |  MediaPipe Face Detection
| & Preprocessing   |  224x224 face crops
+-------------------+
    |
    +------------------+------------------+
    |                                     |
    v                                     v
+-------------------+           +-------------------+
| Visual Feature    |           | rPPG Signal       |
| Extraction        |           | Extraction        |
| (ResNet18 + LSTM) |           | (POS)       |
+-------------------+           +-------------------+
    |                                     |
    v                                     v
+-------------------+           +-------------------+
| Geometric         |           | Physiological     |
| Features (12-dim) |           | Features (9-dim)  |
+-------------------+           +-------------------+
    |                                     |
    +------------------+------------------+
                       |
                       v
              +-------------------+
              | Early Fusion      |
              | Classifier        |
              +-------------------+
                       |
                       v
              +-------------------+
              | Engagement        |
              | Prediction        |
              +-------------------+
```

---

## Installation


### Dependencies

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
mediapipe>=0.10.13
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
flask>=2.0.0
flask-cors>=3.0.0
openpyxl>=3.0.0
Pillow>=8.0.0
```

---

## Project Structure

```
aiforeducation_404notfound_deepam/
|
|-- dataset/
|   |-- train/
|   |   |-- labels_train.xlsx
|   |   |-- 001_1.avi
|   |   |-- ...
|   |-- test/
|
|-- processed_faces/           # Extracted face frames
|-- processed_geo/             # Geometric features
|-- rppg_signals/              # Extracted rPPG signals
|-- fusion_features/           # Physiological features
|-- results/                   # Evaluation results and figures
|
|-- task1_2_visual/            # Phase A: Visual baseline
|   |-- model.pth
|
|-- task_3_rppg/               # Phase B: rPPG extraction
|   |-- rppg_algorithms.py
|
|-- model.pth                  # Final multimodal model (191 MB)
|-- app.py                     # Flask backend server
|-- index.html                 # Web UI dashboard
|-- requirements.txt
|-- README.md
```

---

## Phase A: Visual Baseline

### Approach

1. **Preprocessing**: Face detection using MediaPipe, crop and resize to 224x224
2. **Backbone**: ResNet18 pretrained on ImageNet
3. **Temporal Modeling**: Bidirectional LSTM for frame sequence modeling
4. **Geometric Features**: 12-dimensional face landmark features (eye aspect ratio, mouth aspect ratio, head pose)

### Architecture

```python
EngagementHybridModel:
    - ResNet18 backbone (512-dim features per frame)
    - Geometric projection (12 -> 32 dim)
    - BiLSTM (hidden=128, bidirectional)
    - Temporal attention pooling
    - Classification head
```

### Phase A Results

| Task | Accuracy | F1 Score |
|------|----------|----------|
| Binary (Task 1) | 73.3% | 0.750 |
| Multi-class (Task 2) | 40.0% | 0.180 |

---

## Phase B: rPPG Signal Extraction

### Algorithms Implemented

1. **POS (Plane Orthogonal to Skin)**: Robust to motion artifacts
2. **CHROM (Chrominance-based)**: Illumination invariant
3. **Green Channel**: Baseline method

### Signal Processing Pipeline

1. Face detection and ROI extraction (forehead, cheeks)
2. Spatial averaging of RGB channels
3. Temporal filtering (0.7-4.0 Hz bandpass)
4. Signal quality assessment
5. BPM estimation via FFT peak detection

### Extracted Features

| Feature | Description |
|---------|-------------|
| mean_hr | Mean heart rate (BPM) |
| std_hr | Heart rate variability |
| mean_ibi | Mean inter-beat interval (ms) |
| std_ibi | IBI variability |
| rmssd | Root mean square of successive differences |
| sdnn | Standard deviation of NN intervals |
| lf_power | Low frequency power (0.04-0.15 Hz) |
| hf_power | High frequency power (0.15-0.4 Hz) |
| lf_hf_ratio | LF/HF ratio (sympathovagal balance) |

---

## Phase C: Multimodal Fusion

### Fusion Strategy

**Early Fusion**: Concatenation of visual features (128-dim from BiLSTM) with physiological features (9-dim) followed by joint classification.

### Model Architecture

```python
RegularizedFusionModel:
    - Visual: ResNet18 + BiLSTM (128-dim output)
    - Physio: MLP projection (9 -> 16 dim)
    - Fusion: Concatenation (144-dim)
    - Classifier: MLP with dropout (0.6)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 0.01 |
| Batch Size | 4 |
| Epochs | 40 |
| Early Stopping | Patience 12 |
| Cross-Validation | 5-Fold Stratified |

### Phase C Results (5-Fold Cross-Validation)

| Task | Accuracy | F1 Score | Target | Status |
|------|----------|----------|--------|--------|
| Binary (Task 5a) | 75.0% +/- 3.0% | 0.800 | 70.0% |
| Multi-class (Task 5b) | 41.7% +/- 3.6% | 0.343 | 65.0% |

---

## Phase D: Deployment

### Web Application

- **Backend**: Flask REST API
- **Frontend**: HTML/CSS/JavaScript dashboard
- **Inference**: CPU-optimized for demo

### Running the Application

```bash
python app.py

# Navigate to http://localhost:5000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve web dashboard |
| `/api/predict` | POST | Video engagement prediction |
| `/api/health` | GET | Server health check |

### Inference Latency

| Component | Latency |
|-----------|---------|
| Face Detection | 5-10 ms |
| Feature Extraction | 15-20 ms |
| Model Inference | 10-15 ms |
| Total | 30-45 ms |

---

## Results Summary

### Final Performance Comparison

| Model Configuration | Binary Acc | Binary F1 | Multi Acc | Multi F1 |
|---------------------|------------|-----------|-----------|----------|
| Phase A Visual Baseline | 73.3% | 0.750 | 40.0% | 0.180 |
| Phase C Visual Only | 53.3% | 0.667 | 40.0% | 0.362 |
| Phase C Physio Only | 66.7% | 0.667 | 20.0% | 0.163 |
| Phase C Multimodal | 73.3% | 0.778 | 53.3% | 0.428 |
| Phase C CV Final | 75.0% | 0.800 | 41.7% | 0.343 |

### Qualification Status

- Binary Classification (Task 5a): **QUALIFIED** (75.0% >= 70.0% threshold)
- Multi-class Classification (Task 5b): Below threshold (41.7% < 65.0%)

---

## Ablation Study

### Modality Contribution Analysis

| Modality | Binary Accuracy | Multi-class Accuracy |
|----------|-----------------|---------------------|
| Visual Only | 53.3% | 40.0% |
| Physio Only | 66.7% | 20.0% |
| Multimodal (Fusion) | 75.0% | 41.7% |

### Key Findings

1. **Fusion Benefit**: Multimodal fusion improves binary accuracy by +21.7% over visual-only baseline
2. **Physio Value**: Physiological features alone achieve 66.7% binary accuracy, validating rPPG extraction
3. **F1 Improvement**: Binary F1 score improved from 0.667 to 0.800 with fusion
4. **Multi-class Limitation**: 4-class classification limited by small dataset size (74 videos total)

### Physiological Insights

- Engaged students show lower heart rate variability (more stable HRV)
- Distracted students exhibit higher average heart rate
- LF/HF ratio correlates with attention state
- rPPG signals provide complementary information to visual features

---

## Usage

### Training

```python
# Run Jupyter notebook
jupyter notebook aiforeducation_404notfound_deepam.ipynb
```

### Inference

```python
import torch
from PIL import Image

# Load model
checkpoint = torch.load('model.pth', map_location='cpu')

# Initialize model
model = EngagementHybridModel(num_classes=2)
model.load_state_dict(checkpoint['task5a_binary']['model_state_dict'])
model.eval()
```

### Command Line

```bash
# Start web server
python app.py

# Process single video (example)
python -c "from app import predictor; print(predictor.predict('video.avi'))"
```

---

## API Reference

### Prediction Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "video=@sample_video.avi"
```

### Response Format

```json
{
  "success": true,
  "binary": {
    "prediction": 1,
    "label": "High Attentiveness",
    "confidence": 87.5,
    "probability": 87.5
  },
  "multiclass": {
    "prediction": 3,
    "label": "Highly Engaged",
    "confidence": 65.2,
    "probabilities": {
      "Distracted": 5.1,
      "Disengaged": 8.3,
      "Nominally Engaged": 21.4,
      "Highly Engaged": 65.2
    }
  },
  "rppg": {
    "hr_bpm": 72,
    "sdnn_ms": 42.5,
    "rmssd_ms": 38.2,
    "lf_hf_ratio": 1.35
  }
}
```

---



## Limitations

1. **Dataset Size**: Training data limited to 74 videos, affecting multi-class performance
2. **Class Imbalance**: Uneven distribution across 4 engagement levels
3. **rPPG Quality**: Signal extraction sensitive to motion and lighting conditions
4. **Generalization**: Model trained on specific student population and environment

---

**Model Checkpoint**: `model.pth` (191.1 MB)  
**Binary Accuracy**: 75.0%
**Validation Method**: 5-Fold Stratified Cross-Validation