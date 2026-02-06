# Phase A: Visual Engagement Classification
**Team: 404NotFound**

## Model Architecture
- **Visual Backbone**: ResNet-18 (ImageNet pretrained, fine-tuned layer3+4)
- **Geometric Features**: 12-dimensional pixel-based face analysis
- **Temporal Modeling**: Bidirectional LSTM with attention pooling
- **Classification**: LayerNorm-based MLP head

## Files
- `train.py` - Training script for both tasks
- `inference.py` - Inference script for test videos
- `model.pth` - Trained model weights
