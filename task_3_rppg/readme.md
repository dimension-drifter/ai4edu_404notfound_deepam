# Phase B: rPPG Signal Extraction
**Team: 404NotFound**

## What It Does
- Extracts non-contact physiological signals (rPPG) from RGB video.
- Runs three algorithms for robustness:
  - **POS** (Plane-Orthogonal-to-Skin) — physics-based baseline.
  - **TS-CAN** — temporal shift CNN with attention.
  - **EfficientPhys** — lightweight physiologically guided CNN.

## Key Outputs
- **Raw signals**: `rppg_signals.json` (waveforms + BPM + SQI).
- **Summary CSV**: `rppg_signals_summary.csv` (BPM, SQI per video/algorithm).

## Files
- `train.py` — Training & rPPG extraction pipeline.
- `inference.py` — Inference for engagement + rPPG on test videos.
- `rppg_signals_summary.csv` — Cleaned rPPG outputs.

