# TrackMaster — Multi-Object Video Tracking

> YOLOv8 detection + DeepSORT re-identification for robust real-time multi-object tracking. MOTA 0.812, IDF1 0.784 at 30 FPS.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The Problem

Multi-object tracking (MOT) must solve three coupled sub-problems simultaneously:
1. **Detection**: Localize all targets in each frame (handled by YOLOv8).
2. **Association**: Match detections across frames despite occlusion, ID switches, and new/disappearing targets (DeepSORT with Kalman filter).
3. **Re-identification**: Recover correct track ID after complete occlusion (appearance embedding network).

The key failure mode is "ID switching" — when tracks get swapped during occlusion. Standard IoU-based association fails for objects that disappear for >1 second.

---

## Architecture

```
Video Frame t
      │
      ▼
┌──────────────────┐
│  YOLOv8-M        │   ← Detection (8ms/frame)
│  Object Detector │
└────────┬─────────┘
         │  Detections: [{bbox, conf, class}, ...]
         ▼
┌──────────────────────────────────────────────────────┐
│                   DeepSORT Tracker                   │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  Kalman Filter (per active track)           │    │
│  │  State: [x, y, a, h, ẋ, ẏ, ȧ, ḣ]          │    │
│  │  Predict → Update with matched detection    │    │
│  └──────────────────────┬──────────────────────┘    │
│                         │                           │
│  ┌──────────────────────▼──────────────────────┐    │
│  │  Hungarian Algorithm (assignment)           │    │
│  │  Cost = α×IoU_dist + (1-α)×appearance_dist │    │
│  │  (α=0.7 for fast objects, 0.3 for static)  │    │
│  └──────────────────────┬──────────────────────┘    │
│                         │                           │
│  ┌──────────────────────▼──────────────────────┐    │
│  │  Re-ID Embedding (ResNet-34, 128-d output)  │    │
│  │  Cosine distance threshold: 0.3             │    │
│  │  Gallery: 50-frame appearance history       │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
         │
         ▼
Track outputs: [{track_id, bbox, class, age}, ...]
```

**Why DeepSORT over SORT or ByteTrack?**
- SORT: IoU-only association breaks completely when objects cross paths. ID switches = 847 on MOT17.
- ByteTrack: Low-confidence detections cause ghost tracks in crowded scenes.
- DeepSORT: Appearance embedding reduces ID switches to 124 on MOT17; re-identification works for occlusions up to ~4 seconds.

---

## Training Details — Re-ID Network

| Setting | Value |
|---------|-------|
| Backbone | ResNet-34 (ImageNet pretrained) |
| Output | 128-d L2-normalized embedding |
| Loss | Triplet loss (margin=0.3) + Cross-entropy |
| Dataset | Market-1501 + custom retail footage |
| Hardware | 1× RTX 3090 |
| Epochs | 80 |
| Mining | Online hard triplet mining |

---

## Results on MOT17

| Tracker | MOTA ↑ | IDF1 ↑ | ID Sw. ↓ | FPS ↑ |
|---------|--------|--------|----------|-------|
| SORT | 0.744 | 0.691 | 847 | 48 |
| ByteTrack | 0.803 | 0.761 | 196 | 45 |
| **TrackMaster** | **0.812** | **0.784** | **124** | **30** |
| OC-SORT (reference) | 0.831 | 0.812 | 102 | 28 |

---

## Ablation Study

| Configuration | MOTA | IDF1 | ID Sw. |
|---------------|------|------|--------|
| YOLOv8 + IoU only | 0.744 | 0.691 | 847 |
| + Kalman filter | 0.781 | 0.723 | 412 |
| + Re-ID embedding | 0.806 | 0.771 | 187 |
| + Adaptive α (IoU/appearance) | **0.812** | **0.784** | **124** |

---

## Failure Analysis

- **Long-duration occlusion (>5 sec)**: Gallery appearance history expires (50 frames); track is terminated and new ID assigned on reappearance.
- **Dense crowds (>20 people/frame)**: Hungarian assignment is O(n³); processing delay increases to 45ms/frame. Approximate assignment (auction algorithm) reduces to 32ms with <2% MOTA loss.
- **Very similar appearance (uniforms)**: Re-ID embeddings cluster, increasing ID switch rate in sports tracking scenarios.

---

## Getting Started

```bash
git clone https://github.com/sherifabdelrady/trackmaster
cd trackmaster
pip install -r requirements.txt

# Train Re-ID network
python train_reid.py --config configs/resnet34_triplet.yaml --dataset market1501

# Run tracking on video
python track.py --source video.mp4 --detector yolov8m --reid checkpoints/reid_best.pth

# Evaluate on MOT17
python evaluate_mot.py --benchmark MOT17 --split val
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
