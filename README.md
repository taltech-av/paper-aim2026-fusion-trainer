# Fusion Training

Training and evaluation framework for the paper:

> SAM-Enhanced Segmentation on Road Datasets: Balancing Critical Classes in Autonomous Driving
> Toomas Tahves, Mauro Bellone, Junyi Gu, Raivo Sell
> AIM 2026

| Resource | Link |
|---|---|
| This repo (trainer) | https://github.com/taltech-av/paper-aim2026-fusion-trainer |
| SAM annotation generator | https://github.com/taltech-av/paper-aim2026-zod-sam-generator |
| Datasets | https://app.visin.eu/datasets |
| Training logs & dashboards | https://app.visin.eu/projects/sam-zod |

## Abstract

Dense semantic segmentation is essential for autonomous driving, yet many multi-modal datasets lack pixel-level annotations. The Zenseact Open Dataset (ZOD) provides rich multi-sensor data from Northern European environments but includes only bounding-box labels, limiting its use for segmentation research. We introduce a Segment Anything Model (SAM)-based preprocessing pipeline that converts ZOD's bounding boxes into dense masks. In this pilot study, we process over 100,000 frames and manually curate a 2,300-frame subset (36% acceptance rate) to establish a reliable baseline. This enables the first dense multi-modal object segmentation benchmark on ZOD, focusing specifically on dynamic traffic participants and critical infrastructure. We evaluate transformer-based CLFT and CNN-based DeepLabV3+ architectures across diverse weather conditions, achieving up to 48.1% mIoU with the CLFT-Hybrid fusion model. To address extreme class imbalance, where safety-critical classes such as pedestrians, cyclists, and signs constitute less than 1% of pixels, we explore model specialization with dedicated modules for large-scale (vehicles) and small-scale (vulnerable road users and signs) objects. To assess generalization, we validate our approach on the Iseauto autonomous-vehicle platform using SAM-enhanced manual annotations, achieving 77.5% mIoU. Bidirectional transfer-learning experiments between ZOD and Iseauto further demonstrate that SAM-derived representations transfer effectively across sensor configurations and environmental conditions. All code, training pipelines, and results are released as open source to support reproducible research.

## Reproduction Guide

### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.8+ and a CUDA-capable GPU. The best-performing CLFT-Hybrid model was trained with batch size 8 on a single GPU.

---

### 2. Dataset Download

Download the processed datasets from https://app.visin.eu/datasets and unpack as follows:

```
zod_dataset/                    <- Zenseact Open Dataset with SAM-generated annotations
  annotation_camera_only/
  train.txt
  validation.txt
  test_day_fair.txt
  test_day_rain.txt
  test_night_fair.txt
  test_night_rain.txt
  test_snow.txt
  visualizations.txt

xod_dataset/                    <- Iseauto dataset with SAM-enhanced manual annotations
  train.txt
  validation.txt
  ...
```

The SAM-generated ZOD annotations were produced by the [annotation generator repo](https://github.com/taltech-av/paper-aim2026-zod-sam-generator) from raw ZOD bounding boxes (2,300 curated frames, 36% acceptance rate from 6,400 manually inspected).

---

### 3. Training

All scripts take a single `-c <config.json>` argument. Training logs are written under `logs/` and can be monitored with TensorBoard:

```bash
tensorboard --logdir logs/
```

#### 3a. ZOD Baseline (Table I)

| Config | Model | Mode |
|---|---|---|
| `config/zod/clft/config_1.json` | CLFT-Base | RGB |
| `config/zod/clft/config_2.json` | CLFT-Hybrid | RGB |
| `config/zod/clft/config_3.json` | CLFT-Large | RGB |
| `config/zod/clft/config_4.json` | CLFT-Base | LiDAR |
| `config/zod/clft/config_5.json` | CLFT-Hybrid | LiDAR |
| `config/zod/clft/config_6.json` | CLFT-Large | LiDAR |
| `config/zod/clft/config_7.json` | CLFT-Base | Fusion |
| `config/zod/clft/config_8.json` | CLFT-Hybrid | Fusion — **best: 48.1% mIoU** |
| `config/zod/clft/config_9.json` | CLFT-Large | Fusion |
| `config/zod/deeplab/rgb_baseline.json` | DeepLabV3+ | RGB |
| `config/zod/deeplab/lidar_baseline.json` | DeepLabV3+ | LiDAR |
| `config/zod/deeplab/fusion_baseline.json` | DeepLabV3+ | Fusion |

```bash
# Best ZOD model: CLFT-Hybrid Fusion
python train_clft.py -c config/zod/clft/config_8.json

# DeepLabV3+ fusion baseline
python train_deeplabv3plus.py -c config/zod/deeplab/fusion_baseline.json
```

#### 3b. Model Specialization (Table III)

Dedicated CLFT-Hybrid models trained on subsets of classes to address extreme class imbalance (signs and humans < 1% of pixels). Sign-only specialization improves sign IoU from 29.7% to 43.5%.

| Config | Specialization |
|---|---|
| `config/zod/clft/specialization/baseline.json` | All classes (reference) |
| `config/zod/clft/specialization/vehicle_only.json` | Vehicle only |
| `config/zod/clft/specialization/human_only.json` | Human only |
| `config/zod/clft/specialization/sign_only.json` | Sign only |
| `config/zod/clft/specialization/human_sign.json` | Human + Sign |
| `config/zod/clft/specialization/human_vehicle.json` | Human + Vehicle |
| `config/zod/clft/specialization/vehicle_sign.json` | Vehicle + Sign |

```bash
python train_clft.py -c config/zod/clft/specialization/vehicle_only.json
python train_clft.py -c config/zod/clft/specialization/human_only.json
python train_clft.py -c config/zod/clft/specialization/sign_only.json
```

#### 3c. Iseauto Validation (Table IV)

Identical config structure to ZOD, pointing at `xod_dataset/`:

| Config | Model | Mode |
|---|---|---|
| `config/iseauto/clft/config_1.json` | CLFT-Base | RGB |
| `config/iseauto/clft/config_2.json` | CLFT-Hybrid | RGB |
| `config/iseauto/clft/config_3.json` | CLFT-Large | RGB |
| `config/iseauto/clft/config_4.json` | CLFT-Base | LiDAR |
| `config/iseauto/clft/config_5.json` | CLFT-Hybrid | LiDAR |
| `config/iseauto/clft/config_6.json` | CLFT-Large | LiDAR |
| `config/iseauto/clft/config_7.json` | CLFT-Base | Fusion |
| `config/iseauto/clft/config_8.json` | CLFT-Hybrid | Fusion — **best: 77.5% mIoU** |
| `config/iseauto/clft/config_9.json` | CLFT-Large | Fusion |

```bash
# Best Iseauto model: CLFT-Hybrid Fusion
python train_clft.py -c config/iseauto/clft/config_8.json
```

#### 3d. Transfer Learning (Section V.E)

Bidirectional transfer with reduced learning rates (1e-6 to 3e-5). Models initialized from the opposite dataset reach competitive performance within 20-30 epochs.

```bash
# ZOD -> Iseauto transfer
python train_clft.py -c config/iseauto/clft/transfer_from_zod_wo_sign_config_1.json

# Iseauto -> ZOD transfer
python train_clft.py -c config/zod/clft/transfer_from_iseauto_config_1.json
```

---

### 4. Testing

```bash
# CLFT
python test_clft.py -c config/zod/clft/config_8.json

# DeepLabV3+
python test_deeplabv3plus.py -c config/zod/deeplab/fusion_baseline.json

# Iseauto
python test_clft.py -c config/iseauto/clft/config_8.json

# Specific checkpoint
python test_clft.py -c config/zod/clft/config_8.json --checkpoint path/to/checkpoint.pth
```

---

### 5. Ensemble (51.3% mIoU — best overall result)

The ensemble merges parameters from specialized models (vehicle, human, sign) to achieve the paper's top result. Train all specialization models first (Step 3b), then evaluate the ensemble:

```bash
python ensemble/test_ensemble.py -c config/ensemble/zod_ensemble_config.json
```

Note: The ensemble runs at 10.3 FPS vs 27.8 FPS for the single CLFT-Hybrid baseline, making it suitable as an offline teacher or auto-labeling model rather than real-time inference.

---

### 6. Benchmarking

Speed and accuracy benchmarks across multiple configs:

```bash
# Compare all ZOD fusion variants
python benchmark.py -c config/zod/clft/config_7.json config/zod/clft/config_8.json config/zod/clft/config_9.json

# Explicit device
python benchmark.py -c config/zod/clft/config_8.json --device cuda:0
```

---

### 7. Visualization

```bash
# CLFT predictions
python visualize_clft.py -c config/zod/clft/config_8.json

# DeepLabV3+ predictions
python visualize_deeplabv3plus.py -c config/zod/deeplab/fusion_baseline.json

# Ground truth masks
python visualize_ground_truth.py -c config/zod/clft/config_8.json

# Iseauto
python visualize_clft.py -c config/iseauto/clft/config_8.json
```

---

## Repository Structure

```
train_clft.py                   # CLFT training entry point
train_deeplabv3plus.py          # DeepLabV3+ training entry point
test_clft.py                    # CLFT evaluation
test_deeplabv3plus.py           # DeepLabV3+ evaluation
benchmark.py                    # Multi-config speed + accuracy benchmark
visualize_clft.py               # CLFT prediction visualizer
visualize_deeplabv3plus.py      # DeepLabV3+ prediction visualizer
visualize_ground_truth.py       # Ground truth visualizer
config/
  zod/clft/                     # ZOD CLFT configs (config_1-9, specialization/, transfer)
  zod/deeplab/                  # ZOD DeepLabV3+ configs
  iseauto/clft/                 # Iseauto CLFT configs (config_1-9, transfer)
  ensemble/                     # Ensemble config
clft/                           # CLFT model implementation
models/                         # DeepLabV3+ model implementation
core/                           # Shared training/evaluation engine
ensemble/                       # Ensemble merge and test scripts
utils/                          # Metrics, augmentation, helpers
zod_dataset/                    # ZOD split files (data not included, download separately)
xod_dataset/                    # Iseauto split files (data not included, download separately)
```

## License

See [LICENSE](LICENSE) for details.
