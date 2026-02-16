# Fusion Training: CLFTv2

This repository contains the implementation for the paper "SAM-Enhanced Segmentation on ZOD and Iseauto: Balancing Critical Classes in Autonomous Driving".

## Abstract

Dense semantic segmentation is critical for autonomous driving, yet many multi-modal datasets lack pixel-level annotations.
Although the Zenseact Open Dataset (ZOD) provides rich multi-modal data from Northern European environments, it contains only bounding box annotations unsuitable for semantic segmentation.
We present a Segment Anything Model (SAM)-based pipeline that converts ZOD's bounding boxes into dense pixel-level masks, yielding a carefully curated 2300-frame segmentation subset with 36\% acceptance rate after manual quality inspection.
This enables, for the first time, dense multi-modal segmentation training and benchmarking on ZOD.
We compare transformer-based CLFT and CNN-based DeepLabV3+ architectures across diverse weather conditions, achieving up to 48.1\% mIoU with CLFT-Hybrid Fusion.
To address severe class imbalance, where safety-critical objects comprise less than 1\% of pixels, we investigate model specialization with dedicated modules for large-scale (vehicles) and small-scale (vulnerable road users, signs) objects.
We validate our approach on the Iseauto autonomous vehicle platform with SAM-enhanced manual annotations, achieving 77.5\% mIoU in fusion settings, and demonstrate effective bidirectional transfer learning between ZOD and Iseauto datasets.
All code, training pipelines, and results are released as open-source to enable reproducible research.

## Setup Virtual Environment

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the models, use the provided training scripts. For example:

- For CLFT models: `python train.py` (adjust configurations as needed)
- For DeepLabV3+ models: `python train_deeplabv3plus.py`

### Testing

Run tests with model-specific scripts:

- **CLFT models**: `python test.py -c config/zod/clft/config_1.json`
- **DeepLabV3+ models**: `python test_deeplabv3plus.py -c config/zod/deeplabv3/config_1.json`

### Visualization

Visualize results using:

- **CLFT models**: `python visualize.py -c config/zod/clft/config_1.json -p zod_dataset/visualizations.txt`
- **DeepLabV3+ models**: `python visualize_deeplabv3plus.py -c config/zod/deeplabv3/config_1.json -p zod_dataset/visualizations.txt`
- **Ground truth**: `python visualize_ground_truth.py -c config/zod/clft/config_1.json -p zod_dataset/visualizations.txt`

### Benchmarking

Run benchmarks across multiple configurations:

- **CLFT models**: `python benchmark.py -c config/zod/clft/config_1.json config/zod/clft/config_2.json`

## Dataset

This project uses the Zenseact Open Dataset (ZOD) and Waymo Open Dataset.

### ZOD Download
- Apply for access at [zod.zenseact.com](https://zod.zenseact.com/)
- Install SDK: `pip install "zod[cli]"`
- Download: `zod download -y --url="<link>" --output-dir=./zod_raw --subset=frames --version=full`
- Preprocess to `./zod_dataset/` (SAM-generated masks from bounding boxes)

### Waymo Download
- Download processed dataset: [roboticlab.eu/claude/waymo](https://www.roboticlab.eu/claude/waymo/)
- Extract 'labeled' folder to `./waymo_dataset/`

## Paper

For more details, refer to the paper: "SAM-Enhanced Semantic Segmentation on ZOD: Specialized Models for Vulnerable Road Users".

## License

See LICENSE file for details.
