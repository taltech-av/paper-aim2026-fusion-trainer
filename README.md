# Fusion Training: SAM-Enhanced Semantic Segmentation on ZOD

This repository contains the implementation for the paper "SAM-Enhanced Semantic Segmentation on ZOD: Specialized Models for Vulnerable Road Users".

## Abstract

The Zenseact Open Dataset (ZOD) provides valuable multi-modal data for autonomous driving but lacks dense semantic segmentation annotations, limiting its use for pixel-level perception tasks. We introduce a preprocessing pipeline using the Segment Anything Model (SAM) to convert ZOD's 2D bounding box annotations into dense pixel-level segmentation masks, enabling semantic segmentation training on this dataset for the first time. Due to the imperfect nature of automated mask generation, only 36% of frames passed manual quality control and were included in the final dataset. We present a comprehensive comparison between transformer-based Camera-LiDAR Fusion Transformers (CLFT) and CNN-based DeepLabV3+ architectures for multi-modal semantic segmentation on ZOD across RGB, LiDAR, and fusion modalities under diverse weather conditions. Furthermore, we investigate model specialization techniques to address class imbalance, developing separate modules optimized for large-scale objects (vehicles) and small-scale vulnerable road users (pedestrians, cyclists, traffic signs). The specialized models significantly improve detection of underrepresented safety-critical classes while maintaining overall segmentation accuracy, providing practical insights for deploying multi-modal perception systems in autonomous vehicles. To enable reproducible research, we release the complete open-source implementation of our processing pipeline.

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
