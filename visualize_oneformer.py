#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for OneFormer Fusion models.
Works with ZOD, Waymo and Iseauto datasets.
Supports RGB, LiDAR and fusion modalities.
"""
import os
import sys
import json
import argparse
import torch

from models.oneformer_fusion import OneFormerFusion
from core.data_loader import DataLoader as InferenceDataLoader
from core.visualizer import Visualizer
from utils.helpers import get_model_path, get_annotation_path
from integrations.visualization_uploader import (
    upload_all_visualizations_for_image,
    get_epoch_uuid_from_model_path,
)


def calculate_num_classes(config):
    """Calculate number of training classes."""
    return len(config['Dataset']['train_classes'])


def load_image_paths(path_arg, dataroot):
    """Load image paths from a single image path or a text file of paths."""
    if path_arg.endswith(('.png', '.jpg', '.jpeg')):
        return [path_arg]
    with open(path_arg, 'r') as f:
        return f.read().splitlines()


def get_lidar_path(cam_path, dataset_name):
    """Derive the LiDAR image path from the camera image path."""
    if dataset_name == 'zod':
        return cam_path.replace('camera', 'lidar_png')
    else:  # waymo / iseauto
        return cam_path.replace('camera/', 'lidar_png/')


def prepare_model_inputs(rgb, lidar, modality):
    """Return (rgb_input, lidar_input) tensors according to the modality."""
    if modality == 'rgb':
        return rgb, rgb
    elif modality == 'lidar':
        return lidar, lidar
    else:  # fusion / cross_fusion
        return rgb, lidar


def build_model(config, device):
    """Instantiate and return an OneFormerFusion model (weights not loaded)."""
    num_classes = calculate_num_classes(config)
    model = OneFormerFusion(
        backbone=config['OneFormer']['model_timm'],
        num_classes=num_classes,
        pixel_decoder_channels=config['OneFormer'].get('pixel_decoder_channels', 256),
        transformer_d_model=config['OneFormer'].get('transformer_d_model', 256),
        num_queries=config['OneFormer'].get('num_queries', 100),
        pretrained=False,  # weights come from the checkpoint
    )
    model.to(device)
    return model


def load_checkpoint(model, model_path, device):
    """Load model weights from a checkpoint file."""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {model_path}")


def process_images(model, data_loader, visualizer, image_paths, dataroot,
                   modality, dataset_name, device, config,
                   epoch_uuid=None, upload=False):
    """Run inference and visualize every image in the list."""
    model.eval()

    uploaded_count = 0
    failed_count = 0

    for idx, path in enumerate(image_paths, 1):
        cam_path = path if os.path.isabs(path) else os.path.join(dataroot, path)
        anno_path  = get_annotation_path(cam_path, dataset_name, config)
        lidar_path = get_lidar_path(cam_path, dataset_name)

        rgb_name   = os.path.basename(cam_path).split('.')[0]
        anno_name  = os.path.basename(anno_path).split('.')[0]
        lidar_name = os.path.basename(lidar_path).split('.')[0]

        assert rgb_name == anno_name, \
            f"RGB and annotation names don't match: {rgb_name} vs {anno_name}"
        assert rgb_name == lidar_name, \
            f"RGB and LiDAR names don't match: {rgb_name} vs {lidar_name}"

        print(f'Processing image {idx}/{len(image_paths)}: {rgb_name}')

        rgb = data_loader.load_rgb(cam_path).to(device, non_blocking=True).unsqueeze(0)

        if modality == 'rgb':
            lidar = rgb  # dummy — not used by the model in rgb mode
        else:
            lidar = data_loader.load_lidar(lidar_path).to(device, non_blocking=True).unsqueeze(0)

        rgb_input, lidar_input = prepare_model_inputs(rgb, lidar, modality)

        with torch.no_grad():
            try:
                _, pred_logits = model(rgb_input, lidar_input, modal=modality)
            except Exception as e:
                print(f"  Error during inference for {rgb_name}: {e}")
                failed_count += 1
                continue

        try:
            visualizer.visualize_prediction(pred_logits, cam_path, anno_path, idx)
        except Exception as e:
            print(f"  Visualization failed for {rgb_name}: {e}")
            failed_count += 1
            continue

        if upload and epoch_uuid:
            try:
                results = upload_all_visualizations_for_image(
                    epoch_uuid=epoch_uuid,
                    output_base=visualizer.output_base,
                    image_name=os.path.basename(cam_path),
                )
                success_count = sum(1 for r in results.values() if r is not None)
                if success_count > 0:
                    uploaded_count += 1
                    print(f'  Uploaded {success_count}/{len(results)} visualization types')
                else:
                    failed_count += 1
            except Exception as e:
                print(f"  Upload failed for {rgb_name}: {e}")
                failed_count += 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(image_paths) - failed_count}/{len(image_paths)}")
    if upload:
        print(f"Successfully uploaded: {uploaded_count}/{len(image_paths)}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize OneFormer Fusion Model Predictions'
    )
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='Image file or text file with image paths '
                             '(default: auto-detected from dataset name)')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--upload', action='store_true',
                        help='Upload visualizations to vision service')
    parser.add_argument('--output_dir', type=str, default='./visualizations/',
                        help='Output directory for visualizations')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Auto-detect visualization list when not provided
    if args.path is None:
        dataset_name = config['Dataset']['name']
        if dataset_name == 'waymo':
            args.path = 'waymo_dataset/splits_clft/visualizations.txt'
        elif dataset_name == 'iseauto':
            args.path = 'xod_dataset/visualization.txt'
        else:
            args.path = 'zod_dataset/visualizations.txt'
    print(f"Using visualization list: {args.path}")

    device = torch.device(
        config['General']['device'] if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    model_path = get_model_path(config, best=True)
    if not model_path:
        print("No model checkpoint found!")
        sys.exit(1)
    print(f"Using model: {model_path}")

    epoch_uuid = None
    if args.upload:
        epoch_uuid = get_epoch_uuid_from_model_path(model_path)
        if epoch_uuid:
            print(f"Auto-detected epoch UUID: {epoch_uuid}")
            print(f"Visualizations will be uploaded for epoch: {epoch_uuid}")
        else:
            print("Warning: Could not auto-detect epoch UUID — upload will be skipped")
            args.upload = False

    logdir = config['Log']['logdir'].rstrip('/')
    output_base = f'{logdir}/visualizations'

    model = build_model(config, device)
    load_checkpoint(model, model_path, device)

    data_loader = InferenceDataLoader(config)
    visualizer  = Visualizer(config, output_base)

    dataroot = os.path.abspath(config['Dataset']['dataset_root'])
    image_paths = load_image_paths(args.path, dataroot)
    print(f"Found {len(image_paths)} images to process")

    modality = config['CLI']['mode']
    print(f"Using modality: {modality}")

    process_images(
        model, data_loader, visualizer, image_paths, dataroot,
        modality, config['Dataset']['name'], device, config,
        epoch_uuid, args.upload,
    )


if __name__ == '__main__':
    main()
