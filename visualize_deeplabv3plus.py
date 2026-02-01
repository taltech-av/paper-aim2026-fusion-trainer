#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for DeepLabV3+ models.
Works with both ZOD and Waymo datasets.
Supports RGB, LiDAR, and fusion modalities.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

from models.deeplabv3plus import build_deeplabv3plus
from core.data_loader import DataLoader as InferenceDataLoader
from core.visualizer import Visualizer
from utils.helpers import get_model_path, get_annotation_path
from integrations.visualization_uploader import (
    upload_all_visualizations_for_image,
    get_epoch_uuid_from_model_path
)


def calculate_num_classes(config):
    """Calculate number of training classes."""
    return len(config['Dataset']['train_classes'])


def load_image_paths(path_arg, dataroot):
    """Load image paths from file or single path."""
    if path_arg.endswith(('.png', '.jpg', '.jpeg')):
        # Single image
        return [path_arg]
    else:
        # Text file with multiple paths
        with open(path_arg, 'r') as f:
            paths = f.read().splitlines()
        return paths


def get_lidar_path(cam_path, dataset_name):
    """Get LiDAR path based on dataset."""
    if dataset_name == 'zod':
        return cam_path.replace('camera', 'lidar_png')
    else:  # waymo
        return cam_path.replace('camera/', 'lidar_png/')


def prepare_model_inputs(rgb, lidar, modality):
    """Prepare model inputs based on modality."""
    if modality == 'rgb':
        return rgb, None
    elif modality == 'lidar':
        return lidar, None
    else:  # fusion
        return rgb, lidar


def process_images(model, data_loader, visualizer, image_paths, dataroot,
                   modality, dataset_name, device, config, epoch_uuid=None, upload=False):
    """Process and visualize all images."""
    model.eval()

    uploaded_count = 0
    failed_count = 0

    for idx, path in enumerate(image_paths, 1):
        # Construct full paths
        if os.path.isabs(path):
            cam_path = path
        else:
            cam_path = os.path.join(dataroot, path)

        anno_path = get_annotation_path(cam_path, dataset_name, config)
        lidar_path = get_lidar_path(cam_path, dataset_name)

        # Check if required files exist
        if modality in ['rgb', 'fusion'] and not os.path.exists(cam_path):
            print(f'Warning: Camera image not found: {cam_path}')
            continue
        
        if modality in ['lidar', 'fusion'] and not os.path.exists(lidar_path):
            print(f'Warning: LiDAR file not found: {lidar_path}')
            continue

        # Verify paths match
        rgb_name = os.path.basename(cam_path).split('.')[0]
        anno_name = os.path.basename(anno_path).split('.')[0]
        lidar_name = os.path.basename(lidar_path).split('.')[0]

        assert rgb_name == anno_name, f"RGB and annotation names don't match: {rgb_name} vs {anno_name}"
        assert rgb_name == lidar_name, f"RGB and LiDAR names don't match: {rgb_name} vs {lidar_name}"

        # Load data
        print(f'Processing image {idx}/{len(image_paths)}: {rgb_name}')
        rgb = data_loader.load_rgb(cam_path).to(device, non_blocking=True).unsqueeze(0)
        
        # Only load LiDAR data if not in RGB-only mode
        if modality == 'rgb':
            lidar = None
        else:
            lidar = data_loader.load_lidar(lidar_path).to(device, non_blocking=True).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            rgb_input, lidar_input = prepare_model_inputs(rgb, lidar, modality)
            if modality == 'fusion':
                output_seg, _, _ = model(rgb_input, lidar_input)
            else:
                output_seg = model(rgb_input)

        # Visualize
        visualizer.visualize_prediction(output_seg, cam_path, anno_path, idx)

        # Upload to Vision API if enabled
        if upload and epoch_uuid:
            print(f'Uploading visualizations for {rgb_name}...')
            image_filename = os.path.basename(cam_path)
            results = upload_all_visualizations_for_image(
                epoch_uuid=epoch_uuid,
                output_base=visualizer.output_base,
                image_name=image_filename
            )

            # Count successes
            success_count = sum(1 for r in results.values() if r is not None)
            if success_count > 0:
                uploaded_count += 1
                print(f'Successfully uploaded {success_count}/{len(results)} visualization types')
            else:
                failed_count += 1
                print(f'Failed to upload visualizations for {rgb_name}')

    print(f'\nCompleted visualization of {len(image_paths)} images')
    if upload:
        print(f'Upload summary: {uploaded_count} images uploaded, {failed_count} failed')


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize DeepLabV3+ Model Predictions')
    parser.add_argument('-p', '--path', type=str, required=False,
                       default=None,
                       help='Path to image file or text file with image paths (default: auto-detect based on dataset)')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--upload', action='store_true',
                       help='Upload visualizations to Vision API')
    parser.add_argument('--epoch-uuid', type=str, default=None,
                       help='Epoch UUID for upload (auto-detected from model if not provided)')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Set default path based on dataset if not provided
    if args.path is None:
        if config['Dataset']['name'] == 'waymo':
            args.path = 'waymo_dataset/splits_clft/visualizations.txt'
        else:
            args.path = 'zod_dataset/visualizations.txt'
    print(f"Using visualization list: {args.path}")

    # Setup device
    device = torch.device(config['General']['device']
                         if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Calculate number of classes
    num_classes = calculate_num_classes(config)
    print(f"Number of classes: {num_classes}")

    # Get model path
    model_path = get_model_path(config, best=True)
    if not model_path:
        print("No model checkpoint found!")
        sys.exit(1)

    print(f"Using model: {model_path}")

    # Extract epoch UUID if uploading
    epoch_uuid = args.epoch_uuid
    if args.upload and not epoch_uuid:
        epoch_uuid = get_epoch_uuid_from_model_path(model_path)
        if epoch_uuid:
            print(f"Auto-detected epoch UUID: {epoch_uuid}")
        else:
            print("Warning: Could not auto-detect epoch UUID from model path")
            print("Upload will be skipped unless --epoch-uuid is provided")
            args.upload = False

    if args.upload and epoch_uuid:
        print(f"Visualizations will be uploaded to Vision API for epoch: {epoch_uuid}")

    # Extract config name for output directory
    dataset_name = config['Dataset']['name']
    log_dir = config['Log']['logdir']
    output_base = os.path.join(log_dir, 'visualizations')

    # Build and load model
    modality = config['CLI']['mode']
    fusion_type = config['DeepLabV3Plus'].get('fusion_type') or config['DeepLabV3Plus'].get('fusion_strategy', 'learned')
    backbone = config['DeepLabV3Plus'].get('backbone', 'resnet101')
    is_fusion = modality == 'fusion'

    model = build_deeplabv3plus(
        num_classes=num_classes,
        mode=modality,
        fusion_strategy=fusion_type,
        backbone=backbone,
        pretrained=False  # Don't load ImageNet weights for inference
    )
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")

    # Setup data loader
    data_loader = InferenceDataLoader(config)

    # Setup visualizer
    visualizer = Visualizer(config, output_base)

    # Load image paths
    dataroot = os.path.abspath(config['Dataset']['dataset_root'])
    image_paths = load_image_paths(args.path, dataroot)
    print(f"Found {len(image_paths)} images to process")

    print(f"Using modality: {modality}")

    # Process images
    process_images(
        model, data_loader, visualizer, image_paths,
        dataroot, modality, dataset_name, device, config,
        epoch_uuid=epoch_uuid, upload=args.upload
    )


if __name__ == '__main__':
    main()