#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for Swin Transformer fusion models.
Works with both ZOD and Waymo datasets.
Supports RGB, LiDAR, and cross-fusion modalities.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

from core.advanced_model_builder import AdvancedModelBuilder
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
        return rgb, rgb
    elif modality == 'lidar':
        return lidar, lidar
    else:  # cross_fusion
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
            lidar = rgb  # Use RGB as dummy LiDAR input to avoid unnecessary loading
        else:
            lidar = data_loader.load_lidar(lidar_path).to(device, non_blocking=True).unsqueeze(0)

        # Prepare model inputs
        rgb_input, lidar_input = prepare_model_inputs(rgb, lidar, modality)

        # Forward pass
        with torch.no_grad():
            try:
                _, pred_logits = model(rgb_input, lidar_input, modal=modality)
                pred_segmentation = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()
            except Exception as e:
                print(f"Error during inference for {rgb_name}: {e}")
                failed_count += 1
                continue

        # Load ground truth
        try:
            gt_segmentation = data_loader.load_annotation(anno_path).squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"Error loading ground truth for {rgb_name}: {e}")
            gt_segmentation = None

        # Visualize
        try:
            visualizer.visualize_prediction(
                pred_logits, cam_path, anno_path, idx
            )

            if upload and epoch_uuid:
                try:
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
                except Exception as e:
                    print(f"Upload failed for {rgb_name}: {e}")
                    failed_count += 1

        except Exception as e:
            print(f"Visualization failed for {rgb_name}: {e}")
            failed_count += 1

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(image_paths) - failed_count}/{len(image_paths)}")
    if upload:
        print(f"Successfully uploaded: {uploaded_count}/{len(image_paths)}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Swin Transformer Fusion Model Predictions')
    parser.add_argument('-p', '--path', type=str, required=False,
                        default=None,
                        help='Path to image file or text file with image paths (default: auto-detect based on dataset)')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--upload', action='store_true',
                        help='Upload visualizations to vision service')
    parser.add_argument('--output_dir', type=str, default='./visualizations/',
                        help='Output directory for visualizations')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Set default path based on dataset if not provided
    if args.path is None:
        if config['Dataset']['name'] == 'waymo':
            args.path = 'waymo_dataset/splits_clft/visualizations.txt'
        elif config['Dataset']['name'] == 'iseauto':
            args.path = 'xod_dataset/visualization.txt'
        else:
            args.path = 'zod_dataset/visualizations.txt'
    print(f"Using visualization list: {args.path}")

    # Setup device
    device = torch.device(config['General']['device']
                         if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get model path
    model_path = get_model_path(config, best=True)
    if not model_path:
        print("No model checkpoint found!")
        sys.exit(1)

    print(f"Using model: {model_path}")

    # Extract epoch UUID if uploading
    epoch_uuid = None
    if args.upload:
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
    logdir = config['Log']['logdir'].rstrip('/')
    output_base = f'{logdir}/visualizations'
    dataset_name = config['Dataset']['name']

    # Build model
    model_builder = AdvancedModelBuilder(config, device)
    model = model_builder.build_model()

    # Load checkpoint
    model_builder.load_checkpoint(model, model_path)
    print("Model loaded successfully")

    # Set up data loader and visualizer
    data_loader = InferenceDataLoader(config)
    visualizer = Visualizer(config, output_base)

    # Load image paths
    dataroot = os.path.abspath(config['Dataset']['dataset_root'])
    image_paths = load_image_paths(args.path, dataroot)
    print(f"Found {len(image_paths)} images to process")

    # Get modality from config
    modality = config['CLI']['mode']
    print(f"Using modality: {modality}")

    # Process images
    process_images(
        model, data_loader, visualizer, image_paths, dataroot,
        modality, dataset_name, device, config, epoch_uuid, args.upload
    )


if __name__ == '__main__':
    main()