#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing script for MaskFormer models on ZOD weather conditions.
"""
import os
import json
import argparse
import time
import uuid
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.maskformer_fusion import MaskFormerFusion
from core.metrics_calculator import MetricsCalculator
from utils.metrics import find_overlap_exclude_bg_ignore
from utils.helpers import relabel_annotation, get_model_path, sanitize_for_json
from integrations.vision_service import send_test_results_from_file


def _store_predictions_for_ap(output_seg, anno, all_predictions, all_targets, eval_classes, config):
    """Store pixel-wise predictions and targets for AP calculation."""
    # Apply softmax to get probabilities
    probs = torch.softmax(output_seg, dim=1)  # Shape: [batch, classes, H, W]
    
    # Get predictions (argmax) and targets
    preds = torch.argmax(output_seg, dim=1)  # Shape: [batch, H, W]
    
    # Get eval indices from config
    eval_indices = [cls['index'] for cls in config['Dataset']['train_classes'] if cls['index'] > 0]
    
    # For each evaluation class
    for cls_idx, cls_name in enumerate(eval_classes):
        # Get the training index for this class
        train_idx = eval_indices[cls_idx]
        
        # Get probabilities for this class
        cls_probs = probs[:, train_idx, :, :]  # Shape: [batch, H, W]
        
        # Get binary predictions and targets for this class
        cls_preds = (preds == train_idx).float()  # Shape: [batch, H, W]
        cls_targets = (anno == train_idx).float()  # Shape: [batch, H, W]
        
        # Flatten and store
        cls_probs_flat = cls_probs.flatten()
        cls_preds_flat = cls_preds.flatten()
        cls_targets_flat = cls_targets.flatten()
        
        # Only store pixels that are predicted as this class OR are actually this class
        # This ensures we have both true positives and false positives
        relevant_mask = (cls_preds_flat > 0) | (cls_targets_flat > 0)
        
        if relevant_mask.sum() > 0:
            all_predictions[cls_name].append(cls_probs_flat[relevant_mask])
            all_targets[cls_name].append(cls_targets_flat[relevant_mask])


def _compute_ap_for_class(cls_name, all_predictions, all_targets):
    """Compute Average Precision for a single class using proper method."""
    if cls_name not in all_predictions or not all_predictions[cls_name]:
        return 0.0
    
    # Concatenate all predictions and targets for this class
    try:
        pred_probs = torch.cat(all_predictions[cls_name])  # All predicted probabilities
        pred_targets = torch.cat(all_targets[cls_name])    # All ground truth labels
    except RuntimeError as e:
        print(f"Warning: Failed to concatenate tensors for {cls_name}: {e}")
        return 0.0
    
    if len(pred_probs) == 0:
        return 0.0
    
    # Limit to reasonable size to avoid memory issues (sample if too large)
    max_samples = 100000  # Limit to 100k samples for AP calculation
    if len(pred_probs) > max_samples:
        # Sample a subset for AP calculation
        indices = torch.randperm(len(pred_probs))[:max_samples]
        pred_probs = pred_probs[indices]
        pred_targets = pred_targets[indices]
    
    # Sort by prediction confidence (descending)
    sorted_indices = torch.argsort(pred_probs, descending=True)
    pred_probs = pred_probs[sorted_indices]
    pred_targets = pred_targets[sorted_indices]
    
    # Calculate precision and recall at different thresholds
    num_positives = pred_targets.sum().item()
    if num_positives == 0:
        return 0.0
    
    # Calculate cumulative true positives and false positives
    tp = torch.cumsum(pred_targets, dim=0).float()
    fp = torch.cumsum(1 - pred_targets, dim=0).float()
    
    # Calculate precision and recall
    precision = tp / (tp + fp + 1e-6)
    recall = tp / num_positives
    
    # Use VOC 2010 AP calculation method
    ap = _voc_ap(recall, precision)
    return ap


def _voc_ap(recall, precision):
    """Calculate AP using VOC 2010 method."""
    if len(recall) == 0:
        return 0.0
    
    # Convert to numpy
    recall = recall.cpu().numpy()
    precision = precision.cpu().numpy()
    
    # Add sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Calculate AP
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def extract_epoch_info(model_path):
    """Extract epoch number and UUID from model path."""
    import re
    epoch_match = re.search(r'epoch_(\d+)_([a-f0-9\-]+)\.pth', model_path)
    if epoch_match:
        return int(epoch_match.group(1)), epoch_match.group(2)
    
    # Fallback for old checkpoint format
    epoch_match = re.search(r'checkpoint_(\d+)\.pth', model_path)
    if epoch_match:
        return int(epoch_match.group(1)), None
    
    # Fallback for best_model format
    best_match = re.search(r'best_model_([a-f0-9\-]+)\.pth', model_path)
    if best_match:
        return 0, best_match.group(1)
    
    return 0, None


def calculate_num_classes(config):
    """Calculate number of training classes."""
    return len(config['Dataset']['train_classes'])


def calculate_num_eval_classes(config, num_classes):
    """Calculate number of evaluation classes (excludes background)."""
    eval_count = sum(1 for cls in config['Dataset']['train_classes'] if cls['index'] > 0)
    return eval_count


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def setup_dataset(config, weather_condition=None):
    """Setup dataset based on configuration and weather condition."""
    from tools.dataset_png import DatasetPNG as Dataset

    # Modify config for weather-specific testing
    if weather_condition:
        weather_file_map = {
            'day_fair': 'test_day_fair.txt',
            'day_rain': 'test_day_rain.txt',
            'night_fair': 'test_night_fair.txt',
            'night_rain': 'test_night_rain.txt',
            'snow': 'test_snow.txt'
        }
        if weather_condition in weather_file_map:
            # Check dataset name to determine path
            dataset_name = config['Dataset'].get('name', 'zod')
            if dataset_name == 'waymo':
                base_path = "./waymo_dataset/splits_clft"
            else:
                base_path = "./zod_dataset"
                
            config['Dataset']['val_split'] = f"{base_path}/{weather_file_map[weather_condition]}"
        else:
            print(f"Warning: Unknown weather condition {weather_condition}, using default val_split")

    return Dataset


def _prepare_inputs_for_mode(images, lidar, mode):
    """Prepare inputs for model call based on CLI mode (keeps behavior consistent with TestingEngine)."""
    if mode == 'rgb':
        return images, images
    elif mode == 'lidar':
        return lidar, lidar
    else:  # fusion
        return images, lidar


def test_model_on_weather(config, model, device, weather_condition, checkpoint_path):
    """Test model on a specific weather condition."""
    print(f"\nTesting on {weather_condition}...")

    # Setup dataset
    Dataset = setup_dataset(config, weather_condition)
    num_classes = calculate_num_classes(config)
    num_eval_classes = calculate_num_eval_classes(config, num_classes)

    # Create dataset and dataloader
    val_dataset = Dataset(config, 'val', config['Dataset']['val_split'])

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['General']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Setup metrics calculator
    metrics_calculator = MetricsCalculator(config, num_eval_classes, find_overlap_exclude_bg_ignore)

    # Initialize accumulators
    accumulators = {
        'overlap': torch.zeros(num_eval_classes),
        'pred': torch.zeros(num_eval_classes),
        'label': torch.zeros(num_eval_classes),
        'union': torch.zeros(num_eval_classes)
    }
    # Move accumulators to device to match batch metrics
    accumulators = {k: v.to(device) for k, v in accumulators.items()}

    # Initialize storage for AP calculation
    eval_classes = [cls['name'] for cls in config['Dataset']['train_classes'] if cls['index'] > 0]
    all_predictions = {cls: [] for cls in eval_classes}
    all_targets = {cls: [] for cls in eval_classes}

    # Testing loop
    model.eval()
    total_time = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Testing {weather_condition}")):
            images = batch['rgb'].to(device)
            lidar = batch['lidar'].to(device)
            targets = batch['anno'].to(device)

            # Relabel annotations
            targets = relabel_annotation(targets.cpu(), config).squeeze(0).to(device)

            # Forward pass
            start_time = time.time()
            rgb_input, lidar_input = _prepare_inputs_for_mode(images, lidar, config['CLI']['mode'])
            _, outputs = model(rgb_input, lidar_input, config['CLI']['mode'])
            end_time = time.time()
            total_time += (end_time - start_time)

            # Get predictions
            preds = torch.argmax(outputs, dim=1)

            # Update metrics
            metrics_calculator.update_accumulators(accumulators, outputs, targets, num_classes)

            # Store predictions for AP calculation
            _store_predictions_for_ap(outputs, targets, all_predictions, all_targets, eval_classes, config)

    # Compute final metrics
    epoch_metrics = metrics_calculator.compute_epoch_metrics(accumulators, 0, len(val_loader))

    # Compute AP for each class
    ap_results = {}
    for cls_name in eval_classes:
        ap = _compute_ap_for_class(cls_name, all_predictions, all_targets)
        ap_results[cls_name] = ap

    # Calculate inference time
    if len(val_loader) > 0:
        avg_time = total_time / len(val_loader)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
    else:
        avg_time = 0.0
        fps = 0.0

    # Prepare results
    results = {}
    for cls_name in eval_classes:
        cls_idx = eval_classes.index(cls_name)
        results[cls_name] = {
            'iou': epoch_metrics['epoch_IoU'][cls_idx].item(),
            'precision': epoch_metrics['precision'][cls_idx].item(),
            'recall': epoch_metrics['recall'][cls_idx].item(),
            'ap': ap_results[cls_name]
        }

    print(f"{weather_condition} Results:")
    print(f"Average IoU: {epoch_metrics['epoch_IoU'].mean().item():.4f}")
    print(f"Average Precision: {epoch_metrics['precision'].mean().item():.4f}")
    print(f"Average Recall: {epoch_metrics['recall'].mean().item():.4f}")
    print(f"Average AP: {np.mean(list(ap_results.values())):.4f}")
    print(f"FPS: {fps:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test MaskFormer on ZOD weather conditions")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='test_results.json', help='Output file for results')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device
    device = torch.device(config['General']['device'])

    # Calculate classes
    num_classes = calculate_num_classes(config)
    num_eval_classes = calculate_num_eval_classes(config, num_classes)

    # Build model
    model = MaskFormerFusion(
        backbone=config['MaskFormer']['model_timm'],
        num_classes=num_classes,
        pixel_decoder_channels=config['MaskFormer']['pixel_decoder_channels'],
        transformer_d_model=config['MaskFormer']['transformer_d_model'],
        num_queries=config['MaskFormer']['num_queries'],
        pretrained=config['MaskFormer'].get('pretrained', True)
    )
    model.to(device)

    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = get_model_path(config)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        
        # Extract epoch info
        epoch_num, epoch_uuid = extract_epoch_info(checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using untrained model")
        epoch_num, epoch_uuid = 0, None

    # Weather conditions to test
    weather_conditions = ['day_fair', 'day_rain', 'night_fair', 'night_rain', 'snow']

    # Test on each weather condition
    all_results = {}
    for weather in weather_conditions:
        results = test_model_on_weather(config, model, device, weather, checkpoint_path)
        all_results[weather] = results

    # Save results
    logdir = config['Log']['logdir']
    test_results_dir = os.path.join(logdir, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    # Generate filename
    if epoch_uuid:
        filename = f'epoch_{epoch_num}_{epoch_uuid}.json'
    else:
        filename = f'epoch_{epoch_num}_test_results.json'
    output_path = os.path.join(test_results_dir, filename)
    
    output_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "epoch": epoch_num,
        "epoch_uuid": epoch_uuid,
        "test_uuid": str(uuid.uuid4()),
        "test_results": sanitize_for_json(all_results)
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\nSummary:")
    for weather, results in all_results.items():
        eval_classes = [cls['name'] for cls in config['Dataset']['train_classes'] if cls['index'] > 0]
        mean_iou = sum(results[cls]['iou'] for cls in eval_classes) / len(eval_classes)
        mean_precision = sum(results[cls]['precision'] for cls in eval_classes) / len(eval_classes)
        mean_recall = sum(results[cls]['recall'] for cls in eval_classes) / len(eval_classes)
        print(f"{weather}: mIoU={mean_iou:.4f}, mPrecision={mean_precision:.4f}, mRecall={mean_recall:.4f}")

    # Send to vision service
    print("\nUploading test results to vision service...")
    upload_success = send_test_results_from_file(output_path)
    if upload_success:
        print("✅ Test results successfully uploaded to vision service")
    else:
        print("❌ Failed to upload test results to vision service")


if __name__ == "__main__":
    main()