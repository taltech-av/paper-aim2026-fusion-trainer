#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing script for MaskFormer models on ZOD weather conditions.
"""
import os
import json
import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.maskformer_fusion import MaskFormerFusion
from core.metrics_calculator import MetricsCalculator
from utils.metrics import find_overlap_exclude_bg_ignore
from utils.helpers import relabel_annotation, get_checkpoint_path_with_fallback
from utils.test_aggregator import test_checkpoint_and_save


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
            elif dataset_name == 'iseauto':
                base_path = "./xod_dataset"
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
    accumulators = metrics_calculator.create_accumulators(device)

    # Initialize storage for AP calculation
    eval_classes = [cls['name'] for cls in config['Dataset']['train_classes'] if cls['index'] > 0]
    all_predictions = {cls: [] for cls in eval_classes}
    all_targets = {cls: [] for cls in eval_classes}

    # Testing loop
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Testing {weather_condition}")):
            images = batch['rgb'].to(device)
            lidar = batch['lidar'].to(device)
            targets = batch['anno'].to(device)

            # Relabel annotations
            targets = relabel_annotation(targets.cpu(), config).squeeze(0).to(device)

            # Forward pass
            rgb_input, lidar_input = _prepare_inputs_for_mode(images, lidar, config['CLI']['mode'])
            _, outputs, _, _ = model(rgb_input, lidar_input, config['CLI']['mode'])

            # Get predictions
            preds = torch.argmax(outputs, dim=1)

            # Update metrics
            metrics_calculator.update_accumulators(accumulators, outputs, targets, num_classes)

            # Store predictions for AP calculation
            _store_predictions_for_ap(outputs, targets, all_predictions, all_targets, eval_classes, config)

    # Compute final metrics
    epoch_metrics = metrics_calculator.compute_epoch_metrics(accumulators, 0, len(val_loader))

    # Calculate additional metrics
    pixel_accuracy = epoch_metrics['pixel_accuracy']  # Already computed by accumulators
    mean_accuracy = epoch_metrics['mean_accuracy']    # Already computed by accumulators
    fw_iou = 0.0
    if accumulators['class_pixels'].sum() > 0:
        weights = accumulators['class_pixels'] / accumulators['class_pixels'].sum()
        fw_iou = (weights * epoch_metrics['epoch_IoU']).sum().item()

    # Compute AP for each class
    ap_results = {}
    for cls_name in eval_classes:
        ap = _compute_ap_for_class(cls_name, all_predictions, all_targets)
        ap_results[cls_name] = ap

    # Prepare results
    results = {}
    for cls_name in eval_classes:
        cls_idx = eval_classes.index(cls_name)
        results[cls_name] = {
            'iou': epoch_metrics['epoch_IoU'][cls_idx].item(),
            'precision': epoch_metrics['precision'][cls_idx].item(),
            'recall': epoch_metrics['recall'][cls_idx].item(),
            'f1_score': epoch_metrics['f1'][cls_idx].item(),
            'ap': ap_results[cls_name]
        }

    # Add overall metrics
    confusion_matrix_labels = [cls['name'] for cls in sorted(config['Dataset']['train_classes'], key=lambda x: x['index'])]
    results['overall'] = {
        'mIoU_foreground': epoch_metrics['mean_iou'],
        'mean_accuracy': mean_accuracy,
        'fw_iou': fw_iou,
        'pixel_accuracy': pixel_accuracy,
        'confusion_matrix': accumulators['confusion_matrix'].cpu().tolist(),
        'confusion_matrix_labels': confusion_matrix_labels
    }

    print(f"{weather_condition} Results:")
    print(f"mIoU (foreground): {epoch_metrics['mean_iou']:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Frequency-Weighted IoU: {fw_iou:.4f}")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f} (note: dominated by background)")
    
    print("\nPer-class Dice/F1:")
    for i, cls_name in enumerate(eval_classes):
        print(f"  {cls_name} F1: {epoch_metrics['f1'][i].item():.4f}")

    # Now print the Mean AP after results is populated
    print(f"Mean AP: {np.mean(list(ap_results.values())):.4f}")

    return results


def test_single_checkpoint(checkpoint_path, config, device, weather_conditions, num_classes, num_eval_classes):
    """
    Test a single checkpoint on all weather conditions.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dict
        device: Torch device
        weather_conditions: List of weather condition names
        num_classes: Number of classes
        num_eval_classes: Number of evaluation classes

    Returns:
        Dict with results for each weather condition
    """
    # Build model for this checkpoint
    model = MaskFormerFusion(
        backbone=config['MaskFormer']['model_timm'],
        num_classes=num_classes,
        pixel_decoder_channels=config['MaskFormer']['pixel_decoder_channels'],
        transformer_d_model=config['MaskFormer']['transformer_d_model'],
        num_queries=config['MaskFormer']['num_queries'],
        pretrained=config['MaskFormer'].get('pretrained', True)
    )
    model.to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using untrained model")

    # Test on each weather condition
    checkpoint_results = {}
    for weather in weather_conditions:
        results = test_model_on_weather(config, model, device, weather, checkpoint_path)
        checkpoint_results[weather] = results

    # Compute overall metrics as averages across conditions
    overall_miou = np.mean([checkpoint_results[w]['overall']['mIoU_foreground'] for w in weather_conditions])
    overall_mean_acc = np.mean([checkpoint_results[w]['overall']['mean_accuracy'] for w in weather_conditions])
    overall_fw_iou = np.mean([checkpoint_results[w]['overall']['fw_iou'] for w in weather_conditions])
    overall_pixel_acc = np.mean([checkpoint_results[w]['overall']['pixel_accuracy'] for w in weather_conditions])

    checkpoint_results['overall'] = {
        'mIoU_foreground': overall_miou,
        'mean_accuracy': overall_mean_acc,
        'fw_iou': overall_fw_iou,
        'pixel_accuracy': overall_pixel_acc
    }

    return checkpoint_results


def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Test MaskFormer on ZOD weather conditions")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device
    device = torch.device(config['General']['device'])

    # Calculate classes
    num_classes = calculate_num_classes(config)
    num_eval_classes = calculate_num_eval_classes(config, num_classes)

    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Find the best checkpoint automatically (with fallback to latest)
        checkpoint_path = get_checkpoint_path_with_fallback(config)
    
    if not checkpoint_path:
        print("No checkpoints found. Please train the model first.")
        return

    print(f"Testing checkpoint: {checkpoint_path}")

    # Weather conditions to test
    weather_conditions = ['day_fair', 'day_rain', 'night_fair', 'night_rain', 'snow']

    # Test the checkpoint
    test_checkpoint_and_save(
        checkpoint_path, test_single_checkpoint, config, device, weather_conditions, num_classes, num_eval_classes
    )

    # Calculate total execution time
    end_time = time.time()
    total_execution_time = end_time - start_time

    # Print total execution time
    hours, remainder = divmod(total_execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*50}")
    print(f"Total test execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} ({total_execution_time:.2f} seconds)")
    print(f"Completed testing checkpoint")


if __name__ == "__main__":
    main()