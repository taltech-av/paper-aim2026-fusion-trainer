#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing script for DeepLabV3+ models.
"""
import os
import json
import argparse
import time
import uuid
import datetime
import re
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.deeplabv3plus import build_deeplabv3plus
from core.metrics_calculator import MetricsCalculator
from utils.metrics import find_overlap_exclude_bg_ignore
from utils.helpers import get_all_checkpoint_paths, get_checkpoint_path_with_fallback, sanitize_for_json
from utils.test_aggregator import test_checkpoint_and_save


def calculate_num_classes(config):
    """Calculate number of training classes."""
    return len(config['Dataset']['train_classes'])


def calculate_num_eval_classes(config, num_classes):
    """Calculate number of evaluation classes (excludes background)."""
    eval_count = sum(1 for cls in config['Dataset']['train_classes'] if cls['index'] > 0)
    return eval_count


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


def setup_dataset(config):
    """Setup dataset based on configuration."""
    from tools.dataset_png import DatasetPNG as Dataset
    return Dataset


def relabel_classes(anno, config):
    """Relabel ground truth annotations according to train_classes mapping."""
    train_classes = config['Dataset']['train_classes']
    relabeled = torch.zeros_like(anno)
    
    for train_cls in train_classes:
        class_idx = train_cls['index']
        dataset_indices = train_cls['dataset_mapping']
        
        for dataset_idx in dataset_indices:
            relabeled[anno == dataset_idx] = class_idx
    
    return relabeled


def test_model(model, dataloader, metrics_calc, device, config, num_classes, modality='rgb', is_fusion=False):
    """Test the model on a dataset."""
    model.eval()
    
    accumulators = metrics_calc.create_accumulators(device)
    total_loss = 0.0
    num_batches = 0
    
    # Initialize storage for proper AP calculation (pixel-wise predictions and targets)
    eval_classes = [cls['name'] for cls in config['Dataset']['train_classes'] if cls['index'] > 0]
    all_predictions = {cls: [] for cls in eval_classes}
    all_targets = {cls: [] for cls in eval_classes}
    
    # Initialize accumulators for additional metrics
    pixel_correct = 0.0
    pixel_total = 0.0
    class_pixels = torch.zeros(len(eval_classes))  # For FWIoU
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)  # Include background
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
            rgb = batch['rgb'].to(device)
            anno = batch['anno'].to(device)
            
            # Relabel annotations
            anno = relabel_classes(anno, config)
            
            if is_fusion:
                lidar = batch['lidar'].to(device)
                pred, _, _ = model(rgb, lidar)
            else:
                if modality == 'rgb':
                    pred = model(rgb)
                else:  # lidar
                    lidar = batch['lidar'].to(device)
                    pred = model(lidar)
            
            # Get predictions
            preds = torch.argmax(pred, dim=1)
            
            # Update accumulators
            batch_overlap, batch_pred, batch_label, batch_union = metrics_calc.update_accumulators(
                accumulators, pred, anno, num_classes
            )
            
            # Store predictions and targets for proper AP calculation
            _store_predictions_for_ap(pred, anno, all_predictions, all_targets, eval_classes, config)
            
            # Update pixel accuracy
            correct_pixels = (preds == anno).sum().item()
            total_pixels = anno.numel()
            pixel_correct += correct_pixels
            pixel_total += total_pixels
            
            # Update class pixel counts for FWIoU
            for i in range(len(eval_classes)):
                class_pixels[i] += (anno == (i + 1)).sum().item()  # eval classes start from 1
            
            # Update confusion matrix (vectorized)
            indices = num_classes * anno.flatten() + preds.flatten()
            confusion_matrix += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
            
            num_batches += 1
    
    # Compute metrics
    metrics = metrics_calc.compute_epoch_metrics(accumulators, total_loss, num_batches)
    
    # Calculate additional metrics
    fw_iou = 0.0
    if accumulators['class_pixels'].sum() > 0:
        class_pixels_cpu = accumulators['class_pixels'].cpu()
        weights = class_pixels_cpu / class_pixels_cpu.sum()
        epoch_iou_cpu = metrics['epoch_IoU'].cpu()
        fw_iou = (weights * epoch_iou_cpu).sum().item()
    
    # Add proper AP calculation
    num_eval_classes = len(eval_classes)
    metrics['ap'] = torch.zeros(num_eval_classes)
    for i, cls_name in enumerate(eval_classes):
        ap = _compute_ap_for_class(cls_name, all_predictions, all_targets)
        metrics['ap'][i] = ap
    
    # Add additional metrics
    metrics['fw_iou'] = fw_iou
    metrics['confusion_matrix'] = accumulators['confusion_matrix']
    
    return metrics


def print_metrics(metrics, dataset_name, class_names):
    """Print test metrics."""
    print(f"\n{'='*60}")
    print(f"{dataset_name} Results")
    print(f"{'='*60}")
    print(f"mIoU (foreground): {metrics['mean_iou']:.4f}")
    print(f"Mean Precision: {torch.mean(metrics['precision']).item():.4f}")
    print(f"Mean Recall: {torch.mean(metrics['recall']).item():.4f}")
    print(f"Mean F1: {torch.mean(metrics['f1']).item():.4f}")
    print(f"Mean AP: {torch.mean(metrics['ap']).item():.4f}")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
    print(f"Frequency-Weighted IoU: {metrics['fw_iou']:.4f}")
    
    print("\nPer-class Metrics:")
    for i, class_name in enumerate(class_names):
        iou = metrics['epoch_IoU'][i].item()
        prec = metrics['precision'][i].item()
        rec = metrics['recall'][i].item()
        f1 = metrics['f1'][i].item()
        ap = metrics['ap'][i].item()
        print(f"  {class_name}: IoU={iou:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AP={ap:.4f}")


def extract_epoch_info(model_path):
    """Extract epoch number and UUID from model path."""
    epoch_match = re.search(r'epoch_(\d+)_([a-f0-9\-]+)\.pth', model_path)
    if epoch_match:
        return int(epoch_match.group(1)), epoch_match.group(2)
    
    # Fallback for old checkpoint format
    epoch_match = re.search(r'checkpoint_(\d+)\.pth', model_path)
    if epoch_match:
        return int(epoch_match.group(1)), None
    
    return 0, None


def convert_metrics_to_serializable(metrics, config, num_eval_classes):
    """Convert metrics to JSON-serializable format."""
    train_classes = config['Dataset']['train_classes']
    class_names = [cls['name'] for cls in train_classes if cls['index'] > 0]
    
    serializable = {}
    
    # Add per-class metrics at top level
    for i in range(num_eval_classes):
        class_name = class_names[i] if i < len(class_names) else f'class_{i+1}'
        serializable[class_name] = {
            'iou': float(metrics['epoch_IoU'][i].item()),
            'precision': float(metrics['precision'][i].item()),
            'recall': float(metrics['recall'][i].item()),
            'f1_score': float(metrics['f1'][i].item()),
            'ap': float(metrics['ap'][i].item())
        }
    
    # Add overall metrics under 'overall' key
    confusion_matrix_labels = [cls['name'] for cls in sorted(config['Dataset']['train_classes'], key=lambda x: x['index'])]
    serializable['overall'] = {
        'mIoU_foreground': float(metrics['mean_iou']),
        'mean_precision': float(torch.mean(metrics['precision']).item()),
        'mean_recall': float(torch.mean(metrics['recall']).item()),
        'mean_f1': float(torch.mean(metrics['f1']).item()),
        'mean_ap': float(torch.mean(metrics['ap']).item()),
        'pixel_accuracy': float(metrics['pixel_accuracy']),
        'mean_accuracy': float(metrics['mean_accuracy']),
        'fw_iou': float(metrics['fw_iou']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'confusion_matrix_labels': confusion_matrix_labels
    }
    
    return serializable


def save_test_results(config, all_results, epoch_num, epoch_uuid, test_uuid):
    """Save test results to JSON file."""
    combined_results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model': 'deeplabv3plus',
        'mode': config['CLI']['mode'],
        'fusion_type': config['DeepLabV3Plus'].get('fusion_type', 'N/A'),
        'epoch': epoch_num,
        'epoch_uuid': epoch_uuid,
        'test_uuid': test_uuid,
        'test_results': all_results
    }
    
    # Create results directory
    output_dir = config['Log']['logdir']
    test_results_dir = os.path.join(output_dir, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    # Generate filename
    if epoch_uuid:
        filename = f'deeplabv3plus_epoch_{epoch_num}_{epoch_uuid}.json'
    else:
        filename = f'deeplabv3plus_epoch_{epoch_num}_test_results.json'
    
    filepath = os.path.join(test_results_dir, filename)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(sanitize_for_json(combined_results), f, indent=2)
    
    print(f'\n{"="*60}')
    print(f'Test results saved to: {filepath}')
    if epoch_uuid:
        print(f'Epoch UUID: {epoch_uuid}')
    print(f'Test UUID: {test_uuid}')
    print(f'{"="*60}')
    
    return filepath


def test_single_checkpoint(checkpoint_path, config, device, weather_conditions, num_classes, num_eval_classes, modality, fusion_type, is_fusion):
    """
    Test a single checkpoint on all weather conditions.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dict
        device: Torch device
        weather_conditions: List of weather condition names and file tuples
        num_classes: Number of classes
        num_eval_classes: Number of evaluation classes
        modality: Modality (rgb, lidar, fusion)
        fusion_type: Fusion type
        is_fusion: Whether it's fusion mode

    Returns:
        Dict with results for each weather condition
    """
    # Get backbone from config
    backbone = config['DeepLabV3Plus'].get('backbone', 'resnet101')
    
    # Build model for this checkpoint
    model = build_deeplabv3plus(
        num_classes=num_classes,
        mode=modality,
        fusion_strategy=fusion_type,
        backbone=backbone,
        pretrained=False  # Don't load ImageNet weights for testing
    )
    model.to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Setup metrics calculator
    find_overlap_func = find_overlap_exclude_bg_ignore
    metrics_calc = MetricsCalculator(config, num_eval_classes, find_overlap_func)

    # Setup datasets
    Dataset = setup_dataset(config)

    # Get class names for display
    train_classes = config['Dataset']['train_classes']
    class_names = [cls['name'] for cls in train_classes if cls['index'] > 0]

    # Dictionary to collect all results
    checkpoint_results = {}

    # Define test data path and files
    test_data_path = config['CLI']['path']
    test_data_files = [
        ('day_fair', 'test_day_fair.txt'),
        ('day_rain', 'test_day_rain.txt'),
        ('night_fair', 'test_night_fair.txt'),
        ('night_rain', 'test_night_rain.txt'),
        ('snow', 'test_snow.txt')
    ]

    # Test on each weather condition
    for condition_key, filename in test_data_files:
        filepath = os.path.join(test_data_path, filename)
        if not os.path.exists(filepath):
            print(f"\nSkipping {condition_key}: file not found ({filepath})")
            continue

        condition_name = condition_key.replace('_', ' ').title()
        print(f"\nTesting on {condition_name}...")
        print(f"Using file: {filepath}")

        test_data = Dataset(config, 'test', filepath)
        test_dataloader = DataLoader(
            test_data,
            batch_size=config['General']['batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True
        )

        test_metrics = test_model(model, test_dataloader, metrics_calc, device, config, num_classes, modality, is_fusion)
        print_metrics(test_metrics, condition_name, class_names)
        checkpoint_results[condition_key] = convert_metrics_to_serializable(test_metrics, config, num_eval_classes)

    # Compute overall averages across all weather conditions
    if checkpoint_results:
        # Compute overall metrics as averages across conditions (matching test_swin.py structure)
        overall_miou = np.mean([checkpoint_results[w]['overall']['mIoU_foreground'] for w in checkpoint_results.keys()])
        overall_mean_precision = np.mean([checkpoint_results[w]['overall']['mean_precision'] for w in checkpoint_results.keys()])
        overall_mean_recall = np.mean([checkpoint_results[w]['overall']['mean_recall'] for w in checkpoint_results.keys()])
        overall_mean_f1 = np.mean([checkpoint_results[w]['overall']['mean_f1'] for w in checkpoint_results.keys()])
        overall_mean_ap = np.mean([checkpoint_results[w]['overall']['mean_ap'] for w in checkpoint_results.keys()])
        overall_pixel_acc = np.mean([checkpoint_results[w]['overall']['pixel_accuracy'] for w in checkpoint_results.keys()])
        overall_mean_acc = np.mean([checkpoint_results[w]['overall']['mean_accuracy'] for w in checkpoint_results.keys()])
        overall_fw_iou = np.mean([checkpoint_results[w]['overall']['fw_iou'] for w in checkpoint_results.keys()])

        checkpoint_results['overall'] = {
            'mIoU_foreground': overall_miou,
            'mean_precision': overall_mean_precision,
            'mean_recall': overall_mean_recall,
            'mean_f1': overall_mean_f1,
            'mean_ap': overall_mean_ap,
            'pixel_accuracy': overall_pixel_acc,
            'mean_accuracy': overall_mean_acc,
            'fw_iou': overall_fw_iou
        }

    return checkpoint_results


def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Test DeepLabV3+ Model')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (optional, uses best from config if not specified)')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup device
    device = torch.device(config['General']['device'] 
                         if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Calculate class counts
    num_classes = calculate_num_classes(config)
    num_eval_classes = calculate_num_eval_classes(config, num_classes)
    print(f"Total classes: {num_classes}, Evaluation classes: {num_eval_classes}")
    
    # Determine mode and fusion type
    modality = config['CLI']['mode']
    fusion_type = config['DeepLabV3Plus'].get('fusion_type') or config['DeepLabV3Plus'].get('fusion_strategy', 'learned')
    backbone = config['DeepLabV3Plus'].get('backbone', 'resnet101')
    is_fusion = modality == 'fusion'
    
    # Get checkpoint paths
    if args.checkpoint:
        checkpoint_paths = [args.checkpoint]
    else:
        # Find the best checkpoint automatically (with fallback to latest)
        checkpoint_path = get_checkpoint_path_with_fallback(config)
        if checkpoint_path:
            checkpoint_paths = [checkpoint_path]
        else:
            checkpoint_paths = []
    
    if not checkpoint_paths:
        print("No checkpoints found. Please train the model first.")
        return
    
    print(f"Found {len(checkpoint_paths)} checkpoints to test")
    
    # Weather conditions to test
    weather_conditions = ['day_fair', 'day_rain', 'night_fair', 'night_rain', 'snow']
    eval_classes = [cls['name'] for cls in config['Dataset']['train_classes'] if cls['index'] > 0]
    
    # Test all checkpoints and collect results
    # Test all checkpoints one by one
    all_checkpoint_results = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_data = test_checkpoint_and_save(
            checkpoint_path, test_single_checkpoint, config, device, weather_conditions, 
            num_classes, num_eval_classes, modality, fusion_type, is_fusion
        )
        all_checkpoint_results.append(checkpoint_data)
    
    # Calculate total execution time
    end_time = time.time()
    total_execution_time = end_time - start_time
    
    # Print total execution time
    hours, remainder = divmod(total_execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*50}")
    print(f"Total test execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} ({total_execution_time:.2f} seconds)")
    print(f"Completed testing all {len(checkpoint_paths)} checkpoints")


if __name__ == '__main__':
    main()
