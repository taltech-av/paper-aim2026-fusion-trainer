#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for MaskFormer Fusion model.
"""
import os
import json
import glob
import argparse
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from models.maskformer_fusion import MaskFormerFusion
from core.metrics_calculator import MetricsCalculator
from core.training_engine import TrainingEngine
from utils.metrics import find_overlap_exclude_bg_ignore
from integrations.training_logger import generate_training_uuid
from integrations.vision_service import create_training, create_config, get_training_by_uuid
from utils.helpers import get_model_path


def dice_coeff(pred, target):
    smooth = 1e-5
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def focal_loss_fn(pred, target, alpha=0.25, gamma=2):
    pred = pred.flatten()
    target = target.flatten()
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def maskformer_loss(class_logits, pred_masks, targets, num_classes):
    # class_logits: [b, num_queries, num_classes+1]
    # pred_masks: [b, num_queries, h, w]
    # targets: [b, h, w]
    total_loss = 0
    for b in range(class_logits.shape[0]):
        # Get GT classes and masks
        gt_classes = []
        gt_masks = []
        for c in range(num_classes):
            mask = (targets[b] == c).float()
            if mask.sum() > 0:
                gt_masks.append(mask)
                gt_classes.append(c)
        num_gt = len(gt_classes)
        if num_gt == 0:
            # Penalize all queries for not being no-object
            loss = F.cross_entropy(class_logits[b].view(-1, num_classes+1), torch.full((100,), num_classes, dtype=torch.long, device=class_logits.device))
            total_loss += loss
            continue
        
        class_logits_b = class_logits[b]  # [100, num_classes+1]
        pred_masks_b = pred_masks[b]  # [100, h, w]
        gt_masks = torch.stack(gt_masks).to(class_logits.device)  # [num_gt, h, w]
        
        # Class cost
        class_probs = F.softmax(class_logits_b, dim=-1)
        class_cost = torch.zeros(100, num_gt, device=class_logits.device)
        for i in range(100):
            for j in range(num_gt):
                c = gt_classes[j]
                class_cost[i, j] = -class_probs[i, c]
        
        # Mask cost (dice)
        mask_cost = torch.zeros(100, num_gt, device=class_logits.device)
        for i in range(100):
            for j in range(num_gt):
                dice = 1 - dice_coeff(pred_masks_b[i], gt_masks[j])
                mask_cost[i, j] = dice
        
        cost = class_cost + mask_cost
        cost_np = cost.cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        
        matched_loss = 0
        for r, c in zip(row_ind, col_ind):
            # Class loss
            target_class = gt_classes[c]
            matched_loss += F.cross_entropy(class_logits_b[r].unsqueeze(0), torch.tensor([target_class], device=class_logits.device))
            # Mask loss
            pred = pred_masks_b[r]
            gt = gt_masks[c]
            dice_loss = 1 - dice_coeff(pred, gt)
            focal_loss = focal_loss_fn(pred, gt)
            matched_loss += dice_loss + focal_loss
        
        # Unmatched queries
        matched_queries = set(row_ind)
        for i in range(100):
            if i not in matched_queries:
                matched_loss += F.cross_entropy(class_logits_b[i].unsqueeze(0), torch.tensor([num_classes], device=class_logits.device))
        
        total_loss += matched_loss
    
    return total_loss / class_logits.shape[0]


def calculate_num_classes(config):
    """
    Calculate number of training classes.
    
    Returns the count of classes defined in train_classes.
    """
    return len(config['Dataset']['train_classes'])


def calculate_num_eval_classes(config, num_classes):
    """
    Calculate number of evaluation classes (excludes background).
    
    Excludes only class 0 (background) from evaluation.
    All train_classes with index > 0 are evaluated.
    """
    # Count classes with index > 0
    eval_count = sum(1 for cls in config['Dataset']['train_classes'] if cls['index'] > 0)
    return eval_count


def setup_dataset():
    """Setup dataset based on configuration."""
    from tools.dataset_png import DatasetPNG as Dataset
    return Dataset


def setup_criterion(config):
    """Setup loss criterion with class weights."""
    train_classes = config['Dataset']['train_classes']
    
    # Extract weights in order of class index
    sorted_classes = sorted(train_classes, key=lambda x: x['index'])
    class_weights = [cls['weight'] for cls in sorted_classes]
    
    weight_loss = torch.Tensor(class_weights)
    print(f"Using class weights: {class_weights}")
    print(f"For classes: {[cls['name'] for cls in sorted_classes]}")
    
    return nn.CrossEntropyLoss(weight=weight_loss)


def setup_overlap_function(config):
    """Setup dataset-specific overlap calculation function."""
    dataset_name = config['Dataset']['name']
    if dataset_name in ['zod', 'waymo']:
        print(f"Using unified IoU calculation (excludes background only)")
        return find_overlap_exclude_bg_ignore


def setup_vision_service(config, training_uuid):
    """Setup vision service integration."""
    model_name = config['CLI']['backbone']
    dataset_name = config['Dataset']['name']
    description = config.get('Summary', f"Training {model_name} on {dataset_name} dataset")
    tags = config.get('tags', [])
    
    # Create config
    config_name = f"{dataset_name} - {model_name} Config"
    vision_config_id = create_config(name=config_name, config_data=config)
    
    if vision_config_id:
        print(f"Created config in vision service: {vision_config_id}")
        
        # Create training
        vision_training_id = create_training(
            uuid=training_uuid,
            name=description,
            model=model_name,
            dataset=dataset_name,
            description='',
            tags=tags,
            config_id=vision_config_id
        )
        
        if vision_training_id:
            print(f"Created training in vision service: {vision_training_id}")
            return vision_training_id
        else:
            print("Failed to create training in vision service")
    else:
        print("Failed to create config in vision service")
    
    return None


def get_training_uuid_from_logs(log_dir):
    """Extract training_uuid from existing epoch log files."""
    epochs_dir = os.path.join(log_dir, 'epochs')
    if not os.path.exists(epochs_dir):
        return None
    
    # Find all epoch JSON files
    epoch_files = glob.glob(os.path.join(epochs_dir, 'epochs/epoch_*.json'))
    if not epoch_files:
        return None
    
    # Get the most recent epoch file
    epoch_files.sort()
    latest_epoch_file = epoch_files[-1]
    
    try:
        with open(latest_epoch_file, 'r') as f:
            epoch_data = json.load(f)
        training_uuid = epoch_data.get('training_uuid')
        if training_uuid:
            print(f"Found existing training_uuid from logs: {training_uuid}")
            return training_uuid
    except Exception as e:
        print(f"Warning: Could not read training_uuid from {latest_epoch_file}: {e}")
    
    return None


def load_checkpoint_if_resume(config, model, optimizer, device):
    """Load checkpoint if resuming training."""
    if not config['General']['resume_training']:
        print('Training from the beginning')
        return 0
    
    model_path = get_model_path(config)
    if not model_path:
        print('No checkpoint found, training from beginning')
        return 0
    
    print(f'Resuming training from {model_path}')
    checkpoint = torch.load(model_path, map_location=device)
    
    if config['General']['reset_lr']:
        print('Reset the epoch to 0')
        return 0
    
    finished_epochs = checkpoint['epoch']
    print(f"Finished epochs in previous training: {finished_epochs}")
    
    if config['General']['epochs'] <= finished_epochs:
        print(f'Error: Current epochs ({config["General"]["epochs"]}) <= finished epochs ({finished_epochs})')
        print(f"Please set epochs > {finished_epochs}")
        exit(1)
    
    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print('Loading trained optimizer...')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return finished_epochs


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='MaskFormer Fusion Training')
    parser.add_argument('-c', '--config', type=str, required=False, 
                       default='config.json', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set random seed
    np.random.seed(config['General']['seed'])
    
    # Set multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Generate or retrieve training UUID
    if config['General']['resume_training']:
        # Try to get existing training_uuid from logs
        training_uuid = get_training_uuid_from_logs(config['Log']['logdir'])
        if training_uuid:
            print(f"Resuming training with existing UUID: {training_uuid}")
        else:
            print("Warning: Could not find existing training_uuid, generating new one")
            training_uuid = generate_training_uuid()
            print(f"New Training UUID: {training_uuid}")
    else:
        training_uuid = generate_training_uuid()
        print(f"Training UUID: {training_uuid}")
    
    # Setup device
    device = torch.device(config['General']['device'] 
                         if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Calculate class counts
    num_classes = calculate_num_classes(config)
    num_eval_classes = calculate_num_eval_classes(config, num_classes)
    print(f"Total classes: {num_classes}, Evaluation classes: {num_eval_classes}")
    
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
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config['MaskFormer']['clft_lr'])
    
    # Setup criterion
    criterion = setup_criterion(config)
    criterion.to(device)
    
    # Setup overlap function
    find_overlap_func = setup_overlap_function(config)
    
    # Setup metrics calculator
    metrics_calc = MetricsCalculator(config, num_eval_classes, find_overlap_func)
    
    # Setup vision service
    vision_training_id = None
    if training_uuid:
        if config['General']['resume_training']:
            # Look up existing training by UUID
            print("Resuming training - looking up existing training record...")
            vision_training_id = get_training_by_uuid(training_uuid)
            if vision_training_id:
                print(f"Found existing training in vision service: {vision_training_id}")
            else:
                print("Warning: Could not find existing training in vision service")
        else:
            # Create new training
            vision_training_id = setup_vision_service(config, training_uuid)
    
    # Load checkpoint if resuming
    start_epoch = load_checkpoint_if_resume(config, model, optimizer, device)
    
    # Setup datasets
    Dataset = setup_dataset()
    train_data = Dataset(config, 'train', config['Dataset']['train_split'])
    valid_data = Dataset(config, 'val', config['Dataset']['val_split'])
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=config['General']['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        persistent_workers=True
    )
    
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=config['General']['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=2,
        persistent_workers=True
    )
    
    # Setup training engine
    training_engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics_calculator=metrics_calc,
        config=config,
        training_uuid=training_uuid,
        log_dir=config['Log']['logdir'],
        device=device,
        vision_training_id=vision_training_id
    )
    
    # Train
    modality = config['CLI']['mode']
    training_engine.train_full(
        train_dataloader, 
        valid_dataloader, 
        modality, 
        num_classes,
        start_epoch=start_epoch
    )


if __name__ == '__main__':
    main()