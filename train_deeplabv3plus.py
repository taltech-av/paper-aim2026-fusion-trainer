#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for DeepLabV3+ baseline comparison.
Supports RGB-only and late-fusion modes.
"""
import os
import json
import glob
import argparse
import multiprocessing
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from models.deeplabv3plus import build_deeplabv3plus
from core.metrics_calculator import MetricsCalculator
from utils.metrics import find_overlap_exclude_bg_ignore
from integrations.training_logger import generate_training_uuid, log_epoch_results
from integrations.vision_service import create_training, create_config, get_training_by_uuid
from utils.helpers import get_model_path, manage_checkpoints_by_miou, get_training_uuid_from_logs
from utils.system_monitor import get_epoch_system_snapshot, print_system_info


def calculate_num_classes(config):
    """Calculate number of training classes."""
    return len(config['Dataset']['train_classes'])


def calculate_num_eval_classes(config, num_classes):
    """Calculate number of evaluation classes (excludes background)."""
    eval_count = sum(1 for cls in config['Dataset']['train_classes'] if cls['index'] > 0)
    return eval_count


def setup_dataset(config):
    """Setup dataset based on configuration."""
    from tools.dataset_png import DatasetPNG as Dataset
    return Dataset


def setup_criterion(config):
    """Setup loss criterion with class weights."""
    train_classes = config['Dataset']['train_classes']
    sorted_classes = sorted(train_classes, key=lambda x: x['index'])
    class_weights = [cls['weight'] for cls in sorted_classes]
    
    weight_loss = torch.Tensor(class_weights)
    print(f"Using class weights: {class_weights}")
    print(f"For classes: {[cls['name'] for cls in sorted_classes]}")
    
    return nn.CrossEntropyLoss(weight=weight_loss)


def setup_lr_scheduler(optimizer, config):
    """Setup learning rate scheduler based on configuration."""
    lr_config = config['DeepLabV3Plus'].get('lr_scheduler', None)
    if not lr_config:
        print("No learning rate scheduler configured, using constant learning rate")
        return None
    
    scheduler_type = lr_config.get('type', 'step')
    
    if scheduler_type == 'step':
        step_size = lr_config.get('step_size', 30)
        gamma = lr_config.get('gamma', 0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}")
        
    elif scheduler_type == 'cosine':
        T_max = lr_config.get('T_max', config['General']['epochs'])
        eta_min = lr_config.get('eta_min', 1e-6)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        print(f"Using CosineAnnealingLR scheduler: T_max={T_max}, eta_min={eta_min}")
        
    elif scheduler_type == 'exponential':
        gamma = lr_config.get('gamma', 0.95)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        print(f"Using ExponentialLR scheduler: gamma={gamma}")
        
    elif scheduler_type == 'plateau':
        mode = lr_config.get('mode', 'max')
        factor = lr_config.get('factor', 0.1)
        patience = lr_config.get('patience', 10)
        min_lr = lr_config.get('min_lr', 1e-6)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, 
                                                   patience=patience, min_lr=min_lr)
        print(f"Using ReduceLROnPlateau scheduler: mode={mode}, factor={factor}, patience={patience}")
        
    else:
        print(f"Unknown scheduler type: {scheduler_type}, using constant learning rate")
        return None
    
    return scheduler


def setup_vision_service(config, training_uuid):
    """Setup vision service integration."""
    model_name = config['CLI']['backbone']
    dataset_name = config['Dataset']['name']
    description = config.get('Summary', f"Training {model_name} on {dataset_name} dataset")
    tags = config.get('tags', [])
    
    config_name = f"{dataset_name} - {model_name} Config"
    vision_config_id = create_config(name=config_name, config_data=config)
    
    if vision_config_id:
        print(f"Created config in vision service: {vision_config_id}")
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
    
    return None

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


def train_epoch(model, dataloader, criterion, optimizer, metrics_calc, device, config, 
                num_classes, modality='rgb', is_fusion=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    accumulators = metrics_calc.create_accumulators(device)
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        rgb = batch['rgb'].to(device)
        anno = batch['anno'].to(device)
        
        # Relabel annotations
        anno = relabel_classes(anno, config)
        
        optimizer.zero_grad()
        
        if is_fusion:
            lidar = batch['lidar'].to(device)
            pred, rgb_pred, lidar_pred = model(rgb, lidar)
            loss = criterion(pred, anno)
        else:
            if modality == 'rgb':
                pred = model(rgb)
            else:  # lidar
                lidar = batch['lidar'].to(device)
                pred = model(lidar)
            loss = criterion(pred, anno)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update metrics (pass raw predictions, not argmax)
        metrics_calc.update_accumulators(accumulators, pred, anno, num_classes)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute epoch metrics
    metrics = metrics_calc.compute_epoch_metrics(accumulators, total_loss, num_batches)
    return metrics


def validate_epoch(model, dataloader, criterion, metrics_calc, device, config, 
                   num_classes, modality='rgb', is_fusion=False):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    accumulators = metrics_calc.create_accumulators(device)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating')
        for batch in progress_bar:
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
            
            loss = criterion(pred, anno)
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics (pass raw predictions, not argmax)
            metrics_calc.update_accumulators(accumulators, pred, anno, num_classes)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute epoch metrics
    metrics = metrics_calc.compute_epoch_metrics(accumulators, total_loss, num_batches)
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, config, log_dir, epoch_uuid):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}_{epoch_uuid}.pth')
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    
    # Add scheduler state if it exists
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint_data, checkpoint_path)
    
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint_if_resume(config, model, optimizer, scheduler, device):
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
    
    # Load scheduler state if it exists
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        print('Loading scheduler state...')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return finished_epochs


def main():
    parser = argparse.ArgumentParser(description='DeepLabV3+ Training')
    parser.add_argument('-c', '--config', type=str, required=False, 
                       default='config.json', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set random seed
    np.random.seed(config['General']['seed'])
    multiprocessing.set_start_method('spawn', force=True)
    
    # Generate or retrieve training UUID
    vision_training_id = None
    if config['General']['resume_training']:
        # Try to get existing training_uuid and vision_training_id from logs
        training_uuid, vision_training_id = get_training_uuid_from_logs(config['Log']['logdir'])
        if training_uuid:
            print(f"Resuming training with existing UUID: {training_uuid}")
            if vision_training_id:
                print(f"Using existing vision training ID: {vision_training_id}")
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
    
    # Print system information
    print_system_info()
    
    # Calculate class counts
    num_classes = calculate_num_classes(config)
    num_eval_classes = calculate_num_eval_classes(config, num_classes)
    print(f"Total classes: {num_classes}, Evaluation classes: {num_eval_classes}")
    
    # Determine mode and fusion strategy
    modality = config['CLI']['mode']
    fusion_strategy = config['DeepLabV3Plus'].get('fusion_strategy', 'residual_average')
    is_fusion = modality == 'fusion'
    
    # Build model
    model = build_deeplabv3plus(
        num_classes=num_classes,
        mode=modality,
        fusion_strategy=fusion_strategy,
        pretrained=config['DeepLabV3Plus'].get('pretrained', True),
        backbone=config['DeepLabV3Plus'].get('backbone', 'resnet50')
    )
    model.to(device)
    
    # Setup optimizer
    lr = config['DeepLabV3Plus'].get('learning_rate', 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Setup learning rate scheduler
    scheduler = setup_lr_scheduler(optimizer, config)
    
    # Setup criterion
    criterion = setup_criterion(config)
    criterion.to(device)
    
    # Setup metrics calculator
    find_overlap_func = find_overlap_exclude_bg_ignore
    metrics_calc = MetricsCalculator(config, num_eval_classes, find_overlap_func)
    
    # Setup vision service
    if training_uuid:
        if config['General']['resume_training'] and vision_training_id is None:
            # Look up existing training by UUID only if we don't have it from logs
            print("Resuming training - looking up existing training record...")
            vision_training_id = get_training_by_uuid(training_uuid)
            if vision_training_id:
                print(f"Found existing training in vision service: {vision_training_id}")
            else:
                print("Warning: Could not find existing training in vision service")
        elif not config['General']['resume_training']:
            # Create new training
            vision_training_id = setup_vision_service(config, training_uuid)
    
    # Load checkpoint if resuming
    start_epoch = load_checkpoint_if_resume(config, model, optimizer, scheduler, device)
    
    # Setup datasets
    Dataset = setup_dataset(config)
    train_data = Dataset(config, 'train', config['Dataset']['train_split'])
    valid_data = Dataset(config, 'val', config['Dataset']['val_split'])
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=config['General']['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=8,
        persistent_workers=True
    )
    
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=config['General']['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=8,
        persistent_workers=True
    )
    
    # Training loop
    log_dir = config['Log']['logdir']
    os.makedirs(log_dir, exist_ok=True)
    
    best_val_iou = 0.0
    patience_counter = 0
    early_stop_patience = config['General']['early_stop_patience']
    
    for epoch in range(start_epoch, config['General']['epochs']):
        print(f"{'='*60}")
        print(f"Epoch {epoch + 1}/{config['General']['epochs']}")
        print(f"{'='*60}")
        
        epoch_start_time = time.time()
        
        # Train
        print(f"Training epoch {epoch + 1}/{config['General']['epochs']}...")
        train_metrics = train_epoch(model, train_dataloader, criterion, optimizer, 
                                    metrics_calc, device, config, num_classes, modality, is_fusion)
        metrics_calc.print_metrics(train_metrics, prefix="Training ")
        
        # Validate
        print(f"Validating epoch {epoch + 1}/{config['General']['epochs']}...")
        val_metrics = validate_epoch(model, valid_dataloader, criterion, 
                                     metrics_calc, device, config, num_classes, modality, is_fusion)
        
        metrics_calc.print_metrics(val_metrics, prefix="Validation ")
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Get system resource snapshot
        system_snapshot = get_epoch_system_snapshot()
        
        # Step learning rate scheduler first
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                # For ReduceLROnPlateau, step with validation metric
                scheduler.step(val_metrics['mean_iou'])
            else:
                # For other schedulers, step normally
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate updated to: {current_lr:.6f}")
        
        # Log epoch with CLFT-style format (after scheduler step so LR is updated)
        results_dict = metrics_calc.prepare_results_dict(train_metrics, val_metrics)
        
        epoch_file = log_epoch_results(
            epoch=epoch,
            training_uuid=training_uuid,
            results=results_dict,
            log_dir=log_dir,
            learning_rate=optimizer.param_groups[0]['lr'],
            epoch_time=epoch_time,
            system_info=system_snapshot,
            vision_training_id=vision_training_id
        )
        
        # Extract epoch_uuid from the logged file path
        epoch_uuid = os.path.basename(epoch_file).replace(f'epoch_{epoch}_', '').replace('.json', '')
        
        # Vision service upload is already handled by log_epoch_results above
        
        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, scheduler, epoch, config, log_dir, epoch_uuid)
        
        # Manage checkpoints: keep only top max_checkpoints by validation mIoU
        manage_checkpoints_by_miou(config, log_dir)
        
        # Early stopping
        if val_metrics['mean_iou'] > best_val_iou:
            best_val_iou = val_metrics['mean_iou']
            patience_counter = 0
            print(f"New best mIoU: {best_val_iou:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")
            
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    print(f"\nTraining completed! Best validation mIoU: {best_val_iou:.4f}")


if __name__ == '__main__':
    main()
