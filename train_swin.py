#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Swin Transformer Fusion model.
"""
import os
import json
import glob
import argparse
import multiprocessing
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from core.advanced_model_builder import AdvancedModelBuilder
from core.metrics_calculator import MetricsCalculator
from core.training_engine import TrainingEngine
from utils.metrics import find_overlap_exclude_bg_ignore
from integrations.training_logger import generate_training_uuid
from integrations.vision_service import create_training, create_config, get_training_by_uuid
from utils.helpers import get_model_path


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
    parser = argparse.ArgumentParser(description='Swin Transformer Fusion Training')
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
    model_builder = AdvancedModelBuilder(config, device)
    model = model_builder.build_model()
    model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config['SwinFusion']['clft_lr'])
    
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