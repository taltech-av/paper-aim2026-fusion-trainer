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
import numpy as np
from torch.utils.data import DataLoader

from models.maskformer_fusion import MaskFormerFusion, MaskFormerCriterion
from core.metrics_calculator import MetricsCalculator
from core.training_engine import TrainingEngine
from utils.metrics import find_overlap_exclude_bg_ignore
from integrations.training_logger import generate_training_uuid
from integrations.vision_service import create_training, create_config, get_training_by_uuid
from utils.helpers import get_model_path, get_training_uuid_from_logs



class MaskFormerTrainingEngine(TrainingEngine):
    """TrainingEngine subclass that adds Hungarian matching loss.

    On each training step:
        total_loss = ce_loss(segmap, anno)
                   + hungarian_weight * hungarian_loss(class_logits, masks, anno)

    Validation still uses ce_loss only (segmap is all we need for IoU eval).
    """

    def __init__(self, *args,
                 hungarian_criterion: MaskFormerCriterion,
                 hungarian_weight: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hungarian_criterion = hungarian_criterion
        self.hungarian_weight    = hungarian_weight

    def train_epoch(self, dataloader, modality, num_classes):
        """One training epoch with CE + Hungarian matching loss."""
        from torch.amp import autocast
        from utils.helpers import relabel_annotation

        self.model.train()
        accumulators = self.metrics_calc.create_accumulators(self.device)
        train_loss = 0.0

        from tqdm import tqdm
        progress_bar = tqdm(dataloader)
        for batch in progress_bar:
            rgb   = batch['rgb'].to(self.device,   non_blocking=True)
            lidar = batch['lidar'].to(self.device,  non_blocking=True)
            anno  = batch['anno'].to(self.device,   non_blocking=True)

            self.optimizer.zero_grad()

            rgb_input, lidar_input = self._prepare_inputs(rgb, lidar, modality)

            with autocast('cuda'):
                model_outputs = self.model(rgb_input, lidar_input, modality)
                # model returns (None, segmap, class_logits, masks)
                output_seg    = model_outputs[1].squeeze(1)   # [B,C,H,W]
                class_logits  = model_outputs[2]               # [B,Q,C+1]
                pred_masks    = model_outputs[3]               # [B,Q,h,w]

                anno = relabel_annotation(
                    anno.cpu(), self.config
                ).squeeze(0).to(self.device)

                ce_loss       = self.criterion(output_seg, anno)
                hun_loss      = self.hungarian_criterion(
                    class_logits, pred_masks, anno
                )
                loss = ce_loss + self.hungarian_weight * hun_loss

            self.metrics_calc.update_accumulators(
                accumulators, output_seg, anno, num_classes
            )
            train_loss += loss.item()

            self.scaler.scale(loss).backward()
            # Unscale before clipping so the threshold is in true gradient units
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            progress_bar.set_description(
                f'Train loss: {loss:.4f}  (ce={ce_loss:.3f}, hun={hun_loss:.3f})'
            )

        metrics = self.metrics_calc.compute_epoch_metrics(
            accumulators, train_loss, len(dataloader)
        )
        return metrics


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
    if dataset_name in ['zod', 'waymo', 'iseauto']:
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
        num_workers=8,
        persistent_workers=True
    )
    
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=config['General']['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=8,
        persistent_workers=True
    )
    
    # Setup Hungarian criterion
    train_classes  = config['Dataset']['train_classes']
    sorted_classes = sorted(train_classes, key=lambda x: x['index'])
    class_weights  = [cls['weight'] for cls in sorted_classes]
    eos_coef       = config['MaskFormer'].get('eos_coef', 0.1)
    hungarian_criterion = MaskFormerCriterion(
        num_classes=num_classes,
        no_object_coef=eos_coef,
        class_weights=torch.tensor(class_weights, dtype=torch.float32),
    ).to(device)
    hungarian_weight = config['MaskFormer'].get('hungarian_weight', 1.0)

    # Setup training engine
    training_engine = MaskFormerTrainingEngine(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics_calculator=metrics_calc,
        config=config,
        training_uuid=training_uuid,
        log_dir=config['Log']['logdir'],
        device=device,
        vision_training_id=vision_training_id,
        hungarian_criterion=hungarian_criterion,
        hungarian_weight=hungarian_weight,
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