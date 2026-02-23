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
    """TrainingEngine subclass that uses Hungarian matching loss only.

    On each training step:
        total_loss = hungarian_loss(class_logits, masks, anno)

    Matches the original MaskFormer training setup (Cheng et al., NeurIPS 2021).
    Validation uses the base class CE loss on the segmap for monitoring.
    """

    def __init__(self, *args,
                 hungarian_criterion: MaskFormerCriterion,
                 scheduler=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hungarian_criterion = hungarian_criterion
        self.scheduler           = scheduler

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

                loss = self.hungarian_criterion(
                    class_logits, pred_masks, anno
                )

            self.metrics_calc.update_accumulators(
                accumulators, output_seg, anno, num_classes
            )
            train_loss += loss.item()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            progress_bar.set_description(f'Train loss: {loss:.4f}')

        metrics = self.metrics_calc.compute_epoch_metrics(
            accumulators, train_loss, len(dataloader)
        )
        if self.scheduler is not None:
            self.scheduler.step()
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
    
    # Setup optimizer — differential LR: pretrained backbone at 10x lower rate
    # so it is fine-tuned gently, while the pixel-decoder / transformer-decoder /
    # heads start from random init and need a higher LR to converge.
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
    base_lr = config['MaskFormer']['clft_lr']
    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': base_lr * 0.1},
            {'params': other_params,    'lr': base_lr},
        ],
        weight_decay=0.05,
    )
    
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
        shuffle=False,      # deterministic — noisy batch selection masks model quality
        pin_memory=True,
        drop_last=False,    # evaluate every validation sample
        num_workers=8,
        persistent_workers=True
    )
    
    # Setup Hungarian criterion (no per-class weights — original MaskFormer uses eos_coef only)
    eos_coef = config['MaskFormer'].get('eos_coef', 0.1)
    hungarian_criterion = MaskFormerCriterion(
        num_classes=num_classes,
        no_object_coef=eos_coef,
    ).to(device)

    # Prevent legacy momentum-based decay from overriding the scheduler.
    # (Some older configs carry lr_momentum; train_full's adjust_learning_rate
    # is a no-op when lr_momentum is absent.)
    config['MaskFormer'].pop('lr_momentum', None)

    warmup_epochs  = config['MaskFormer'].get('warmup_epochs', 10)
    total_epochs   = config['General']['epochs']
    post_warmup    = max(1, total_epochs - warmup_epochs)
    lr_sched_type  = config['MaskFormer'].get('lr_scheduler', 'poly')

    # Linear warm-up shared by both schedule types.
    # Matches the original MaskFormer Swin config (WARMUP_FACTOR: 1e-6).
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
    )

    if lr_sched_type == 'step':
        # MultiStepLR (×gamma at each milestone epoch).
        # Milestones expressed as fractions [0,1) of post-warmup epochs so
        # they scale automatically with different epoch budgets.
        # Default fractions match the original Detectron2 COCO-stuff config:
        #   STEPS=(119600, 144400) / MAX_ITER=160000 ≈ 0.75, 0.90 of total,
        #   offset by warmup → 0.75 and 0.90 of post-warmup epochs.
        step_fractions = config['MaskFormer'].get('lr_step_milestones', [0.75, 0.90])
        step_gamma     = config['MaskFormer'].get('lr_step_gamma', 0.1)
        milestones     = [max(1, int(f * post_warmup)) for f in step_fractions]
        print(f"LR schedule: MultiStepLR  milestones={milestones} (post-warmup epochs)  gamma={step_gamma}")
        main_sched = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=step_gamma
        )
    else:  # 'poly' — default, matches original WarmupPolyLR from Detectron2
        poly_power = config['MaskFormer'].get('lr_poly_power', 0.9)
        print(f"LR schedule: PolynomialLR  power={poly_power}  T={post_warmup} post-warmup epochs")
        main_sched = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=post_warmup, power=poly_power
        )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs]
    )

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
        scheduler=scheduler,
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