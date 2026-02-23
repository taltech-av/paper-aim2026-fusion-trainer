#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training engine for model training.
"""
import time
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from utils.helpers import relabel_annotation, adjust_learning_rate, save_model_dict, EarlyStopping, manage_checkpoints_by_miou
from integrations.training_logger import log_epoch_results
from integrations.vision_service import send_epoch_results_from_file
from utils.system_monitor import get_epoch_system_snapshot


class TrainingEngine:
    """Handles the training loop and epoch execution."""
    
    def __init__(self, model, optimizer, criterion, metrics_calculator, config, 
                 training_uuid, log_dir, device, vision_training_id=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics_calc = metrics_calculator
        self.config = config
        self.training_uuid = training_uuid
        self.log_dir = log_dir
        self.device = device
        self.vision_training_id = vision_training_id
        self.writer = SummaryWriter()
        self.early_stopping = EarlyStopping(config)
        self.scaler = GradScaler('cuda')
    
    def train_epoch(self, dataloader, modality, num_classes):
        """Execute one training epoch."""
        self.model.train()
        
        accumulators = self.metrics_calc.create_accumulators(self.device)
        train_loss = 0.0
        
        progress_bar = tqdm(dataloader)
        for batch in progress_bar:
            # Move data to device
            rgb = batch['rgb'].to(self.device, non_blocking=True)
            lidar = batch['lidar'].to(self.device, non_blocking=True)
            anno = batch['anno'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Prepare inputs based on modality
            rgb_input, lidar_input = self._prepare_inputs(rgb, lidar, modality)
            
            # Forward pass with mixed precision
            with autocast('cuda'):
                model_outputs = self.model(rgb_input, lidar_input, modality)
                if modality == 'cross_fusion':
                    # For fusion: check if first output is None (depth models)
                    if model_outputs[0] is None:
                        output_seg = model_outputs[1]  # segmentation output
                    else:
                        output_seg = model_outputs[0]  # direct segmentation output
                else:
                    # For single modality: check if first output is None (depth models)
                    if model_outputs[0] is None:
                        output_seg = model_outputs[1]  # segmentation output
                    else:
                        output_seg = model_outputs[0]  # direct segmentation output
                output_seg = output_seg.squeeze(1)
                
                # Relabel annotation
                anno = relabel_annotation(anno.cpu(), self.config).squeeze(0).to(self.device)
                
                # Compute loss
                loss = self.criterion(output_seg, anno)
            
            # Update metrics (outside autocast for precision)
            self.metrics_calc.update_accumulators(
                accumulators, output_seg, anno, num_classes
            )
            
            train_loss += loss.item()
            
            # Backprop with scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            progress_bar.set_description(f'Train loss: {loss:.4f}')
        
        # Compute epoch metrics
        metrics = self.metrics_calc.compute_epoch_metrics(
            accumulators, train_loss, len(dataloader)
        )
        
        return metrics
    
    def validate_epoch(self, dataloader, modality, num_classes):
        """Execute one validation epoch."""
        self.model.eval()
        
        accumulators = self.metrics_calc.create_accumulators(self.device)
        valid_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader)
            for batch in progress_bar:
                # Move data to device
                rgb = batch['rgb'].to(self.device, non_blocking=True)
                lidar = batch['lidar'].to(self.device, non_blocking=True)
                anno = batch['anno'].to(self.device, non_blocking=True)
                
                # Prepare inputs based on modality
                rgb_input, lidar_input = self._prepare_inputs(rgb, lidar, modality)
                
                # Forward pass with mixed precision
                with autocast('cuda'):
                    model_outputs = self.model(rgb_input, lidar_input, modality)
                    if modality == 'cross_fusion':
                        # For fusion: check if first output is None (depth models)
                        if model_outputs[0] is None:
                            output_seg = model_outputs[1]  # segmentation output
                        else:
                            output_seg = model_outputs[0]  # direct segmentation output
                    else:
                        # For single modality: check if first output is None (depth models)
                        if model_outputs[0] is None:
                            output_seg = model_outputs[1]  # segmentation output
                        else:
                            output_seg = model_outputs[0]  # direct segmentation output
                    output_seg = output_seg.squeeze(1)
                    
                    # Relabel annotation
                    anno = relabel_annotation(anno.cpu(), self.config).squeeze(0).to(self.device)
                    
                    # Compute loss
                    loss = self.criterion(output_seg, anno)
                
                # Update metrics
                self.metrics_calc.update_accumulators(
                    accumulators, output_seg, anno, num_classes
                )
                
                valid_loss += loss.item()
                
                progress_bar.set_description(f'Valid loss: {loss:.4f}')
        
        # Compute epoch metrics
        metrics = self.metrics_calc.compute_epoch_metrics(
            accumulators, valid_loss, len(dataloader)
        )
        
        return metrics
    
    def _prepare_inputs(self, rgb, lidar, modality):
        """Prepare model inputs based on training modality."""
        if modality == 'rgb':
            return rgb, rgb
        elif modality == 'lidar':
            return lidar, lidar
        else:  # cross_fusion
            return rgb, lidar
    
    def train_full(self, train_dataloader, valid_dataloader, modality, num_classes, start_epoch=0):
        """Execute full training loop."""
        epochs = self.config['General']['epochs']
        
        last_epoch_uuid = None
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            # Adjust learning rate
            lr = adjust_learning_rate(self.config, self.optimizer, epoch)
            print(f'Epoch: {epoch}, LR: {lr:.6f}')
            
            # Train
            print('Training...')
            train_metrics = self.train_epoch(train_dataloader, modality, num_classes)
            self.metrics_calc.print_metrics(train_metrics, prefix="Training ")
            
            # Validate
            print('Validating...')
            val_metrics = self.validate_epoch(valid_dataloader, modality, num_classes)
            self.metrics_calc.print_metrics(val_metrics, prefix="Validation ")
            
            epoch_time = time.time() - epoch_start_time
            
            # Capture system info
            system_info = get_epoch_system_snapshot()
            
            # Log to tensorboard
            self._log_tensorboard(train_metrics, val_metrics, epoch)
            
            # Log and upload results
            epoch_uuid = self._log_and_upload_results(
                epoch, train_metrics, val_metrics, lr, epoch_time, system_info
            )
            last_epoch_uuid = epoch_uuid  # Store for final checkpoint
            
            # Save checkpoints
            self._handle_checkpoints(epoch, val_metrics, epoch_uuid)
            
            # Check early stopping — based on val mIoU (higher=better; negate for min-tracking)
            early_stop_index = -round(val_metrics['mean_iou'], 4)
            self.early_stopping(early_stop_index, epoch, self.model, self.optimizer, epoch_uuid)

            if self.early_stopping.early_stop_trigger:
                break
        
        # Save final checkpoint with UUID from last epoch
        print('Saving final model checkpoint...')
        final_epoch = epochs - 1 if epochs > 0 else 0
        save_model_dict(self.config, final_epoch, self.model, self.optimizer, last_epoch_uuid)
        print('Training Complete')
    
    def _log_tensorboard(self, train_metrics, val_metrics, epoch):
        """Log metrics to tensorboard."""
        self.writer.add_scalars('Loss', {
            'train': train_metrics['epoch_loss'], 
            'valid': val_metrics['epoch_loss']
        }, epoch)
        
        for i, cls in enumerate(self.metrics_calc.eval_classes):
            self.writer.add_scalars(f'{cls}_IoU', {
                'train': train_metrics['epoch_IoU'][i], 
                'valid': val_metrics['epoch_IoU'][i]
            }, epoch)
        
        self.writer.close()
    
    def _log_and_upload_results(self, epoch, train_metrics, val_metrics, lr, epoch_time, system_info=None):
        """Log results locally and upload to vision service."""
        epoch_uuid = None
        
        if self.training_uuid and self.log_dir:
            results = self.metrics_calc.prepare_results_dict(train_metrics, val_metrics)
            logged_file_path = log_epoch_results(
                epoch, self.training_uuid, results, self.log_dir, 
                learning_rate=lr, epoch_time=epoch_time, system_info=system_info,
                vision_training_id=self.vision_training_id
            )
            
            # Extract epoch UUID
            import os
            epoch_uuid = os.path.basename(logged_file_path).replace(
                f'epoch_{epoch}_', ''
            ).replace('.json', '')
        
        return epoch_uuid
    
    def _handle_checkpoints(self, epoch, val_metrics, epoch_uuid):
        """Handle checkpoint saving and early stopping."""
        # Save checkpoint every epoch
        print('Saving model checkpoint...')
        save_model_dict(self.config, epoch, self.model, self.optimizer, epoch_uuid)
        print('Checkpoint saved')
        
        # Manage checkpoints: keep only top max_checkpoints by validation mIoU
        manage_checkpoints_by_miou(self.config, self.log_dir)
        # Early stopping is handled in train_full
