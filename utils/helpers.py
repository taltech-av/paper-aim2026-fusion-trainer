#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import torch
import numpy as np
import shutil
import datetime
import glob

def creat_dir(config):
    logdir = config['Log']['logdir']
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        print(f'Making log directory {logdir}...')
    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

def get_annotation_path(cam_path, dataset_name, config):
    """Get annotation path based on dataset and config."""
    if dataset_name in ['zod', 'iseauto']:
        anno_folder = config['Dataset']['annotation_path']
        return cam_path.replace('camera', anno_folder)
    else:  # waymo
        # Use same annotation path as training: /annotation/ directories
        return cam_path.replace('camera/', 'annotation/')

def relabel_annotation(annotation, config):
    """
    Relabel annotation from dataset indices to training indices.
    
    Uses the new config format with dataset_classes and train_classes.
    Supports class merging through dataset_mapping.
    
    Args:
        annotation: numpy array or torch tensor with dataset class indices
        config: configuration dictionary with Dataset.train_classes
        
    Returns:
        torch tensor with training indices [1, H, W]
    """
    annotation = np.array(annotation)
    
    train_classes = config['Dataset']['train_classes']
    
    # Find max dataset index to create mapping array
    max_dataset_index = max(
        max(mapping) for cls in train_classes 
        for mapping in [cls['dataset_mapping']]
    )
    
    # Create mapping from dataset index to training index
    # Default to 0 (background) for unmapped indices
    dataset_to_train_mapping = np.zeros(max_dataset_index + 1, dtype=int)
    
    for train_cls in train_classes:
        train_index = train_cls['index']
        for dataset_index in train_cls['dataset_mapping']:
            dataset_to_train_mapping[dataset_index] = train_index
    
    # Apply mapping
    relabeled = dataset_to_train_mapping[annotation]
    
    return torch.from_numpy(relabeled).unsqueeze(0).long()  # [H,W]->[1,H,W]


def draw_test_segmentation_map(outputs, config=None):
    """
    Create segmentation visualization with colors based on config class definitions.
    
    Args:
        outputs: Model output tensor
        config: Configuration dictionary with Dataset.train_classes containing color field.
    """
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    
    # Create color mapping based on config or use default
    if config is not None and 'train_classes' in config.get('Dataset', {}):
        train_classes = config['Dataset']['train_classes']
        
        # Create color list for training indices using colors from config
        color_list = []
        for cls in train_classes:
            # Use color from config if available, otherwise use default
            if 'color' in cls:
                color_list.append(tuple(cls['color']))
            else:
                # Fallback to hardcoded colors based on class name
                if cls['name'] == 'background':
                    color_list.append((0, 0, 0))  # Black
                elif cls['name'] == 'sign':
                    color_list.append((0, 0, 255))  # Blue
                elif cls['name'] == 'vehicle':
                    color_list.append((128, 0, 128))  # Purple
                elif cls['name'] == 'human':
                    color_list.append((255, 255, 0))  # Yellow
                else:
                    color_list.append((255, 255, 255))  # White fallback
    
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(color_list)):
        idx = labels == label_num
        red_map[idx] = color_list[label_num][0]
        green_map[idx] = color_list[label_num][1]
        blue_map[idx] = color_list[label_num][2]

    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image

def image_overlay(image, segmented_image):
    """
    Create overlay with transparent masks on original image.
    Only predicted classes are shown with transparency, background remains original.
    Both image and segmented_image should be in BGR format.
    """
    # Create a copy of the original image
    overlay = image.copy().astype(np.float32)

    # Find non-black pixels in segmented image (predicted classes)
    # Background is black [0, 0, 0] in BGR
    mask = np.any(segmented_image != [0, 0, 0], axis=2)

    # Apply alpha blending only to predicted regions
    alpha = 0.6  # transparency level
    overlay[mask] = alpha * segmented_image[mask].astype(np.float32) + (1 - alpha) * overlay[mask]

    return overlay.astype(np.uint8)

def get_all_checkpoint_paths(config, ignore_model_path=False):
    """Get all checkpoint file paths, sorted by epoch number.
    
    Args:
        config: Configuration dictionary
        ignore_model_path: If True, ignore config['General']['model_path'] and return all checkpoints
    """
    import glob
    import os
    
    # If model path is specified and we're not ignoring it, return just that one
    model_path = config['General'].get('model_path', '')
    if model_path != '' and not ignore_model_path:
        return [model_path]
    
    # Otherwise, find all checkpoints
    checkpoint_dir = os.path.join(config['Log']['logdir'], 'checkpoints')
    files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if len(files) == 0:
        return []
    
    # Sort by checkpoint number (not by file creation time which can be unreliable)
    def get_checkpoint_num(filepath):
        try:
            filename = os.path.basename(filepath)
            # Handle both old format (checkpoint_0.pth) and new format (epoch_0_uuid.pth)
            if filename.startswith('checkpoint_'):
                num_str = filename.replace('checkpoint_', '').replace('.pth', '')
            elif filename.startswith('epoch_'):
                # Extract epoch number from epoch_0_uuid.pth format
                parts = filename.replace('epoch_', '').replace('.pth', '').split('_')
                num_str = parts[0] if parts else '0'
            else:
                num_str = '0'
            return int(num_str)
        except:
            return 0
    
    # Sort files by epoch number
    sorted_files = sorted(files, key=get_checkpoint_num)
    return sorted_files

def get_best_checkpoint_path(config):
    """Find the checkpoint with the best validation mIoU."""
    import re
    logdir = config['Log']['logdir']
    epochs_dir = os.path.join(logdir, 'epochs')
    
    if not os.path.exists(epochs_dir):
        print(f"Epochs directory not found: {epochs_dir}")
        return None
    
    best_epoch = None
    best_miou = -1.0
    
    for file in os.listdir(epochs_dir):
        if file.endswith('.json'):
            filepath = os.path.join(epochs_dir, file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    val_miou = data['results']['val'].get('mean_iou', 0)
                    if val_miou > best_miou:
                        best_miou = val_miou
                        # Extract epoch and uuid from filename
                        match = re.search(r'epoch_(\d+)_([a-f0-9\-]+)\.json', file)
                        if match:
                            epoch_num = int(match.group(1))
                            epoch_uuid = match.group(2)
                            best_epoch = f"epoch_{epoch_num}_{epoch_uuid}.pth"
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
    
    if best_epoch:
        checkpoint_path = os.path.join(logdir, 'checkpoints', best_epoch)
        if os.path.exists(checkpoint_path):
            print(f"Found best checkpoint: {checkpoint_path} (val mIoU: {best_miou:.4f})")
            return checkpoint_path
        else:
            print(f"Best checkpoint file not found: {checkpoint_path}")
    
    return None

def get_model_path(config, best=False):
    """Get the model checkpoint file path. If best=True, get the best checkpoint."""
    if best:
        return get_best_checkpoint_path(config)
    else:
        checkpoint_paths = get_all_checkpoint_paths(config)
        if not checkpoint_paths:
            return False
        # Return the latest checkpoint (last in sorted list)
        return checkpoint_paths[-1]

def get_checkpoint_path_with_fallback(config):
    """Get the best checkpoint path, or fall back to the latest checkpoint if best is not found."""
    # Try to get the best checkpoint first
    checkpoint_path = get_best_checkpoint_path(config)
    if checkpoint_path:
        return checkpoint_path
    
    # Fall back to the latest checkpoint
    checkpoint_paths = get_all_checkpoint_paths(config, ignore_model_path=True)
    if checkpoint_paths:
        return checkpoint_paths[-1]  # Latest checkpoint
    
    return None

def save_model_dict(config, epoch, model, optimizer, epoch_uuid=None):
    creat_dir(config)
    if epoch_uuid:
        filename = f"epoch_{epoch}_{epoch_uuid}.pth"
    else:
        filename = f"checkpoint_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(config['Log']['logdir'], 'checkpoints', filename)
    )

def get_model_config_key(config):
    """Get the model configuration key (CLFT, SwinFusion, etc.)"""
    backbone = config['CLI']['backbone']
    if backbone == 'clft':
        return 'CLFT'
    elif backbone == 'swin_fusion':
        return 'SwinFusion'
    elif backbone == 'maskformer':
        return 'MaskFormer'
    elif backbone == 'mask2former':
        return 'Mask2Former'
    else:
        return 'CLFT'  # default

def adjust_learning_rate(config, optimizer, epoch):
    """Decay the learning rate based on schedule.

    If 'lr_momentum' is absent from the model config block (e.g. when an
    external scheduler already manages the LR), the function is a no-op and
    returns the current LR from the optimizer.
    """
    model_key = get_model_config_key(config)
    momentum = config[model_key].get('lr_momentum', None)
    if momentum is None:
        # Scheduler-managed LR — just read and return current value
        return optimizer.param_groups[0]['lr']
    lr = config[model_key]['clft_lr'] * (momentum ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def manage_checkpoints_by_miou(config, log_dir):
    """Keep only the top max_checkpoints checkpoints based on validation mIoU from JSON files.
    
    Only deletes .pth checkpoint files, preserving JSON files for analysis and testing.
    """
    max_checkpoints = config['General'].get('max_epochs', 10)  # Default to 10 if not specified
    import glob
    import os
    import json
    
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    epochs_dir = os.path.join(log_dir, 'epochs')
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, '*.pth')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if len(checkpoint_files) <= max_checkpoints:
        return  # No need to delete anything
    
    # Get validation mIoU for each checkpoint from JSON files
    checkpoints_with_miou = []
    
    for checkpoint_path in checkpoint_files:
        checkpoint_filename = os.path.basename(checkpoint_path)
        
        # Extract epoch and uuid from checkpoint filename (format: epoch_X_uuid.pth)
        if checkpoint_filename.startswith('epoch_'):
            parts = checkpoint_filename.replace('epoch_', '').replace('.pth', '').split('_')
            if len(parts) >= 2:
                epoch_num = int(parts[0])
                epoch_uuid = parts[1]
                
                # Find corresponding JSON file
                json_pattern = os.path.join(epochs_dir, f'epoch_{epoch_num}_{epoch_uuid}.json')
                json_files = glob.glob(json_pattern)
                
                if json_files:
                    try:
                        with open(json_files[0], 'r') as f:
                            json_data = json.load(f)
                        
                        # Get validation mIoU
                        val_miou = json_data.get('results', {}).get('val', {}).get('mean_iou', -1.0)
                        
                        checkpoints_with_miou.append({
                            'path': checkpoint_path,
                            'epoch': epoch_num,
                            'uuid': epoch_uuid,
                            'miou': val_miou
                        })
                    except (json.JSONDecodeError, KeyError, FileNotFoundError):
                        # If we can't read the JSON, treat as lowest priority
                        checkpoints_with_miou.append({
                            'path': checkpoint_path,
                            'epoch': epoch_num,
                            'uuid': epoch_uuid,
                            'miou': -1.0
                        })
                else:
                    # No JSON file found, treat as lowest priority
                    checkpoints_with_miou.append({
                        'path': checkpoint_path,
                        'epoch': epoch_num,
                        'uuid': epoch_uuid,
                        'miou': -1.0
                    })
            else:
                # Can't parse filename, treat as lowest priority
                checkpoints_with_miou.append({
                    'path': checkpoint_path,
                    'epoch': -1,
                    'uuid': 'unknown',
                    'miou': -1.0
                })
        else:
            # Not an epoch checkpoint, treat as lowest priority
            checkpoints_with_miou.append({
                'path': checkpoint_path,
                'epoch': -1,
                'uuid': 'unknown',
                'miou': -1.0
            })
    
    # Sort by validation mIoU descending (highest first)
    checkpoints_with_miou.sort(key=lambda x: x['miou'], reverse=True)
    
    # Keep only top max_checkpoints
    to_delete = checkpoints_with_miou[max_checkpoints:]
    
    for checkpoint_info in to_delete:
        try:
            # Delete the checkpoint file only (keep JSON files)
            os.remove(checkpoint_info['path'])
            print(f"Deleted checkpoint: {os.path.basename(checkpoint_info['path'])} (mIoU: {checkpoint_info['miou']:.4f})")
        except OSError as e:
            print(f"Error deleting checkpoint {checkpoint_info['path']}: {e}")

class EarlyStopping(object):
    def __init__(self, config):
        self.patience = config['General']['early_stop_patience']
        self.config = config
        self.min_param = None
        self.early_stop_trigger = False
        self.count = 0

    def __call__(self, valid_param, epoch, model, optimizer, epoch_uuid=None):
        if self.min_param is None:
            self.min_param = valid_param
        elif valid_param >= self.min_param:
            self.count += 1
            print(f'Early Stopping Counter: {self.count} of {self.patience}')
            if self.count >= self.patience:
                self.early_stop_trigger = True
                print('Saving model for last epoch...')
                save_model_dict(self.config, epoch, model, optimizer, epoch_uuid)
                print('Saving Model Complete')
                print('Early Stopping Triggered!')
        else:
            print(f'Valid loss decreased from {self.min_param:.4f} ' + f'to {valid_param:.4f}')
            self.min_param = valid_param
            # No need to save additional checkpoint - we save every epoch now
            self.count = 0

def create_config_snapshot():
    source_file = 'config.json'
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    destination_file = f'config_{timestamp}.json'
    shutil.copy(source_file, destination_file)
    print(f'Config snapshot created {destination_file}')


def sanitize_for_json(data):
    """Recursively sanitize data for JSON serialization (handle NaN/Inf)."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return 0.0
        return data
    elif isinstance(data, (np.float32, np.float64)):
        if np.isnan(data) or np.isinf(data):
            return 0.0
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    return data


def get_training_uuid_from_logs(log_dir):
    """Extract training_uuid and vision_training_id from existing epoch log files."""
    epochs_dir = os.path.abspath(os.path.join(log_dir, 'epochs'))
    if not os.path.exists(epochs_dir):
        return None, None
    
    # Find all epoch JSON files
    epoch_files = glob.glob(os.path.join(epochs_dir, 'epoch_*.json'))
    if not epoch_files:
        return None, None
    
    # Get the most recent epoch file
    epoch_files.sort()
    latest_epoch_file = epoch_files[-1]
    
    try:
        with open(latest_epoch_file, 'r') as f:
            epoch_data = json.load(f)
        training_uuid = epoch_data.get('training_uuid')
        vision_training_id = epoch_data.get('vision_training_id')
        if training_uuid:
            print(f"Found existing training_uuid from logs: {training_uuid}")
            if vision_training_id:
                print(f"Found existing vision_training_id from logs: {vision_training_id}")
            return training_uuid, vision_training_id
    except Exception as e:
        print(f"Warning: Could not read training data from {latest_epoch_file}: {e}")
    
    return None, None
