#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
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
    if not os.path.exists(logdir + 'checkpoints'):
        os.makedirs(logdir + 'checkpoints')

def get_annotation_path(cam_path, dataset_name, config):
    """Get annotation path based on dataset and config."""
    if dataset_name == 'zod':
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

def get_model_path(config):
    model_path = config['General']['model_path']
    if model_path != '':
        return config['General']['model_path']
    # If model path not specified then take latest checkpoint
    files = glob.glob(config['Log']['logdir']+'checkpoints/*.pth')
    if len(files) == 0:
        return False
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
    
    latest_file = max(files, key=get_checkpoint_num)
    return latest_file

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
        config['Log']['logdir']+'checkpoints/'+filename
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
    else:
        return 'CLFT'  # default

def adjust_learning_rate(config, optimizer, epoch):
    """Decay the learning rate based on schedule"""
    epoch_max = config['General']['epochs']
    model_key = get_model_config_key(config)
    momentum = config[model_key]['lr_momentum']
    # lr = config['General']['dpt_lr'] * (1-epoch/epoch_max)**0.9
    lr = config[model_key]['clft_lr'] * (momentum ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

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
            # Check if this epoch will also be saved as a regular checkpoint
            save_epoch = self.config['General']['save_epoch']
            if not (epoch == 0 or (epoch + 1) % save_epoch == 0):
                save_model_dict(self.config, epoch, model, optimizer, epoch_uuid)
                print('Saving Model...')
            else:
                print('Skipping early stopping save (regular checkpoint will be saved)')
            self.count = 0

def create_config_snapshot():
    source_file = 'config.json'
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    destination_file = f'config_{timestamp}.json'
    shutil.copy(source_file, destination_file)
    print(f'Config snapshot created {destination_file}')
