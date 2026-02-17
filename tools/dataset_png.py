#!/usr/bin/env python3

"""
Modified dataset loader that uses pre-computed PNG projections instead of pickle files.

PNG Format Specification:
- RGB image where R=X, G=Y, B=Z coordinate projections
- Values: 1-255 (uint8) for normalized coordinates, 0 for empty pixels
- Original dimensions: 1363x768 for ZOD, 1920x1280 for Waymo
- File naming: frame_XXXXXX.png (same as original pickle but .png extension)
"""

import os
import torch
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from utils.lidar_process import *
from utils.helpers import get_annotation_path


class DatasetPNG(Dataset):
    """
    Dataset class for loading LiDAR projections from PNG files.
    Replaces the original Dataset class but uses PNG files instead of pickle files.
    """

    def __init__(self, config, split=None, path=None):
        """
        Initialize dataset.

        Args:
            config (dict): Configuration dictionary
            split (str): Data split ('train', 'val', 'test')
            path (str): Path to split file (e.g., 'train.txt')
        """
        np.random.seed(789)
        self.config = config

        # Read the split file to get list of examples
        list_examples_file = open(path, 'r')
        self.list_examples_cam = np.array(list_examples_file.read().splitlines())
        list_examples_file.close()

        # Set augmentation probabilities based on split
        if split == 'train':  # only augment for training.
            self.p_flip = config['Dataset']['transforms']['p_flip']
            self.p_crop = config['Dataset']['transforms']['p_crop']
            self.p_rot = config['Dataset']['transforms']['p_rot']
        else:
            self.p_flip = 0
            self.p_crop = 0
            self.p_rot = 0

        self.img_size = config['Dataset']['transforms']['resize']
        self.rgb_normalize = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['Dataset']['transforms']['image_mean'],
                std=config['Dataset']['transforms']['image_std'])
        ])

        # For Swin models, annotations should match the output resolution (resample_dim)
        if 'SwinFusion' in config:
            self.anno_size = config['SwinFusion']['resample_dim']
        else:
            self.anno_size = self.img_size

        self.anno_resize = transforms.Resize((self.anno_size, self.anno_size),
                                           interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        """Return dataset length."""
        return len(self.list_examples_cam)

    def __getitem__(self, idx):
        """
        Get item from dataset.

        Args:
            idx (int): Index of the item

        Returns:
            dict: Dictionary containing processed data
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dataroot = self.config['Dataset']['dataset_root']

        # Construct paths (generic approach)
        cam_path = os.path.join(dataroot, self.list_examples_cam[idx])
        dataset_name = self.config['Dataset']['name']
        
        # Try to get annotation path - if it doesn't exist, we'll create dummy annotations
        anno_path = None
        try:
            anno_path = get_annotation_path(cam_path, dataset_name, self.config)
            # Check if annotation file actually exists
            if os.path.exists(anno_path):
                anno = torch.from_numpy(np.array(Image.open(anno_path))).unsqueeze(0).long()
            else:
                # Create dummy annotations if file doesn't exist
                anno = torch.zeros((1, 1920, 1280), dtype=torch.long)
        except:
            # Fallback to dummy annotations
            anno = torch.zeros((1, 1920, 1280), dtype=torch.long)

        # Generic PNG path replacement (handles both /camera and /camera/ patterns)
        png_path = cam_path.replace('/camera', '/lidar_png').replace('/camera/', '/lidar_png/')

        rgb = Image.open(cam_path).convert('RGB')

        # Load LiDAR projection from PNG only if not in RGB-only mode
        rgb_only = self.config.get('CLI', {}).get('mode') == 'rgb'
        if rgb_only:
            # For RGB-only mode, create a dummy LiDAR image
            lidar_pil = Image.new('RGB', rgb.size, (0, 0, 0))
        else:
            # Load LiDAR projection from PNG
            lidar_pil = self.load_lidar_png(png_path)

        # Validate filenames match
        rgb_name = cam_path.split('/')[-1].split('.')[0]
        # Only validate annotation name match if we loaded real annotations (not dummy)
        if anno_path and os.path.exists(anno_path):
            anno_name = anno_path.split('/')[-1].split('.')[0]
            assert (rgb_name == anno_name), "rgb and anno input not matching"
        if not rgb_only:
            png_name = png_path.split('/')[-1].split('.')[0]
            assert (rgb_name == png_name), "rgb and png input not matching"

        # Keep full image (no cropping)
        rgb_orig = rgb.copy()
        rgb_aug = rgb
        anno_aug = anno
        lidar_aug = lidar_pil

        # Apply horizontal flip
        if random.random() < self.p_flip:
            rgb_aug = TF.hflip(rgb_aug)
            anno_aug = TF.hflip(anno_aug)
            lidar_aug = TF.hflip(lidar_aug)

        # Apply random crop
        if random.random() < self.p_crop:
            # Use resized crop for more variation (like DataAugment)
            random_size = random.randint(128, self.img_size - 1)
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                rgb_aug, scale=(0.2, 1.0), ratio=(3./4., 4./3.))
            rgb_aug = TF.resized_crop(rgb_aug, i, j, h, w, (random_size, random_size),
                                     interpolation=TF.InterpolationMode.BILINEAR)
            anno_aug = TF.resized_crop(anno_aug, i, j, h, w, (random_size, random_size),
                                      interpolation=TF.InterpolationMode.NEAREST)
            lidar_aug = TF.resized_crop(lidar_aug, i, j, h, w, (random_size, random_size),
                                       interpolation=TF.InterpolationMode.BILINEAR)

        # Apply random rotation
        if random.random() < self.p_rot:
            rotate_range = self.config['Dataset']['transforms']['random_rotate_range']
            angle = (-rotate_range + 2 * rotate_range * torch.rand(1)[0]).item()
            rgb_aug = TF.rotate(rgb_aug, angle)
            anno_aug = TF.rotate(anno_aug, angle, interpolation=TF.InterpolationMode.NEAREST)
            lidar_aug = TF.rotate(lidar_aug, angle)

        # Normalize and resize (same as original)
        rgb = self.rgb_normalize(rgb_aug)  # Tensor [3, 384, 384]
        anno = self.anno_resize(anno_aug).squeeze(0)  # Tensor [384, 384]

        rgb_orig = transforms.ToTensor()(rgb_orig)

        # Resize lidar to match image size and convert to tensor
        lidar_resized = transforms.Resize((self.img_size, self.img_size))(lidar_aug)
        lidar_tensor = TF.to_tensor(lidar_resized)

        # Create dummy camera coordinates (not used in PNG version)
        max_len = 300000
        padded_coord = torch.full((max_len, 2), -1, dtype=torch.float)

        return {
            'rgb': rgb,
            'rgb_orig': rgb_orig,
            'lidar': lidar_tensor,
            'anno': anno,
            'camera_coord': padded_coord
        }

    def load_lidar_png(self, png_path):
        """
        Load LiDAR projection from PNG file.

        Args:
            png_path (str): Path to PNG file containing LiDAR projection

        Returns:
            PIL.Image: LiDAR projection as PIL image
        """
        # Load PNG image
        img = Image.open(png_path)
        return img