#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground truth visualization script.
Creates overlays of ground truth annotations on camera images.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2

from core.data_loader import DataLoader as InferenceDataLoader
from utils.helpers import get_annotation_path, relabel_annotation


class GroundTruthVisualizer:
    """Handles visualization of ground truth annotations."""
    
    def __init__(self, config, output_base):
        self.config = config
        self.output_base = output_base
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Create output directories for visualizations."""
        self.dirs = {
            'ground_truth': os.path.join(self.output_base, 'ground_truth')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def visualize_ground_truth(self, rgb_path, anno_path, idx):
        """
        Create ground truth overlay visualization.
        
        Args:
            rgb_path: Path to original RGB image
            anno_path: Path to ground truth annotation
            idx: Image index for logging
        """
        # Load original image
        rgb_cv2 = cv2.imread(rgb_path)
        if rgb_cv2 is None:
            print(f'Warning: Could not load RGB image {rgb_path}')
            return
        
        h, w = rgb_cv2.shape[:2]
        
        # Load and process ground truth
        gt_anno = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
        if gt_anno is None:
            print(f'Warning: Could not load annotation for {anno_path}')
            return
        
        # Resize ground truth to match image dimensions
        gt_anno_resized = cv2.resize(gt_anno, (w, h), 
                                     interpolation=cv2.INTER_NEAREST)
        gt_anno_tensor = torch.from_numpy(gt_anno_resized).unsqueeze(0).long()
        gt_relabeled = relabel_annotation(gt_anno_tensor, self.config)
        
        # Create colored segmentation map from ground truth
        gt_labels = gt_relabeled.squeeze(0).squeeze(0).numpy()
        gt_segmented = self._draw_segmentation_map_from_labels(gt_labels, self.config)
        
        # Convert RGB to BGR for OpenCV
        gt_segmented_bgr = cv2.cvtColor(gt_segmented, cv2.COLOR_RGB2BGR)
        
        # Resize to match original image
        gt_segmented_resized = cv2.resize(gt_segmented_bgr, (w, h), 
                                         interpolation=cv2.INTER_NEAREST)
        
        # Create overlay with transparency
        overlay = self._create_overlay(rgb_cv2, gt_segmented_resized, alpha=0.6)
        
        # Save overlay
        output_path = self._get_output_path('ground_truth', rgb_path)
        print(f'Saving ground truth overlay {idx}...')
        cv2.imwrite(output_path, overlay)
    
    def _create_overlay(self, image, segmented_image, alpha=0.6):
        """
        Create overlay with transparent masks on original image.
        Both image and segmented_image should be in BGR format.
        
        Args:
            image: Original image
            segmented_image: Segmented image to overlay
            alpha: Transparency level
        """
        # Create a copy of the original image
        overlay = image.copy().astype(np.float32)

        # Find non-black pixels in segmented image (predicted classes)
        mask = np.any(segmented_image != [0, 0, 0], axis=2)

        # Apply alpha blending only to predicted regions
        overlay[mask] = alpha * segmented_image[mask].astype(np.float32) + (1 - alpha) * overlay[mask]

        return overlay.astype(np.uint8)
    
    def _draw_segmentation_map_from_labels(self, labels, config=None):
        """
        Create segmentation visualization from class labels with colors based on config.
        
        Args:
            labels: Class label array (H, W)
            config: Configuration dictionary with Dataset.train_classes containing color field.
        """
        # Create color mapping based on config or use default
        if config is not None and 'train_classes' in config.get('Dataset', {}):
            train_classes = config['Dataset']['train_classes']
            
            # Create color list for training indices using colors from config
            color_list = []
            for cls in train_classes:
                # Use color from config if available, otherwise use default
                if 'color' in cls:
                    # Config colors are in RGB, convert to BGR for OpenCV
                    rgb_color = cls['color']
                    color_list.append((rgb_color[2], rgb_color[1], rgb_color[0]))  # BGR
                else:
                    # Fallback to hardcoded colors in BGR format
                    if cls['name'] == 'background':
                        color_list.append((0, 0, 0))  # Black
                    elif cls['name'] == 'sign':
                        color_list.append((255, 0, 0))  # Red (BGR)
                    elif cls['name'] == 'vehicle':
                        color_list.append((128, 0, 128))  # Purple (BGR)
                    elif cls['name'] == 'human':
                        color_list.append((0, 255, 255))  # Yellow (BGR)
                    else:
                        color_list.append((255, 255, 255))  # White fallback
        
        blue_map = np.zeros_like(labels).astype(np.uint8)
        green_map = np.zeros_like(labels).astype(np.uint8)
        red_map = np.zeros_like(labels).astype(np.uint8)

        for label_num in range(0, len(color_list)):
            idx = labels == label_num
            blue_map[idx] = color_list[label_num][0]
            green_map[idx] = color_list[label_num][1]
            red_map[idx] = color_list[label_num][2]

        segmented_image = np.stack([blue_map, green_map, red_map], axis=2)
        return segmented_image
    
    def _get_output_path(self, output_type, input_path):
        """Get output path for a given visualization type."""
        filename = os.path.basename(input_path)
        return os.path.join(self.dirs[output_type], filename)


def load_image_paths(path_arg, dataroot, dataset_name):
    """Load image paths from frame number, file path, or text file."""
    if not path_arg.startswith('frame_'):
        # Handle file paths as before
        if path_arg.endswith(('.png', '.jpg', '.jpeg')):
            # Single image
            return [path_arg]
        else:
            # Text file with multiple paths
            with open(path_arg, 'r') as f:
                paths = f.read().splitlines()
            return paths
    else:
        # Frame number - construct camera path
        if dataset_name == 'zod':
            cam_path = f'camera/{path_arg}.png'
        else:  # waymo
            cam_path = f'camera/{path_arg}.png'  # Adjust if different for waymo
        return [cam_path]


def process_images(data_loader, visualizer, image_paths, dataroot, dataset_name, config):
    """Process and visualize all images."""
    for idx, path in enumerate(image_paths, 1):
        # Construct full paths
        if os.path.isabs(path):
            cam_path = path
        else:
            cam_path = os.path.join(dataroot, path)
        
        anno_path = get_annotation_path(cam_path, dataset_name, config)
        
        # Verify paths match
        rgb_name = os.path.basename(cam_path).split('.')[0]
        anno_name = os.path.basename(anno_path).split('.')[0]
        
        assert rgb_name == anno_name, f"RGB and annotation names don't match: {rgb_name} vs {anno_name}"
        
        print(f'Processing image {idx}/{len(image_paths)}: {rgb_name}')
        
        # Visualize ground truth
        visualizer.visualize_ground_truth(cam_path, anno_path, idx)
    
    print(f'\nCompleted ground truth visualization of {len(image_paths)} images')


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize Ground Truth Annotations')
    parser.add_argument('-p', '--path', type=str, required=True,
                       help='Frame number (e.g., frame_017119), image path, or text file with paths')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup device (not needed for ground truth visualization)
    device = torch.device("cpu")
    
    # Extract config name for output directory
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    dataset_name = config['Dataset']['name']
    output_base = f'logs/{dataset_name}/{config_name}/visualizations'
    
    # Setup data loader (for potential future use)
    data_loader = InferenceDataLoader(config)
    
    # Setup visualizer
    visualizer = GroundTruthVisualizer(config, output_base)
    
    # Load image paths
    dataroot = os.path.abspath(config['Dataset']['dataset_root'])
    image_paths = load_image_paths(args.path, dataroot, dataset_name)
    print(f"Found {len(image_paths)} images to process")
    
    # Process images
    process_images(data_loader, visualizer, image_paths, dataroot, dataset_name, config)


if __name__ == '__main__':
    main()