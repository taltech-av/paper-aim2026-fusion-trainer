#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored testing script using modular components.
"""
import json
import argparse
import uuid
import datetime
import re
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from core.model_builder import ModelBuilder
from core.metrics_calculator import MetricsCalculator
from core.testing_engine import TestingEngine
from utils.metrics import find_overlap_exclude_bg_ignore
from integrations.vision_service import send_test_results_from_file
from utils.helpers import get_all_checkpoint_paths, sanitize_for_json


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


def setup_dataset(config):
    """Setup dataset based on configuration."""
    from tools.dataset_png import DatasetPNG as Dataset
    return Dataset


def setup_overlap_function(config):
    """Setup dataset-specific overlap calculation function."""
    dataset_name = config['Dataset']['name']
    if dataset_name in ['zod', 'waymo']:
        print(f"Using unified IoU calculation (excludes background only)")
        return find_overlap_exclude_bg_ignore



def extract_epoch_info(model_path):
    """Extract epoch number and UUID from model path."""
    epoch_match = re.search(r'epoch_(\d+)_([a-f0-9\-]+)\.pth', model_path)
    if epoch_match:
        return int(epoch_match.group(1)), epoch_match.group(2)
    
    # Fallback for old checkpoint format
    epoch_match = re.search(r'checkpoint_(\d+)\.pth', model_path)
    if epoch_match:
        return int(epoch_match.group(1)), None
    
    return 0, None


def save_test_results(config, all_results, epoch_num, epoch_uuid, test_uuid):
    """Save test results to JSON file."""
    combined_results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'epoch': epoch_num,
        'epoch_uuid': epoch_uuid,
        'test_uuid': test_uuid,
        'test_results': all_results
    }
    
    # Create results directory
    test_results_dir = os.path.join(config['Log']['logdir'], 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    # Generate filename
    if epoch_uuid:
        filename = f'epoch_{epoch_num}_{epoch_uuid}.json'
    else:
        filename = f'epoch_{epoch_num}_test_results.json'
    
    filepath = os.path.join(test_results_dir, filename)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(sanitize_for_json(combined_results), f, indent=2)
    
    print(f'All test results saved to: {filepath}')
    if epoch_uuid:
        print(f'Epoch UUID: {epoch_uuid}')
    print(f'Test UUID: {test_uuid}')
    
    return filepath


def run_test_suite(tester, config, test_data_files, test_data_path, num_classes):
    """Run tests on all test files and collect results."""
    Dataset = setup_dataset(config)
    all_results = {}
    
    for file in test_data_files:
        path = os.path.join(test_data_path, file)
        print(f"Testing with: {path}")
        
        # Check if test file exists
        if not os.path.exists(path):
            print(f"Test file not found: {path}, skipping")
            continue
        
        # Create dataset and dataloader
        test_data = Dataset(config, 'test', path)
        
        # Check if test file has zero rows
        if len(test_data) == 0:
            print(f"Test file is empty: {path}, skipping")
            continue
        
        test_dataloader = DataLoader(
            test_data,
            batch_size=config['General']['batch_size'],
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            persistent_workers=True
        )
        
        # Get modality
        modality = config['CLI']['mode']
        
        # Run test
        results, inference_stats = tester.test(test_dataloader, modality, num_classes)
        
        # Extract weather condition from filename
        weather_condition = file.replace('test_', '').replace('.txt', '')
        all_results[weather_condition] = results
        
        print(f'Testing completed for {weather_condition}\n')
    
    return all_results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='CLFT Testing (Refactored)')
    parser.add_argument('-c', '--config', type=str, required=False,
                       default='config.json', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific model checkpoint (optional, tests all if not specified)')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set random seed
    np.random.seed(config['General']['seed'])
    
    # Setup device
    device = torch.device(config['General']['device']
                         if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get checkpoint paths
    if args.checkpoint:
        checkpoint_paths = [args.checkpoint]
    else:
        checkpoint_paths = get_all_checkpoint_paths(config, ignore_model_path=True)
    
    if not checkpoint_paths:
        print("No model checkpoints found. Please train the model first.")
        exit(1)
    
    print(f"Found {len(checkpoint_paths)} checkpoints to test")
    
    # Test each checkpoint
    for i, model_path in enumerate(checkpoint_paths):
        print(f"\n{'='*50}")
        print(f"Testing checkpoint {i+1}/{len(checkpoint_paths)}: {model_path}")
        print(f"{'='*50}")
        
        # Extract epoch info
        epoch_num, epoch_uuid = extract_epoch_info(model_path)
        test_uuid = str(uuid.uuid4())
        
        # Calculate class counts
        num_classes = calculate_num_classes(config)
        num_eval_classes = calculate_num_eval_classes(config, num_classes)
        print(f"Total classes: {num_classes}, Evaluation classes: {num_eval_classes}")
        
        # Build and load model
        model_builder = ModelBuilder(config, device)
        model = model_builder.build_model()
        model, _ = model_builder.load_checkpoint(model, model_path)
        model.eval()
        
        # Setup overlap function
        find_overlap_func = setup_overlap_function(config)
        
        # Setup metrics calculator
        metrics_calc = MetricsCalculator(config, num_eval_classes, find_overlap_func)
        
        # Setup testing engine
        tester = TestingEngine(model, metrics_calc, config, device)
        
        # Define test files
        test_data_path = config['CLI']['path']
        test_data_files = [
            'test_day_fair.txt',
            'test_night_fair.txt',
            'test_day_rain.txt',
            'test_night_rain.txt',
            'test_snow.txt'
        ]
        
        # Run test suite
        all_results = run_test_suite(tester, config, test_data_files, test_data_path, num_classes)
        
        # Save results
        results_file = save_test_results(config, all_results, epoch_num, epoch_uuid, test_uuid)
        
        # Upload to vision service
        print("Uploading test results to vision service...")
        upload_success = send_test_results_from_file(results_file)
        if upload_success:
            print("✅ Test results successfully uploaded to vision service")
        else:
            print("❌ Failed to upload test results to vision service")
        
        print(f'Completed testing checkpoint {i+1}/{len(checkpoint_paths)}')
    
    print(f"\n{'='*50}")
    print(f"Completed testing all {len(checkpoint_paths)} checkpoints")


if __name__ == '__main__':
    main()
