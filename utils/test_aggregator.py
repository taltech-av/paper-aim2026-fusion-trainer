#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test aggregation utilities for collecting and analyzing results across multiple checkpoints.
"""
import os
import json
import datetime
import uuid
from integrations.vision_service import send_test_results_from_file
from utils.helpers import sanitize_for_json


def test_checkpoint_and_save(checkpoint_path, test_function, config, *test_args):
    """
    Test a single checkpoint and save/send results to Vision API.

    Args:
        checkpoint_path: Single checkpoint file path
        test_function: Function that takes (checkpoint_path, *test_args) and returns results dict
        config: Configuration dict for saving results
        *test_args: Additional arguments to pass to test_function

    Returns:
        Dict with checkpoint data and results
    """
    print(f"\n{'='*50}")
    print(f"Testing checkpoint: {checkpoint_path}")
    print(f"{'='*50}")

    # Extract epoch info
    epoch_num, epoch_uuid = extract_epoch_info(checkpoint_path)
    test_uuid = str(uuid.uuid4())

    # Run test function for this checkpoint
    checkpoint_results = test_function(checkpoint_path, config, *test_args)

    # Create individual results structure
    individual_results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'epoch': epoch_num,
        'epoch_uuid': epoch_uuid,
        'test_uuid': test_uuid,
        'test_results': checkpoint_results
    }

    # Save results locally
    test_results_dir = os.path.join(config['Log']['logdir'], 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)

    # Generate filename
    if epoch_uuid:
        filename = f'epoch_{epoch_num}_{epoch_uuid}.json'
    else:
        filename = f'epoch_{epoch_num}_test_results.json'

    filepath = os.path.join(test_results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(sanitize_for_json(individual_results), f, indent=2)

    print(f'Individual test results saved to: {filepath}')
    if epoch_uuid:
        print(f'Epoch UUID: {epoch_uuid}')
    print(f'Test UUID: {test_uuid}')

    print("Uploading individual test results to vision service...")
    upload_success = send_test_results_from_file(filepath)
    if upload_success:
        print("✅ Individual test results successfully uploaded to vision service")
    else:
        print("❌ Failed to upload individual test results to vision service")

    # Return checkpoint data
    checkpoint_data = {
        'epoch': epoch_num,
        'epoch_uuid': epoch_uuid,
        'checkpoint_path': checkpoint_path,
        'results': individual_results,
        'saved_file_path': filepath
    }

    print(f"Completed testing checkpoint")

    return checkpoint_data


def extract_epoch_info(checkpoint_path):
    """
    Extract epoch number and UUID from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (epoch_num, epoch_uuid)
    """
    import re
    filename = os.path.basename(checkpoint_path)

    # Try new format first: epoch_{num}_{uuid}.pth
    new_format_match = re.search(r'epoch_(\d+)_([a-f0-9\-]+)\.pth', filename)
    if new_format_match:
        epoch_num = int(new_format_match.group(1))
        epoch_uuid = new_format_match.group(2)
        return epoch_num, epoch_uuid

    # Try old format: checkpoint_{num}.pth
    old_format_match = re.search(r'checkpoint_(\d+)\.pth', filename)
    if old_format_match:
        epoch_num = int(old_format_match.group(1))
        return epoch_num, None

    # Default
    return 0, None