#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaskFormer Model Benchmarking Script

This script benchmarks MaskFormer models across different configurations, modalities,
and hardware to provide comprehensive performance metrics.
"""
import os
import json
import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import psutil
import GPUtil
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import glob
import re
import sys

# Try to import FLOPS calculation libraries
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. Install with: pip install thop")

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not available. Install with: pip install fvcore")


class MaskFormerBenchmarker:
    """Comprehensive benchmarking for MaskFormer models."""

    def __init__(self, config_paths=None, device='auto', epoch_file=None):
        """
        Initialize benchmarker.

        Args:
            config_paths: List of paths to config files to benchmark
            device: Device to use ('auto', 'cpu', 'cuda', or specific GPU)
            epoch_file: Path to epoch JSON file to associate benchmark with
        """
        self.config_paths = config_paths or []
        self.device = self._setup_device(device)
        self.results = []

        # Extract epoch information from file path
        self.training_uuid = None
        self.epoch_uuid = None
        self.epoch = None

        self._extract_epoch_info(epoch_file)

    def _setup_device(self, device):
        """Setup the computation device."""
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cpu':
            device = torch.device('cpu')
        elif device.startswith('cuda'):
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = torch.device('cpu')
            else:
                device = torch.device(device)
        else:
            raise ValueError(f"Unsupported device: {device}")

        print(f"Using device: {device}")
        return device

    def _extract_epoch_info(self, epoch_file):
        """Extract epoch information from epoch file path and contents."""
        import re

        if not epoch_file:
            # If no epoch file provided, find the latest one
            epoch_file = self._find_latest_epoch_file()
            if not epoch_file:
                print("Warning: No epoch file found, benchmarks will not be associated with any epoch")
                return

        # Extract epoch number and UUID from filename
        # Pattern: epoch_{number}_{uuid}.json
        filename = os.path.basename(epoch_file)
        match = re.match(r'epoch_(\d+)_([a-f0-9\-]+)\.json', filename)

        if match:
            self.epoch = int(match.group(1))
            self.epoch_uuid = match.group(2)

            # Try to extract training UUID from the epoch file contents
            try:
                with open(epoch_file, 'r') as f:
                    epoch_data = json.load(f)
                    self.training_uuid = epoch_data.get('training_uuid')
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Warning: Could not read epoch file {epoch_file}")
        else:
            print(f"Warning: Could not parse epoch information from filename {filename}")

    def _find_latest_epoch_file(self):
        """Find the latest epoch JSON file from the logs directory."""
        if not self.config_paths:
            return None

        # Get log directory from first config
        try:
            first_config = self._load_config(self.config_paths[0])
            logdir = first_config['Log']['logdir']
            epochs_dir = os.path.join(logdir, 'epochs')

            if not os.path.exists(epochs_dir):
                return None

            # Find all epoch JSON files
            epoch_files = glob.glob(os.path.join(epochs_dir, 'epoch_*.json'))
            if not epoch_files:
                return None

            # Get the most recent epoch file
            epoch_files.sort(key=os.path.getmtime, reverse=True)
            return epoch_files[0]

        except Exception as e:
            print(f"Error finding latest epoch file: {e}")
            return None

    def _load_config(self, config_path):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def _create_dummy_input(self, config, modality='rgb'):
        """Create dummy input tensors for benchmarking."""
        resize = config['Dataset']['transforms']['resize']

        # Create dummy RGB input
        rgb = torch.randn(1, 3, resize, resize)

        # Create dummy LiDAR input (3 channels for XYZ)
        lidar = torch.randn(1, 3, resize, resize)

        return rgb, lidar

    def _count_parameters(self, model):
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_parameters_m': total_params / 1e6,
            'trainable_parameters_m': trainable_params / 1e6
        }

    def _calculate_flops(self, model, config):
        """Calculate FLOPS for MaskFormer model."""
        rgb, lidar = self._create_dummy_input(config, 'fusion')  # Use fusion for FLOPS calc

        # Move inputs to same device as model
        rgb = rgb.to(self.device)
        lidar = lidar.to(self.device)
        modality = config['CLI']['mode']

        flops_info = {
            'flops_available': False,
            'total_flops': None,
            'flops_giga': None,
            'flops_method': None
        }

        # Try thop first
        if THOP_AVAILABLE:
            try:
                if modality == 'rgb':
                    flops, params = profile(model, inputs=(rgb, rgb, modality), verbose=False)
                elif modality == 'lidar':
                    flops, params = profile(model, inputs=(lidar, lidar, modality), verbose=False)
                else:  # fusion
                    flops, params = profile(model, inputs=(rgb, lidar, modality), verbose=False)

                flops_info.update({
                    'flops_available': True,
                    'total_flops': flops,
                    'flops_giga': flops / 1e9,
                    'flops_method': 'thop'
                })
            except Exception as e:
                print(f"thop profiling failed: {e}")

        # Try fvcore as fallback
        if not flops_info['flops_available'] and FVCORE_AVAILABLE:
            try:
                if modality == 'rgb':
                    flop_analyzer = FlopCountAnalysis(model, (rgb, rgb, modality))
                elif modality == 'lidar':
                    flop_analyzer = FlopCountAnalysis(model, (lidar, lidar, modality))
                else:  # fusion
                    flop_analyzer = FlopCountAnalysis(model, (rgb, lidar, modality))

                total_flops = flop_analyzer.total()
                flops_info.update({
                    'flops_available': True,
                    'total_flops': total_flops,
                    'flops_giga': total_flops / 1e9,
                    'flops_method': 'fvcore'
                })
            except Exception as e:
                print(f"fvcore profiling failed: {e}")

        return flops_info

    def _measure_inference_time(self, model, config, num_runs=100, warmup_runs=10):
        """Measure inference time and memory usage for MaskFormer on current device."""
        model.eval()
        rgb, lidar = self._create_dummy_input(config, 'fusion')
        modality = config['CLI']['mode']

        # Move inputs to device
        rgb = rgb.to(self.device)
        lidar = lidar.to(self.device)

        # Get baseline memory usage after model loading
        baseline_gpu_memory = torch.cuda.memory_allocated() / (1024**2) if self.device.type == 'cuda' else 0
        baseline_ram_memory = psutil.Process().memory_info().rss / (1024**2)

        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                if modality == 'rgb':
                    _ = model(rgb, rgb, modality)
                elif modality == 'lidar':
                    _ = model(lidar, lidar, modality)
                else:  # fusion
                    _ = model(rgb, lidar, modality)

        # Synchronize before timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Initialize memory tracking
        gpu_memory_usage = []
        ram_memory_usage = []

        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()

                if modality == 'rgb':
                    _ = model(rgb, rgb, modality)
                elif modality == 'lidar':
                    _ = model(lidar, lidar, modality)
                else:  # fusion
                    _ = model(rgb, lidar, modality)

                # Synchronize after inference
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.time()
                times.append(end_time - start_time)

                # Track memory usage
                if self.device.type == 'cuda':
                    # GPU memory usage in MB
                    gpu_memory_usage.append(torch.cuda.memory_allocated() / (1024**2))
                else:
                    # RAM usage in MB
                    ram_memory_usage.append(psutil.Process().memory_info().rss / (1024**2))

        times = np.array(times)

        result = {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times),
            'num_runs': num_runs,
            'baseline_gpu_memory_mb': baseline_gpu_memory,
            'baseline_ram_memory_mb': baseline_ram_memory
        }

        # Add memory statistics if collected
        if gpu_memory_usage:
            gpu_memory_usage = np.array(gpu_memory_usage)
            result.update({
                'gpu_memory_mean_mb': np.mean(gpu_memory_usage),
                'gpu_memory_std_mb': np.std(gpu_memory_usage),
                'gpu_memory_min_mb': np.min(gpu_memory_usage),
                'gpu_memory_max_mb': np.max(gpu_memory_usage)
            })

        if ram_memory_usage:
            ram_memory_usage = np.array(ram_memory_usage)
            result.update({
                'ram_memory_mean_mb': np.mean(ram_memory_usage),
                'ram_memory_std_mb': np.std(ram_memory_usage),
                'ram_memory_min_mb': np.min(ram_memory_usage),
                'ram_memory_max_mb': np.max(ram_memory_usage)
            })

        return result

    def _get_system_info(self):
        """Get system information."""
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                gpu = GPUtil.getGPUs()[i] if i < len(GPUtil.getGPUs()) else None
                if gpu:
                    gpu_info.append({
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_free_mb': gpu.memoryFree,
                        'temperature': gpu.temperature
                    })
                else:
                    gpu_info.append({
                        'name': torch.cuda.get_device_name(i),
                        'memory_total_mb': torch.cuda.get_device_properties(i).total_memory / (1024**2),
                        'memory_free_mb': None,
                        'temperature': None
                    })
            system_info['gpu_info'] = gpu_info

        return system_info

    def benchmark_config(self, config_path):
        """Benchmark a single MaskFormer configuration."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {os.path.basename(config_path)}")
        print(f"{'='*60}")

        config = self._load_config(config_path)
        modality = config['CLI']['mode']  # Get modality from config

        # Build MaskFormer model
        from models.maskformer_fusion import MaskFormerFusion
        num_classes = len(config['Dataset']['train_classes'])
        model = MaskFormerFusion(
            backbone=config['MaskFormer']['model_timm'],
            num_classes=num_classes,
            pixel_decoder_channels=config['MaskFormer']['pixel_decoder_channels'],
            transformer_d_model=config['MaskFormer']['transformer_d_model'],
            num_queries=config['MaskFormer']['num_queries'],
            pretrained=config['MaskFormer'].get('pretrained', True)
        )
        model.to(self.device)

        # Get model info
        model_info = self._count_parameters(model)
        model_info.update({
            'model_name': config.get('Summary', 'Unknown'),
            'backbone': config['CLI']['backbone'],
            'dataset': config['Dataset']['name'],
            'image_size': config['Dataset']['transforms']['resize'],
            'pretrained': config['MaskFormer'].get('pretrained', True)
        })

        print(f"\nTesting modality: {modality.upper()}")

        # Calculate FLOPS
        flops_info = self._calculate_flops(model, config)

        # Measure inference time
        timing_info = self._measure_inference_time(model, config)

        # Combine results
        result = {
            'modality': modality,
            **model_info,
            **flops_info,
            **timing_info,
            'device': str(self.device),
            'device_type': self.device.type
        }

        self.results.append(result)

        # Print summary
        print(f"  Parameters: {model_info['total_parameters_m']:.1f}M")
        if flops_info['flops_available']:
            print(f"  FLOPS: {flops_info['flops_giga']:.2f}G ({flops_info['flops_method']})")
        print(f"  Inference: {timing_info['mean_time_ms']:.2f}±{timing_info['std_time_ms']:.2f}ms")
        print(f"  FPS: {timing_info['fps']:.1f}")
        if 'gpu_memory_mean_mb' in timing_info:
            print(f"  GPU Memory: {timing_info['gpu_memory_mean_mb']:.1f}±{timing_info['gpu_memory_std_mb']:.1f}MB")
        if 'ram_memory_mean_mb' in timing_info:
            print(f"  RAM Memory: {timing_info['ram_memory_mean_mb']:.1f}±{timing_info['ram_memory_std_mb']:.1f}MB")

    def benchmark_all_configs(self):
        """Benchmark all configurations."""
        print(f"Starting MaskFormer Model Benchmarking")
        print(f"Device: {self.device}")
        print(f"Configs to test: {len(self.config_paths)}")

        for config_path in self.config_paths:
            try:
                config = self._load_config(config_path)
                backbone = config['CLI']['backbone']

                if backbone == 'maskformer':
                    self.benchmark_config(config_path)
                else:
                    print(f"Skipping non-MaskFormer config {config_path} (backbone: {backbone})")
                    continue
            except Exception as e:
                print(f"Error benchmarking {config_path}: {e}")
                continue

    def save_results(self, output_path=None):
        """Save results to JSON file."""
        # Use config's logdir if no output path specified
        if output_path is None and self.config_paths:
            first_config = self._load_config(self.config_paths[0])
            logdir = first_config['Log']['logdir']
            benchmark_dir = os.path.join(logdir, 'benchmark')
            os.makedirs(benchmark_dir, exist_ok=True)

            # Create timestamped filename to avoid overwriting
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(benchmark_dir, f'maskformer_benchmark_results_{timestamp}.json')

        # Add system info to results
        system_info = self._get_system_info()

        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': self.epoch,
            'epoch_uuid': self.epoch_uuid,
            'training_uuid': self.training_uuid,
            'system_info': system_info,
            'results': self.results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nBenchmark results saved to: {output_path}")

        # Send results to vision service
        try:
            from integrations.vision_service import send_benchmark_results_from_file
            
            # Only send training association if we have a valid training UUID
            if self.training_uuid:
                success = send_benchmark_results_from_file(
                    output_path,
                    training_uuid=self.training_uuid,
                    epoch_uuid=self.epoch_uuid,
                    epoch=self.epoch
                )
            else:
                success = send_benchmark_results_from_file(output_path)
            
            if success:
                print("Benchmark results successfully sent to vision service")
            else:
                print("Failed to send benchmark results to vision service")
        except ImportError:
            print("Warning: vision_service module not found, skipping upload to vision service")
        except Exception as e:
            print(f"Error sending benchmark results to vision service: {e}")

        # Create summary DataFrame
        if self.results:
            df = pd.DataFrame(self.results)
            summary_path = output_path.replace('.json', '_summary.csv')
            df.to_csv(summary_path, index=False)
            print(f"Summary CSV saved to: {summary_path}")

            # Print overall summary
            print(f"\n{'='*80}")
            print("OVERALL SUMMARY")
            print(f"{'='*80}")
            print(f"Total configurations tested: {len(self.results)}")
            print(f"Average FPS: {df['fps'].mean():.1f}")
            print(f"Average parameters: {df['total_parameters_m'].mean():.1f}M")
            if 'flops_giga' in df.columns:
                flops_available = df['flops_available'].sum()
                if flops_available > 0:
                    print(f"Average FLOPS: {df[df['flops_available']]['flops_giga'].mean():.2f}G")
            print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark MaskFormer models')
    parser.add_argument('-c', '--configs', nargs='+', required=True,
                        help='Paths to config JSON files')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--single', action='store_true',
                        help='Benchmark on single device only (default: benchmark on both CPU and GPU if available)')
    parser.add_argument('--epoch-file', type=str, default=None,
                        help='Path to epoch JSON file for associating benchmarks')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results JSON file')
    parser.add_argument('--num-runs', type=int, default=100,
                        help='Number of timing runs per configuration')
    parser.add_argument('--warmup-runs', type=int, default=10,
                        help='Number of warmup runs before timing')

    args = parser.parse_args()

    # Find all config files
    config_files = []
    for base_path in args.configs:
        if os.path.isfile(base_path):
            if base_path.endswith('.json'):
                config_files.append(base_path)
        elif os.path.isdir(base_path):
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.json'):
                        config_files.append(os.path.join(root, file))

    print(f"Found {len(config_files)} config files")

    if not config_files:
        print("No config files found!")
        return

    all_results = []
    devices_to_test = []

    # Default behavior: test both CPU and GPU if CUDA available and --single not specified
    if not args.single and torch.cuda.is_available():
        devices_to_test = ['cpu', 'cuda']
        print("Benchmarking on both CPU and GPU (use --single to test single device)")
    else:
        devices_to_test = [args.device]
        device_name = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Benchmarking on single device: {device_name}")

    for device in devices_to_test:
        print(f"\n{'='*80}")
        print(f"Running benchmarks on {device.upper()}")
        print(f"{'='*80}")

        # Create benchmarker for this device
        benchmarker = MaskFormerBenchmarker(
            config_paths=config_files,
            device=device,
            epoch_file=args.epoch_file
        )

        # Update timing parameters
        if hasattr(benchmarker, '_measure_inference_time'):
            # Monkey patch to use custom num_runs
            original_measure = benchmarker._measure_inference_time
            def custom_measure(model, config):
                return original_measure(model, config, num_runs=args.num_runs, warmup_runs=args.warmup_runs)
            benchmarker._measure_inference_time = custom_measure

        # Full benchmarking
        benchmarker.benchmark_all_configs()

        # Collect results
        all_results.extend(benchmarker.results)

    # Create combined benchmarker for saving (use first device for config paths)
    combined_benchmarker = MaskFormerBenchmarker(config_files, devices_to_test[0], args.epoch_file)
    combined_benchmarker.results = all_results

    # Save combined results
    combined_benchmarker.save_results(args.output)


if __name__ == '__main__':
    main()