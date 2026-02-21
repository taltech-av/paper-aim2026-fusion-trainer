#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OneFormer Model Benchmarking Script

Benchmarks OneFormer models across different configurations, modalities and
hardware to provide comprehensive performance metrics (parameters, FLOPS,
inference latency, memory usage).
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
import pandas as pd
from tqdm import tqdm
import glob
import re
import sys

# Optional FLOPS libraries
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


class OneFormerBenchmarker:
    """Comprehensive benchmarking for OneFormer models."""

    def __init__(self, config_paths=None, device='auto', epoch_file=None):
        """
        Args:
            config_paths: List of paths to config files to benchmark
            device: 'auto', 'cpu', 'cuda', or a specific device string
            epoch_file: Path to epoch JSON file to associate benchmark with
        """
        self.config_paths = config_paths or []
        self.device = self._setup_device(device)
        self.results = []

        self.training_uuid = None
        self.epoch_uuid = None
        self.epoch = None
        self._extract_epoch_info(epoch_file)

    # ── Device setup ──────────────────────────────────────────────────────────

    def _setup_device(self, device):
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

    # ── Epoch info ────────────────────────────────────────────────────────────

    def _extract_epoch_info(self, epoch_file):
        if not epoch_file:
            epoch_file = self._find_latest_epoch_file()
            if not epoch_file:
                print("Warning: No epoch file found — benchmarks will not be "
                      "associated with any epoch")
                return

        filename = os.path.basename(epoch_file)
        match = re.match(r'epoch_(\d+)_([a-f0-9\-]+)\.json', filename)
        if match:
            self.epoch      = int(match.group(1))
            self.epoch_uuid = match.group(2)
            try:
                with open(epoch_file, 'r') as f:
                    self.training_uuid = json.load(f).get('training_uuid')
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Warning: Could not read epoch file {epoch_file}")
        else:
            print(f"Warning: Could not parse epoch info from filename {filename}")

    def _find_latest_epoch_file(self):
        if not self.config_paths:
            return None
        try:
            first_config = self._load_config(self.config_paths[0])
            epochs_dir   = os.path.join(first_config['Log']['logdir'], 'epochs')
            if not os.path.exists(epochs_dir):
                return None
            epoch_files = sorted(
                glob.glob(os.path.join(epochs_dir, 'epoch_*.json')),
                key=os.path.getmtime, reverse=True
            )
            return epoch_files[0] if epoch_files else None
        except Exception as e:
            print(f"Error finding latest epoch file: {e}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def _create_dummy_input(self, config):
        """Create (rgb, lidar) dummy tensors matching the config's resize."""
        size = config['Dataset']['transforms']['resize']
        return torch.randn(1, 3, size, size), torch.randn(1, 3, size, size)

    def _count_parameters(self, model):
        total      = sum(p.numel() for p in model.parameters())
        trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'total_parameters':       total,
            'trainable_parameters':   trainable,
            'total_parameters_m':     total   / 1e6,
            'trainable_parameters_m': trainable / 1e6,
        }

    # ── FLOPS ─────────────────────────────────────────────────────────────────

    def _calculate_flops(self, model, config):
        rgb, lidar = self._create_dummy_input(config)
        rgb, lidar = rgb.to(self.device), lidar.to(self.device)
        modality   = config['CLI']['mode']

        flops_info = {
            'flops_available': False,
            'total_flops':     None,
            'flops_giga':      None,
            'flops_method':    None,
        }

        if THOP_AVAILABLE:
            try:
                if modality == 'rgb':
                    inputs = (rgb,   rgb,   modality)
                elif modality == 'lidar':
                    inputs = (lidar, lidar, modality)
                else:
                    inputs = (rgb,   lidar, modality)
                flops, _ = profile(model, inputs=inputs, verbose=False)
                flops_info.update({
                    'flops_available': True,
                    'total_flops':     flops,
                    'flops_giga':      flops / 1e9,
                    'flops_method':    'thop',
                })
            except Exception as e:
                print(f"thop profiling failed: {e}")

        if not flops_info['flops_available'] and FVCORE_AVAILABLE:
            try:
                if modality == 'rgb':
                    inputs = (rgb,   rgb,   modality)
                elif modality == 'lidar':
                    inputs = (lidar, lidar, modality)
                else:
                    inputs = (rgb,   lidar, modality)
                analyzer   = FlopCountAnalysis(model, inputs)
                total_flops = analyzer.total()
                flops_info.update({
                    'flops_available': True,
                    'total_flops':     total_flops,
                    'flops_giga':      total_flops / 1e9,
                    'flops_method':    'fvcore',
                })
            except Exception as e:
                print(f"fvcore profiling failed: {e}")

        return flops_info

    # ── Inference timing ──────────────────────────────────────────────────────

    def _measure_inference_time(self, model, config, num_runs=100, warmup_runs=10):
        model.eval()
        rgb, lidar = self._create_dummy_input(config)
        rgb, lidar = rgb.to(self.device), lidar.to(self.device)
        modality   = config['CLI']['mode']

        baseline_gpu_memory = (torch.cuda.memory_allocated() / (1024 ** 2)
                               if self.device.type == 'cuda' else 0)
        baseline_ram_memory = psutil.Process().memory_info().rss / (1024 ** 2)

        def _forward():
            if modality == 'rgb':
                return model(rgb,   rgb,   modality)
            elif modality == 'lidar':
                return model(lidar, lidar, modality)
            else:
                return model(rgb,   lidar, modality)

        with torch.no_grad():
            for _ in range(warmup_runs):
                _forward()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        gpu_memory_usage = []
        ram_memory_usage = []
        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.time()
                _forward()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - t0)

                if self.device.type == 'cuda':
                    gpu_memory_usage.append(torch.cuda.memory_allocated() / (1024 ** 2))
                else:
                    ram_memory_usage.append(psutil.Process().memory_info().rss / (1024 ** 2))

        times = np.array(times)
        result = {
            'mean_time_ms':         np.mean(times) * 1000,
            'std_time_ms':          np.std(times)  * 1000,
            'min_time_ms':          np.min(times)  * 1000,
            'max_time_ms':          np.max(times)  * 1000,
            'fps':                  1.0 / np.mean(times),
            'num_runs':             num_runs,
            'baseline_gpu_memory_mb': baseline_gpu_memory,
            'baseline_ram_memory_mb': baseline_ram_memory,
        }

        if gpu_memory_usage:
            g = np.array(gpu_memory_usage)
            result.update({
                'gpu_memory_mean_mb': np.mean(g),
                'gpu_memory_std_mb':  np.std(g),
                'gpu_memory_min_mb':  np.min(g),
                'gpu_memory_max_mb':  np.max(g),
            })

        if ram_memory_usage:
            r = np.array(ram_memory_usage)
            result.update({
                'ram_memory_mean_mb': np.mean(r),
                'ram_memory_std_mb':  np.std(r),
                'ram_memory_min_mb':  np.min(r),
                'ram_memory_max_mb':  np.max(r),
            })

        return result

    # ── System info ───────────────────────────────────────────────────────────

    def _get_system_info(self):
        info = {
            'cpu_count':         psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb':   psutil.virtual_memory().total / (1024 ** 3),
            'python_version':    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'torch_version':     torch.__version__,
            'cuda_available':    torch.cuda.is_available(),
            'cuda_version':      torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count':         torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                gpu = gpus[i] if i < len(gpus) else None
                if gpu:
                    gpu_info.append({
                        'name':             gpu.name,
                        'memory_total_mb':  gpu.memoryTotal,
                        'memory_free_mb':   gpu.memoryFree,
                        'temperature':      gpu.temperature,
                    })
                else:
                    gpu_info.append({
                        'name':             torch.cuda.get_device_name(i),
                        'memory_total_mb':  torch.cuda.get_device_properties(i).total_memory / (1024 ** 2),
                        'memory_free_mb':   None,
                        'temperature':      None,
                    })
            info['gpu_info'] = gpu_info

        return info

    # ── Per-config benchmark ──────────────────────────────────────────────────

    def benchmark_config(self, config_path, num_runs=100, warmup_runs=10):
        """Benchmark a single OneFormer configuration."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {os.path.basename(config_path)}")
        print(f"{'='*60}")

        config   = self._load_config(config_path)
        modality = config['CLI']['mode']

        from models.oneformer_fusion import OneFormerFusion
        num_classes = len(config['Dataset']['train_classes'])
        model = OneFormerFusion(
            backbone=config['OneFormer']['model_timm'],
            num_classes=num_classes,
            pixel_decoder_channels=config['OneFormer'].get('pixel_decoder_channels', 256),
            transformer_d_model=config['OneFormer'].get('transformer_d_model', 256),
            num_queries=config['OneFormer'].get('num_queries', 100),
            pretrained=config['OneFormer'].get('pretrained', True),
        )
        model.to(self.device)

        model_info = self._count_parameters(model)
        model_info.update({
            'model_name': config.get('Summary', 'Unknown'),
            'backbone':   config['CLI']['backbone'],
            'dataset':    config['Dataset']['name'],
            'image_size': config['Dataset']['transforms']['resize'],
            'pretrained': config['OneFormer'].get('pretrained', True),
        })

        print(f"\nTesting modality: {modality.upper()}")

        flops_info  = self._calculate_flops(model, config)
        timing_info = self._measure_inference_time(model, config, num_runs, warmup_runs)

        result = {
            'modality':   modality,
            **model_info,
            **flops_info,
            **timing_info,
            'device':      str(self.device),
            'device_type': self.device.type,
        }
        self.results.append(result)

        print(f"  Parameters: {model_info['total_parameters_m']:.1f}M")
        if flops_info['flops_available']:
            print(f"  FLOPS: {flops_info['flops_giga']:.2f}G ({flops_info['flops_method']})")
        print(f"  Inference: {timing_info['mean_time_ms']:.2f}±{timing_info['std_time_ms']:.2f}ms")
        print(f"  FPS: {timing_info['fps']:.1f}")
        if 'gpu_memory_mean_mb' in timing_info:
            print(f"  GPU Memory: {timing_info['gpu_memory_mean_mb']:.1f}±"
                  f"{timing_info['gpu_memory_std_mb']:.1f}MB")
        if 'ram_memory_mean_mb' in timing_info:
            print(f"  RAM Memory: {timing_info['ram_memory_mean_mb']:.1f}±"
                  f"{timing_info['ram_memory_std_mb']:.1f}MB")

    def benchmark_all_configs(self, num_runs=100, warmup_runs=10):
        """Benchmark all OneFormer configurations in config_paths."""
        print(f"Starting OneFormer Model Benchmarking")
        print(f"Device: {self.device}")
        print(f"Configs to test: {len(self.config_paths)}")

        for config_path in self.config_paths:
            try:
                config  = self._load_config(config_path)
                backbone = config['CLI']['backbone']
                if backbone == 'oneformer':
                    self.benchmark_config(config_path, num_runs, warmup_runs)
                else:
                    print(f"Skipping non-OneFormer config {config_path} "
                          f"(backbone: {backbone})")
            except Exception as e:
                print(f"Error benchmarking {config_path}: {e}")

    # ── Save results ──────────────────────────────────────────────────────────

    def save_results(self, output_path=None):
        if output_path is None and self.config_paths:
            first_config = self._load_config(self.config_paths[0])
            benchmark_dir = os.path.join(first_config['Log']['logdir'], 'benchmark')
            os.makedirs(benchmark_dir, exist_ok=True)
            timestamp   = time.strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(benchmark_dir,
                                       f'oneformer_benchmark_results_{timestamp}.json')

        output_data = {
            'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S'),
            'epoch':        self.epoch,
            'epoch_uuid':   self.epoch_uuid,
            'training_uuid': self.training_uuid,
            'system_info':  self._get_system_info(),
            'results':      self.results,
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nBenchmark results saved to: {output_path}")

        try:
            from integrations.vision_service import send_benchmark_results_from_file
            if self.training_uuid:
                success = send_benchmark_results_from_file(
                    output_path,
                    training_uuid=self.training_uuid,
                    epoch_uuid=self.epoch_uuid,
                    epoch=self.epoch,
                )
            else:
                success = send_benchmark_results_from_file(output_path)
            if success:
                print("Benchmark results successfully sent to vision service")
            else:
                print("Failed to send benchmark results to vision service")
        except ImportError:
            print("Warning: vision_service not found, skipping upload")
        except Exception as e:
            print(f"Error sending benchmark results to vision service: {e}")

        if self.results:
            df = pd.DataFrame(self.results)
            summary_path = output_path.replace('.json', '_summary.csv')
            df.to_csv(summary_path, index=False)
            print(f"Summary CSV saved to: {summary_path}")

            print(f"\n{'='*80}")
            print("OVERALL SUMMARY")
            print(f"{'='*80}")
            print(f"Total configurations tested : {len(self.results)}")
            print(f"Average FPS                 : {df['fps'].mean():.1f}")
            print(f"Average parameters          : {df['total_parameters_m'].mean():.1f}M")
            if 'flops_giga' in df.columns and df['flops_available'].any():
                print(f"Average FLOPS               : "
                      f"{df[df['flops_available']]['flops_giga'].mean():.2f}G")
            print(f"{'='*80}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Benchmark OneFormer models')
    parser.add_argument('-c', '--config', nargs='+', required=True,
                        help='Paths to config JSON files or directories')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, cuda:0, …)')
    parser.add_argument('--single', action='store_true',
                        help='Benchmark on a single device only '
                             '(default: both CPU and GPU if available)')
    parser.add_argument('--epoch-file', type=str, default=None,
                        help='Path to epoch JSON file for associating benchmarks')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results JSON file')
    parser.add_argument('--num-runs', type=int, default=100,
                        help='Number of timed inference runs per configuration')
    parser.add_argument('--warmup-runs', type=int, default=10,
                        help='Number of warmup runs before timing')
    args = parser.parse_args()

    # Collect config files
    config_files = []
    for base_path in args.config:
        if os.path.isfile(base_path) and base_path.endswith('.json'):
            config_files.append(base_path)
        elif os.path.isdir(base_path):
            for root, _, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.json'):
                        config_files.append(os.path.join(root, file))

    print(f"Found {len(config_files)} config files")
    if not config_files:
        print("No config files found!")
        return

    # Decide which devices to test
    if not args.single and torch.cuda.is_available():
        devices_to_test = ['cpu', 'cuda']
        print("Benchmarking on both CPU and GPU (use --single to test one device)")
    else:
        devices_to_test = [args.device]
        device_name = (args.device if args.device != 'auto'
                       else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Benchmarking on single device: {device_name}")

    all_results = []
    for device in devices_to_test:
        print(f"\n{'='*80}")
        print(f"Running benchmarks on {device.upper()}")
        print(f"{'='*80}")

        benchmarker = OneFormerBenchmarker(
            config_paths=config_files,
            device=device,
            epoch_file=args.epoch_file,
        )
        benchmarker.benchmark_all_configs(
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
        )
        all_results.extend(benchmarker.results)

    # Save combined results via a fresh instance (preserves epoch info)
    combined = OneFormerBenchmarker(config_files, devices_to_test[0], args.epoch_file)
    combined.results = all_results
    combined.save_results(args.output)


if __name__ == '__main__':
    main()
