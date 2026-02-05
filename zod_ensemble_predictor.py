#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictor for merged CLFT ensemble model (single checkpoint).
Loads a merged checkpoint that combines vehicle, sign, and human specialized models.
"""
import os
import json
import torch
import numpy as np
from pathlib import Path

from clft.clft import CLFT


class ZODMergedEnsemblePredictor:
    """Predictor for pseudo-merged ensemble (single file containing all 3 models)."""

    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.models = []
        self.class_names = []
        self.class_indices = []

        # Load pseudo-merged checkpoint
        self._load_pseudo_merged_checkpoint()

    def _load_pseudo_merged_checkpoint(self):
        """Load the pseudo-merged ensemble checkpoint containing all 3 models."""
        checkpoint_path = "/media/tom/ml/projects/fusion-training/logs/zod/clft/specialization/ensemble/merged_ensemble_checkpoint.pth"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pseudo-merged checkpoint not found: {checkpoint_path}")

        print(f"Loading pseudo-merged ensemble checkpoint: {checkpoint_path}")

        # Load the combined checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'ensemble_models' not in checkpoint:
            # Fallback to single model loading
            print("No ensemble_models found, loading as single model...")
            self._load_single_model_checkpoint(checkpoint)
            return

        # Load each model from the ensemble
        model_configs = [
            ('vehicle', 1, checkpoint['ensemble_models']['vehicle']),
            ('sign', 2, checkpoint['ensemble_models']['sign']),
            ('human', 3, checkpoint['ensemble_models']['human'])
        ]

        print(f"Loading ensemble from pseudo-merged checkpoint")

        for class_name, class_idx, model_ckpt in model_configs:
            print(f"Loading {class_name} model...")

            # Use the same config as ensemble predictor
            config_path = f"/media/tom/ml/projects/fusion-training/config/zod/clft/specialization/{class_name}_only.json"
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Build model using ModelBuilder
            from core.model_builder import ModelBuilder
            model_builder = ModelBuilder(config, self.device)
            model = model_builder.build_model()
            model.to(self.device)
            model.eval()

            # Load checkpoint
            model.load_state_dict(model_ckpt['model_state_dict'])
            print(f"  Loaded checkpoint from epoch {model_ckpt.get('epoch', 0)}")

            self.models.append(model)
            self.class_names.append(class_name)
            self.class_indices.append(class_idx)

        print(f"Pseudo-merged ensemble loaded with {len(self.models)} models: {self.class_names}")
        print("Class mapping: background=0, vehicle=1, sign=2, human=3")

    def _load_single_model_checkpoint(self, checkpoint):
        """Fallback: load as single merged model."""
        # Use the baseline config
        config_path = "/media/tom/ml/projects/fusion-training/config/zod/clft/specialization/baseline.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Build model using ModelBuilder
        from core.model_builder import ModelBuilder
        model_builder = ModelBuilder(config, self.device)
        self.model = model_builder.build_model()
        self.model.to(self.device)

        # Load the merged state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print("✓ Single merged model loaded")

    def predict(self, rgb_batch, lidar_batch, modal='cross_fusion'):
        """
        Make ensemble prediction.

        Args:
            rgb_batch: RGB input tensor [B, 3, H, W]
            lidar_batch: LiDAR input tensor [B, 3, H, W]
            modal: Fusion mode ('rgb', 'lidar', 'cross_fusion')

        Returns:
            Combined segmentation output [B, 4, H, W] (background + 3 classes)
        """
        if not self.models:
            # Single model prediction
            with torch.no_grad():
                _, seg_output = self.model(rgb_batch, lidar_batch, modal=modal)
                return seg_output

        # Ensemble prediction (same as ZODEnsemblePredictor)
        batch_size = rgb_batch.shape[0]
        height, width = rgb_batch.shape[2], rgb_batch.shape[3]

        # Collect predictions from each single-class model
        class_predictions = []

        with torch.no_grad():
            for i, model in enumerate(self.models):
                # Forward pass
                _, seg_output = model(rgb_batch, lidar_batch, modal=modal)
                # seg_output shape: [B, 2, H, W] (background + class_i)

                # Apply softmax and extract probability for the positive class (index 1)
                class_prob = torch.softmax(seg_output, dim=1)[:, 1:2]  # [B, 1, H, W]
                class_predictions.append(class_prob)

        # Combine predictions: [B, 3, H, W] in order: vehicle, sign, human
        # class_predictions order: [vehicle, sign, human]
        combined_classes = torch.cat([class_predictions[0], class_predictions[1], class_predictions[2]], dim=1)

        # Create background channel as 1 - max(class probabilities)
        background_prob = 1.0 - torch.max(combined_classes, dim=1, keepdim=True)[0]

        # Final output: [B, 4, H, W] (background, vehicle, sign, human)
        final_output = torch.cat([background_prob, combined_classes], dim=1)

        return final_output


def test_merged_predictor():
    """Quick test of the merged predictor."""
    print("Testing merged ensemble predictor...")

    try:
        predictor = ZODMergedEnsemblePredictor()

        # Create dummy batch
        batch_size = 2
        rgb_batch = torch.randn(batch_size, 3, 384, 384).to(predictor.device)
        lidar_batch = torch.randn(batch_size, 3, 384, 384).to(predictor.device)

        # Run prediction
        with torch.no_grad():
            output = predictor.predict(rgb_batch, lidar_batch)

        print(f"✓ Prediction successful! Output shape: {output.shape}")
        print(f"✓ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print(f"✓ Output sum per sample (should be ~1.0 for probabilities): {output.sum(dim=1).mean().item():.3f}")

        # Check if it's probabilities or logits
        pred_classes = torch.argmax(output, dim=1)
        print(f"✓ Predicted classes: {pred_classes.unique()}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == '__main__':
    test_merged_predictor()