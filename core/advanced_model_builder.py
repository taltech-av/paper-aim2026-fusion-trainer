#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced model builder for state-of-the-art transformers.
"""
import torch
from models.swin_transformer_fusion import SwinTransformerFusion


class AdvancedModelBuilder:
    """Handles model creation for advanced transformers."""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.num_unique_classes = self._calculate_unique_classes()
    
    def _calculate_unique_classes(self):
        """Calculate number of training classes."""
        return len(self.config['Dataset']['train_classes'])
    
    def build_model(self):
        """Build advanced model based on config."""

        resize = self.config['Dataset']['transforms']['resize']
        pretrained = self.config['SwinFusion'].get('pretrained', True)
        fusion_strategy = self.config['SwinFusion'].get('fusion_strategy', 'cross_attention')
        
        model = SwinTransformerFusion(
            RGB_tensor_size=(3, resize, resize),
            XYZ_tensor_size=(3, resize, resize),
            patch_size=self.config['SwinFusion']['patch_size'],
            emb_dims=self.config['SwinFusion'].get('emb_dims', None),
            resample_dim=self.config['SwinFusion']['resample_dim'],
            read=self.config['SwinFusion']['read'],
            hooks=self.config['SwinFusion']['hooks'],
            reassemble_s=self.config['SwinFusion']['reassembles'],
            nclasses=self.num_unique_classes,
            type=self.config['SwinFusion']['type'],
            model_timm=self.config['SwinFusion']['model_timm'],
            pretrained=pretrained,
            fusion_strategy=fusion_strategy
        )
        
        print(f"Built SwinTransformerFusion model with {self.num_unique_classes} classes (pretrained: {pretrained}, fusion: {fusion_strategy})")
        return model
    
    def load_checkpoint(self, model, checkpoint_path):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(self.device)