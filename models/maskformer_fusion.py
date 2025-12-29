#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaskFormer-based fusion model for camera-lidar segmentation.
Simplified implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange


class PixelDecoder(nn.Module):
    """Simplified Pixel Decoder for MaskFormer."""
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # Lateral convs
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
        
        # Output conv
        self.output_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
    def forward(self, features):
        # features: list of [b, c, h, w] from high to low res
        laterals = []
        for feat, conv in zip(features, self.lateral_convs):
            laterals.append(conv(feat))
        
        # Upsample and add
        out = laterals[-1]  # lowest res
        for lateral in reversed(laterals[:-1]):
            out = F.interpolate(out, size=lateral.shape[2:], mode='bilinear', align_corners=False)
            out = out + lateral
        
        out = self.output_conv(out)
        return out


class TransformerDecoder(nn.Module):
    """Simplified Transformer Decoder for MaskFormer."""
    
    def __init__(self, d_model=256, nhead=8, num_layers=6, num_queries=100, num_classes=4):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for no object
        self.mask_embed = nn.Linear(d_model, d_model)
        
    def forward(self, pixel_features):
        b, c, h, w = pixel_features.shape
        pixel_features = pixel_features.flatten(2).permute(0, 2, 1)  # [b, h*w, c]
        
        queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)  # [b, num_queries, c]
        
        # Decoder
        decoded = self.decoder(queries, pixel_features)
        
        # Outputs
        class_logits = self.class_embed(decoded)  # [b, num_queries, num_classes+1]
        mask_features = self.mask_embed(decoded)  # [b, num_queries, c]
        
        # Predict masks
        masks = torch.einsum('bqc,bchw->bqhw', mask_features, pixel_features.permute(0, 2, 1).view(b, c, h, w))
        
        return class_logits, masks


class MaskFormerFusion(nn.Module):
    """MaskFormer-based fusion model."""
    
    def __init__(self,
                 backbone='swin_base_patch4_window7_224',
                 num_classes=4,
                 pixel_decoder_channels=256,
                 transformer_d_model=256,
                 num_queries=100,
                 pretrained=True):
        super().__init__()
        
        # Backbone for Lidar
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        self.backbone_channels = self.backbone.feature_info.channels()
        
        # Pixel Decoder
        self.pixel_decoder = PixelDecoder(self.backbone_channels, pixel_decoder_channels)
        
        # Final segmentation head
        self.final_conv = nn.Conv2d(pixel_decoder_channels, num_classes, 1)
        
    def forward(self, rgb, lidar, modal='fusion'):
        if modal == 'rgb':
            features = self.backbone(rgb)
        elif modal == 'lidar':
            features = self.backbone(lidar)
        elif modal == 'fusion':
            features_rgb = self.backbone(rgb)
            features_lidar = self.backbone(lidar)
            features = [fr + fl for fr, fl in zip(features_rgb, features_lidar)]
        else:
            raise ValueError(f"Invalid modal: {modal}")
        
        # Permute features from [b, h, w, c] to [b, c, h, w] if necessary
        features = [f.permute(0, 3, 1, 2) if f.shape[1] < f.shape[-1] else f for f in features]
        
        # Pixel decoder
        pixel_features = self.pixel_decoder(features)
        
        # Final prediction
        output = self.final_conv(pixel_features)
        
        # Upsample to input resolution
        output = F.interpolate(output, size=(rgb.shape[-2], rgb.shape[-1]), mode='bilinear', align_corners=False)
        
        return None, output