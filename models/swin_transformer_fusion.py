#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swin Transformer based fusion model for camera-lidar segmentation.
"""

import torch
import torch.nn as nn
import timm

class Read_ignore(nn.Module):
    def __init__(self, start_index=1):
        super(Read_ignore, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]

class Read_add(nn.Module):
    def __init__(self, start_index=1):
        super(Read_add, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)

class Read_projection(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(Read_projection, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)
        return self.project(features)

class Resample(nn.Module):
    def __init__(self, s, emb_dim, resample_dim):
        super(Resample, self).__init__()
        assert (s in [4, 8, 16, 32]), "s must be in [4, 8, 16, 32]"
        self.emb_dim = emb_dim
        self.resample_dim = resample_dim
        self.s = s
        if self.emb_dim is not None:
            self.conv1 = nn.Conv2d(self.emb_dim, self.resample_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.emb_dim is None:
            self.emb_dim = x.shape[1]
            self.conv1 = nn.Conv2d(self.emb_dim, self.resample_dim, kernel_size=1, stride=1, padding=0).to(x.device)
        x = self.conv1(x)
        x = nn.functional.interpolate(x, scale_factor=self.s, mode='bilinear', align_corners=False)
        return x

class SpatialReassemble(nn.Module):
    def __init__(self, read, s, emb_dim, resample_dim):
        """
        Modified for spatial inputs [b, c, h, w]
        """
        super(SpatialReassemble, self).__init__()

        # For spatial, read and concat are identity
        self.read = nn.Identity()
        self.concat = nn.Identity()

        #Projection + Resample
        self.resample = Resample(s, emb_dim, resample_dim)

    def forward(self, x):
        # x is already [b, c, h, w]
        x = self.read(x)
        x = self.concat(x)
        x = self.resample(x)
        return x

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        # x: [B, C, H, W] (Query - Camera)
        # context: [B, C, H, W] (Key/Value - LiDAR)
        B, C, H, W = x.shape
        
        # Optimization: Downsample if spatial dimensions are too large to prevent OOM
        # Target size 64x64 results in 4096 tokens, which is manageable
        target_h, target_w = 64, 64
        is_downsampled = False
        if H > target_h or W > target_w:
            x_in = nn.functional.adaptive_avg_pool2d(x, (target_h, target_w))
            context_in = nn.functional.adaptive_avg_pool2d(context, (target_h, target_w))
            is_downsampled = True
        else:
            x_in = x
            context_in = context
        
        # Flatten: [B, C, H', W'] -> [B, C, N] -> [B, N, C]
        x_flat = x_in.flatten(2).transpose(1, 2)
        context_flat = context_in.flatten(2).transpose(1, 2)
        
        # Attention
        # query=x, key=context, value=context
        attn_out, _ = self.multihead_attn(query=x_flat, key=context_flat, value=context_flat)
        
        # Dropout
        attn_out = self.proj_drop(attn_out)
        
        # Reshape back: [B, N, C] -> [B, C, H', W']
        if is_downsampled:
            out = attn_out.transpose(1, 2).reshape(B, C, target_h, target_w)
            # Upsample back to original resolution
            out = nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        else:
            out = attn_out.transpose(1, 2).reshape(B, C, H, W)
            
        return out

class Fusion(nn.Module):
    def __init__(self, resample_dim, fusion_strategy=''):
        super(Fusion, self).__init__()
        if not fusion_strategy:
            raise ValueError("fusion_strategy must be specified and cannot be empty")
        self.res_conv_xyz = ResidualConvUnit(resample_dim)
        self.res_conv_rgb = ResidualConvUnit(resample_dim)
        self.res_conv2 = ResidualConvUnit(resample_dim)
        
        self.fusion_strategy = fusion_strategy
        
        if self.fusion_strategy == 'cross_attention':
            # Cross Attention for Fusion
            self.cross_attn = CrossAttention(resample_dim)
            self.alpha = nn.Parameter(torch.tensor(0.0)) # Learnable scaling parameter
        elif self.fusion_strategy == 'gmf':
            # Geometric Modulation Fusion
            self.modulation_mlp = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(resample_dim, resample_dim * 2),
                nn.ReLU(),
            )
            self.alpha = nn.Parameter(torch.zeros(1))  # Gating parameter
        elif self.fusion_strategy == 'sarf':
            self.alpha = nn.Parameter(torch.zeros(1))  # Gating parameter
            self.ln_rgb = nn.LayerNorm(resample_dim)
            self.ln_lidar = nn.LayerNorm(resample_dim)
            # Spatial attention for beta: outputs 1 channel [B, 1, H, W]
            self.beta_attn = nn.Sequential(
                nn.Conv2d(resample_dim * 2, 1, kernel_size=1),
                nn.Sigmoid()
            )
        elif self.fusion_strategy == 'gated_average':
            self.alpha = nn.Parameter(torch.zeros(1))
        elif self.fusion_strategy == 'residual_average':
            # Middle ground: residual convs + average + previous stage (no gating)
            pass
        elif self.fusion_strategy == 'gated_average_no_prev':
            # Ablate previous_stage: residual convs + gating, no previous stage
            self.alpha = nn.Parameter(torch.zeros(1))
        elif self.fusion_strategy == 'weighted_average':
            # Global weighted average with gating
            self.alpha = nn.Parameter(torch.zeros(1))
            self.beta = nn.Parameter(torch.tensor(0.5))
        elif self.fusion_strategy == 'simple_average':
            # No additional parameters for true simple average
            pass

    def forward(self, rgb, lidar, previous_stage=None, modal = 'rgb'):
        if previous_stage == None:
                previous_stage = torch.zeros_like(rgb)

        attn_out = torch.zeros_like(rgb)

        if modal == 'rgb':
            output_stage1_rgb = self.res_conv_rgb(rgb)
            output_stage1 = output_stage1_rgb + previous_stage
        elif modal == 'lidar':
            output_stage1_lidar = self.res_conv_xyz(lidar)
            output_stage1 = output_stage1_lidar + previous_stage
        elif modal == 'cross_fusion':
            if self.fusion_strategy == 'cross_attention':
                output_stage1_rgb = self.res_conv_rgb(rgb)
                output_stage1_lidar = self.res_conv_xyz(lidar)
                # Apply Cross Attention: Query=RGB, Key/Value=LiDAR
                attn_out = self.cross_attn(output_stage1_rgb, output_stage1_lidar)
                # Formula: F_f = F_c + alpha * Attention(F_c, F_l) + F_l
                output_stage1 = output_stage1_rgb + (self.alpha * attn_out) + output_stage1_lidar + previous_stage
            elif self.fusion_strategy == 'sarf':
                output_stage1_rgb = self.res_conv_rgb(rgb)
                output_stage1_lidar = self.res_conv_xyz(lidar)
                # Apply layer normalization
                output_stage1_rgb = self.ln_rgb(output_stage1_rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                output_stage1_lidar = self.ln_lidar(output_stage1_lidar.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                
                # Spatial adaptive weighting
                beta_map = self.beta_attn(torch.cat([output_stage1_rgb, output_stage1_lidar], dim=1))  # [B, C, H, W]
                weighted_avg = (1 - beta_map) * output_stage1_rgb + beta_map * output_stage1_lidar
                output_stage1 = output_stage1_rgb + output_stage1_lidar + previous_stage + self.alpha * weighted_avg
            elif self.fusion_strategy == 'gmf':
                output_stage1_rgb = self.res_conv_rgb(rgb)
                output_stage1_lidar = self.res_conv_xyz(lidar)
                # Geometric Modulation Fusion
                lidar_pooled = self.modulation_mlp(output_stage1_lidar)
                gamma, beta = lidar_pooled.chunk(2, dim=-1)
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # For broadcasting
                beta = beta.unsqueeze(-1).unsqueeze(-1)
                rgb_mod = gamma * output_stage1_rgb + beta
                # Gated residual
                output_stage1 = rgb_mod + output_stage1_lidar + previous_stage + self.alpha * (rgb_mod + output_stage1_lidar)
            elif self.fusion_strategy == 'gated_average':
                output_stage1_rgb = self.res_conv_rgb(rgb)
                output_stage1_lidar = self.res_conv_xyz(lidar)
                simple_avg = (output_stage1_rgb + output_stage1_lidar) / 2
                output_stage1 = output_stage1_rgb + output_stage1_lidar + previous_stage + self.alpha * simple_avg
            elif self.fusion_strategy == 'residual_average':
                output_stage1_rgb = self.res_conv_rgb(rgb)
                output_stage1_lidar = self.res_conv_xyz(lidar)
                simple_avg = (output_stage1_rgb + output_stage1_lidar) / 2
                output_stage1 = simple_avg + previous_stage
            elif self.fusion_strategy == 'residual_average_no_prev':
                output_stage1_rgb = self.res_conv_rgb(rgb)
                output_stage1_lidar = self.res_conv_xyz(lidar)
                output_stage1 = (output_stage1_rgb + output_stage1_lidar) / 2
            elif self.fusion_strategy == 'weighted_average':
                output_stage1_rgb = self.res_conv_rgb(rgb)
                output_stage1_lidar = self.res_conv_xyz(lidar)
                weighted_avg = (1 - self.beta) * output_stage1_rgb + self.beta * output_stage1_lidar
                output_stage1 = output_stage1_rgb + output_stage1_lidar + previous_stage + self.alpha * weighted_avg
            elif self.fusion_strategy == 'simple_average':
                # True simple average fusion without resblocks and previous stage
                output_stage1 = (rgb + lidar) / 2
            else:
                raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        output_stage2 = self.res_conv2(output_stage1)
        
        # output_stage2 = nn.functional.interpolate(output_stage2, scale_factor=2, mode="bilinear", align_corners=True)
        return output_stage2


class HeadDepth(nn.Module):
    def __init__(self, resample_dim):
        super(HeadDepth, self).__init__()
        self.conv = nn.Conv2d(resample_dim, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)

class HeadSeg(nn.Module):
    def __init__(self, resample_dim, nclasses):
        super(HeadSeg, self).__init__()
        self.conv = nn.Conv2d(resample_dim, nclasses, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class SwinTransformerFusion(nn.Module):
    def __init__(self,
                 emb_dims=[128, 256, 512, 1024],
                 resample_dim=256,
                 read='ignore',
                 reassemble_s=[4, 8, 16, 32],
                 nclasses=None,
                 type='segmentation',
                 model_timm='swin_base_patch4_window7_224',
                 pretrained=True,
                 fusion_strategy=''
                 ):
        """
        Swin-based fusion model.
        """
        super().__init__()
        if not fusion_strategy:
            raise ValueError("fusion_strategy must be specified and cannot be empty")

        self.transformer_encoders = timm.create_model(model_timm, pretrained=pretrained, features_only=True)
        self.type_ = type
        self.is_spatial = True
        
        # Automatically detect embedding dimensions from the backbone
        if hasattr(self.transformer_encoders, 'feature_info'):
            self.emb_dims = self.transformer_encoders.feature_info.channels()
        elif emb_dims is not None:
            self.emb_dims = emb_dims
        else:
            raise ValueError("emb_dims must be provided if backbone does not have feature_info")
            
        # Reassembles Fusion
        self.reassembles_RGB = []
        self.reassembles_XYZ = []
        self.fusions = []
        for i, s in enumerate(reassemble_s):
            emb_dim_i = self.emb_dims[i]
            self.reassembles_RGB.append(SpatialReassemble(read, s, emb_dim_i, resample_dim))
            self.reassembles_XYZ.append(SpatialReassemble(read, s, emb_dim_i, resample_dim))
            self.fusions.append(Fusion(resample_dim, fusion_strategy=fusion_strategy))
        self.reassembles_RGB = nn.ModuleList(self.reassembles_RGB)
        self.reassembles_XYZ = nn.ModuleList(self.reassembles_XYZ)
        self.fusions = nn.ModuleList(self.fusions)

        #Head
        if type == "full":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        elif type == "depth":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = None
        else:
            self.head_depth = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

    def forward(self, rgb, lidar, modal='rgb'):
        if modal == 'rgb':
            features_rgb = self.transformer_encoders(rgb)
            features_lidar = None
        elif modal == 'lidar':
            features_lidar = self.transformer_encoders(lidar)
            features_rgb = None
        elif modal == 'cross_fusion':
            features_rgb = self.transformer_encoders(rgb)
            features_lidar = self.transformer_encoders(lidar)
        else:
            raise ValueError(f"Invalid modal: {modal}")
        
        previous_stage = None
        for i in range(len(self.fusions)):
            if modal == 'rgb':
                activation_result_rgb = features_rgb[i]
                activation_result_lidar = torch.zeros_like(activation_result_rgb)
            elif modal == 'lidar':
                activation_result_lidar = features_lidar[i]
                activation_result_rgb = torch.zeros_like(activation_result_lidar)
            elif modal == 'cross_fusion':
                # For all fusion strategies, use separate features
                activation_result_rgb = features_rgb[i]
                activation_result_lidar = features_lidar[i]
            
            # Ensure [b, c, h, w]
            if activation_result_rgb.shape[1] != self.emb_dims[i]:
                activation_result_rgb = activation_result_rgb.permute(0, 3, 1, 2)
            if activation_result_lidar.shape[1] != self.emb_dims[i]:
                activation_result_lidar = activation_result_lidar.permute(0, 3, 1, 2)
            
            reassemble_result_RGB = self.reassembles_RGB[i](activation_result_rgb)
            reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result_lidar)
            
            fusion_result = self.fusions[i](reassemble_result_RGB, reassemble_result_XYZ, previous_stage, modal)
            previous_stage = fusion_result
        out_depth = None
        out_segmentation = None
        if self.head_depth != None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation != None:
            out_segmentation = self.head_segmentation(previous_stage)
        return out_depth, out_segmentation

