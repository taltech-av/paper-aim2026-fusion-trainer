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
    def __init__(self, p, s, image_height, emb_dim, resample_dim):
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
    def __init__(self, image_size, read, p, s, emb_dim, resample_dim):
        """
        Modified for spatial inputs [b, c, h, w]
        """
        super(SpatialReassemble, self).__init__()
        channels, image_height, image_width = image_size

        # For spatial, read and concat are identity
        self.read = nn.Identity()
        self.concat = nn.Identity()

        #Projection + Resample
        self.resample = Resample(p, s, image_height, emb_dim, resample_dim)

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

class Fusion(nn.Module):
    def __init__(self, resample_dim):
        super(Fusion, self).__init__()
        self.res_conv_xyz = ResidualConvUnit(resample_dim)
        self.res_conv_rgb = ResidualConvUnit(resample_dim)
        self.res_conv2 = ResidualConvUnit(resample_dim)

    def forward(self, rgb, lidar, previous_stage=None, modal = 'rgb'):
        if previous_stage == None:
                previous_stage = torch.zeros_like(rgb)

        if modal == 'rgb':
            output_stage1_rgb = self.res_conv_rgb(rgb)
            output_stage1_lidar = torch.zeros_like(output_stage1_rgb)
        if modal == 'lidar':
            output_stage1_lidar = self.res_conv_xyz(lidar)
            output_stage1_rgb = torch.zeros_like(output_stage1_lidar)
        if modal == 'cross_fusion': 
            output_stage1_rgb = self.res_conv_rgb(rgb)
            output_stage1_lidar = self.res_conv_xyz(lidar)

        output_stage1 = output_stage1_lidar + output_stage1_rgb + previous_stage
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
                 RGB_tensor_size=None,
                 XYZ_tensor_size=None,
                 patch_size=4,
                 emb_dims=[128, 256, 512, 1024],
                 resample_dim=256,
                 read='ignore',
                 hooks=[1, 3, 21, 23],
                 reassemble_s=[4, 8, 16, 32],
                 nclasses=None,
                 type='segmentation',
                 model_timm='swin_base_patch4_window7_224',
                 pretrained=True
                 ):
        """
        Swin-based fusion model.
        """
        super().__init__()

        self.transformer_encoders = timm.create_model(model_timm, pretrained=pretrained, features_only=True)
        self.type_ = type
        self.is_spatial = True
        if emb_dims is None:
            self.emb_dims = self.transformer_encoders.feature_info.channels()
        else:
            self.emb_dims = emb_dims
        self.patch_size = patch_size  # Not used for spatial, but for Reassemble

        # Reassembles Fusion
        self.reassembles_RGB = []
        self.reassembles_XYZ = []
        self.fusions = []
        for i, s in enumerate(reassemble_s):
            emb_dim_i = self.emb_dims[i]
            self.reassembles_RGB.append(SpatialReassemble(RGB_tensor_size, read, self.patch_size, s, emb_dim_i, resample_dim))
            self.reassembles_XYZ.append(SpatialReassemble(XYZ_tensor_size, read, self.patch_size, s, emb_dim_i, resample_dim))
            self.fusions.append(Fusion(resample_dim))
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
        if modal == 'rgb' or modal == 'cross_fusion':
            features = self.transformer_encoders(rgb)  # Use RGB for camera-based modes
        else:
            features = self.transformer_encoders(lidar)  # Use LiDAR for LiDAR-only
        previous_stage = None
        for i in range(len(self.fusions)):
            activation_result = features[i]
            # Ensure [b, c, h, w]
            if activation_result.shape[1] != self.emb_dims[i]:
                activation_result = activation_result.permute(0, 3, 1, 2)  # assume [b, h, w, c] -> [b, c, h, w]
            if modal == 'rgb':
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result)
                reassemble_result_XYZ = torch.zeros_like(reassemble_result_RGB)
            if modal == 'lidar':
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result)
                reassemble_result_RGB = torch.zeros_like(reassemble_result_XYZ)
            if modal == 'cross_fusion':
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result)
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result)
            
            fusion_result = self.fusions[i](reassemble_result_RGB, reassemble_result_XYZ, previous_stage, modal)
            previous_stage = fusion_result
        out_depth = None
        out_segmentation = None
        if self.head_depth != None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation != None:
            out_segmentation = self.head_segmentation(previous_stage)
        return out_depth, out_segmentation

