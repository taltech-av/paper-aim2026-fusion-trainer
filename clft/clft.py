import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from clft.reassemble import Reassemble
from clft.fusion import Fusion
from clft.head import HeadDepth, HeadSeg

torch.manual_seed(0)


class CLFT(nn.Module):
    def __init__(self,
                 RGB_tensor_size=None,
                 XYZ_tensor_size=None,
                 patch_size=None,
                 emb_dim=None,
                 resample_dim=None,
                 read=None,
                 hooks=None,
                 reassemble_s=None,
                 nclasses=None,
                 type=None,
                 model_timm=None,
                 pretrained=True
                 # num_layers_encoder=24,
                 # transformer_dropout=0,
                 ):
        """
        Focus on Depth
        type : {"full", "depth", "segmentation"}
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        self.transformer_encoders = timm.create_model(model_timm, pretrained=pretrained)
        self.type_ = type

        # Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        # Reassembles Fusion
        self.reassembles_RGB = []
        self.reassembles_XYZ = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles_RGB.append(Reassemble(RGB_tensor_size, read, patch_size, s, emb_dim, resample_dim))
            self.reassembles_XYZ.append(Reassemble(XYZ_tensor_size, read, patch_size, s, emb_dim, resample_dim))
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
        if modal == 'rgb':
            self.transformer_encoders(rgb)
            activation_result = self.activation
        elif modal == 'lidar':
            self.transformer_encoders(lidar)
            activation_result = self.activation
        elif modal == 'cross_fusion':
            # Run on rgb to get RGB activations
            self.transformer_encoders(rgb)
            activation_rgb = self.activation.copy()
            # Run on lidar to get LiDAR activations
            self.transformer_encoders(lidar)
            activation_lidar = self.activation.copy()
        
        previous_stage = None
        for i in np.arange(len(self.fusions)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])
            if modal == 'rgb':
                activation_result_current = activation_result[hook_to_take]
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result_current)
                reassemble_result_XYZ = torch.zeros_like(reassemble_result_RGB)
            elif modal == 'lidar':
                activation_result_current = activation_result[hook_to_take]
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result_current)
                reassemble_result_RGB = torch.zeros_like(reassemble_result_XYZ)
            elif modal == 'cross_fusion':
                activation_rgb_current = activation_rgb[hook_to_take]
                activation_lidar_current = activation_lidar[hook_to_take]
                reassemble_result_RGB = self.reassembles_RGB[i](activation_rgb_current)
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_lidar_current)
            
            fusion_result = self.fusions[i](reassemble_result_RGB, reassemble_result_XYZ, previous_stage, modal)
            previous_stage = fusion_result
        out_depth = None
        out_segmentation = None
        if self.head_depth != None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation != None:
            out_segmentation = self.head_segmentation(previous_stage)
        return out_depth, out_segmentation

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t'+str(h)))