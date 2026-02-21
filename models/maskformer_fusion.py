#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaskFormer-based fusion model for camera-lidar segmentation.

Pure-PyTorch implementation — no Detectron2 dependency.

Architecture (faithful to Cheng et al., "Per-Pixel Classification is Not All
You Need for Semantic Segmentation", NeurIPS 2021):

  1. Shared backbone (timm) extracts multi-scale features.
  2. PixelDecoder (FPN-style) produces rich pixel-level feature maps.
  3. TransformerDecoder cross-attends N learnable object queries against the
     pixel features, producing per-query class logits and binary mask
     predictions.
  4. At inference the query outputs are merged into a dense segmentation map:
       segmap[c] = Σ_q  softmax(class_logits_q)[c] · sigmoid(mask_q)
     so the output is compatible with the existing CrossEntropyLoss training
     engine without any wiring changes.

Camera-lidar fusion is handled by element-wise addition of rgb and lidar
backbone features (early-fusion variant), identical to other models in this
codebase.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ResidualConvUnit(nn.Module):
    """3×3 residual conv block — matches swin_transformer_fusion.py."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class PixelDecoder(nn.Module):
    """FPN-style pixel decoder (multi-scale → single high-res feature map).

    Takes the list of backbone feature maps (from finest to coarsest resolution)
    and produces a single feature map at the finest resolution via lateral
    connections and top-down upsampling.
    """

    def __init__(self, in_channels_list: list[int], out_channels: int = 256):
        super().__init__()

        # 1×1 lateral projections for each scale
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels_list]
        )
        # Output refinement
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of [B, C_i, H_i, W_i], ordered fine→coarse.
        Returns:
            [B, out_channels, H_0, W_0] at the finest resolution.
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down path: start from coarsest, upsample and add
        out = laterals[-1]
        for lateral in reversed(laterals[:-1]):
            out = F.interpolate(out, size=lateral.shape[2:],
                                mode='bilinear', align_corners=False)
            out = out + lateral

        return self.output_conv(out)


class TransformerDecoder(nn.Module):
    """Per-object-query transformer decoder (MaskFormer §3.2).

    N learnable query embeddings cross-attend to the pixel-level memory
    produced by the PixelDecoder.  Each query predicts:
      - a class distribution over (num_classes + 1) labels (the extra label is
        the "∅ / no-object" token)
      - a binary foreground mask over the spatial grid

    The masks are computed as the dot product between the query's mask
    embedding and every pixel's feature vector, following the original paper.
    """

    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 num_queries: int = 100,
                 num_classes: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Standard PyTorch transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Classification head: num_classes + 1 (no-object)
        self.class_embed = nn.Linear(d_model, num_classes + 1)

        # Mask embedding: projects query features → spatial mask logits
        self.mask_embed = nn.Linear(d_model, d_model)

    def forward(self, pixel_features: torch.Tensor):
        """
        Args:
            pixel_features: [B, C, H, W]  (output of PixelDecoder)
        Returns:
            class_logits : [B, num_queries, num_classes+1]
            masks        : [B, num_queries, H, W]   (raw logits, not sigmoid)
        """
        b, c, h, w = pixel_features.shape
        # Flatten spatial dims for cross-attention memory
        memory = pixel_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # Broadcast queries over the batch
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # [B, Q, C]

        # Cross-attend queries to pixel memory
        decoded = self.decoder(queries, memory)  # [B, Q, C]

        # Per-query class distribution
        class_logits = self.class_embed(decoded)  # [B, Q, num_classes+1]

        # Per-query mask: dot product with every pixel feature.
        # Scale by 1/√d_model to prevent sigmoid saturation at initialisation
        # (same reasoning as scaled dot-product attention).
        mask_feat = self.mask_embed(decoded)                          # [B, Q, C]
        masks = torch.einsum('bqc,bchw->bqhw', mask_feat,
                             pixel_features) / math.sqrt(self.d_model)

        return class_logits, masks


class MaskFormerFusion(nn.Module):
    """MaskFormer with camera-lidar fusion backbone.

    Differences from the original MaskFormer:
    - Backbone is any timm model (not necessarily a Swin Transformer).
    - Camera and LiDAR streams share a single backbone; their features are
      fused by element-wise addition before the pixel decoder.
    - Returns (None, segmap) where segmap is a dense [B, C, H, W] logit
      tensor, so the model is a drop-in replacement for all other models in
      this codebase and works with the existing CrossEntropyLoss training
      engine.

    The dense segmap is produced by the standard MaskFormer inference step:
        segmap[:, c, :, :] = Σ_q  softmax(class_logits_q)[c] · sigmoid(mask_q)
    """

    def __init__(self,
                 backbone: str = 'swin_base_patch4_window7_224',
                 num_classes: int = 4,
                 pixel_decoder_channels: int = 256,
                 transformer_d_model: int = 256,
                 num_queries: int = 100,
                 pretrained: bool = True):
        super().__init__()

        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True
        )
        bb_channels = self.backbone.feature_info.channels()

        # ── Residual fusion units (residual_average strategy) ─────────────────
        # One ResidualConvUnit per backbone scale, per modality stream.
        # Mirrors the design in swin_transformer_fusion.py:
        #   fused_i = res_rgb_i(feat_rgb_i) + res_lidar_i(feat_lidar_i) + prev
        self.fusion_res_rgb   = nn.ModuleList(
            [ResidualConvUnit(c) for c in bb_channels])
        self.fusion_res_lidar = nn.ModuleList(
            [ResidualConvUnit(c) for c in bb_channels])

        # ── Pixel Decoder (FPN) ───────────────────────────────────────────────
        self.pixel_decoder = PixelDecoder(bb_channels, pixel_decoder_channels)

        # Project pixel decoder output to transformer d_model if sizes differ
        self.pixel_proj = (
            nn.Conv2d(pixel_decoder_channels, transformer_d_model, 1)
            if pixel_decoder_channels != transformer_d_model else nn.Identity()
        )

        # ── Transformer Decoder ───────────────────────────────────────────────
        self.transformer_decoder = TransformerDecoder(
            d_model=transformer_d_model,
            nhead=max(1, transformer_d_model // 64),   # ≥1 head, ~1 head/64ch
            num_layers=6,
            num_queries=num_queries,
            num_classes=num_classes,
        )

        # ── Direct pixel classification head ─────────────────────────────────
        # Provides an immediate dense-prediction shortcut that produces strong
        # gradient signal from epoch 0.  The transformer decoder output is
        # *added* to this, so both paths are trained jointly.  Without this,
        # early in training the Q=100 softmax attention is nearly uniform
        # (1/Q per query), averaging all query class logits into a background-
        # dominated prediction that collapses eval-mode metrics to zero.
        self.direct_head = nn.Conv2d(transformer_d_model, num_classes, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_bchw(f: torch.Tensor) -> torch.Tensor:
        """Normalise a backbone feature map to [B, C, H, W]."""
        if f.ndim == 4 and f.shape[1] < f.shape[-1]:   # BHWC → BCHW
            return f.permute(0, 3, 1, 2).contiguous()
        return f

    def _extract_features(self, rgb: torch.Tensor,
                          lidar: torch.Tensor,
                          modal: str) -> list[torch.Tensor]:
        """Run backbone and fuse with residual_average strategy.

        Applies a ResidualConvUnit to each stream at each backbone scale,
        then sums the two streams.  Mirrors swin_transformer_fusion.py's
        residual_average, adapted for multi-scale features with varying
        channel widths (no cross-scale previous-stage threading: the FPN
        PixelDecoder already provides that cross-scale integration).

            fused_i = res_rgb_i(feat_rgb_i) + res_lidar_i(feat_lidar_i)
        """
        if modal == 'rgb':
            return [self.fusion_res_rgb[i](self._to_bchw(f))
                    for i, f in enumerate(self.backbone(rgb))]

        elif modal == 'lidar':
            return [self.fusion_res_lidar[i](self._to_bchw(f))
                    for i, f in enumerate(self.backbone(lidar))]

        elif modal in ('fusion', 'cross_fusion'):
            raw_rgb   = [self._to_bchw(f) for f in self.backbone(rgb)]
            raw_lidar = [self._to_bchw(f) for f in self.backbone(lidar)]
            return [
                self.fusion_res_rgb[i](fr) + self.fusion_res_lidar[i](fl)
                for i, (fr, fl) in enumerate(zip(raw_rgb, raw_lidar))
            ]

        else:
            raise ValueError(f"Unknown modal: {modal!r}")

    def forward(self, rgb: torch.Tensor,
                lidar: torch.Tensor,
                modal: str = 'fusion'):
        """
        Args:
            rgb   : [B, 3, H, W]
            lidar : [B, C_l, H, W]
            modal : 'rgb' | 'lidar' | 'fusion' | 'cross_fusion'
        Returns:
            (None, segmap)  where segmap is [B, num_classes, H, W]
        """
        H, W = rgb.shape[-2], rgb.shape[-1]

        # 1. Backbone + fusion
        features = self._extract_features(rgb, lidar, modal)

        # 2. FPN pixel decoder
        pixel_features = self.pixel_decoder(features)          # [B, pd_ch, h, w]
        pixel_features = self.pixel_proj(pixel_features)       # [B, d_model, h, w]

        # 3. Transformer decoder → per-query class logits + binary masks
        class_logits, masks = self.transformer_decoder(pixel_features)
        # class_logits : [B, Q, num_classes+1]
        # masks        : [B, Q, h, w]  (raw dot-product logits)

        # 4. Direct pixel classification (FCN-style shortcut)
        direct_seg = self.direct_head(pixel_features)              # [B, C, h, w]

        # 5. Transformer decoder merge.
        #    Each pixel attends over Q queries weighted by the mask scores,
        #    then takes a weighted average of the queries' raw class logits.
        b, _, h, w = pixel_features.shape
        attn = F.softmax(
            masks.view(b, self.transformer_decoder.num_queries, -1)  # [B, Q, hw]
                 .permute(0, 2, 1),                                  # [B, hw, Q]
            dim=-1
        )                                                             # [B, hw, Q]
        class_logits_valid = class_logits[..., :self.num_classes]     # [B, Q, C]
        query_seg = torch.bmm(attn, class_logits_valid)              # [B, hw, C]
        query_seg = query_seg.permute(0, 2, 1).view(b, self.num_classes, h, w)

        # Sum the two paths: direct head provides reliable signal from epoch 0;
        # the query path adds structural, object-level refinement over time.
        segmap = direct_seg + query_seg

        # 6. Upsample to original input resolution
        segmap = F.interpolate(segmap, size=(H, W),
                               mode='bilinear', align_corners=False)

        return None, segmap