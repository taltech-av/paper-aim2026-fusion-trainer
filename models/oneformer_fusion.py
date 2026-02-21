#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OneFormer-based fusion model for camera-lidar segmentation.

Pure-PyTorch implementation — **no Detectron2 dependency**.

Architecture (faithful to Jain et al., "OneFormer: One Transformer to Rule
Universal Image Segmentation", CVPR 2023):

  Backbone (timm)
       │  multi-scale features
  PixelDecoder (FPN)
       │  dense pixel features  [B, d_model, h, w]
  TaskConditionedDecoder
       │  task token  +  object queries → class logits + binary masks
  Merge: segmap via cross-attention (pixels attend over queries, then weighted
         sum of per-query class logits — gives proper logits for CE loss)

The key difference from MaskFormer is the **task-conditioned query scheme**:
  - A small set of learnable task tokens (one per task: semantic / instance /
    panoptic) is prepended to the N object queries.
  - All queries (task + object) jointly attend to the pixel memory.
  - Task tokens additionally cross-attend to all object-query features,
    so that the task context propagates to every object query.
  - At inference the task token is selected by passing a task index, which
    allows a single model to handle multiple segmentation tasks.

For this codebase only the "semantic" task (index 0) is exercised, but the
architecture is fully multi-task capable.

Camera-lidar fusion is handled by element-wise addition of backbone feature
maps (early-fusion variant), identical to MaskFormerFusion.

Returns (None, segmap) — drop-in replacement for all other models in this
codebase; compatible with the CrossEntropyLoss training engine.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ──────────────────────────────────────────────────────────────────────────────
# Shared components (mirrors maskformer_fusion.py; kept here for
# self-containedness so each model file is independently importable)
# ──────────────────────────────────────────────────────────────────────────────

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
    """FPN-style pixel decoder.  See maskformer_fusion.py for documentation."""

    def __init__(self, in_channels_list: list[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels_list]
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        out = laterals[-1]
        for lat in reversed(laterals[:-1]):
            out = F.interpolate(out, size=lat.shape[2:],
                                mode='bilinear', align_corners=False)
            out = out + lat
        return self.output_conv(out)


# ──────────────────────────────────────────────────────────────────────────────
# OneFormer-specific components
# ──────────────────────────────────────────────────────────────────────────────

class TaskConditionedDecoder(nn.Module):
    """Transformer decoder with task-conditional query scheme (OneFormer §3.2).

    Compared to the plain MaskFormer TransformerDecoder:
    1. k learnable *task tokens* are prepended to the N object queries,
       giving a combined sequence of length (k + N).
    2. All tokens jointly cross-attend to the pixel memory.
    3. An additional *task guidance* cross-attention layer propagates the task
       context from the task tokens back into the object-query features.

    At inference ``task_idx`` selects which task token to use (semantic=0,
    instance=1, panoptic=2).  Predictions are built only from the N object
    query outputs (task token outputs are discarded).
    """

    # Number of supported tasks (semantic / instance / panoptic)
    NUM_TASKS: int = 3
    # Number of task tokens prepended per forward pass (= 1, the chosen task)
    NUM_TASK_TOKENS: int = 1

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

        # ── Task token bank ───────────────────────────────────────────────────
        # One embedding per task; one is selected per forward call.
        self.task_embed = nn.Embedding(self.NUM_TASKS, d_model)

        # ── Object query embeddings ───────────────────────────────────────────
        self.query_embed = nn.Embedding(num_queries, d_model)

        # ── Main transformer decoder (queries + task token → pixel memory) ────
        joint_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.joint_decoder = nn.TransformerDecoder(
            joint_decoder_layer, num_layers=num_layers
        )

        # ── Task-guidance layer (object queries cross-attend to task token) ───
        # Applied once after the main decoder to let task context modulate
        # each object query.  Uses a single cross-attention layer.
        self.task_guidance = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=0.1, batch_first=True,
        )
        self.task_guidance_norm = nn.LayerNorm(d_model)

        # ── Output heads ──────────────────────────────────────────────────────
        self.class_embed = nn.Linear(d_model, num_classes + 1)   # +1 no-object
        self.mask_embed  = nn.Linear(d_model, d_model)

    def forward(self,
                pixel_features: torch.Tensor,
                task_idx: int = 0):
        """
        Args:
            pixel_features : [B, C, H, W]  (output of PixelDecoder)
            task_idx       : int in {0=semantic, 1=instance, 2=panoptic}
        Returns:
            class_logits : [B, num_queries, num_classes+1]
            masks        : [B, num_queries, H, W]   (raw logits)
        """
        b, c, h, w = pixel_features.shape
        memory = pixel_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # ── Build combined token sequence: [task_token | object_queries] ──────
        # task token: [B, 1, C]
        task_idx_t = torch.tensor([task_idx], device=pixel_features.device)
        task_token = self.task_embed(task_idx_t)              # [1, C]
        task_token = task_token.unsqueeze(0).expand(b, -1, -1) # [B, 1, C]

        # object queries: [B, Q, C]
        obj_queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)

        # concatenate → [B, 1+Q, C]
        combined = torch.cat([task_token, obj_queries], dim=1)

        # ── Joint cross-attention to pixel memory ─────────────────────────────
        decoded = self.joint_decoder(combined, memory)  # [B, 1+Q, C]

        # Split back
        task_out  = decoded[:, :self.NUM_TASK_TOKENS, :]   # [B, 1, C]
        query_out = decoded[:, self.NUM_TASK_TOKENS:, :]   # [B, Q, C]

        # ── Task-guidance: object queries cross-attend to task output ─────────
        guided, _ = self.task_guidance(
            query=query_out,    # [B, Q, C]  queries
            key=task_out,       # [B, 1, C]  keys
            value=task_out,     # [B, 1, C]  values
        )
        query_out = self.task_guidance_norm(query_out + guided)

        # ── Prediction heads ──────────────────────────────────────────────────
        class_logits = self.class_embed(query_out)          # [B, Q, C+1]
        mask_feat    = self.mask_embed(query_out)           # [B, Q, d_model]
        # Scale by 1/√d_model to prevent sigmoid saturation at initialisation
        masks = torch.einsum('bqc,bchw->bqhw', mask_feat,
                             pixel_features) / math.sqrt(self.d_model)

        return class_logits, masks


# ──────────────────────────────────────────────────────────────────────────────
# Top-level fusion model
# ──────────────────────────────────────────────────────────────────────────────

class OneFormerFusion(nn.Module):
    """OneFormer with camera-lidar fusion backbone.

    Architectural differences from ``MaskFormerFusion``:
    - Uses ``TaskConditionedDecoder`` instead of ``TransformerDecoder``.
    - Supports multi-task inference via ``task_idx`` (0=semantic by default).
    - A ``config_file`` argument is accepted for backward-compatibility with
      the training script but is ignored (no Detectron2 config needed).

    Forward interface is otherwise identical: returns ``(None, segmap)`` where
    segmap is ``[B, num_classes, H, W]``.
    """

    def __init__(self,
                 backbone: str = 'swin_base_patch4_window7_224',
                 num_classes: int = 4,
                 pixel_decoder_channels: int = 256,
                 transformer_d_model: int = 256,
                 num_queries: int = 100,
                 pretrained: bool = True,
                 config_file: str | None = None,   # legacy arg, unused
                 task_idx: int = 0):               # 0=semantic (default)
        super().__init__()

        self.num_classes = num_classes
        self.task_idx = task_idx   # can be overridden at call time

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True
        )
        bb_channels = self.backbone.feature_info.channels()

        # ── Residual fusion units (residual_average strategy) ─────────────────
        # One ResidualConvUnit per backbone scale, per modality stream.
        # Mirrors swin_transformer_fusion.py's residual_average:
        #   fused_i = res_rgb_i(feat_rgb_i) + res_lidar_i(feat_lidar_i) + prev
        self.fusion_res_rgb   = nn.ModuleList(
            [ResidualConvUnit(c) for c in bb_channels])
        self.fusion_res_lidar = nn.ModuleList(
            [ResidualConvUnit(c) for c in bb_channels])

        # ── Pixel Decoder (FPN) ───────────────────────────────────────────────
        self.pixel_decoder = PixelDecoder(bb_channels, pixel_decoder_channels)

        # Project to d_model if needed
        self.pixel_proj = (
            nn.Conv2d(pixel_decoder_channels, transformer_d_model, 1)
            if pixel_decoder_channels != transformer_d_model else nn.Identity()
        )

        # ── Task-Conditioned Transformer Decoder ──────────────────────────────
        self.transformer_decoder = TaskConditionedDecoder(
            d_model=transformer_d_model,
            nhead=max(1, transformer_d_model // 64),
            num_layers=6,
            num_queries=num_queries,
            num_classes=num_classes,
        )

        # ── Direct pixel classification head ─────────────────────────────────
        # FCN-style shortcut that provides reliable dense-prediction gradient
        # from epoch 0, preventing query-attention collapse during eval mode.
        # The transformer decoder output is added on top of this.
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
                modal: str = 'fusion',
                task_idx: int | None = None):
        """
        Args:
            rgb      : [B, 3, H, W]
            lidar    : [B, C_l, H, W]
            modal    : 'rgb' | 'lidar' | 'fusion' | 'cross_fusion'
            task_idx : override default task (0=semantic, 1=instance, 2=panoptic)
        Returns:
            (None, segmap)  where segmap is [B, num_classes, H, W]
        """
        H, W = rgb.shape[-2], rgb.shape[-1]
        task = task_idx if task_idx is not None else self.task_idx

        # 1. Backbone + fusion
        features = self._extract_features(rgb, lidar, modal)

        # 2. FPN pixel decoder
        pixel_features = self.pixel_decoder(features)          # [B, pd_ch, h, w]
        pixel_features = self.pixel_proj(pixel_features)       # [B, d_model, h, w]

        # 3. Task-conditioned transformer decoder
        class_logits, masks = self.transformer_decoder(pixel_features, task_idx=task)
        # class_logits : [B, Q, num_classes+1]
        # masks        : [B, Q, h, w]

        # 4. Direct pixel classification (FCN-style shortcut)
        direct_seg = self.direct_head(pixel_features)              # [B, C, h, w]

        # 5. Transformer decoder merge — MaskFormer paper formula:
        #    segmap[c] = Σ_q  softmax(class_logits)[q,c] · sigmoid(mask)[q]
        b, _, h, w = masks.shape
        cls_probs  = F.softmax(class_logits, dim=-1)[..., :self.num_classes]  # [B, Q, C]
        mask_probs = torch.sigmoid(masks)                                       # [B, Q, h, w]
        query_seg  = torch.einsum('bqc,bqhw->bchw', cls_probs, mask_probs)    # [B, C, h, w]

        # Sum direct + query paths
        segmap = direct_seg + query_seg

        # 6. Upsample to original input resolution
        segmap = F.interpolate(segmap, size=(H, W),
                               mode='bilinear', align_corners=False)

        return None, segmap, class_logits, masks
