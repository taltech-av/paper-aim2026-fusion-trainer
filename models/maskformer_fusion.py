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
from scipy.optimize import linear_sum_assignment


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


class MaskFormerCriterion(nn.Module):
    """Bipartite matching loss for MaskFormer / OneFormer.

    For each image in the batch:
      1. Extract GT segments from the semantic annotation map
         (one binary mask + class index per unique class present).
      2. Build a [Q × M] cost matrix (class + mask-BCE + mask-Dice).
      3. Run Hungarian algorithm to find the optimal query → GT assignment.
      4. Compute matched-pair loss (CE + mask BCE + Dice) and
         unmatched-query no-object CE loss.

    The result is averaged over the batch.
    """

    def __init__(self,
                 num_classes: int,
                 cost_class: float = 1.0,
                 cost_mask: float = 20.0,   # paper §A.2: λ_bce=20 in cost matrix
                 cost_dice: float = 1.0,    # paper §A.2: λ_dice=1 in cost matrix
                 weight_class: float = 2.0,
                 weight_mask: float = 5.0,
                 weight_dice: float = 5.0,
                 no_object_coef: float = 0.1,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.num_classes    = num_classes
        self.cost_class     = cost_class
        self.cost_mask      = cost_mask
        self.cost_dice      = cost_dice
        self.weight_class   = weight_class
        self.weight_mask    = weight_mask
        self.weight_dice    = weight_dice
        self.no_object_coef = no_object_coef

        # CE weight tensor: [class_0_w, …, class_{C-1}_w, no_object_w]
        if class_weights is not None:
            eos = torch.cat([class_weights,
                             class_weights.new_tensor([no_object_coef])])
        else:
            eos = None
        # register_buffer so it moves with .to(device) but is not a parameter
        if eos is not None:
            self.register_buffer('ce_weight', eos)
        else:
            self.ce_weight = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _build_gt(self, anno_b: torch.Tensor):
        """Extract per-class binary masks from a [H, W] annotation map.

        Returns list of (class_idx: int, binary_mask: [H, W] float32).
        Background (class 0) is included so the no-object class always has
        at least one segment to compete against.
        """
        segments = []
        for c in range(self.num_classes):
            mask = (anno_b == c).float()
            if mask.sum() > 0:
                segments.append((c, mask))
        return segments

    @torch.no_grad()
    def _cost_matrix(self,
                     cls_logits_q: torch.Tensor,   # [Q, C+1]
                     pred_masks_q: torch.Tensor,   # [Q, h, w]
                     gt_classes:   list[int],
                     gt_masks:     torch.Tensor,   # [M, h, w]
                     ) -> torch.Tensor:             # [Q, M]
        Q = cls_logits_q.shape[0]
        M = len(gt_classes)
        dev = cls_logits_q.device

        # ── Class cost: −p(gt_class) ──────────────────────────────────────────
        cls_probs   = F.softmax(cls_logits_q, dim=-1)          # [Q, C+1]
        gt_idx      = torch.tensor(gt_classes, device=dev)     # [M]
        class_cost  = -cls_probs[:, gt_idx]                    # [Q, M]

        # ── Mask costs (vectorised Q×M) ───────────────────────────────────────
        pred_flat = pred_masks_q.flatten(1)       # [Q, HW]
        pred_sig  = pred_flat.sigmoid()           # [Q, HW]
        gt_flat   = gt_masks.flatten(1)           # [M, HW]
        HW        = pred_flat.shape[1]

        # Binary cross-entropy cost
        p_exp = pred_flat.unsqueeze(1).expand(Q, M, HW)  # [Q, M, HW]
        g_exp = gt_flat.unsqueeze(0).expand(Q, M, HW)    # [Q, M, HW]
        bce_cost = F.binary_cross_entropy_with_logits(
            p_exp, g_exp, reduction='none').mean(-1)      # [Q, M]

        # Dice cost
        num      = 2 * torch.einsum('qn,mn->qm', pred_sig, gt_flat) # [Q, M]
        den      = pred_sig.sum(-1, keepdim=True) + gt_flat.sum(-1).unsqueeze(0) + 1e-5
        dice_cost = 1.0 - num / den                                  # [Q, M]

        return (self.cost_class * class_cost
                + self.cost_mask  * bce_cost
                + self.cost_dice  * dice_cost)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self,
                class_logits: torch.Tensor,   # [B, Q, C+1]
                pred_masks:   torch.Tensor,   # [B, Q, h, w]
                anno:         torch.Tensor,   # [B, H, W]  long
                ) -> torch.Tensor:
        B, Q, _ = class_logits.shape
        _, _, h, w = pred_masks.shape
        dev = class_logits.device

        # Downsample annotation to pixel-decoder resolution for mask matching
        anno_down = F.interpolate(
            anno.float().unsqueeze(1), size=(h, w), mode='nearest'
        ).long().squeeze(1)                          # [B, h, w]

        total_loss = class_logits.new_tensor(0.0)
        no_obj_idx = torch.tensor([self.num_classes], dtype=torch.long, device=dev)

        for b in range(B):
            segments = self._build_gt(anno_down[b])

            if not segments:
                # Degenerate: push all queries to no-object
                targets = no_obj_idx.expand(Q)
                total_loss = total_loss + self.no_object_coef * F.cross_entropy(
                    class_logits[b], targets,
                    weight=self.ce_weight,
                )
                continue

            gt_classes = [s[0] for s in segments]
            gt_masks   = torch.stack([s[1] for s in segments]).to(dev)  # [M,h,w]

            # ── Hungarian matching ────────────────────────────────────────────
            cost = self._cost_matrix(
                class_logits[b], pred_masks[b], gt_classes, gt_masks)
            row_ind, col_ind = linear_sum_assignment(
                cost.cpu().detach().numpy())

            matched     = set(row_ind.tolist())
            batch_loss  = class_logits.new_tensor(0.0)

            # ── Matched-pair loss ─────────────────────────────────────────────
            for r, c in zip(row_ind, col_ind):
                tgt = torch.tensor([gt_classes[c]], dtype=torch.long, device=dev)

                # Classification
                batch_loss = batch_loss + self.weight_class * F.cross_entropy(
                    class_logits[b, r].unsqueeze(0), tgt,
                    weight=self.ce_weight,
                )
                # Mask – binary cross-entropy
                batch_loss = batch_loss + self.weight_mask * F.binary_cross_entropy_with_logits(
                    pred_masks[b, r], gt_masks[c]
                )
                # Mask – Dice
                p_sig = pred_masks[b, r].sigmoid().flatten()
                g     = gt_masks[c].flatten()
                dice  = 1.0 - (2*(p_sig*g).sum() + 1e-5) / (p_sig.sum() + g.sum() + 1e-5)
                batch_loss = batch_loss + self.weight_dice * dice

            # ── Unmatched queries → no-object + zero-mask ─────────────────────
            unmatched = torch.tensor(
                [q for q in range(Q) if q not in matched],
                dtype=torch.long, device=dev)
            if unmatched.numel() > 0:
                no_obj_targets = no_obj_idx.expand(unmatched.numel())
                batch_loss = batch_loss + self.no_object_coef * F.cross_entropy(
                    class_logits[b][unmatched], no_obj_targets,
                    weight=self.ce_weight,
                )
                # Push unmatched masks toward zero so they don't pollute the segmap
                zero_target = torch.zeros_like(pred_masks[b][unmatched])
                batch_loss = batch_loss + self.no_object_coef * F.binary_cross_entropy_with_logits(
                    pred_masks[b][unmatched], zero_target
                )

            total_loss = total_loss + batch_loss

        return total_loss / B


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
            nhead=8,   # matches MaskFormer paper (Cheng et al., NeurIPS 2021)
            num_layers=6,
            num_queries=num_queries,
            num_classes=num_classes,
        )

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

        # 4. Merge — paper formula (MaskFormer §3.3):
        #    segmap[c] = Σ_q  softmax(class_logits)[q,c] · sigmoid(mask)[q]
        b, _, h, w = masks.shape
        cls_probs  = F.softmax(class_logits, dim=-1)[..., :self.num_classes]  # [B, Q, C]
        mask_probs = torch.sigmoid(masks)                                       # [B, Q, h, w]
        segmap     = torch.einsum('bqc,bqhw->bchw', cls_probs, mask_probs)    # [B, C, h, w]

        # 5. Upsample to original input resolution
        segmap = F.interpolate(segmap, size=(H, W),
                               mode='bilinear', align_corners=False)

        return None, segmap, class_logits, masks