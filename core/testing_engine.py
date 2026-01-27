#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing engine for model evaluation.
"""
import time
import torch
import numpy as np
from tqdm import tqdm
from utils.helpers import relabel_annotation


class TestingEngine:
    """Handles model testing and evaluation."""
    
    def __init__(self, model, metrics_calculator, config, device):
        self.model = model
        self.metrics_calc = metrics_calculator
        self.config = config
        self.device = device
        self.dataset_name = config['Dataset']['name']
    
    def test(self, dataloader, modality, num_classes):
        """Run testing on a dataloader and return results."""
        self.model.eval()
        
        # Initialize accumulators for IoU/precision/recall
        accumulators = self.metrics_calc.create_accumulators(self.device)
        
        # Initialize storage for AP calculation (pixel-wise predictions and targets)
        all_predictions = {cls: [] for cls in self.metrics_calc.eval_classes}
        all_targets = {cls: [] for cls in self.metrics_calc.eval_classes}
        
        # Track inference time
        total_inference_time = 0.0
        total_samples = 0
        total_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader)
            for i, batch in enumerate(progress_bar):
                # Move data to device
                rgb = batch['rgb'].to(self.device, non_blocking=True)
                lidar = batch['lidar'].to(self.device, non_blocking=True)
                anno = batch['anno'].to(self.device, non_blocking=True)
                
                # Prepare inputs
                rgb_input, lidar_input = self._prepare_inputs(rgb, lidar, modality)
                
                # Synchronize for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Time the forward pass
                inference_start = time.time()
                
                # Forward pass
                _, output_seg = self.model(rgb_input, lidar_input, modality)
                output_seg = output_seg.squeeze(1)
                
                # Synchronize and record time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.time() - inference_start
                total_inference_time += inference_time
                total_samples += rgb.size(0)
                total_batches += 1
                
                # Relabel annotation
                anno = relabel_annotation(anno.cpu(), self.config).squeeze(0).to(self.device)
                
                # Update accumulators for IoU/precision/recall
                batch_overlap, batch_pred, batch_label, batch_union = self.metrics_calc.update_accumulators(
                    accumulators, output_seg, anno, num_classes
                )
                
                # Store predictions and targets for AP calculation
                self._store_predictions_for_ap(output_seg, anno, all_predictions, all_targets)
                
                # Calculate batch metrics for progress bar
                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                array_indices = self._get_array_indices()
                
                # Update progress bar
                progress_desc = ' '.join([
                    f'{cls.upper()}:IoU->{batch_IoU[array_indices[j]]:.4f}'
                    for j, cls in enumerate(self.metrics_calc.eval_classes)
                ])
                progress_bar.set_description(progress_desc)
        
        # Compute final metrics including proper AP
        results = self._compute_final_results(accumulators, all_predictions, all_targets)
        
        # Print results
        self._print_results(results)
        
        # Return results and inference stats
        inference_stats = {
            'total_inference_time': total_inference_time,
            'total_samples': total_samples,
            'total_batches': total_batches
        }
        
        return results, inference_stats
    
    def _prepare_inputs(self, rgb, lidar, modality):
        """Prepare model inputs based on modality."""
        if modality == 'rgb':
            return rgb, rgb
        elif modality == 'lidar':
            return lidar, lidar
        else:  # cross_fusion
            return rgb, lidar
    
    def _get_array_indices(self):
        """Get array indices for eval classes."""
        if self.dataset_name in ['zod', 'waymo']:
            return [idx - 1 for idx in self.metrics_calc.eval_indices]
        else:
            return self.metrics_calc.eval_indices
    
    def _store_predictions_for_ap(self, output_seg, anno, all_predictions, all_targets):
        """Store pixel-wise predictions and targets for AP calculation."""
        # Apply softmax to get probabilities
        probs = torch.softmax(output_seg, dim=1)  # Shape: [batch, classes, H, W]
        
        # Get predictions (argmax) and targets
        preds = torch.argmax(output_seg, dim=1)  # Shape: [batch, H, W]
        
        # For each evaluation class
        for cls_idx, cls_name in enumerate(self.metrics_calc.eval_classes):
            # Get the training index for this class
            train_idx = self.metrics_calc.eval_indices[cls_idx]
            
            # Get probabilities for this class
            cls_probs = probs[:, train_idx, :, :]  # Shape: [batch, H, W]
            
            # Get binary predictions and targets for this class
            cls_preds = (preds == train_idx).float()  # Shape: [batch, H, W]
            cls_targets = (anno == train_idx).float()  # Shape: [batch, H, W]
            
            # Flatten and store
            cls_probs_flat = cls_probs.flatten()
            cls_preds_flat = cls_preds.flatten()
            cls_targets_flat = cls_targets.flatten()
            
            # Only store pixels that are predicted as this class OR are actually this class
            # This ensures we have both true positives and false positives
            relevant_mask = (cls_preds_flat > 0) | (cls_targets_flat > 0)
            
            if relevant_mask.sum() > 0:
                all_predictions[cls_name].append(cls_probs_flat[relevant_mask])
                all_targets[cls_name].append(cls_targets_flat[relevant_mask])
    
    def _compute_final_results(self, accumulators, all_predictions, all_targets):
        """Compute final test results with proper AP calculation."""
        cum_IoU = accumulators['overlap'] / accumulators['union']
        cum_precision = accumulators['overlap'] / accumulators['pred']
        cum_recall = accumulators['overlap'] / accumulators['label']
        
        # Filter to eval classes
        array_indices = self._get_array_indices()
        eval_IoU = cum_IoU[array_indices]
        eval_precision = cum_precision[array_indices]
        eval_recall = cum_recall[array_indices]
        
        # Calculate F1 and AP
        results = {}
        for i, cls in enumerate(self.metrics_calc.eval_classes):
            iou = self.metrics_calc.sanitize_value(eval_IoU[i].item())
            precision = self.metrics_calc.sanitize_value(eval_precision[i].item())
            recall = self.metrics_calc.sanitize_value(eval_recall[i].item())
            
            # Calculate F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            f1 = self.metrics_calc.sanitize_value(f1)
            
            # Calculate proper AP using stored predictions
            ap = self._compute_ap_for_class(cls, all_predictions, all_targets)
            ap = self.metrics_calc.sanitize_value(ap)
            
            results[cls] = {
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'ap': ap
            }
        
        # Calculate additional overall metrics
        pixel_accuracy = accumulators['pixel_correct'] / accumulators['pixel_total'] if accumulators['pixel_total'] > 0 else 0.0
        mean_accuracy = torch.mean(eval_recall).item()  # Mean of per-class recalls
        fw_iou = 0.0
        if accumulators['class_pixels'].sum() > 0:
            weights = accumulators['class_pixels'] / accumulators['class_pixels'].sum()
            fw_iou = (weights * eval_IoU).sum().item()
        
        # Add overall metrics
        results['overall'] = {
            'mIoU_foreground': torch.mean(eval_IoU).item(),
            'mean_accuracy': mean_accuracy,
            'fw_iou': fw_iou,
            'pixel_accuracy': pixel_accuracy.item(),
            'confusion_matrix': accumulators['confusion_matrix'].cpu().tolist()
        }
        
        return results
    
    def _compute_ap_for_class(self, cls_name, all_predictions, all_targets):
        """Compute Average Precision for a single class using proper method."""
        if cls_name not in all_predictions or not all_predictions[cls_name]:
            return 0.0
        
        # Concatenate all predictions and targets for this class
        pred_probs = torch.cat(all_predictions[cls_name])  # All predicted probabilities
        pred_targets = torch.cat(all_targets[cls_name])    # All ground truth labels
        
        if len(pred_probs) == 0:
            return 0.0
        
        # Sort by prediction confidence (descending)
        sorted_indices = torch.argsort(pred_probs, descending=True)
        pred_probs = pred_probs[sorted_indices]
        pred_targets = pred_targets[sorted_indices]
        
        # Calculate precision and recall at different thresholds
        num_positives = pred_targets.sum().item()
        if num_positives == 0:
            return 0.0
        
        # Calculate cumulative true positives and false positives
        tp = torch.cumsum(pred_targets, dim=0).float()
        fp = torch.cumsum(1 - pred_targets, dim=0).float()
        
        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-6)
        recall = tp / num_positives
        
        # Use VOC 2010 AP calculation method
        ap = self._voc_ap(recall, precision)
        return ap
    
    def _voc_ap(self, recall, precision):
        """Calculate AP using VOC 2010 method."""
        if len(recall) == 0:
            return 0.0
        
        # Convert to numpy
        recall = recall.cpu().numpy()
        precision = precision.cpu().numpy()
        
        # Add sentinel values
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        
        # Make precision monotonically decreasing
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # Find points where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # Calculate AP
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    
    def _print_results(self, results):
        """Print test results."""
        print('-----------------------------------------')
        for cls, metrics in results.items():
            if cls == 'overall':
                continue
            print(f'{cls.upper()}: IoU->{metrics["iou"]:.4f} '
                  f'Precision->{metrics["precision"]:.4f} '
                  f'Recall->{metrics["recall"]:.4f} '
                  f'F1->{metrics["f1_score"]:.4f} '
                  f'AP->{metrics["ap"]:.4f}')
        
        # Print overall metrics
        overall = results.get('overall', {})
        if overall:
            print('-----------------------------------------')
            print(f'mIoU (foreground): {overall.get("mIoU_foreground", 0):.4f}')
            print(f'Mean Accuracy: {overall.get("mean_accuracy", 0):.4f}')
            print(f'Frequency-Weighted IoU: {overall.get("fw_iou", 0):.4f}')
            print(f'Pixel Accuracy: {overall.get("pixel_accuracy", 0):.4f}')
        print('-----------------------------------------')
