#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics calculation utilities.
"""
import torch
import math


class MetricsCalculator:
    """Calculate and aggregate training/testing metrics."""
    
    def __init__(self, config, num_eval_classes, find_overlap_func):
        self.config = config
        self.num_eval_classes = num_eval_classes
        self.find_overlap_func = find_overlap_func
        self.eval_classes = self._extract_eval_classes()
        self.eval_indices = self._extract_eval_indices()
        
        # For simple update/compute interface
        self.reset()
    
    def reset(self):
        """Reset accumulators for new evaluation."""
        self.total_overlap = torch.zeros(self.num_eval_classes)
        self.total_pred = torch.zeros(self.num_eval_classes)
        self.total_label = torch.zeros(self.num_eval_classes)
        self.total_union = torch.zeros(self.num_eval_classes)
        self.num_samples = 0
    
    def _extract_eval_classes(self):
        """Extract evaluation class names from config (excludes background at index 0)."""
        train_classes = self.config['Dataset']['train_classes']
        # Get all classes with index > 0, sorted by index
        eval_classes = [
            cls['name'] 
            for cls in sorted(train_classes, key=lambda x: x['index'])
            if cls['index'] > 0
        ]
        return eval_classes
    
    def _extract_eval_indices(self):
        """Extract evaluation class indices from config (excludes background at index 0)."""
        train_classes = self.config['Dataset']['train_classes']
        # Get all indices > 0, sorted
        eval_indices = [
            cls['index'] 
            for cls in sorted(train_classes, key=lambda x: x['index'])
            if cls['index'] > 0
        ]
        return eval_indices
    
    def update(self, outputs, targets):
        """Update metrics with batch predictions and targets."""
        batch_overlap, batch_pred, batch_label, batch_union = \
            self.find_overlap_func(self.num_eval_classes + 1, outputs, targets)
        
        self.total_overlap += batch_overlap.cpu()
        self.total_pred += batch_pred.cpu()
        self.total_label += batch_label.cpu()
        self.total_union += batch_union.cpu()
        self.num_samples += outputs.size(0)
    
    def compute(self):
        """Compute final metrics."""
        # IoU and metrics
        iou = self.total_overlap / (self.total_union + 1e-6)
        precision = self.total_overlap / (self.total_pred + 1e-6)
        recall = self.total_overlap / (self.total_label + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            'mean_iou': torch.mean(iou).item(),
            'mean_precision': torch.mean(precision).item(),
            'mean_recall': torch.mean(recall).item(),
            'mean_f1': torch.mean(f1).item(),
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def compute(self):
        """Compute final metrics."""
        # Calculate IoU and other metrics
        iou = self.total_overlap / (self.total_union + 1e-6)
        precision = self.total_overlap / (self.total_pred + 1e-6)
        recall = self.total_overlap / (self.total_label + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_iou': torch.mean(iou).item(),
            'mean_precision': torch.mean(precision).item(),
            'mean_recall': torch.mean(recall).item(),
            'mean_f1': torch.mean(f1).item()
        }
    
    def create_accumulators(self, device):
        """Create metric accumulator tensors."""
        num_classes = len(self.config['Dataset']['train_classes'])
        return {
            'overlap': torch.zeros(self.num_eval_classes).to(device),
            'pred': torch.zeros(self.num_eval_classes).to(device),
            'label': torch.zeros(self.num_eval_classes).to(device),
            'union': torch.zeros(self.num_eval_classes).to(device),
            'pixel_correct': torch.tensor(0.0).to(device),
            'pixel_total': torch.tensor(0.0).to(device),
            'class_pixels': torch.zeros(self.num_eval_classes).to(device),  # For FWIoU
            'confusion_matrix': torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)  # Include background
        }
    
    def update_accumulators(self, accumulators, output_seg, anno, num_classes):
        """Update metric accumulators with batch results."""
        batch_overlap, batch_pred, batch_label, batch_union = \
            self.find_overlap_func(num_classes, output_seg, anno)
        
        accumulators['overlap'] += batch_overlap
        accumulators['pred'] += batch_pred
        accumulators['label'] += batch_label
        accumulators['union'] += batch_union
        
        # Calculate pixel accuracy
        _, pred_indices = torch.max(output_seg, dim=1)
        correct_pixels = (pred_indices == anno).sum().float()
        total_pixels = torch.tensor(anno.numel(), dtype=torch.float, device=accumulators['pixel_correct'].device)
        
        accumulators['pixel_correct'] += correct_pixels
        accumulators['pixel_total'] += total_pixels
        
        # Update class pixel counts for FWIoU
        for i in range(self.num_eval_classes):
            accumulators['class_pixels'][i] += (anno == (i + 1)).sum().item()  # eval classes start from 1
        
        # Update confusion matrix (vectorized)
        indices = num_classes * anno.flatten() + pred_indices.flatten()
        accumulators['confusion_matrix'] += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
        
        return batch_overlap, batch_pred, batch_label, batch_union
    
    def compute_epoch_metrics(self, accumulators, total_loss, num_batches):
        """Compute final metrics for an epoch."""
        # IoU and metrics are already in correct order (class 1, 2, 3, ... -> indices 0, 1, 2, ...)
        epoch_IoU = accumulators['overlap'] / (accumulators['union'] + 1e-6)
        
        # Additional metrics
        precision = accumulators['overlap'] / (accumulators['pred'] + 1e-6)
        recall = accumulators['overlap'] / (accumulators['label'] + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        # Pixel accuracy
        pixel_accuracy = accumulators['pixel_correct'] / (accumulators['pixel_total'] + 1e-6)
        
        # Mean accuracy (average of per-class accuracies)
        # Per-class accuracy = recall for each class
        mean_accuracy = torch.mean(recall)
        
        # Dice score (F1 score) - already calculated as f1
        
        # Average loss
        if num_batches > 0:
            epoch_loss = total_loss / num_batches
        else:
            epoch_loss = 0.0
        
        return {
            'epoch_IoU': epoch_IoU,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pixel_accuracy': pixel_accuracy.item(),
            'mean_accuracy': mean_accuracy.item(),
            'dice_score': torch.mean(f1).item(),  # Mean Dice score
            'epoch_loss': epoch_loss,
            'mean_iou': torch.mean(epoch_IoU).item()
        }
    
    def sanitize_value(self, value):
        """Replace NaN/Inf with 0."""
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return 0.0
        return value
    
    def prepare_results_dict(self, train_metrics, val_metrics):
        """Prepare results dictionary for logging."""
        results = {"train": {}, "val": {}}
        
        for i, cls in enumerate(self.eval_classes):
            results["train"][cls] = {
                "iou": self.sanitize_value(train_metrics["epoch_IoU"][i].item()),
                "precision": self.sanitize_value(train_metrics["precision"][i].item()),
                "recall": self.sanitize_value(train_metrics["recall"][i].item()),
                "f1": self.sanitize_value(train_metrics["f1"][i].item())
            }
            results["val"][cls] = {
                "iou": self.sanitize_value(val_metrics["epoch_IoU"][i].item()),
                "precision": self.sanitize_value(val_metrics["precision"][i].item()),
                "recall": self.sanitize_value(val_metrics["recall"][i].item()),
                "f1": self.sanitize_value(val_metrics["f1"][i].item())
            }
        
        results["train"]["loss"] = self.sanitize_value(train_metrics["epoch_loss"])
        results["train"]["mean_iou"] = self.sanitize_value(train_metrics["mean_iou"])
        results["train"]["pixel_accuracy"] = self.sanitize_value(train_metrics["pixel_accuracy"])
        results["train"]["mean_accuracy"] = self.sanitize_value(train_metrics["mean_accuracy"])
        results["train"]["dice_score"] = self.sanitize_value(train_metrics["dice_score"])
        results["val"]["loss"] = self.sanitize_value(val_metrics["epoch_loss"])
        results["val"]["mean_iou"] = self.sanitize_value(val_metrics["mean_iou"])
        results["val"]["pixel_accuracy"] = self.sanitize_value(val_metrics["pixel_accuracy"])
        results["val"]["mean_accuracy"] = self.sanitize_value(val_metrics["mean_accuracy"])
        results["val"]["dice_score"] = self.sanitize_value(val_metrics["dice_score"])
        
        return results
    
    def print_metrics(self, metrics, prefix="", accumulators=None):
        """Print metrics in a formatted way."""
        print(f'{prefix}Mean IoU: {metrics["mean_iou"]:.4f}')
        print(f'{prefix}Pixel Accuracy: {metrics["pixel_accuracy"]:.4f}')
        print(f'{prefix}Mean Accuracy: {metrics["mean_accuracy"]:.4f}')
        print(f'{prefix}Dice Score: {metrics["dice_score"]:.4f}')
        for i, cls in enumerate(self.eval_classes):
            print(f'{prefix}{cls} IoU: {metrics["epoch_IoU"][i]:.4f}')
        print(f'{prefix}Average Loss: {metrics["epoch_loss"]:.4f}')
