
import torch

# Metrics calculator (keeping original implementation)
class MetricsCalculator:
    @staticmethod
    def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
        pred_mask = (torch.sigmoid(pred_mask) > threshold).float()
        
        pred_mask = pred_mask.view(pred_mask.size(0), -1)
        true_mask = true_mask.view(true_mask.size(0), -1)
        
        intersection = (pred_mask * true_mask).sum(dim=1)
        union = pred_mask.sum(dim=1) + true_mask.sum(dim=1) - intersection
        
        iou = intersection / (union + 1e-8)
        return iou.mean().item()
    
    @staticmethod
    def calculate_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
        pred_mask = (torch.sigmoid(pred_mask) > threshold).float()
        
        pred_mask = pred_mask.view(pred_mask.size(0), -1)
        true_mask = true_mask.view(true_mask.size(0), -1)
        
        intersection = (pred_mask * true_mask).sum(dim=1)
        dice = (2 * intersection) / (pred_mask.sum(dim=1) + true_mask.sum(dim=1) + 1e-8)
        
        return dice.mean().item()
