import torch
import torch.nn as nn
import torch.nn.functional as F 


class MultiClassDiceLoss(nn.Module):
    """Dice loss for multi-class segmentation"""
    def __init__(self, smooth: float = 1e-6, num_classes: int = 3, 
                 ignore_index: int = -1, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs shape: (B, C, H, W), targets shape: (B, H, W)
        if inputs.dim() != 4:
            raise ValueError(f"Expected 4D input, got {inputs.dim()}D")
        
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate dice loss for each class
        dice_scores = []
        for i in range(self.num_classes):
            if i == self.ignore_index:
                continue
                
            pred_i = probs[:, i, :, :].contiguous().view(-1)
            target_i = targets_one_hot[:, i, :, :].contiguous().view(-1)
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_score)
        
        # Average dice scores
        dice_scores = torch.stack(dice_scores)
        
        if self.reduction == 'mean':
            return 1 - dice_scores.mean()
        elif self.reduction == 'sum':
            return dice_scores.size(0) - dice_scores.sum()
        else:
            return 1 - dice_scores


class MultiClassFocalLoss(nn.Module):
    """Focal loss for multi-class segmentation"""
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, 
                 num_classes: int = 3, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs shape: (B, C, H, W), targets shape: (B, H, W)
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        
        # Get class probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            
            # Gather alpha values for each pixel's class
            alpha_t = self.alpha.gather(0, targets.view(-1)).view_as(targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiClassCombinedLoss(nn.Module):
    """Combined loss for multi-class segmentation"""
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.3, 
                 focal_weight: float = 0.2, num_classes: int = 3,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Cross entropy with optional class weights
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = MultiClassDiceLoss(num_classes=num_classes)
        self.focal = MultiClassFocalLoss(alpha=class_weights, num_classes=num_classes)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure targets are long type for CE loss
        targets = targets.squeeze(1) if targets.dim() == 4 else targets
        
        ce_loss = self.ce(inputs, targets.long())
        dice_loss = self.dice(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        
        return (self.ce_weight * ce_loss + 
                self.dice_weight * dice_loss + 
                self.focal_weight * focal_loss)
