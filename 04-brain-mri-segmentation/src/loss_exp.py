
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss (原始版本)"""
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dsc


class FocalLoss(nn.Module):
    """Focal Loss (用于处理类别不平衡)"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        # y_pred: [B, 1, H, W], y_true: [B, 1, H, W]
        y_pred = y_pred[:, 0].contiguous()
        y_true = y_true[:, 0].contiguous()
        
        # 二分类交叉熵
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        
        # Focal loss
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class DiceBCELoss(nn.Module):
    """Dice Loss + BCE Loss (组合损失)"""
    def __init__(self, dice_weight=1.0, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true):
        dice = self.dice_loss(y_pred, y_true)
        bce = self.bce_loss(y_pred, y_true)
        return self.dice_weight * dice + self.bce_weight * bce
