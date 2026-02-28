import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, weight=torch.tensor([1.0, 50.0, 35.0, 45.0]), smooth=1e-5, dice_weight=1.0, ce_weight=1.0):
        super(DiceCELoss, self).__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.accum_dice_score = None

        if weight is not None:
            self.register_buffer('weight', weight.float())
        else:
            self.weight = None

        self.ce = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, logits, targets):

        target_indices = targets.argmax(dim=1)        
        ce_loss = self.ce(logits, target_indices)

        probs = F.softmax(logits, dim=1)

        dims = (2, 3, 4)
        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs + targets, dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        self.accum_dice_score += dice_score.mean(dim=0)
        dice_loss_per_class = 1.0 - dice_score.mean(dim=0)  

        if self.weight is not None:
            dice_loss = (dice_loss_per_class * self.weight).sum() / self.weight.sum()
        else:
            dice_loss = dice_loss_per_class.mean()

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss