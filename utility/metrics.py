import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, weight=torch.tensor([[0.1, 15, 25, 130]]), smooth=1e-5, dice_weight=1.0, ce_weight=1.0):
        super(DiceCELoss, self).__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss(weight=weight)
        # Register as buffer so it moves with .to(device) automatically
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        target_indices = targets.argmax(dim=1)
        ce_loss = self.ce(logits, target_indices)

        probs = F.softmax(logits, dim=1)

        dims = (2, 3, 4)
        intersection = torch.sum(probs * targets, dims)   # [B, C]
        cardinality = torch.sum(probs + targets, dims)    # [B, C]

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)  # [B, C]
        dice_loss_per_class = 1.0 - dice_score.mean(dim=0)  # [C]

        if self.weight is not None:
            dice_loss = (dice_loss_per_class * self.weight).sum() / self.weight.sum()
        else:
            dice_loss = dice_loss_per_class.mean()

        loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        return loss
    

def compare_masks(real_mask : np.array, predicted_mask : torch.Tensor):

    pass

def calculate_IoU():

    pass