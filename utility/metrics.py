import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-5, dice_weight=1.0, ce_weight=1.0):
 
        super(DiceCELoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):

        # ---- Cross Entropy ----
        # Convert one-hot target to class indices
        targets_ce = torch.argmax(targets, dim=1)  # (B, D, H, W)
        ce_loss = self.ce(logits, targets_ce)

        probs = F.softmax(logits, dim=1)

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs + targets, dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        dice_loss = 1.0 - dice_score.mean()

        # ---- Combined ----
        loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        return loss
    

def compare_masks(real_mask : np.array, predicted_mask : torch.Tensor):

    pass

def calculate_IoU():

    pass