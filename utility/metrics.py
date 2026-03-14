import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, weight=torch.tensor([1.0, 40.0, 35.0, 45.0]), smooth=1e-5, dice_weight=1.0, ce_weight=1.0):
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



def predict_mask(model, features_tensor):
    pred_labels = model(features_tensor).detach()
    pred_labels = torch.argmax(pred_labels, dim=1)
    return pred_labels  


def calculate_dice(pred_mask, true_mask, num_classes=4, smooth=1e-6):

    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()

    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()

        intersection = (pred_cls * true_cls).sum()
        union = pred_cls.sum() + true_cls.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())

    return dice_scores


def calculate_iou(pred_mask, true_mask, num_classes=4, smooth=1e-6):

    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()

    iou_scores = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()

        intersection = (pred_cls * true_cls).sum()
        union = pred_cls.sum() + true_cls.sum() - intersection

        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())

    return iou_scores


def calculate_mean_iou_dice(model, title):

    n_samples = 350
    num_classes = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    cumulative_dice = [0.0] * num_classes
    cumulative_iou = [0.0] * num_classes

    with torch.no_grad():

        for idx in range(1, n_samples + 1):

            path_scan = f"data/BraTS2020_npy/data/{idx}/combined_scan.npy"
            path_mask = f"data/BraTS2020_npy/data/{idx}/mask.npy"

            features = np.load(path_scan).reshape(1, 3, 128, 128, 128)
            labels = np.argmax(np.load(path_mask), axis=0)

            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
            labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

            pred_mask = predict_mask(model=model, features_tensor=features_tensor)

            dice_scores = calculate_dice(pred_mask, labels_tensor, num_classes=num_classes)
            iou_scores = calculate_iou(pred_mask, labels_tensor, num_classes=num_classes)

            for cls in range(num_classes):
                cumulative_dice[cls] += dice_scores[cls]
                cumulative_iou[cls] += iou_scores[cls]

    mean_dice = [cumulative_dice[cls] / n_samples for cls in range(num_classes)]
    mean_iou = [cumulative_iou[cls] / n_samples for cls in range(num_classes)]

    print(title)
    print(f"Running on: {device}")
    print("\nPer-class Mean Dice Scores:")
    for cls, score in enumerate(mean_dice):
        print(f"  Class {cls}: {score:.4f}")

    print("\nPer-class Mean IoU Scores:")
    for cls, score in enumerate(mean_iou):
        print(f"  Class {cls}: {score:.4f}")

    print(f"\nOverall Mean Dice: {np.mean(mean_dice):.4f}")
    print(f"Overall Mean IoU:  {np.mean(mean_iou):.4f}")
