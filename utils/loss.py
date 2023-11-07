import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def iou_loss(pred, target, eps=1e-6):
    if np.max(target) == 0.0:
        return iou_loss(1-pred, 1-target)  ## empty image; calc IoU of zeros
    intersection = torch.sum(pred * target, axis=[1, 2, 3])
    union = torch.sum(target, axis=[1, 2, 3]) + torch.sum(pred, axis=[1, 2, 3]) - intersection
    return -torch.mean((intersection + eps) / (union + eps), axis=0)


class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1e-15

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)

