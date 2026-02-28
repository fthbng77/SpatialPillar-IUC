import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianFocalLoss(nn.Module):
    """Gaussian Focal Loss for heatmap training (CenterNet/CenterPoint style).

    Modified focal loss where positive targets are continuous Gaussian values
    instead of hard 0/1 labels.

    Reference: Law & Deng, "CornerNet: Detecting Objects as Paired Keypoints", ECCV 2018
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) predicted heatmaps (raw logits)
            target: (B, C, H, W) ground truth Gaussian heatmaps [0, 1]

        Returns:
            loss: scalar
        """
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)

        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, self.alpha) * \
                   torch.pow(1 - target, self.beta) * neg_mask

        num_pos = pos_mask.sum().clamp(min=1.0)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


class RegLoss(nn.Module):
    """Regression loss for center offset, dimensions, height, rotation."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred: (B, C, H, W)
            target: (B, max_objs, C)
            mask: (B, max_objs) — 1 for valid objects
            ind: (B, max_objs) — flattened indices

        Returns:
            loss: scalar
        """
        # pred is already gathered: (B, max_objs, C)
        mask = mask.unsqueeze(2).expand_as(target).float()
        num_pos = mask.sum().clamp(min=1.0)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / num_pos
        return loss


def gaussian_radius(det_size, min_overlap=0.5):
    """Compute minimum Gaussian radius for a given bounding box size.

    Args:
        det_size: (height, width) of the bounding box on the feature map
        min_overlap: minimum IoU overlap

    Returns:
        radius: int
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_gaussian(heatmap, center, radius, k=1):
    """Draw a 2D Gaussian on the heatmap.

    Args:
        heatmap: (H, W) tensor
        center: (x, y) center position
        radius: Gaussian radius
        k: peak value
    """
    diameter = 2 * radius + 1
    gaussian = _gaussian2d(diameter, sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _gaussian2d(diameter, sigma):
    """Generate a 2D Gaussian kernel."""
    m = np.arange(0, diameter, 1, dtype=np.float32)
    m = m[:, np.newaxis]
    n = np.arange(0, diameter, 1, dtype=np.float32)
    n = n[np.newaxis, :]
    center = diameter // 2
    gaussian = np.exp(-((m - center) ** 2 + (n - center) ** 2) / (2 * sigma ** 2))
    return torch.from_numpy(gaussian)
