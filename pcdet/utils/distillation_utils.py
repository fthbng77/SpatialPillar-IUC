"""
Knowledge Distillation Utilities

Implements feature mimicry and response distillation losses for
transferring knowledge from a LiDAR teacher to a radar student model.

Reference: SCKD — Spatially Consistent Knowledge Distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMimicryLoss(nn.Module):
    """BEV Feature Alignment Loss between teacher and student.

    Aligns the BEV feature maps from student (radar) and teacher (LiDAR)
    using L2 loss with optional channel adaptation layer.

    Args:
        student_channels: number of student BEV feature channels
        teacher_channels: number of teacher BEV feature channels
    """

    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        # Channel adaptation if dimensions differ
        if student_channels != teacher_channels:
            self.adapt = nn.Conv2d(student_channels, teacher_channels,
                                   kernel_size=1, bias=False)
        else:
            self.adapt = nn.Identity()

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: (B, C_s, H, W) student BEV features
            teacher_feat: (B, C_t, H, W) teacher BEV features (detached)

        Returns:
            loss: scalar MSE loss
        """
        adapted_feat = self.adapt(student_feat)

        # Handle spatial size mismatch
        if adapted_feat.shape[2:] != teacher_feat.shape[2:]:
            adapted_feat = F.interpolate(
                adapted_feat,
                size=teacher_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        loss = F.mse_loss(adapted_feat, teacher_feat.detach())
        return loss


class ResponseDistillationLoss(nn.Module):
    """Response-level distillation using KL divergence.

    Matches the class prediction distributions between teacher and student.

    Args:
        temperature: softmax temperature for distillation
    """

    def __init__(self, temperature=3.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: (B, N, C) student classification logits
            teacher_logits: (B, N, C) teacher classification logits (detached)

        Returns:
            loss: scalar KL divergence loss
        """
        T = self.temperature

        student_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits.detach() / T, dim=-1)

        loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (T ** 2)
        return loss
