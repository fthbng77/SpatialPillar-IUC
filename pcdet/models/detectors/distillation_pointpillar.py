"""
Distillation PointPillar: Teacher-Student Knowledge Distillation

Wraps a radar PointPillar (student) and a pretrained LiDAR PointPillar (teacher)
for knowledge distillation training. The teacher is frozen and provides
supervision signals via BEV feature alignment and response distillation.

Reference: SCKD — Spatially Consistent Knowledge Distillation
"""

import torch
import torch.nn as nn

from .detector3d_template import Detector3DTemplate
from .pointpillar import PointPillar
from ...utils.distillation_utils import FeatureMimicryLoss, ResponseDistillationLoss


class DistillationPointPillar(Detector3DTemplate):
    """Teacher-Student distillation wrapper for PointPillar.

    The student model is the radar PointPillar being trained.
    The teacher model is a pretrained LiDAR PointPillar (frozen).

    Args:
        model_cfg: config with:
            - STUDENT: student model config
            - TEACHER: teacher model config
            - TEACHER_CKPT: path to pretrained teacher checkpoint
            - DISTILL_ALPHA: weight for feature mimicry loss
            - DISTILL_BETA: weight for response distillation loss
            - DISTILL_TEMPERATURE: KL divergence temperature
        num_class: number of classes
        dataset: dataset object
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        # Build student (radar model) — uses the regular module_topology
        self.module_list = self.build_networks()

        # Distillation config
        self.distill_alpha = model_cfg.get('DISTILL_ALPHA', 1.0)
        self.distill_beta = model_cfg.get('DISTILL_BETA', 0.5)
        temperature = model_cfg.get('DISTILL_TEMPERATURE', 3.0)

        # Build teacher model if config is provided
        self.teacher = None
        self.feature_loss = None
        self.response_loss = None

        if model_cfg.get('TEACHER_CKPT', None) is not None:
            # Losses
            student_channels = self.backbone_2d.num_bev_features if self.backbone_2d else 96
            teacher_channels = model_cfg.get('TEACHER_BEV_CHANNELS', student_channels)

            self.feature_loss = FeatureMimicryLoss(student_channels, teacher_channels)
            self.response_loss = ResponseDistillationLoss(temperature=temperature)

    def load_teacher(self, teacher_model):
        """Load and freeze a pretrained teacher model.

        Args:
            teacher_model: nn.Module — pretrained LiDAR PointPillar
        """
        self.teacher = teacher_model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, batch_dict):
        # Forward through student
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            # Distillation losses (only if teacher is loaded)
            if self.teacher is not None:
                distill_loss, distill_tb = self._compute_distillation_loss(batch_dict)
                loss = loss + distill_loss
                tb_dict.update(distill_tb)

            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def _compute_distillation_loss(self, student_batch_dict):
        """Compute distillation losses from teacher.

        Requires teacher's batch_dict to be computed separately
        with LiDAR data (teacher_points in the batch).
        """
        tb_dict = {}
        total_distill_loss = torch.tensor(0.0, device=student_batch_dict['spatial_features_2d'].device)

        # Feature mimicry loss on BEV features
        if self.feature_loss is not None and 'teacher_spatial_features_2d' in student_batch_dict:
            feat_loss = self.feature_loss(
                student_batch_dict['spatial_features_2d'],
                student_batch_dict['teacher_spatial_features_2d']
            )
            total_distill_loss = total_distill_loss + self.distill_alpha * feat_loss
            tb_dict['distill_feat_loss'] = feat_loss.item()

        # Response distillation loss
        if self.response_loss is not None and 'teacher_cls_preds' in student_batch_dict:
            resp_loss = self.response_loss(
                student_batch_dict['batch_cls_preds'],
                student_batch_dict['teacher_cls_preds']
            )
            total_distill_loss = total_distill_loss + self.distill_beta * resp_loss
            tb_dict['distill_resp_loss'] = resp_loss.item()

        tb_dict['distill_total_loss'] = total_distill_loss.item()
        return total_distill_loss, tb_dict

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }
        loss = loss_rpn
        return loss, tb_dict, disp_dict
