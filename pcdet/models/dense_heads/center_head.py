import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .center_head_utils import GaussianFocalLoss, RegLoss, gaussian_radius, draw_gaussian


class SeparateHead(nn.Module):
    """Separate prediction head for each attribute (heatmap, offset, dim, etc.)."""

    def __init__(self, input_channels, head_channels, num_conv, num_output, use_bias=True):
        super().__init__()
        layers = []
        for k in range(num_conv - 1):
            in_ch = input_channels if k == 0 else head_channels
            layers.extend([
                nn.Conv2d(in_ch, head_channels, kernel_size=3, padding=1, bias=use_bias),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(head_channels, num_output, kernel_size=3, padding=1, bias=True))
        self.head = nn.Sequential(*layers)

        # Initialize final conv
        self.head[-1].bias.data.fill_(0)

    def forward(self, x):
        return self.head(x)


class CenterHead(nn.Module):
    """CenterPoint-style anchor-free detection head for BEV feature maps.

    Predicts heatmap + regression targets (center offset, height, dimensions, rotation).
    Uses Gaussian focal loss for heatmap and L1 loss for regression.

    Interface matches AnchorHeadSingle for compatibility with Detector3DTemplate.

    Args:
        model_cfg: config node
        input_channels: number of input BEV feature channels
        num_class: number of object classes
        class_names: list of class name strings
        grid_size: (nx, ny, nz) voxel grid dimensions
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        predict_boxes_when_training: whether to decode boxes during training
    """

    def __init__(self, model_cfg, input_channels, num_class, class_names,
                 grid_size, point_cloud_range, predict_boxes_when_training=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.point_cloud_range = np.array(point_cloud_range)
        self.grid_size = grid_size

        self.feature_map_stride = model_cfg.get('FEATURE_MAP_STRIDE', 2)
        head_channels = model_cfg.get('HEAD_CHANNELS', 64)
        num_conv = model_cfg.get('NUM_CONV', 2)

        # Class-grouped or all-together
        self.class_names_each_head = []
        if model_cfg.get('SEPARATE_HEAD_CFG', None) is not None:
            for cur_class_names in model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER:
                self.class_names_each_head.append([cur_class_names])
        else:
            self.class_names_each_head = [[name] for name in class_names]

        self.num_heads = len(self.class_names_each_head)

        # Build separate heads for each class group
        self.heads_list = nn.ModuleList()
        for idx in range(self.num_heads):
            num_cls = len(self.class_names_each_head[idx])
            head_dict = nn.ModuleDict({
                'hm': SeparateHead(input_channels, head_channels, num_conv, num_cls),
                'center': SeparateHead(input_channels, head_channels, num_conv, 2),
                'center_z': SeparateHead(input_channels, head_channels, num_conv, 1),
                'dim': SeparateHead(input_channels, head_channels, num_conv, 3),
                'rot': SeparateHead(input_channels, head_channels, num_conv, 2),
            })
            self.heads_list.append(head_dict)

        # Initialize heatmap bias for focal loss
        for head in self.heads_list:
            head['hm'].head[-1].bias.data.fill_(-2.19)  # -log((1-0.1)/0.1)

        # Losses
        self.hm_loss_func = GaussianFocalLoss(
            alpha=model_cfg.LOSS_CONFIG.get('ALPHA', 2.0),
            beta=model_cfg.LOSS_CONFIG.get('BETA', 4.0)
        )
        self.reg_loss_func = RegLoss()

        self.loss_weights = model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        self.forward_ret_dict = {}

        # Max objects per frame
        self.max_objs = model_cfg.get('MAX_OBJS', 100)

        # Voxel size for coordinate mapping
        voxel_size = model_cfg.get('VOXEL_SIZE', [0.16, 0.16, 5.0])
        self.voxel_size = np.array(voxel_size)

        # Feature map size (after stride)
        self.feature_map_size = grid_size[:2] // self.feature_map_stride

        # Post-processing
        self.score_thresh = model_cfg.get('SCORE_THRESH', 0.1)
        self.nms_kernel_size = model_cfg.get('NMS_KERNEL_SIZE', 3)

    def assign_targets(self, gt_boxes):
        """Generate heatmap and regression targets from ground truth boxes.

        Args:
            gt_boxes: (B, M, 8) — [x, y, z, dx, dy, dz, heading, class_id]

        Returns:
            target_dict with heatmaps, regression targets, masks, indices
        """
        batch_size = gt_boxes.shape[0]
        feat_h, feat_w = int(self.feature_map_size[1]), int(self.feature_map_size[0])

        heatmaps = []
        target_centers = []
        target_center_zs = []
        target_dims = []
        target_rots = []
        target_masks = []
        target_inds = []

        for head_idx in range(self.num_heads):
            cur_class_names = self.class_names_each_head[head_idx]
            num_cls = len(cur_class_names)

            heatmap = torch.zeros((batch_size, num_cls, feat_h, feat_w), dtype=torch.float32)
            center = torch.zeros((batch_size, self.max_objs, 2), dtype=torch.float32)
            center_z = torch.zeros((batch_size, self.max_objs, 1), dtype=torch.float32)
            dim = torch.zeros((batch_size, self.max_objs, 3), dtype=torch.float32)
            rot = torch.zeros((batch_size, self.max_objs, 2), dtype=torch.float32)
            mask = torch.zeros((batch_size, self.max_objs), dtype=torch.float32)
            ind = torch.zeros((batch_size, self.max_objs), dtype=torch.long)

            for b in range(batch_size):
                cur_gt = gt_boxes[b]
                # Filter out padding boxes (all zeros)
                valid_mask = cur_gt[:, -1] > 0
                cur_gt = cur_gt[valid_mask]

                for k in range(cur_gt.shape[0]):
                    cls_id = int(cur_gt[k, 7]) - 1  # 1-indexed to 0-indexed
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else None

                    # Check if this class belongs to current head
                    local_cls_id = -1
                    for c_idx, c_name in enumerate(cur_class_names):
                        if c_name == cls_name:
                            local_cls_id = c_idx
                            break
                    if local_cls_id < 0:
                        continue

                    # Convert box center to feature map coordinates
                    x = cur_gt[k, 0]
                    y = cur_gt[k, 1]
                    z = cur_gt[k, 2]
                    dx = cur_gt[k, 3]
                    dy = cur_gt[k, 4]
                    dz = cur_gt[k, 5]
                    heading = cur_gt[k, 6]

                    # Map to feature map coordinates
                    cx = (x - self.point_cloud_range[0]) / (self.voxel_size[0] * self.feature_map_stride)
                    cy = (y - self.point_cloud_range[1]) / (self.voxel_size[1] * self.feature_map_stride)

                    cx_int = int(cx.item() if isinstance(cx, torch.Tensor) else cx)
                    cy_int = int(cy.item() if isinstance(cy, torch.Tensor) else cy)

                    if cx_int < 0 or cx_int >= feat_w or cy_int < 0 or cy_int >= feat_h:
                        continue

                    # Compute Gaussian radius
                    box_w = dx / (self.voxel_size[0] * self.feature_map_stride)
                    box_h = dy / (self.voxel_size[1] * self.feature_map_stride)
                    if isinstance(box_w, torch.Tensor):
                        box_w = box_w.item()
                        box_h = box_h.item()

                    radius = max(0, int(gaussian_radius((box_h, box_w), min_overlap=0.5)))

                    # Draw Gaussian on heatmap
                    draw_gaussian(heatmap[b, local_cls_id], [cx_int, cy_int], radius)

                    # Count objects
                    obj_count = int(mask[b].sum().item())
                    if obj_count >= self.max_objs:
                        continue

                    # Store regression targets
                    cx_val = cx.item() if isinstance(cx, torch.Tensor) else cx
                    cy_val = cy.item() if isinstance(cy, torch.Tensor) else cy
                    center[b, obj_count] = torch.tensor([cx_val - cx_int, cy_val - cy_int])
                    center_z[b, obj_count] = z if isinstance(z, torch.Tensor) else torch.tensor([z])
                    dim[b, obj_count] = cur_gt[k, 3:6].log()
                    rot[b, obj_count, 0] = torch.sin(heading) if isinstance(heading, torch.Tensor) else np.sin(heading)
                    rot[b, obj_count, 1] = torch.cos(heading) if isinstance(heading, torch.Tensor) else np.cos(heading)
                    ind[b, obj_count] = cy_int * feat_w + cx_int
                    mask[b, obj_count] = 1

            heatmaps.append(heatmap.cuda())
            target_centers.append(center.cuda())
            target_center_zs.append(center_z.cuda())
            target_dims.append(dim.cuda())
            target_rots.append(rot.cuda())
            target_masks.append(mask.cuda())
            target_inds.append(ind.cuda())

        return {
            'heatmaps': heatmaps,
            'target_centers': target_centers,
            'target_center_zs': target_center_zs,
            'target_dims': target_dims,
            'target_rots': target_rots,
            'target_masks': target_masks,
            'target_inds': target_inds,
        }

    def get_loss(self):
        tb_dict = {}
        total_loss = 0

        for head_idx in range(self.num_heads):
            pred_dict = self.forward_ret_dict['pred_dicts'][head_idx]
            target_dict = self.forward_ret_dict['target_dicts']

            # Heatmap loss
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dict['heatmaps'][head_idx])
            hm_loss *= self.loss_weights.get('hm_weight', 1.0)

            # Gather predicted regression values at target locations
            target_ind = target_dict['target_inds'][head_idx]
            target_mask = target_dict['target_masks'][head_idx]
            batch_size = pred_dict['center'].shape[0]

            # Gather predictions
            pred_center = self._gather_feat(pred_dict['center'], target_ind)
            pred_center_z = self._gather_feat(pred_dict['center_z'], target_ind)
            pred_dim = self._gather_feat(pred_dict['dim'], target_ind)
            pred_rot = self._gather_feat(pred_dict['rot'], target_ind)

            # Regression losses
            center_loss = self.reg_loss_func(pred_center, target_dict['target_centers'][head_idx], target_mask)
            center_z_loss = self.reg_loss_func(pred_center_z, target_dict['target_center_zs'][head_idx], target_mask)
            dim_loss = self.reg_loss_func(pred_dim, target_dict['target_dims'][head_idx], target_mask)
            rot_loss = self.reg_loss_func(pred_rot, target_dict['target_rots'][head_idx], target_mask)

            reg_loss = center_loss + center_z_loss + dim_loss + rot_loss
            reg_loss *= self.loss_weights.get('reg_weight', 2.0)

            head_loss = hm_loss + reg_loss
            total_loss += head_loss

            tb_dict[f'center_head_{head_idx}_hm_loss'] = hm_loss.item()
            tb_dict[f'center_head_{head_idx}_reg_loss'] = reg_loss.item()

        tb_dict['rpn_loss'] = total_loss.item()
        return total_loss, tb_dict

    @staticmethod
    def _gather_feat(feat, ind):
        """Gather features at specific indices.

        Args:
            feat: (B, C, H, W) feature map
            ind: (B, K) flattened indices

        Returns:
            gathered: (B, K, C)
        """
        B, C, H, W = feat.shape
        feat = feat.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        ind = ind.unsqueeze(2).expand(-1, -1, C)  # (B, K, C)
        gathered = feat.gather(1, ind)
        return gathered

    def _nms_by_maxpool(self, heatmap):
        """Simple NMS using max pooling (no explicit NMS needed).

        Args:
            heatmap: (B, C, H, W)

        Returns:
            heatmap with non-peak values zeroed out
        """
        padding = self.nms_kernel_size // 2
        hmax = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size,
                            stride=1, padding=padding)
        keep = (hmax == heatmap).float()
        return heatmap * keep

    def decode_predictions(self, pred_dicts):
        """Decode predictions to 3D bounding boxes.

        Returns:
            batch_box_preds: (B, N, 7) [x, y, z, dx, dy, dz, heading]
            batch_cls_preds: (B, N, num_classes) class scores
        """
        all_boxes = []
        all_scores = []

        batch_size = pred_dicts[0]['hm'].shape[0]

        for b in range(batch_size):
            boxes_per_sample = []
            scores_per_sample = []

            for head_idx, pred_dict in enumerate(pred_dicts):
                hm = torch.sigmoid(pred_dict['hm'][b:b+1])  # (1, C, H, W)
                hm = self._nms_by_maxpool(hm)

                C, H, W = hm.shape[1], hm.shape[2], hm.shape[3]

                center = pred_dict['center'][b]      # (2, H, W)
                center_z = pred_dict['center_z'][b]  # (1, H, W)
                dim = pred_dict['dim'][b]             # (3, H, W)
                rot = pred_dict['rot'][b]             # (2, H, W)

                # Get all positions above threshold
                for cls_idx in range(C):
                    cls_hm = hm[0, cls_idx]  # (H, W)
                    score_mask = cls_hm > self.score_thresh

                    if score_mask.sum() == 0:
                        continue

                    scores = cls_hm[score_mask]
                    ys, xs = torch.where(score_mask)

                    # Decode center
                    cx = xs.float() + center[0][score_mask]
                    cy = ys.float() + center[1][score_mask]

                    # Map back to world coordinates
                    x = cx * self.voxel_size[0] * self.feature_map_stride + self.point_cloud_range[0]
                    y = cy * self.voxel_size[1] * self.feature_map_stride + self.point_cloud_range[1]
                    z = center_z[0][score_mask]

                    dx = dim[0][score_mask].exp()
                    dy = dim[1][score_mask].exp()
                    dz = dim[2][score_mask].exp()

                    sin_h = rot[0][score_mask]
                    cos_h = rot[1][score_mask]
                    heading = torch.atan2(sin_h, cos_h)

                    boxes = torch.stack([x, y, z, dx, dy, dz, heading], dim=-1)  # (K, 7)

                    # Map local class index to global class index
                    cur_class_names = self.class_names_each_head[head_idx]
                    cls_name = cur_class_names[cls_idx]
                    global_cls_idx = self.class_names.index(cls_name)

                    # Create one-hot class scores
                    cls_scores = torch.zeros((boxes.shape[0], self.num_class),
                                           device=boxes.device, dtype=boxes.dtype)
                    cls_scores[:, global_cls_idx] = scores

                    boxes_per_sample.append(boxes)
                    scores_per_sample.append(cls_scores)

            if len(boxes_per_sample) > 0:
                boxes_per_sample = torch.cat(boxes_per_sample, dim=0)
                scores_per_sample = torch.cat(scores_per_sample, dim=0)
            else:
                boxes_per_sample = torch.zeros((0, 7), device=hm.device)
                scores_per_sample = torch.zeros((0, self.num_class), device=hm.device)

            all_boxes.append(boxes_per_sample)
            all_scores.append(scores_per_sample)

        # Pad to same size
        max_boxes = max(b.shape[0] for b in all_boxes)
        max_boxes = max(max_boxes, 1)

        batch_box_preds = torch.zeros((batch_size, max_boxes, 7), device=all_boxes[0].device)
        batch_cls_preds = torch.zeros((batch_size, max_boxes, self.num_class), device=all_boxes[0].device)

        for b in range(batch_size):
            n = all_boxes[b].shape[0]
            if n > 0:
                batch_box_preds[b, :n] = all_boxes[b]
                batch_cls_preds[b, :n] = all_scores[b]

        return batch_cls_preds, batch_box_preds

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        pred_dicts = []
        for head in self.heads_list:
            pred_dict = {
                'hm': head['hm'](spatial_features_2d),
                'center': head['center'](spatial_features_2d),
                'center_z': head['center_z'](spatial_features_2d),
                'dim': head['dim'](spatial_features_2d),
                'rot': head['rot'](spatial_features_2d),
            }
            pred_dicts.append(pred_dict)

        if self.training:
            target_dicts = self.assign_targets(gt_boxes=data_dict['gt_boxes'])
            self.forward_ret_dict['pred_dicts'] = pred_dicts
            self.forward_ret_dict['target_dicts'] = target_dicts

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.decode_predictions(pred_dicts)
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = True  # already sigmoided

        return data_dict
