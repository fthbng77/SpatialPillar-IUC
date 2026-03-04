"""
CQCA: Cluster Query Cross-Attention Module (SpatialPillar-IUC)

Spatio-velocity clustering followed by enriched cross-attention between
pillar features and cluster query embeddings.

Key differences from MAFF-Net's CQCA:
1. 4D clustering in (x, y, vx, vy) space using decomposed velocity,
   rather than 3D (x, y, v_r_comp). Decomposed velocity captures
   directional motion and removes angular dependency.
2. Pillar-level fusion (before BEV scatter) instead of BEV-level fusion.
   This provides earlier and more direct information flow.
3. Cluster statistics embedding: each cluster is represented by its
   mean features enriched with learned projections of statistical
   properties (count, velocity mean/std, spatial centroid).

Pipeline:
1. Extract per-pillar mean position (x, y) and velocity (v_r_comp)
2. Decompose velocity: vx = v_r_comp * cos(phi), vy = v_r_comp * sin(phi)
3. DBSCAN clustering in weighted (x, y, vx, vy) space
4. Compute cluster embeddings = mean_features + projected_statistics
5. Cross-attention: pillar features (Q) attend to cluster embeddings (K, V)
6. FFN + residual connections
"""

import torch
import torch.nn as nn
import numpy as np

from .velocity_clustering import (
    spatio_velocity_clustering,
    compute_cluster_features,
    compute_cluster_statistics,
)


class CQCAModule(nn.Module):
    """Cluster Query Cross-Attention with spatio-velocity clustering.

    Args:
        model_cfg: config dict with:
            ATTN_CHANNELS, NUM_HEADS, DROPOUT, DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
            VELOCITY_INDEX, MAX_CLUSTERS, CLUSTER_STATS_DIM,
            SPATIAL_WEIGHT, VELOCITY_WEIGHT
        input_channels: pillar feature dimension from PillarAttention
    """

    STATS_DIM = 7  # count, vx_mean, vy_mean, vx_std, vy_std, cx, cy

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = input_channels

        attn_channels = model_cfg.get('ATTN_CHANNELS', input_channels)
        num_heads = model_cfg.get('NUM_HEADS', 4)
        dropout = model_cfg.get('DROPOUT', 0.0)

        # Clustering parameters
        self.dbscan_eps = model_cfg.get('DBSCAN_EPS', 0.4)
        self.dbscan_min_samples = model_cfg.get('DBSCAN_MIN_SAMPLES', 5)
        self.velocity_index = model_cfg.get('VELOCITY_INDEX', 5)
        self.max_clusters = model_cfg.get('MAX_CLUSTERS', 32)
        self.spatial_weight = model_cfg.get('SPATIAL_WEIGHT', 1.0)
        self.velocity_weight = model_cfg.get('VELOCITY_WEIGHT', 2.0)

        # Cluster statistics projection
        self.stats_proj = nn.Sequential(
            nn.Linear(self.STATS_DIM, attn_channels),
            nn.GELU(),
            nn.Linear(attn_channels, attn_channels),
        )

        # Feature projections
        self.proj_q = nn.Linear(input_channels, attn_channels)
        self.proj_kv = nn.Linear(input_channels, attn_channels)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(attn_channels)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(attn_channels, attn_channels * 2),
            nn.GELU(),
            nn.Linear(attn_channels * 2, attn_channels),
        )
        self.norm2 = nn.LayerNorm(attn_channels)

        # Project back to input dim
        self.proj_out = nn.Linear(attn_channels, input_channels)

        # Learnable residual gate
        self.gate = nn.Parameter(torch.zeros(1))

    def _extract_pillar_xyv(self, voxels, voxel_num_points, device):
        """Extract per-pillar mean (x, y, vx, vy) from raw voxel data.

        Velocity decomposition: v_r_comp → (vx, vy) using point angle.
        This removes angular dependency and captures directional motion.

        Args:
            voxels: (N_b, max_pts, F) raw point features
            voxel_num_points: (N_b,) valid point counts
            device: torch device

        Returns:
            positions_xy: (N_b, 2) mean pillar positions
            velocities_vxvy: (N_b, 2) decomposed velocity vectors
        """
        max_pts = voxels.shape[1]
        pt_range = torch.arange(max_pts, device=device).unsqueeze(0)
        valid_mask = pt_range < voxel_num_points.unsqueeze(1)  # (N_b, max_pts)
        safe_counts = voxel_num_points.clamp(min=1).float().unsqueeze(1)  # (N_b, 1)

        # Mean positions
        x_vals = (voxels[:, :, 0] * valid_mask.float()).sum(dim=1, keepdim=True) / safe_counts
        y_vals = (voxels[:, :, 1] * valid_mask.float()).sum(dim=1, keepdim=True) / safe_counts
        positions_xy = torch.cat([x_vals, y_vals], dim=1)  # (N_b, 2)

        # Velocity decomposition per point, then mean
        if voxels.shape[2] > self.velocity_index:
            point_x = voxels[:, :, 0]  # (N_b, max_pts)
            point_y = voxels[:, :, 1]
            phi = torch.atan2(point_y, point_x + 1e-6)  # (N_b, max_pts)

            v_r_comp = voxels[:, :, self.velocity_index]  # (N_b, max_pts)
            vx = v_r_comp * torch.cos(phi)
            vy = v_r_comp * torch.sin(phi)

            vx_mean = (vx * valid_mask.float()).sum(dim=1, keepdim=True) / safe_counts
            vy_mean = (vy * valid_mask.float()).sum(dim=1, keepdim=True) / safe_counts
            velocities_vxvy = torch.cat([vx_mean, vy_mean], dim=1)  # (N_b, 2)
        else:
            velocities_vxvy = torch.zeros_like(positions_xy)

        return positions_xy, velocities_vxvy

    def forward(self, batch_dict):
        """
        Args:
            batch_dict with: pillar_features, voxel_coords, voxels, voxel_num_points

        Returns:
            batch_dict with updated pillar_features
        """
        pillar_features = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']
        batch_size = coords[:, 0].max().int().item() + 1
        device = pillar_features.device

        voxels = batch_dict.get('voxels', None)
        voxel_num_points = batch_dict.get('voxel_num_points', None)

        updated_features = []

        for b in range(batch_size):
            batch_mask = coords[:, 0] == b
            cur_features = pillar_features[batch_mask]
            num_pillars = cur_features.shape[0]

            if num_pillars == 0:
                updated_features.append(cur_features)
                continue

            # --- Step 1: Extract spatial + velocity info ---
            if voxels is not None and voxel_num_points is not None:
                cur_voxels = voxels[batch_mask]
                cur_num_pts = voxel_num_points[batch_mask]
                positions_xy, velocities_vxvy = self._extract_pillar_xyv(
                    cur_voxels, cur_num_pts, device
                )
            else:
                positions_xy = torch.zeros(num_pillars, 2, device=device)
                velocities_vxvy = torch.zeros(num_pillars, 2, device=device)

            # --- Step 2: 4D spatio-velocity DBSCAN ---
            pos_np = positions_xy.detach().cpu().numpy()
            vel_np = velocities_vxvy.detach().cpu().numpy()

            cluster_labels = spatio_velocity_clustering(
                pos_np, vel_np,
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                spatial_weight=self.spatial_weight,
                velocity_weight=self.velocity_weight
            )

            num_clusters = int(cluster_labels.max()) + 1 if len(cluster_labels) > 0 else 1
            num_clusters = min(num_clusters, self.max_clusters)

            # Remap if exceeding max
            if cluster_labels.max() >= self.max_clusters:
                unique_labels = np.unique(cluster_labels)
                label_map = {old: new % self.max_clusters for new, old in enumerate(unique_labels)}
                cluster_labels = np.array([label_map[l] for l in cluster_labels])
                num_clusters = min(len(unique_labels), self.max_clusters)

            cluster_labels_t = torch.from_numpy(cluster_labels).long().to(device)

            # --- Step 3: Cluster embeddings = mean features + statistics ---
            cluster_mean_feats, cluster_counts = compute_cluster_features(
                cur_features, cluster_labels_t, num_clusters
            )  # (K, C)

            cluster_stats = compute_cluster_statistics(
                positions_xy, velocities_vxvy, cluster_labels_t, num_clusters
            )  # (K, 7)

            stats_embed = self.stats_proj(cluster_stats)  # (K, D)

            # --- Step 4: Cross-attention ---
            q = self.proj_q(cur_features).unsqueeze(0)                    # (1, N_b, D)
            kv = (self.proj_kv(cluster_mean_feats) + stats_embed).unsqueeze(0)  # (1, K, D)

            attn_out, _ = self.cross_attn(q, kv, kv)
            x = self.norm1(q + attn_out)

            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

            out = self.proj_out(x.squeeze(0))  # (N_b, C)

            # Gated residual: model learns how much CQCA contributes
            out = cur_features + self.gate.sigmoid() * out

            updated_features.append(out)

        batch_dict['pillar_features'] = torch.cat(updated_features, dim=0)
        return batch_dict
