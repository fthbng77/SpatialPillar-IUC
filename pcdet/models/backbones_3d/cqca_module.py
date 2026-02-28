"""
CQCA: Cluster Query Cross-Attention Module

Implements velocity-based clustering followed by cross-attention between
pillar features and cluster query features. This captures global context
by allowing each pillar to attend to velocity cluster representations.

Reference: MAFF-Net — Cluster Query Cross-Attention is a key component
that enables information exchange between points grouped by velocity.

Pipeline:
1. Extract velocity from voxel features
2. Cluster by velocity (DBSCAN)
3. Compute cluster mean features (queries)
4. Cross-attention: pillar features (Q) x cluster queries (K, V)
5. Residual connection + layer norm
"""

import torch
import torch.nn as nn
import numpy as np

from .velocity_clustering import velocity_clustering_dbscan, compute_cluster_features


class CQCAModule(nn.Module):
    """Cluster Query Cross-Attention module.

    Sits in the backbone_3d pipeline after PillarAttention and before
    map_to_bev. Operates on pillar_features and voxel_coords.

    Args:
        model_cfg: config with:
            - ATTN_CHANNELS: attention embedding dimension
            - NUM_HEADS: number of attention heads
            - DROPOUT: attention dropout
            - VELOCITY_EPS: DBSCAN eps for velocity clustering (m/s)
            - VELOCITY_INDEX: index of v_r_comp in raw point features
        input_channels: number of input pillar feature channels
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = input_channels  # output dim = input dim

        attn_channels = model_cfg.get('ATTN_CHANNELS', input_channels)
        num_heads = model_cfg.get('NUM_HEADS', 2)
        dropout = model_cfg.get('DROPOUT', 0.0)
        self.velocity_eps = model_cfg.get('VELOCITY_EPS', 0.5)
        self.velocity_index = model_cfg.get('VELOCITY_INDEX', 5)  # v_r_comp column
        self.max_clusters = model_cfg.get('MAX_CLUSTERS', 32)

        # Project pillar features to attention dim if needed
        self.proj_q = nn.Linear(input_channels, attn_channels) if input_channels != attn_channels else nn.Identity()
        self.proj_kv = nn.Linear(input_channels, attn_channels) if input_channels != attn_channels else nn.Identity()

        # Cross-attention: Q = pillar features, K/V = cluster features
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

        # Project back if needed
        self.proj_out = nn.Linear(attn_channels, input_channels) if input_channels != attn_channels else nn.Identity()

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                pillar_features: (N_total, C) pillar features
                voxel_coords: (N_total, 4) [batch_idx, z, y, x]
                voxels: (N_total, max_points, F) raw voxel point features

        Returns:
            batch_dict with updated pillar_features
        """
        pillar_features = batch_dict['pillar_features']  # (N, C)
        coords = batch_dict['voxel_coords']              # (N, 4)

        batch_size = coords[:, 0].max().int().item() + 1
        device = pillar_features.device

        # Get velocity information from raw voxels
        voxels = batch_dict.get('voxels', None)
        voxel_num_points = batch_dict.get('voxel_num_points', None)

        updated_features = []

        for b in range(batch_size):
            batch_mask = coords[:, 0] == b
            cur_features = pillar_features[batch_mask]  # (N_b, C)
            num_pillars = cur_features.shape[0]

            if num_pillars == 0:
                updated_features.append(cur_features)
                continue

            # Extract velocity for clustering
            if voxels is not None and voxel_num_points is not None:
                cur_voxels = voxels[batch_mask]           # (N_b, max_pts, F)
                cur_num_pts = voxel_num_points[batch_mask]  # (N_b,)

                # Mean velocity per pillar
                velocities = []
                for i in range(num_pillars):
                    n_pts = int(cur_num_pts[i].item())
                    if n_pts > 0 and cur_voxels.shape[2] > self.velocity_index:
                        vel = cur_voxels[i, :n_pts, self.velocity_index].mean().item()
                    else:
                        vel = 0.0
                    velocities.append(vel)
                velocities = np.array(velocities, dtype=np.float32)
            else:
                # Fallback: no velocity info, use single cluster
                velocities = np.zeros(num_pillars, dtype=np.float32)

            # Cluster by velocity
            cluster_labels = velocity_clustering_dbscan(
                velocities, eps=self.velocity_eps, min_samples=2
            )
            num_clusters = cluster_labels.max() + 1 if len(cluster_labels) > 0 else 1
            num_clusters = min(num_clusters, self.max_clusters)

            # Remap labels if too many clusters
            if cluster_labels.max() >= self.max_clusters:
                unique_labels = np.unique(cluster_labels)
                label_map = {old: new % self.max_clusters for new, old in enumerate(unique_labels)}
                cluster_labels = np.array([label_map[l] for l in cluster_labels])
                num_clusters = min(len(unique_labels), self.max_clusters)

            cluster_labels_t = torch.from_numpy(cluster_labels).long().to(device)

            # Compute cluster query features
            cluster_features, cluster_counts = compute_cluster_features(
                cur_features.detach(), cluster_labels_t, num_clusters
            )  # (num_clusters, C)

            # Cross-attention: pillars attend to clusters
            q = self.proj_q(cur_features).unsqueeze(0)    # (1, N_b, D)
            kv = self.proj_kv(cluster_features).unsqueeze(0)  # (1, K, D)

            attn_out, _ = self.cross_attn(q, kv, kv)      # (1, N_b, D)
            q_residual = self.norm1(q + attn_out)

            ffn_out = self.ffn(q_residual)
            out = self.norm2(q_residual + ffn_out)         # (1, N_b, D)

            out = self.proj_out(out.squeeze(0))            # (N_b, C)

            # Residual connection with original features
            out = out + cur_features

            updated_features.append(out)

        batch_dict['pillar_features'] = torch.cat(updated_features, dim=0)
        return batch_dict
