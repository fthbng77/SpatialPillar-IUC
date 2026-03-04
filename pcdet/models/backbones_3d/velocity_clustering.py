"""
Spatio-Velocity Clustering for Radar Point Clouds

Clusters radar pillars using 4D features: spatial position (x, y) and
decomposed velocity (vx, vy). Unlike MAFF-Net which uses raw v_r_comp
in 3D space (x, y, v_r_comp), we use decomposed velocity components
to capture directional motion information.

The decomposed velocity removes angular dependency:
  vx = v_r_comp * cos(phi),  vy = v_r_comp * sin(phi)
where phi = atan2(y, x). This ensures points moving in the same direction
cluster together regardless of their angular position relative to the sensor.
"""

import torch
import numpy as np
from sklearn.cluster import DBSCAN


def spatio_velocity_clustering(positions_xy, velocities_vxvy, eps=0.4, min_samples=5,
                               spatial_weight=1.0, velocity_weight=2.0):
    """Cluster pillars in 4D (x, y, vx, vy) space using DBSCAN.

    Spatial and velocity dimensions are weighted separately to account
    for their different scales and importance.

    Args:
        positions_xy: (N, 2) numpy array of pillar center positions
        velocities_vxvy: (N, 2) numpy array of decomposed velocities (vx, vy)
        eps: DBSCAN neighborhood radius
        min_samples: minimum points to form a cluster
        spatial_weight: weight for (x, y) dimensions
        velocity_weight: weight for (vx, vy) dimensions

    Returns:
        labels: (N,) cluster labels (no -1, noise assigned to nearest)
    """
    N = len(positions_xy)
    if N == 0:
        return np.array([], dtype=np.int32)

    # Build weighted 4D feature vector
    features = np.concatenate([
        positions_xy * spatial_weight,
        velocities_vxvy * velocity_weight
    ], axis=1)  # (N, 4)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(features)

    # Assign noise points to nearest valid cluster
    noise_mask = labels == -1
    if noise_mask.any() and (~noise_mask).any():
        valid_features = features[~noise_mask]
        valid_labels = labels[~noise_mask]

        noise_features = features[noise_mask]
        # Vectorized nearest neighbor assignment
        dists = np.linalg.norm(noise_features[:, None] - valid_features[None, :], axis=2)
        nearest_indices = np.argmin(dists, axis=1)
        labels[noise_mask] = valid_labels[nearest_indices]
    elif noise_mask.all():
        labels[:] = 0

    return labels


def compute_cluster_features(pillar_features, cluster_labels, num_clusters):
    """Compute mean feature vector for each cluster using scatter operations.

    Args:
        pillar_features: (N, C) tensor of pillar features
        cluster_labels: (N,) tensor of cluster indices
        num_clusters: total number of clusters

    Returns:
        cluster_features: (num_clusters, C) mean features per cluster
        cluster_counts: (num_clusters,) number of points per cluster
    """
    C = pillar_features.shape[1]
    device = pillar_features.device

    cluster_counts = torch.zeros(num_clusters, device=device)
    cluster_counts.scatter_add_(0, cluster_labels, torch.ones(cluster_labels.shape[0], device=device))

    cluster_sum = torch.zeros(num_clusters, C, device=device)
    cluster_sum.scatter_add_(0, cluster_labels.unsqueeze(1).expand(-1, C), pillar_features)

    safe_counts = cluster_counts.clamp(min=1).unsqueeze(1)
    cluster_features = cluster_sum / safe_counts

    return cluster_features, cluster_counts


def compute_cluster_statistics(positions_xy, velocities_vxvy, cluster_labels, num_clusters):
    """Compute per-cluster statistics for enriched cluster embedding.

    For each cluster computes: [count, vx_mean, vy_mean, vx_std, vy_std, cx, cy]

    Args:
        positions_xy: (N, 2) tensor of pillar positions
        velocities_vxvy: (N, 2) tensor of decomposed velocities
        cluster_labels: (N,) tensor of cluster indices
        num_clusters: total number of clusters

    Returns:
        stats: (num_clusters, 7) cluster statistics tensor
    """
    device = positions_xy.device

    counts = torch.zeros(num_clusters, device=device)
    counts.scatter_add_(0, cluster_labels, torch.ones(cluster_labels.shape[0], device=device))
    safe_counts = counts.clamp(min=1)

    # Mean positions (centroid)
    pos_sum = torch.zeros(num_clusters, 2, device=device)
    pos_sum.scatter_add_(0, cluster_labels.unsqueeze(1).expand(-1, 2), positions_xy)
    pos_mean = pos_sum / safe_counts.unsqueeze(1)

    # Mean velocities
    vel_sum = torch.zeros(num_clusters, 2, device=device)
    vel_sum.scatter_add_(0, cluster_labels.unsqueeze(1).expand(-1, 2), velocities_vxvy)
    vel_mean = vel_sum / safe_counts.unsqueeze(1)

    # Velocity std (variance → sqrt)
    vel_diff = velocities_vxvy - vel_mean[cluster_labels]  # (N, 2)
    vel_sq = vel_diff ** 2
    vel_var_sum = torch.zeros(num_clusters, 2, device=device)
    vel_var_sum.scatter_add_(0, cluster_labels.unsqueeze(1).expand(-1, 2), vel_sq)
    vel_std = (vel_var_sum / safe_counts.unsqueeze(1)).sqrt()

    # Normalize count to [0, 1] range
    count_norm = counts / counts.max().clamp(min=1)

    # Stack: [count_norm, vx_mean, vy_mean, vx_std, vy_std, cx, cy]
    stats = torch.stack([
        count_norm,
        vel_mean[:, 0], vel_mean[:, 1],
        vel_std[:, 0], vel_std[:, 1],
        pos_mean[:, 0], pos_mean[:, 1]
    ], dim=1)  # (num_clusters, 7)

    return stats
