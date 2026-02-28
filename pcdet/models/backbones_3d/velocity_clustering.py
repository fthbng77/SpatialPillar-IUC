"""
Velocity-based Point Cloud Clustering

Clusters radar points based on radial velocity (v_r_comp) using DBSCAN.
Points with similar velocities are likely to belong to the same moving object
or static background.

Reference: MAFF-Net — velocity clustering is a key component for grouping
radar points and enabling cross-attention between velocity clusters.
"""

import torch
import numpy as np
from sklearn.cluster import DBSCAN


def velocity_clustering_dbscan(velocities, eps=0.5, min_samples=2):
    """Cluster points by radial velocity using DBSCAN.

    Args:
        velocities: (N,) numpy array of radial velocities
        eps: maximum velocity difference within a cluster (m/s)
        min_samples: minimum points to form a cluster

    Returns:
        labels: (N,) cluster labels (-1 for noise)
    """
    if len(velocities) == 0:
        return np.array([], dtype=np.int32)

    # DBSCAN on 1D velocity
    vel_2d = velocities.reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(vel_2d)

    # Assign noise points to nearest cluster
    noise_mask = labels == -1
    if noise_mask.any() and (~noise_mask).any():
        valid_labels = labels[~noise_mask]
        valid_vels = velocities[~noise_mask]

        for i in np.where(noise_mask)[0]:
            dists = np.abs(valid_vels - velocities[i])
            nearest = np.argmin(dists)
            labels[i] = valid_labels[nearest]
    elif noise_mask.all():
        # All points are noise — put in single cluster
        labels[:] = 0

    return labels


def compute_cluster_features(pillar_features, cluster_labels, num_clusters):
    """Compute mean feature vector for each velocity cluster.

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

    cluster_features = torch.zeros(num_clusters, C, device=device)
    cluster_counts = torch.zeros(num_clusters, device=device)

    for c in range(num_clusters):
        mask = cluster_labels == c
        if mask.any():
            cluster_features[c] = pillar_features[mask].mean(dim=0)
            cluster_counts[c] = mask.sum().float()

    return cluster_features, cluster_counts
