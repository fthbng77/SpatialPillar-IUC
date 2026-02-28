"""
GeoSPA: Geometric Spatial Features for Point Clouds

Computes Lalonde geometric features (scatterness, linearness, surfaceness)
from local point cloud neighborhoods using KNN + covariance eigenvalue analysis.

Reference: MUFASA - "Multi-Feature Aggregation for Semantic Analysis"
These features capture local geometric structure:
  - scatterness: λ3/λ1 — high for uniformly distributed points
  - linearness: (λ1-λ2)/λ1 — high for points along a line (e.g., poles, edges)
  - surfaceness: (λ2-λ3)/λ1 — high for planar surfaces
"""

import numpy as np
from scipy.spatial import cKDTree


def compute_geospa_features(points, k=16):
    """Compute geometric features for each point using KNN neighborhoods.

    Args:
        points: (N, C) numpy array where first 3 columns are x, y, z
        k: number of nearest neighbors

    Returns:
        features: (N, 3) numpy array [scatterness, linearness, surfaceness]
    """
    N = points.shape[0]
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32)

    xyz = points[:, :3].astype(np.float64)

    # Build KD-tree
    tree = cKDTree(xyz)

    # Query k nearest neighbors (including self)
    k_actual = min(k, N)
    _, indices = tree.query(xyz, k=k_actual)

    # If N < k, indices shape may differ
    if indices.ndim == 1:
        indices = indices[:, np.newaxis]

    features = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        neighbors = xyz[indices[i]]  # (k, 3)

        if neighbors.shape[0] < 3:
            # Not enough neighbors for meaningful covariance
            continue

        # Compute covariance matrix
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        cov = centered.T @ centered / neighbors.shape[0]

        # Compute eigenvalues (sorted ascending)
        try:
            eigenvalues = np.linalg.eigvalsh(cov)
        except np.linalg.LinAlgError:
            continue

        # Sort descending: λ1 >= λ2 >= λ3
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Avoid division by zero
        lambda1 = max(eigenvalues[0], 1e-10)
        lambda2 = eigenvalues[1]
        lambda3 = eigenvalues[2]

        # Geometric features
        features[i, 0] = lambda3 / lambda1         # scatterness
        features[i, 1] = (lambda1 - lambda2) / lambda1  # linearness
        features[i, 2] = (lambda2 - lambda3) / lambda1  # surfaceness

    return features


def compute_geospa_features_batch(points, k=16):
    """Vectorized batch version using matrix operations for better performance.

    Args:
        points: (N, C) numpy array
        k: number of nearest neighbors

    Returns:
        features: (N, 3) numpy array
    """
    N = points.shape[0]
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32)

    xyz = points[:, :3].astype(np.float64)
    k_actual = min(k, N)

    tree = cKDTree(xyz)
    _, indices = tree.query(xyz, k=k_actual)

    if indices.ndim == 1:
        indices = indices[:, np.newaxis]

    # Gather all neighbor points
    neighbors = xyz[indices]  # (N, k, 3)

    # Compute centroids and centered coordinates
    centroids = neighbors.mean(axis=1, keepdims=True)  # (N, 1, 3)
    centered = neighbors - centroids  # (N, k, 3)

    # Batch covariance: (N, 3, 3)
    cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / k_actual

    # Batch eigenvalue computation
    try:
        eigenvalues = np.linalg.eigvalsh(cov_matrices)  # (N, 3) ascending
    except np.linalg.LinAlgError:
        return compute_geospa_features(points, k)

    # Sort descending per row
    eigenvalues = eigenvalues[:, ::-1]  # λ1 >= λ2 >= λ3

    lambda1 = np.maximum(eigenvalues[:, 0], 1e-10)
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]

    features = np.zeros((N, 3), dtype=np.float32)
    features[:, 0] = lambda3 / lambda1              # scatterness
    features[:, 1] = (lambda1 - lambda2) / lambda1  # linearness
    features[:, 2] = (lambda2 - lambda3) / lambda1  # surfaceness

    return features
