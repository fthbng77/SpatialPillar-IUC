"""
KDE Density Branch for BEV Feature Augmentation

Computes a Gaussian Kernel Density Estimation (KDE) map from point cloud
(x, y) coordinates in BEV space, then processes it through a small CNN
and concatenates the resulting features with the BEV backbone output.

Motivation: SMURF paper — BEV density maps help identify regions with
higher point concentrations, improving detection in sparse radar data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KDEDensityBranch(nn.Module):
    """Computes point density in BEV and fuses with backbone features.

    This module:
    1. Creates a BEV density map from point cloud (x, y) coordinates using
       Gaussian kernel density estimation
    2. Processes the density map through a small CNN
    3. Concatenates the result with BEV backbone features

    Args:
        model_cfg: config node with:
            - BANDWIDTH: KDE bandwidth in meters (default: 1.0)
            - NUM_DENSITY_FEATURES: output channels from density CNN (default: 16)
        input_channels: number of input BEV features (from backbone_2d)
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: [vx, vy, vz]
        grid_size: [nx, ny, nz]
    """

    def __init__(self, model_cfg, input_channels, point_cloud_range, voxel_size, grid_size):
        super().__init__()
        self.model_cfg = model_cfg
        self.point_cloud_range = np.array(point_cloud_range)
        self.voxel_size = np.array(voxel_size)
        self.grid_size = grid_size

        self.bandwidth = model_cfg.get('BANDWIDTH', 1.0)
        num_density_features = model_cfg.get('NUM_DENSITY_FEATURES', 16)

        # BEV grid dimensions (same as spatial_features)
        self.nx = int(grid_size[0])
        self.ny = int(grid_size[1])

        # Small CNN to process density map
        self.density_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, num_density_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_density_features, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # Output channels = backbone features + density features
        self.num_bev_features = input_channels + num_density_features

    def _compute_density_map(self, points, batch_size, device):
        """Compute Gaussian KDE density map in BEV.

        Args:
            points: (N_total, C) with columns [batch_idx, x, y, z, ...]
            batch_size: int
            device: torch device

        Returns:
            density_maps: (B, 1, ny, nx) normalized density maps
        """
        density_maps = torch.zeros((batch_size, 1, self.ny, self.nx),
                                   device=device, dtype=torch.float32)

        for b in range(batch_size):
            if points.shape[1] > 4:
                # Points have batch index in first column (collated format)
                mask = points[:, 0].long() == b
                pts = points[mask, 1:3]  # x, y
            else:
                # Raw points without batch index
                mask = torch.ones(points.shape[0], dtype=torch.bool)
                pts = points[:, :2]

            if pts.shape[0] == 0:
                continue

            # Convert to voxel grid indices (continuous)
            x_idx = (pts[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
            y_idx = (pts[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]

            # Bandwidth in voxel units
            bw_voxels = self.bandwidth / self.voxel_size[0]

            # Create density map using scatter with Gaussian weights
            # For efficiency, use a simplified scatter approach
            x_int = x_idx.long().clamp(0, self.nx - 1)
            y_int = y_idx.long().clamp(0, self.ny - 1)

            # Flat index for scatter_add
            flat_idx = y_int * self.nx + x_int
            density_flat = torch.zeros(self.ny * self.nx, device=device)
            ones = torch.ones(flat_idx.shape[0], device=device)
            density_flat.scatter_add_(0, flat_idx, ones)

            density_map = density_flat.view(1, self.ny, self.nx)

            # Apply Gaussian smoothing (approximates KDE)
            kernel_size = int(2 * bw_voxels) * 2 + 1
            kernel_size = max(3, min(kernel_size, 15))  # Clamp kernel size
            sigma = bw_voxels
            density_map = density_map.unsqueeze(0)  # (1, 1, H, W)
            density_map = self._gaussian_blur(density_map, kernel_size, sigma)
            density_map = density_map.squeeze(0)  # (1, H, W)

            # Normalize to [0, 1]
            max_val = density_map.max()
            if max_val > 0:
                density_map = density_map / max_val

            density_maps[b] = density_map

        return density_maps

    @staticmethod
    def _gaussian_blur(x, kernel_size, sigma):
        """Apply 2D Gaussian blur using separable convolution."""
        channels = x.shape[1]

        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        # Reshape for depthwise convolution
        kernel_h = g.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
        kernel_w = g.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)

        padding_h = kernel_size // 2
        padding_w = kernel_size // 2

        # Apply separable convolution
        x = F.conv2d(x, kernel_h, padding=(padding_h, 0), groups=channels)
        x = F.conv2d(x, kernel_w, padding=(0, padding_w), groups=channels)
        return x

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features_2d: (B, C, H, W) from backbone_2d
                points: (N, C) original point cloud data
                batch_size: int

        Returns:
            data_dict with updated spatial_features_2d (concatenated with density features)
        """
        spatial_features_2d = data_dict['spatial_features_2d']
        batch_size = data_dict['batch_size']
        device = spatial_features_2d.device

        # Get points from batch_dict
        if 'points' in data_dict:
            points = data_dict['points']
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points).float().to(device)
            elif not points.is_cuda:
                points = points.to(device)
        else:
            # Fallback: use voxel coordinates
            coords = data_dict['voxel_coords']  # (N, 4) [batch, z, y, x]
            points = torch.zeros((coords.shape[0], 3), device=device)
            points[:, 0] = coords[:, 0].float()  # batch idx
            points[:, 1] = coords[:, 3].float() * self.voxel_size[0] + self.point_cloud_range[0]
            points[:, 2] = coords[:, 2].float() * self.voxel_size[1] + self.point_cloud_range[1]

        # Compute density map
        density_maps = self._compute_density_map(points, batch_size, device)

        # Match spatial resolution of backbone output
        if density_maps.shape[2:] != spatial_features_2d.shape[2:]:
            density_maps = F.interpolate(
                density_maps,
                size=spatial_features_2d.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Process through density CNN
        density_features = self.density_cnn(density_maps)

        # Concatenate with backbone features
        data_dict['spatial_features_2d'] = torch.cat(
            [spatial_features_2d, density_features], dim=1
        )

        return data_dict
