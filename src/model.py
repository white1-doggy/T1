"""
Model definitions for the Cortical Multi-View Network.

The network contains three branches:
- MeshBranch: processes vertex-level mesh features via SplineCNN layers.
- RoiBranch: processes ROI graph features via GraphSAGE.
- SphereBranch: processes spherical maps via lightweight 2D CNNs.

A gated fusion module learns per-branch importance before classification.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SplineConv, SAGEConv, global_mean_pool
except ImportError:  # pragma: no cover - torch_geometric may not be installed in CI
    SplineConv = None  # type: ignore[assignment]
    SAGEConv = None  # type: ignore[assignment]
    global_mean_pool = None  # type: ignore[assignment]


class MeshBranch(nn.Module):
    """SplineCNN branch for full-surface mesh processing."""

    def __init__(self, in_channels: int, hidden: int = 64, out_channels: int = 128):
        super().__init__()
        if SplineConv is None:
            raise ImportError("torch_geometric is required for MeshBranch")
        self.conv1 = SplineConv(in_channels, hidden, dim=3, kernel_size=5)
        self.conv2 = SplineConv(hidden, hidden, dim=3, kernel_size=5)
        self.conv3 = SplineConv(hidden, out_channels, dim=3, kernel_size=5)

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.pos
        x = F.elu(self.conv1(x, edge_index, pos))
        x = F.elu(self.conv2(x, edge_index, pos))
        x = self.conv3(x, edge_index, pos)
        return global_mean_pool(x, batch=data.batch)


class RoiBranch(nn.Module):
    """GraphSAGE branch for ROI-level graph processing."""

    def __init__(self, in_channels: int, hidden: int = 64, out_channels: int = 64):
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for RoiBranch")
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch=data.batch)


class SphereBranch(nn.Module):
    """Lightweight CNN over resampled spherical maps."""

    def __init__(self, in_channels: int, base_channels: int = 32, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(base_channels * 4, out_channels)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.head(x)


class GatedFusion(nn.Module):
    """Learnable gating across multiple embeddings."""

    def __init__(self, dims: List[int]):
        super().__init__()
        self.gates = nn.Parameter(torch.ones(len(dims)))
        self.proj = nn.ModuleList([nn.Identity() for _ in dims])

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        assert len(embeddings) == len(self.gates)
        weights = torch.softmax(self.gates, dim=0)
        weighted = [w * e for w, e in zip(weights, embeddings)]
        return torch.cat(weighted, dim=-1)


class CorticalMultiViewNet(nn.Module):
    """Three-branch cortical surface classifier."""

    def __init__(self, mesh_feat: int, roi_feat: int, sphere_channels: int):
        super().__init__()
        self.mesh_branch = MeshBranch(mesh_feat)
        self.roi_branch = RoiBranch(roi_feat)
        self.sphere_branch = SphereBranch(sphere_channels)
        self.fusion = GatedFusion([128, 64, 128])
        self.classifier = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, mesh_data, roi_data, sphere_tensor):
        mesh_embed = self.mesh_branch(mesh_data)
        roi_embed = self.roi_branch(roi_data)
        sphere_embed = self.sphere_branch(sphere_tensor)
        fused = self.fusion([mesh_embed, roi_embed, sphere_embed])
        return self.classifier(fused).squeeze(-1)
