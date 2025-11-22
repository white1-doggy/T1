"""
Utilities for loading FreeSurfer-derived cortical surface data and building
multi-view tensors suitable for the Cortical Multi-View Network.

This module handles three kinds of inputs:
1. Mesh-level features (vertices/faces + per-vertex metrics).
2. ROI-level graph features derived from annotation files.
3. Spherical map tensors built from resampled morphometric data.

The code is written to be reusable for both left/right hemispheres and to
support common FreeSurfer outputs (``*.white``, ``*.pial``, ``*.midthickness``
meshes; ``*.curv``, ``*.sulc``, ``*.thickness``, ``*.area`` morphometrics; and
``*.annot`` labels).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

try:
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - torch_geometric may not be installed in CI
    Data = object  # type: ignore[misc,assignment]


@dataclasses.dataclass
class SurfaceMeshes:
    """Container for a subject's cortical meshes and morphometric features."""

    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (F, 3)
    features: np.ndarray  # (N, C) vertex-wise metrics


@dataclasses.dataclass
class RoiGraph:
    """Graph representation of ROI-level features."""

    x: torch.Tensor  # (num_rois, num_features)
    edge_index: torch.Tensor  # (2, num_edges)


@dataclasses.dataclass
class SphericalMap:
    """Spherical morphometric data resampled to a 2D grid."""

    tensor: torch.Tensor  # (channels, H, W)


@dataclasses.dataclass
class CorticalSample:
    """Aggregate sample returned by the dataset."""

    mesh: Data
    roi: Data
    sphere: torch.Tensor
    label: torch.Tensor


def read_surface_with_features(mesh_path: Path, metrics: Sequence[Path]) -> SurfaceMeshes:
    """Load a FreeSurfer surface and stack per-vertex metrics as features.

    Args:
        mesh_path: Path to a FreeSurfer surface (``.white``, ``.pial``, or ``.midthickness``).
        metrics: Paths to per-vertex metric files (e.g., ``.curv``, ``.sulc``).

    Returns:
        SurfaceMeshes containing vertices, faces, and stacked features.
    """

    coords, faces = nib.freesurfer.read_geometry(str(mesh_path))
    feature_list = []
    for metric_path in metrics:
        data = nib.freesurfer.read_morph_data(str(metric_path))
        feature_list.append(data.astype(np.float32))
    features = np.stack(feature_list, axis=-1) if feature_list else np.zeros((coords.shape[0], 0), dtype=np.float32)
    return SurfaceMeshes(vertices=coords.astype(np.float32), faces=faces.astype(np.int64), features=features)


def build_mesh_data(mesh: SurfaceMeshes) -> Data:
    """Convert SurfaceMeshes into a torch_geometric ``Data`` object."""

    verts = torch.from_numpy(mesh.vertices)
    faces = torch.from_numpy(mesh.faces).t().contiguous()  # shape (3, F)
    x = torch.from_numpy(mesh.features)
    edge_index = torch.cat(
        [faces[[0, 1]], faces[[1, 2]], faces[[2, 0]], faces[[1, 0]], faces[[2, 1]], faces[[0, 2]]],
        dim=1,
    )
    edge_index = torch.unique(edge_index, dim=1)
    return Data(x=x, edge_index=edge_index, pos=verts, face=faces)


def load_annot_roi_features(annot_path: Path, metric_paths: Sequence[Path], min_roi_size: int = 10) -> RoiGraph:
    """Compute ROI-level features by averaging per-vertex metrics within each label.

    Args:
        annot_path: FreeSurfer ``.annot`` path.
        metric_paths: Metric files matching the same mesh used for the annotation.
        min_roi_size: Ignore ROIs with fewer vertices than this threshold.

    Returns:
        RoiGraph with GraphSAGE-friendly inputs.
    """

    labels, ctab, names = nib.freesurfer.read_annot(str(annot_path))
    labels = labels.astype(np.int64)
    valid_mask = labels != -1
    metrics = [nib.freesurfer.read_morph_data(str(p)).astype(np.float32) for p in metric_paths]
    metrics_arr = np.stack(metrics, axis=-1)  # (N, C)

    roi_features = []
    roi_centers = []
    roi_indices = []
    for roi_id, roi_name in enumerate(names):
        roi_mask = labels == roi_id
        if roi_mask.sum() < min_roi_size:
            continue
        roi_indices.append(roi_id)
        roi_features.append(metrics_arr[roi_mask].mean(axis=0))
        roi_centers.append(np.argwhere(roi_mask)[0][0])

    roi_features_arr = np.stack(roi_features, axis=0)
    x = torch.from_numpy(roi_features_arr)

    # Build edges based on spatial proximity of ROI centroids in label index space.
    centers = np.array(roi_centers).reshape(-1, 1).astype(np.float32)
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=5, output_type="ndarray")  # connect ROIs with close indices
    if pairs.size == 0:
        # Fallback to fully connected small graph if query_pairs returns nothing
        src, dst = np.meshgrid(np.arange(len(roi_indices)), np.arange(len(roi_indices)))
        pairs = np.stack([src.ravel(), dst.ravel()], axis=1)
    edge_index = torch.from_numpy(np.vstack([pairs[:, 0], pairs[:, 1]])).long()
    return RoiGraph(x=x, edge_index=edge_index)


def load_spherical_map(gifti_paths: Sequence[Path], target_size: Tuple[int, int] = (128, 128)) -> SphericalMap:
    """Load one or more GIFTI/CIFTI shape files and resize them to a grid.

    Args:
        gifti_paths: Paths to ``.shape.gii`` or ``.dscalar.nii`` files.
        target_size: Output grid size (H, W).

    Returns:
        SphericalMap with concatenated channels.
    """

    channel_list: List[np.ndarray] = []
    for path in gifti_paths:
        img = nib.load(str(path))
        if isinstance(img, nib.gifti.GiftiImage):
            for da in img.darrays:
                channel_list.append(np.asarray(da.data, dtype=np.float32))
        else:
            data = np.asarray(img.get_fdata(), dtype=np.float32)
            if data.ndim == 4:  # (X, Y, Z, C)
                data = data.reshape(-1, data.shape[-1]).T
            channel_list.append(data)
    stacked = np.stack(channel_list, axis=0)
    # Normalize per-channel and reshape into square grid (naive mapping).
    normalized = []
    for ch in stacked:
        ch = (ch - ch.mean()) / (ch.std() + 1e-6)
        side = int(math.sqrt(ch.shape[0]))
        side = side if side * side == ch.shape[0] else int(math.ceil(math.sqrt(ch.shape[0])))
        grid = np.zeros((side * side,), dtype=np.float32)
        grid[: ch.shape[0]] = ch
        grid = grid.reshape(side, side)
        grid = torch.from_numpy(grid).unsqueeze(0)
        grid = TF.resize(grid, target_size, antialias=True)
        normalized.append(grid)
    tensor = torch.cat(normalized, dim=0)
    return SphericalMap(tensor=tensor)


class CorticalDataset(Dataset):
    """Dataset that yields multi-view cortical representations and labels.

    The dataset expects a CSV file with at least two columns:
    - ``subject_id``: Identifier that matches the directory layout of FreeSurfer outputs.
    - ``label``: Binary label (0/1) indicating treatment response.

    The ``data_root`` directory should contain per-subject subdirectories with
    ``surf/`` and ``label/`` content as generated by FreeSurfer.
    """

    def __init__(
        self,
        csv_path: Path,
        data_root: Path,
        metric_names: Sequence[str] = ("curv", "sulc", "thickness"),
        sphere_maps: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.metric_names = metric_names
        self.sphere_maps = sphere_maps or ["thickness.shape.gii"]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.df)

    def _metric_paths(self, subject_dir: Path, hemisphere: str) -> List[Path]:
        surf_dir = subject_dir / "surf"
        return [surf_dir / f"{hemisphere}.{name}" for name in self.metric_names]

    def _mesh_paths(self, subject_dir: Path, hemisphere: str) -> Path:
        return subject_dir / "surf" / f"{hemisphere}.midthickness"

    def _annot_path(self, subject_dir: Path, hemisphere: str) -> Path:
        return subject_dir / "label" / f"{hemisphere}.aparc.annot"

    def _sphere_paths(self, subject_dir: Path, hemisphere: str) -> List[Path]:
        sphere_dir = subject_dir / "surf"
        return [sphere_dir / f"{hemisphere}.{name}" for name in self.sphere_maps]

    def __getitem__(self, idx: int) -> CorticalSample:
        row = self.df.iloc[idx]
        subject_dir = self.data_root / str(row["subject_id"])
        label = torch.tensor([row["label"]], dtype=torch.float32)

        mesh_lh = read_surface_with_features(self._mesh_paths(subject_dir, "lh"), self._metric_paths(subject_dir, "lh"))
        mesh_rh = read_surface_with_features(self._mesh_paths(subject_dir, "rh"), self._metric_paths(subject_dir, "rh"))
        mesh = merge_meshes(mesh_lh, mesh_rh)
        mesh_data = build_mesh_data(mesh)

        roi = load_annot_roi_features(self._annot_path(subject_dir, "lh"), self._metric_paths(subject_dir, "lh"))
        roi_data = Data(x=roi.x, edge_index=roi.edge_index)

        sphere_map = load_spherical_map(self._sphere_paths(subject_dir, "lh"))

        return CorticalSample(mesh=mesh_data, roi=roi_data, sphere=sphere_map.tensor, label=label)


def merge_meshes(left: SurfaceMeshes, right: SurfaceMeshes) -> SurfaceMeshes:
    """Concatenate left/right hemisphere meshes into a single mesh space."""

    right_faces = right.faces + left.vertices.shape[0]
    vertices = np.vstack([left.vertices, right.vertices])
    faces = np.vstack([left.faces, right_faces])
    features = np.vstack([left.features, right.features])
    return SurfaceMeshes(vertices=vertices, faces=faces, features=features)


def save_dataset_config(path: Path, config: Dict) -> None:
    path.write_text(json.dumps(config, indent=2))


__all__ = [
    "SurfaceMeshes",
    "RoiGraph",
    "SphericalMap",
    "CorticalSample",
    "CorticalDataset",
    "read_surface_with_features",
    "build_mesh_data",
    "load_annot_roi_features",
    "load_spherical_map",
    "merge_meshes",
    "save_dataset_config",
]
