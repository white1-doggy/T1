"""
Training script for depression treatment response prediction using the
Cortical Multi-View Network.

The script stitches together dataset, dataloaders, model, optimizer, and a
simple training/evaluation loop. It is intentionally lightweight and modular so
it can be adapted to different cohort sizes and cross-validation schemes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_utils import CorticalDataset
from src.model import CorticalMultiViewNet


def collate_pyg(batch):
    """Custom collate function to handle PyG Data objects and tensors."""

    from torch_geometric.loader import DataLoader as GeoLoader

    mesh_list = [item.mesh for item in batch]
    roi_list = [item.roi for item in batch]
    sphere_list = [item.sphere for item in batch]
    labels = torch.cat([item.label for item in batch], dim=0)

    mesh_batch = GeoLoader(mesh_list, batch_size=len(mesh_list)).collate_fn(mesh_list)
    roi_batch = GeoLoader(roi_list, batch_size=len(roi_list)).collate_fn(roi_list)
    sphere_batch = torch.stack(sphere_list, dim=0)
    return mesh_batch, roi_batch, sphere_batch, labels


def train_one_epoch(model, loader, optimizer, device) -> Tuple[float, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for mesh, roi, sphere, label in loader:
        mesh, roi, sphere, label = mesh.to(device), roi.to(device), sphere.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(mesh, roi, sphere)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        preds = (logits > 0).long()
        total_correct += (preds == label.long()).sum().item()
        total += label.size(0)
    return total_loss / total, total_correct / total


def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for mesh, roi, sphere, label in loader:
            mesh, roi, sphere, label = mesh.to(device), roi.to(device), sphere.to(device), label.to(device)
            logits = model(mesh, roi, sphere)
            loss = criterion(logits, label)
            total_loss += loss.item() * label.size(0)
            preds = (logits > 0).long()
            total_correct += (preds == label.long()).sum().item()
            total += label.size(0)
    return total_loss / total, total_correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train cortical multi-view network")
    parser.add_argument("csv", type=Path, help="CSV with subject_id and label columns")
    parser.add_argument("data_root", type=Path, help="Root directory containing subject folders")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = CorticalDataset(args.csv, args.data_root)

    # Infer feature dimensions from a tiny probe sample
    sample = dataset[0]
    mesh_feat = sample.mesh.num_node_features
    roi_feat = sample.roi.num_node_features
    sphere_channels = sample.sphere.shape[0]

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_pyg)

    model = CorticalMultiViewNet(mesh_feat=mesh_feat, roi_feat=roi_feat, sphere_channels=sphere_channels).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_acc = evaluate(model, train_loader, args.device)  # Placeholder; swap with real val loader
        print(
            f"Epoch {epoch+1:03d} | Train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

    torch.save(model.state_dict(), "cortical_multiview.pt")


if __name__ == "__main__":  # pragma: no cover
    main()
