# T1

Some measurement data of T1 are used to predict the prognosis of depression diagnosis and treatment.

This repository now contains a **Cortical Multi-View Network** implementation for FreeSurfer-derived cortical surface data. It fuses three complementary views (mesh-level, ROI-level, and spherical-map CNN features) to predict binary treatment response labels.

## Structure

- `src/data_utils.py`: Data loaders for FreeSurfer surfaces, ROI annotations, and spherical maps, plus the `CorticalDataset` for multi-view samples.
- `src/model.py`: Three-branch network with SplineCNN (mesh), GraphSAGE (ROI), and CNN (sphere) plus gated fusion and classifier.
- `src/train.py`: Training script that wires dataset, model, optimizer, and loop for binary classification.

## Usage

1. Organize data as:
   ```
   data_root/
     subj_001/
       surf/lh.midthickness rh.midthickness lh.curv rh.curv ...
       label/lh.aparc.annot rh.aparc.annot
     subj_002/
       ...
   labels.csv  # columns: subject_id,label
   ```

2. Install dependencies (PyTorch, torch-geometric, nibabel, torchvision, pandas, scipy).

3. Train:
   ```bash
   python -m src.train labels.csv data_root --batch-size 2 --epochs 50 --lr 1e-4
   ```

The script infers feature dimensions from the first sample and saves `cortical_multiview.pt` after training.
