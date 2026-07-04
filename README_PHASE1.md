# Phase 1 - Sign Language Dataset Analysis and Preprocessing

This folder contains the modular implementation of Phase 1 for Project 3.

## Purpose

Phase 1 includes:

- Loading the sign language image dataset
- Analyzing class distribution
- Visualizing sample images from each class
- Resizing images
- Normalizing pixel values to `[0, 1]`
- Splitting the dataset into train, validation, and test sets
- Creating reusable PyTorch Dataset and DataLoader objects

## Expected Dataset Structure

Place your dataset in a folder named `dataset`:

```text
dataset/
├── A/
│   ├── image1.jpg
│   └── image2.jpg
├── B/
│   ├── image1.jpg
│   └── image2.jpg
└── Nothing/
    ├── image1.jpg
    └── image2.jpg
```

Each class must have its own folder.

## Files

```text
src/
├── phase1_config.py
├── phase1_dataset.py
├── phase1_eda.py
├── phase1_preprocessing.py
└── __init__.py

phase1_run.py
notebooks/phase1.ipynb
requirements.txt
```

## How to Run

Install requirements:

```bash
pip install -r requirements.txt
```

Run Phase 1:

```bash
python phase1_run.py
```

## Outputs

The script creates:

```text
outputs/plots/class_distribution.png
outputs/plots/sample_grid.png
data_splits/train.csv
data_splits/val.csv
data_splits/test.csv
```

## Import Example

```python
from src.phase1_dataset import scan_image_folder, split_dataset
from src.phase1_preprocessing import build_dataloaders
```
