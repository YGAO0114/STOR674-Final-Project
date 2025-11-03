# Vessel Network Reconstruction

This folder contains scripts to reconstruct vascular networks (skeletons, nodes, and edges) from **binary vessel masks** and export graph data for downstream analysis.

## Overview

The reconstruction pipeline converts binary masks into a **topology‑preserving graph** required for quantitative network analysis. The main script handles:

- **Skeletonization**: One‑pixel centerlines via Guo & Hall (1989)
- **Node Detection**: Identifies branching (degree ≥3) and terminal points
- **Edge Tracing**: Connects nodes along skeleton paths to form segments
- **Graph Export**: Writes node/edge tables and a network visualization for QC

## Files

- `feature_extraction.py` – Main reconstruction and export script
- `_utility.py` – Helper utilities used by the main script
- `feature_extraction.ipynb` – Notebook for quick inspection
- `connectivity_matrix_test.py` – Optional connectivity diagnostic

## Dataset Structure

### Input Structure
The script expects **binary** masks (1 = vessel, 0 = background) in:
```
Data/validation_data/feature_extraction/img/
├── N_129_groundtruth.tif
├── N_129_predicted.tif
└── N_129_predicted_viz.tif
```

### Output Structure
Reconstruction outputs are written to:
```
Data/validation_data/feature_extraction/feature/
├── N_129_alldata.xlsx
├── N_129_degreedata.xlsx
├── N_129_network.png
├── N_129_groundtruth.tif_alldata.xlsx
├── N_129_groundtruth.tif_degreedata.xlsx
├── N_129_groundtruth.tif_network.png
├── N_129_predicted_viz.tif_alldata.xlsx
├── N_129_predicted_viz.tif_degreedata.xlsx
├── N_129_predicted_viz.tif_network.png
├── N_129_predicted.tif_alldata.xlsx
├── N_129_predicted.tif_degreedata.xlsx
└── N_129_predicted.tif_network.png
```
> One trio of files (edge table, node table, network PNG) is produced **per input mask**.

## Usage

### Prerequisites
- Python 3.x
- Required packages installed
  ```bash
  pip install networkx scikit-image opencv-python numpy pandas openpyxl
  ```

### Running the Reconstruction
```bash
cd Data/validation_data/feature_extraction
python feature_extraction.py
```

## What the Script Does

1. **Image Processing**
   - Loads binary masks from `img/`
   - Performs thinning to obtain 1‑px skeletons

2. **Node & Edge Generation**
   - Detects endpoints and branching nodes (merges near‑duplicates)
   - Traces edges between nodes and measures segment geometry (length, mean/SD area, perimeter)

3. **Export & QC**
   - Saves `*_alldata.xlsx` (edges), `*_degreedata.xlsx` (nodes)
   - Renders `*_network.png` overlay for visual verification

## Output

The script prints progress (images processed, nodes/edges counts) and confirms saved files in `feature/`.

### Example Output
```
=== Vessel Network Reconstruction ===
Processing: N_129_predicted.tif
 - Skeletonization complete
 - Nodes detected: 3,4xx | Edges: 3,4xx
 - Exported: N_129_predicted.tif_alldata.xlsx, N_129_predicted.tif_degreedata.xlsx, N_129_predicted.tif_network.png
=== Complete ===
```

## Integration with Analysis

This reconstruction step is stage **02‑reconstruction** in our project:
1. **01‑segmentation** → produce binary masks
2. **02‑reconstruction** (this) → graph & QC image
3. **03‑feature‑extraction** → compute angles, density, fractal, Betti, regional metrics

The reconstructed node/edge tables are required inputs for the next stage.
