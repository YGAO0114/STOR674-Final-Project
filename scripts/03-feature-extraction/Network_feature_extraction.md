# Network Feature Extraction

This folder contains scripts to compute morphological, geometric, and topological features from the **reconstructed vascular graphs** produced in the previous step.

## Overview

The feature extraction pipeline reads node/edge tables from reconstruction and produces **per‑image** Excel workbooks with quantitative metrics. The main script handles:

- **Regional Area Analysis**: Spatial vessel area statistics
- **Branching Angle Statistics**: Junction angle distributions
- **Vascular Density**: Area and length density (global & regional)
- **Fractal Dimension**: Box‑counting complexity (vessel/skeleton)
- **Topological Metrics (Betti Numbers)**: Components, loops, voids

## Files

- `feature_extraction.py` – Core feature computation script
- `_utility.py` – Shared helpers
- **Reconstruction inputs used by this step (examples):**
  - `N_129_predicted_viz.tif_alldata.xlsx`
  - `N_129_predicted_viz.tif_degreedata.xlsx`
  - `N_129_predicted.tif_alldata.xlsx`
  - `N_129_predicted.tif_degreedata.xlsx`
  - `N_129_groundtruth.tif_alldata.xlsx`
  - `N_129_groundtruth.tif_degreedata.xlsx`

## Dataset Structure

### Input Structure
Reconstructed graph tables in:
```
Data/validation_data/feature_extraction/feature/
├── N_129_predicted_viz.tif_alldata.xlsx
├── N_129_predicted_viz.tif_degreedata.xlsx
├── N_129_predicted.tif_alldata.xlsx
├── N_129_predicted.tif_degreedata.xlsx
├── N_129_groundtruth.tif_alldata.xlsx
└── N_129_groundtruth.tif_degreedata.xlsx
```

### Output Structure
Feature workbooks generated in the same folder:
```
Data/validation_data/feature_extraction/feature/
├── N_129_regional_area.xlsx
├── N_129_groundtruth.tif_angle_statistics.xlsx
├── N_129_groundtruth.tif_betti_numbers.xlsx
├── N_129_groundtruth.tif_fractal_dimension.xlsx
├── N_129_groundtruth.tif_vascular_density.xlsx
├── N_129_predicted_viz.tif_angle_statistics.xlsx
├── N_129_predicted_viz.tif_betti_numbers.xlsx
├── N_129_predicted_viz.tif_fractal_dimension.xlsx
├── N_129_predicted_viz.tif_vascular_density.xlsx
├── N_129_predicted.tif_angle_statistics.xlsx
├── N_129_predicted.tif_fractal_dimension.xlsx
└── N_129_predicted.tif_vascular_density.xlsx
```
> Only files present in your project are listed above.

## Usage

### Prerequisites
- Python 3.x
- Required packages installed
  ```bash
  pip install pandas numpy openpyxl networkx scikit-image scipy matplotlib
  ```

### Running the Feature Extraction
```bash
cd Data/validation_data/feature_extraction
python feature_extraction.py
```

## What the Script Does

1. **Regional Area Analysis**
   - Computes regional totals, means, and density ratios
   - Output: `N_129_regional_area.xlsx`

2. **Branching Angle Statistics**
   - Calculates per‑junction angles and global summaries
   - Outputs: `N_129_groundtruth.tif_angle_statistics.xlsx`, `N_129_predicted_viz.tif_angle_statistics.xlsx`, `N_129_predicted.tif_angle_statistics.xlsx`

3. **Vascular Density**
   - Computes VAD & VLD globally and by region
   - Outputs: `N_129_groundtruth.tif_vascular_density.xlsx`, `N_129_predicted_viz.tif_vascular_density.xlsx`, `N_129_predicted.tif_vascular_density.xlsx`

4. **Fractal Dimension**
   - Box‑counting FD for vessels/skeletons; reports R²
   - Outputs: `N_129_groundtruth.tif_fractal_dimension.xlsx`, `N_129_predicted_viz.tif_fractal_dimension.xlsx`, `N_129_predicted.tif_fractal_dimension.xlsx`

5. **Betti Numbers**
   - Topology: β₀ (components), β₁ (loops), β₂ (voids)
   - Outputs: `N_129_groundtruth.tif_betti_numbers.xlsx`, `N_129_predicted_viz.tif_betti_numbers.xlsx`

## Output

The script logs progress and confirms generated files in `feature/`.

### Example Output
```
=== Network Feature Extraction ===
Inputs: N_129_predicted_viz.tif_alldata.xlsx, N_129_predicted_viz.tif_degreedata.xlsx
 - Regional area ... done
 - Angles ... done
 - Density ... done
 - Fractal ... done
 - Betti ... done
Outputs written to ./feature/
```

## Integration with Pipeline

This is stage **03‑feature‑extraction** in our project:
1. **01‑segmentation** → binary masks
2. **02‑reconstruction** → node/edge tables & network PNG
3. **03‑feature‑extraction** (this) → angle, density, fractal, Betti, regional metrics


