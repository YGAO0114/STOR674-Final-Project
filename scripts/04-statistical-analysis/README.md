# Statistical Analysis - Vascular Development (P2-P7)

## Overview

This directory contains statistical analysis scripts and results for analyzing vascular development in mouse retinal vessels from postnatal day 2 (P2) to postnatal day 7 (P7).

## Files

- **`statistical_analysis.py`**: Main analysis script
- **`config.py`**: Centralized configuration (paths, parameters, constants)
- **`test_statistical_analysis.py`**: Unit tests for reproducibility
- **`requirements.txt`**: Python dependencies with exact versions
- **`README.md`**: This file

## Environment Setup

### Prerequisites

**Python Version:**
- Python 3.12+ recommended
- Tested with Python 3.12.8

### Installation

**Option 1: Using requirements.txt (Recommended)**
```bash
# From the statistical-analysis directory
cd scripts/04-statistical-analysis
pip install -r requirements.txt
```

**Option 2: Manual installation**
```bash
pip install pandas==2.2.3 numpy==2.2.1 matplotlib==3.10.0 scipy==1.15.1 openpyxl==3.1.5
```

**Option 3: Using conda (if using project environment)**
```bash
# The project's environment.yml includes most dependencies
# But you may need to add openpyxl manually:
conda activate nnunet_env  # or your environment name
pip install openpyxl==3.1.5
```

### Testing Installation

```bash
# Run tests to verify everything is set up correctly
pytest test_statistical_analysis.py -v

# Or run a quick smoke test
python -c "from statistical_analysis import VascularDevelopmentAnalysis; print('✓ Import successful')"
```

## Usage

### Running the Analysis

From the project root directory:

```bash
python scripts/04-statistical-analysis/statistical_analysis.py
```

### What the Script Does

The script performs the following analyses:

1. **Loads all feature data** from P2-P7 timepoints:
   - Vascular density (VAD, VLD)
   - Fractal dimension (FD)
   - Branching angle statistics
   - Topological metrics (Betti numbers)
   - Regional area distribution

2. **Creates summary statistics table**: Aggregates all metrics across timepoints

3. **Generates temporal trend plots**: 9-panel visualization showing:
   - Vessel area density over time
   - Vessel length density over time
   - Fractal dimension evolution
   - Branching angle changes
   - Network fragmentation (β₀)
   - Network loops (β₁)
   - Network redundancy
   - Void area fraction
   - Number of branch points

4. **Performs trend analysis**: Linear regression to quantify temporal trends:
   - Slope, R², p-value for each metric
   - Identifies significant vs. non-significant trends

5. **Creates correlation matrix**: Heatmap showing relationships between all features

6. **Generates comprehensive report**: Text summary of key findings

## Output Files

All results are saved to `results/statistical_analysis/`:

| File | Description |
|------|-------------|
| `summary_statistics.xlsx` | Table of all metrics across P2-P7 |
| `temporal_trends.png` | 9-panel time series plots |
| `trend_analysis.xlsx` | Statistical significance of trends |
| `correlation_matrix.png` | Feature correlation heatmap |
| `correlations.xlsx` | Numeric correlation values |
| `analysis_report.txt` | Text summary of findings |

## Key Results

### Major Findings

1. **Vessel Area Density**: +663% increase from P2 to P7x (p < 0.0001)
2. **Fractal Dimension**: +16% increase (p = 0.004)
3. **Network Loops (β₁)**: +1,030% increase (p < 0.0001)
4. **Branching Angles**: Stable at ~65-70° (p = 0.20, not significant)

### Interpretation Highlights

- **P2-P5**: Active angiogenesis phase (sprouting, rapid expansion)
- **P5-P7**: Vascular remodeling phase (optimization, maturation)
- **Fractal complexity**: Approaches theoretical limit for 2D space-filling
- **Branching angles**: Deviate from Murray's Law (37.5°) due to:
  - Astrocyte guidance constraints
  - Asymmetric bifurcations
  - Anastomotic connections

See `INTERPRETATION.md` for detailed biological interpretation and literature context.

## Customization

### Analyzing Different Time Points

Edit `__init__` method in `VascularDevelopmentAnalysis` class:

```python
self.time_points = ['p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p7x']
```

### Adding New Metrics

Add to `load_all_features()` method:

```python
# Example: Add new feature
new_feature_file = self.feature_dir / f'vessel_{tp}_new_feature.xlsx'
if new_feature_file.exists():
    df = pd.read_excel(new_feature_file)
    df['time_point'] = tp
    df['day'] = int(tp[1]) if tp != 'p7x' else 7.5
    self.new_feature_data.append(df)
```

### Modifying Visualizations

The plotting functions use matplotlib. Example modification:

```python
# Change color scheme
plt.plot(x, y, 'o-', linewidth=2, markersize=8, color='your_color')

# Add additional subplot
fig, axes = plt.subplots(4, 3, figsize=(18, 18))  # Was 3x3
```

## Statistical Methods

### Trend Analysis

- **Method**: Linear regression (scipy.stats.linregress)
- **Metrics**: Slope, R², p-value
- **Significance threshold**: α = 0.05

### Correlation Analysis

- **Method**: Pearson correlation (pandas.DataFrame.corr)
- **Interpretation**:
  - |r| > 0.9: Very strong
  - 0.7 < |r| < 0.9: Strong
  - 0.5 < |r| < 0.7: Moderate
  - |r| < 0.5: Weak

## Reproducibility Notes

### Random Seed

No random processes used - analysis is fully deterministic.

### Software Versions

Tested with:
- Python 3.12+
- pandas 2.0+
- numpy 1.24+
- matplotlib 3.7+
- scipy 1.11+

### Data Requirements

The script expects this directory structure:

```
Data/validation_data/feature_extraction/feature/p2-p7/
├── vessel_p2_vascular_density.xlsx
├── vessel_p2_fractal_dimension.xlsx
├── vessel_p2_angle_statistics.xlsx
├── vessel_p2_betti_numbers.xlsx
├── vessel_p2_regional_area.xlsx
├── ... (same for p3, p4, p5, p6, p7, p7x)
```

## Integration with Project Pipeline

This is **stage 04** of the project workflow:

1. **00-EDA**: Exploratory data analysis
2. **01-segmentation**: Blood vessel segmentation (nnUNet)
3. **02-reconstruction**: Network skeleton extraction
4. **03-feature-extraction**: Morphological feature computation
5. **04-statistical-analysis** (this): Temporal trend analysis ← **YOU ARE HERE**

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: 'openpyxl'**

```bash
pip install openpyxl
```

**2. ValueError: Worksheet named 'X' not found**

Check sheet names in your Excel files:

```python
import pandas as pd
xl = pd.ExcelFile('your_file.xlsx')
print(xl.sheet_names)
```

**3. KeyError: Column not found**

Verify column names match between script and data files. Update script if needed:

```python
# Check actual columns
df = pd.read_excel('your_file.xlsx')
print(df.columns)
```

**4. Empty plots**

Ensure data was loaded correctly:

```python
analyzer = VascularDevelopmentAnalysis(data_dir)
analyzer.load_all_features()
print(f"Density records: {len(analyzer.density_df)}")
print(f"Fractal records: {len(analyzer.fractal_df)}")
```

## Future Enhancements

Potential additions

## Contact & Questions

For questions about this analysis, please see:             
- **Interpretation guide**: `INTERPRETATION.md`          
- **Feature extraction guide**:   `../03-feature-extraction/Network_feature_extraction.md`         
- **Project documentation**: `../../README.md`               

---


