# STOR674 Final Project: Multi-Scale Topological and Geometric Analysis of Vascular Network Development in Mouse Retina

A comprehensive pipeline for blood vessel segmentation, network reconstruction, and feature extraction from medical images using deep learning (nnU-Net) and topological analysis.

**Course:** Statistical Computing (STOR 674)  
**Institution:** UNC Chapel Hill
**Authors:** Yuan Gao, Jiaxin Ying, Meihan Chen
---

## 🎯 Project Overview

This project implements a complete pipeline for analyzing blood vessel networks in medical images:

1. **Segmentation**: Use nnU-Net (a self-configuring deep learning framework) to segment blood vessels from 2D medical images
2. **Reconstruction**: Convert segmented binary masks into vessel network graphs with nodes and edges
3. **Feature Extraction**: Compute morphological, geometric, and topological features from the reconstructed networks
4. **Statistical Analysis**: Perform longitudinal analysis of vascular development (P2-P7 postnatal days)

The pipeline is designed to process mouse retinal vessel images at multiple developmental timepoints.

### Key Technologies

- **Deep Learning**: nnU-Net v2 for medical image segmentation
- **Image Processing**: scikit-image, nibabel, SimpleITK
- **Graph Analysis**: networkx for vascular network topology
- **Statistical Analysis**: scipy, pandas, matplotlib
- **Containerization**: Singularity (optional, for reproducibility)

---

## 📊 Dataset Description

### Data Structure

```
Data/
├── train_data/              # Training data for model
│   ├── image/              # Raw vessel images (.tif/.tiff)
│   └── ground_truth/       # Annotated segmentation masks (.npy)
├── validation_data/         # Validation data (P2-P7 timepoints)
│   ├── image/              # Vessel images for each timepoint
│   ├── ground_truth/       # Corresponding masks
│   ├── feature_extraction/ # Intermediate reconstruction outputs
│   └── img/                # Visualization outputs
└── test_data/              # Test set (optional)
```

### Expected Image Properties

- **Format**: TIFF (.tif/.tiff) or compatible medical image formats
- **Modality**: 2D medical images
- **Task**: Binary segmentation (background vs. blood vessels)
- **Preprocessing**: Converts RGB to grayscale for consistent input

---

## 🚀 Installation & Setup

### Option 1: Using Conda Environment (Recommended)

This is the simplest approach for most users.

#### Prerequisites

- Anaconda or Miniconda installed
- Python 3.10+
- CUDA-compatible GPU (optional, but recommended for faster training)

#### Installation Steps

```bash
# 1. Navigate to project directory
cd /path/to/STOR674-Final-Project

# 2. Create conda environment from file
conda env create -f environment.yml

# 3. Activate the environment
conda activate nnunet_env

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Optional: GPU Support

If you have a CUDA-compatible GPU:

```bash
# Verify GPU access
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

### Option 2: Using Singularity Container

For maximum reproducibility across different systems (HPC clusters, etc.).

#### Prerequisites

- Singularity installed (see [Singularity documentation](https://sylabs.io/singularity/))
- ~10GB disk space for the container
- Sylabs Cloud account (for remote building, no root required)

#### Building the Container

```bash
# Option A: Remote build (recommended, no sudo needed)
singularity remote login  # First time only, follow prompts to add token from https://cloud.sylabs.io
singularity build --remote stor674.sif Singularity.def

# Option B: Local build (requires sudo or fakeroot)
sudo singularity build stor674.sif Singularity.def
# OR with fakeroot (if configured)
singularity build --fakeroot stor674.sif Singularity.def
```

#### Running Commands in Container

```bash
# Interactive shell
singularity shell stor674.sif

# Run Python script
singularity exec stor674.sif python scripts/00-EDA/EDA.ipynb

# Run Jupyter Lab
singularity exec --bind $(pwd):/workspace stor674.sif jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Check installed packages
singularity exec stor674.sif pip list
```

---

### Option 3: Manual Installation

If you prefer to set up dependencies manually.

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch (choose appropriate version for your system)
pip install torch==2.9.0 torchvision==0.24.0  # CPU
# For GPU (CUDA 12.1):
# pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Install nnU-Net
pip install nnunetv2==2.6.2

# 6. Verify installation
python -c "import nnunetv2; print('nnU-Net successfully installed')"
```

---

## 📈 Project Pipeline

```
Raw Images
    ↓
[0] Exploratory Data Analysis (EDA)
    ├─ Load and visualize images
    ├─ Analyze image properties (shape, dtype, intensity distribution)
    ├─ Spatial analysis by rings
    └─ Generate EDA report
    ↓
[1] Preprocessing & Data Conversion
    ├─ Convert RGB to grayscale
    ├─ Standardize image format for nnU-Net
    └─ Organize training/validation datasets
    ↓
[2] Image Segmentation (nnU-Net)
    ├─ Preprocess data for nnU-Net
    ├─ Train segmentation model
    ├─ Evaluate on validation set
    └─ Generate binary vessel masks
    ↓
[3] Vascular Network Reconstruction
    ├─ Skeletonize binary masks
    ├─ Extract vessel network graph (nodes & edges)
    ├─ Compute node degrees and edge connections
    └─ Generate Excel workbooks with topology data
    ↓
[4] Feature Extraction
    ├─ Compute morphological features (branching angles, vessel area)
    ├─ Calculate geometric properties (vascular density, fractal dimension)
    ├─ Compute topological metrics (Betti numbers)
    └─ Generate per-image feature workbooks
    ↓
[5] Statistical Analysis
    ├─ Analyze longitudinal vascular development (P2-P7)
    ├─ Compare predicted vs. ground truth segmentations
    ├─ Statistical tests and hypothesis testing
    └─ Generate analysis reports and visualizations
```

---

## ⚡ Quick Start Guide

For experienced users who want to run the pipeline quickly:

```bash
# 1. Set up environment
conda activate nnunet_env

# 2. Run EDA
cd scripts/00-EDA
jupyter lab EDA.ipynb

# 3. Preprocess data
cd ../01-segmentation/preprocess
python preprocess.py

# 4. Set up nnU-Net environment variables
export nnUNet_raw="$(pwd)/../nnUnet/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/../nnUnet/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/../nnUnet/nnUNet_results"

# 5. Run segmentation pipeline
cd ../nnUnet
python -m nnunetv2.nnunetv2_plan_and_preprocess -d 001 -pl nnUNetPlannerResEncL -gpu
python -m nnunetv2.run.run_training 001 2d 0 -tr nnUNetTrainer

# 6. Run reconstruction
cd ../../02-reconstruction
python vessel_reconstruction.py

# 7. Run feature extraction
cd ../../03-feature-extraction
python feature_extraction.py

# 8. Run statistical analysis
cd ../../04-statistical-analysis
python statistical_analysis.py
```

---

## 📝 Detailed Step-by-Step Instructions

### Step 0: Exploratory Data Analysis (EDA)

**Purpose**: Understand the dataset properties, distribution, and quality

**Location**: `scripts/00-EDA/`

#### Files

- `EDA.ipynb` - Main EDA notebook
- `README.md` - Detailed EDA documentation

#### Running the EDA

```bash
cd scripts/00-EDA

# Option 1: Using Jupyter Lab
jupyter lab EDA.ipynb

# Option 2: Using Jupyter Notebook
jupyter notebook EDA.ipynb
```

#### What EDA Analyzes

1. **Image Properties**:
   - Shape, data type, min/max values
   - Intensity distributions
   - Spatial statistics

2. **Mask Analysis**:
   - Mask pixel value distributions
   - Connected component analysis
   - Vessel area statistics

3. **Preprocessing Analysis**:
   - Effect of different preprocessing techniques
   - Histogram equalization analysis
   - Contrast normalization

4. **Patch Analysis**:
   - Training patch distribution
   - Patch size statistics

**Outputs**: Analysis report in notebook, visualizations

---

### Step 1: Image Segmentation

The segmentation pipeline uses nnU-Net, a self-configuring deep learning framework for medical image segmentation.

#### 1.1 Data Preprocessing

**Location**: `scripts/01-segmentation/preprocess/`

First, convert raw images to nnU-Net format:

```bash
cd scripts/01-segmentation/preprocess

# Run the preprocessing script
python preprocess.py
```

**What it does**:
- Converts RGB images to grayscale
- Standardizes file format
- Organizes into nnU-Net dataset structure
- Creates `dataset.json` configuration file

**Output**: Preprocessed data in `../nnUnet/nnUNet_raw/Dataset001_vessel/`

**Configuration in preprocess.py**:

```python
BASE_PATH = "/path/to/STOR674-Final-Project"
DATASET_NAME = "Dataset001_vessel"
```

Edit these paths if your data is in a different location.

#### 1.2 nnU-Net Setup & Configuration

**Location**: `scripts/01-segmentation/nnUnet/`

**Setup Steps**:

```bash
cd scripts/01-segmentation/nnUnet

# Set environment variables
export nnUNet_raw="$(pwd)/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"

# Verify nnU-Net installation
python -c "import nnunetv2; print('nnU-Net v2 ready')"
```

**Reference**: See `nnUNet_setup_guide.md` for detailed setup instructions

#### 1.3 Model Planning & Preprocessing

```bash
# Plan dataset and preprocess
python -m nnunetv2.nnunetv2_plan_and_preprocess -d 001 -pl nnUNetPlannerResEncL -gpu

# Parameters:
# -d 001      : Dataset ID
# -pl         : Planner (ResEncL for reasonable computational cost)
# -gpu        : Use GPU acceleration
```

**What it does**:
- Analyzes dataset properties
- Creates preprocessing plan
- Preprocesses all images
- Generates fingerprint statistics

**Output**: Preprocessed data in `nnUNet_preprocessed/Dataset001_vessel/`

#### 1.4 Model Training

```bash
# Train the model (2D U-Net variant for vessel segmentation)
python -m nnunetv2.run.run_training 001 2d 0 -tr nnUNetTrainer

# Parameters:
# 001          : Dataset ID
# 2d           : 2D model (appropriate for slice-by-slice images)
# 0            : Fold 0 (single fold for small datasets)
# -tr          : Trainer class
```

**Training Tips**:
- Training time depends on GPU (typically 2-24 hours)
- Monitor with tensorboard: `tensorboard --logdir nnUNet_results`
- Checkpoints saved in `nnUNet_results/Dataset001_vessel/`

**Output**: Trained model weights saved in results directory

#### 1.5 Model Evaluation

```bash
# Evaluate segmentation metrics
python evaluate_segmentation_metrics.py

# This computes:
# - Dice coefficient
# - Intersection over Union (IoU)
# - Hausdorff distance
# - Sensitivity and Specificity
```

**Output**: Evaluation metrics report

---

### Step 2: Vascular Network Reconstruction

**Purpose**: Convert binary segmentation masks into vascular network graphs (nodes, edges, topology)

**Location**: `scripts/02-reconstruction/`

**Files**:
- `Vessel network reconstruction.md` - Reconstruction guide
- `note.txt` - Implementation notes

#### Reconstruction Process

```bash
cd scripts/02-reconstruction

# Run reconstruction (command depends on implementation)
python reconstruct_vessel_network.py
```

#### What Reconstruction Produces

For each image, generates Excel workbooks:

1. **`image_name_alldata.xlsx`**:
   - Node table: (x, y, degree, type, order, ...)
   - Edge table: (source, target, length, width, ...)
   - Full topology information

2. **`image_name_degreedata.xlsx`**:
   - Degree distribution statistics
   - Node type classification
   - Branching patterns

#### Expected Output Structure

```
Data/validation_data/feature_extraction/feature/p2-p7/
├── vessel_p2_alldata.xlsx
├── vessel_p2_degreedata.xlsx
├── vessel_p3_alldata.xlsx
├── vessel_p3_degreedata.xlsx
... (P4-P7 similarly)
└── vessel_p7x_alldata.xlsx
```

**Key Metrics**:
- **Node Degree**: Number of connections per junction
- **Vessel Segments**: Connections between junctions
- **Network Topology**: Overall structure and organization

---

### Step 3: Feature Extraction

**Purpose**: Compute quantitative morphological, geometric, and topological features from reconstructed networks

**Location**: `scripts/03-feature-extraction/`

**Files**:
- `feature_extraction.py` - Main feature computation
- `_utility.py` - Helper functions
- `Network_feature_extraction.md` - Detailed documentation

#### Feature Categories

##### 1. Morphological Features
- **Branching Angles**: Junction angle distributions
- **Vessel Area**: Regional area statistics
- **Vessel Length**: Total and segmental lengths

##### 2. Geometric Features
- **Vascular Density**: 
  - Area density (vessel area / total area)
  - Length density (vessel length / total length)
- **Fractal Dimension**: Box-counting complexity analysis

##### 3. Topological Features
- **Betti Numbers**: 
  - β₀ (Components): Number of disconnected parts
  - β₁ (Loops): Number of independent cycles
  - β₂ (Voids): Number of enclosed volumes

#### Running Feature Extraction

```bash
cd scripts/03-feature-extraction

# Run feature extraction
python feature_extraction.py

# Configuration in feature_extraction.py:
# - Input directory: Data/validation_data/feature_extraction/feature/p2-p7/
# - Output directory: same location
```

#### Output

For each image, generates feature Excel workbooks:

```
vessel_p2_features.xlsx
├── Morphological Features
├── Geometric Features
├── Topological Features
└── Summary Statistics
```

**Key Features Computed**:

| Feature | Description | Units |
|---------|-------------|-------|
| Branching Angle | Mean angle at junctions | degrees |
| Vascular Density | Vessel area per unit tissue | % |
| Fractal Dimension | Network complexity | 1-2 |
| Betti Number β₀ | Network components | count |
| Betti Number β₁ | Network loops | count |

---

### Step 4: Statistical Analysis

**Purpose**: Analyze longitudinal vascular development and compare predicted vs. ground truth

**Location**: `scripts/04-statistical-analysis/`

**Files**:
- `statistical_analysis.py` - Main analysis script
- `config.py` - Configuration and paths
- `test_statistical_analysis.py` - Unit tests
- `requirements.txt` - Analysis dependencies

#### Setup

```bash
cd scripts/04-statistical-analysis

# Install analysis-specific dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas; import scipy; print('Ready for analysis')"
```

#### Running Analysis

```bash
# Run full statistical analysis
python statistical_analysis.py

# Run with tests
pytest test_statistical_analysis.py -v
```

#### Analysis Components

1. **Longitudinal Analysis (P2-P7)**:
   - Temporal trends in vascular features
   - Growth rates and developmental patterns
   - Statistical significance tests

2. **Predicted vs. Ground Truth Comparison**:
   - Segmentation accuracy metrics
   - Agreement analysis (Dice, IoU)
   - Systematic differences

3. **Statistical Tests**:
   - ANOVA for multi-timepoint comparisons
   - Paired t-tests for predicted vs. ground truth
   - Correlation analysis between features

4. **Visualizations**:
   - Time-series plots (P2-P7)
   - Scatter plots and regression analysis
   - Heatmaps and distribution plots

#### Configuration

Edit `config.py` to customize:

```python
# Paths
BASE_PATH = "/path/to/STOR674-Final-Project"
DATA_PATH = os.path.join(BASE_PATH, "Data/validation_data/feature_extraction/feature/p2-p7")
OUTPUT_PATH = os.path.join(BASE_PATH, "results/statistical_analysis")

# Analysis parameters
TIMEPOINTS = ['P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P7x']
SIGNIFICANCE_LEVEL = 0.05
```

#### Expected Outputs

```
results/statistical_analysis/
├── analysis_report.txt          # Summary statistics
├── developmental_trends.png     # Time-series visualizations
├── comparison_metrics.csv       # Numerical results
├── statistical_tests_results.txt # P-values and test statistics
└── feature_heatmaps.png         # Feature correlation heatmaps
```

---

## 📁 File Structure

```
STOR674-Final-Project/
├── README.md                          # This file
├── BUILD.md                          # Singularity build instructions
├── environment.yml                   # Conda environment file
├── requirements.txt                  # pip requirements (all dependencies)
├── requirements_full.txt             # Extended requirements
├── Singularity.def                   # Singularity container definition
│
├── Data/
│   ├── train_data/
│   │   ├── image/                   # Training images
│   │   └── ground_truth/            # Training masks
│   ├── validation_data/
│   │   ├── image/                   # Validation images
│   │   ├── ground_truth/            # Validation masks
│   │   ├── feature_extraction/      # Intermediate outputs
│   │   │   ├── __init__.py
│   │   │   ├── _utility.py
│   │   │   ├── feature_extraction.py
│   │   │   ├── EXTRACTED_FEATURES_GUIDE.md
│   │   │   ├── connectivity_matrix_test.py
│   │   │   └── feature/
│   │   │       ├── N_129/           # Sample outputs
│   │   │       └── p2-p7/           # P2-P7 timepoint data
│   │   └── img/                     # Visualizations
│   └── test_data/                   # Test set (optional)
│
├── scripts/
│   ├── 00-EDA/
│   │   ├── README.md                # EDA documentation
│   │   └── EDA.ipynb                # Exploratory data analysis
│   │
│   ├── 01-segmentation/
│   │   ├── nnUNet_setup_guide.md    # nnU-Net guide
│   │   ├── preprocess/
│   │   │   ├── README.md
│   │   │   ├── preprocess.py        # Data preprocessing script
│   │   │   └── preprocess_test_data.py
│   │   └── nnUnet/
│   │       ├── setup.py
│   │       ├── pyproject.toml
│   │       ├── evaluate_segmentation_metrics.py
│   │       ├── nnUNet_raw/          # Raw data for training
│   │       ├── nnUNet_preprocessed/ # Preprocessed data
│   │       ├── nnUNet_results/      # Model checkpoints
│   │       └── nnunetv2/            # nnU-Net source code
│   │
│   ├── 02-reconstruction/
│   │   ├── Vessel network reconstruction.md
│   │   └── note.txt
│   │
│   ├── 03-feature-extraction/
│   │   ├── Network_feature_extraction.md
│   │   ├── feature_extraction.py
│   │   ├── _utility.py
│   │   └── note.txt
│   │
│   └── 04-statistical-analysis/
│       ├── README.md
│       ├── config.py                # Configuration file
│       ├── statistical_analysis.py  # Main analysis script
│       ├── test_statistical_analysis.py  # Unit tests
│       └── requirements.txt         # Analysis dependencies
│
└── results/
    └── statistical_analysis/
        ├── analysis_report.txt      # Summary statistics
        └── [Analysis outputs]
```

---

## 📦 Requirements & Dependencies

### Core Dependencies

#### Python Version
- **Python 3.10+** (tested with 3.10, 3.12)

#### Deep Learning & Image Processing
- `torch>=2.9.0` - PyTorch framework
- `nnunetv2>=2.6.2` - nnU-Net segmentation framework
- `torchvision>=0.24.0` - Vision utilities
- `nibabel>=5.3.2` - Neuroimaging I/O
- `SimpleITK>=2.5.2` - Medical image processing
- `scikit-image>=0.24.0` - Image processing algorithms
- `imageio>=2.33.1` - Image I/O

#### Scientific Computing
- `numpy>=1.26.4` - Numerical computing
- `scipy>=1.13.1` - Scientific functions
- `pandas>=2.2.2` - Data manipulation

#### Visualization
- `matplotlib>=3.9.2` - Plotting
- `seaborn>=0.13.2` - Statistical visualization
- `napari>=0.6.2` - Image viewer

#### Utilities
- `tqdm>=4.66.5` - Progress bars
- `h5py>=3.11.0` - HDF5 file handling
- `pyyaml>=6.0.1` - YAML configuration
- `einops>=0.8.1` - Tensor operations

### Installation Methods

See [Installation & Setup](#-installation--setup) section above for detailed instructions.

---

## ⚠️ Troubleshooting

### Common Issues & Solutions

#### 1. CUDA/GPU Issues

**Problem**: `RuntimeError: CUDA out of memory`

```bash
# Solution: Reduce batch size in config
# Edit nnU-Net configuration to use smaller batch sizes
# Or run on CPU (slower):
python -m nnunetv2.run.run_training 001 2d 0 -tr nnUNetTrainer --device cpu
```

**Problem**: `torch.cuda.is_available()` returns False

```bash
# Solution: Install correct CUDA-compatible PyTorch
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu121

# Verify:
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

#### 2. nnU-Net Setup Issues

**Problem**: `ModuleNotFoundError: No module named 'nnunetv2'`

```bash
# Solution: Install nnU-Net
pip install nnunetv2==2.6.2

# Or install from source:
cd scripts/01-segmentation/nnUnet
pip install -e .
```

**Problem**: Environment variables not set

```bash
# Solution: Set nnU-Net paths
export nnUNet_raw="$(pwd)/scripts/01-segmentation/nnUnet/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/scripts/01-segmentation/nnUnet/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/scripts/01-segmentation/nnUnet/nnUNet_results"

# Verify:
python -c "import os; print(os.environ.get('nnUNet_raw'))"
```

#### 3. Data Path Issues

**Problem**: `FileNotFoundError: Data directory not found`

```bash
# Solution: Update paths in config files
# Edit preprocess.py:
BASE_PATH = "/correct/path/to/STOR674-Final-Project"

# Or set as environment variable:
export STOR674_PATH="/path/to/STOR674-Final-Project"
```

#### 4. Memory Issues

**Problem**: Out of memory during training

```bash
# Solution 1: Use smaller model configuration
# Edit nnU-Net configuration

# Solution 2: Use gradient accumulation
# Reduce batch size

# Solution 3: Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 5. Singularity Issues

**Problem**: `FATAL: could not build Singularity image`

```bash
# Solution 1: Use remote build (recommended)
singularity remote login
singularity build --remote stor674.sif Singularity.def

# Solution 2: Use sudo
sudo singularity build stor674.sif Singularity.def

# Solution 3: Use fakeroot (if configured)
singularity build --fakeroot stor674.sif Singularity.def
```

#### 6. Jupyter Issues

**Problem**: `jupyter: command not found`

```bash
# Solution: Install Jupyter
pip install jupyter jupyterlab

# Then run:
jupyter lab
```

### Getting Help

1. **Check log files**: Most scripts generate detailed logs
2. **Run tests**: Use `pytest` to identify issues
3. **Check documentation**: See `README.md` files in each `scripts/` subdirectory
4. **Verify data paths**: Ensure all input data exists and is accessible

---

## 📊 Results & Outputs

### Output Directory Structure

```
results/
├── statistical_analysis/
│   ├── analysis_report.txt
│   ├── developmental_trends.png
│   ├── comparison_metrics.csv
│   └── statistical_tests_results.txt
│
└── [Other analysis outputs]
```

### Key Results

1. **Segmentation Results**:
   - Binary vessel masks for each image
   - Segmentation metrics (Dice, IoU, Hausdorff distance)
   - Comparison with ground truth

2. **Network Reconstruction Results**:
   - Node-edge graphs for each image
   - Topology statistics
   - Excel workbooks with connectivity matrices

3. **Feature Extraction Results**:
   - Morphological features (branching angles, vessel areas)
   - Geometric features (vascular density, fractal dimension)
   - Topological features (Betti numbers)

4. **Statistical Analysis Results**:
   - Developmental trends (P2-P7 vascular growth)
   - Segmentation accuracy comparison
   - Visualizations and summary statistics

---

## 📚 References

### Key Papers

1. **nnU-Net Framework**:
   - Isensee, F., & Maier-Hein, K. H. (2019). An attempt at beating the 3D U-Net. arXiv preprint arXiv:1908.02182.
   - Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

2. **Medical Image Segmentation**:
   - Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. MICCAI 2015.

3. **Network Analysis**:
   - Newman, M. E. (2003). The structure and function of complex networks. SIAM review, 45(2), 167-256.

### External Resources

- [nnU-Net GitHub Repository](https://github.com/MIC-DKFZ/nnUNet)
- [nnU-Net Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

## 🤝 Contributing



---

## ✅ Checklist for Reproducing the Project

Use this checklist to ensure you've completed all steps:

- [ ] Environment set up (Conda or Singularity)
- [ ] Dependencies installed and verified
- [ ] Data structure validated
- [ ] EDA completed (Step 0)
- [ ] Data preprocessing completed (Step 1.1)
- [ ] nnU-Net setup and environment variables configured (Step 1.2)
- [ ] Model planning and preprocessing completed (Step 1.3)
- [ ] Model training completed (Step 1.4)
- [ ] Model evaluation completed (Step 1.5)
- [ ] Vascular reconstruction completed (Step 2)
- [ ] Feature extraction completed (Step 3)
- [ ] Statistical analysis completed (Step 4)
- [ ] Results reviewed and saved (Step 5)

---

For questions or issues:
1. Check the troubleshooting section
2. Review relevant README files in each `scripts/` subdirectory
3. Examine log files for detailed error messages
   
*For questions or bug reports, please refer to the troubleshooting section or check individual step READMEs.*
