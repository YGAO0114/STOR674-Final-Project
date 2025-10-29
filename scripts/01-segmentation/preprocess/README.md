# Preprocessing for Blood Vessel Segmentation

This folder contains preprocessing scripts to prepare blood vessel image data for nnU-Net training.

## Overview

The preprocessing pipeline converts raw blood vessel images and ground truth masks into the nnU-Net format required for training deep learning models. The main script handles:

- **Image Conversion**: RGB to grayscale conversion for consistent input format
- **Format Standardization**: Converting to nnU-Net naming conventions
- **Data Organization**: Separating training and validation data into proper directories

## Files

- `preprocess.py` - Main preprocessing script for data conversion

## Dataset Structure

### Input Structure
The script expects the following input directory structure:
```
STOR674/
├── Data/
│   ├── train_data/
│   │   ├── image/          # Training RGB images (.tif/.tiff)
│   │   └── ground_truth/   # Training ground truth masks (.tif/.tiff)
│   └── validation_data/
│       ├── image/          # Validation RGB images (.tif/.tiff)
│       └── ground_truth/   # Validation ground truth masks (.tif/.tiff)
```

### Output Structure
The script creates nnU-Net compatible structure:
```
STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw/Dataset001_vessel/
├── imagesTr/    # Training images (vessel_000_0000.tif, vessel_001_0000.tif, ...)
├── imagesTs/    # Validation images (vessel_001_0000.tif, vessel_002_0000.tif, ...)
├── labelsTr/    # Training labels (vessel_000.tif, vessel_001.tif, ...)
└── labelsTs/    # Validation labels (vessel_001.tif, vessel_002.tif, ...)
```

## Usage

### Prerequisites
- Python 3.x
- PIL (Pillow)
- Required directories must exist with input data

### Running the Preprocessing

```bash
cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/preprocess
python preprocess.py
```

### What the Script Does

1. **Image Processing**:
   - Loads RGB images from training and validation directories
   - Converts RGB images to grayscale using PIL
   - Saves with nnU-Net naming convention: `vessel_XXX_0000.tif`

2. **Label Processing**:
   - Copies ground truth mask files
   - Renames to nnU-Net format: `vessel_XXX.tif`
   - Maintains original mask values and properties

3. **Directory Management**:
   - Creates output directories if they don't exist
   - Organizes data into nnU-Net required structure
   - Provides detailed logging of conversion process

### Naming Convention

- **Images**: `vessel_XXX_0000.tif` where XXX is zero-padded case number
- **Labels**: `vessel_XXX.tif` where XXX matches corresponding image
- **Indexing**: Training starts at 000, validation starts at 001

## Output

The script provides detailed console output including:
- Conversion progress for each file
- Count of processed images and labels
- Final verification of output directories
- Error messages for any failed conversions

### Example Output
```
=== Converting Training Images ===
Converted: image_001.tif -> vessel_000_0000.tif
Converted: image_002.tif -> vessel_001_0000.tif
...

=== Conversion Complete ===
Training images converted: 14
Training labels copied: 14
Validation images converted: 4
Validation labels copied: 4
Total images processed: 18
Total labels processed: 18
```

## Integration with nnU-Net

This preprocessing step is the first in the nnU-Net pipeline:
1. **Preprocessing** (this script) → Raw data conversion
2. **Planning** → `nnUNetv2_plan_and_preprocess`
3. **Training** → `nnUNetv2_train`

The converted data is ready for nnU-Net's automatic preprocessing and training phases.
