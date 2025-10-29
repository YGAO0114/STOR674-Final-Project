# EDA for Blood Vessel Image Analysis

This file contains:  
1. load and check the image
2. EDA: check the training image properties:  
 - a. shape, data type, min and max values ...  
 - b. visualization  
 - c. check image intensity distribution  
 - d. Spatial Analysis by Rings  
 - e. different preprocessing  
 - f. iamge mask analysis: Mask dtype, Pixel value distribution ...  
 - g. Patch Analysis for Training  
3. data convertion: specific for validation mask image  



## Quick Start

```python
# Set up paths
BASE_PATH = "/path/to/your/dataset"
paths = setup_paths(BASE_PATH)

# some function usage examples: 
# Analyze dataset structure
analyze_dataset_structure(paths)

# Get image lists
train_images = get_image_list(paths['train_images'])
train_masks = get_image_list(paths['train_masks'])

# Visualize samples
visualize_samples(train_images, train_masks, n_samples=4)

# Analyze image properties
props_df = analyze_image_properties(train_images[:5])
print(props_df)
```

## Package Structure

### Some Functions

#### Data Utilities 
- `setup_paths(base_path)`: Set up dataset paths
- `load_image(path)`: Load images in various formats
- `get_image_list(folder_path, extensions)`: Get list of image files
- `analyze_dataset_structure(paths)`: Analyze dataset structure

#### Analysis 
- `analyze_image_properties(image_list)`: Analyze basic image properties
- `analyze_mask_detailed(mask_path)`: Comprehensive mask analysis
- `analyze_spatial_distribution(mask_paths)`: Spatial distribution analysis
- `analyze_patch_distribution(image_path, mask_path)`: Patch analysis for training

#### Visualization 
- `visualize_samples(image_paths, mask_paths)`: Visualize image samples
- `visualize_intensity_distribution(image_paths)`: Plot intensity distributions

#### Metrics 
- `dice_score(y_true, y_pred)`: Calculate Dice score


## Dataset Structure Expected

The package expects the following dataset structure:

```
base_path/
├── Data/
│   ├── train_data/
│   │   ├── image/
│   │   └── ground_truth/
│   ├── validation_data/
│   │   ├── image/
│   │   └── ground_truth/
│   └── test_data/
```
