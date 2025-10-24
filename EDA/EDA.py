
# # Blood Vessel Segmentation - Exploratory Data Analysis
# 
# This notebook performs comprehensive EDA on mouse retinal blood vessel images to understand the data characteristics before building a segmentation model.

# ## 1. Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
from pathlib import Path
from skimage import io, exposure, morphology, measure
from skimage.filters import threshold_otsu, gaussian
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Set visualization parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# ## 2. Define Data Paths and Load Functions

def setup_paths(base_path):
    """Set up all necessary paths for the project"""
    paths = {
        'base': Path(base_path),
        'train_images': Path(base_path) / 'Data' / 'train_data' / 'image',
        'train_masks': Path(base_path) / 'Data' / 'train_data' / 'ground truth',
        'val_images': Path(base_path) / 'Data' / 'validation_data' / 'image',
        'val_masks': Path(base_path) / 'Data' / 'validation_data' ,
        'test_images': Path(base_path) / 'Data' / 'test_data' 
    }
    return paths

def load_image(path):
    """Load an image and return as numpy array"""
    if path.suffix.lower() in ['.tif', '.tiff']:
        img = io.imread(str(path))
    else:
        img = np.array(Image.open(path))
    return img

def get_image_list(folder_path, extensions=['.tif', '.tiff', '.png', '.jpg']):
    """Get list of image files in a folder"""
    image_files = []
    if folder_path.exists():
        for ext in extensions:
            image_files.extend(list(folder_path.glob(f'*{ext}')))
    return sorted(image_files)

# ## 3. Data Overview and Basic Statistics