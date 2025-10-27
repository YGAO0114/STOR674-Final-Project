#!/usr/bin/env python3
"""
Comprehensive Segmentation Metrics Calculator for nnUNet Results

This script calculates various segmentation metrics including:
- DICE Score
- Hausdorff Distance (HD)
- Mean Surface Distance (MSD)
- Jaccard Index (IoU)
- Sensitivity, Specificity, Precision
- Volume Similarity
- And other important metrics

Supports TIFF, NIfTI, and other common medical image formats.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Image loading libraries
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available. NIfTI support disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. TIFF support may be limited.")

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    print("Warning: tifffile not available. Using PIL for TIFF files.")

# Scientific computing libraries
try:
    from scipy import ndimage
    from scipy.spatial.distance import directed_hausdorff, cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Error: scipy is required for surface distance calculations.")
    sys.exit(1)

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Using alternative surface extraction.")

class SegmentationMetricsCalculator:
    """Comprehensive segmentation metrics calculator."""
    
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
        
    def load_image(self, file_path):
        """Load image from various formats (TIFF, NIfTI, etc.)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file format and load accordingly
        if file_path.suffix.lower() in ['.tif', '.tiff']:
            return self._load_tiff(file_path)
        elif file_path.suffix.lower() in ['.nii', '.nii.gz']:
            return self._load_nifti(file_path)
        else:
            # Try PIL as fallback
            try:
                img = Image.open(file_path)
                return np.array(img)
            except Exception as e:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_tiff(self, file_path):
        """Load TIFF file."""
        if TIFFFILE_AVAILABLE:
            try:
                return tifffile.imread(file_path)
            except Exception as e:
                print(f"tifffile failed, trying PIL: {e}")
        
        if PIL_AVAILABLE:
            try:
                img = Image.open(file_path)
                return np.array(img)
            except Exception as e:
                raise ValueError(f"Failed to load TIFF with PIL: {e}")
        else:
            raise ValueError("No TIFF loading library available")
    
    def _load_nifti(self, file_path):
        """Load NIfTI file."""
        if not NIBABEL_AVAILABLE:
            raise ValueError("nibabel not available for NIfTI loading")
        
        try:
            img = nib.load(file_path)
            return img.get_fdata()
        except Exception as e:
            raise ValueError(f"Failed to load NIfTI: {e}")
    
    def dice_score(self, pred, target):
        """Calculate DICE score between prediction and ground truth."""
        pred = pred.astype(np.bool_)
        target = target.astype(np.bool_)
        
        intersection = np.sum(pred & target)
        union = np.sum(pred) + np.sum(target)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice
    
    def jaccard_score(self, pred, target):
        """Calculate Jaccard Index (IoU)."""
        pred = pred.astype(np.bool_)
        target = target.astype(np.bool_)
        
        intersection = np.sum(pred & target)
        union = np.sum(pred | target)
        
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        return jaccard
    
    def sensitivity(self, pred, target):
        """Calculate Sensitivity (Recall/True Positive Rate)."""
        pred = pred.astype(np.bool_)
        target = target.astype(np.bool_)
        
        tp = np.sum(pred & target)
        fn = np.sum(~pred & target)
        
        sensitivity = (tp + self.smooth) / (tp + fn + self.smooth)
        return sensitivity
    
    def specificity(self, pred, target):
        """Calculate Specificity (True Negative Rate)."""
        pred = pred.astype(np.bool_)
        target = target.astype(np.bool_)
        
        tn = np.sum(~pred & ~target)
        fp = np.sum(pred & ~target)
        
        specificity = (tn + self.smooth) / (tn + fp + self.smooth)
        return specificity
    
    def precision_score(self, pred, target):
        """Calculate Precision (Positive Predictive Value)."""
        pred = pred.astype(np.bool_)
        target = target.astype(np.bool_)
        
        tp = np.sum(pred & target)
        fp = np.sum(pred & ~target)
        
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        return precision
    
    def f1_score(self, pred, target):
        """Calculate F1 Score."""
        precision = self.precision_score(pred, target)
        sensitivity = self.sensitivity(pred, target)
        
        f1 = (2 * precision * sensitivity + self.smooth) / (precision + sensitivity + self.smooth)
        return f1
    
    def volume_similarity(self, pred, target):
        """Calculate Volume Similarity."""
        pred_vol = np.sum(pred)
        target_vol = np.sum(target)
        
        vs = 1 - abs(pred_vol - target_vol) / (pred_vol + target_vol + self.smooth)
        return vs
    
    def get_surface_points(self, volume):
        """Extract surface points from a binary volume."""
        if SKIMAGE_AVAILABLE:
            try:
                # For 2D images, use contour detection
                if len(volume.shape) == 2:
                    contours = measure.find_contours(volume.astype(float), level=0.5)
                    if len(contours) > 0:
                        # Combine all contours
                        all_points = np.vstack(contours)
                        return all_points
                    else:
                        return np.array([]).reshape(0, 2)
                else:
                    # For 3D images, use marching cubes
                    verts, faces, _, _ = measure.marching_cubes(volume, level=0.5, spacing=(1, 1, 1))
                    return verts
            except Exception as e:
                print(f"skimage surface extraction failed: {e}")
        
        # Fallback: use edge detection
        if len(volume.shape) == 2:
            edges = ndimage.binary_erosion(volume) != volume
            coords = np.where(edges)
            return np.column_stack(coords)
        else:
            edges = ndimage.binary_erosion(volume) != volume
            coords = np.where(edges)
            return np.column_stack(coords)
    
    def hausdorff_distance(self, pred, target):
        """Calculate Hausdorff Distance between two binary volumes."""
        pred = pred.astype(np.bool_)
        target = target.astype(np.bool_)
        
        # If either volume is empty, return infinity
        if not np.any(pred) or not np.any(target):
            return np.inf
        
        # Get surface points
        pred_surface = self.get_surface_points(pred.astype(np.float32))
        target_surface = self.get_surface_points(target.astype(np.float32))
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            return np.inf
        
        try:
            # Calculate directed Hausdorff distances
            hd_1 = directed_hausdorff(pred_surface, target_surface)[0]
            hd_2 = directed_hausdorff(target_surface, pred_surface)[0]
            
            # Return the maximum (symmetric Hausdorff distance)
            return max(hd_1, hd_2)
        except Exception as e:
            print(f"Hausdorff calculation failed: {e}")
            return np.inf
    
    def mean_surface_distance(self, pred, target):
        """Calculate Mean Surface Distance between two binary volumes."""
        pred = pred.astype(np.bool_)
        target = target.astype(np.bool_)
        
        # If either volume is empty, return infinity
        if not np.any(pred) or not np.any(target):
            return np.inf
        
        # Get surface points
        pred_surface = self.get_surface_points(pred.astype(np.float32))
        target_surface = self.get_surface_points(target.astype(np.float32))
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            return np.inf
        
        try:
            # Calculate distances from pred surface to target surface
            distances_1 = np.min(cdist(pred_surface, target_surface), axis=1)
            distances_2 = np.min(cdist(target_surface, pred_surface), axis=1)
            
            # Mean surface distance
            msd = (np.mean(distances_1) + np.mean(distances_2)) / 2
            return msd
        except Exception as e:
            print(f"Mean surface distance calculation failed: {e}")
            return np.inf
    
    def calculate_all_metrics(self, pred_path, target_path):
        """Calculate all metrics for a single prediction-target pair."""
        try:
            # Load images
            pred_data = self.load_image(pred_path)
            target_data = self.load_image(target_path)
            
            # Ensure same shape
            if pred_data.shape != target_data.shape:
                print(f"Warning: Shape mismatch - pred: {pred_data.shape}, target: {target_data.shape}")
                # Resize to match the smaller dimensions
                min_shape = tuple(min(pred_data.shape[i], target_data.shape[i]) for i in range(len(pred_data.shape)))
                pred_data = pred_data[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else pred_data[:min_shape[0], :min_shape[1], :min_shape[2]]
                target_data = target_data[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else target_data[:min_shape[0], :min_shape[1], :min_shape[2]]
            
            # Convert to binary if needed
            if pred_data.max() > 1:
                pred_data = (pred_data > 0).astype(np.uint8)
            if target_data.max() > 1:
                target_data = (target_data > 0).astype(np.uint8)
            
            # Calculate all metrics
            metrics = {
                'DICE': self.dice_score(pred_data, target_data),
                'Jaccard': self.jaccard_score(pred_data, target_data),
                'Sensitivity': self.sensitivity(pred_data, target_data),
                'Specificity': self.specificity(pred_data, target_data),
                'Precision': self.precision_score(pred_data, target_data),
                'F1_Score': self.f1_score(pred_data, target_data),
                'Volume_Similarity': self.volume_similarity(pred_data, target_data),
                'Hausdorff_Distance': self.hausdorff_distance(pred_data, target_data),
                'Mean_Surface_Distance': self.mean_surface_distance(pred_data, target_data)
            }
            
            return metrics, pred_data, target_data
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None, None, None

def print_metrics_summary(metrics):
    """Print a formatted summary of metrics."""
    print("\n" + "="*80)
    print("SEGMENTATION METRICS SUMMARY")
    print("="*80)
    
    # Basic overlap metrics
    print(f"OVERLAP METRICS:")
    print(f"   DICE Score:           {metrics['DICE']:.4f}")
    print(f"   Jaccard Index (IoU):  {metrics['Jaccard']:.4f}")
    print(f"   Volume Similarity:    {metrics['Volume_Similarity']:.4f}")
    
    # Classification metrics
    print(f"\nCLASSIFICATION METRICS:")
    print(f"   Sensitivity (Recall): {metrics['Sensitivity']:.4f}")
    print(f"   Specificity:          {metrics['Specificity']:.4f}")
    print(f"   Precision:            {metrics['Precision']:.4f}")
    print(f"   F1 Score:             {metrics['F1_Score']:.4f}")
    
    # Distance metrics
    print(f"\nDISTANCE METRICS:")
    hd = metrics['Hausdorff_Distance']
    msd = metrics['Mean_Surface_Distance']
    
    if np.isfinite(hd):
        print(f"   Hausdorff Distance:   {hd:.2f} pixels")
    else:
        print(f"   Hausdorff Distance:    ∞ (infinite)")
    
    if np.isfinite(msd):
        print(f"   Mean Surface Distance: {msd:.2f} pixels")
    else:
        print(f"   Mean Surface Distance: ∞ (infinite)")

def main():
    parser = argparse.ArgumentParser(description='Calculate comprehensive segmentation metrics')
    parser.add_argument('--pred', type=str, required=True, 
                       help='Path to prediction file')
    parser.add_argument('--target', type=str, required=True,
                       help='Path to ground truth file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results CSV file')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = SegmentationMetricsCalculator()
    

    print(f"Prediction file: {args.pred}")
    print(f"Ground truth file: {args.target}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics, pred_data, target_data = calculator.calculate_all_metrics(args.pred, args.target)
    
    if metrics is None:
        print("Failed to calculate metrics!")
        return 1
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save results if requested
    if args.output:
        df = pd.DataFrame([metrics])
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    
    # Print detailed info if verbose
    if args.verbose and pred_data is not None and target_data is not None:
        print(f"\nDETAILED INFORMATION:")
        print(f"   Prediction shape: {pred_data.shape}")
        print(f"   Target shape:      {target_data.shape}")
        print(f"   Prediction range:  [{pred_data.min()}, {pred_data.max()}]")
        print(f"   Target range:     [{target_data.min()}, {target_data.max()}]")
        print(f"   Prediction volume: {np.sum(pred_data)} pixels")
        print(f"   Target volume:     {np.sum(target_data)} pixels")
    
    return 0

if __name__ == "__main__":
    if len(sys.argv) == 1:
        
        pred_file = "/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results/Dataset001_vessel/nnUNetTrainer_50epochs__nnUNetResEncUNetLPlans__2d/predictions/vessel_001.tif"
        target_file = "/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw/Dataset001_vessel/labelsTs/vessel_001.tif"
        
        # Check if files exist
        if not os.path.exists(pred_file):
            print(f"Prediction file not found: {pred_file}")
            sys.exit(1)
        
        if not os.path.exists(target_file):
            print(f"Target file not found: {target_file}")
            sys.exit(1)
        
        # Initialize calculator
        calculator = SegmentationMetricsCalculator()
        
        print(f"Prediction file: {pred_file}")
        print(f"Ground truth file: {target_file}")
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics, pred_data, target_data = calculator.calculate_all_metrics(pred_file, target_file)
        
        if metrics is None:
            print("Failed to calculate metrics!")
            sys.exit(1)
        
        # Print summary
        print_metrics_summary(metrics)
        
        # Save results
        output_file = "/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/vessel_001_metrics.csv"
        df = pd.DataFrame([metrics])
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Print detailed info
        print(f"\nDETAILED INFORMATION:")
        print(f"   Prediction shape: {pred_data.shape}")
        print(f"   Target shape:      {target_data.shape}")
        print(f"   Prediction range:  [{pred_data.min()}, {pred_data.max()}]")
        print(f"   Target range:     [{target_data.min()}, {target_data.max()}]")
        print(f"   Prediction volume: {np.sum(pred_data)} pixels")
        print(f"   Target volume:     {np.sum(target_data)} pixels")
        
    else:
        sys.exit(main())
