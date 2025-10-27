#!/usr/bin/env python3
"""
Simple DICE Score Calculator for nnUNet Results

This script calculates only the DICE score between prediction and ground truth.
Supports TIFF, NIfTI, and other common medical image formats.

Usage:
    python calculate_dice_score.py
    python calculate_dice_score.py --pred path/to/pred.tif --target path/to/target.tif
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

class DiceScoreCalculator:
    """Simple DICE score calculator."""
    
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
    
    def calculate_dice(self, pred_path, target_path):
        """Calculate DICE score for a single prediction-target pair."""
        try:
            # Load images
            print("Loading images...")
            pred_data = self.load_image(pred_path)
            target_data = self.load_image(target_path)
            
            print(f"Prediction: shape={pred_data.shape}, range=[{pred_data.min()}, {pred_data.max()}]")
            print(f"Target:     shape={target_data.shape}, range=[{target_data.min()}, {target_data.max()}]")
            
            # Ensure same shape
            if pred_data.shape != target_data.shape:
                print(f"Warning: Shape mismatch - pred: {pred_data.shape}, target: {target_data.shape}")
                # Resize to match the smaller dimensions
                min_shape = tuple(min(pred_data.shape[i], target_data.shape[i]) for i in range(len(pred_data.shape)))
                if len(min_shape) == 2:
                    pred_data = pred_data[:min_shape[0], :min_shape[1]]
                    target_data = target_data[:min_shape[0], :min_shape[1]]
                else:
                    pred_data = pred_data[:min_shape[0], :min_shape[1], :min_shape[2]]
                    target_data = target_data[:min_shape[0], :min_shape[1], :min_shape[2]]
                print(f"Resized to: {pred_data.shape}")
            
            # Convert to binary if needed
            if pred_data.max() > 1:
                pred_data = (pred_data > 0).astype(np.uint8)
            if target_data.max() > 1:
                target_data = (target_data > 0).astype(np.uint8)
            
            print("Calculating DICE score...")
            # Calculate DICE score
            dice = self.dice_score(pred_data, target_data)
            
            return dice, pred_data, target_data
            
        except Exception as e:
            print(f"Error calculating DICE score: {e}")
            return None, None, None

def print_dice_summary(dice_score):
    """Print a formatted summary of DICE score."""
    print("\n" + "="*60)
    print("DICE SCORE SUMMARY")
    print("="*60)
    
    print(f"DICE Score: {dice_score:.4f}")
    
    # Interpretation
    print(f"\nINTERPRETATION:")
    if dice_score >= 0.8:
        print(f"   Excellent segmentation quality (DICE ≥ 0.8)")
    elif dice_score >= 0.7:
        print(f"   Good segmentation quality (DICE ≥ 0.7)")
    elif dice_score >= 0.5:
        print(f"   Moderate segmentation quality (DICE ≥ 0.5)")
    else:
        print(f"   Poor segmentation quality (DICE < 0.5)")
    
    print(f"\nDICE SCORE GUIDE:")
    print(f"   • 1.0 = Perfect overlap")
    print(f"   • 0.8+ = Excellent")
    print(f"   • 0.7+ = Good")
    print(f"   • 0.5+ = Moderate")
    print(f"   • <0.5 = Poor")

def main():
    parser = argparse.ArgumentParser(description='Calculate DICE score')
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
    calculator = DiceScoreCalculator()
    
    print(f"Prediction file: {args.pred}")
    print(f"Ground truth file: {args.target}")
    
    # Calculate DICE score
    print("\nCalculating DICE score...")
    dice_score, pred_data, target_data = calculator.calculate_dice(args.pred, args.target)
    
    if dice_score is None:
        print("Failed to calculate DICE score!")
        return 1
    
    # Print summary
    print_dice_summary(dice_score)
    
    # Save results if requested
    if args.output:
        df = pd.DataFrame([{'DICE_Score': dice_score}])
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
        # Use the specific files mentioned in the query
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
        calculator = DiceScoreCalculator()
        
        print(f"Prediction file: {pred_file}")
        print(f"Ground truth file: {target_file}")
        
        # Calculate DICE score
        print("\nCalculating DICE score...")
        dice_score, pred_data, target_data = calculator.calculate_dice(pred_file, target_file)
        
        if dice_score is None:
            print("Failed to calculate DICE score!")
            sys.exit(1)
        
        # Print summary
        print_dice_summary(dice_score)
        
        # Save results
        output_file = "/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/vessel_001_dice_score.csv"
        df = pd.DataFrame([{'DICE_Score': dice_score}])
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
