# nnU-Net Setup Guide for Blood Vessel Segmentation

## Overview

This guide provides step-by-step instructions for setting up and training nnU-Net for blood vessel segmentation. nnU-Net is a self-configuring framework for deep learning-based biomedical image segmentation that automatically adapts to different datasets without manual intervention.

**Key Resources:**
- **Paper**: [nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation](https://arxiv.org/abs/1809.10486)
- **GitHub Repository**: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- **Documentation**: [nnU-Net Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/)

## Dataset Overview

- **Dataset ID**: 001
- **Dataset Name**: Dataset001_BloodVessels  
- **Total Images**: 1 image with blood vessel masks
- **Training Cases**: 1 
- **Test Cases**: 1 
- **Task**: Binary segmentation (background vs blood vessels)
- **Modality**: 2D medical images
- **File Format**: TIFF images

## Prerequisites

Before starting, ensure you have:
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Sufficient disk space for preprocessed data and model checkpoints
- The preprocessed data from the `preprocess/` folder

## Environment Setup

```bash
# Install nnU-Net from source
cd /home/ygao130/Auto/BVSeg/nnUNet && pip install -e .

# Install nnU-Net v2
pip install nnunetv2
```

## Step-by-Step Workflow

### **Step 1: Dataset Verification and Planning**

First, verify that your dataset is properly formatted and extract dataset properties:

```bash
# Set environment variables
cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw"
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed"
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results"

# Verify dataset integrity and extract fingerprint
nnUNetv2_extract_fingerprint -d 001 --verify_dataset_integrity
```

Next, plan the experiment and preprocess the data:

```bash
# Plan and preprocess the dataset
nnUNetv2_plan_and_preprocess -d 001 -pl nnUNetPlannerResEncL --verify_dataset_integrity
```

**What this step does:**
- Analyzes dataset properties (image dimensions, spacing, intensity distributions)
- Creates preprocessing plans optimized for your specific dataset
- Generates preprocessed data in `nnUNet_preprocessed/`
- Validates data integrity and format compliance

### **Step 2: Train the Model**

Choose one of the following training configurations based on your computational resources and time constraints:

#### Option A: Quick Training (50 epochs)
```bash
# Set environment variables
cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw"
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed"
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results"

# Train for 50 epochs (faster, good for initial testing)
nnUNetv2_train 001 2d all -p nnUNetResEncUNetLPlans \
-tr nnUNetTrainer_50epochs --npz
```

#### Option B: Extended Training (250 epochs)
```bash
# Set environment variables
cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw"
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed"
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results"

# Train for 250 epochs (longer training, potentially better results)
nnUNetv2_train 001 2d all -p nnUNetResEncUNetLPlans \
-tr nnUNetTrainer_250epochs_NoMirroring --npz
```

**Training Parameters Explained:**
- `001`: Dataset ID
- `2d`: 2D configuration (suitable for 2D images)
- `all`: Train on all available folds
- `-p nnUNetResEncUNetLPlans`: Use the ResEncUNetLPlans configuration
- `-tr`: Trainer configuration (epochs and augmentation settings)
- `--npz`: Save predictions in .npz format for evaluation

### **Step 3: Run Inference**

After training, use the trained model to perform inference on new images. Choose the appropriate command based on which model you trained:

#### Option A: Inference with 50-epoch model
```bash
# Set environment variables
cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw"
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed"
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results"

# Run inference with 50-epoch model
nnUNetv2_predict -d 001 \
-i /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw/Dataset001_vessel/imagesTs \
-o /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results/Dataset001_vessel/nnUNetTrainer_50epochs__nnUNetResEncUNetLPlans__2d/predictions \
-c 2d -tr nnUNetTrainer_50epochs -p nnUNetResEncUNetLPlans -f all
```

#### Option B: Inference with 250-epoch model
```bash
# Set environment variables
cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw"
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed"
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results"

# Run inference with 250-epoch model
cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet && 
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw" && 
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed" &&
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results" &&
nnUNetv2_predict -d 001 \
-i /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw/Dataset001_vessel/imagesTs \
-o /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results/Dataset001_vessel/nnUNetTrainer_250epochs_NoMirroring__nnUNetResEncUNetLPlans__2d/predictions \
-c 2d -tr nnUNetTrainer_250epochs_NoMirroring -p nnUNetResEncUNetLPlans -f all
```


**Inference Parameters Explained:**
- `-d 001`: Dataset ID
- `-i`: Input directory containing images to segment
- `-o`: Output directory for segmentation results
- `-c 2d`: Use 2D configuration
- `-tr`: Trainer configuration (must match the trained model)
- `-p`: Plans configuration (must match the trained model)
- `-f all`: Use all available folds

## Monitoring and Evaluation

### Training Progress
Monitor training progress by checking:
- Training logs in the terminal output
- TensorBoard logs (if enabled)
- Model checkpoints saved in `nnUNet_results/`

### Evaluation Metrics
After inference, evaluate your model using:
- Dice coefficient
- Hausdorff distance
- Surface distance metrics
- Visual inspection of segmentation results

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU training
2. **Dataset format errors**: Ensure data follows nnU-Net naming conventions
3. **Missing dependencies**: Reinstall nnU-Net and required packages

### Getting Help
- Check the [nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/)
- Visit the [nnU-Net GitHub issues](https://github.com/MIC-DKFZ/nnUNet/issues)
- Refer to the original [nnU-Net paper](https://arxiv.org/abs/1809.10486) for theoretical background

## Additional Resources

- **nnU-Net Paper**: [Self-adapting Framework for U-Net-Based Medical Image Segmentation](https://arxiv.org/abs/1809.10486)
- **nnU-Net GitHub**: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- **nnU-Net Documentation**: [https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/)
- **Medical Image Segmentation**: [nnU-Net v2 Paper](https://arxiv.org/abs/2306.08104)