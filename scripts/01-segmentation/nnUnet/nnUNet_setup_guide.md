# 🏥 nnU-Net Setup Guide for Blood Vessel Segmentation

## 📊 **Dataset Overview**
- **Dataset ID**: 001
- **Dataset Name**: Dataset001_BloodVessels  
- **Total Patients**: 18 patients with CT images and blood vessel masks
- **Training Cases**: 14 patients
- **Test Cases**: 4 patients (pat_011, pat_038, pat_063, pat_130)
- **Modality**: CT (3D)
- **Task**: Binary segmentation (background vs blood vessels)


## 🚀 **Next Steps to Train nnU-Net**
## set up env:

```bash
cd /home/ygao130/Auto/BVSeg/nnUNet && pip install -e .
pip install nnunetv2
```

### **Step 1: Plan and Preprocess**
```bash
cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet &&
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw" && 
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed" &&
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results" &&
nnUNetv2_extract_fingerprint -d 001 --verify_dataset_integrity





cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet && 
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw" && 
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed" && 
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results" && 
nnUNetv2_plan_and_preprocess -d 001 -pl nnUNetPlannerResEncL --verify_dataset_integrity

cd /home/ygao130/Auto/BVSeg/nnUNet && export nnUNet_raw="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_raw" && export nnUNet_preprocessed="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_preprocessed" && export nnUNet_results="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_results" && nnUNetv2_plan_experiment -d 001 -pl nnUNetPlannerResEncL --verify_dataset_integrity
```
This will:
- Analyze your dataset properties
- Create preprocessing plans
- Generate preprocessed data in `nnUNet_preprocessed/`

### **Step 2: Train the Model**
```bash
# Train 3D full resolution model (recommended for 3D data)
nnUNetv2_train -d 001 -c 3d_fullres
nnUNetv2_train -d 001 -c 2d
nnUNetv2_train -d 001 -c 3d_lowres
nnUNetv2_train -d 001 -c 3d_cascade_fullres


cd /home/ygao130/Auto/BVSeg/nnUNet && export nnUNet_raw="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_raw" && export nnUNet_preprocessed="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_preprocessed" && export nnUNet_results="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_results" && nnUNetv2_train 001 3d_fullres all -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_250epochs_NoMirroring

nnUNetv2_train 001 3d_fullres all -p nnUNetResEncUNetLPlans -c
# The -c flag continues from the last checkpoint.

cd /home/ygao130/Auto/BVSeg/nnUNet && export nnUNet_raw="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_raw" && export nnUNet_preprocessed="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_preprocessed" && export nnUNet_results="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_results" &&
nnUNetv2_train 001 3d_fullres all -tr nnUNetTrainer_250epochs_NoMirroring

```

### **Step 3: Run Inference**
```bash
# Predict on new images
nnUNetv2_predict -d 001 -i /path/to/input/folder -o /path/to/output/folder

# Or use the Python API
python nnunetv2/inference/predict_from_raw_data.py \
    -i /path/to/input/folder \
    -o /path/to/output/folder \
    -d 001 \
    -c 3d_fullres


# use this: 
cd /home/ygao130/Auto/BVSeg/nnUNet && conda activate nnunet && export nnUNet_raw="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_raw" && export nnUNet_preprocessed="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_preprocessed" && export nnUNet_results="/home/ygao130/Auto/BVSeg/nnUNet/nnUNet_results" && 
nnUNetv2_predict -d 001 -i /home/ygao130/Auto/BVSeg/nnUNet/nnUNet_raw/Dataset001_BloodVessels/imagesTs -o /home/ygao130/Auto/BVSeg/nnUNet/nnUNet_results/Dataset001_BloodVessels/nnUNetTrainer_250epochs_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres/predictions -c 3d_fullres -tr nnUNetTrainer_250epochs_NoMirroring -p nnUNetResEncUNetLPlans -f all
```

## 📋 **Dataset Details**
- **Images**: CT scans (3D NIfTI format)
- **Labels**: Binary masks (0=background, 1=blood vessel)
- **Training cases**: 18 patients
- **File format**: `.nii.gz`
- **Spatial alignment**: Already aligned using your preprocessing pipeline


## 🎯 **Model Configurations Available**
1. **2d**: 2D U-Net (fastest, least memory)
2. **3d_fullres**: 3D U-Net on full resolution (recommended)
3. **3d_lowres**: 3D U-Net on low resolution
4. **3d_cascade_fullres**: 3D cascade (for large images)
