# nnU-Net Setup Guide for Blood Vessel Segmentation

## **Dataset Overview**
- **Dataset ID**: 001
- **Dataset Name**: Dataset001_BloodVessels  
- **Total Patients**: 18 patients with CT images and blood vessel masks
- **Training Cases**: 14 patients
- **Test Cases**: 4 patients (pat_011, pat_038, pat_063, pat_130)
- **Modality**: CT (3D)
- **Task**: Binary segmentation (background vs blood vessels)


## **Next Steps to Train nnU-Net**
## set up env:

```bash
cd /home/ygao130/Auto/BVSeg/nnUNet && pip install -e .
pip install nnunetv2
```

### **Step 1: Plan and Preprocess**
```bash
# sanity check
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

# cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet && 
# export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw" && 
# export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed" && 
# export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results" && 
# nnUNetv2_plan_and_preprocess -d 001 -pl nnUNetPlannerResEncL_custom --verify_dataset_integrity
```
This will:
- Analyze your dataset properties
- Create preprocessing plans
- Generate preprocessed data in `nnUNet_preprocessed/`

### **Step 2: Train the Model**
```bash
cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet && 
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw" && 
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed" && 
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results" && 
nnUNetv2_train 001 2d all -p nnUNetResEncUNetLPlans \
-tr nnUNetTrainer_250epochs_NoMirroring --npz 

cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet && 
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw" && 
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed" && 
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results" && 
nnUNetv2_train 001 2d all -p nnUNetResEncUNetLPlans \
-tr nnUNetTrainer_50epochs --npz 

```

### **Step 3: Run Inference**
```bash
# Predict on new images
# use this: 
cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet && 
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw" && 
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed" && 
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results" && 
nnUNetv2_predict -d 001 -i /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw/Dataset001_vessel/imagesTs -o /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results/Dataset001_vessel/nnUNetTrainer_50epochs__nnUNetResEncUNetLPlans__2d/predictions -c 2d -tr nnUNetTrainer_50epochs -p nnUNetResEncUNetLPlans -f all


cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet && 
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw" && 
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed" && 
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results" && 
nnUNetv2_predict -d 001 -i /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw/Dataset001_vessel/imagesTs -o /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results/Dataset001_vessel/nnUNetTrainer_100epochs__nnUNetResEncUNetLPlans__2d/predictions -c 2d -tr nnUNetTrainer_100epochs -p nnUNetResEncUNetLPlans -f all 


cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet && 
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw" && 
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed" && 
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results" && 
nnUNetv2_predict -d 001 -i /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw/Dataset001_vessel/imagesTs -o /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results/Dataset001_vessel/nnUNetTrainer_250epochs_NoMirroring__nnUNetResEncUNetLPlans__2d/predictions -c 2d -tr nnUNetTrainer_250epochs_NoMirroring -p nnUNetResEncUNetLPlans -f all 
```

## find best config:

```bash
cd /home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet && 
export nnUNet_raw="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw" && 
export nnUNet_preprocessed="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_preprocessed" && 
export nnUNet_results="/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results" && 
nUNetv2_find_best_configuration Dataset001_vessel -c 2d -tr nnUNetTrainer_50epochs nnUNetTrainer_100epochs -p nnUNetResEncUNetLPlans
```