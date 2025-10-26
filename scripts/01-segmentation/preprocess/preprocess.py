import os
from PIL import Image
import shutil
from pathlib import Path

# Define paths
base_path = "/home/ygao130/Auto/STOR674"
train_rgb_dir = os.path.join(base_path, "Data/train_data/image")
val_rgb_dir = os.path.join(base_path, "Data/validation_data/image")
train_label_dir = os.path.join(base_path, "Data/train_data/ground_truth")
val_label_dir = os.path.join(base_path, "Data/validation_data/ground_truth")

# nnUNet output directories
nnunet_base = "/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_raw/Dataset001_vessel"
images_tr_dir = os.path.join(nnunet_base, "imagesTr")
images_ts_dir = os.path.join(nnunet_base, "imagesTs")
labels_tr_dir = os.path.join(nnunet_base, "labelsTr")
labels_ts_dir = os.path.join(nnunet_base, "labelsTs")

# Create output directories
os.makedirs(images_tr_dir, exist_ok=True)
os.makedirs(images_ts_dir, exist_ok=True)
os.makedirs(labels_tr_dir, exist_ok=True)
os.makedirs(labels_ts_dir, exist_ok=True)

def convert_rgb_to_grayscale_and_save(rgb_dir, label_dir, output_img_dir, output_label_dir, prefix, start_idx=0):
    """Convert RGB images to grayscale and save with nnUNet naming convention"""
    
    if not os.path.exists(rgb_dir):
        print(f"Warning: {rgb_dir} does not exist")
        return 0
    
    converted_count = 0
    
    for fname in os.listdir(rgb_dir):
        if fname.endswith((".tif", ".tiff")):
            try:
                # Load and convert RGB to grayscale
                img_path = os.path.join(rgb_dir, fname)
                img = Image.open(img_path)
                
                # Convert to grayscale if it's RGB
                if img.mode == 'RGB':
                    img_gray = img.convert("L")  # "L" = grayscale
                else:
                    img_gray = img  # Already grayscale
                
                # Create nnUNet naming convention: vessel_XXX_0000.tif
                case_num = start_idx + converted_count
                output_fname = f"{prefix}_{case_num:03d}_0000.tif"
                output_path = os.path.join(output_img_dir, output_fname)
                
                # Save grayscale image
                img_gray.save(output_path)
                print(f"Converted: {fname} -> {output_fname}")
                
                # Copy the grayscale image to labels directory with vessel_XXX.tif naming
                output_label_img_fname = f"{prefix}_{case_num:03d}.tif"
                output_label_img_path = os.path.join(output_label_dir, output_label_img_fname)
                img_gray.save(output_label_img_path)
                print(f"Copied image to labels: {fname} -> {output_label_img_fname}")
                
                converted_count += 1
                
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                continue
    
    return converted_count

print("=== Converting Training Data ===")
train_count = convert_rgb_to_grayscale_and_save(
    train_rgb_dir, train_label_dir, images_tr_dir, labels_tr_dir, "vessel", start_idx=0
)

print("\n=== Converting Validation Data ===")
val_count = convert_rgb_to_grayscale_and_save(
    val_rgb_dir, val_label_dir, images_ts_dir, labels_ts_dir, "vessel", start_idx=1
)

print(f"\n=== Conversion Complete ===")
print(f"Training images converted: {train_count}")
print(f"Validation images converted: {val_count}")
print(f"Total images processed: {train_count + val_count}")

# Verify the output
print(f"\n=== Output Verification ===")
print(f"Training images: {os.listdir(images_tr_dir)}")
print(f"Training labels: {os.listdir(labels_tr_dir)}")
print(f"Validation images: {os.listdir(images_ts_dir)}")
print(f"Validation labels: {os.listdir(labels_ts_dir)}")
