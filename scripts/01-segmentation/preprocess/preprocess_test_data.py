import os
import re
from PIL import Image
from pathlib import Path

# Define paths
test_data_dir = "/home/ygao130/Auto/STOR674/Data/test_data"

def extract_number_from_filename(fname):
    """
    Extract number from filename pattern like 'p2-from 5-5-2 M.tif'
    Returns the number after 'p' (e.g., 2 for 'p2-from...')
    """
    # Look for pattern 'p' followed by digits (e.g., p2, p3, p10)
    match = re.search(r'p(\d+)', fname, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def convert_rgb_to_grayscale_and_rename(input_dir, prefix="vessel"):
    """
    Convert RGB images to grayscale and rename them with nnUNet naming convention.
    The vessel number is extracted from the filename (e.g., p2 -> vessel_002_0000.tif)
    
    Args:
        input_dir: Directory containing the input images
        prefix: Prefix for the output filenames (default: "vessel")
    """
    
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} does not exist")
        return
    
    # Get all TIF files
    tif_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".tiff"))]
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return
    
    # Extract numbers and create a list of (filename, number) tuples
    file_num_pairs = []
    for fname in tif_files:
        num = extract_number_from_filename(fname)
        if num is not None:
            file_num_pairs.append((fname, num))
        else:
            print(f"Warning: Could not extract number from filename '{fname}', skipping")
    
    # Sort by the extracted number
    file_num_pairs.sort(key=lambda x: x[1])
    
    if not file_num_pairs:
        print("No files with extractable numbers found")
        return
    
    print(f"Found {len(file_num_pairs)} TIF file(s) to process")
    print(f"Output naming pattern: {prefix}_XXX_0000.tif\n")
    
    for fname, case_num in file_num_pairs:
        try:
            input_path = os.path.join(input_dir, fname)
            
            # Load the image
            img = Image.open(input_path)
            print(f"Processing: {fname} (mode: {img.mode}, size: {img.size})")
            
            # Convert to grayscale if it's RGB or has color channels
            if img.mode in ('RGB', 'RGBA', 'CMYK', 'LAB', 'HSV'):
                img_gray = img.convert("L")  # "L" = grayscale
                print(f"  Converted from {img.mode} to grayscale")
            elif img.mode == 'L':
                img_gray = img  # Already grayscale
                print(f"  Already grayscale, no conversion needed")
            else:
                # Try to convert other modes to grayscale
                img_gray = img.convert("L")
                print(f"  Converted from {img.mode} to grayscale")
            
            # Create new filename with nnUNet naming convention using extracted number
            output_fname = f"{prefix}_{case_num:03d}_0000.tif"
            output_path = os.path.join(input_dir, output_fname)
            
            # Check if output file already exists (and is different from input)
            if output_path == input_path:
                print(f"  Skipping rename (filename already matches pattern)")
            elif os.path.exists(output_path):
                print(f"  Warning: {output_fname} already exists, skipping")
                continue
            else:
                # Save grayscale image with new name
                img_gray.save(output_path, format='TIFF')
                
                # If the old filename is different, remove it
                if input_path != output_path:
                    os.remove(input_path)
                    print(f"  Saved and removed old file: {fname} -> {output_fname}")
                else:
                    print(f"  Saved: {output_fname}")
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n=== Conversion Complete ===")
    print(f"Processed {len(file_num_pairs)} file(s)")
    
    # List final files
    final_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".tiff"))])
    print(f"\nFinal files in {input_dir}:")
    for f in final_files:
        print(f"  - {f}")

if __name__ == "__main__":
    print("=== Converting Test Data: RGB to Grayscale and Renaming ===\n")
    convert_rgb_to_grayscale_and_rename(test_data_dir, prefix="vessel")

