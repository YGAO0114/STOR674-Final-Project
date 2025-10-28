#!/usr/bin/env python3
"""
Simple script to compare nnUNet model performance and determine the best configuration
"""

import json
import os
from pathlib import Path

def load_validation_summary(model_path):
    """Load validation summary from a model folder"""
    summary_path = Path(model_path) / "fold_all" / "validation" / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None

def main():
    # Define the models to compare
    base_path = "/home/ygao130/Auto/STOR674/scripts/01-segmentation/nnUnet/nnUNet_results/Dataset001_vessel"
    
    models = {
        "nnUNetTrainer_50epochs__nnUNetResEncUNetLPlans__2d": "50 epochs",
        "nnUNetTrainer_100epochs__nnUNetResEncUNetLPlans__2d": "100 epochs", 
        "nnUNetTrainer_250epochs_NoMirroring__nnUNetResEncUNetLPlans__2d": "250 epochs (No Mirroring)"
    }
    
    results = {}
    
    print("="*80)
    print("nnUNet Model Performance Comparison")
    print("="*80)
    print()
    
    # Load results for each model
    for model_folder, description in models.items():
        model_path = os.path.join(base_path, model_folder)
        summary = load_validation_summary(model_path)
        
        if summary:
            dice_score = summary['foreground_mean']['Dice']
            iou_score = summary['foreground_mean']['IoU']
            tp = summary['foreground_mean']['TP']
            fp = summary['foreground_mean']['FP']
            fn = summary['foreground_mean']['FN']
            
            results[model_folder] = {
                'description': description,
                'dice': dice_score,
                'iou': iou_score,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'path': model_path
            }
            
            print(f"Model: {description}")
            print(f"  Dice Score: {dice_score:.6f}")
            print(f"  IoU Score:  {iou_score:.6f}")
            print(f"  TP: {tp:,}, FP: {fp:,}, FN: {fn:,}")
            print()
        else:
            print(f"Warning: Could not load results for {description}")
            print()
    
    # Find the best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['dice'])
        best_name, best_data = best_model
        
        print("="*80)
        print("BEST CONFIGURATION")
        print("="*80)
        print(f"Best Model: {best_data['description']}")
        print(f"Model Folder: {best_name}")
        print(f"Dice Score: {best_data['dice']:.6f}")
        print(f"IoU Score: {best_data['iou']:.6f}")
        print()
        
        print("INFERENCE COMMAND:")
        print(f"nnUNetv2_predict -d Dataset001_vessel -i INPUT_FOLDER -o OUTPUT_FOLDER -tr {best_name.split('__')[0]} -p nnUNetResEncUNetLPlans -c 2d")
        print()
        
        print("MODEL PATH:")
        print(f"{best_data['path']}")
        print()
        
        # Create inference information file
        inference_info = {
            'dataset_name': 'Dataset001_vessel',
            'best_model': {
                'trainer': best_name.split('__')[0],
                'plans': 'nnUNetResEncUNetLPlans',
                'configuration': '2d',
                'model_folder': best_name,
                'dice_score': best_data['dice'],
                'iou_score': best_data['iou'],
                'path': best_data['path']
            },
            'all_results': {k: {'dice': v['dice'], 'iou': v['iou']} for k, v in results.items()}
        }
        
        output_file = os.path.join(base_path, 'inference_information.json')
        with open(output_file, 'w') as f:
            json.dump(inference_info, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        
        # Create inference instructions file
        instructions_file = os.path.join(base_path, 'inference_instructions.txt')
        with open(instructions_file, 'w') as f:
            f.write("nnUNet Inference Instructions\n")
            f.write("="*50 + "\n\n")
            f.write(f"Best Model: {best_data['description']}\n")
            f.write(f"Dice Score: {best_data['dice']:.6f}\n\n")
            f.write("Command:\n")
            f.write(f"nnUNetv2_predict -d Dataset001_vessel -i INPUT_FOLDER -o OUTPUT_FOLDER -tr {best_name.split('__')[0]} -p nnUNetResEncUNetLPlans -c 2d\n\n")
            f.write("Model Path:\n")
            f.write(f"{best_data['path']}\n")
        
        print(f"Instructions saved to: {instructions_file}")

if __name__ == "__main__":
    main()
