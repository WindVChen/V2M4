"""
Overview:
---------
(!! You can also do the similar conversions to white backgrounds by slightly modifying this code. !!)

This script converts images with black backgrounds into PNG images with alpha (transparency) channels. 
It scans all folders in the current directory ending with "_black", detects background pixels (nearly black), 
and sets their alpha value to 0 (transparent), while keeping the foreground opaque.

Workflow:
---------
1. For each folder ending with "_black" in the current directory:
   - Creates a corresponding "_alpha" folder.
   - Processes each image in the "_black" folder:
     - Detects background pixels (all channels < 10).
     - Sets those pixels as transparent in the output PNG.
     - Saves the result in the "_alpha" folder.

Requirements:
-------------
- Python packages: opencv-python, numpy

Usage:
------
Place your "_black" image folders in the script's directory and run:
    python convert_black_white_background_images_to_alpha_images.py

The script will create corresponding "_alpha" folders with PNG images containing transparency.
"""
# ...existing code...

import os
import cv2
import numpy as np

def add_alpha_channel(image_path, save_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read image without alpha channel
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    white_mask = np.all(image < 10, axis=-1)  # Detect background pixels where all channels are less than 10
    alpha_channel = np.where(white_mask, 0, 255).astype(np.uint8)  # Create alpha channel
    
    image_with_alpha = np.dstack([image, alpha_channel])  # Stack with alpha channel
    cv2.imwrite(save_path, image_with_alpha)  # Save as PNG

def process_folders(base_dir):
    for folder in os.listdir(base_dir):
        if folder.endswith("_black") and os.path.isdir(os.path.join(base_dir, folder)):
            white_folder_path = os.path.join(base_dir, folder)
            alpha_folder_path = os.path.join(base_dir, folder.replace("_black", "_alpha"))
            
            os.makedirs(alpha_folder_path, exist_ok=True)  # Create target directory
            
            for file in os.listdir(white_folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(white_folder_path, file)
                    save_path = os.path.join(alpha_folder_path, os.path.splitext(file)[0] + ".png")
                    add_alpha_channel(image_path, save_path)
                    print(f"Processed: {save_path}")

if __name__ == "__main__":
    base_directory = os.getcwd()  # Set base directory to current working directory
    process_folders(base_directory)
