"""
Overview:
---------
This script automates the process of extracting frames from MP4 videos, removing their backgrounds, and reassembling the processed frames back into new videos with either white or black backgrounds.

Workflow:
1. For each MP4 file in the current directory (excluding files already processed), the script:
   - Detects the video framerate.
   - Extracts all frames using ffmpeg.
   - Removes the background from each frame using the rembg library (with the 'birefnet-massive' model).
   - Generates two sets of frames: one with a white background, one with a black background.
   - Reassembles the processed frames into two new MP4 videos (one with white background, one with black).

Requirements:
-------------
- Python packages: opencv-python, numpy, rembg, tqdm
- ffmpeg and ffprobe installed and available in your system PATH

Usage:
------
Place your MP4 files in the script's directory and run:
    python split_videos_to_image_frames.py

The script will create subdirectories for frames and output videos for each input file.
"""
# ...existing code...

import os
import subprocess
import cv2
import numpy as np
import rembg
from rembg import remove
from tqdm import tqdm

# Function to get the framerate of an input video
def get_framerate(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=nokey=1:noprint_wrappers=1", video_path],
        capture_output=True, text=True
    )
    framerate = result.stdout.strip()
    if framerate:
        return eval(framerate)  # Convert "30000/1001" to float
    raise ValueError("Framerate not found")

# Function to process images and remove background
def remove_background(input_path, output_white_path, output_black_path):
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return
    
    # Ensure image has 4 channels (RGBA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) if image.shape[2] == 3 else image
    
    # Remove background using rembg (SAM model is used automatically if installed)
    result = remove(image, session=rembg_session)
    background_region = result[:, :, 3] < 0.8 * 255  # Find regions that are not transparent enough

    # Create versions with white and black backgrounds
    white_bg = result.copy()
    white_bg[background_region] = [255, 255, 255, 255]  # Set to white

    black_bg = result.copy()
    black_bg[background_region] = [0, 0, 0, 255]  # Set to black

    # Convert alpha images to BGR by replacing transparent regions
    white_bg_bgr = cv2.cvtColor(white_bg, cv2.COLOR_BGRA2BGR)
    black_bg_bgr = cv2.cvtColor(black_bg, cv2.COLOR_BGRA2BGR)

    cv2.imwrite(output_white_path, white_bg_bgr)
    cv2.imwrite(output_black_path, black_bg_bgr)

rembg_session = rembg.new_session('birefnet-massive')

# Process each mp4 file in the directory
i = 0
for filename in os.listdir('.'):
    if filename.endswith('.mp4') and '_rmbg_' not in filename:
        i += 1
        
        base_name = os.path.splitext(filename)[0]
        frame_dir = base_name  # Directory to store extracted frames

        os.makedirs(frame_dir, exist_ok=True)

        # Get framerate from input video
        framerate = get_framerate(filename)
        print(f"Detected framerate for {filename}: {framerate} fps")

        # Extract frames using ffmpeg
        frame_pattern = f"{frame_dir}/frame_%04d.png"
        subprocess.run(["ffmpeg", "-i", filename, "-vsync", "vfr", frame_pattern], check=True)

        print(f"Extracted frames for {filename}")

        # Create output directories
        white_dir = f"{frame_dir}_white"
        black_dir = f"{frame_dir}_black"
        os.makedirs(white_dir, exist_ok=True)
        os.makedirs(black_dir, exist_ok=True)

        # Process frames to remove backgrounds
        frame_files = sorted(os.listdir(frame_dir))
        for frame in tqdm(frame_files, desc=f"Processing {filename}"):
            frame_path = os.path.join(frame_dir, frame)
            white_path = os.path.join(white_dir, frame)
            black_path = os.path.join(black_dir, frame)

            remove_background(frame_path, white_path, black_path)

        # Convert processed frames back to video
        output_white_video = f"{base_name}_rmbg_white.mp4"
        output_black_video = f"{base_name}_rmbg_black.mp4"

        # Reassemble videos with original framerate
        subprocess.run(["ffmpeg", "-framerate", str(framerate), "-i", f"{white_dir}/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_white_video], check=True)
        subprocess.run(["ffmpeg", "-framerate", str(framerate), "-i", f"{black_dir}/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_black_video], check=True)

        print(f"Created videos: {output_white_video}, {output_black_video}")
