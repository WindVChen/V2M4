import torch
import torchvision.transforms as transforms
import cv2
import os
import json
import numpy as np
from calculate_fvd import calculate_fvd
from calculate_lpips import calculate_lpips, calculate_dreamsim_loss, calculate_clip_loss
import argparse

# Constants
FRAME_SIZE = 512  # Resize all frames to 512x512
TIMESTAMP_LIMIT = 32  # Split videos into 32-frame subvideos

# Define transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor()  # Converts to [0,1] range
])

def load_video_as_tensor(video_path):
    """Load a video and convert it into a tensor [timestamp, channel, width, height]."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame = transform(frame)  # Resize & normalize
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < TIMESTAMP_LIMIT:
        return None  # Ignore videos with fewer than 32 frames
    
    video_tensor = torch.stack(frames)  # Shape: [timestamp, channel, width, height]
    return video_tensor

def process_videos(folder):
    """Load all videos in a folder, convert them into tensors, and split into 32-frame subvideos."""
    video_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp4') and 'normal' not in f and 'interpolated' not in f])
    all_subvideos = []

    for video_file in video_files:
        tensor = load_video_as_tensor(video_file)
        if tensor is not None:
            # Split into 32-frame subvideos
            subvideos = [tensor[i:i+TIMESTAMP_LIMIT] for i in range(0, tensor.shape[0] - TIMESTAMP_LIMIT + 1, TIMESTAMP_LIMIT)]
            print(f"Loaded {len(subvideos)} subvideos from {video_file}")
            all_subvideos.extend(subvideos)
    
    if not all_subvideos:
        raise ValueError(f"No valid subvideos found in {folder}")

    # Stack into batch: [batch, timestamp, channel, width, height]
    return torch.stack(all_subvideos)

parser = argparse.ArgumentParser(description="Evaluate video metrics between ground truth and generated videos.")
parser.add_argument('--gt_videos_path', type=str, required=True, help='Path to ground truth videos folder')
parser.add_argument('--result_videos_path', type=str, required=True, help='Path to generated/result videos folder')
args = parser.parse_args()

GT_videos_path = args.gt_videos_path
result_videos_path = args.result_videos_path

# # Process videos into 32-frame subvideos
videos1 = process_videos(GT_videos_path)  # Shape: [batch, 32, channel, width, height]
videos2 = process_videos(result_videos_path)

# Move tensors to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute metrics
only_final = True

lpips, lpips_eachvideo = calculate_lpips(videos1, videos2, device, only_final=only_final)

fvd = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=only_final)

dreamsim, dreamsim_eachvideo = calculate_dreamsim_loss(videos1, videos2, device, only_final=only_final)

clip_loss, clip_loss_eachvideo = calculate_clip_loss(videos1, videos2, device, only_final=only_final)

result = {
    'fvd': fvd,
    'lpips': lpips,
    'dreamsim': dreamsim,
    'clip_loss': clip_loss,
}

# Print results
print(json.dumps(result, indent=4))