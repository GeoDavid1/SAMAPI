# Imports
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms

# Check if running under SLURM by looking for SLURM environment variables
if "SLURM_JOB_ID" in os.environ and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")
model_type = 'vit_h'
checkpoint_path = 'sam_vit_h_4b8939.pth'
image_path = '1.tiff'

# Load the SAM model from the registry
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)

# Increase the limit for image size to prevent DecompressionBombError
Image.MAX_IMAGE_PIXELS = None  # or set to a specific higher value

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")  # Open the image and convert to RGB

# Get the desired height and width for resizing (adjust as needed)
desired_size = (640, 480)  # Example size, modify as required

transform = transforms.Compose([
    transforms.Resize(desired_size),  # Resize to the desired size
    transforms.ToTensor(),  # Convert to tensor
    # Add normalization if needed (mean, std)
])
image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Set the image for prediction
mask_generator = SamAutomaticMaskGenerator(sam)

# Remove the batch dimension before generating masks
masks = mask_generator.generate(image_tensor.squeeze(0))  # Squeeze to get rid of the batch dimension

# Output results (optional)
print(f"Masks: {masks}")
