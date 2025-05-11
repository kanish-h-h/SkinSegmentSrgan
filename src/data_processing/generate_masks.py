from segmentation_models import Unet
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import os

# Set paths
# Directory with 100 images
input_dir = Path("data/raw/segmentation/images")
output_dir = Path("data/raw/segmentation/masks")
output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory

# Load pre-trained U-Net model (using segmentation_models)

model = Unet(
    backbone_name="efficientnetb7",       # Pre-trained backbone
    encoder_weights="imagenet",     # Use ImageNet weights
    input_shape=(512, 512, 3),   # Flexible input size
    classes=1,                      # Binary mask output
    activation="sigmoid"
)
model.trainable = False  # Freeze weights for inference

# Preprocessing function


def preprocess(image):
    image = cv2.resize(image, (512, 512))          # Resize to common size
    image = image.astype(np.float32) / 255.0        # Normalize to [0, 1]
    image = (image - np.mean(image)) / np.std(image)  # Zerp-mean noralisation
    return np.expand_dims(image, axis=0)            # Add batch dimension


# Process each image
for img_path in tqdm(list(input_dir.glob("*.*"))):
    # Load image
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Preprocess and predict
    input_tensor = preprocess(image)
    mask = model.predict(input_tensor, verbose=0)[0]  # Get mask

    # Postprocess mask
    mask = (mask > 0.5).astype(np.uint8) * 255       # Threshold
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Original size

    # Save mask
    output_path = output_dir / f"{img_path.stem}_mask.png"
    cv2.imwrite(str(output_path), mask)
