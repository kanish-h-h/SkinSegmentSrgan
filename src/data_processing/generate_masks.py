
from segmentation_models import Unet
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Set paths
input_dir = Path("data/raw/segmentation/images")
output_dir = Path("data/raw/segmentation/masks")
output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory

# Load pre-trained U-Net model
print("Loading pre-trained U-Net model...")
model = Unet(
    backbone_name="efficientnetb3",  # Less resource-intensive backbone
    encoder_weights="imagenet",       # Use ImageNet weights
    input_shape=(512, 512, 3),        # Flexible input size
    classes=1,                        # Binary mask output
    activation="sigmoid"
)
model.trainable = False  # Freeze weights for inference

# Preprocessing function
def preprocess(image):
    """Preprocess image for U-Net prediction"""
    image = cv2.resize(image, (512, 512))          # Resize to model input size
    image = image.astype(np.float32) / 255.0        # Normalize to [0, 1]
    # Standardize
    image = (image - np.mean(image)) / (np.std(image) + 1e-7)
    return np.expand_dims(image, axis=0)            # Add batch dimension

# Process each image with visualization
def generate_masks(input_dir, output_dir, visualize_results=False):
    """Generate segmentation masks for all images in the input directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory if needed
    if visualize_results:
        vis_dir = output_dir.parent / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for img_path in tqdm(list(input_dir.glob("*.*"))):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Preprocess and predict
        input_tensor = preprocess(image)
        mask = model.predict(input_tensor, verbose=0)[0]  # Get mask

        # Postprocess mask - use dynamic thresholding
        threshold = np.mean(mask) + 0.5 * np.std(mask)  # Adaptive threshold
        binary_mask = (mask > threshold).astype(np.uint8) * 255  
        
        # Resize to original image dimensions
        binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Save mask
        output_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(output_path), binary_mask)
        
        # Visualize results if requested
        if visualize_results:
            # Create visualization with original, mask, and overlay
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axs[0].imshow(image)
            axs[0].set_title('Original')
            axs[0].axis('off')
            
            # Mask
            axs[1].imshow(binary_mask, cmap='gray')
            axs[1].set_title('Segmentation Mask')
            axs[1].axis('off')
            
            # Overlay
            overlay = image.copy()
            mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
            overlay_mask = np.where(mask_rgb > 0, [0, 255, 0], [0, 0, 0]).astype(np.uint8)
            overlay = cv2.addWeighted(overlay, 1.0, overlay_mask, 0.5, 0)
            axs[2].imshow(overlay)
            axs[2].set_title('Overlay')
            axs[2].axis('off')
            
            plt.tight_layout()
            vis_path = vis_dir / f"{img_path.stem}_visualization.png"
            plt.savefig(str(vis_path))
            plt.close()
    
    print(f"Generated {len(list(output_dir.glob('*.*')))} masks")

if __name__ == "__main__":
    # Generate masks with visualization
    generate_masks(input_dir, output_dir, visualize_results=True)
    print("Mask generation complete!")

