from architecture.srgan import SRGAN
import os
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import segmentation_models as sm
import sys

# Add parent directory to path for importing modules
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))


class SkinSegmentSRGANPipeline:
    """Pipeline that connects segmentation and SRGAN for skin disease images"""

    def __init__(self,
                 seg_model_path="models/segmentation/best_model.h5",
                 srgan_model_path="models/srgan/generator_epoch_100.h5"):

        # Load configurations
        with open("configs/segmentation.yaml", 'r') as f:
            self.seg_config = yaml.safe_load(f)

        with open("configs/srgan.yaml", 'r') as f:
            self.srgan_config = yaml.safe_load(f)

        # Initialize segmentation model
        self.input_shape = tuple(self.seg_config['input_shape'])
        self.seg_model = self._load_segmentation_model(seg_model_path)

        # Initialize SRGAN model
        self.upscale_factor = self.srgan_config['upscale_factor']
        self.hr_shape = tuple(self.srgan_config['hr_shape'])
        self.lr_shape = (
            self.hr_shape[0] // self.upscale_factor,
            self.hr_shape[1] // self.upscale_factor,
            self.hr_shape[2]
        )
        self.srgan = self._load_srgan_model(srgan_model_path)

        # Create output directory
        self.output_dir = Path("results/pipeline")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_segmentation_model(self, model_path):
        """Load segmentation model"""
        if not os.path.exists(model_path):
            print(f"Warning: Segmentation model not found at {model_path}")
            print("Initializing new model without weights...")
            # Initialize model architecture
            model = sm.Unet(
                backbone_name='efficientnetb3',
                input_shape=self.input_shape,
                classes=1,
                activation='sigmoid',
                encoder_weights='imagenet'
            )
        else:
            print(f"Loading segmentation model from {model_path}")
            # Initialize model architecture
            model = sm.Unet(
                backbone_name='efficientnetb3',
                input_shape=self.input_shape,
                classes=1,
                activation='sigmoid',
                encoder_weights=None  # Don't load ImageNet weights when loading our weights
            )
            # Load weights
            model.load_weights(model_path)

        return model

    def _load_srgan_model(self, model_path):
        """Load SRGAN model"""
        # Initialize SRGAN
        srgan = SRGAN(
            input_shape_lr=self.lr_shape,
            input_shape_hr=self.hr_shape,
            upscale_factor=self.upscale_factor
        )

        if not os.path.exists(model_path):
            print(f"Warning: SRGAN model not found at {model_path}")
            print("Using initialized model without weights...")
        else:
            print(f"Loading SRGAN generator from {model_path}")
            srgan.generator.load_weights(model_path)

        return srgan

    def preprocess_image(self, image_path):
        """Load and preprocess an image for the pipeline"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize for segmentation
        seg_input = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        seg_input = seg_input.astype(np.float32) / 255.0  # Normalize to [0, 1]

        return img, seg_input

    def segment_image(self, image):
        """Segment the skin lesion in the image"""
        # Add batch dimension
        input_tensor = np.expand_dims(image, axis=0)

        # Predict mask
        mask = self.seg_model.predict(input_tensor)[0]

        # Threshold the mask and ensure it's 2D
        binary_mask = (mask > 0.5).astype(np.float32)

        # Ensure binary_mask is 2D (H, W) not (H, W, 1)
        if binary_mask.ndim == 3 and binary_mask.shape[2] == 1:
            binary_mask = binary_mask[:, :, 0]

        return binary_mask

    def apply_mask_to_image(self, image, mask):
        """Apply the binary segmentation mask to the image"""
        print(f"Input - Image shape: {image.shape}, Mask shape: {mask.shape}")

        # Ensure the mask has shape (H, W)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]  # Convert from (H, W, 1) to (H, W)
        elif mask.ndim != 2:
            raise ValueError(
                f"Expected mask to be 2D or 3D with 1 channel, got shape: {mask.shape}")

        # Expand mask to 3 channels: (H, W) -> (H, W, 3)
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        print(
            f"After processing - Image shape: {image.shape}, Mask_3ch shape: {mask_3ch.shape}")

        # Ensure image and mask have compatible shapes
        if image.shape != mask_3ch.shape:
            raise ValueError(
                f"Shape mismatch: image {image.shape}, mask {mask_3ch.shape}")

        return image * mask_3ch

    def prepare_for_srgan(self, image):
        """Prepare the masked image for SRGAN enhancement"""
        # Resize to LR shape for SRGAN input
        lr_input = cv2.resize(image, (self.lr_shape[0], self.lr_shape[1]))

        # Convert from [0, 1] to [-1, 1] for SRGAN
        lr_input = lr_input * 2 - 1

        # Add batch dimension
        lr_input = np.expand_dims(lr_input, axis=0)

        return lr_input

    def enhance_image(self, lr_input):
        """Enhance the image using SRGAN"""
        # Generate super-resolution image
        sr_output = self.srgan.generator.predict(lr_input)
        print("SRGAN raw output shape:", sr_output.shape)

        # Validate and safely squeeze batch dim
        if sr_output.ndim == 4 and sr_output.shape[0] == 1:
            sr_output = sr_output[0]
        elif sr_output.ndim == 3:
            pass
        else:
            raise ValueError(
                f"Unexpected SRGAN output shape: {sr_output.shape}")

        # Convert back from [-1, 1] to [0, 1]
        sr_output = (sr_output + 1) / 2.0
        sr_output = np.clip(sr_output, 0, 1)

        # ✅ Resize SRGAN output to match segmentation input resolution
        sr_output = tf.image.resize(sr_output, self.input_shape[:2]).numpy()

        # Validate shape
        if sr_output.ndim != 3 or sr_output.shape[2] != 3:
            raise ValueError(
                f"Expected SRGAN output shape (H, W, 3), got {sr_output.shape}")

        return sr_output

    def process_image(self, image_path, visualize=True):
        """Process an image through the complete pipeline"""
        # Load and preprocess image
        original_img, seg_input = self.preprocess_image(image_path)

        # Step 1: Segment the image
        mask = self.segment_image(seg_input)

        # Step 2: Apply mask to extract the lesion
        masked_image = self.apply_mask_to_image(seg_input, mask)

        # Step 3: Prepare for SRGAN
        lr_input = self.prepare_for_srgan(masked_image)

        # Step 4: Enhance with SRGAN
        enhanced_image = self.enhance_image(lr_input)

        # Save results
        if visualize:
            self.visualize_results(
                image_path,
                original_img,
                seg_input,
                mask,
                masked_image,
                enhanced_image
            )

        return {
            'original': original_img,
            'segmentation_input': seg_input,
            'mask': mask,
            'masked_image': masked_image,
            'enhanced_image': enhanced_image
        }

    def visualize_results(self, image_path, original, seg_input, mask, masked, enhanced):
        """Visualize and save the pipeline results"""
        # Create figure
        plt.figure(figsize=(15, 10))

        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(original)
        plt.title('Original Image')
        plt.axis('off')

        # Segmentation input
        plt.subplot(2, 3, 2)
        plt.imshow(seg_input)
        plt.title('Preprocessed Input')
        plt.axis('off')

        # Segmentation mask - Fixed: mask is now 2D, no need for [:,:,0]
        plt.subplot(2, 3, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('Segmentation Mask')
        plt.axis('off')

        # Masked image
        plt.subplot(2, 3, 4)
        plt.imshow(masked)
        plt.title('Masked Lesion')
        plt.axis('off')

        # Enhanced image
        plt.subplot(2, 3, 5)
        plt.imshow(enhanced)
        plt.title('Enhanced (SRGAN)')
        plt.axis('off')

        # Overlay of enhanced on original
        plt.subplot(2, 3, 6)
        # Resize enhanced to match original
        h, w = original.shape[:2]
        enhanced_resized = cv2.resize(enhanced, (w, h))

        # Create overlay mask based on the segmentation
        mask_resized = cv2.resize(mask, (w, h))
        # Ensure mask_resized is 3D for broadcasting
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        overlay = original.copy().astype(np.float32) / 255.0

        # Place enhanced image in the masked region
        enhanced_region = enhanced_resized * mask_resized
        background = overlay * (1 - mask_resized)
        combined = enhanced_region + background

        plt.imshow(combined)
        plt.title('Overlay Result')
        plt.axis('off')

        # Save the visualization
        out_path = self.output_dir / f"{Path(image_path).stem}_results.png"
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150)
        plt.close()

        print(f"Saved visualization to {out_path}")

        # Save individual results - Fixed: mask is now 2D
        cv2.imwrite(
            str(self.output_dir / f"{Path(image_path).stem}_mask.png"),
            (mask * 255).astype(np.uint8)
        )

        print("Enhanced shape:", enhanced.shape, "dtype:", enhanced.dtype,
              "min:", enhanced.min(), "max:", enhanced.max())

        cv2.imwrite(
            str(self.output_dir / f"{Path(image_path).stem}_enhanced.png"),
            cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
