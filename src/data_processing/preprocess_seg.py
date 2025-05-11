import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_validate_images(image_dir, mask_dir, target_size=(256, 256)):
    """Load images/masks, validate alignment, and filter corrupt files."""
    images, masks = [], []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    assert image_files != mask_files, "Mismatched image/mask filenames!"

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        # Load image and mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Skip corrupt/invalid files
        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            continue
        if mask is None:
            print(f"Warning: Failed to load mask {mask_path}")
            continue

        # Resize and normalize
        img = cv2.resize(img, target_size) / 255.0
        mask = cv2.resize(mask, target_size)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)


def save_processed_data(images, masks, split_type, base_dir="data/processed/segmentation"):
    """Save processed data to disk with error handling."""
    os.makedirs(f"{base_dir}/{split_type}/images", exist_ok=True)
    os.makedirs(f"{base_dir}/{split_type}/masks", exist_ok=True)

    for idx, (img, mask) in enumerate(zip(images, masks)):
        try:
            cv2.imwrite(
                f"{base_dir}/{split_type}/images/{idx:04d}.png",
                (img * 255).astype(np.uint8)
            )
            cv2.imwrite(
                f"{base_dir}/{split_type}/masks/{idx:04d}.png",
                mask.astype(np.uint8)
            )
        except Exception as e:
            print(f"Error saving {idx:04d}: {str(e)}")


# Main execution
if __name__ == "__main__":
    image_dir = "data/raw/segmentation/images"
    mask_dir = "data/raw/segmentation/masks"

    # Load and validate data
    images, masks = load_and_validate_images(image_dir, mask_dir)
    print(f"Loaded {len(images)} valid image-mask pairs.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, masks, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Save processed data
    save_processed_data(X_train, y_train, "train")
    save_processed_data(X_val, y_val, "val")
    save_processed_data(X_test, y_test, "test")
    print("Preprocessing complete!")
