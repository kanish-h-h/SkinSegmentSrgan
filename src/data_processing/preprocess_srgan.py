import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_hr_images(hr_dir, target_size=(256, 256)):
    """Load and validate high-res images."""
    hr_images = []
    valid_files = []

    for filename in sorted(os.listdir(hr_dir)):
        hr_path = os.path.join(hr_dir, filename)
        hr_img = cv2.imread(hr_path)

        if hr_img is None:
            print(f"Warning: Failed to load HR image {hr_path}")
            continue

        # Resize to target dimensions if needed
        if hr_img.shape[:2] != target_size:
            hr_img = cv2.resize(hr_img, target_size)

        hr_images.append(hr_img)
        valid_files.append(filename)

    return np.array(hr_images), valid_files


def generate_lr_images(hr_images, scale_factor=4):
    """Generate low-res images from high-res."""
    lr_images = []
    for hr_img in hr_images:
        h, w = hr_img.shape[:2]
        lr_img = cv2.resize(
            hr_img,
            (w // scale_factor, h // scale_factor),
            interpolation=cv2.INTER_CUBIC
        )
        lr_images.append(lr_img)
    return np.array(lr_images)


def save_srgan_pairs(lr_images, hr_images, filenames, split_type, base_dir="data/processed/srgan"):
    """Save LR/HR pairs with proper naming."""
    os.makedirs(f"{base_dir}/{split_type}/lr", exist_ok=True)
    os.makedirs(f"{base_dir}/{split_type}/hr", exist_ok=True)

    for idx, (lr, hr, filename) in enumerate(zip(lr_images, hr_images, filenames)):
        try:
            base_name = os.path.splitext(filename)[0]

            # Save HR (e.g., data/processed/srgan/train/hr/ISIC_001.png)
            hr_path = f"{base_dir}/{split_type}/hr/{base_name}.png"
            cv2.imwrite(hr_path, hr)

            # Save LR (e.g., data/processed/srgan/train/lr/ISIC_001_x4.png)
            lr_path = f"{base_dir}/{split_type}/lr/{base_name}_x4.png"
            cv2.imwrite(lr_path, lr)

        except Exception as e:
            print(f"Error saving {base_name}: {str(e)}")


if __name__ == "__main__":
    hr_dir = "data/raw/srgan/HR"

    # Load and validate HR images
    hr_images, valid_files = load_hr_images(hr_dir)
    print(f"Loaded {len(hr_images)} valid HR images.")

    # Generate LR images
    lr_images = generate_lr_images(hr_images)

    # Split into train/val/test (80/10/10)
    indices = np.arange(len(hr_images))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, random_state=42)

    # Save processed pairs
    save_srgan_pairs(
        lr_images[train_idx], hr_images[train_idx],
        np.array(valid_files)[train_idx], "train"
    )
    save_srgan_pairs(
        lr_images[val_idx], hr_images[val_idx],
        np.array(valid_files)[val_idx], "val"
    )
    save_srgan_pairs(
        lr_images[test_idx], hr_images[test_idx],
        np.array(valid_files)[test_idx], "test"
    )

    print("SRGAN preprocessing complete!")
