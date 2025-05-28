import cv2
import numpy as np

def generate_lr_image(hr_image, scale_factor=4):
    """Generate a low-resolution image from a high-resolution image using bicubic downsampling."""
    if hr_image is None:
        raise ValueError("Input image is None. Check if the file path is correct and the image exists.")
    
    h, w = hr_image.shape[:2]
    lr_image = cv2.resize(
        hr_image,
        (w // scale_factor, h // scale_factor),
        interpolation=cv2.INTER_CUBIC
    )
    return lr_image

# Usage example
image_path = '0161.png'
hr_image = cv2.imread(image_path)

if hr_image is None:
    print(f"Failed to load image from path: {image_path}")
else:
    lr_image = generate_lr_image(hr_image, scale_factor=4)
    cv2.imwrite('0161_lr.png', lr_image)

