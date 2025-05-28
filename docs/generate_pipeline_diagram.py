
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# Create directories
Path("docs").mkdir(exist_ok=True)

# Create a simple pipeline diagram
fig, ax = plt.subplots(1, 5, figsize=(15, 3))

# Generate sample images for the pipeline
# 1. Original image (simulated skin lesion)
img_size = (200, 200)
original = np.ones((img_size[0], img_size[1], 3)) * 0.8  # Light skin tone
# Create an elliptical lesion
center = (img_size[0]//2, img_size[1]//2)
axes = (60, 40)
angle = 30
# Draw the lesion
for i in range(img_size[0]):
    for j in range(img_size[1]):
        # Calculate if point is inside ellipse
        dist = ((i - center[0]) * np.cos(np.radians(angle)) + (j - center[1]) * np.sin(np.radians(angle)))**2 / axes[0]**2 + \
               ((i - center[0]) * np.sin(np.radians(angle)) - (j - center[1]) * np.cos(np.radians(angle)))**2 / axes[1]**2
        if dist <= 1:
            # Dark brown lesion with some variation
            original[i, j, :] = [0.3 + np.random.rand() * 0.1, 
                                0.2 + np.random.rand() * 0.1, 
                                0.1 + np.random.rand() * 0.1]

# 2. Segmentation mask
mask = np.zeros((img_size[0], img_size[1], 3))
for i in range(img_size[0]):
    for j in range(img_size[1]):
        dist = ((i - center[0]) * np.cos(np.radians(angle)) + (j - center[1]) * np.sin(np.radians(angle)))**2 / axes[0]**2 + \
               ((i - center[0]) * np.sin(np.radians(angle)) - (j - center[1]) * np.cos(np.radians(angle)))**2 / axes[1]**2
        if dist <= 1:
            mask[i, j, :] = [1, 1, 1]

# 3. Masked image
masked = original.copy()
masked = masked * mask

# 4. Enhanced image (simulated SRGAN output)
enhanced = masked.copy()
# Add more details to simulate enhancement
for i in range(img_size[0]):
    for j in range(img_size[1]):
        dist = ((i - center[0]) * np.cos(np.radians(angle)) + (j - center[1]) * np.sin(np.radians(angle)))**2 / axes[0]**2 + \
               ((i - center[0]) * np.sin(np.radians(angle)) - (j - center[1]) * np.cos(np.radians(angle)))**2 / axes[1]**2
        if dist <= 1:
            # Add texture details
            if i % 10 < 5 and j % 10 < 5:
                enhanced[i, j, :] = enhanced[i, j, :] * 1.2
            elif i % 8 < 4 and j % 8 < 4:
                enhanced[i, j, :] = enhanced[i, j, :] * 0.9
            
            # Add border details
            if 0.9 <= dist <= 1:
                enhanced[i, j, :] = [0.4, 0.3, 0.2]

# 5. Final result (enhanced lesion on original background)
final = original.copy()
final = final * (1 - mask) + enhanced * mask

# Plot the pipeline
ax[0].imshow(original)
ax[0].set_title('1. Original Image')
ax[0].axis('off')

ax[1].imshow(mask)
ax[1].set_title('2. Segmentation Mask')
ax[1].axis('off')

ax[2].imshow(masked)
ax[2].set_title('3. Masked Lesion')
ax[2].axis('off')

ax[3].imshow(enhanced)
ax[3].set_title('4. Enhanced (SRGAN)')
ax[3].axis('off')

ax[4].imshow(final)
ax[4].set_title('5. Final Result')
ax[4].axis('off')

# Add arrows between images
for i in range(4):
    plt.annotate('', xy=(0.2 + i*0.2, 0.5), xytext=(0.1 + i*0.2, 0.5),
                xycoords='figure fraction', textcoords='figure fraction',
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

plt.tight_layout()
plt.savefig('docs/pipeline_overview.png', dpi=150, bbox_inches='tight')

# Create example results image
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

# Simulate different lesion types
def create_lesion(img_size, center, axes, angle, color_base, texture_type='random'):
    img = np.ones((img_size[0], img_size[1], 3)) * 0.8  # Light skin tone
    
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            # Calculate if point is inside ellipse
            dist = ((i - center[0]) * np.cos(np.radians(angle)) + (j - center[1]) * np.sin(np.radians(angle)))**2 / axes[0]**2 + \
                   ((i - center[0]) * np.sin(np.radians(angle)) - (j - center[1]) * np.cos(np.radians(angle)))**2 / axes[1]**2
            if dist <= 1:
                # Base color with variation
                if texture_type == 'random':
                    img[i, j, :] = [color_base[0] + np.random.rand() * 0.1, 
                                    color_base[1] + np.random.rand() * 0.1, 
                                    color_base[2] + np.random.rand() * 0.1]
                elif texture_type == 'pattern':
                    if (i + j) % 10 < 5:
                        img[i, j, :] = [color_base[0] * 1.2, color_base[1] * 1.2, color_base[2] * 1.2]
                    else:
                        img[i, j, :] = color_base
                elif texture_type == 'gradient':
                    factor = 1 - dist
                    img[i, j, :] = [color_base[0] * factor, color_base[1] * factor, color_base[2] * factor]
    
    return img

# Create different lesion types
lesion1 = create_lesion(img_size, (100, 100), (50, 40), 0, [0.6, 0.3, 0.2], 'random')  # Reddish-brown
lesion2 = create_lesion(img_size, (100, 100), (60, 30), 45, [0.2, 0.2, 0.2], 'pattern')  # Dark with pattern
lesion3 = create_lesion(img_size, (100, 100), (40, 40), 0, [0.1, 0.1, 0.5], 'gradient')  # Blue with gradient

# Add noise and blur to simulate low quality
def degrade_image(img):
    # Add noise
    noise = np.random.normal(0, 0.05, img.shape)
    img_noisy = np.clip(img + noise, 0, 1)
    
    # Add blur
    img_blurry = cv2.GaussianBlur(img_noisy, (5, 5), 2)
    
    return img_blurry

# Degrade images
lesion1_degraded = degrade_image(lesion1)
lesion2_degraded = degrade_image(lesion2)
lesion3_degraded = degrade_image(lesion3)

# Simulate enhanced versions
def enhance_image(img):
    # Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    img_sharp = cv2.filter2D(img, -1, kernel)
    
    # Enhance contrast
    img_enhanced = np.clip((img_sharp - 0.5) * 1.3 + 0.5, 0, 1)
    
    return img_enhanced

# Enhance images
lesion1_enhanced = enhance_image(lesion1)
lesion2_enhanced = enhance_image(lesion2)
lesion3_enhanced = enhance_image(lesion3)

# Plot example results
ax[0, 0].imshow(lesion1_degraded)
ax[0, 0].set_title('Original (Type 1)')
ax[0, 0].axis('off')

ax[0, 1].imshow(lesion2_degraded)
ax[0, 1].set_title('Original (Type 2)')
ax[0, 1].axis('off')

ax[0, 2].imshow(lesion3_degraded)
ax[0, 2].set_title('Original (Type 3)')
ax[0, 2].axis('off')

ax[1, 0].imshow(lesion1_enhanced)
ax[1, 0].set_title('Enhanced (Type 1)')
ax[1, 0].axis('off')

ax[1, 1].imshow(lesion2_enhanced)
ax[1, 1].set_title('Enhanced (Type 2)')
ax[1, 1].axis('off')

ax[1, 2].imshow(lesion3_enhanced)
ax[1, 2].set_title('Enhanced (Type 3)')
ax[1, 2].axis('off')

plt.tight_layout()
plt.savefig('docs/example_results.png', dpi=150, bbox_inches='tight')

print("Pipeline diagrams generated successfully!")
