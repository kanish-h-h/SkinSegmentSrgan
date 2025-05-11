# Skin Disease Segmentation + SRGAN Enhancement
**Segment skin diseases from low-contrast images and enhance resolution using SRGAN.**

## 📌 Features
- **U-Net** for precise disease segmentation.
- **SRGAN** for 4x super-resolution of segmented regions.
- CLI interface for inference.

## 🛠️ Installation
1. Clone the repo:
  ```bash
   git clone https://github.com/yourusername/skin-disease-segmentation-srgan.git
   ```

2. Install Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

# 🚀 Usage

## Inference (CLI):
  ```bash
  skinsegmentsrgan \
    --input_img path/to/image.jpg \
    --output_dir results/ \
    --seg_model models/trained/unet.h5 \
    --srgan_model models/trained/srgan.h5
  ```

# 📂 Directory Structure
```
skin-disease-segmentation-srgan/  
├── data/               # Raw/processed datasets  
├── models/             # Pretrained/trained models  
├── src/                # Source code  
└── outputs/            # Generated results  
```
📝 Research Paper
Work in progress (to be published).

---

### **Key Notes**  
1. **`.gitignore`**: Excludes large datasets/models but keeps empty directories (via `.gitkeep`).  
2. **GitHub Workflows**:  
   - `ci.yml`: Runs tests on every push/PR.  
   - `deploy.yml`: Auto-publishes to PyPI on new releases.  
3. **`setup.py`**:  
   - Packages only the `src/` directory.  
   - CLI command maps to `src/inference/cli.py`.  
4. **`README.md`**:  
   - Includes minimal setup/usage instructions.  
   - Update with paper details later.  
