
from src.inference.pipeline import SkinSegmentSRGANPipeline
import argparse
import os
from pathlib import Path
import sys
import glob
from tqdm import tqdm

# Add parent directory to path for importing modules
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Skin Disease Segmentation + SRGAN Enhancement Pipeline"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input",
        help="Path to input image"
    )
    input_group.add_argument(
        "-d", "--directory",
        help="Directory containing multiple images to process"
    )

    # Model paths
    parser.add_argument(
        "--seg-model",
        default="models/segmentation/best_model.h5",
        help="Path to segmentation model weights"
    )
    parser.add_argument(
        "--srgan-model",
        default="models/srgan/generator_epoch_100.h5",
        help="Path to SRGAN generator model weights"
    )

    # Output options
    parser.add_argument(
        "-o", "--output-dir",
        default="results/pipeline",
        help="Directory to save results"
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable visualization of results"
    )

    return parser.parse_args()


def process_single_image(pipeline, image_path, visualize=True):
    """Process a single image through the pipeline"""
    print(f"Processing {image_path}...")
    try:
        results = pipeline.process_image(image_path, visualize=visualize)
        print(f"Successfully processed {image_path}")
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False


def process_directory(pipeline, directory, visualize=True):
    """Process all images in a directory"""
    # Get supported image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        image_files.extend(
            glob.glob(os.path.join(directory, f"*{ext.upper()}")))

    if not image_files:
        print(f"No supported image files found in {directory}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process each image
    success_count = 0
    for image_path in tqdm(image_files):
        if process_single_image(pipeline, image_path, visualize):
            success_count += 1

    print(
        f"Successfully processed {success_count} out of {len(image_files)} images")


def main():
    """Main function for CLI"""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize pipeline
    pipeline = SkinSegmentSRGANPipeline(
        seg_model_path=args.seg_model,
        srgan_model_path=args.srgan_model
    )

    # Set output directory for the pipeline
    pipeline.output_dir = Path(args.output_dir)

    # Process input
    if args.input:
        process_single_image(pipeline, args.input, visualize=not args.no_vis)
    elif args.directory:
        process_directory(pipeline, args.directory, visualize=not args.no_vis)


if __name__ == "__main__":
    main()
