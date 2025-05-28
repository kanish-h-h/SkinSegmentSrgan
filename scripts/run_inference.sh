#!/bin/bash

# Default paths
INPUT_DIR="data/raw/test_images"
SEG_MODEL="models/segmentation/best_model.h5"
SRGAN_MODEL="models/srgan/generator_epoch_100.h5"
OUTPUT_DIR="results/pipeline"

# Help function
function show_help {
  echo "Usage: $0 [OPTIONS]"
  echo "Run the skin disease segmentation + SRGAN pipeline on test images."
  echo ""
  echo "Options:"
  echo "  -i, --input DIR      Input directory containing test images"
  echo "  -s, --seg-model PATH Path to segmentation model weights"
  echo "  -g, --srgan-model PATH Path to SRGAN generator model weights"
  echo "  -o, --output DIR     Output directory for results"
  echo "  -h, --help           Show this help message"
  echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  -i | --input)
    INPUT_DIR="$2"
    shift
    shift
    ;;
  -s | --seg-model)
    SEG_MODEL="$2"
    shift
    shift
    ;;
  -g | --srgan-model)
    SRGAN_MODEL="$2"
    shift
    shift
    ;;
  -o | --output)
    OUTPUT_DIR="$2"
    shift
    shift
    ;;
  -h | --help)
    show_help
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    show_help
    exit 1
    ;;
  esac
done

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory '$INPUT_DIR' does not exist."
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python CLI script
echo "Running skin segmentation + SRGAN pipeline..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Segmentation model: $SEG_MODEL"
echo "SRGAN model: $SRGAN_MODEL"

python -m src.inference.cli \
  --directory "$INPUT_DIR" \
  --seg-model "$SEG_MODEL" \
  --srgan-model "$SRGAN_MODEL" \
  --output-dir "$OUTPUT_DIR"

echo "Pipeline completed. Results saved to $OUTPUT_DIR"
