#!/bin/bash

# Example usage of SEARLE CIR with Grad-ECLIP Saliency Maps
# This script demonstrates how to run the saliency-enabled inference

echo "🚀 Running SEARLE CIR with Grad-ECLIP Saliency Maps Example"
echo "============================================================"

# Configuration
DATABASE_PATH="/home/ikapetan/Frameworks/Projects-Master/MMA/dbs/imagenet-r-database"
REFERENCE_IMAGE="/home/ikapetan/Frameworks/Projects-Master/MMA/SEARLE/src/example_scripts/banana.jpg"
CAPTION="as a cartoon character with other cartoon fruits around it"
DATASET_PATH="/home/ikapetan/Frameworks/Projects-Master/MMA/data/imagenet-r"
OUTPUT_DIR="/home/ikapetan/Frameworks/Projects-Master/MMA/SEARLE/saliency_output_2"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📋 Configuration:"
echo "  Database: $DATABASE_PATH"
echo "  Reference Image: $REFERENCE_IMAGE"
echo "  Caption: '$CAPTION'"
echo "  Dataset Path: $DATASET_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Run the saliency-enabled CIR script
echo "🔍 Running inference with saliency generation..."
python3 simple_cir_inference_with_saliency.py \
    --database-path "$DATABASE_PATH" \
    --reference-image "$REFERENCE_IMAGE" \
    --caption "$CAPTION" \
    --top-k 10 \
    --dataset-path "$DATASET_PATH" \
    --generate-saliency \
    --generate-reference-saliency \
    --generate-candidate-saliency \
    --max-candidate-saliency 3 \
    --save-saliency-dir "$OUTPUT_DIR" \
    --output-format text

echo ""
echo "🎉 Inference complete!"
echo "📁 Check the output directory for saliency visualizations: $OUTPUT_DIR"
echo ""
echo "📊 Generated files should include:"
echo "  - reference_heatmap.png: Reference image with saliency overlay"
echo "  - reference_saliency.npy: Raw reference saliency map"
echo "  - result_*_heatmap_*.png: Candidate images with saliency overlays"
echo "  - result_*_saliency_*.npy: Raw candidate saliency maps" 