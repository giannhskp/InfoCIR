# Simple Composed Image Retrieval Inference

This directory contains a simple inference script for Composed Image Retrieval (CIR) that allows you to query a pre-created database with a reference image and text description.

## Quick Start

### 1. Create a Database (One-time setup)

First, create a database from your dataset:

```bash
# For FashionIQ dataset
python compose_image_retrieval_demo.py \
  --dataset-path /path/to/fashioniq \
  --dataset-type fashioniq \
  --eval-type searle \
  --split val \
  --save-database fashioniq_val_database.pt

# For CIRR dataset  
python compose_image_retrieval_demo.py \
  --dataset-path /path/to/cirr \
  --dataset-type cirr \
  --eval-type searle \
  --split val \
  --save-database cirr_val_database.pt
```

### 2. Perform Inference

Use the simple inference script to query the database:

```bash
python simple_cir_inference.py \
  --database-path fashioniq_val_database.pt \
  --reference-image /path/to/reference/image.jpg \
  --caption "white with stripes" \
  --top-k 10
```

## Script Overview

### `simple_cir_inference.py`

The main inference script that:
- ✅ Loads a pre-created database
- ✅ Takes a reference image and text caption as input  
- ✅ Returns top-k most similar images
- ✅ Supports both text and JSON output formats
- ✅ **Visual grid display** with reference + top-k images
- ✅ Color-coded similarity borders and smart image resolution
- ✅ Validates inputs and provides helpful error messages

### `example_scripts/inference_example.py`

Example script showing:
- 📖 How to use the inference script programmatically
- 📖 Command-line usage examples
- 📖 Complete workflow from database creation to inference
- 📖 Dataset-specific query examples

## Usage

### Command Line Arguments

**Required:**
- `--database-path`: Path to the saved database file (.pt)
- `--reference-image`: Path to the reference image
- `--caption`: Text describing the desired modification

**Optional:**
- `--top-k`: Number of top results to return (default: 10)
- `--clip-model-name`: CLIP model name (default: ViT-B/32)
- `--eval-type`: Evaluation type (default: searle)
- `--preprocess-type`: Preprocessing type (default: targetpad)  
- `--output-format`: Output format - text or json (default: text)

**Visualization:**
- `--display-grid`: Display results in a visual grid
- `--dataset-path`: Path to dataset (needed for displaying result images)
- `--save-grid`: Path to save the results grid image (e.g., results.png)
- `--no-show`: Don't show the plot window (useful when only saving)

### Example Commands

#### Basic Fashion Query
```bash
python simple_cir_inference.py \
  --database-path fashioniq_shirt_database.pt \
  --reference-image shirt_image.jpg \
  --caption "has long sleeves instead of short" \
  --top-k 5
```

#### JSON Output
```bash
python simple_cir_inference.py \
  --database-path cirr_database.pt \
  --reference-image person_image.jpg \
  --caption "is wearing a red dress instead" \
  --output-format json \
  --top-k 3
```

#### Using SEARLE-XL Model
```bash
python simple_cir_inference.py \
  --database-path database.pt \
  --reference-image image.jpg \
  --caption "is more formal" \
  --eval-type searle-xl \
  --clip-model-name "ViT-L/14"
```

#### Grid Visualization
```bash
python simple_cir_inference.py \
  --database-path fashioniq_database.pt \
  --reference-image shirt.jpg \
  --caption "white with stripes" \
  --top-k 8 \
  --display-grid \
  --dataset-path /path/to/fashioniq
```

#### Save Grid Without Display
```bash
python simple_cir_inference.py \
  --database-path cirr_database.pt \
  --reference-image person.jpg \
  --caption "is wearing a red dress" \
  --display-grid \
  --dataset-path /path/to/cirr \
  --save-grid results.png \
  --no-show
```

## Programmatic Usage

You can also use the inference system programmatically:

```python
from simple_cir_inference import SimpleCIRInference

# Initialize the inference system
inference = SimpleCIRInference(
    database_path="fashioniq_val_database.pt",
    clip_model_name="ViT-B/32",
    eval_type="searle"
)

# Perform a query
results = inference.query(
    reference_image_path="reference_image.jpg",
    caption="is shorter",
    top_k=10
)

# Process results
for rank, (image_name, score) in enumerate(results, 1):
    print(f"{rank}. {image_name} (similarity: {score:.4f})")

# Display visual grid (optional)
inference.display_results_grid(
    reference_image_path="reference_image.jpg",
    caption="is shorter",
    results=results,
    dataset_base_path="/path/to/dataset",  # For resolving image paths
    save_path="results_grid.png",  # Optional: save grid
    show_plot=True  # Show interactive plot
)
```

## Dataset-Specific Examples

### FashionIQ (Fashion Items)
```bash
# Style modifications
--caption "is more formal"
--caption "is more casual"

# Color changes
--caption "is black instead of white"
--caption "has stripes instead of solid color"

# Clothing features
--caption "has long sleeves instead of short"
--caption "is sleeveless"
--caption "is shorter"
--caption "has a belt"
```

### CIRR (Real-world Images)
```bash
# Clothing modifications
--caption "is wearing a red shirt instead"
--caption "is wearing a dress instead of pants"

# Appearance changes  
--caption "without glasses"
--caption "has dark hair instead of blonde"
--caption "is smiling"

# Scene modifications
--caption "is outdoors instead of indoors"
--caption "in a different pose"
```

### CIRCO (Object-centric)
```bash
# Color changes
--caption "is red instead of blue"
--caption "is yellow"

# Size modifications
--caption "is larger"
--caption "is smaller"

# Material changes
--caption "is wooden instead of metal"
--caption "has different texture"

# Position changes
--caption "is on a table instead of floor"
--caption "is in the background"
```

## Output Formats

### Text Output (Default)
```
📋 Top 5 Results:
============================================================
 1. image_001.jpg                         (similarity: 0.8542)
 2. image_023.jpg                         (similarity: 0.8234)
 3. image_045.jpg                         (similarity: 0.8156)
 4. image_067.jpg                         (similarity: 0.8089)
 5. image_089.jpg                         (similarity: 0.7934)
```

### JSON Output
```json
{
  "query": {
    "reference_image": "reference.jpg",
    "caption": "is shorter",
    "top_k": 5
  },
  "results": [
    {"rank": 1, "image_name": "image_001.jpg", "similarity_score": 0.8542},
    {"rank": 2, "image_name": "image_023.jpg", "similarity_score": 0.8234},
    {"rank": 3, "image_name": "image_045.jpg", "similarity_score": 0.8156},
    {"rank": 4, "image_name": "image_067.jpg", "similarity_score": 0.8089},
    {"rank": 5, "image_name": "image_089.jpg", "similarity_score": 0.7934}
  ]
}
```

## Tips for Best Results

1. **Model Consistency**: Use the same CLIP model and evaluation type that were used to create the database
2. **Relative Descriptions**: Write captions that describe changes relative to the reference image
3. **Specific Language**: Use specific, descriptive language rather than vague terms
4. **Database Reuse**: Once created, databases can be reused for multiple queries
5. **Top-K Selection**: Experiment with different top-k values to get the right number of results

## Grid Visualization Features

The script includes a powerful grid visualization feature that displays:

### 🖼️ Visual Layout
- **Reference Image**: Displayed with blue border in top-left position
- **Top-K Results**: Arranged in a grid layout (max 5 columns)
- **Adaptive Grid**: Automatically adjusts based on number of results

### 🎨 Color-Coded Borders
- 🟦 **Blue**: Reference image
- 🟢 **Green**: High similarity (>0.8)
- 🟠 **Orange**: Medium similarity (>0.6)
- 🔴 **Red**: Lower similarity (≤0.6)

### 📊 Information Display
- **Similarity Scores**: Shown under each result image
- **Query Caption**: Displayed as main title
- **Image Names**: Truncated filenames as subtitles
- **Ranking**: Clear numbered ranking (#1, #2, etc.)

### 💾 Save Options
- Save grid as high-resolution PNG
- Optional display (useful for batch processing)
- Customizable output paths

## Troubleshooting

### Common Issues

**"Database file not found"**
- Ensure the database path is correct
- Create the database first using `compose_image_retrieval_demo.py`

**"Reference image not found"**  
- Check that the image path exists and is accessible
- Supported formats: JPG, PNG, etc.

**"Model mismatch warnings"**
- The database was created with different model settings
- Either recreate the database or use matching model parameters

**"Query failed"**
- Check that the caption is not empty
- Ensure the reference image is valid and readable

**"Grid visualization not working"**
- Install matplotlib: `pip install matplotlib`
- Provide `--dataset-path` for image resolution
- Check that dataset contains `images/` directory
- Use `--save-grid` if display issues occur

**"Images not showing in grid"**
- Verify dataset path is correct
- Check image file extensions (.jpg, .png, etc.)
- Ensure images are readable and not corrupted

### Getting Help

Run the example script to see usage patterns:
```bash
python example_scripts/inference_example.py
```

For more detailed examples and explanations, check the example scripts in the `example_scripts/` directory. 