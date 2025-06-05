# Composed Image Retrieval Demo

This directory contains demo scripts for performing Composed Image Retrieval (CIR) using the SEARLE framework. The scripts allow you to:

1. Create a searchable database from image datasets (CIRR, CIRCO, FashionIQ)
2. Perform queries using a reference image + text modification
3. Retrieve the most similar images based on the composed query

## Files

- `compose_image_retrieval_demo.py` - Main CIR system implementation
- `cir_example.py` - Example usage scripts
- `CIR_DEMO_README.md` - This documentation

## Quick Start

### 1. Basic Usage

Create a database and perform a query:

```bash
# Create database and perform a single query
python src/compose_image_retrieval_demo.py \
  --dataset-path /path/to/cirr/dataset \
  --dataset-type cirr \
  --eval-type searle \
  --reference-image /path/to/reference.jpg \
  --caption "is wearing a red shirt" \
  --top-k 5
```

### 2. Save Database for Reuse

```bash
# Create and save database
python src/compose_image_retrieval_demo.py \
  --dataset-path /path/to/cirr/dataset \
  --dataset-type cirr \
  --eval-type searle \
  --save-database cirr_val_database.pt

# Later, load the saved database for querying
python src/compose_image_retrieval_demo.py \
  --dataset-path /path/to/cirr/dataset \
  --dataset-type cirr \
  --eval-type searle \
  --load-database cirr_val_database.pt \
  --reference-image /path/to/reference.jpg \
  --caption "without glasses" \
  --top-k 10
```

## Supported Evaluation Types

### 1. SEARLE (Recommended)
Uses the pre-trained SEARLE model:
```bash
--eval-type searle  # Uses ViT-B/32 backbone
--eval-type searle-xl  # Uses ViT-L/14 backbone (better quality)
```

### 2. Phi Network
Uses a trained Phi network (requires pre-trained model):
```bash
--eval-type phi \
--exp-name your_phi_experiment \
--phi-checkpoint-name phi_20.pt
```

### 3. OTI (Optimization-based Textual Inversion)
Uses pre-computed OTI pseudo tokens:
```bash
--eval-type oti \
--exp-name your_oti_experiment
```

## Dataset Support

### CIRR Dataset
```bash
--dataset-type cirr \
--dataset-path /path/to/cirr
```

### CIRCO Dataset
```bash
--dataset-type circo \
--dataset-path /path/to/circo
```

### FashionIQ Dataset
```bash
--dataset-type fashioniq \
--dataset-path /path/to/fashioniq
```

## Command Line Arguments

### Required Arguments
- `--dataset-path`: Path to the dataset directory
- `--dataset-type`: Type of dataset (`cirr`, `circo`, `fashioniq`)

### Model Configuration
- `--clip-model-name`: CLIP model to use (default: `ViT-B/32`)
- `--eval-type`: Evaluation method (`searle`, `searle-xl`, `phi`, `oti`)
- `--preprocess-type`: Preprocessing pipeline (`clip`, `targetpad`)

### Database Options
- `--split`: Dataset split for database (`train`, `val`, `test`)
- `--save-database`: Path to save the created database
- `--load-database`: Path to load a pre-saved database

### Query Options
- `--reference-image`: Path to reference image for query
- `--caption`: Text describing the desired modification
- `--top-k`: Number of top results to return (default: 10)

### Phi-specific Options (when using `--eval-type phi`)
- `--exp-name`: Name of the Phi experiment
- `--phi-checkpoint-name`: Phi checkpoint filename

## Example Queries

### CIRR Dataset Examples
```bash
# Person wearing different clothing
--caption "is wearing a blue jacket instead"

# Remove/add accessories
--caption "without sunglasses"
--caption "wearing a hat"

# Change pose or action
--caption "is sitting instead of standing"
--caption "is smiling"
```

### FashionIQ Examples
```bash
# Color changes
--caption "is red instead of blue"

# Pattern changes  
--caption "has stripes instead of solid color"

# Style modifications
--caption "is shorter"
--caption "has long sleeves"
```

### CIRCO Examples
```bash
# Object modifications
--caption "is smaller"
--caption "has a different color"

# Scene changes
--caption "in a different setting"
--caption "with more objects"
```

## Programmatic Usage

You can also use the `ComposedImageRetrievalSystem` class directly in your Python code:

```python
from compose_image_retrieval_demo import ComposedImageRetrievalSystem

# Initialize the system
cir_system = ComposedImageRetrievalSystem(
    dataset_path="/path/to/cirr",
    dataset_type="cirr",
    clip_model_name="ViT-B/32",
    eval_type="searle"
)

# Create database
cir_system.create_database(split='val')

# Perform query
results = cir_system.query(
    reference_image_path="/path/to/reference.jpg",
    relative_caption="is wearing a red dress",
    top_k=5
)

# Print results
for i, (image_name, score) in enumerate(results, 1):
    print(f"{i}. {image_name} (similarity: {score:.4f})")
```

## Performance Notes

### Model Loading
- SEARLE models are downloaded automatically on first use
- CLIP models are also downloaded automatically
- GPU is used if available, CPU otherwise

### Database Creation
- Creating databases can take several minutes for large datasets
- Save databases to disk to avoid recreating them
- Databases contain pre-computed image features for fast querying

### Memory Usage
- Large datasets may require significant GPU memory
- Consider using smaller batch sizes if you encounter memory issues
- Database features are normalized and stored efficiently

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Dataset not found**: Check dataset path and structure
3. **Model download fails**: Check internet connection
4. **Image not found**: Verify reference image path exists

### Dataset Structure

Ensure your datasets follow the expected structure:

**CIRR:**
```
cirr/
├── cirr/
│   ├── captions/
│   ├── image_splits/
│   └── ...
└── [image directories]
```

**FashionIQ:**
```
fashioniq/
├── captions/
├── image_splits/
├── images/
└── ...
```

## Tips for Best Results

1. **Use SEARLE-XL** for best quality results
2. **Use targetpad preprocessing** for better performance
3. **Save databases** for frequently used datasets
4. **Use specific captions** for better retrieval accuracy
5. **Experiment with different top-k values** based on your needs

## Dependencies

Make sure you have the required packages installed:
- torch
- clip-by-openai
- PIL/Pillow
- numpy
- tqdm
- pandas (for some dataset operations)

The SEARLE repository should have all necessary dependencies listed in their requirements. 