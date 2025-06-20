import os
from pathlib import Path

# UI CONFIGURATION
IMAGE_GALLERY_SIZE = 24
IMAGE_GALLERY_ROW_SIZE = 4

WORDCLOUD_IMAGE_HEIGHT = 600
WORDCLOUD_IMAGE_WIDTH = 800

SCATTERPLOT_COLOR = 'rgba(31, 119, 180, 0.5)'
SCATTERPLOT_SELECTED_COLOR = 'red'

# Colors for selected image and neighbors in scatterplot
SELECTED_IMAGE_COLOR = 'green'  # Color for the specifically selected image
SELECTED_CLASS_COLOR = 'red'    # Color for other images of the same class as selected image

MAX_IMAGES_ON_SCATTERPLOT = 100
DEFAULT_PROJECTION = 'UMAP'

# Server configuration
PORT = 8052  # Port number for the Dash application

# Dataset configuration
DATASET_SAMPLE_SIZE = 30000  # Sample size from dataset

# NEW UMAP CONFIGURATION - Enhanced UMAP with debiasing techniques
USE_NEW_UMAP = True  # Set to True to use the new enhanced UMAP implementation, False for original

# NEW UMAP Parameters (based on optimal command)
NEW_UMAP_CONFIG = {
    # Basic UMAP parameters
    'pca_components': 80,
    'n_neighbors': 25,
    'min_dist': 0.1,
    'spread': 55.0,
    'target_weight': 0.3,
    'local_connectivity': 3.0,
    'n_epochs': 1200,
    
    # Debiasing techniques
    'style_debiasing': True,
    'contrastive_debiasing': True,
    'contrastive_weight': 0.7,
    'alternative_projection': 'ica',
    'semantic_enhancement': False,
    'augment_embeddings': False,
  
    'force_approximation_algorithm': False,  # Force UMAP approximation algorithm
    'enhanced_parameter_tuning': True,  # Apply enhanced parameter tuning for semantic clustering
    'calculate_quality_metrics': True,  # Calculate comprehensive quality metrics
    'use_hdbscan': False,  # Apply HDBSCAN clustering post-processing
    'hdbscan_min_cluster_size': 10,  # Minimum cluster size for HDBSCAN
}

# Path configuration
# You should set DATASET_ROOT_PATH to point to your actual dataset directory
DATASET_ROOT_PATH = '/home/scur1151/multimedia-analytics/imagenet-r'

# Working directory for processed data (relative to cir_app/)
APP_DIR = Path(__file__).parent.parent  # cir_app/
WORK_DIR = APP_DIR / 'data'  # cir_app/data/

# Dataset paths (where your actual dataset is located)
IMAGES_DIR = DATASET_ROOT_PATH  # Your dataset images
CLASS_NAMES_PATH = os.path.join(DATASET_ROOT_PATH, 'class_names.csv')  # Your class names file

# Generated/processed data paths (where the app stores its processed data)
DATASET_PATH = WORK_DIR / 'dataset.csv'
AUGMENTED_DATASET_PATH = WORK_DIR / 'augmented_dataset.csv'

# CLIP model configuration
CLIP_MODEL_NAME = 'ViT-B/32'  # CLIP model to use for image retrieval

# Prompt enhancement configuration
ENHANCEMENT_CANDIDATE_PROMPTS = 10  # Number of candidate prompts to generate for enhancement

# SEARLE CIR configuration
CIR_DATASET_PATH = AUGMENTED_DATASET_PATH  # Path to CIR augmented dataset CSV
CIR_DATASET_TYPE = 'imagenet-r'            # Dataset type: 'cirr', 'circo', or 'fashioniq'
CIR_CLIP_MODEL_NAME = CLIP_MODEL_NAME # CLIP model to use for CIR
CIR_EVAL_TYPE = 'searle'             # Evaluation type: 'searle', 'searle-xl', 'phi', 'oti'
CIR_PREPROCESS_TYPE = 'targetpad'     # Preprocessing type: 'targetpad' or 'clip'
CIR_EXP_NAME = None                  # Experiment name for phi/oti if needed
CIR_PHI_CHECKPOINT_NAME = None       # Phi checkpoint name if using phi evaluation
CIR_SPLIT = 'val'                    # Dataset split for building the database

# CIR search visualization colors
QUERY_COLOR = 'magenta'
FINAL_QUERY_COLOR = 'cyan'  # Color for the final composed query (image + text)
TOP_K_COLOR = 'orange'
TOP_1_COLOR = 'red' 

CIR_FEATURES_PATH = WORK_DIR / 'features'  # Path to store CIR features
CIR_LOAD_FEATURES = True  # Whether to load precomputed features or not
CIR_FREEDOM_FEATURES_PATH = WORK_DIR / 'clip_features'  # Path for Freedom features

# Saliency configuration
SALIENCY_ENABLED = True  # Whether to generate saliency maps for CIR queries
SALIENCY_OUTPUT_DIR = WORK_DIR / 'saliency_output'  # Directory to save saliency visualizations
SALIENCY_MAX_CANDIDATES = None  # Maximum number of candidates to generate saliency maps for (set to None for all top-k)
SALIENCY_GENERATE_REFERENCE = True  # Whether to generate reference image saliency
SALIENCY_GENERATE_CANDIDATES = True  # Whether to generate candidate image saliency
SALIENCY_GENERATE_TEXT_ATTRIBUTION = True  # Whether to generate text token attribution
