import os
import pandas as pd

from src import config, feature_engineering
from src.dataloaders import dataset_loader

class Dataset:
    data = None
    count = None
    
    @staticmethod
    def load():
        """Load the augmented dataset"""
        Dataset.data = pd.read_csv(config.AUGMENTED_DATASET_PATH, index_col='image_id')
        Dataset.count = Dataset.data['class_name'].value_counts()
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total number of images: {len(Dataset.data)}")
        print(f"Number of classes: {len(Dataset.count)}")

    @staticmethod
    def get():
        """Get the dataset"""
        return Dataset.data

    @staticmethod
    def class_count():
        """Get class counts"""
        return Dataset.count

    @staticmethod
    def processed_files_exist():
        """Check if processed/augmented dataset exists"""
        return os.path.isfile(config.AUGMENTED_DATASET_PATH)
    
    @staticmethod
    def source_files_exist():
        """Check if source dataset files exist"""
        return (os.path.isdir(config.IMAGES_DIR) and 
                os.path.isfile(config.CLASS_NAMES_PATH))

    @staticmethod
    def download():
        """Process the dataset (load and create embeddings)"""
        print("Processing dataset...")
        
        # Check if dataset path is configured
        if not config.DATASET_ROOT_PATH or config.DATASET_ROOT_PATH == '/path/to/your/dataset':
            raise ValueError(
                "Please configure DATASET_ROOT_PATH in config.py\n"
                "to point to your actual dataset directory"
            )
        
        # Check if source files exist
        if not Dataset.source_files_exist():
            missing_files = []
            if not os.path.isdir(config.IMAGES_DIR):
                missing_files.append(f"Images directory: {config.IMAGES_DIR}")
            if not os.path.isfile(config.CLASS_NAMES_PATH):
                missing_files.append(f"Class names file: {config.CLASS_NAMES_PATH}")
            
            raise FileNotFoundError(
                f"Source dataset files missing:\n" + 
                "\n".join(f"- {f}" for f in missing_files)
            )
        
        # Load dataset
        dataset_loader.load_dataset()
        
        # Generate CLIP embeddings and projections
        feature_engineering.generate_projection_data() 