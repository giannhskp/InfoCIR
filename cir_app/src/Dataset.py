import sys
import os
import threading
import pandas as pd

from pathlib import Path

from src import config, feature_engineering
from src.dataloaders import dataset_loader

from src.shared import cir_systems

sys.path.append(os.path.join(os.path.dirname(__file__), 'callbacks', 'SEARLE'))
from compose_image_retrieval_demo import ComposedImageRetrievalSystem

sys.path.append(os.path.join(os.path.dirname(__file__), 'callbacks', 'freedom'))
from composed_image_retrieval_freedom import FreedomRetrievalSystem

class Dataset:
    data = None
    count = None
    
    @staticmethod
    def load():
        """Load the augmented dataset"""
        Dataset.data = pd.read_csv(config.AUGMENTED_DATASET_PATH, index_col='image_id')
        Dataset.count = Dataset.data['class_name'].value_counts()

        # Initialize the CIR system
        with cir_systems.lock:
            if cir_systems.cir_system_searle is None:
                cir_systems.cir_system_searle = ComposedImageRetrievalSystem(
                    dataset_path=config.CIR_DATASET_PATH,
                    dataset_type=config.CIR_DATASET_TYPE,
                    clip_model_name=config.CIR_CLIP_MODEL_NAME,
                    eval_type=config.CIR_EVAL_TYPE,
                    preprocess_type=config.CIR_PREPROCESS_TYPE,
                    exp_name=config.CIR_EXP_NAME,
                    phi_checkpoint_name=config.CIR_PHI_CHECKPOINT_NAME,
                    features_path=config.CIR_FEATURES_PATH,
                    load_features=config.CIR_LOAD_FEATURES,
                )
                cir_systems.cir_system_searle.create_database(split=config.CIR_SPLIT)

        with cir_systems.lock:
            if cir_systems.cir_system_freedom is None:
                cir_systems.cir_system_freedom = FreedomRetrievalSystem(
                    load_features=config.CIR_LOAD_FEATURES, features_path=config.CIR_FREEDOM_FEATURES_PATH
                )
                cir_systems.cir_system_freedom.create_database(config.WORK_DIR)
        
        # Initialize the saliency manager
        with cir_systems.lock:
            if cir_systems.saliency_manager is None:
                from src.saliency import SaliencyManager
                cir_systems.saliency_manager = SaliencyManager()
        
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
        features_path = Path(config.CIR_FEATURES_PATH)
        umap_reducer_path = Path(config.WORK_DIR)
        
        freedom_save_dir = os.path.join(config.WORK_DIR, "clip_features")
        freedom_corpus_file = os.path.join(freedom_save_dir, "corpus", "open_image_v7_class_names.pkl")
        freedom_features_file = os.path.join(freedom_save_dir, "imagenet_r", "full_imagenet_r_features.pkl")
        freedom_names_file = os.path.join(freedom_save_dir, "imagenet_r", "full_imagenet_r_names.pkl")

        return (os.path.isfile(config.AUGMENTED_DATASET_PATH) and os.path.isfile(features_path / "index_features.pt")
                and os.path.isfile(features_path / "index_names.pkl")
                and os.path.isfile(umap_reducer_path / "umap_reducer.pkl")
                and os.path.isfile(freedom_corpus_file)
                and os.path.isfile(freedom_features_file)
                and os.path.isfile(freedom_names_file))
    
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