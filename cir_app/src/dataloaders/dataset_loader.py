import os
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src import config

def load_dataset():
    """
    Load dataset and create a CSV file with image paths and class information.
    
    Expects:
    - DATASET_ROOT_PATH pointing to your dataset directory
    - Dataset structure:
      - DATASET_ROOT_PATH/ with subdirectories for each class
      - DATASET_ROOT_PATH/class_names.csv with dir_name,class_name mapping
        (CSV can have headers or just two columns: dir_name, class_name)
    """
    print("Loading dataset...")
    
    # Create working directory if it doesn't exist
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset path exists
    if not os.path.exists(config.DATASET_ROOT_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {config.DATASET_ROOT_PATH}\n"
            f"Please set the DATASET_ROOT_PATH in config.py"
        )
    
    # Check if class_names.csv exists
    if not os.path.exists(config.CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"class_names.csv not found at {config.CLASS_NAMES_PATH}\n"
            f"Expected structure: {config.DATASET_ROOT_PATH}/class_names.csv"
        )
    
    # Check if images directory exists
    if not os.path.exists(config.IMAGES_DIR):
        raise FileNotFoundError(
            f"Images directory not found at {config.IMAGES_DIR}\n"
            f"Expected structure: {config.DATASET_ROOT_PATH}/"
        )
    
    # Load class names mapping - handle both with and without headers
    try:
        # First try reading with headers
        class_names_df = pd.read_csv(config.CLASS_NAMES_PATH)
        if 'dir_name' not in class_names_df.columns or 'class_name' not in class_names_df.columns:
            raise ValueError("Headers not found or incorrect")
    except (ValueError, KeyError):
        # If headers are missing or incorrect, read without headers and add them
        print("CSV file doesn't have proper headers. Reading as dir_name,class_name...")
        class_names_df = pd.read_csv(config.CLASS_NAMES_PATH, header=None, names=['dir_name', 'class_name'])
    
    print(f"Found {len(class_names_df)} classes")
    
    # Collect all image paths
    data = []
    image_id = 1
    
    for _, row in tqdm(class_names_df.iterrows(), desc="Processing classes", total=len(class_names_df)):
        dir_name = row['dir_name']
        class_name = row['class_name']
        class_dir = os.path.join(config.IMAGES_DIR, dir_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found, skipping...")
            continue
        
        # Get all image files in this class directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(class_dir) if f.lower().endswith(ext.lower())])
        
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            data.append({
                'image_id': image_id,
                'class_name': class_name,
                'dir_name': dir_name,
                'image_path': image_path
            })
            image_id += 1
    
    # Create DataFrame
    dataset_df = pd.DataFrame(data)
    dataset_df['class_id'] = pd.Categorical(dataset_df['class_name']).codes
    
    # Save to CSV
    dataset_df.to_csv(config.DATASET_PATH, index=False)
    print(f"Dataset saved to {config.DATASET_PATH}")
    print(f"Total images: {len(dataset_df)}")
    print(f"Total classes: {dataset_df['class_name'].nunique()}")
    
    return dataset_df

def resize_images(target_size=(224, 224)):
    """
    Resize all images in the dataset to a target size (optional preprocessing step).
    Warning: This will modify your original dataset images!
    """
    print(f"Resizing images to {target_size}...")
    print("WARNING: This will modify your original images!")
    
    # Load dataset
    if os.path.exists(config.DATASET_PATH):
        dataset_df = pd.read_csv(config.DATASET_PATH)
    else:
        raise FileNotFoundError("Dataset CSV not found. Run load_dataset() first.")
    
    for _, row in tqdm(dataset_df.iterrows(), desc="Resizing images", total=len(dataset_df)):
        image_path = row['image_path']
        if os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')  # Ensure RGB format
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    img.save(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

if __name__ == '__main__':
    load_dataset()
    # Optionally resize images (WARNING: modifies original files)
    # resize_images() 