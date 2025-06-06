import PIL.Image
from tqdm import tqdm
import clip
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
import pickle
import os

from src import config

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = PIL.Image.open(image_path).convert('RGB')
            image = self.preprocess(image)
            return image
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return torch.zeros(3, 224, 224)  # fallback zero image if corrupted

def calculate_clip_embeddings(dataset, clip_model_name=None, batch_size=32):
    """
    Calculate CLIP embeddings for images using the same approach as compose_image_retrieval_demo.py
    """
    # Use the configured CLIP model if none provided
    if clip_model_name is None:
        clip_model_name = config.CLIP_MODEL_NAME
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CLIP model
    model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    model = model.float().eval().requires_grad_(False)

    # Prepare dataset and loader
    image_paths = dataset['image_path'].values
    img_dataset = ImagePathDataset(image_paths, preprocess)
    data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_embeddings = []

    print("Calculating CLIP embeddings in batches...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            features = model.encode_image(batch)
            features = F.normalize(features.float(), dim=-1)
            all_embeddings.append(features.cpu().numpy())
    
    clip_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"CLIP embeddings shape: {clip_embeddings.shape}")
    return clip_embeddings

def calculate_umap(clip_embeddings, n_components=2, metric='cosine'):
    """Calculate UMAP projection"""
    print("Calculating UMAP projection...")
    umap_reducer = UMAP(
        n_components=n_components, 
        metric=metric, 
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    umap_embeddings = umap_reducer.fit_transform(clip_embeddings)
    return umap_embeddings[:, 0], umap_embeddings[:, 1]

def calculate_tsne(clip_embeddings, n_components=2, metric='cosine'):
    """Calculate t-SNE projection"""
    print("Calculating t-SNE projection...")
    tsne_reducer = TSNE(
        n_components=n_components,
        metric=metric,
        random_state=42,
        perplexity=30,
        n_iter=1000
    )
    tsne_embeddings = tsne_reducer.fit_transform(clip_embeddings)
    return tsne_embeddings[:, 0], tsne_embeddings[:, 1]

def generate_projection_data():
    """
    Generate CLIP embeddings and projections for the dataset
    """
    # Load dataset
    dataset = pd.read_csv(config.DATASET_PATH)
    
    # Sample dataset if specified
    if config.DATASET_SAMPLE_SIZE and config.DATASET_SAMPLE_SIZE < len(dataset):
        print(f"Sampling {config.DATASET_SAMPLE_SIZE} images from {len(dataset)} total images")
        dataset_sample = dataset.sample(n=config.DATASET_SAMPLE_SIZE, random_state=42)
    else:
        dataset_sample = dataset
    
    print(f"Processing {len(dataset_sample)} images")
    
    # Calculate CLIP embeddings using configured model
    clip_embeddings = calculate_clip_embeddings(dataset_sample)
    
    # Calculate projections
    umap_x, umap_y = calculate_umap(clip_embeddings)
    tsne_x, tsne_y = calculate_tsne(clip_embeddings)
    
    # Create augmented dataset
    augmented_dataset = dataset_sample.assign(
        umap_x=umap_x, 
        umap_y=umap_y, 
        tsne_x=tsne_x, 
        tsne_y=tsne_y
    )
    
    # Save augmented dataset
    augmented_dataset.to_csv(config.AUGMENTED_DATASET_PATH, index=False)
    print(f'Augmented dataset saved to {config.AUGMENTED_DATASET_PATH}')
    
    # Save CLIP embeddings and names for CIR retrieval
    features_dir = config.CIR_FEATURES_PATH
    # ensure features directory exists
    if isinstance(features_dir, str):
        os.makedirs(features_dir, exist_ok=True)
    else:
        features_dir.mkdir(parents=True, exist_ok=True)
    # convert numpy embeddings to tensor and save
    features_tensor = torch.from_numpy(clip_embeddings)
    torch.save(features_tensor, os.path.join(str(features_dir), 'index_features.pt'))
    # save the corresponding image IDs as names
    names_list = dataset_sample['image_id'].astype(str).tolist()
    with open(os.path.join(str(features_dir), 'index_names.pkl'), 'wb') as f:
        pickle.dump(names_list, f)
    print(f"Saved CLIP embeddings and names to {features_dir}")
    
    return augmented_dataset

if __name__ == '__main__':
    generate_projection_data() 