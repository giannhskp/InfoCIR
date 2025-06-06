import PIL.Image
from tqdm import tqdm
import clip
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE

from src import config

def calculate_clip_embeddings(dataset, clip_model_name='ViT-B/32'):
    """
    Calculate CLIP embeddings for images using the same approach as compose_image_retrieval_demo.py
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CLIP model
    model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    model = model.float().eval().requires_grad_(False)
    
    clip_embeddings = []
    
    print("Calculating CLIP embeddings...")
    for image_path in tqdm(dataset['image_path']):
        try:
            image = PIL.Image.open(image_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                # Normalize features as done in compose_image_retrieval_demo.py
                image_features = F.normalize(image_features.float())
                clip_embeddings.append(image_features.cpu().numpy())
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Add zero embedding for failed images
            clip_embeddings.append(np.zeros((1, 512)))  # ViT-B/32 has 512 dim
    
    clip_embeddings = np.concatenate(clip_embeddings, axis=0)
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

def generate_projection_data(clip_model_name='ViT-B/32'):
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
    
    # Calculate CLIP embeddings
    clip_embeddings = calculate_clip_embeddings(dataset_sample, clip_model_name)
    
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
    
    return augmented_dataset

if __name__ == '__main__':
    generate_projection_data() 