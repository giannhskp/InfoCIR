import PIL.Image
from tqdm import tqdm
import clip
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from umap import UMAP

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
import random
from PIL import ImageFilter
import json
import os

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

# ORIGINAL UMAP IMPLEMENTATION
def calculate_umap(clip_embeddings, n_components=2, metric='cosine'):
    """
    ORIGINAL UMAP CALCULATION - Simple unsupervised UMAP
    This is the old implementation, kept for compatibility
    """
    print("=== ORIGINAL UMAP CALCULATION ===")
    print("Calculating UMAP projection...")
    umap_reducer = UMAP(
        n_components=n_components, 
        metric=metric, 
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    umap_embeddings = umap_reducer.fit_transform(clip_embeddings)
    return umap_embeddings[:, 0], umap_embeddings[:, 1], umap_reducer
# END ORIGINAL UMAP IMPLEMENTATION

# NEW ENHANCED UMAP IMPLEMENTATION

def embed_images_enhanced(paths, device, clip_model_name=None, clip_layer=-1, 
                         augment_embeddings=False, semantic_enhancement=False):
    """Enhanced embedding function with additional techniques to reduce style bias."""
    if clip_model_name is None:
        clip_model_name = config.CLIP_MODEL_NAME
        
    model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    model = model.float().eval().requires_grad_(False)

    embeddings = []
    batch = []
    BATCH_SIZE = 128 if augment_embeddings else 256

    for idx, img_path in enumerate(paths):
        try:
            img = PIL.Image.open(img_path).convert("RGB")
            
            if augment_embeddings:
                # Create multiple views of the same image to focus on content over style
                augmented_imgs = []
                
                # Original
                augmented_imgs.append(preprocess(img))
                
                # Grayscale to reduce color style bias
                gray_img = img.convert('L').convert('RGB')
                augmented_imgs.append(preprocess(gray_img))
                
                # Different crops to focus on content
                width, height = img.size
                if min(width, height) > 224:
                    # Center crop
                    left = (width - 224) // 2
                    top = (height - 224) // 2
                    center_crop = img.crop((left, top, left + 224, top + 224))
                    augmented_imgs.append(preprocess(center_crop))
                    
                    # Random crop for different perspective
                    max_left = width - 224
                    max_top = height - 224
                    rand_left = random.randint(0, max_left)
                    rand_top = random.randint(0, max_top)
                    random_crop = img.crop((rand_left, rand_top, rand_left + 224, rand_top + 224))
                    augmented_imgs.append(preprocess(random_crop))
                
                # Edge detection to focus on shape over texture
                edges = img.filter(ImageFilter.FIND_EDGES).convert('RGB')
                augmented_imgs.append(preprocess(edges))
                
                batch.extend(augmented_imgs)
            else:
                batch.append(preprocess(img))
                
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            continue

        if len(batch) >= BATCH_SIZE or idx == len(paths) - 1:
            with torch.no_grad():
                inp = torch.stack(batch).to(device)
                
                if clip_layer == -1:
                    # Use final embeddings
                    feats = model.encode_image(inp).float()
                else:
                    # Extract intermediate layer features (would need model modification)
                    feats = model.encode_image(inp).float()
                
                feats = F.normalize(feats.float(), dim=-1)
                
                if augment_embeddings and len(batch) > 1:
                    # Calculate actual views per image based on current batch
                    # We need to track how many augmented views we created for each image in this batch
                    total_features = feats.shape[0]
                    
                    # Count images processed in this batch iteration
                    images_in_batch = 0
                    current_start = max(0, idx + 1 - len(batch))  # Starting index for this batch
                    for i in range(current_start, min(idx + 1, len(paths))):
                        images_in_batch += 1
                    
                    if images_in_batch > 0:
                        views_per_image = total_features // images_in_batch
                        
                        # Only reshape if we have the expected number of features
                        if total_features == views_per_image * images_in_batch:
                            feats = feats.reshape(images_in_batch, views_per_image, feats.shape[1]).mean(dim=1)
                        else:
                            # Fallback: just use the features as-is without averaging
                            print(f"Warning: Unexpected feature count {total_features} for {images_in_batch} images, skipping view averaging")
                
                embeddings.append(feats.cpu().numpy())
            batch.clear()
            print(f"Enhanced embedding: Processed {idx + 1}/{len(paths)} images", end="\r")

    return np.concatenate(embeddings, axis=0)


def apply_style_debiasing(embeddings, labels, *, return_transform: bool = False):
    """Apply the style-debiasing procedure (scaling → PCA → semantic-dimension
    selection).

    Parameters
    ----------
    embeddings : np.ndarray
        Matrix of CLIP (or previous-stage) embeddings of shape (N, D).
    labels : list[str] | np.ndarray
        Class labels (length N) – used to pick the semantic dimensions.
    return_transform : bool, default = False
        When *True* the function returns a 4-tuple *(embeddings_sd, scaler,
        pca, semantic_dims)* so that the exact same mapping can be applied to
        new query vectors at inference time.  When *False* (default) behaves
        exactly like before and returns only the transformed embeddings.
    """
    # Method 1: Remove the most style-correlated principal components
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Apply PCA to find principal components
    pca = PCA(n_components=min(100, embeddings.shape[1]))
    pca_embeddings = pca.fit_transform(embeddings_scaled)
    
    # Method 2: Focus on features that are more consistent within classes
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_indices = np.array([label_to_idx[label] for label in labels])
    
    # Calculate within-class vs between-class variance for each dimension
    within_class_var = []
    between_class_var = []
    
    for dim in range(pca_embeddings.shape[1]):
        # Within-class variance
        within_var = 0
        for label_idx in range(len(unique_labels)):
            class_mask = label_indices == label_idx
            if np.sum(class_mask) > 1:
                class_features = pca_embeddings[class_mask, dim]
                within_var += np.var(class_features)
        within_class_var.append(within_var / len(unique_labels))
        
        # Between-class variance
        class_means = []
        for label_idx in range(len(unique_labels)):
            class_mask = label_indices == label_idx
            if np.sum(class_mask) > 0:
                class_means.append(np.mean(pca_embeddings[class_mask, dim]))
        between_class_var.append(np.var(class_means) if len(class_means) > 1 else 0)
    
    # Select dimensions with high between-class / within-class variance ratio
    variance_ratios = np.array(between_class_var) / (np.array(within_class_var) + 1e-8)
    
    # Keep top 50% of dimensions with highest semantic discriminability
    n_keep = int(0.5 * len(variance_ratios))
    semantic_dims = np.argsort(variance_ratios)[-n_keep:]
    
    embeddings_sd = pca_embeddings[:, semantic_dims]

    if return_transform:
        return embeddings_sd, scaler, pca, semantic_dims
    return embeddings_sd


def apply_alternative_projection(embeddings, method, *, return_transform: bool = False):
    """Apply alternative dimensionality reduction techniques."""
    if method == "none":
        return (embeddings, None, None) if return_transform else embeddings
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    if method == "pca_then_umap":
        # First apply PCA to reduce noise, then UMAP in a subsequent step
        pca = PCA(n_components=min(200, embeddings.shape[1]))
        proj = pca.fit_transform(embeddings_scaled)
        return (proj, scaler, pca) if return_transform else proj
    
    elif method == "ica":
        # Independent Component Analysis to separate style and content
        ica = FastICA(n_components=min(200, embeddings.shape[1]), random_state=42)
        proj = ica.fit_transform(embeddings_scaled)
        return (proj, scaler, ica) if return_transform else proj

    # Fallback (should not reach here)
    return (embeddings_scaled, scaler, None) if return_transform else embeddings_scaled


def apply_contrastive_debiasing(embeddings, labels, contrastive_weight=0.1, *, return_transform: bool = False):
    """
    Apply contrastive debiasing to reduce style bias.
    
    For each class, identifies features that are:
    1. Consistent within the class (semantic)
    2. Variable across different styles within the class (style-invariant)
    """
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Group by class
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # For each class, compute style-invariant features
    class_prototypes = []
    for label in unique_labels:
        class_mask = np.array([l == label for l in labels])
        class_embeddings = embeddings_scaled[class_mask]
        
        if len(class_embeddings) > 1:
            # Compute centroid (style-invariant semantic core)
            centroid = np.mean(class_embeddings, axis=0)
            class_prototypes.append(centroid)
        else:
            class_prototypes.append(class_embeddings[0])
    
    class_prototypes = np.array(class_prototypes)
    
    # Project embeddings toward their class prototypes
    debiased_embeddings = []
    for i, (embedding, label) in enumerate(zip(embeddings_scaled, labels)):
        class_idx = label_to_idx[label]
        prototype = class_prototypes[class_idx]
        
        # Move embedding toward class prototype (reduces within-class style variation)
        debiased = embedding + contrastive_weight * (prototype - embedding)
        debiased_embeddings.append(debiased)
    
    debiased_arr = np.array(debiased_embeddings)

    if return_transform:
        return debiased_arr, scaler, class_prototypes, contrastive_weight
    return debiased_arr


def check_for_nans(data, name):
    """Check for NaNs in data and raise error if found."""
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        total_count = data.size
        raise ValueError(f"Found {nan_count} NaN values in {name} (out of {total_count} total values). "
                        f"This indicates a problem in the data processing pipeline that must be fixed.")


def calculate_umap_enhanced(clip_embeddings, labels, umap_config):
    """
    NEW ENHANCED UMAP CALCULATION
    """
    print("=== NEW ENHANCED UMAP CALCULATION ===", flush=True)
    print("Applying enhanced UMAP with debiasing techniques...", flush=True)
    
    # ------------------------------------------------------------------
    # Fit the *same* transformation chain that will later be applied to
    # novel query vectors at runtime.  We collect the fitted pieces in
    # *projection_pipeline* so that the Dash app can load & reuse them.
    # ------------------------------------------------------------------

    projection_pipeline = {}

    if umap_config.get('style_debiasing', False):
        print("Applying style debiasing techniques...", flush=True)
        clip_embeddings, style_scaler, style_pca, semantic_dims = apply_style_debiasing(
            clip_embeddings, labels, return_transform=True)
        projection_pipeline.update({
            'style_scaler': style_scaler,
            'style_pca': style_pca,
            'style_dims': semantic_dims
        })
        print(f"\u2713 Style debiasing applied. New shape: {clip_embeddings.shape}", flush=True)
    
    # Apply contrastive debiasing if enabled
    if umap_config.get('contrastive_debiasing', False):
        contrastive_weight = umap_config.get('contrastive_weight', 0.1)
        print(f"Applying contrastive debiasing with weight={contrastive_weight}...", flush=True)
        clip_embeddings, ctr_scaler, prototypes, c_weight = apply_contrastive_debiasing(
            clip_embeddings, labels, contrastive_weight, return_transform=True)
        projection_pipeline.update({
            'contrastive_scaler': ctr_scaler,
            'contrastive_prototypes': prototypes,
            'contrastive_weight': c_weight
        })
        print(f"\u2713 Contrastive debiasing applied. New shape: {clip_embeddings.shape}", flush=True)
    
    # Apply alternative projection if specified
    alt_projection = umap_config.get('alternative_projection', 'none')
    if alt_projection != 'none':
        print(f"Applying alternative projection: {alt_projection}", flush=True)
        clip_embeddings, alt_scaler, alt_model = apply_alternative_projection(
            clip_embeddings, alt_projection, return_transform=True)
        projection_pipeline.update({
            'alt_scaler': alt_scaler,
            'alt_model': alt_model
        })
        print(f"\u2713 Alternative projection applied. New shape: {clip_embeddings.shape}", flush=True)
    
    # ------------------------------------------------------------------
    # Fit the *same* transformation chain that will later be applied to
    # novel query vectors at runtime.  We collect the fitted pieces in
    # *projection_pipeline* so that the Dash app can load & reuse them.
    # ------------------------------------------------------------------

    pca_components = umap_config.get('pca_components', 100)
    pca = PCA(n_components=min(pca_components, clip_embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(clip_embeddings)
    check_for_nans(embeddings_pca, "PCA embeddings")

    # -----------------------------------------------------------
    # Save the final PCA into the projection pipeline dictionary
    # -----------------------------------------------------------
    projection_pipeline['final_pca'] = pca

    # Create label to integer mapping for supervised UMAP
    unique_labels = sorted(list(set(labels)))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Enhanced UMAP configuration
    umap_params = {
        "n_components": 2,
        "n_neighbors": umap_config.get('n_neighbors', 15),
        "min_dist": umap_config.get('min_dist', 0.1),
        "spread": umap_config.get('spread', 1.0),
        "metric": "cosine",
        "target_metric": "categorical",
        "target_weight": umap_config.get('target_weight', 0.5),
        "n_epochs": umap_config.get('n_epochs', 500),
        "random_state": 42,
        "local_connectivity": umap_config.get('local_connectivity', 1.0),
        "verbose": True
    }
    
    print(f"UMAP parameters: {umap_params}", flush=True)
    
    # Verify no NaNs before UMAP
    check_for_nans(embeddings_pca, "embeddings before UMAP")
    
    # Enhanced parameter tuning for semantic clustering (from advanced script)
    if umap_config.get('enhanced_parameter_tuning', True):
        if umap_config.get('style_debiasing', False) or umap_config.get('semantic_enhancement', False):
            print("Applying enhanced UMAP configuration for semantic clustering...", flush=True)
            umap_params.update({
                "target_weight": max(umap_params["target_weight"], 0.8),  # Ensure high target weight
                "min_dist": min(umap_params["min_dist"], 0.01),  # Tighter clusters
                "spread": max(umap_params["spread"], 1.0),  # Better global structure
                "n_epochs": max(umap_params["n_epochs"], 500),  # More training
            })
            print(f"✓ Enhanced parameters: target_weight={umap_params['target_weight']}, min_dist={umap_params['min_dist']}, spread={umap_params['spread']}, n_epochs={umap_params['n_epochs']}", flush=True)
    
    # Force approximation algorithm if requested
    if umap_config.get('force_approximation_algorithm', False):
        umap_params["force_approximation_algorithm"] = True
        print("✓ Forcing approximation algorithm", flush=True)
    
    reducer = UMAP(**umap_params)
    print("[LOG] Starting UMAP fit_transform... (this may take a long time)", flush=True)
    projection = reducer.fit_transform(embeddings_pca, y=np.array([label_to_int[lab] for lab in labels]))
    print("[LOG] UMAP fit_transform completed.", flush=True)
    
    check_for_nans(projection, "UMAP projection")
    print(f"✓ Enhanced UMAP projection completed. Shape: {projection.shape}", flush=True)
    
    # Optional HDBSCAN post-processing
    cluster_labels = None
    if umap_config.get('use_hdbscan', False):
        try:
            import hdbscan
            min_cluster_size = umap_config.get('hdbscan_min_cluster_size', 10)
            print(f"Applying HDBSCAN clustering with min_cluster_size={min_cluster_size}", flush=True)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
            cluster_labels = clusterer.fit_predict(projection)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print(f"✓ Found {n_clusters} clusters with HDBSCAN", flush=True)
            
            # Save cluster labels to CSV if we have access to dataset
            try:
                import pandas as pd_local
                cluster_df = pd_local.DataFrame({
                    'hdbscan_cluster': cluster_labels
                })
                cluster_path = os.path.join('data', 'hdbscan_clusters.csv')
                cluster_df.to_csv(cluster_path, index=False)
                print(f"✓ HDBSCAN clusters saved to {cluster_path}", flush=True)
            except Exception as e:
                print(f"Warning: Could not save HDBSCAN clusters: {e}", flush=True)
                
        except ImportError:
            print("Warning: HDBSCAN not available. Skipping clustering.", flush=True)
        except Exception as e:
            print(f"Warning: HDBSCAN clustering failed: {e}", flush=True)
    
    # Calculate quality metrics if enabled
    metrics = None
    if umap_config.get('calculate_quality_metrics', True):
        metrics = calculate_quality_metrics(embeddings_pca, projection, labels)
        
        # Save metrics to file
        metrics_path = os.path.join('data', 'umap_quality_metrics.json')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Quality metrics saved to {metrics_path}", flush=True)
    
    # ------------------------------------------------------------------
    # Persist the UMAP reducer *and* the full pre-UMAP projection pipeline so
    # that the Dash callbacks can reproduce the exact mapping.  We store the
    # reducer right here as well (even though `generate_projection_data()`
    # will save it again) so that intermediate runs of this function alone
    # also leave a usable artefact on disk.
    # ------------------------------------------------------------------

    umap_model_path = config.WORK_DIR / 'umap_reducer.pkl'
    try:
        with open(str(umap_model_path), 'wb') as f:
            pickle.dump(reducer, f)
        print(f'UMAP reducer saved to {umap_model_path}', flush=True)
    except Exception as e:
        print(f"Warning: could not save UMAP reducer: {e}", flush=True)
    
    # Save complete projection pipeline (excluding UMAP)
    pipeline_path = config.WORK_DIR / 'projection_pipeline.pkl'
    try:
        with open(str(pipeline_path), 'wb') as f:
            pickle.dump(projection_pipeline, f)
        print(f'Projection pipeline saved to {pipeline_path}')
    except Exception as e:
        print(f"Warning: failed to save projection pipeline: {e}")
    
    return projection[:, 0], projection[:, 1], reducer

# END NEW ENHANCED UMAP IMPLEMENTATION

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
    
    # ==================================================================================
    # ENHANCED EMBEDDING CALCULATION
    # ==================================================================================
    if config.USE_NEW_UMAP and (config.NEW_UMAP_CONFIG.get('semantic_enhancement', False) or 
                                config.NEW_UMAP_CONFIG.get('augment_embeddings', False)):
        print("=== USING NEW ENHANCED EMBEDDING CALCULATION ===")
        # Use enhanced embedding function with debiasing techniques
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_paths = dataset_sample['image_path'].values.tolist()
        clip_embeddings = embed_images_enhanced(
            image_paths, 
            device,
            augment_embeddings=config.NEW_UMAP_CONFIG.get('augment_embeddings', False),
            semantic_enhancement=config.NEW_UMAP_CONFIG.get('semantic_enhancement', False)
        )
        print(f"Enhanced CLIP embeddings shape: {clip_embeddings.shape}")
    else:
        # OLD EMBEDDING CALCULATION (ORIGINAL)
        print("=== USING ORIGINAL EMBEDDING CALCULATION ===")
        clip_embeddings = calculate_clip_embeddings(dataset_sample)
    
    # ==================================================================================
    # ENHANCED UMAP CALCULATION (NEW vs OLD)
    # ==================================================================================
    if config.USE_NEW_UMAP:
        print("=== USING NEW ENHANCED UMAP ===")
        # NEW ENHANCED UMAP with debiasing and supervision
        labels = dataset_sample['class_name'].values.tolist()
        umap_x, umap_y, umap_reducer = calculate_umap_enhanced(
            clip_embeddings, 
            labels, 
            config.NEW_UMAP_CONFIG
        )
    else:
        print("=== USING ORIGINAL UMAP ===")
        # OLD ORIGINAL UMAP CALCULATION (COMMENTED BUT FUNCTIONAL)
        # umap_x, umap_y, umap_reducer = calculate_umap(clip_embeddings)
        umap_x, umap_y, umap_reducer = calculate_umap(clip_embeddings)
    
    # ==================================================================================
    # POST-HOC RESCALING - Expand scatterplot range while keeping clusters distinct
    # ==================================================================================
    print("Applying post-hoc rescaling to expand scatterplot range...")
    desired_max = 300                                     # target half-width
    current_max = np.abs(np.concatenate([umap_x, umap_y])).max()
    scale = desired_max / current_max                     # e.g. ≈ 2.0
    print(f"Scaling factor: {scale:.3f} (from ±{current_max:.1f} to ±{desired_max})")
    umap_x *= scale
    umap_y *= scale
    
    # Create augmented dataset
    augmented_dataset = dataset_sample.assign(
        umap_x=umap_x, 
        umap_y=umap_y
    )
    
    # Save augmented dataset
    augmented_dataset.to_csv(config.AUGMENTED_DATASET_PATH, index=False)
    print(f'Augmented dataset saved to {config.AUGMENTED_DATASET_PATH}')
    # Save UMAP reducer for transforming new query images at runtime
    umap_model_path = config.WORK_DIR / 'umap_reducer.pkl'
    with open(str(umap_model_path), 'wb') as f:
        pickle.dump(umap_reducer, f)
    print(f'UMAP reducer saved to {umap_model_path}')
    
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

# ==================================================================================
# QUALITY METRICS FUNCTIONS - From advanced script
# ==================================================================================

def continuity(high_X, low_X, n_neighbors: int = 10):
    """
    Continuity metric (Venna & Kaski).
    Measures how well the embedding preserves local structure.
    Higher values indicate better preservation of local neighborhoods.
    """
    n_samples = high_X.shape[0]
    knn_high = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(high_X)
    knn_low = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(low_X)

    high_idx = knn_high.kneighbors(return_distance=False)[:, 1:]
    low_idx = knn_low.kneighbors(return_distance=False)[:, 1:]

    score = 0.0
    for i in range(n_samples):
        score += len(set(high_idx[i]) & set(low_idx[i])) / n_neighbors
    return score / n_samples


def neighborhood_hit(labels, low_X, n_neighbors: int = 10):
    """
    Neighborhood hit metric - measures label consistency in neighborhoods.
    This IS a supervised metric that considers class labels.
    Higher values indicate better class separation.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(low_X)
    idx = nbrs.kneighbors(return_distance=False)[:, 1:]
    hit = 0.0
    for i, nbr in enumerate(idx):
        hit += (labels[nbr] == labels[i]).mean()
    return hit / len(labels)


def normalized_stress(high_X, low_X, sample: int = 2000):
    """
    Normalized stress metric - measures how well distances are preserved.
    Lower values indicate better distance preservation.
    """
    if high_X.shape[0] > sample:
        sel = np.random.RandomState(42).choice(high_X.shape[0], sample, replace=False)
        high_X = high_X[sel]
        low_X = low_X[sel]
    D_high = pairwise_distances(high_X)
    D_low = pairwise_distances(low_X)
    stress = np.sum((D_high - D_low) ** 2) / np.sum(D_high ** 2)
    return float(stress)


def shepard_goodness(high_X, low_X, sample_size: int = 2000):
    """
    Shepard goodness metric - Spearman correlation between high and low-dim distances.
    Higher values indicate better distance preservation.
    """
    if high_X.shape[0] > sample_size:
        idx = np.random.RandomState(42).choice(high_X.shape[0], sample_size, replace=False)
        high_X = high_X[idx]
        low_X = low_X[idx]
    dist_high = pairwise_distances(high_X).flatten()
    dist_low = pairwise_distances(low_X).flatten()
    corr, _ = spearmanr(dist_high, dist_low)
    return float(corr)


def class_separation_score(embedding, labels, n_neighbors=10):
    """
    Calculate how well classes are separated in the embedding space.
    Higher scores indicate better separation between different classes.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    
    # Calculate average intra-class vs inter-class distances
    intra_class_dists = []
    inter_class_dists = []
    
    for i, label in enumerate(labels):
        neighbor_labels = labels[indices[i][1:]]  # Exclude self
        neighbor_dists = distances[i][1:]
        
        same_class_mask = neighbor_labels == label
        if same_class_mask.any():
            intra_class_dists.extend(neighbor_dists[same_class_mask])
        if (~same_class_mask).any():
            inter_class_dists.extend(neighbor_dists[~same_class_mask])
    
    if len(intra_class_dists) == 0 or len(inter_class_dists) == 0:
        return 0.0
    
    # Separation score: ratio of inter-class to intra-class distances
    separation = np.mean(inter_class_dists) / np.mean(intra_class_dists)
    return separation


def calculate_quality_metrics(embeddings_pca, projection, labels):
    """Calculate comprehensive quality metrics for UMAP projection."""
    print("Calculating quality metrics...", flush=True)
    labels_arr = np.array(labels)
    
    # Calculate comprehensive metrics including supervised ones
    trust = trustworthiness(embeddings_pca, projection, n_neighbors=10)
    cont = continuity(embeddings_pca, projection, n_neighbors=10)
    nhit = neighborhood_hit(labels_arr, projection, n_neighbors=10)  # Supervised metric
    stress = normalized_stress(embeddings_pca, projection)
    shep = shepard_goodness(embeddings_pca, projection)
    sep_score = class_separation_score(projection, labels_arr)  # New supervised metric
    
    metrics = {
        "trustworthiness": float(trust),
        "continuity": float(cont),
        "neighborhood_hit": float(nhit),  # Higher = better class separation
        "class_separation_score": float(sep_score),  # Higher = better separation
        "stress": float(stress),
        "shepard_goodness": float(shep),
        "n_classes": len(set(labels)),
        "n_samples": len(labels_arr),
    }
    
    print(f"✓ Quality Metrics:", flush=True)
    print(f"  Trustworthiness: {trust:.4f}", flush=True)
    print(f"  Neighborhood Hit: {nhit:.4f} (higher = better class separation)", flush=True)
    print(f"  Class Separation Score: {sep_score:.4f} (higher = better separation)", flush=True)
    print(f"  Continuity: {cont:.4f}", flush=True)
    
    return metrics

# ==================================================================================
# END QUALITY METRICS FUNCTIONS
# ==================================================================================

if __name__ == '__main__':
    generate_projection_data() 