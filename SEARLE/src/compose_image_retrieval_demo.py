import json
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import os

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm
import PIL.Image
import pandas as pd

from data_utils import PROJECT_ROOT, targetpad_transform, collate_fn
from datasets import CIRRDataset, CIRCODataset, FashionIQDataset
from encode_with_pseudo_tokens import encode_with_pseudo_tokens
from phi import Phi
from utils import extract_image_features, device, extract_pseudo_tokens_with_phi


class ComposedImageRetrievalSystem:
    """
    A system for performing Composed Image Retrieval using SEARLE techniques.
    Supports creating databases from datasets and querying with reference image + text.
    """
    
    def __init__(self, dataset_path: str, dataset_type: str, clip_model_name: str, 
                 eval_type: str = 'searle', preprocess_type: str = 'targetpad',
                 exp_name: Optional[str] = None, phi_checkpoint_name: Optional[str] = None):
        """
        Initialize the CIR system.
        
        Args:
            dataset_path: Path to the dataset
            dataset_type: Type of dataset ('cirr', 'circo', 'fashioniq', 'imagenet', 'imagenet-r')
            clip_model_name: CLIP model to use
            eval_type: Evaluation type ('searle', 'searle-xl', 'phi', 'oti')
            preprocess_type: Preprocessing type ('clip', 'targetpad')
            exp_name: Experiment name (required for phi/oti)
            phi_checkpoint_name: Phi checkpoint name (required for phi)
        """
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type.lower()
        self.clip_model_name = clip_model_name
        self.eval_type = eval_type
        self.preprocess_type = preprocess_type
        self.exp_name = exp_name
        self.phi_checkpoint_name = phi_checkpoint_name
        
        # Initialize models and preprocessing
        self._setup_models_and_preprocessing()
        
        # Database storage
        self.database_features = None
        self.database_names = None
        self.database_created = False
        
    def _setup_models_and_preprocessing(self):
        """Setup CLIP model, phi model (if needed), and preprocessing pipeline."""
        print(f"Setting up models for {self.eval_type} evaluation...")
        
        # Load CLIP model
        self.clip_model, clip_preprocess = clip.load(self.clip_model_name, device=device, jit=False)
        self.clip_model = self.clip_model.float().eval().requires_grad_(False)
        
        # Setup preprocessing
        if self.preprocess_type == 'targetpad':
            print('Using target pad preprocess pipeline')
            self.preprocess = targetpad_transform(1.25, self.clip_model.visual.input_resolution)
        elif self.preprocess_type == 'clip':
            print('Using CLIP preprocess pipeline')
            self.preprocess = clip_preprocess
        else:
            raise ValueError("Preprocess type not supported")
            
        # Setup phi model if needed
        self.phi = None
        if self.eval_type in ['phi', 'searle', 'searle-xl']:
            if self.eval_type == 'phi':
                if not self.exp_name or not self.phi_checkpoint_name:
                    raise ValueError("exp_name and phi_checkpoint_name required for phi evaluation")
                    
                phi_path = PROJECT_ROOT / 'data' / "phi_models" / self.exp_name
                if not phi_path.exists():
                    raise ValueError(f"Experiment {self.exp_name} not found")
                    
                hyperparameters = json.load(open(phi_path / "hyperparameters.json"))
                
                self.phi = Phi(
                    input_dim=self.clip_model.visual.output_dim, 
                    hidden_dim=self.clip_model.visual.output_dim * 4,
                    output_dim=self.clip_model.token_embedding.embedding_dim, 
                    dropout=hyperparameters['phi_dropout']
                ).to(device)
                
                self.phi.load_state_dict(
                    torch.load(phi_path / 'checkpoints' / self.phi_checkpoint_name, map_location=device)[
                        self.phi.__class__.__name__])
                self.phi = self.phi.eval()
                
            else:  # searle or searle-xl
                print(f"Loading pre-trained {self.eval_type} model...")
                if self.eval_type == 'searle':
                    backbone = 'ViT-B/32'
                else:  # searle-xl
                    backbone = 'ViT-L/14'
                    
                self.phi, _ = torch.hub.load(
                    repo_or_dir='miccunifi/SEARLE', 
                    model='searle', 
                    source='github',
                    backbone=backbone
                )
                self.phi = self.phi.to(device).eval()
                
    def create_database(self, split: str = 'val'):
        """
        Create a database of image features from the specified dataset split.
        
        Args:
            split: Dataset split to use ('train', 'val', 'test')
        """
        print(f"Creating database from {self.dataset_type} {split} split...")
        
        # Create dataset
        if self.dataset_type == 'cirr':
            dataset = CIRRDataset(self.dataset_path, split, 'classic', self.preprocess)
        elif self.dataset_type == 'circo':
            dataset = CIRCODataset(self.dataset_path, split, 'classic', self.preprocess)
        elif self.dataset_type == 'fashioniq':
            dataset = FashionIQDataset(
                self.dataset_path, split, ['dress', 'toptee', 'shirt'], 'classic', self.preprocess
            )
        elif self.dataset_type == 'imagenet':
            df = pd.read_csv(self.dataset_path)
            from torch.utils.data import Dataset as TorchDataset
            class SimpleImageDataset(TorchDataset):
                def __init__(self, df, preprocess):
                    self.df = df
                    self.preprocess = preprocess
                def __len__(self):
                    return len(self.df)
                def __getitem__(self, idx):
                    row = self.df.iloc[idx]
                    img = PIL.Image.open(row['image_path']).convert('RGB')
                    img = self.preprocess(img)
                    return {'image': img, 'image_name': str(row.get('image_id', row.get('image_name', idx)))}
            dataset = SimpleImageDataset(df, self.preprocess)
        elif self.dataset_type == 'imagenet-r':
            root = Path(self.dataset_path)
            csv_files = list(root.glob("*.csv"))
            if not csv_files:
                raise ValueError("No mapping CSV found in imagenet-r directory")
            mapping_df = pd.read_csv(csv_files[0], header=None, names=['dir_name','class_name'])
            image_paths, image_names = [], []
            for dir_name in mapping_df['dir_name']:
                class_dir = root / dir_name
                if not class_dir.exists():
                    continue
                for file in class_dir.iterdir():
                    if file.is_file():
                        image_paths.append(str(file))
                        image_names.append(f"{dir_name}/{file.name}")
            from torch.utils.data import Dataset as TorchDataset
            class ImageNetRDataset(TorchDataset):
                def __init__(self, paths, names, preprocess):
                    self.paths = paths
                    self.names = names
                    self.preprocess = preprocess
                def __len__(self):
                    return len(self.paths)
                def __getitem__(self, idx):
                    img = PIL.Image.open(self.paths[idx]).convert('RGB')
                    img = self.preprocess(img)
                    return {'image': img, 'image_name': self.names[idx]}
            dataset = ImageNetRDataset(image_paths, image_names, self.preprocess)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
        # Extract image features
        self.database_features, self.database_names = extract_image_features(dataset, self.clip_model)
        self.database_features = F.normalize(self.database_features.float()).to(device)
        
        self.database_created = True
        print(f"Database created with {len(self.database_names)} images")
        
    def query(self, reference_image_path: str, relative_caption: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Perform a composed image retrieval query.
        
        Args:
            reference_image_path: Path to the reference image
            relative_caption: Text describing the desired modification
            top_k: Number of top results to return
            
        Returns:
            List of (image_name, similarity_score) tuples
        """
        if not self.database_created:
            raise ValueError("Database not created. Call create_database() first.")
            
        print(f"Querying with: '{relative_caption}'")
        
        # Load and preprocess reference image
        reference_image = PIL.Image.open(reference_image_path)
        reference_image = self.preprocess(reference_image).unsqueeze(0).to(device)
        
        # Extract reference image features
        with torch.no_grad():
            reference_features = self.clip_model.encode_image(reference_image)
            
        if self.eval_type == 'oti':
            # For OTI, would need to load pre-computed pseudo tokens
            raise NotImplementedError("OTI evaluation not implemented in this demo")
            
        elif self.eval_type in ['phi', 'searle', 'searle-xl']:
            # Use phi network to generate pseudo tokens
            with torch.no_grad():
                pseudo_tokens = self.phi(reference_features)
                
            # Create text with pseudo token placeholder
            input_caption = f"a photo of $ that {relative_caption}"
            tokenized_caption = clip.tokenize([input_caption], context_length=77).to(device)
            
            # Encode text with pseudo tokens
            with torch.no_grad():
                query_features = encode_with_pseudo_tokens(
                    self.clip_model, tokenized_caption, pseudo_tokens
                )
                query_features = F.normalize(query_features)
                
        else:
            raise ValueError(f"Unsupported evaluation type: {self.eval_type}")
            
        # Compute similarities
        similarities = query_features @ self.database_features.T
        similarities = similarities.squeeze().cpu()
        
        # Get top-k results
        top_indices = torch.topk(similarities, k=min(top_k, len(similarities))).indices
        
        results = []
        for idx in top_indices:
            image_name = self.database_names[idx]
            score = similarities[idx].item()
            results.append((image_name, score))
            
        return results
        
    def save_database(self, save_path: str):
        """Save the created database to disk."""
        if not self.database_created:
            raise ValueError("No database to save")
            
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'features': self.database_features.cpu(),
            'names': self.database_names,
            'dataset_info': {
                'dataset_type': self.dataset_type,
                'clip_model_name': self.clip_model_name,
                'eval_type': self.eval_type
            }
        }, save_path)
        print(f"Database saved to {save_path}")
        
    def load_database(self, load_path: str):
        """Load a previously saved database."""
        load_path = Path(load_path)
        if not load_path.exists():
            raise ValueError(f"Database file not found: {load_path}")
            
        data = torch.load(load_path, map_location='cpu')
        self.database_features = data['features'].to(device)
        self.database_names = data['names']
        self.database_created = True
        
        print(f"Database loaded with {len(self.database_names)} images")


def main():
    parser = ArgumentParser(description="Composed Image Retrieval Demo using SEARLE techniques")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--dataset-type", type=str, required=True, choices=['cirr', 'circo', 'fashioniq', 'imagenet', 'imagenet-r'],
                        help="Type of dataset")
    parser.add_argument("--clip-model-name", type=str, default='ViT-B/32', 
                        help="CLIP model to use")
    parser.add_argument("--eval-type", type=str, choices=['searle', 'searle-xl', 'phi', 'oti'], 
                        default='searle', help="Evaluation type")
    parser.add_argument("--preprocess-type", type=str, choices=['clip', 'targetpad'], 
                        default='targetpad', help="Preprocessing type")
    parser.add_argument("--split", type=str, default='val', choices=['train', 'val', 'test'], 
                        help="Dataset split for database")
    
    # Optional arguments for phi evaluation
    parser.add_argument("--exp-name", type=str, help="Experiment name (required for phi)")
    parser.add_argument("--phi-checkpoint-name", type=str, help="Phi checkpoint name (required for phi)")
    
    # Database options
    parser.add_argument("--save-database", type=str, help="Path to save the database")
    parser.add_argument("--load-database", type=str, help="Path to load a pre-saved database")
    
    # Query options
    parser.add_argument("--reference-image", type=str, help="Path to reference image for query")
    parser.add_argument("--caption", type=str, help="Relative caption for query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top results to return")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.eval_type == 'phi':
        if not args.exp_name or not args.phi_checkpoint_name:
            raise ValueError("--exp-name and --phi-checkpoint-name required for phi evaluation")
    
    # Initialize the CIR system
    cir_system = ComposedImageRetrievalSystem(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        clip_model_name=args.clip_model_name,
        eval_type=args.eval_type,
        preprocess_type=args.preprocess_type,
        exp_name=args.exp_name,
        phi_checkpoint_name=args.phi_checkpoint_name
    )
    
    # Load or create database
    if args.load_database:
        cir_system.load_database(args.load_database)
    else:
        cir_system.create_database(args.split)
        
        if args.save_database:
            cir_system.save_database(args.save_database)
    
    # Perform query if specified
    if args.reference_image and args.caption:
        print(f"\nPerforming query...")
        print(f"Reference image: {args.reference_image}")
        print(f"Caption: '{args.caption}'")
        
        results = cir_system.query(
            reference_image_path=args.reference_image,
            relative_caption=args.caption,
            top_k=args.top_k
        )
        
        print(f"\nTop {len(results)} results:")
        for i, (image_name, score) in enumerate(results, 1):
            print(f"{i:2d}. {image_name} (similarity: {score:.4f})")
            
    else:
        print("\nDatabase created successfully!")
        print("To perform a query, provide --reference-image and --caption arguments")
        
    # Example queries for demonstration
    if not (args.reference_image and args.caption):
        print("\nExample usage for querying:")
        print(f"python {__file__} \\")
        print(f"  --dataset-path {args.dataset_path} \\")
        print(f"  --dataset-type {args.dataset_type} \\")
        print(f"  --eval-type {args.eval_type} \\")
        if args.load_database:
            print(f"  --load-database {args.load_database} \\")
        print(f"  --reference-image path/to/reference/image.jpg \\")
        print(f"  --caption 'is wearing a red shirt' \\")
        print(f"  --top-k 5")


if __name__ == '__main__':
    main() 