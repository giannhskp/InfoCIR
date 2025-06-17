#!/usr/bin/env python3
"""
Simple Composed Image Retrieval Inference Script

This script performs inference using a pre-created database.
It takes a reference image and text prompt as input and returns top-k similar images.

Usage:
    python simple_cir_inference.py \
        --database-path /path/to/saved/database.pt \
        --reference-image /path/to/reference/image.jpg \
        --caption "description of desired modification" \
        --top-k 10

The database should be created beforehand using compose_image_retrieval_demo.py
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from math import ceil, sqrt

from compose_image_retrieval_demo import ComposedImageRetrievalSystem


class SimpleCIRInference:
    """
    Simple interface for Composed Image Retrieval inference.
    Loads a pre-created database and performs queries.
    """
    
    def __init__(self, database_path: str, clip_model_name: str = "ViT-B/32", 
                 eval_type: str = "searle", preprocess_type: str = "targetpad"):
        """
        Initialize the inference system.
        
        Args:
            database_path: Path to the saved database file
            clip_model_name: CLIP model name (should match the one used to create database)
            eval_type: Evaluation type (should match the one used to create database)
            preprocess_type: Preprocessing type (should match the one used to create database)
        """
        self.database_path = database_path
        self.clip_model_name = clip_model_name
        self.eval_type = eval_type
        self.preprocess_type = preprocess_type
        
        # Store dataset info for image path resolution
        self.dataset_info = None
        
        # Load database info
        self._load_database_info()
        
        # Initialize CIR system
        self.cir_system = ComposedImageRetrievalSystem(
            dataset_path="",  # Not needed when loading database
            dataset_type=self.dataset_type,
            clip_model_name=self.clip_model_name,
            eval_type=self.eval_type,
            preprocess_type=self.preprocess_type
        )
        
        # Load the pre-created database  
        self.cir_system.load_database(database_path)
        print(f"✅ Database loaded successfully from {database_path}")
        
    def _load_database_info(self):
        """Load database metadata to get configuration info."""
        if not Path(self.database_path).exists():
            raise FileNotFoundError(f"Database file not found: {self.database_path}")
            
        try:
            data = torch.load(self.database_path, map_location='cpu')
            dataset_info = data.get('dataset_info', {})
            self.dataset_info = dataset_info
            
            # Use database info if available, otherwise use provided defaults
            self.dataset_type = dataset_info.get('dataset_type', 'cirr')
            db_clip_model = dataset_info.get('clip_model_name', self.clip_model_name)
            db_eval_type = dataset_info.get('eval_type', self.eval_type)
            
            # Warn if there are mismatches
            if db_clip_model != self.clip_model_name:
                print(f"⚠️  Warning: Database was created with {db_clip_model}, but using {self.clip_model_name}")
            if db_eval_type != self.eval_type:
                print(f"⚠️  Warning: Database was created with {db_eval_type}, but using {self.eval_type}")
                
        except Exception as e:
            print(f"⚠️  Could not read database metadata: {e}")
            print("Using provided configuration parameters...")
            self.dataset_type = 'cirr'  # Default fallback
    
    def query(self, reference_image_path: str, caption: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Perform a composed image retrieval query.
        
        Args:
            reference_image_path: Path to the reference image
            caption: Text describing the desired modification
            top_k: Number of top results to return
            
        Returns:
            List of (image_name, similarity_score) tuples sorted by similarity
        """
        # Validate inputs
        if not Path(reference_image_path).exists():
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
            
        if not caption.strip():
            raise ValueError("Caption cannot be empty")
        
        # Perform query
        print(f"\n🔍 Performing query...")
        print(f"Reference image: {Path(reference_image_path).name}")
        print(f"Caption: '{caption}'")
        print(f"Requesting top-{top_k} results")
        
        try:
            results = self.cir_system.query(
                reference_image_path=reference_image_path,
                relative_caption=caption,
                top_k=top_k
            )
            
            print(f"\n✅ Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            raise
    
    def _resolve_image_path(self, image_name: str, dataset_base_path: Optional[str] = None) -> Optional[str]:
        """
        Try to resolve the full path of an image from its name.
        
        Args:
            image_name: Name of the image from the database
            dataset_base_path: Base path to the dataset (optional)
            
        Returns:
            Full path to the image if found, None otherwise
        """
        if not dataset_base_path:
            return None
            
        dataset_path = Path(dataset_base_path)
        
        # Common image directory patterns for different datasets
        possible_dirs = [
            dataset_path / "images",
            dataset_path / "val2017", 
            dataset_path / "unlabeled2017",
            dataset_path / "COCO2017_unlabeled" / "unlabeled2017",
            dataset_path,
        ]
        
        # Try different extensions
        extensions = ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG']
        
        for directory in possible_dirs:
            if not directory.exists():
                continue
                
            for ext in extensions:
                # Try direct path
                image_path = directory / f"{image_name}{ext}"
                if image_path.exists():
                    return str(image_path)
                
                # Try without extension (image_name might already have it)
                image_path = directory / image_name
                if image_path.exists():
                    return str(image_path)
                    
                # For ImageNet-R style (subdirectory/filename)
                if '/' in image_name:
                    image_path = directory / image_name
                    if image_path.exists():
                        return str(image_path)
        
        return None
    
    def display_results_grid(self, reference_image_path: str, caption: str, 
                           results: List[Tuple[str, float]], dataset_base_path: Optional[str] = None,
                           save_path: Optional[str] = None, show_plot: bool = True) -> None:
        """
        Display the reference image and top-k results in a grid.
        
        Args:
            reference_image_path: Path to the reference image
            caption: Query caption
            results: List of (image_name, similarity_score) tuples
            dataset_base_path: Base path to the dataset for resolving image paths
            save_path: Optional path to save the grid image
            show_plot: Whether to display the plot
        """
        print(f"\n🖼️  Creating results grid...")
        
        # Calculate grid dimensions (reference + results)
        total_images = len(results) + 1  # +1 for reference image
        grid_cols = min(5, total_images)  # Max 5 columns
        grid_rows = ceil(total_images / grid_cols)
        
        # Create figure
        fig_width = grid_cols * 3
        fig_height = grid_rows * 3.5  # Extra space for text
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))
        fig.suptitle(f'Composed Image Retrieval Results\nQuery: "{caption}"', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Handle single row case
        if grid_rows == 1:
            axes = [axes] if grid_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Display reference image
        try:
            ref_img = PIL.Image.open(reference_image_path).convert('RGB')
            axes[0].imshow(ref_img)
            axes[0].set_title("Reference Image", fontweight='bold', color='blue', fontsize=10)
            axes[0].axis('off')
            
            # Add blue border for reference
            for spine in axes[0].spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(3)
                spine.set_visible(True)
                
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Error loading\nreference image\n{e}", 
                        ha='center', va='center', fontsize=8)
            axes[0].set_title("Reference Image", fontweight='bold', color='red')
            axes[0].axis('off')
        
        # Display result images
        for i, (image_name, score) in enumerate(results):
            ax_idx = i + 1  # +1 to account for reference image
            if ax_idx >= len(axes):
                break
                
            try:
                # Try to resolve the image path
                image_path = None
                
                if dataset_base_path:
                    image_path = self._resolve_image_path(image_name, dataset_base_path)
                
                if image_path and Path(image_path).exists():
                    img = PIL.Image.open(image_path).convert('RGB')
                    axes[ax_idx].imshow(img)
                    axes[ax_idx].set_title(f"#{i+1}: {Path(image_name).stem}\nSim: {score:.3f}", 
                                         fontsize=9, pad=5)
                else:
                    # Show placeholder with image name
                    axes[ax_idx].text(0.5, 0.5, f"Image not found\n{image_name}\nSimilarity: {score:.3f}", 
                                    ha='center', va='center', fontsize=8, 
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    axes[ax_idx].set_title(f"#{i+1}: {Path(image_name).stem}", fontsize=9)
                    
                axes[ax_idx].axis('off')
                
                # Color-code by similarity score
                if score > 0.8:
                    border_color = 'green'
                elif score > 0.6:
                    border_color = 'orange'
                else:
                    border_color = 'red'
                    
                # Add colored border
                for spine in axes[ax_idx].spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2)
                    spine.set_visible(True)
                    
            except Exception as e:
                axes[ax_idx].text(0.5, 0.5, f"Error loading\n{image_name}\n{e}", 
                                ha='center', va='center', fontsize=8)
                axes[ax_idx].set_title(f"#{i+1}: Error", color='red', fontsize=9)
                axes[ax_idx].axis('off')
        
        # Hide unused subplots
        for i in range(total_images, len(axes)):
            axes[i].axis('off')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='blue', label='Reference Image'),
            mpatches.Patch(color='green', label='High Similarity (>0.8)'),
            mpatches.Patch(color='orange', label='Medium Similarity (>0.6)'),
            mpatches.Patch(color='red', label='Lower Similarity (≤0.6)')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
                  bbox_to_anchor=(0.5, 0.02), fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.15)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Grid saved to: {save_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        print(f"✅ Results grid created successfully!")


def main(args):
    
    # Validate arguments
    if args.top_k < 1:
        print("❌ Error: top-k must be at least 1")
        sys.exit(1)
    
    if args.display_grid and not args.dataset_path:
        print("⚠️  Warning: --dataset-path recommended for displaying result images")
    
    try:
        # Initialize inference system
        print("🚀 Initializing Composed Image Retrieval system...")
        inference = SimpleCIRInference(
            database_path=args.database_path,
            clip_model_name=args.clip_model_name,
            eval_type=args.eval_type,
            preprocess_type=args.preprocess_type
        )
        
        # Perform query
        results = inference.query(
            reference_image_path=args.reference_image,
            caption=args.caption,
            top_k=args.top_k
        )
        
        # Output results in text/json format
        if args.output_format == "json":
            import json
            output = {
                "query": {
                    "reference_image": args.reference_image,
                    "caption": args.caption,
                    "top_k": args.top_k
                },
                "results": [
                    {"rank": i+1, "image_name": name, "similarity_score": float(score)}
                    for i, (name, score) in enumerate(results)
                ]
            }
            print("\n" + json.dumps(output, indent=2))
        else:
            # Text format
            print(f"\n📋 Top {len(results)} Results:")
            print("=" * 60)
            for i, (image_name, score) in enumerate(results, 1):
                print(f"{i:2d}. {image_name:<40} (similarity: {score:.4f})")
        
        # Display grid if requested
        if args.display_grid:
            try:
                inference.display_results_grid(
                    reference_image_path=args.reference_image,
                    caption=args.caption,
                    results=results,
                    dataset_base_path=args.dataset_path,
                    save_path=args.save_grid,
                    show_plot=not args.no_show
                )
            except Exception as e:
                print(f"⚠️  Could not display grid: {e}")
                print("    Make sure matplotlib is installed: pip install matplotlib")
        
        
        print(f"\n🎉 Query completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    args = {
        "database_path": "/home/ikapetan/Frameworks/Projects-Master/MMA/dbs/imagenet-r-database",
        "reference_image": "/home/ikapetan/Frameworks/Projects-Master/MMA/SEARLE/src/example_scripts/green-apple-isolated-white.jpg",
        "caption": "as a cartoon character with other cartoon fruits around it",
        "top_k": 10,
        "clip_model_name": "ViT-B/32",
        "eval_type": "searle",
        "preprocess_type": "targetpad",
        "output_format": "text",
        "display_grid": True,
        "dataset_path": "/home/ikapetan/Frameworks/Projects-Master/MMA/data/imagenet-r",
        "save_grid": None,
        "no_show": False
    }
    args = argparse.Namespace(**args)
    main(args) 