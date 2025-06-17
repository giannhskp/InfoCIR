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


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Simple Composed Image Retrieval Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  python simple_cir_inference.py \\
    --database-path cirr_database.pt \\
    --reference-image /path/to/image.jpg \\
    --caption "is wearing a red shirt"
    
  # With visual grid display
  python simple_cir_inference.py \\
    --database-path fashioniq_database.pt \\
    --reference-image /path/to/dress.jpg \\
    --caption "is shorter" \\
    --top-k 8 \\
    --display-grid \\
    --dataset-path /path/to/fashioniq
        """
    )
    
    # Required arguments
    parser.add_argument("--database-path", type=str, required=True,
                        help="Path to the saved database file (.pt)")
    parser.add_argument("--reference-image", type=str, required=True,
                        help="Path to the reference image")
    parser.add_argument("--caption", type=str, required=True,
                        help="Text describing the desired modification")
    
    # Optional arguments
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top results to return (default: 10)")
    parser.add_argument("--clip-model-name", type=str, default="ViT-B/32",
                        help="CLIP model name (default: ViT-B/32)")
    parser.add_argument("--eval-type", type=str, default="searle",
                        choices=['searle', 'searle-xl', 'phi', 'oti'],
                        help="Evaluation type (default: searle)")
    parser.add_argument("--preprocess-type", type=str, default="targetpad",
                        choices=['clip', 'targetpad'],
                        help="Preprocessing type (default: targetpad)")
    parser.add_argument("--output-format", type=str, default="text",
                        choices=['text', 'json'],
                        help="Output format (default: text)")
    
    # Visualization arguments
    parser.add_argument("--display-grid", action="store_true",
                        help="Display results in a visual grid")
    parser.add_argument("--dataset-path", type=str,
                        help="Path to dataset (needed for displaying result images)")
    parser.add_argument("--save-grid", type=str,
                        help="Path to save the results grid image (e.g., results.png)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't show the plot window (useful when only saving)")
    
    # Explanation arguments
    parser.add_argument("--explain-reference", action="store_true",
                        help="Generate heat-map showing which parts of reference image influence the query")
    parser.add_argument("--explain-result", type=int, metavar="RANK",
                        help="Generate heat-map for result at given rank (1-based)")
    parser.add_argument("--save-heatmaps", type=str,
                        help="Directory to save heat-map images (default: current directory)")
    
    args = parser.parse_args()
    
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
        
        # Generate explanations if requested
        if args.explain_reference or args.explain_result:
            try:
                import clip
                import torch.nn.functional as F
                from explanations import SearleGradCAM
                from encode_with_pseudo_tokens import encode_with_pseudo_tokens
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm
                from pathlib import Path
                
                # Get the CIR system components
                cir = inference.cir_system
                device = next(cir.clip_model.parameters()).device
                
                # Load and preprocess reference image
                ref_img = PIL.Image.open(args.reference_image).convert("RGB")
                ref_tensor = cir.preprocess(ref_img).unsqueeze(0).to(device)
                
                # Compute the composed query features (same as in CIR system)
                with torch.no_grad():
                    ref_features = cir.clip_model.encode_image(ref_tensor)
                    pseudo_tokens = cir.phi(ref_features)
                    
                    # Create text with pseudo token placeholder
                    input_caption = f"a photo of $ that {args.caption}"
                    tokenized_caption = clip.tokenize([input_caption], context_length=77).to(device)
                    
                    # Encode text with pseudo tokens to get query features
                    query_features = encode_with_pseudo_tokens(
                        cir.clip_model, tokenized_caption, pseudo_tokens
                    )
                    query_features = F.normalize(query_features, dim=-1)
                
                # Initialize explainer
                explainer = SearleGradCAM(cir.clip_model, cir.preprocess, phi_network=cir.phi, device=device)
                
                # Set up save directory
                save_dir = Path(args.save_heatmaps) if args.save_heatmaps else Path(".")
                save_dir.mkdir(exist_ok=True)
                
                # Generate reference heat-map
                if args.explain_reference:
                    print(f"\n🔍 Generating reference image heat-map...")
                    ref_heatmap = explainer.reference_heatmap(ref_img)
                    
                    # Save heat-map
                    ref_save_path = save_dir / "reference_heatmap.png"
                    plt.imsave(ref_save_path, cm.jet(ref_heatmap)[:, :, :3])
                    print(f"✅ Reference heat-map saved to: {ref_save_path}")
                
                # Generate candidate result heat-map
                if args.explain_result:
                    if 1 <= args.explain_result <= len(results):
                        rank = args.explain_result
                        result_name, result_score = results[rank - 1]
                        
                        print(f"\n🔍 Generating heat-map for result #{rank}: {result_name}")
                        
                        # Resolve candidate image path
                        cand_path = inference._resolve_image_path(result_name, args.dataset_path)
                        if cand_path and Path(cand_path).exists():
                            cand_img = PIL.Image.open(cand_path).convert("RGB")
                            
                            # Generate heat-map and token scores
                            cand_heatmap, token_scores = explainer.candidate_heatmap(
                                cand_img, query_features, args.caption
                            )
                            
                            # Save candidate heat-map
                            cand_save_path = save_dir / f"result_{rank}_heatmap.png"
                            plt.imsave(cand_save_path, cm.jet(cand_heatmap)[:, :, :3])
                            print(f"✅ Result #{rank} heat-map saved to: {cand_save_path}")
                            print(f"   Image: {result_name} (similarity: {result_score:.4f})")
                            
                            # Show token importance if available
                            if token_scores is not None:
                                tokens = clip.tokenize([input_caption])[0]
                                vocab = clip.simple_tokenizer.SimpleTokenizer()
                                token_words = [vocab.decode([t.item()]) for t in tokens if t != 0]
                                
                                print(f"   Token importance (top 5):")
                                token_importance = list(zip(token_words, token_scores))
                                token_importance.sort(key=lambda x: x[1], reverse=True)
                                for word, score in token_importance[:5]:
                                    if score > 0.1:  # Only show significant tokens
                                        print(f"     '{word}': {score:.3f}")
                        else:
                            print(f"❌ Could not find candidate image: {result_name}")
                            if not args.dataset_path:
                                print("   Hint: Provide --dataset-path to resolve image paths")
                    else:
                        print(f"❌ Invalid result rank: {args.explain_result} (available: 1-{len(results)})")
                
                print(f"\n🎉 Explanation generation completed!")
                
            except ImportError as e:
                print(f"❌ Missing dependencies for explanations: {e}")
                print("   Make sure matplotlib is installed: pip install matplotlib")
            except Exception as e:
                print(f"❌ Error generating explanations: {e}")
        
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
    main() 