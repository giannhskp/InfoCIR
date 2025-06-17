#!/usr/bin/env python3
"""
Example: Adding Visual Explanations to SEARLE Inference

This script demonstrates how to generate saliency heat-maps for SEARLE
composed image retrieval queries. It shows two types of explanations:

1. Reference heat-map: Which parts of the reference image influence 
   the pseudo tokens that SEARLE generates
2. Candidate heat-map: Which parts of a result image are most similar 
   to the composed query

Usage:
    python explanation_example.py
"""

import torch
import torch.nn.functional as F
import clip
import PIL.Image
from pathlib import Path

# Import the existing inference system and new explanation module
from simple_cir_inference import SimpleCIRInference
from explanations import SearleGradCAM
from encode_with_pseudo_tokens import encode_with_pseudo_tokens

def explain_searle_query(database_path: str, reference_image: str, caption: str, 
                        dataset_path: str = None, result_rank: int = 1):
    """
    Perform a SEARLE query and generate explanatory heat-maps.
    
    Args:
        database_path: Path to pre-created database (.pt file)
        reference_image: Path to reference image
        caption: Relative caption describing desired changes
        dataset_path: Path to dataset (for resolving result image paths)
        result_rank: Which result to explain (1-based ranking)
    """
    print("🚀 SEARLE Query with Visual Explanations")
    print("=" * 50)
    
    # Step 1: Initialize inference system and perform query
    print("1️⃣ Performing composed image retrieval...")
    inference = SimpleCIRInference(
        database_path=database_path,
        clip_model_name="ViT-B/32",
        eval_type="searle"
    )
    
    results = inference.query(
        reference_image_path=reference_image,
        caption=caption,
        top_k=5
    )
    
    print(f"   Found {len(results)} results")
    for i, (name, score) in enumerate(results[:3], 1):
        print(f"   {i}. {name} (similarity: {score:.4f})")
    
    # Step 2: Set up explanation components
    print("\n2️⃣ Setting up explanation components...")
    cir = inference.cir_system
    device = next(cir.clip_model.parameters()).device
    
    # Load reference image
    ref_img = PIL.Image.open(reference_image).convert("RGB")
    ref_tensor = cir.preprocess(ref_img).unsqueeze(0).to(device)
    
    # Compute SEARLE query features (exactly as the system does internally)
    with torch.no_grad():
        # Reference image → image features
        ref_features = cir.clip_model.encode_image(ref_tensor)
        
        # Image features → pseudo tokens (via Φ network)
        pseudo_tokens = cir.phi(ref_features)
        
        # Caption + pseudo tokens → composed query features
        input_caption = f"a photo of $ that {caption}"
        tokenized_caption = clip.tokenize([input_caption], context_length=77).to(device)
        query_features = encode_with_pseudo_tokens(
            cir.clip_model, tokenized_caption, pseudo_tokens
        )
        query_features = F.normalize(query_features, dim=-1)
    
    # Initialize explainer
    explainer = SearleGradCAM(cir.clip_model, cir.preprocess, phi_network=cir.phi, device=device)
    
    # Step 3: Generate reference image heat-map
    print("\n3️⃣ Generating reference image heat-map...")
    print(f"   This shows which parts of '{Path(reference_image).name}' influence the pseudo tokens")
    
    ref_heatmap = explainer.reference_heatmap(ref_img)
    
    # Save reference heat-map
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    ref_save_path = "reference_heatmap.png"
    plt.imsave(ref_save_path, cm.jet(ref_heatmap)[:, :, :3])
    print(f"   💾 Saved to: {ref_save_path}")
    
    # Step 4: Generate candidate result heat-map
    print(f"\n4️⃣ Generating heat-map for result #{result_rank}...")
    
    if result_rank <= len(results):
        result_name, result_score = results[result_rank - 1]
        print(f"   Target: {result_name} (similarity: {result_score:.4f})")
        
        # Resolve candidate image path
        if dataset_path:
            cand_path = inference._resolve_image_path(result_name, dataset_path)
            if cand_path and Path(cand_path).exists():
                cand_img = PIL.Image.open(cand_path).convert("RGB")
                
                # Generate heat-map showing similarity to composed query
                cand_heatmap, token_scores = explainer.candidate_heatmap(
                    cand_img, query_features, caption
                )
                
                # Save candidate heat-map
                cand_save_path = f"result_{result_rank}_heatmap.png"
                plt.imsave(cand_save_path, cm.jet(cand_heatmap)[:, :, :3])
                print(f"   💾 Saved to: {cand_save_path}")
                
                # Show token importance
                if token_scores is not None:
                    print(f"   📊 Token importance analysis:")
                    tokens = clip.tokenize([input_caption])[0]
                    vocab = clip.simple_tokenizer.SimpleTokenizer()
                    token_words = [vocab.decode([t.item()]) for t in tokens if t != 0]
                    
                    token_importance = list(zip(token_words, token_scores))
                    token_importance.sort(key=lambda x: x[1], reverse=True)
                    
                    for word, score in token_importance[:5]:
                        if score > 0.1:  # Only show significant tokens
                            print(f"      '{word}': {score:.3f}")
            else:
                print(f"   ❌ Could not find candidate image: {result_name}")
        else:
            print(f"   ⚠️  Dataset path not provided - cannot resolve result image")
    else:
        print(f"   ❌ Invalid result rank: {result_rank}")
    
    print("\n✅ Explanation complete!")
    print("\nInterpretation:")
    print("• Reference heat-map: Red areas contribute most to the pseudo tokens")
    print("• Candidate heat-map: Red areas are most similar to the composed query")
    print("• Token scores: Higher values = more influential words in the caption")


def main():
    """Example usage with placeholder paths."""
    
    # Update these paths for your setup
    database_path = "/home/ikapetan/Frameworks/Projects-Master/MMA/dbs/imagenet-r-database"  # Pre-created database
    reference_image = "/home/ikapetan/Frameworks/Projects-Master/MMA/SEARLE/src/example_scripts/green-apple-isolated-white.jpg"  # Your reference image
    caption = "as a cartoon character alonside with other cartoon fruits"  # Modification description
    dataset_path = "/home/ikapetan/Frameworks/Projects-Master/MMA/data/imagenet-r"  # Dataset root (for resolving results)
    
    print("SEARLE Visual Explanation Example")
    print("=" * 40)
    print(f"Database: {database_path}")
    print(f"Reference: {reference_image}")
    print(f"Caption: '{caption}'")
    print(f"Dataset: {dataset_path}")
    
    # Check if files exist
    if not Path(database_path).exists():
        print(f"\n❌ Database not found: {database_path}")
        print("Create one first using compose_image_retrieval_demo.py")
        return
    
    if not Path(reference_image).exists():
        print(f"\n❌ Reference image not found: {reference_image}")
        print("Update the reference_image path in this script")
        return
    
    try:
        explain_searle_query(
            database_path=database_path,
            reference_image=reference_image,
            caption=caption,
            dataset_path=dataset_path,
            result_rank=1  # Explain the top result
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure all dependencies are installed and paths are correct")


if __name__ == "__main__":
    main() 