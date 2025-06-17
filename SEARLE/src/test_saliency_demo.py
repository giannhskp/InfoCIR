#!/usr/bin/env python3
"""
Test script for SEARLE CIR with Grad-ECLIP Saliency Maps

This script demonstrates the saliency functionality using hardcoded parameters.
"""

import sys
from pathlib import Path

# Import the saliency-enabled system
from simple_cir_inference_with_saliency import SaliencyEnabledCIRSystem

def main():
    """Test the saliency-enabled CIR system."""
    
    # Configuration (same as the original simple_cir_inference.py)
    config = {
        "database_path": "/home/ikapetan/Frameworks/Projects-Master/MMA/dbs/imagenet-r-database",
        "reference_image": "/home/ikapetan/Frameworks/Projects-Master/MMA/SEARLE/src/example_scripts/green-apple-isolated-white.jpg",
        "caption": "as a cartoon character with other cartoon fruits around it",
        "top_k": 10,
        "clip_model_name": "ViT-B/32",
        "eval_type": "searle",
        "preprocess_type": "targetpad",
        "dataset_path": "/home/ikapetan/Frameworks/Projects-Master/MMA/data/imagenet-r",
    }
    
    print("🚀 Testing SEARLE CIR with Grad-ECLIP Saliency Maps...")
    print(f"Reference image: {config['reference_image']}")
    print(f"Caption: '{config['caption']}'")
    print(f"Database: {config['database_path']}")
    
    try:
        # Initialize saliency-enabled inference system
        print("\n🔧 Initializing system...")
        inference = SaliencyEnabledCIRSystem(
            database_path=config["database_path"],
            clip_model_name=config["clip_model_name"],
            eval_type=config["eval_type"],
            preprocess_type=config["preprocess_type"]
        )
        
        # Store dataset path for candidate image resolution
        inference._dataset_path = config["dataset_path"]
        
        print("✅ System initialized successfully!")
        
        # Perform query with saliency generation
        print("\n🔍 Performing query with saliency generation...")
        query_results = inference.query_with_saliency(
            reference_image_path=config["reference_image"],
            relative_caption=config["caption"],
            top_k=config["top_k"],
            generate_reference_saliency=True,
            generate_candidate_saliency=True,
            max_candidate_saliency=5,
            generate_text_attribution=True
        )
        
        # Display results
        print(f"\n📋 Top {len(query_results['results'])} Results:")
        print("=" * 70)
        for result in query_results['results']:
            print(f"{result['rank']:2d}. {result['image_name']:<45} (similarity: {result['similarity_score']:.4f})")
        
        # Report saliency map generation
        print(f"\n🎨 Saliency Maps Generated:")
        if 'reference' in query_results['saliency_maps']:
            ref_shape = query_results['saliency_maps']['reference'].shape
            print(f"   ✅ Reference image saliency map: {ref_shape}")
        else:
            print(f"   ❌ Reference image saliency map: Failed")
            
        if 'candidates' in query_results['saliency_maps']:
            num_candidates = len(query_results['saliency_maps']['candidates'])
            print(f"   ✅ Candidate saliency maps: {num_candidates} generated")
            for image_name, data in query_results['saliency_maps']['candidates'].items():
                shape = data['saliency_map'].shape
                rank = data['rank']
                print(f"      - Rank {rank}: {Path(image_name).name} ({shape})")
        else:
            print(f"   ❌ Candidate saliency maps: None generated")
        
        # Report text attribution generation
        print(f"\n🔤 Text Attribution Analysis:")
        if 'text_attribution' in query_results:
            if 'reference' in query_results['text_attribution']:
                ref_attr = query_results['text_attribution']['reference']
                print(f"   ✅ Reference text attribution: {len(ref_attr['tokens'])} tokens analyzed")
                # Show top attributed tokens
                top_tokens = sorted(
                    zip(ref_attr['tokens'], ref_attr['attributions']), 
                    key=lambda x: x[1], reverse=True
                )[:3]
                print(f"      Top tokens: {[f'{token}({attr:.3f})' for token, attr in top_tokens]}")
            else:
                print(f"   ❌ Reference text attribution: Failed")
                
            if 'candidates' in query_results['text_attribution']:
                num_candidates_attr = len(query_results['text_attribution']['candidates'])
                print(f"   ✅ Candidate text attributions: {num_candidates_attr} generated")
                for image_name, text_attr in query_results['text_attribution']['candidates'].items():
                    # Show top attributed token for each candidate
                    top_token = max(
                        zip(text_attr['tokens'], text_attr['attributions']), 
                        key=lambda x: x[1]
                    )
                    print(f"      - {Path(image_name).name}: Top token '{top_token[0]}' ({top_token[1]:.3f})")
            else:
                print(f"   ❌ Candidate text attributions: None generated")
        else:
            print(f"   ❌ Text attribution analysis: Not performed")
        
        # Save saliency visualizations
        save_dir = "/home/ikapetan/Frameworks/Projects-Master/MMA/SEARLE/saliency_output"
        print(f"\n💾 Saving visualizations to: {save_dir}")
        
        inference.save_saliency_visualizations(query_results, save_dir)
        
        print(f"\n🎉 Test completed successfully!")
        print(f"📁 Check the output directory: {save_dir}")
        print(f"📊 Generated files include:")
        print(f"   - Saliency heatmaps (.png) and raw data (.npy)")
        print(f"   - Text attribution visualizations (.png)")
        print(f"   - Reference and candidate analysis results")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 