#!/usr/bin/env python3
"""
SEARLE Composed Image Retrieval with Grad-ECLIP Saliency Maps

This script extends the simple CIR inference with visual explanation capabilities.
It generates saliency maps for both reference images (showing where φ "looks" when building 
pseudo-words) and candidate images (showing regions driving similarity with text).

Usage:
    python simple_cir_inference_with_saliency.py \
        --database-path /path/to/saved/database.pt \
        --reference-image /path/to/reference/image.jpg \
        --caption "description of desired modification" \
        --top-k 10 \
        --generate-saliency

Dependencies:
    - pytorch-grad-cam: pip install pytorch-grad-cam
    - Grad-ECLIP repository (cloned locally)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn.functional as F
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt
import cv2

# Import from existing modules
from compose_image_retrieval_demo import ComposedImageRetrievalSystem
from simple_cir_inference import SimpleCIRInference

# Add Grad-ECLIP path
sys.path.append(str(Path(__file__).parent.parent.parent / "Grad-Eclip"))


class GradECLIPHelper:
    """Helper class for Grad-ECLIP saliency map generation."""
    
    @staticmethod
    def grad_eclip(c, q_out, k_out, v, att_output, map_size, withksim=True):
        """
        Generate Grad-ECLIP saliency map.
        
        Args:
            c: Target scalar (similarity or norm)
            q_out, k_out, v: Attention components from Vision Transformer
            att_output: Attention output
            map_size: Tuple of (height, width) for the output map
            withksim: Whether to include cosine similarity weighting
            
        Returns:
            Saliency map as a 2D tensor
        """
        # Gradient on last attention output
        grad = torch.autograd.grad(c, att_output, retain_graph=True)[0]
        grad = grad.detach()
        grad_cls = grad[:1, 0, :]  # CLS token gradient
        
        if withksim:
            # Compute cosine similarity between CLS and patch tokens
            q_cls = q_out[:1, 0, :]
            k_patch = k_out[1:, 0, :]
            q_cls = F.normalize(q_cls, dim=-1)
            k_patch = F.normalize(k_patch, dim=-1)
            cosine_qk = (q_cls * k_patch).sum(-1)
            cosine_qk = (cosine_qk - cosine_qk.min()) / (cosine_qk.max() - cosine_qk.min())
            emap_lastv = F.relu_((grad_cls * v[1:, 0, :] * cosine_qk[:, None]).detach().sum(-1))
        else:
            emap_lastv = F.relu_((grad_cls * v[1:, 0, :]).detach().sum(-1))
            
        return emap_lastv.reshape(*map_size)
    
    @staticmethod
    def normalize_saliency_map(saliency_map):
        """Normalize saliency map to 0-1 range."""
        smap = saliency_map.clone()
        smap -= smap.min()
        if smap.max() > 0:
            smap /= smap.max()
        return smap
    
    @staticmethod
    def create_heatmap_overlay(image_array, saliency_map, alpha=0.4):
        """
        Create a heatmap overlay on the original image.
        
        Args:
            image_array: Original image as np.array (H, W, 3)
            saliency_map: Saliency map as np.array (H, W)
            alpha: Overlay transparency
            
        Returns:
            Overlaid image as np.array
        """
        # Apply colormap
        heatmap = cv2.applyColorMap((saliency_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = np.clip(image_array * (1 - alpha) + heatmap * alpha, 0, 255).astype(np.uint8)
        return overlay


class SaliencyEnabledCIRSystem(SimpleCIRInference):
    """
    Extended CIR system with Grad-ECLIP saliency map generation.
    """
    
    def __init__(self, database_path: str, clip_model_name: str = "ViT-B/32", 
                 eval_type: str = "searle", preprocess_type: str = "targetpad"):
        """Initialize the saliency-enabled CIR system."""
        super().__init__(database_path, clip_model_name, eval_type, preprocess_type)
        
        # Storage for hook outputs
        self.activation_hooks = {}
        self.gradient_hooks = {}
        self.hooked_activations = {}
        self.hooked_gradients = {}
        
        # Register hooks on the CLIP vision transformer
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the last transformer block."""
        # Get the last transformer block
        if hasattr(self.cir_system.clip_model.visual, 'transformer'):
            # ViT architecture
            last_block = list(self.cir_system.clip_model.visual.transformer.resblocks)[-1]
        else:
            raise ValueError("Unsupported CLIP architecture for saliency")
        
        # Forward hook to capture activations
        def forward_hook(module, input, output):
            self.hooked_activations['last_block'] = {
                'input': input[0] if isinstance(input, tuple) else input,
                'output': output
            }
        
        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.hooked_gradients['last_block'] = {
                'grad_input': grad_input,
                'grad_output': grad_output
            }
        
        # Register hooks
        self.activation_hooks['last_block'] = last_block.register_forward_hook(forward_hook)
        self.gradient_hooks['last_block'] = last_block.register_full_backward_hook(backward_hook)
    
    def _extract_attention_components(self, image_tensor):
        """
        Extract attention components (q, k, v) from the CLIP vision transformer.
        Adapted from Grad-ECLIP's clip_encode_dense function.
        """
        # Enable gradient computation
        image_tensor = image_tensor.requires_grad_(True)
        
        # Get vision transformer
        visual = self.cir_system.clip_model.visual
        
        # Convert to half precision for consistency with Grad-ECLIP
        x = image_tensor.to(visual.conv1.weight.dtype)
        
        # Process through initial layers (following Grad-ECLIP exactly)
        x = visual.conv1(x)  
        feah, feaw = x.shape[-2:]
        
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 
        class_embedding = visual.class_embedding.to(x.dtype)
        
        x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)
        
        # Handle positional embedding (adaptive for different resolutions)
        pos_embedding = visual.positional_embedding.to(x.dtype)
        tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
        
        clip_inres = visual.input_resolution
        clip_ksize = visual.conv1.kernel_size
        pos_h = clip_inres // clip_ksize[0]
        pos_w = clip_inres // clip_ksize[1]
        
        if img_pos.size(0) == (pos_h * pos_w):
            # Need to interpolate positional embedding
            img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
            img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic', align_corners=False)
            img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
            pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)
        else:
            # Use original positional embedding
            pos_embedding = pos_embedding.unsqueeze(0)
        
        x = x + pos_embedding
        x = visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Process through all transformer blocks except the last one
        x_in = x
        for block in visual.transformer.resblocks[:-1]:
            x_in = block(x_in)
        
        # Handle the last block separately to extract attention components
        target_tr = visual.transformer.resblocks[-1]
        x_before_attn = target_tr.ln_1(x_in)
        
        # Extract Q, K, V using the same approach as Grad-ECLIP
        linear = torch._C._nn.linear    
        q, k, v = linear(x_before_attn, target_tr.attn.in_proj_weight, target_tr.attn.in_proj_bias).chunk(3, dim=-1)
        
        # Compute attention using Grad-ECLIP's attention_layer function
        attn_output, attn = self._attention_layer(q, k, v, num_heads=1)
        x_after_attn = linear(attn_output, target_tr.attn.out_proj.weight, target_tr.attn.out_proj.bias)
        
        x = x_after_attn + x_in
        x_out = x + target_tr.mlp(target_tr.ln_2(x))
        
        # Complete the vision transformer
        x = x_out.permute(1, 0, 2)  # LND -> NLD
        x = visual.ln_post(x)
        final_features = x @ visual.proj
        
        # Get q_out, k_out for Grad-ECLIP (following their approach)
        with torch.no_grad():
            qkv = torch.stack((q, k, v), dim=0)
            qkv = linear(qkv, target_tr.attn.out_proj.weight, target_tr.attn.out_proj.bias)
            q_out, k_out, v_out = qkv[0], qkv[1], qkv[2]
        
        # Calculate map size (assuming square patches)
        num_patches = q.shape[0] - 1  # Subtract CLS token
        map_size = int(num_patches ** 0.5), int(num_patches ** 0.5)
        
        return final_features, q_out, k_out, v, attn_output, map_size
    
    def _attention_layer(self, q, k, v, num_heads=1, attn_mask=None):
        """
        Compute 'Scaled Dot Product Attention'.
        Adapted from Grad-ECLIP's attention_layer function.
        """
        tgt_len, bsz, embed_dim = q.shape
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_output_weights += attn_mask
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_heads = torch.bmm(attn_output_weights, v)
        attn_output = attn_output_heads.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
        attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights
    
    def generate_reference_saliency(self, reference_image_path: str) -> np.ndarray:
        """
        Generate saliency map for the reference image showing where φ looks.
        
        Args:
            reference_image_path: Path to the reference image
            
        Returns:
            Normalized saliency map as numpy array
        """
        print("🔍 Generating reference image saliency map...")
        
        # Load and preprocess image
        ref_image = PIL.Image.open(reference_image_path).convert('RGB')
        ref_tensor = self.cir_system.preprocess(ref_image).unsqueeze(0).to('cuda')
        
        # Extract attention components
        image_features, q_out, k_out, v_out, att_output, map_size = self._extract_attention_components(ref_tensor)
        
        # Pass through phi to get pseudo-word (use CLS token features)
        cls_features = image_features[:, 0, :]  # Extract CLS token features
        pseudo_word = self.cir_system.phi(cls_features)
        
        # Use L2 norm as the target (unbiased across embedding dimensions)
        norm_target = torch.norm(pseudo_word, p=2, dim=-1)
        
        # Generate saliency map
        saliency_map = GradECLIPHelper.grad_eclip(
            c=norm_target,
            q_out=q_out,
            k_out=k_out,
            v=v_out,
            att_output=att_output,
            map_size=map_size,
            withksim=True
        )
        
        # Normalize and convert to numpy
        saliency_map = GradECLIPHelper.normalize_saliency_map(saliency_map)
        
        # Resize to match original image size
        original_size = ref_image.size[::-1]  # PIL size is (W, H), we need (H, W)
        saliency_map = F.interpolate(
            saliency_map.unsqueeze(0).unsqueeze(0), 
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze().cpu().numpy()
        
        return saliency_map
    
    def generate_candidate_saliency(self, candidate_image_path: str, text_features: torch.Tensor) -> np.ndarray:
        """
        Generate saliency map for a candidate image showing similarity-driving regions.
        
        Args:
            candidate_image_path: Path to the candidate image
            text_features: Pre-computed text features with pseudo-word
            
        Returns:
            Normalized saliency map as numpy array
        """
        print(f"🔍 Generating candidate saliency map for {Path(candidate_image_path).name}...")
        
        # Load and preprocess image
        candidate_image = PIL.Image.open(candidate_image_path).convert('RGB')
        candidate_tensor = self.cir_system.preprocess(candidate_image).unsqueeze(0).to('cuda')
        
        # Extract attention components
        image_features, q_out, k_out, v_out, att_output, map_size = self._extract_attention_components(candidate_tensor)
        
        # Use CLS token features for similarity computation
        cls_features = image_features[:, 0, :]  # Extract CLS token features
        
        # Normalize features for cosine similarity
        image_features_norm = F.normalize(cls_features, dim=-1)
        text_features_norm = F.normalize(text_features, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.sum(image_features_norm * text_features_norm, dim=-1)
        
        # Generate saliency map
        saliency_map = GradECLIPHelper.grad_eclip(
            c=similarity,
            q_out=q_out,
            k_out=k_out,
            v=v_out,
            att_output=att_output,
            map_size=map_size,
            withksim=True
        )
        
        # Normalize and convert to numpy
        saliency_map = GradECLIPHelper.normalize_saliency_map(saliency_map)
        
        # Resize to match original image size
        original_size = candidate_image.size[::-1]  # PIL size is (W, H), we need (H, W)
        saliency_map = F.interpolate(
            saliency_map.unsqueeze(0).unsqueeze(0), 
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze().cpu().numpy()
        
        return saliency_map
    
    def query_with_saliency(self, reference_image_path: str, relative_caption: str, 
                           top_k: int = 10, generate_reference_saliency: bool = True,
                           generate_candidate_saliency: bool = True, 
                           max_candidate_saliency: int = 3) -> Dict:
        """
        Perform CIR query with saliency map generation.
        
        Args:
            reference_image_path: Path to the reference image
            relative_caption: Text describing desired modification
            top_k: Number of top results to return
            generate_reference_saliency: Whether to generate reference saliency map
            generate_candidate_saliency: Whether to generate candidate saliency maps
            max_candidate_saliency: Maximum number of candidates to generate saliency for
            
        Returns:
            Dictionary containing query results and saliency maps
        """
        # Perform regular query
        results = self.query(reference_image_path, relative_caption, top_k)
        
        output = {
            'query': {
                'reference_image': reference_image_path,
                'caption': relative_caption,
                'top_k': top_k
            },
            'results': [
                {'rank': i+1, 'image_name': name, 'similarity_score': float(score)}
                for i, (name, score) in enumerate(results)
            ],
            'saliency_maps': {}
        }
        
        # Generate reference saliency map
        if generate_reference_saliency:
            try:
                ref_saliency = self.generate_reference_saliency(reference_image_path)
                output['saliency_maps']['reference'] = ref_saliency
                print("✅ Reference saliency map generated")
            except Exception as e:
                print(f"⚠️  Failed to generate reference saliency: {e}")
        
        # Generate candidate saliency maps
        if generate_candidate_saliency and results:
            try:
                # Get text features with pseudo-word (needed for candidate saliency)
                # This requires re-implementing some of the query logic
                ref_image = PIL.Image.open(reference_image_path).convert('RGB')
                ref_tensor = self.cir_system.preprocess(ref_image).unsqueeze(0).to('cuda')
                ref_features = self.cir_system.clip_model.encode_image(ref_tensor)
                pseudo_tokens = self.cir_system.phi(ref_features)
                
                # Encode text with pseudo tokens (following SEARLE's format)
                import clip
                input_caption = f"a photo of $ that {relative_caption}"
                text_inputs = clip.tokenize([input_caption]).to('cuda')
                from encode_with_pseudo_tokens import encode_with_pseudo_tokens
                text_features = encode_with_pseudo_tokens(
                    self.cir_system.clip_model, text_inputs, pseudo_tokens
                )
                text_features = F.normalize(text_features.float()).to('cuda')
                
                output['saliency_maps']['candidates'] = {}
                
                # Generate saliency for top candidates
                num_candidates = min(max_candidate_saliency, len(results))
                for i in range(num_candidates):
                    image_name, score = results[i]
                    
                    # Try to resolve full image path
                    dataset_path = None
                    if self.dataset_info and 'dataset_path' in self.dataset_info:
                        dataset_path = self.dataset_info['dataset_path']
                    elif hasattr(self, '_dataset_path'):
                        dataset_path = self._dataset_path
                    
                    candidate_path = self._resolve_image_path(image_name, dataset_path)
                    
                    if candidate_path and Path(candidate_path).exists():
                        try:
                            candidate_saliency = self.generate_candidate_saliency(candidate_path, text_features)
                            output['saliency_maps']['candidates'][image_name] = {
                                'saliency_map': candidate_saliency,
                                'image_path': candidate_path,
                                'rank': i + 1,
                                'similarity_score': score
                            }
                            print(f"✅ Candidate saliency map generated for rank {i+1}")
                        except Exception as e:
                            print(f"⚠️  Failed to generate saliency for {image_name}: {e}")
                    else:
                        print(f"⚠️  Could not resolve path for {image_name}")
                
            except Exception as e:
                print(f"⚠️  Failed to generate candidate saliency maps: {e}")
        
        return output
    
    def save_saliency_visualizations(self, query_results: Dict, save_dir: str) -> None:
        """
        Save saliency map visualizations to disk.
        
        Args:
            query_results: Results from query_with_saliency
            save_dir: Directory to save visualizations
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        print(f"💾 Saving saliency visualizations to {save_path}")
        
        # Save reference saliency
        if 'reference' in query_results['saliency_maps']:
            ref_image = PIL.Image.open(query_results['query']['reference_image']).convert('RGB')
            ref_array = np.array(ref_image)
            ref_saliency = query_results['saliency_maps']['reference']
            
            # Create overlay
            overlay = GradECLIPHelper.create_heatmap_overlay(ref_array, ref_saliency)
            
            # Save
            PIL.Image.fromarray(overlay).save(save_path / "reference_heatmap.png")
            np.save(save_path / "reference_saliency.npy", ref_saliency)
            print("✅ Reference saliency visualization saved")
        
        # Save candidate saliency maps
        if 'candidates' in query_results['saliency_maps']:
            for image_name, candidate_data in query_results['saliency_maps']['candidates'].items():
                candidate_image = PIL.Image.open(candidate_data['image_path']).convert('RGB')
                candidate_array = np.array(candidate_image)
                candidate_saliency = candidate_data['saliency_map']
                
                # Create overlay
                overlay = GradECLIPHelper.create_heatmap_overlay(candidate_array, candidate_saliency)
                
                # Save with rank and similarity info
                rank = candidate_data['rank']
                score = candidate_data['similarity_score']
                safe_name = Path(image_name).stem.replace('/', '_')
                
                PIL.Image.fromarray(overlay).save(save_path / f"result_{rank}_heatmap_{safe_name}.png")
                np.save(save_path / f"result_{rank}_saliency_{safe_name}.npy", candidate_saliency)
                
            print(f"✅ {len(query_results['saliency_maps']['candidates'])} candidate saliency visualizations saved")
    
    def __del__(self):
        """Clean up hooks when the object is destroyed."""
        for hook in self.activation_hooks.values():
            hook.remove()
        for hook in self.gradient_hooks.values():
            hook.remove()


def main(args):
    """Main function for saliency-enabled CIR inference."""
    
    # Validate arguments
    if args.top_k < 1:
        print("❌ Error: top-k must be at least 1")
        sys.exit(1)
    
    try:
        # Initialize saliency-enabled inference system
        print("🚀 Initializing Saliency-Enabled Composed Image Retrieval system...")
        inference = SaliencyEnabledCIRSystem(
            database_path=args.database_path,
            clip_model_name=args.clip_model_name,
            eval_type=args.eval_type,
            preprocess_type=args.preprocess_type
        )
        # Store dataset path for candidate image resolution
        inference._dataset_path = args.dataset_path
        
        if args.generate_saliency:
            # Perform query with saliency generation
            query_results = inference.query_with_saliency(
                reference_image_path=args.reference_image,
                relative_caption=args.caption,
                top_k=args.top_k,
                generate_reference_saliency=args.generate_reference_saliency,
                generate_candidate_saliency=args.generate_candidate_saliency,
                max_candidate_saliency=args.max_candidate_saliency
            )
            
            # Output results
            if args.output_format == "json":
                import json
                print("\n" + json.dumps(query_results, indent=2, default=str))
            else:
                # Text format
                print(f"\n📋 Top {len(query_results['results'])} Results:")
                print("=" * 60)
                for result in query_results['results']:
                    print(f"{result['rank']:2d}. {result['image_name']:<40} (similarity: {result['similarity_score']:.4f})")
            
            # Save saliency visualizations
            if args.save_saliency_dir:
                inference.save_saliency_visualizations(query_results, args.save_saliency_dir)
            
        else:
            # Regular query without saliency
            results = inference.query(
                reference_image_path=args.reference_image,
                caption=args.caption,
                top_k=args.top_k
            )
            
            # Output results
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
        
        print(f"\n🎉 Query completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEARLE CIR with Grad-ECLIP Saliency Maps')
    
    # Required arguments
    parser.add_argument('--database-path', required=True, help='Path to the saved database file')
    parser.add_argument('--reference-image', required=True, help='Path to the reference image')
    parser.add_argument('--caption', required=True, help='Text describing the desired modification')
    
    # Optional arguments
    parser.add_argument('--top-k', type=int, default=10, help='Number of top results to return')
    parser.add_argument('--clip-model-name', default='ViT-B/32', help='CLIP model name')
    parser.add_argument('--eval-type', default='searle', help='Evaluation type')
    parser.add_argument('--preprocess-type', default='targetpad', help='Preprocessing type')
    parser.add_argument('--output-format', choices=['text', 'json'], default='text', help='Output format')
    parser.add_argument('--dataset-path', help='Base path to the dataset (for resolving image paths)')
    
    # Saliency-specific arguments
    parser.add_argument('--generate-saliency', action='store_true', help='Generate saliency maps')
    parser.add_argument('--generate-reference-saliency', action='store_true', default=True, 
                       help='Generate saliency map for reference image')
    parser.add_argument('--generate-candidate-saliency', action='store_true', default=True,
                       help='Generate saliency maps for candidate images')
    parser.add_argument('--max-candidate-saliency', type=int, default=3,
                       help='Maximum number of candidates to generate saliency for')
    parser.add_argument('--save-saliency-dir', help='Directory to save saliency visualizations')
    
    args = parser.parse_args()
    main(args) 