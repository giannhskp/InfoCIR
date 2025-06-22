#!/usr/bin/env python3
"""
Saliency-Enabled CIR System for the CIR App

This module provides saliency map generation capabilities that can be applied
to existing CIR systems. It wraps the existing ComposedImageRetrievalSystem
and adds visual explanation capabilities.

Usage:
    from src.saliency import SaliencyManager
    saliency_manager = SaliencyManager()
    results = saliency_manager.query_with_saliency(cir_system, reference_path, caption, top_k)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import tempfile
import torch
import torch.nn.functional as F
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import clip
import re
import concurrent.futures  # Added for parallel saving
from functools import partial


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
    
    @staticmethod
    def sim_qk(q, k, eos_position):
        """Compute similarity between query and key vectors (for text attribution)."""
        q_cls = F.normalize(q[eos_position, 0, :], dim=-1) 
        k_patch = F.normalize(k[:, 0, :], dim=-1)
        
        cosine_qk = (q_cls * k_patch).sum(-1)  
        cosine_qk = (cosine_qk - cosine_qk.min()) / (cosine_qk.max() - cosine_qk.min())
        return cosine_qk
    
    @staticmethod
    def grad_eclip_text(c, qs, ks, vs, attn_outputs, eos_position):
        """
        Generate Grad-ECLIP text token attribution.
        
        Args:
            c: Target scalar (similarity score)
            qs, ks, vs: Query, key, value tensors from text transformer
            attn_outputs: Attention outputs from text transformer layers
            eos_position: Position of end-of-sequence token
            
        Returns:
            Token attribution scores
        """
        tmp_maps = []
        for q, k, v, attn_output in zip(qs, ks, vs, attn_outputs):
            try:
                grad = torch.autograd.grad(
                    c, attn_output, 
                    retain_graph=True, 
                    allow_unused=True,
                    create_graph=False
                )[0]
                
                if grad is None:
                    # Skip this layer if gradient is None
                    continue
                    
                grad_cls = grad[eos_position, 0, :]
                
                # Use gradient on the EOS token position  
                cosine_qk = GradECLIPHelper.sim_qk(q, k, eos_position)
                tmp_maps.append((grad_cls * v[:, 0, :] * cosine_qk[:, None]).sum(-1))
            except Exception as e:
                print(f"⚠️  Skipping layer due to gradient error: {e}")
                continue

        if not tmp_maps:
            # If no gradients were computed, return zeros
            return torch.zeros(eos_position - 1)
            
        emap = F.relu_(torch.stack(tmp_maps, dim=0).sum(0))
        emap = emap[1:eos_position].flatten()  # Exclude start/end tokens
        emap = emap / (emap.sum() + 1e-8)  # Normalize with small epsilon
        return emap


class SaliencyManager:
    """
    A manager class that wraps existing CIR systems to add saliency capabilities.
    
    This class can work with any CIR system that has:
    - .clip_model attribute
    - .preprocess attribute  
    - .phi attribute (for SEARLE-based systems)
    - .query() method
    """
    
    def __init__(self):
        """Initialize the saliency manager."""
        # Storage for hook outputs
        self.activation_hooks = {}
        self.gradient_hooks = {}
        self.hooked_activations = {}
        self.hooked_gradients = {}
        self._current_cir_system = None
        
    def _register_hooks(self, cir_system):
        """Register forward and backward hooks on the last transformer block."""
        # Clear any existing hooks
        self._clear_hooks()
        
        # Get the last transformer block
        if hasattr(cir_system.clip_model.visual, 'transformer'):
            # ViT architecture
            last_block = list(cir_system.clip_model.visual.transformer.resblocks)[-1]
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
        
        self._current_cir_system = cir_system
    
    def _clear_hooks(self):
        """Clear all registered hooks."""
        for hook in self.activation_hooks.values():
            hook.remove()
        for hook in self.gradient_hooks.values():
            hook.remove()
        self.activation_hooks.clear()
        self.gradient_hooks.clear()
        self.hooked_activations.clear()
        self.hooked_gradients.clear()
    
    def _extract_attention_components(self, cir_system, image_tensor):
        """
        Extract attention components (q, k, v) from the CLIP vision transformer.
        Adapted from Grad-ECLIP's clip_encode_dense function.
        """
        # Enable gradient computation
        image_tensor = image_tensor.requires_grad_(True)
        
        # Get vision transformer
        visual = cir_system.clip_model.visual
        
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
    
    def generate_reference_saliency(self, cir_system, reference_image_path: str) -> np.ndarray:
        """
        Generate saliency map for the reference image showing where φ looks.
        
        Args:
            cir_system: The CIR system to use
            reference_image_path: Path to the reference image
            
        Returns:
            Normalized saliency map as numpy array
        """
        print("🔍 Generating reference image saliency map...")
        
        # Load and preprocess image
        ref_image = PIL.Image.open(reference_image_path).convert('RGB')
        ref_tensor = cir_system.preprocess(ref_image).unsqueeze(0).to('cuda')
        
        # Register hooks if not already done or if CIR system changed
        if self._current_cir_system != cir_system:
            self._register_hooks(cir_system)
        
        # Set model to training mode temporarily for gradient computation
        was_training = cir_system.clip_model.training
        cir_system.clip_model.train()
        
        try:
            # Extract attention components
            image_features, q_out, k_out, v_out, att_output, map_size = self._extract_attention_components(cir_system, ref_tensor)
            
            # Pass through phi to get pseudo-word (use CLS token features)
            cls_features = image_features[:, 0, :]  # Extract CLS token features
            pseudo_word = cir_system.phi(cls_features)
            
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
            
        finally:
            # Restore original model mode
            cir_system.clip_model.train(was_training)
    
    def generate_candidate_saliency(self, cir_system, candidate_image_path: str, text_features: torch.Tensor) -> np.ndarray:
        """
        Generate saliency map for a candidate image showing similarity-driving regions.
        
        Args:
            cir_system: The CIR system to use
            candidate_image_path: Path to the candidate image
            text_features: Pre-computed text features with pseudo-word
            
        Returns:
            Normalized saliency map as numpy array
        """
        print(f"🔍 Generating candidate saliency map for {Path(candidate_image_path).name}...")
        
        # Load and preprocess image
        candidate_image = PIL.Image.open(candidate_image_path).convert('RGB')
        candidate_tensor = cir_system.preprocess(candidate_image).unsqueeze(0).to('cuda')
        
        # Register hooks if not already done or if CIR system changed
        if self._current_cir_system != cir_system:
            self._register_hooks(cir_system)
        
        # Set model to training mode temporarily for gradient computation
        was_training = cir_system.clip_model.training
        cir_system.clip_model.train()
        
        try:
            # Extract attention components
            image_features, q_out, k_out, v_out, att_output, map_size = self._extract_attention_components(cir_system, candidate_tensor)
            
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
            
        finally:
            # Restore original model mode
            cir_system.clip_model.train(was_training)
    
    def generate_reference_saliency_for_candidate(self, cir_system, reference_image_path: str, candidate_features: torch.Tensor, relative_caption: str) -> np.ndarray:
        """
        Generate a per-candidate saliency map on the reference image.
        
        The map is obtained by back-propagating the *same* retrieval score that is
        used for ranking (cosine of CLS_candidate and text features that include
        φ(reference) as the `$` placeholder) all the way to the reference image.
        """
        print(f"🔍 Generating reference saliency for candidate...")
        
        # Register hooks for this CIR system
        self._register_hooks(cir_system)
        
        # Store original training mode
        was_training = cir_system.clip_model.training
        cir_system.clip_model.eval()
        
        try:
            # 1. Pre-process reference image and extract attention components (with grads)
            ref_img = PIL.Image.open(reference_image_path).convert('RGB')
            ref_tensor = cir_system.preprocess(ref_img).unsqueeze(0).to('cuda')

            img_features, q_out, k_out, v_out, att_output, map_size = self._extract_attention_components(cir_system, ref_tensor)

            # CLS token features with gradient
            cls_ref = img_features[:, 0, :]  # [1, D]

            # 2. Compute pseudo-tokens via φ(reference) (keeps gradient path)
            pseudo_tokens = cir_system.phi(cls_ref)

            # 3. Build text features with `$` → pseudo_tokens
            input_caption = f"a photo of $ that {relative_caption}"
            text_inputs = clip.tokenize([input_caption]).to('cuda')
            
            # Import the encode function
            from src.callbacks.SEARLE.src.encode_with_pseudo_tokens import encode_with_pseudo_tokens
            text_features = encode_with_pseudo_tokens(
                cir_system.clip_model, text_inputs, pseudo_tokens
            )  # shape [1, D]

            # 4. Compute similarity scalar with *candidate* features (no grad on candidate)
            cand_norm = F.normalize(candidate_features.detach(), dim=-1)
            text_norm = F.normalize(text_features, dim=-1)
            similarity = torch.sum(cand_norm * text_norm, dim=-1)

            # 5. Grad-ECLIP on reference image
            saliency_map = GradECLIPHelper.grad_eclip(
                c=similarity,
                q_out=q_out,
                k_out=k_out,
                v=v_out,
                att_output=att_output,
                map_size=map_size,
                withksim=True
            )

            saliency_map = GradECLIPHelper.normalize_saliency_map(saliency_map)
            original_size = ref_img.size[::-1]
            saliency_map = F.interpolate(
                saliency_map.unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()

            return saliency_map
            
        finally:
            # Restore original model mode
            cir_system.clip_model.train(was_training)
    
    def _clip_encode_text_dense_with_pseudo(self, cir_system, text_tokens: torch.Tensor, pseudo_tokens: torch.Tensor, n_layers: int = 8):
        """Same as _clip_encode_text_dense but replaces the $ token embedding with provided pseudo token(s)."""
        clip_model = cir_system.clip_model
        
        # Initial embedding
        x = clip_model.token_embedding(text_tokens).type(clip_model.dtype)  # [B, N_ctx, D]
        
        # Replace $ (token id 259) embedding with pseudo token
        dollar_mask = (text_tokens == 259)  # shape [B, N_ctx]
        if dollar_mask.sum() > 0:
            # Ensure shape compatibility (B, D)
            if pseudo_tokens.dim() == 1:
                pseudo_tokens = pseudo_tokens.unsqueeze(0)
            x[dollar_mask] = pseudo_tokens.to(x.dtype)
        
        attn_mask = clip_model.build_attention_mask().to(dtype=x.dtype, device=x.device)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Forward through early layers
        for block in clip_model.transformer.resblocks[:-n_layers]:
            x = block(x)
        
        # Collect components from last n_layers
        attns, attn_outputs = [], []
        qs, ks, vs = [], [], []
        for TR in clip_model.transformer.resblocks[-n_layers:]:
            x_in = x
            x = TR.ln_1(x_in)
            linear = torch._C._nn.linear
            q, k, v = linear(x, TR.attn.in_proj_weight, TR.attn.in_proj_bias).chunk(3, dim=-1)
            
            # Ensure Q, K, V require gradients
            q.requires_grad_(True)
            k.requires_grad_(True) 
            v.requires_grad_(True)
            
            attn_output, attn = self._attention_layer(q, k, v, num_heads=1, attn_mask=attn_mask)
            
            # Ensure attention output requires gradients
            attn_output.requires_grad_(True)
            
            attns.append(attn)
            attn_outputs.append(attn_output)
            vs.append(v)
            qs.append(q)
            ks.append(k)
            
            x_after_attn = linear(attn_output, TR.attn.out_proj.weight, TR.attn.out_proj.bias)       
            x = x_after_attn + x_in
            x = x + TR.mlp(TR.ln_2(x))
                
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)
        
        # Take features from the EOT embedding
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ clip_model.text_projection
        
        return x, (qs, ks, vs), attns, attn_outputs
    
    def generate_text_attribution(self, cir_system, text_caption: str, image_features: torch.Tensor, 
                                 attribution_type: str = "candidate", image_path: str = None,
                                 pseudo_tokens: torch.Tensor = None) -> Dict:
        """
        Generate token-level attribution for text prompt based on image-text interaction.
        
        Args:
            cir_system: The CIR system to use
            text_caption: The input caption (e.g., "as a cartoon character")
            image_features: Pre-computed image features for this specific image
            attribution_type: Type of attribution ("reference" or "candidate")
            image_path: Path to the image (for debugging/logging)
            pseudo_tokens: Pseudo tokens from φ network
            
        Returns:
            Dict with token attributions and readable tokens
        """
        print(f"🔍 Generating {attribution_type} text attribution for {Path(image_path).name if image_path else 'image'}...")
        
        # Create SEARLE format text
        full_prompt = f"a photo of $ that {text_caption}"
        
        # Tokenize text
        tokenized_text = clip.tokenize([full_prompt]).to('cuda')
        
        # Ensure image features are properly formatted and require gradients
        if image_features.dim() == 3:  # If it has sequence dimension, extract CLS token
            image_features = image_features[:, 0, :]  # Extract CLS token
        image_features = F.normalize(image_features.detach().requires_grad_(True), dim=-1)
        
        # Use provided pseudo tokens or fallback to zeros
        if pseudo_tokens is None:
            pseudo_tokens = torch.zeros((1, image_features.shape[-1]), device='cuda', dtype=image_features.dtype)
        
        text_features_new, (qs, ks, vs), attns, attn_outputs = self._clip_encode_text_dense_with_pseudo(cir_system, tokenized_text, pseudo_tokens)
        
        # Find EOS position
        eos_position = tokenized_text.argmax(dim=-1).item()
        
        # Normalize text features for proper similarity computation
        text_features_normalized = F.normalize(text_features_new, dim=-1)
        
        # Compute ACTUAL IMAGE-TEXT INTERACTION as target
        if attribution_type == "reference":
            # For reference: How text helps φ network process the reference image
            # Use the reference image to get pseudo-word features
            pseudo_word = cir_system.phi(image_features)
            # Target: How well text features align with pseudo-word enhanced features
            target = F.cosine_similarity(text_features_normalized, pseudo_word, dim=-1)
        else:
            # For candidates: Actual similarity with this specific candidate image
            target = F.cosine_similarity(image_features, text_features_normalized, dim=-1)
        
        # Generate attribution using Grad-ECLIP
        attribution_scores = GradECLIPHelper.grad_eclip_text(
            c=target,
            qs=qs,
            ks=ks, 
            vs=vs,
            attn_outputs=attn_outputs,
            eos_position=eos_position
        )
        
        # Decode tokens for human readability
        try:
            # Use CLIP's tokenizer for accurate decoding
            from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
            tokenizer = _Tokenizer()
            token_ids = tokenized_text.squeeze().cpu().numpy()
            tokens = [tokenizer.decode([token_id]) for token_id in token_ids[1:eos_position]]
        except (ImportError, AttributeError):
            # Fallback to basic CLIP token decoding
            tokens = [f"token_{i}" for i in range(len(attribution_scores))]
        
        # Ensure tokens and scores match in length
        min_len = min(len(tokens), len(attribution_scores))
        tokens = tokens[:min_len]
        attribution_scores = attribution_scores[:min_len]
        
        return {
            'tokens': tokens,
            'attributions': attribution_scores.detach().cpu().numpy(),
            'full_prompt': full_prompt,
            'eos_position': eos_position
        }
    
    def create_text_attribution_visualization(self, text_attribution: Dict, save_path: str) -> None:
        """
        Create and save text attribution visualization.
        
        Args:
            text_attribution: Output from generate_text_attribution
            save_path: Path to save the visualization
        """
        tokens = text_attribution['tokens']
        attributions = text_attribution['attributions']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color map for attributions (normalize to 0-1)
        norm_attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)
        
        # Create bar plot
        bars = ax.bar(range(len(tokens)), attributions, color=plt.cm.Reds(norm_attributions))
        
        # Set labels
        ax.set_xlabel('Tokens', fontsize=12)
        ax.set_ylabel('Attribution Score', fontsize=12)
        ax.set_title(f'Text Token Attribution\n"{text_attribution["full_prompt"]}"', fontsize=14, fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (token, attr) in enumerate(zip(tokens, attributions)):
            ax.text(i, attr + 0.01, f'{attr:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Text attribution visualization saved to: {save_path}")

    def query_with_saliency(self, cir_system, reference_image_path: str, relative_caption: str, 
                           top_k: int = 10, dataset_path: Optional[str] = None,
                           generate_reference_saliency: bool = True,
                           generate_candidate_saliency: bool = True, 
                           max_candidate_saliency: int = 3,
                           generate_text_attribution: bool = True) -> Dict:
        """
        Perform CIR query with saliency map generation.
        
        Args:
            cir_system: The CIR system to use
            reference_image_path: Path to the reference image
            relative_caption: Text describing desired modification
            top_k: Number of top results to return
            dataset_path: Base path to dataset for resolving candidate image paths
            generate_reference_saliency: Whether to generate reference saliency map
            generate_candidate_saliency: Whether to generate candidate saliency maps
            max_candidate_saliency: Maximum number of candidates to generate saliency for (None = all top_k)
            generate_text_attribution: Whether to generate text token attribution
            
        Returns:
            Dictionary containing query results and saliency maps
        """
        print(f"🚀 Performing CIR query with saliency generation...")
        
        # Perform regular query using the existing CIR system
        results = cir_system.query(reference_image_path, relative_caption, top_k)
        
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
        if generate_reference_saliency and hasattr(cir_system, 'phi'):
            try:
                ref_saliency = self.generate_reference_saliency(cir_system, reference_image_path)
                output['saliency_maps']['reference'] = ref_saliency
                print("✅ Reference saliency map generated")
            except Exception as e:
                print(f"⚠️  Failed to generate reference saliency: {e}")
        
        # Generate candidate saliency maps
        if generate_candidate_saliency and results and hasattr(cir_system, 'phi'):
            try:
                # Get text features with pseudo-word (needed for candidate saliency)
                ref_image = PIL.Image.open(reference_image_path).convert('RGB')
                ref_tensor = cir_system.preprocess(ref_image).unsqueeze(0).to('cuda')
                
                with torch.no_grad():
                    ref_features = cir_system.clip_model.encode_image(ref_tensor)
                    
                    # Compute φ(reference) pseudo tokens
                    pseudo_tokens = cir_system.phi(ref_features)
                    
                    # Encode text with pseudo tokens (following SEARLE's format) for candidate saliency
                    input_caption = f"a photo of $ that {relative_caption}"
                    text_inputs = clip.tokenize([input_caption]).to('cuda')
                    
                    # Import the encode function
                    from src.callbacks.SEARLE.src.encode_with_pseudo_tokens import encode_with_pseudo_tokens
                    text_features = encode_with_pseudo_tokens(
                        cir_system.clip_model, text_inputs, pseudo_tokens
                    )
                    text_features = F.normalize(text_features.float()).to('cuda')
                
                output['saliency_maps']['candidates'] = {}
                
                # Generate saliency for top candidates
                # If max_candidate_saliency is None, generate for all results
                if max_candidate_saliency is None:
                    num_candidates = len(results)
                else:
                    num_candidates = min(max_candidate_saliency, len(results))
                for i in range(num_candidates):
                    image_name, score = results[i]
                    
                    # Try to resolve full image path
                    candidate_path = self._resolve_image_path(image_name, dataset_path)
                    
                    if candidate_path and Path(candidate_path).exists():
                        try:
                            # Generate candidate saliency
                            candidate_saliency = self.generate_candidate_saliency(cir_system, candidate_path, text_features)
                            
                            # Generate per-candidate reference saliency
                            candidate_image = PIL.Image.open(candidate_path).convert('RGB')
                            candidate_tensor = cir_system.preprocess(candidate_image).unsqueeze(0).to('cuda')
                            with torch.no_grad():
                                candidate_features = cir_system.clip_model.encode_image(candidate_tensor)
                            
                            reference_cond_saliency = self.generate_reference_saliency_for_candidate(
                                cir_system,
                                reference_image_path,
                                candidate_features=candidate_features,
                                relative_caption=relative_caption
                            )
                            
                            output['saliency_maps']['candidates'][image_name] = {
                                'saliency_map': candidate_saliency,
                                'reference_saliency': reference_cond_saliency,
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
        
        # Generate text attribution
        if generate_text_attribution and hasattr(cir_system, 'phi'):
            try:
                output['text_attribution'] = {}
                
                # Reference text attribution (how text relates to reference processing)
                ref_image = PIL.Image.open(reference_image_path).convert('RGB')
                ref_tensor = cir_system.preprocess(ref_image).unsqueeze(0).to('cuda')
                with torch.no_grad():
                    ref_features = cir_system.clip_model.encode_image(ref_tensor)
                    # Compute φ(reference) pseudo tokens for attribution generation
                    pseudo_tokens = cir_system.phi(ref_features)
                
                ref_text_attr = self.generate_text_attribution(
                    cir_system,
                    relative_caption, 
                    ref_features, 
                    attribution_type="reference",
                    image_path=reference_image_path,
                    pseudo_tokens=pseudo_tokens
                )
                output['text_attribution']['reference'] = ref_text_attr
                print("✅ Reference text attribution generated")
                
                # Candidate text attribution (how text relates to each candidate)
                output['text_attribution']['candidates'] = {}
                
                # Process top candidates
                # If max_candidate_saliency is None, generate for all results
                if max_candidate_saliency is None:
                    num_candidates = len(results)
                else:
                    num_candidates = min(max_candidate_saliency, len(results))
                for i in range(num_candidates):
                    image_name, score = results[i]
                    
                    try:
                        # Try to resolve full image path
                        candidate_path = self._resolve_image_path(image_name, dataset_path)
                        
                        if candidate_path and Path(candidate_path).exists():
                            # Load candidate image and get features
                            candidate_image = PIL.Image.open(candidate_path).convert('RGB')
                            candidate_tensor = cir_system.preprocess(candidate_image).unsqueeze(0).to('cuda')
                            with torch.no_grad():
                                candidate_features = cir_system.clip_model.encode_image(candidate_tensor)
                            
                            candidate_text_attr = self.generate_text_attribution(
                                cir_system,
                                relative_caption, 
                                candidate_features, 
                                attribution_type="candidate",
                                image_path=candidate_path,
                                pseudo_tokens=pseudo_tokens
                            )
                            output['text_attribution']['candidates'][image_name] = candidate_text_attr
                            print(f"✅ Text attribution generated for {image_name}")
                        else:
                            print(f"⚠️  Could not resolve path for {image_name}")
                            
                    except Exception as e:
                        print(f"⚠️  Failed to generate text attribution for {image_name}: {e}")
                
                print("✅ Text attribution analysis completed")
                
            except Exception as e:
                print(f"⚠️  Failed to generate text attribution: {e}")
        
        return output
    
    def _resolve_image_path(self, image_name: str, dataset_path: Optional[str]) -> Optional[str]:
        """
        Resolve the full path to an image given its name/ID and dataset path.
        
        Args:
            image_name: Image name or ID
            dataset_path: Base dataset path
            
        Returns:
            Full path to the image or None if not found
        """
        if not dataset_path:
            return None
            
        from src.Dataset import Dataset
        df = Dataset.get()
        
        try:
            # Try numeric index lookup
            idx = int(image_name)
            if idx in df.index:
                return df.loc[idx]['image_path']
        except (ValueError, TypeError):
            # Try string index lookup
            if image_name in df.index:
                return df.loc[image_name]['image_path']
        
        return None
    
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
        
        # ---------------------------------------------------------------
        # 1. Pre-load reference image once to avoid redundant disk I/O
        # ---------------------------------------------------------------
        ref_array = None
        if 'reference' in query_results['saliency_maps']:
            ref_image = PIL.Image.open(query_results['query']['reference_image']).convert('RGB')
            ref_array = np.array(ref_image)
            ref_saliency = query_results['saliency_maps']['reference']
            
            # Create overlay
            overlay = GradECLIPHelper.create_heatmap_overlay(ref_array, ref_saliency)
            
            # Save with a faster PNG compression level (1 ≈ fastest, 9 ≈ slowest)
            _fast_write_image(overlay, save_path / "reference_heatmap.png", compression=1)
            # np.save(save_path / "reference_saliency.npy", ref_saliency)
            print("✅ Reference saliency visualization saved")
        
        # -----------------------------------------------
        # 2. Save candidate saliency maps in parallel
        # -----------------------------------------------
        if 'candidates' in query_results['saliency_maps']:
            def _process_candidate(item):
                """Inner helper to process a single candidate (runs in a thread)."""
                image_name, candidate_data = item
                try:
                    candidate_image = PIL.Image.open(candidate_data['image_path']).convert('RGB')
                    candidate_array = np.array(candidate_image)
                    candidate_saliency = candidate_data['saliency_map']
                    
                    # Create candidate overlay
                    overlay = GradECLIPHelper.create_heatmap_overlay(candidate_array, candidate_saliency)
                    
                    rank = candidate_data['rank']
                    safe_name = Path(image_name).stem.replace('/', '_')
                    
                    # Save candidate heatmap
                    _fast_write_image(
                        overlay,
                        save_path / f"result_{rank}_heatmap_{safe_name}.png",
                        compression=1,
                    )
                    
                    # Optional: per-candidate reference saliency
                    if 'reference_saliency' in candidate_data and ref_array is not None:
                        ref_sal = candidate_data['reference_saliency']
                        ref_overlay = GradECLIPHelper.create_heatmap_overlay(ref_array, ref_sal)
                        _fast_write_image(ref_overlay, save_path / f"result_{rank}_ref_heatmap_{safe_name}.png", compression=1)
                except Exception as e:
                    print(f"⚠️  Failed to save visualization for {image_name}: {e}")
            
            candidates_items = list(query_results['saliency_maps']['candidates'].items())
            max_workers = min(os.cpu_count() or 4, len(candidates_items))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_process_candidate, candidates_items))
            
            print(f"✅ {len(candidates_items)} candidate saliency visualizations saved")
        
        # # Save text attribution visualizations
        # if 'text_attribution' in query_results:
        #     print("💾 Saving text attribution visualizations...")
            
        #     # Save reference text attribution
        #     if 'reference' in query_results['text_attribution']:
        #         ref_attr_path = save_path / "reference_text_attribution.png"
        #         self.create_text_attribution_visualization(
        #             query_results['text_attribution']['reference'], 
        #             str(ref_attr_path)
        #         )
            
        #     # Save candidate text attributions
        #     if 'candidates' in query_results['text_attribution']:
        #         for image_name, text_attr in query_results['text_attribution']['candidates'].items():
        #             safe_name = Path(image_name).stem.replace('/', '_')
        #             attr_path = save_path / f"text_attribution_{safe_name}.png"
        #             self.create_text_attribution_visualization(text_attr, str(attr_path))
                
        #         print(f"✅ {len(query_results['text_attribution']['candidates'])} text attribution visualizations saved")
    
    def __del__(self):
        """Clean up hooks when the object is destroyed."""
        self._clear_hooks() 

# -----------------------------------------------------------------------------
# Fast image writer (OpenCV is ~2-3× faster than Pillow for PNG/JPEG encoding)
# -----------------------------------------------------------------------------

def _fast_write_image(rgb_array: np.ndarray, out_path: Path, *, format: str = "png", compression: int = 1) -> None:
    """Save an RGB uint8 numpy array to disk quickly using OpenCV."""
    if format.lower() == "png":
        # OpenCV expects BGR
        bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    elif format.lower() == "jpg" or format.lower() == "jpeg":
        bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        # Fallback to Pillow if unsupported format
        PIL.Image.fromarray(rgb_array).save(out_path) 