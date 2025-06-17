"""
Visual explanation utilities for SEARLE / CLIP models.

This module implements lightweight, gradient-based saliency maps for images and
attribution scores for tokens.  Only the default pipeline (eval-type *searle*
/ *searle-xl* and *targetpad* preprocessing) is considered.

The core class – `SearleGradCAM` – registers forward + backward hooks on the
last residual block of the CLIP visual transformer and on the token embedding
layer of the text tower.  In a single backward pass we recover:

1. A per-patch heat-map for the inspected **candidate** image that highlights
   the regions most responsible for the dot-product similarity with the
   composed query.
2. (Optional) An equivalent saliency vector for every token in the caption so
   that UIs can emphasise influential words.

For the **reference** image we back-propagate the L2-norm of the *pseudo*
(tokens produced by the Φ / SEARLE network) to reveal which pixels inform that
synthetic token.

All heavy lifting is handled in pure PyTorch – no external Grad-CAM package is
required – and runs happily under `torch.cuda.amp` for speed.
"""

from __future__ import annotations

import hashlib
import math
import types
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torchvision.transforms.functional import resize as tv_resize
from torchvision.transforms import InterpolationMode
from PIL import Image

# Fall-back device (will be overwritten by utils.device if available)
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _HookStore:
    """Container to keep activations / gradients captured by hooks."""

    def __init__(self):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.text_embeddings: Optional[torch.Tensor] = None
        self.text_grads: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------------
    # Forward & backward hooks used internally by `SearleGradCAM`
    # ---------------------------------------------------------------------

    def forward_hook(self, _module, _inp, output):  # noqa: D401, N802 <- (pytorch convention)
        # ViT output shape → (batch, seq_len, channels)
        self.activations = output.detach()
        # Retain gradient on tensor that is *not* a leaf in the graph.
        output.retain_grad()

    def backward_hook(self, _module, _grad_input, grad_output):  # noqa: D401, N802
        # grad_output is a tuple – grab first element.
        self.gradients = grad_output[0].detach()

    def text_forward(self, _module, _inp, output):  # noqa: D401
        self.text_embeddings = output.detach()
        output.retain_grad()

    def text_backward(self, _module, _grad_input, grad_output):  # noqa: D401
        self.text_grads = grad_output[0].detach()


class SearleGradCAM:
    """Generate image + token saliency maps for SEARLE composed retrieval."""

    def __init__(
        self,
        clip_model,
        preprocess_fn,
        phi_network=None,
        device: torch.device | None = None,
        cache_size: int = 128,
    ):
        """Construct the Grad-CAM helper.

        Parameters
        ----------
        clip_model
            A loaded CLIP model exactly as used by the retrieval code.
        preprocess_fn
            Callable performing the *targetpad* preprocessing and normalisation
            expected by the model.
        phi_network
            The phi network for generating pseudo tokens (needed for reference heatmaps).
        device
            Where to run the computations – defaults to CUDA if available.
        cache_size
            Maximum #entries retained in the heat-map LRU cache.
        """
        self.clip_model = clip_model.eval()
        self.preprocess = preprocess_fn
        self.phi = phi_network
        self.device = device or DEFAULT_DEVICE

        # Very small LRU cache:  (key -> (heatmap[N,H,W], token_scores[M]))
        self._cache: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self._cache_size = cache_size

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enable_hooks(self):
        """No-op for compatibility - we use input gradients instead of hooks."""
        pass

    def remove_hooks(self):
        """No-op for compatibility - we use input gradients instead of hooks."""
        pass

    # ------------------------------------------------------------------
    # Core saliency helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        """Helper that preprocesses *image* and encodes it with the CLIP visual tower."""
        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        else:
            # Already tensor with range expected by model.
            image_tensor = image.to(self.device, non_blocking=True)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            return self.clip_model.encode_image(image_tensor)

    # ------------------------------------------------------------------
    # Public API – candidate & reference heat-maps
    # ------------------------------------------------------------------

    def candidate_heatmap(
        self,
        candidate_pil: Image.Image,
        query_features: torch.Tensor,
        prompt: str,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Return (image_heatmap, token_saliency) for *candidate_pil*.

        Uses input gradients to generate saliency maps.
        """
        self.enable_hooks()

        # -- Check cache – key combines image path hash & prompt hash --------
        cache_key = self._make_cache_key(candidate_pil, prompt)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Preprocess image and enable gradients
        image_tensor = self.preprocess(candidate_pil).unsqueeze(0).to(self.device)
        image_tensor.requires_grad_(True)

        # Temporarily set model to train mode for gradient computation
        was_training = self.clip_model.training
        phi_was_training = self.phi.training if self.phi else None
        self.clip_model.train()
        if self.phi:
            self.phi.train()
        
        try:
            # Forward pass
            cand_features = self.clip_model.encode_image(image_tensor)
            cand_features = F.normalize(cand_features, dim=-1)
            sim = (cand_features * query_features).sum()

            # Backward to get input gradients
            self.clip_model.zero_grad(set_to_none=True)
            if self.phi:
                self.phi.zero_grad(set_to_none=True)
            sim.backward(retain_graph=False)
            
            # Get gradients w.r.t. input image
            input_grads = image_tensor.grad

            # Generate heat-map from input gradients
            heatmap = self._build_input_heatmap(input_grads, candidate_pil.size)

            # Token scores not implemented for input gradient method
            token_scores = None

        finally:
            # Restore original training state
            self.clip_model.train(was_training)
            if self.phi and phi_was_training is not None:
                self.phi.train(phi_was_training)

        # Cache result (simple FIFO eviction).
        if len(self._cache) >= self._cache_size:
            # Pop first inserted item (Python 3.7+ dict preserves order).
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = (heatmap, token_scores)
        return heatmap, token_scores

    def reference_heatmap(
        self,
        reference_pil: Image.Image,
        pseudo_tokens: torch.Tensor = None,
        prompt: str = "$",
    ) -> np.ndarray:
        """Generate heat-map that highlights regions influencing pseudo tokens.

        Uses input gradients to show which parts of the image influence the phi network output.
        """
        self.enable_hooks()

        cache_key = self._make_cache_key(reference_pil, prompt + "|REF")
        if cache_key in self._cache:
            return self._cache[cache_key][0]  # Only heatmap stored.

        if self.phi is None:
            raise ValueError("phi_network must be provided to SearleGradCAM for reference heatmaps")

        # Preprocess image and enable gradients
        image_tensor = self.preprocess(reference_pil).unsqueeze(0).to(self.device)
        image_tensor.requires_grad_(True)
        
        # Temporarily set model to train mode for gradient computation
        was_training = self.clip_model.training
        phi_was_training = self.phi.training
        self.clip_model.train()
        self.phi.train()
        
        try:
            # Forward pass: image -> features -> pseudo tokens -> norm
            ref_features = self.clip_model.encode_image(image_tensor)
            pseudo_tokens_grad = self.phi(ref_features)
            scalar_obj = pseudo_tokens_grad.norm(p=2)
            
            # Backward to get input gradients
            self.clip_model.zero_grad(set_to_none=True)
            if self.phi:
                self.phi.zero_grad(set_to_none=True)
            scalar_obj.backward(retain_graph=False)
            
            # Get gradients w.r.t. input image
            input_grads = image_tensor.grad

            # Generate heat-map from input gradients
            heatmap = self._build_input_heatmap(input_grads, reference_pil.size)

        finally:
            # Restore original training state
            self.clip_model.train(was_training)
            self.phi.train(phi_was_training)

        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = (heatmap, None)
        return heatmap

    # ------------------------------------------------------------------
    # Helpers – building heat-map & token attribution
    # ------------------------------------------------------------------

    @staticmethod
    def _build_heatmap(activations: torch.Tensor, gradients: torch.Tensor, im_size: Tuple[int, int]) -> np.ndarray:
        """Combine *gradient × activation* → resize → normalise → NumPy."""
        if activations is None or gradients is None:
            raise RuntimeError("Hooks did not capture activations / gradients – make sure hooks are enabled *before* the forward pass.")

        # Handle different tensor layouts
        if activations.dim() == 3:
            if activations.shape[0] == 1:
                # Shape: (1, seq_len, C) - standard format
                activations = activations[0, 1:, :]  # Remove batch and class token
                gradients = gradients[0, 1:, :]
            else:
                # Shape: (seq_len, batch, C) - transformer format
                activations = activations[1:, 0, :]  # Remove class token and select batch
                gradients = gradients[1:, 0, :]
        else:
            raise ValueError(f"Unexpected activation shape: {activations.shape}")

        # Element-wise product then mean over channels → (patches,)
        product = activations * gradients
        weights = product.mean(dim=1)  # (P,)

        # Re-shape back to spatial grid.  For ViT-B/32 input 224 → grid 7×7.
        num_patches = weights.shape[0]
        grid_size = int(math.sqrt(num_patches))
        
        if num_patches == 0:
            raise ValueError("No patches found - check that hooks are capturing the right layer")
        
        cam = weights.reshape(grid_size, grid_size)
        cam = F.relu(cam)  # ReLU – only positive influence.

        # Upsample to original image size.
        cam = tv_resize(
            cam.unsqueeze(0).unsqueeze(0),
            size=im_size[::-1],
            interpolation=InterpolationMode.BILINEAR,
            antialias=False,
        ).squeeze()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()

    @staticmethod
    def _build_token_saliency(tokens: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        """Compute per-token saliency as *‖g ⊙ a‖* → normalise 0-1."""
        # tokens, grads shape: (B, seq_len, d)
        token_importance = (tokens[0] * grads[0]).norm(dim=1)  # (seq_len,)
        token_importance = token_importance - token_importance.min()
        if token_importance.max() > 0:
            token_importance = token_importance / token_importance.max()
        return token_importance.cpu().numpy()

    # ------------------------------------------------------------------
    # Helpers – building heat-map from input gradients
    # ------------------------------------------------------------------

    @staticmethod
    def _build_input_heatmap(input_grads: torch.Tensor, im_size: Tuple[int, int]) -> np.ndarray:
        """Generate heat-map from input gradients."""
        if input_grads is None:
            raise RuntimeError("No input gradients available")

        # input_grads shape: (1, C, H, W)
        grads = input_grads[0]  # Remove batch dimension: (C, H, W)
        
        # 1. Channel-wise magnitude → spatial saliency map
        heatmap = torch.mean(torch.abs(grads), dim=0)  # (H, W)

        # 2. SmoothGrad style – simple spatial smoothing via average pooling
        #    This removes high-frequency noise and makes the map more
        #    interpretable without incurring the heavy cost of sampling-based
        #    SmoothGrad.
        heatmap = F.avg_pool2d(heatmap.unsqueeze(0).unsqueeze(0), kernel_size=7, stride=1, padding=3).squeeze()

        # 3. Optional gamma correction (sqrt) to increase contrast in the
        #    most relevant regions – empirically improves visual quality.
        heatmap = torch.pow(heatmap, 0.5)
        
        # Resize to original image size if needed
        if heatmap.shape != im_size[::-1]:  # im_size is (W, H), we need (H, W)
            heatmap = F.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=im_size[::-1],
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Normalize to [0, 1]
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        return heatmap.cpu().numpy()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(image: Image.Image, prompt: str) -> str:
        # Hash output bytes of image to avoid keeping path (works for tests).
        img_bytes = image.tobytes()
        h_img = hashlib.sha1(img_bytes).hexdigest()[:8]
        h_prompt = hashlib.sha1(prompt.encode()).hexdigest()[:8]
        return f"{h_img}_{h_prompt}"

    # ------------------------------------------------------------------
    # Context-manager niceties
    # ------------------------------------------------------------------

    def __enter__(self):
        self.enable_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks() 