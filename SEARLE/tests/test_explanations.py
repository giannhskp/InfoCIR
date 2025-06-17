import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pathlib import Path

# Ensure the src directory is on sys.path so we can import explanations when
# tests are executed from repository root.
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from explanations import SearleGradCAM  # noqa: E402

###############################################################
# Tiny *dummy* CLIP-like model – keeps tests lightweight & fast
###############################################################

class DummyResBlock(torch.nn.Module):
    def forward(self, x):
        return x * 1.0  # Identity – keeps gradient flow intact


class DummyVisual(torch.nn.Module):
    def __init__(self, output_dim: int = 8, seq_len: int = 49 + 1):
        super().__init__()
        # Emulate last residual block list expected by SearleGradCAM
        self.transformer = type("_", (), {"resblocks": [DummyResBlock()]})()
        self.output_dim = output_dim
        self.seq_len = seq_len

    def forward(self, img: torch.Tensor) -> torch.Tensor:  # noqa: D401
        B = img.shape[0]
        scalar = img.mean(dim=(1, 2, 3), keepdim=True)  # (B,1,1,1)
        # Build (B, seq_len, C) activations filled with scalar value
        x = scalar.repeat(1, self.seq_len * self.output_dim).view(B, self.seq_len, self.output_dim)
        # Pass through dummy residual block to trigger hooks
        x = self.transformer.resblocks[0](x)
        # Pool tokens → feature vector
        return x.mean(dim=1)


class DummyCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(1000, 8)
        self.visual = DummyVisual()

    # Dummy encode functions ------------------------------------------------
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.visual(img)

    def encode_text(self, tokenized: torch.Tensor) -> torch.Tensor:  # noqa: D401
        emb = self.token_embedding(tokenized)
        return emb.mean(dim=1)

    # CLIP compatibility attributes
    text_projection = torch.nn.Identity()
    dtype = torch.float32


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def pil_rand_image(size=(224, 224)) -> Image.Image:
    arr = np.random.randint(0, 255, size + (3,), dtype=np.uint8)
    return Image.fromarray(arr)


def simple_preprocess(pil_img: Image.Image) -> torch.Tensor:
    # Convert to tensor in [0,1]
    img = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
    # Clamp just in case & normalise poorly (sufficient for dummy model)
    return img


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

def test_self_match_uniform_heatmap():
    clip = DummyCLIP()
    explainer = SearleGradCAM(clip, simple_preprocess, device=torch.device("cpu"))
    img = pil_rand_image()

    # Query features from same image
    q_feat = F.normalize(clip.encode_image(simple_preprocess(img).unsqueeze(0)), dim=-1)

    heatmap, _ = explainer.candidate_heatmap(img, q_feat, prompt="self-match")

    # Uniform map → very low variance
    assert heatmap.var() < 1e-6, "Self-match should yield uniform heatmap"


def test_prompt_swap_changes_heatmap():
    clip = DummyCLIP()
    explainer = SearleGradCAM(clip, simple_preprocess, device=torch.device("cpu"))
    img = pil_rand_image()

    # Two artificial query vectors pointing in different directions
    q1 = torch.ones(1, 8)
    q2 = -q1
    q1, q2 = F.normalize(q1, dim=-1), F.normalize(q2, dim=-1)

    h1, _ = explainer.candidate_heatmap(img, q1, prompt="red")
    h2, _ = explainer.candidate_heatmap(img, q2, prompt="blue")

    diff = np.abs(h1 - h2).mean()
    assert diff > 0.0, "Heatmaps for different prompts should differ"


def test_noise_control_heatmap_flat():
    clip = DummyCLIP()
    explainer = SearleGradCAM(clip, simple_preprocess, device=torch.device("cpu"))
    noise_img = pil_rand_image()

    q_feat = torch.randn(1, 8)
    q_feat = F.normalize(q_feat, dim=-1)

    heatmap, _ = explainer.candidate_heatmap(noise_img, q_feat, prompt="noise")

    # For random noise & dummy model expect near uniform map
    assert heatmap.var() < 1e-6, "Noise control heatmap should be flat" 