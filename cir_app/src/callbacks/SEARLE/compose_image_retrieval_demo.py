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

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


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
                 exp_name: Optional[str] = None, phi_checkpoint_name: Optional[str] = None,
                 features_path: Optional[str] = None, load_features: bool = True):
        """
        Initialize the CIR system.
        
        Args:
            dataset_path: Path to the dataset
            dataset_type: Type of dataset ('cirr', 'circo', 'fashioniq')
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

        self.features_path = features_path
        self.load_features = load_features
        
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
        
        # Create dataset based on type
        if self.dataset_type == 'cirr':
            dataset = CIRRDataset(self.dataset_path, split, 'classic', self.preprocess)
        elif self.dataset_type == 'circo':
            dataset = CIRCODataset(self.dataset_path, split, 'classic', self.preprocess)
        elif self.dataset_type == 'fashioniq':
            dataset = FashionIQDataset(
                self.dataset_path, split, ['dress', 'toptee', 'shirt'], 'classic', self.preprocess
            )
        elif self.dataset_type in ['imagenet', 'imagenet-r']:
            # Load our augmented dataset CSV with image paths
            df = pd.read_csv(self.dataset_path)
            # Define a simple dataset
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
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
        # Extract image features
        self.database_features, self.database_names = extract_image_features(dataset, self.clip_model, features_path=self.features_path, load_features=self.load_features)
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


    # def compute_token_attribution(
    #     self,
    #     reference_image,           # Tensor [1,3,H,W] | [3,H,W] | PIL.Image
    #     relative_caption: str,
    #     target_image,              # Tensor or PIL.Image
    #     normalise: bool = True
    # ) -> List[Tuple[str, float]]:
    #     """
    #     Grad×Input saliency for every prompt token + the IMAGE pseudo-token.

    #     Returns
    #     -------
    #     List[(token, importance)] sorted - highest first.
    #     """

    #     # ---------- helper -------------------------------------------------
    #     dev = next(self.clip_model.parameters()).device

    #     def _to_tensor(img):
    #         if isinstance(img, torch.Tensor):
    #             if img.dim() == 3:           # CHW ➜ 1CHW
    #                 img = img.unsqueeze(0)
    #             return img.to(dev).float()
    #         if isinstance(img, PIL.Image.Image):
    #             return self.preprocess(img).unsqueeze(0).to(dev)
    #         raise TypeError("Image must be a Tensor or PIL.Image")

    #     # ------------------------------------------------------------------
    #     ref_tensor = _to_tensor(reference_image)
    #     tgt_tensor = _to_tensor(target_image)

    #     print(f"Reference image shape: {ref_tensor.shape}, Target image shape: {tgt_tensor.shape}")

    #     # 1) reference-image feature ➜ pseudo-token
    #     with torch.no_grad():
    #         ref_feat = self.clip_model.encode_image(ref_tensor)              # [1,D]
    #         pseudo   = self.phi(ref_feat).squeeze(0)                         # [d_token]

    #     print(f"Pseudo-token shape: {pseudo.shape}")

    #     # 2) prompt with placeholder + tokenise
    #     prompt  = f"a photo of $ that {relative_caption}"
    #     tok_ids = clip.tokenize([prompt], context_length=77).to(dev)         # [1,77]
        
    #     print(f"Tokenized prompt shape: {tok_ids.shape}, Prompt: {prompt}")

    #     placeholder_id = clip.tokenize(["$"])[0][1].item()
    #     mask = (tok_ids == placeholder_id)                                   # [1,77] bool
        
    #     print(f"Placeholder token ID: {placeholder_id}, Mask shape: {mask.shape}")

    #     # 3) embed tokens & substitute pseudo-word
    #     emb = self.clip_model.token_embedding(tok_ids).clone()               # [1,77,d]
    #     emb[mask] = pseudo
    #     emb = emb.detach().requires_grad_(True)
        
    #     print(f"Token embeddings shape: {emb.shape}")

    #     # 4) forward through text tower (replicates CLIP.encode_text)
    #     x = emb + self.clip_model.positional_embedding
    #     x = x.permute(1, 0, 2)                 # [seq, batch, d]
    #     x = self.clip_model.transformer(x)
    #     x = x.permute(1, 0, 2)                 # [batch, seq, d]
    #     x = self.clip_model.ln_final(x)

    #     eos_idx   = tok_ids.argmax(dim=-1).item()            # scalar
    #     text_feat = x[0, eos_idx, :] @ self.clip_model.text_projection
    #     text_feat = F.normalize(text_feat, dim=-1)            # [d]
        
    #     print(f"Text feature shape: {text_feat.shape}, EOS index: {eos_idx}")

    #     # 5) target-image feature
    #     with torch.no_grad():
    #         tgt_feat = F.normalize(
    #             self.clip_model.encode_image(tgt_tensor).squeeze(0), dim=-1
    #         )                                                # [d]

    #     print(f"Target image feature shape: {tgt_feat.shape}")

    #     # 6) similarity & backward
    #     sim = (text_feat * tgt_feat).sum()
    #     self.clip_model.zero_grad()
    #     self.phi.zero_grad(set_to_none=True)
    #     sim.backward()
        
    #     print(f"Similarity score: {sim.item()}")

    #     # 7) Grad×Input saliency
    #     # grad = emb.grad.squeeze(0)                           # [77,d]
    #     # sal  = (grad * emb.detach().squeeze(0)).sum(dim=-1).abs()   # [77]  <-- flat!

    #     grad = emb.grad.squeeze(0) 
    #     # sal = (grad * emb.detach().squeeze(0)).norm(dim=-1)      # L2-norm
    #     sal = (grad * emb.detach().squeeze(0)).abs().sum(dim=-1)
    #     sal = sal - sal.min()
    #     if sal.max() > 0:
    #         sal = sal / sal.max()
        
    #     print(f"Saliency scores shape: {sal.shape}")

    #     # 8) readable tokens
    #     from clip.simple_tokenizer import SimpleTokenizer
    #     dec  = SimpleTokenizer()
    #     toks_scores: List[Tuple[str, float]] = []

    #     tok_ids_np = tok_ids.cpu().numpy()[0]
    #     for i, tid in enumerate(tok_ids_np):
    #         if tid in [0, 49406, 49407]:  # PAD, SOS, EOS
    #             continue

    #         score = float(sal[i])                     # sal[i] is 0-D – cast to Python
    #         is_image_token = mask[0, i].item() != 0   # *** key change ***

    #         token = "IMAGE" if is_image_token else dec.decode([int(tid)]).strip()
    #         toks_scores.append((token, score))
            
    #     print(f"Number of tokens with scores: {len(toks_scores)}")

    #     # # 9) optional L1 normalisation
    #     # if normalise:
    #     #     tot = sum(v for _, v in toks_scores) + 1e-8
    #     #     toks_scores = [(t, v / tot) for t, v in toks_scores]

    #     toks_scores.sort(key=lambda x: x[1], reverse=True)
    #     print("Token attribution computed successfully.")
    #     return toks_scores

    def compute_token_attribution(
        self,
        reference_image,           # Tensor [1,3,H,W] | [3,H,W] | PIL.Image
        relative_caption: str,
        target_image,              # Tensor or PIL.Image
        normalise: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Return a list of (token, importance) pairs showing how much every
        caption   **and**   pseudo-image token contributes to the similarity
        between the composed query and *target_image*.

        Notes
        -----
        • Works for *searle*, *searle-xl* and *phi* evaluation types  
        • Importance is the ‖g ⊙ a‖ L2-norm of gradient × activation  
        • Scores are normalised to [0, 1] when *normalise* is True
        """
        if self.eval_type not in {"searle", "searle-xl", "phi"}:
            raise NotImplementedError(
                "Token attribution currently only supported for "
                "eval_type 'searle', 'searle-xl' and 'phi'"
            )

        device_ = next(self.clip_model.parameters()).device
        vocab     = clip.simple_tokenizer.SimpleTokenizer()

        print(f"Computing token attribution for '{relative_caption}' ")

        # ------------------------------------------------------------
        # 1. Pre-process inputs
        # ------------------------------------------------------------
        def _prep_image(img):
            if isinstance(img, PIL.Image.Image):
                return self.preprocess(img).unsqueeze(0).to(device_)
            if torch.is_tensor(img):
                if img.ndim == 3:         # [C,H,W]  -> add batch
                    img = img.unsqueeze(0)
                return img.to(device_, non_blocking=True)
            raise TypeError("reference_image / target_image must be "
                            "PIL.Image or torch.Tensor")

        ref_tensor    = _prep_image(reference_image)
        target_tensor = _prep_image(target_image)

        print(f"Reference image shape: {ref_tensor.shape}, Target image shape: {target_tensor.shape}")

        # Extract reference features → Φ → pseudo tokens
        with torch.no_grad():
            ref_feats     = self.clip_model.encode_image(ref_tensor)
            pseudo_tokens = self.phi(ref_feats)                 # (1, P, d)

        print(f"Pseudo tokens shape: {pseudo_tokens.shape}")

        # ------------------------------------------------------------
        # 2. Build caption with placeholder and tokenise
        # ------------------------------------------------------------
        input_caption      = f"a photo of $ that {relative_caption.strip()}"
        tokenised_caption  = clip.tokenize([input_caption], context_length=77).to(device_)
        tokens_ids         = tokenised_caption[0].tolist()

        print(f"Tokenized caption shape: {tokenised_caption.shape}, Caption: {input_caption}")

        # ------------------------------------------------------------
        # 3. Hooks to capture embeddings & gradients
        # ------------------------------------------------------------
        hook_data = dict(emb=None, grad=None)

        # -- ❶ temporarily switch gradients **on** for the lookup table ----
        tok_emb_weight = self.clip_model.token_embedding.weight
        prev_req_grad  = tok_emb_weight.requires_grad         # remember old state
        tok_emb_weight.requires_grad_(True)                   # ✨ ENABLED ✨

        def _fwd_hook(_m, _inp, out):
            out.retain_grad()                 # now allowed ✔
            hook_data["emb"] = out            # keep reference (no .detach() !)

        def _bwd_hook(_m, _gin, gout):
            hook_data["grad"] = gout[0]

        h_fwd = self.clip_model.token_embedding.register_forward_hook(_fwd_hook)
        h_bwd = self.clip_model.token_embedding.register_backward_hook(_bwd_hook)

        print(f"Registered hooks for token embedding. ")

        # ------------------------------------------------------------
        # 4. Forward pass – encode text with pseudo tokens
        # ------------------------------------------------------------
        # IMPORTANT:  run *with* autograd, NO torch.no_grad()
        query_features = encode_with_pseudo_tokens(
            self.clip_model, tokenised_caption, pseudo_tokens
        )                                   # (1, d)
        query_features = F.normalize(query_features, dim=-1)

        # Encode target image (no grad needed)
        with torch.no_grad():
            target_features = self.clip_model.encode_image(target_tensor)
            target_features = F.normalize(target_features, dim=-1)

        # Similarity scalar
        similarity = (query_features * target_features).sum()
        
        print(f"Query features shape: {query_features.shape}, Target features shape: {target_features.shape}")

        # ------------------------------------------------------------
        # 5. Back-prop to obtain gradients
        # ------------------------------------------------------------
        self.clip_model.zero_grad(set_to_none=True)
        if self.phi:
            self.phi.zero_grad(set_to_none=True)

        similarity.backward()

        print(f"Computed similarity: {similarity.item()}, Backpropagated gradients.")

        # ------------------------------------------------------------
        # 6. Importance for *text* tokens
        # ------------------------------------------------------------
        if hook_data["emb"] is None or hook_data["grad"] is None:
            h_fwd.remove(), h_bwd.remove()
            raise RuntimeError("Embedding hooks did not fire.  Make sure "
                               "compute_token_attribution is called before "
                               "any text forward pass.")

        emb   = hook_data["emb"]   # (1, L, d)
        grad  = hook_data["grad"]  # (1, L, d)
        token_imp = (emb * grad).norm(dim=-1)[0]     # (L,)

        print(f"Token importance computed for {len(tokens_ids)} tokens.")

        # ------------------------------------------------------------
        # 7. Importance for *pseudo* tokens (image slot '$')
        # ------------------------------------------------------------
        if pseudo_tokens.grad is not None:
            pseudo_imp = (pseudo_tokens * pseudo_tokens.grad).norm(dim=-1)[0]  # (P,)
        else:                # Should not really happen
            pseudo_imp = torch.zeros(pseudo_tokens.shape[1], device=device_)

        # Insert pseudo importance back into the right positions
        placeholder_id = vocab.encode("$")[0]
        placeholder_idxs = [i for i, t in enumerate(tokens_ids) if t == placeholder_id]

        print(f"Found {len(placeholder_idxs)} placeholders for pseudo tokens.")

        # Align counts – clip to min(P, #placeholders)
        n_insert = min(len(placeholder_idxs), pseudo_imp.shape[0])
        for k in range(n_insert):
            token_imp[placeholder_idxs[k]] = pseudo_imp[k]

        print(f"Inserted pseudo token importance into token importance vector.")

        # ------------------------------------------------------------
        # 8. Build readable token list & optionally normalise
        # ------------------------------------------------------------
        words, scores = [], []
        for tok_id, score in zip(tokens_ids, token_imp.tolist()):
            if tok_id == 0:                     # padding
                continue
            word = vocab.decode([tok_id])
            words.append(word)
            scores.append(score)

        print(f"Extracted {len(words)} tokens with scores.")

        # Optional min-max normalisation → [0,1]
        if normalise and scores:
            s = torch.tensor(scores)
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            scores = s.tolist()

        # Clean-up hooks
        h_fwd.remove(), h_bwd.remove()

        print(f"Hooks removed, token attribution computation completed.")

        # Return pairs **sorted** by descending importance (handy for UI)
        token_attr = list(zip(words, scores))
        token_attr.sort(key=lambda x: x[1], reverse=True)

        h_fwd.remove(); h_bwd.remove()
        tok_emb_weight.requires_grad_(prev_req_grad)   # ❷ RESTORE
        return token_attr


def main():
    parser = ArgumentParser(description="Composed Image Retrieval Demo using SEARLE techniques")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--dataset-type", type=str, required=True, choices=['cirr', 'circo', 'fashioniq'], 
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