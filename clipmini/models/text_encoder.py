import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyTextEncoder(nn.Module):
    """
    Frozen stand-in for a CLIP text encoder.
    - One embedding per CIFAR-100 class.
    - Two learned position vectors (left/right).
    - Outputs a normalized caption embedding.
    """
    def __init__(self, num_classes: int = 100, d: int = 256, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)  # deterministic init
        self.class_emb = nn.Embedding(num_classes, d)
        self.pos_left  = nn.Parameter(torch.randn(d))
        self.pos_right = nn.Parameter(torch.randn(d))
        # Freeze all params
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, left_idx: torch.Tensor, right_idx: torch.Tensor) -> torch.Tensor:
        e_left  = self.class_emb(left_idx)  + self.pos_left
        e_right = self.class_emb(right_idx) + self.pos_right
        return F.normalize(e_left + e_right, dim=-1)


# Optional: OpenCLIP-backed text encoder (frozen)
class OpenClipTextEncoder(nn.Module):
    """
    Uses open-clip-torch to encode the exact caption template:
    "The photo on the left is {L}, the photo on the right is {R}."
    All weights are frozen.
    """
    def __init__(
        self,
        model: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
        class_names = None,
    ):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise RuntimeError(
                "open-clip-torch is not installed. "
                "Install with: poetry add open-clip-torch -E clip"
            ) from e

        self._oc = open_clip
        self.tokenizer = open_clip.get_tokenizer(model)
        self.model, _, _ = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=device
        )
        # freeze
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.class_names = class_names or [str(i) for i in range(100)]

    @torch.no_grad()
    def _encode_prompts(self, prompts, device):
        toks = self.tokenizer(prompts).to(device)
        z = self.model.encode_text(toks)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def forward(self, left_idx: torch.Tensor, right_idx: torch.Tensor) -> torch.Tensor:
        L = [self.class_names[i] for i in left_idx.tolist()]
        R = [self.class_names[i] for i in right_idx.tolist()]
        prompts = [f"The photo on the left is {l}, the photo on the right is {r}."
                   for l, r in zip(L, R)]
        return self._encode_prompts(prompts, device=left_idx.device)

    @torch.no_grad()
    def encode_all_pairs(self, device: str):
        """
        Convenience for evaluation: returns all 10,000 caption embeddings (100x100).
        """
        names = self.class_names
        prompts = [
            f"The photo on the left is {names[i]}, the photo on the right is {names[j]}."
            for i in range(len(names)) for j in range(len(names))
        ]
        return self._encode_prompts(prompts, device=device)  # (10000, D)
