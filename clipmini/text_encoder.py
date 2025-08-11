import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Tiny frozen text encoder (default) ----
class TinyTextEncoder(nn.Module):
    def __init__(self, num_classes=100, d=256, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.class_emb = nn.Embedding(num_classes, d)
        self.pos_left  = nn.Parameter(torch.randn(d))
        self.pos_right = nn.Parameter(torch.randn(d))
        for p in self.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, left_idx, right_idx):
        e_left  = self.class_emb(left_idx)  + self.pos_left
        e_right = self.class_emb(right_idx) + self.pos_right
        return F.normalize(e_left + e_right, dim=-1)

# ---- Optional: OpenCLIP text encoder (requires open-clip-torch) ----
class OpenClipTextEncoder(nn.Module):
    def __init__(self, model="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu", class_names=None):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise RuntimeError("open-clip-torch not installed. Install with: poetry add open-clip-torch -E clip") from e
        self.open_clip = open_clip
        self.tokenizer = open_clip.get_tokenizer(model)
        self.model, _, _ = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
        for p in self.model.parameters(): p.requires_grad_(False)
        self.model.eval()
        self.class_names = class_names or [str(i) for i in range(100)]

    @torch.no_grad()
    def encode_prompts(self, prompts, device):
        toks = self.tokenizer(prompts).to(device)
        z = self.model.encode_text(toks)
        return torch.nn.functional.normalize(z, dim=-1)

    @torch.no_grad()
    def forward(self, left_idx, right_idx):
        # Build batch prompts on the fly
        left = [self.class_names[i] for i in left_idx.tolist()]
        right = [self.class_names[i] for i in right_idx.tolist()]
        prompts = [f"The photo on the left is {l}, the photo on the right is {r}." for l,r in zip(left,right)]
        return self.encode_prompts(prompts, device=left_idx.device)

    @torch.no_grad()
    def encode_all_pairs(self, device):
        # Precompute 10,000 caption embeddings for eval
        names = self.class_names
        prompts = [f"The photo on the left is {names[i]}, the photo on the right is {names[j]}."
                   for i in range(len(names)) for j in range(len(names))]
        return self.encode_prompts(prompts, device=device)  # (10000, D)
