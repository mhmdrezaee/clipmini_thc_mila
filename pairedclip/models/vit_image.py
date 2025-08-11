import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_h=32, img_w=64, patch=4, in_ch=3, dim=256):
        super().__init__()
        assert img_h % patch == 0 and img_w % patch == 0
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.num_patches = (img_h // patch) * (img_w // patch)

    def forward(self, x):
        x = self.proj(x)                      # (B, dim, H/patch, W/patch)
        return x.flatten(2).transpose(1, 2)   # (B, N, dim)

class Block(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTImageEncoder(nn.Module):
    """
    ViT from scratch for (3,32,64) images. Outputs a normalized embedding of size emb_dim.
    """
    def __init__(self, emb_dim=512, dim=256, depth=6, heads=4, patch=4, img_h=32, img_w=64):
        super().__init__()
        self.patch = PatchEmbed(img_h, img_w, patch, 3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.patch.num_patches, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, emb_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)                 # (B, N, dim)
        cls = self.cls.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls, x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, 0]            # CLS token
        z = self.proj(x)                  # (B, emb_dim)
        return F.normalize(z, dim=-1)
