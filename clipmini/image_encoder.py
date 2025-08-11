import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyImageEncoder(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # -> (64,16,32)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # -> (128,8,16)
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),     # -> (128,1,1)
        )
        self.proj = nn.Sequential(
            nn.Linear(128, d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d)
        )

    def forward(self, x):
        h = self.net(x).flatten(1)  # (B,128)
        z = self.proj(h)
        return F.normalize(z, dim=-1)

# --- Micro ViT (toy) ---
class PatchEmbed(nn.Module):
    def __init__(self, img_h=32, img_w=64, patch=4, in_ch=3, dim=256):
        super().__init__()
        assert img_h % patch == 0 and img_w % patch == 0
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.num_patches = (img_h // patch) * (img_w // patch)

    def forward(self, x):
        x = self.proj(x)                  # (B, dim, H/patch, W/patch)
        return x.flatten(2).transpose(1,2)# (B, N, dim)

class Block(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MicroViT(nn.Module):
    def __init__(self, d=256, dim=256, depth=6, heads=4, patch=4, img_h=32, img_w=64):
        super().__init__()
        self.patch = PatchEmbed(img_h, img_w, patch, 3, dim)
        self.cls = nn.Parameter(torch.zeros(1,1,dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.patch.num_patches, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, d)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)                 # (B, N, dim)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)[:,0]             # CLS
        z = self.proj(x)
        return torch.nn.functional.normalize(z, dim=-1)
