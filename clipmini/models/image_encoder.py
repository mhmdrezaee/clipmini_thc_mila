import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyImageEncoder(nn.Module):
    """
    Small CNN that embeds a paired CIFAR image of shape (B, 3, 32, 64)
    into a normalized vector of size d.
    """

    def __init__(self, d: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (3,32,64) -> (64,16,32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (128,8,16)

            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (128,1,1)
        )
        # Slightly stronger projection head
        self.proj = nn.Sequential(
            nn.Linear(128, d), nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)  # (B, 128)
        z = self.proj(h)  # (B, d)
        return F.normalize(z, dim=-1)  # cosine space


# ---- Tiny ViT-ish encoder for (3,32,64) ----

class PatchEmbed(nn.Module):
    def __init__(self, img_h=32, img_w=64, patch=4, in_ch=3, dim=256):
        """
        Converts image to a sequence of patches.
        (B, 3, 32, 64) -> (B, N, dim), where N=(32/patch)*(64/patch)
        """
        super().__init__()
        assert img_h % patch == 0 and img_w % patch == 0
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.num_patches = (img_h // patch) * (img_w // patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, dim, H/patch, W/patch)
        return x.flatten(2).transpose(1, 2)  # (B, N, dim)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MicroViT(nn.Module):
    """
    Tiny ViT-style encoder trained from scratch.
    Outputs a normalized embedding vector of size d.
    """

    def __init__(
            self,
            d: int = 256,
            dim: int = 256,
            depth: int = 6,
            heads: int = 4,
            patch: int = 4,
            img_h: int = 32,
            img_w: int = 64,
    ):
        super().__init__()
        self.patch = PatchEmbed(img_h, img_w, patch, 3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.patch.num_patches, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch(x)  # (B, N, dim)
        cls = self.cls.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls, x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, 0]  # take CLS token
        z = self.proj(x)
        return F.normalize(z, dim=-1)
