import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

MEAN = (0.5071, 0.4866, 0.4409)
STD  = (0.2673, 0.2564, 0.2762)

def base_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class PairedCIFAR100(Dataset):
    """
    Returns:
      paired: (3,32,64) = two CIFAR-100 images concatenated along width
      cL, cR: left/right fine-label indices

    If different_superclass=True and coarse labels are available, enforce coarse_left != coarse_right.
    If coarse labels aren't available in your torchvision, we fall back to fine_left != fine_right.
    """
    def __init__(self, root="./data", train=True, size=20000, different_superclass: bool=False):
        tfm = base_transform()
        self.size = size
        self.different_superclass = different_superclass

        self.has_coarse = False
        # Try the modern API first
        try:
            self.base = datasets.CIFAR100(
                root=root, train=train, download=True, transform=tfm,
                target_type=("fine", "coarse"),
            )
            self.has_coarse = True
        except TypeError:
            # Fallback: older torchvision (no target_type)
            self.base = datasets.CIFAR100(root=root, train=train, download=True, transform=tfm)
            # Some builds still expose coarse_targets as a list attribute
            if hasattr(self.base, "coarse_targets"):
                self.coarse_targets = self.base.coarse_targets
                self.has_coarse = True

        self.class_names = getattr(self.base, "classes", [str(i) for i in range(100)])

    def __len__(self):
        return self.size

    def _unpack(self, sample, index):
        """Return (img, fine, coarse_or_minus1) for a sample that may or may not include coarse labels."""
        if isinstance(sample[1], tuple):           # (fine, coarse) case
            img, (fine, coarse) = sample
        else:                                      # fine-only case
            img, fine = sample
            coarse = self.coarse_targets[index] if self.has_coarse and hasattr(self, "coarse_targets") else -1
        return img, fine, coarse

    def __getitem__(self, idx):
        # Left sample
        i1 = random.randrange(len(self.base))
        s1 = self.base[i1]
        img1, fine1, coarse1 = self._unpack(s1, i1)

        # Right sample, with optional superclass constraint
        tries = 0
        while True:
            i2 = random.randrange(len(self.base))
            s2 = self.base[i2]
            img2, fine2, coarse2 = self._unpack(s2, i2)

            if not self.different_superclass:
                break

            # Prefer different coarse superclasses when available;
            # otherwise fall back to different fine classes.
            if (self.has_coarse and coarse1 != coarse2) or (not self.has_coarse and fine1 != fine2):
                break

            tries += 1
            if tries > 20:   # escape hatch to avoid rare infinite loops
                break

        paired = torch.cat([img1, img2], dim=2)  # (3,32,64)
        return paired, fine1, fine2
