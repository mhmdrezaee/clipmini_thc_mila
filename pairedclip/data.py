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
      paired: (3,32,64) two CIFAR-100 images concatenated along width
      cL, cR: left/right fine-label indices
    If different_superclass=True, enforce coarse_left != coarse_right (harder negatives).
    """
    def __init__(self, root="./data", train=True, size=20000, different_superclass: bool=False):
        # ask for both fine & coarse labels
        self.base = datasets.CIFAR100(
            root=root, train=train, download=True, transform=base_transform(),
            target_type=("fine", "coarse"),
        )
        self.size = size
        self.class_names = self.base.classes
        self.different_superclass = different_superclass

    def __len__(self): return self.size

    def __getitem__(self, idx):
        # sample left
        i1 = random.randrange(len(self.base))
        img1, (fine1, coarse1) = self.base[i1]

        # sample right with optional constraint
        tries = 0
        while True:
            i2 = random.randrange(len(self.base))
            img2, (fine2, coarse2) = self.base[i2]
            if not self.different_superclass or coarse1 != coarse2:
                break
            tries += 1
            if tries > 20:  # escape hatch
                break

        paired = torch.cat([img1, img2], dim=2)  # (3,32,64)
        return paired, fine1, fine2
