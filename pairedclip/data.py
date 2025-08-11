import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

MEAN = (0.5071, 0.4866, 0.4409)
STD  = (0.2673, 0.2564, 0.2762)

def base_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

class PairedCIFAR100(Dataset):
    """
    Returns:
      paired: (3, 32, 64) = two CIFAR-100 images concatenated along width
      cL, cR: left/right fine-label indices
    """
    def __init__(self, root="./data", train=True, size=20000):
        self.base = datasets.CIFAR100(root=root, train=train, download=True,
                                      transform=base_transform())
        self.size = size
        self.class_names = self.base.classes

    def __len__(self): return self.size

    def __getitem__(self, idx):
        i1 = random.randrange(len(self.base))
        i2 = random.randrange(len(self.base))
        img1, c1 = self.base[i1]
        img2, c2 = self.base[i2]
        paired = torch.cat([img1, img2], dim=2)  # (3, 32, 64)
        return paired, c1, c2
