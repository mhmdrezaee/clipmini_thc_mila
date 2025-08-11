import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# CIFAR-100 stats
MEAN = (0.5071, 0.4866, 0.4409)
STD  = (0.2673, 0.2564, 0.2762)

def train_transform(augment: bool = True):
    tfms = []
    if augment:
        tfms += [
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.1),
        ]
    tfms += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(tfms)

def eval_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class PairedCIFAR100(Dataset):
    """
    Returns:
      paired image: (3, 32, 64)  = concat along width of two CIFAR-100 samples
      cL, cR     : left/right class indices
    """
    def __init__(self, root="./data", train=True, size=2000, transform=None, augment=True):
        base_tfm = train_transform(augment) if train else eval_transform()
        self.base = datasets.CIFAR100(root=root, train=train, download=True, transform=transform or base_tfm)
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
