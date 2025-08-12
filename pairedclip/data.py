import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# CIFAR-100 stats
MEAN = (0.5071, 0.4866, 0.4409)
STD  = (0.2673, 0.2564, 0.2762)

def train_transform(augment: bool = True):
    if not augment:
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    # Light but effective for 32×32
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),       # flips each image independently (OK)
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.1),
        # Optional extra punch (comment out if too slow/noisy):
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), value="random"),
    ])

def eval_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class PairedCIFAR100(Dataset):
    """
    Returns:
      paired: (3,32,64) = two CIFAR-100 images concatenated along width
      cL, cR: left/right fine-label indices

    If different_superclass=True and coarse labels are available, enforce coarse_left != coarse_right.
    If coarse labels aren't available, fall back to fine_left != fine_right.
    """
    def __init__(self, root="./data", train=True, size=20000,
                 different_superclass: bool=False, augment: bool=True):
        tfm = train_transform(augment) if train else eval_transform()
        self.size = size
        self.different_superclass = different_superclass
        self.has_coarse = False

        # Try modern API with target_type; fall back gracefully
        try:
            self.base = datasets.CIFAR100(
                root=root, train=train, download=True, transform=tfm,
                target_type=("fine", "coarse"),
            )
            self.has_coarse = True
        except TypeError:
            self.base = datasets.CIFAR100(root=root, train=train, download=True, transform=tfm)
            if hasattr(self.base, "coarse_targets"):
                self.coarse_targets = self.base.coarse_targets
                self.has_coarse = True

        self.class_names = getattr(self.base, "classes", [str(i) for i in range(100)])

    def __len__(self):
        return self.size

    def _unpack(self, sample, index):
        if isinstance(sample[1], tuple):           # (fine, coarse)
            img, (fine, coarse) = sample
        else:                                      # fine only
            img, fine = sample
            coarse = self.coarse_targets[index] if self.has_coarse and hasattr(self, "coarse_targets") else -1
        return img, fine, coarse

    def __getitem__(self, idx):
        # Left
        i1 = random.randrange(len(self.base))
        s1 = self.base[i1]
        img1, fine1, coarse1 = self._unpack(s1, i1)

        # Right with optional superclass constraint
        tries = 0
        while True:
            i2 = random.randrange(len(self.base))
            s2 = self.base[i2]
            img2, fine2, coarse2 = self._unpack(s2, i2)

            if not self.different_superclass:
                break
            if (self.has_coarse and coarse1 != coarse2) or (not self.has_coarse and fine1 != fine2):
                break
            tries += 1
            if tries > 20:
                break

        # IMPORTANT: transforms already applied per-image above; now just stitch
        paired = torch.cat([img1, img2], dim=2)  # (3, 32, 64)
        return paired, fine1, fine2
