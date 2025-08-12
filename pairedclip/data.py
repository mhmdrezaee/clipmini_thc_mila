import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# CIFAR-100 stats
CIFAR_MEAN = (0.5071, 0.4866, 0.4409)
CIFAR_STD  = (0.2673, 0.2564, 0.2762)

def train_transform_by_policy(policy: str):
    # All ops are per-image (left/right independently), preserving order semantics.
    if policy == "none":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    if policy == "light_basic":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=2, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    if policy == "light_color":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=2, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.1, 0.1, 0.05, 0.0),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    if policy == "light_blur":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=2, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    if policy == "light_erase":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=2, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), value="random"),
        ])

    if policy == "light_all":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=2, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.1, 0.1, 0.05, 0.0),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), value="random"),
        ])

    # very minimal scale/translate (optional)
    if policy == "light_rrc":
        return transforms.Compose([
            transforms.RandomResizedCrop((32, 32), scale=(0.90, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    raise ValueError(f"Unknown aug_policy: {policy}")

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
                 different_superclass: bool=False, augment: bool=True, aug_policy: str='light_basic'):

        self.size = size
        self.different_superclass = different_superclass
        self.has_coarse = False
        if train:
            tfm = train_transform_by_policy(aug_policy if augment else "none")
        else:
            tfm = eval_transform
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
