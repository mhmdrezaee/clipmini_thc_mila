import numpy as np
import matplotlib.pyplot as plt
from .data import MEAN, STD

mean = np.array(MEAN); std = np.array(STD)

def denorm(img_t):
    x = img_t.detach().cpu().permute(1,2,0).numpy()
    x = (x * std + mean).clip(0, 1)
    return (x * 255).astype(np.uint8)

def show_samples(ds, n=8, cols=4):
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(4*cols, 3*rows))
    for i in range(n):
        img, cL, cR = ds[i]
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(denorm(img))
        ax.set_title(f"left: {ds.class_names[cL]} | right: {ds.class_names[cR]}", fontsize=10)
        ax.axis('off')
    plt.tight_layout(); plt.show()
