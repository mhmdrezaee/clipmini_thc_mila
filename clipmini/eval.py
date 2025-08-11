import torch
from torch.utils.data import DataLoader
from .data import PairedCIFAR100

@torch.no_grad()
def evaluate_topk(model_img, model_txt, device, class_count=100, k_list=(1,10,100),
                  root="./data", size=1000, num_workers=4, eval_batches=None):
    # Precompute all caption embeddings (10k). If OpenCLIP encoder supports it, use its fast path.
    if hasattr(model_txt, "encode_all_pairs"):
        all_txt = model_txt.encode_all_pairs(device=device, batch_size=256, run_on="cpu")
    else:
        # Fallback: tiny encoder builds from indices
        left  = torch.arange(class_count, device=device).unsqueeze(1).repeat(1, class_count).flatten()
        right = torch.arange(class_count, device=device).unsqueeze(0).repeat(class_count, 1).flatten()
        all_txt = model_txt(left, right)

    eval_ds = PairedCIFAR100(root=root, train=True, size=size, augment=False)
    loader = DataLoader(eval_ds, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)

    correct = {k: 0 for k in k_list}
    total = 0

    for b, (imgs, cL, cR) in enumerate(loader):
        if eval_batches is not None and b >= eval_batches: break
        imgs, cL, cR = imgs.to(device), cL.to(device), cR.to(device)
        iz = model_img(imgs)             # (B, D)
        sims = iz @ all_txt.t()          # (B, 10000)
        gt = cL * class_count + cR

        for k in k_list:
            topk = sims.topk(k, dim=1).indices
            correct[k] += (topk == gt.unsqueeze(1)).any(dim=1).sum().item()
        total += imgs.size(0)

    return {f"top-{k}": correct[k] / total for k in k_list}
