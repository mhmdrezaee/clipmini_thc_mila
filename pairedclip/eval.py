import torch
from torch.utils.data import DataLoader
from .data import PairedCIFAR100

CAPTION_TMPL = "The photo on the left is {L}, the photo on the right is {R}."

@torch.no_grad()
def evaluate_topk(model_img, clip_text_enc, device, class_names,
                  class_count=100, k_list=(1,10,100), root="./data",
                  size=2000, num_workers=4, eval_batches=None, batch_prompts=256):
    # Build all 10k prompts
    prompts = [
        CAPTION_TMPL.format(L=class_names[i], R=class_names[j])
        for i in range(class_count) for j in range(class_count)
    ]
    # Encode text (batched)
    all_txt = clip_text_enc.encode_prompts(prompts, device=device, batch_size=batch_prompts)  # (10000, D)

    # Eval dataset (train split, as per brief)
    eval_ds = PairedCIFAR100(root=root, train=True, size=size)
    loader = DataLoader(eval_ds, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)

    correct = {k: 0 for k in k_list}
    total = 0
    for b, (imgs, cL, cR) in enumerate(loader):
        if eval_batches is not None and b >= eval_batches: break
        imgs, cL, cR = imgs.to(device), cL.to(device), cR.to(device)

        iz = model_img(imgs)                  # (B, D)
        sims = iz @ all_txt.t()               # (B, 10000)

        gt = cL * class_count + cR
        for k in k_list:
            topk = sims.topk(k, dim=1).indices
            correct[k] += (topk == gt.unsqueeze(1)).any(dim=1).sum().item()
        total += imgs.size(0)

    return {f"top-{k}": correct[k] / total for k in k_list}
