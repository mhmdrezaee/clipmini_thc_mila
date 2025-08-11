import torch
from torch.utils.data import DataLoader
from .data import PairedCIFAR100

@torch.no_grad()
def evaluate_topk(model_img, model_txt, device, class_count=100, k_list=(1,10,100),
                  root="./data", size=1000, num_workers=4, eval_batches=None):
    device = device

    # 10k text embeddings precomputed on CPU, moved in chunks
    if hasattr(model_txt, "encode_all_pairs"):
        all_txt_cpu = model_txt.encode_all_pairs(target_device="cpu", batch_size=256)  # stays on CPU
    else:
        left = torch.arange(class_count).unsqueeze(1).repeat(1, class_count).flatten()
        right = torch.arange(class_count).unsqueeze(0).repeat(class_count, 1).flatten()
        all_txt_cpu = model_txt(left, right).cpu()

    eval_ds = PairedCIFAR100(...)
    loader = DataLoader(...)

    correct = {k: 0 for k in k_list}
    total = 0

    for b, (imgs, cL, cR) in enumerate(loader):
        if eval_batches is not None and b >= eval_batches: break

        imgs, cL, cR = imgs.to(device), cL.to(device), cR.to(device)
        iz = model_img(imgs)  # (B, D) on GPU

        # compute iz @ all_txt^T in chunks to avoid GPU blowup
        sims_chunks = []
        chunk = 2000  # 10000/2000 = 5 chunks
        for j in range(0, all_txt_cpu.size(0), chunk):
            txt_chunk = all_txt_cpu[j:j + chunk].to(device, non_blocking=True)  # (C, D)
            sims_chunks.append(iz @ txt_chunk.t())  # (B, C)
        sims = torch.cat(sims_chunks, dim=1)  # (B, 10000)

        gt = cL * class_count + cR
        for k in k_list:
            topk = sims.topk(k, dim=1).indices
            correct[k] += (topk == gt.unsqueeze(1)).any(dim=1).sum().item()

        total += imgs.size(0)

    return {f"top-{k}": correct[k] / total for k in k_list}
