import torch
import torch.nn.functional as F

TEMPLATES = [
    "The photo on the left is {L}, the photo on the right is {R}.",  # official
    "Left: {L}. Right: {R}.",
    "On the left we see {L}, on the right we see {R}.",
]

@torch.no_grad()
def build_caption_bank(txt_enc, class_names, device="cpu", batch_size=128,
                       templates=TEMPLATES) -> torch.Tensor:
    """Average a few equivalent phrasings for robustness; returns (10000, D) on CPU."""
    banks = []
    for tmpl in templates:
        prompts = [tmpl.format(L=class_names[i], R=class_names[j]) for i in range(100) for j in range(100)]
        E = txt_enc.encode_prompts(prompts, device=device, batch_size=batch_size)  # (10000, D)
        banks.append(F.normalize(E, dim=-1).cpu())
    bank = torch.stack(banks, dim=0).mean(dim=0)
    return F.normalize(bank, dim=-1).cpu()
