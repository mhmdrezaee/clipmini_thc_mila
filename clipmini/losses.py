import torch
import torch.nn.functional as F

def contrastive_loss(img_z, txt_z, temperature=0.07):
    logits = img_z @ txt_z.t() / temperature
    labels = torch.arange(img_z.size(0), device=img_z.device)
    li2t = F.cross_entropy(logits, labels)
    lt2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (li2t + lt2i)
