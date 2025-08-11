import math
import torch
import torch.nn.functional as F

def contrastive_loss_with_logit_scale(img_z, txt_z, logit_scale_param):
    """
    Bidirectional InfoNCE with learnable logit_scale (as in CLIP).
    logit_scale_param is a nn.Parameter in log-space (initialized with log(1/τ)).
    """
    # clamp to CLIP’s typical max (~100) in linear space
    scale = torch.clamp(logit_scale_param, max=math.log(100.0)).exp()
    logits = scale * (img_z @ txt_z.t())
    labels = torch.arange(img_z.size(0), device=img_z.device)
    li2t = F.cross_entropy(logits, labels)
    lt2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (li2t + lt2i)
