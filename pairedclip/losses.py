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

def _clip_bidir_ce(img_z, txt_z, logit_scale_param):
    scale = torch.clamp(logit_scale_param, max=math.log(100.)).exp()
    logits = scale * (img_z @ txt_z.t())          # (B,B)
    labels = torch.arange(img_z.size(0), device=img_z.device)
    li2t = F.cross_entropy(logits, labels)
    lt2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (li2t + lt2i)

def clip_loss_label_smooth(img_z, txt_z, logit_scale_param, eps=0.05):
    scale = torch.clamp(logit_scale_param, max=math.log(100.)).exp()
    logits = scale * (img_z @ txt_z.t())          # (B,B)

    B = logits.size(0)
    # soft targets (diagonal 1-eps, off-diagonal eps/(B-1))
    target = torch.full_like(logits, eps / (B - 1))
    idx = torch.arange(B, device=img_z.device)
    target[idx, idx] = 1.0 - eps

    logp = torch.log_softmax(logits, dim=1)
    li2t = -(target * logp).sum(dim=1).mean()

    logp_t = torch.log_softmax(logits.t(), dim=1)
    lt2i = -(target * logp_t).sum(dim=1).mean()
    return 0.5 * (li2t + lt2i)

def swap_margin_hinge(img_z, txt_pos, txt_swapped, margin=0.1):
    # encourage sim(pos) > sim(swapped) + margin
    s_pos = (img_z * txt_pos).sum(dim=-1)
    s_swp = (img_z * txt_swapped).sum(dim=-1)
    return F.relu(margin + s_swp - s_pos).mean()

def partial_info_nce(img_z, txt_pos, txt_negs, logit_scale_param):
    """
    img_z: (B,D)  txt_pos: (B,D)
    txt_negs: (B,M,D) — M semi-hard negatives per sample
    """
    scale = torch.clamp(logit_scale_param, max=math.log(100.)).exp()
    all_txt = torch.cat([txt_pos.unsqueeze(1), txt_negs], dim=1)   # (B,1+M,D)
    logits = scale * torch.einsum('bd,bkd->bk', img_z, all_txt)    # (B,1+M)
    labels = torch.zeros(img_z.size(0), dtype=torch.long, device=img_z.device)  # index 0 = positive
    return F.cross_entropy(logits, labels)

def combined_contrastive_loss(
    img_z, txt_pos, logit_scale_param,
    *,
    txt_swapped=None,            # (B,D) or None
    txt_negs=None,               # (B,M,D) or None
    label_smoothing_eps=0.0,
    swap_margin=0.1, swap_weight=0.5,
    partial_weight=1.0,
    reg_logit_scale_tau=None, reg_logit_scale_weight=0.0
):
    # base CLIP term
    if label_smoothing_eps and label_smoothing_eps > 0.0:
        total = clip_loss_label_smooth(img_z, txt_pos, logit_scale_param, eps=label_smoothing_eps)
    else:
        total = _clip_bidir_ce(img_z, txt_pos, logit_scale_param)

    # explicit order term
    if txt_swapped is not None and swap_weight > 0.0:
        total = total + swap_weight * swap_margin_hinge(img_z, txt_pos, txt_swapped, margin=swap_margin)

    # partial softmax with semi-hard negatives
    if txt_negs is not None and partial_weight > 0.0:
        total = total + partial_weight * partial_info_nce(img_z, txt_pos, txt_negs, logit_scale_param)

    # light regularization to keep logit_scale healthy
    if reg_logit_scale_weight > 0.0 and reg_logit_scale_tau is not None:
        center = math.log(1.0 / float(reg_logit_scale_tau))
        total = total + reg_logit_scale_weight * (logit_scale_param - center) ** 2

    return total

