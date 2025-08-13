import math, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pairedclip.losses import combined_contrastive_loss
from pairedclip.losses import contrastive_loss_with_logit_scale
from pairedclip.data import PairedCIFAR100


def half_mixup(imgs, idx_perm, lam: float):
    # imgs: (B,3,32,64) → split along width and mix left-with-left, right-with-right
    left, right = imgs[..., :32], imgs[..., 32:]
    left_m  = lam * left  + (1.0 - lam) * left[idx_perm]
    right_m = lam * right + (1.0 - lam) * right[idx_perm]
    return torch.cat([left_m, right_m], dim=-1)

def build_classnames(data_root: str):
    # keep augment OFF for this tiny probe; policy doesn't matter here
    tmp = PairedCIFAR100(root=data_root, train=True, size=10, augment=False, aug_policy="none")
    return tmp.class_names


def build_loader(cfg, hard: bool):
    ds = PairedCIFAR100(
        root=cfg.data_root,
        train=True,
        size=20000,
        different_superclass=hard,
        augment=bool(cfg.use_augs),
        aug_policy=getattr(cfg, "aug_policy", "light_basic"),  # <<< pass it
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    return ds, loader


def steps_per_epoch_estimate(cfg):
    ds = PairedCIFAR100(
        root=cfg.data_root,
        train=True,
        size=20000,
        different_superclass=True,
        augment=bool(cfg.use_augs),
        aug_policy=getattr(cfg, "aug_policy", "light_basic"),  # <<< pass it
    )
    return (len(ds) + cfg.batch_size - 1) // cfg.batch_size


def _sample_semi_hard_indices(cL, cR, M=8):
    """
    Build M indices per sample focusing on same-left or same-right hard negatives:
    (L, r') and (l', R). Returns LongTensor (B, M).
    """
    B = cL.size(0)
    # split quota between same-left and same-right
    mL = M // 2
    mR = M - mL

    # sample right variants for same-left
    r_rand = torch.randint(0, 100, (B, mR), dtype=torch.long)
    # avoid picking the true right
    maskR = (r_rand == cR.unsqueeze(1))
    if maskR.any(): r_rand[maskR] = (r_rand[maskR] + 1) % 100
    idx_sameL = (cL.unsqueeze(1) * 100 + r_rand)   # (B, mR)

    # sample left variants for same-right
    l_rand = torch.randint(0, 100, (B, mL), dtype=torch.long)
    maskL = (l_rand == cL.unsqueeze(1))
    if maskL.any(): l_rand[maskL] = (l_rand[maskL] + 1) % 100
    idx_sameR = (l_rand * 100 + cR.unsqueeze(1))   # (B, mL)

    return torch.cat([idx_sameL, idx_sameR], dim=1)  # (B, M)

def train_one_epoch(img_enc, caption_bank_cpu, loader, device, optimizer, scaler,
                    logit_scale, sched, cfg, logger=None, writer=None, epoch: int = 1):
    img_enc.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)

    import time
    t0 = time.time()
    for it, (imgs, cL, cR) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)

        # ---- Optional Half-Mixup (from a later epoch) ----
        use_mix = bool(cfg.mixup) and (epoch >= int(cfg.mixup_start_epoch))
        if use_mix:
            B = imgs.size(0)
            idx_perm = torch.randperm(B, device=device)
            # draw a single lambda per batch (simple and effective)
            lam = float(torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample())
            imgs = half_mixup(imgs, idx_perm, lam)

            # mix the text targets the same way
            idx_a = (cL.cpu() * 100 + cR.cpu())                      # (B,)
            idx_b = (cL[idx_perm.cpu()].cpu() * 100 + cR[idx_perm.cpu()].cpu())
            tz_pos = lam * caption_bank_cpu[idx_a] + (1.0 - lam) * caption_bank_cpu[idx_b]
            tz_pos = F.normalize(tz_pos, dim=-1).to(device, non_blocking=True)

            # with mixup active, keep loss simple (no swap/partials this batch)
            tz_swp = None
            tz_negs = None
        else:
            # plain (non-mixed) targets from the caption bank
            idx_pos = (cL.cpu() * 100 + cR.cpu())
            tz_pos  = caption_bank_cpu[idx_pos].to(device, non_blocking=True)

            tz_swp = None
            if getattr(cfg, "use_swap_margin", False):
                idx_swp = (cR.cpu() * 100 + cL.cpu())
                tz_swp  = caption_bank_cpu[idx_swp].to(device, non_blocking=True)

            tz_negs = None
            if getattr(cfg, "use_partial_softmax", False) and getattr(cfg, "partial_m", 0) > 0:
                M = int(cfg.partial_m)
                # sample semi-hard negatives: (L, r') and (l', R)
                B = cL.size(0)
                mL = M // 2
                mR = M - mL
                r_rand = torch.randint(0, 100, (B, mR), dtype=torch.long)
                maskR = (r_rand == cR.unsqueeze(1))
                if maskR.any(): r_rand[maskR] = (r_rand[maskR] + 1) % 100
                idx_sameL = (cL.unsqueeze(1) * 100 + r_rand)

                l_rand = torch.randint(0, 100, (B, mL), dtype=torch.long)
                maskL = (l_rand == cL.unsqueeze(1))
                if maskL.any(): l_rand[maskL] = (l_rand[maskL] + 1) % 100
                idx_sameR = (l_rand * 100 + cR.unsqueeze(1))

                idx_negs = torch.cat([idx_sameL, idx_sameR], dim=1)  # (B,M)
                tz_negs = caption_bank_cpu[idx_negs.reshape(-1)].to(device, non_blocking=True)
                tz_negs = tz_negs.view(B, M, -1)                     # (B,M,D)

        # ---- Forward + loss ----
        from pairedclip.losses import combined_contrastive_loss
        with torch.cuda.amp.autocast(enabled=cfg.amp and torch.cuda.is_available()):
            iz = img_enc(imgs)  # (B,D), already normalized by the head
            loss = combined_contrastive_loss(
                iz, tz_pos, logit_scale,
                txt_swapped=tz_swp,
                txt_negs=tz_negs,
                label_smoothing_eps=getattr(cfg, "label_smoothing_eps", 0.0),
                swap_margin=getattr(cfg, "swap_margin", 0.1),
                swap_weight=getattr(cfg, "swap_weight", 0.5),
                partial_weight=getattr(cfg, "partial_weight", 1.0),
                reg_logit_scale_tau=getattr(cfg, "reg_logit_scale_tau", None),
                reg_logit_scale_weight=getattr(cfg, "reg_logit_scale_weight", 0.0)
            )

        # ---- Optim step with grad accumulation ----
        scaler.scale(loss / cfg.accum_steps).backward()
        if it % cfg.accum_steps == 0:
            if cfg.amp and torch.cuda.is_available():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(img_enc.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            sched.step()

        running += loss.item()

        # heartbeat (unchanged)
        if logger and (it % 100 == 0 or it == 1):
            ls_val = float(logit_scale.detach().exp().clamp(max=100))
            last_lr = sched.get_last_lr()[0]
            msg = f"ep {epoch} it {it}/{len(loader)} | loss {loss.item():.4f} | lr {last_lr:.3e} | logit_scale {ls_val:.2f}"
            if device == "cuda":
                msg += f" | cuda_mem {torch.cuda.memory_allocated()/1024**2:.0f}MB"
            logger.info(msg)
            if writer:
                step = (epoch-1)*len(loader) + it
                writer.add_scalar("train/loss_step", float(loss.item()), step)
                writer.add_scalar("train/lr", last_lr, step)
                writer.add_scalar("train/logit_scale", ls_val, step)

    avg = running / len(loader)
    secs = time.time() - t0
    return avg, secs


@torch.no_grad()
def evaluate(img_enc, txt_enc, device, class_names, cfg):
    # Use the provided evaluator (unchanged API)
    from pairedclip.eval import evaluate_topk
    return evaluate_topk(img_enc, txt_enc, device, class_names,
                         root=cfg.data_root, size=cfg.eval_size, batch_prompts=128)

def save_checkpoint(run_dir, epoch, img_enc, logit_scale, cfg, metrics, logger=None):
    import torch, json
    ckpt = {
        "img_enc": img_enc.state_dict(),
        "logit_scale": logit_scale.detach().cpu(),
        "config": cfg.__dict__,
        "epoch": epoch,
    }
    path = (run_dir / f"epoch_{epoch:03d}.pt")
    torch.save(ckpt, path)
    if logger: logger.info(f"Saved checkpoint → {path}")

    # track best by top-100
    best_path = run_dir / "best.pt"
    best_score_path = run_dir / "best_score.txt"
    cur = metrics["top-100"]
    prev = float(best_score_path.read_text().strip()) if best_score_path.exists() else -1.0
    if cur > prev:
        torch.save(ckpt, best_path)
        best_score_path.write_text(str(cur))
        if logger: logger.info(f"New best (top-100={cur:.4f}) → {best_path}")
