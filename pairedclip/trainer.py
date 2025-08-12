import math, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pairedclip.losses import contrastive_loss_with_logit_scale
from pairedclip.data import PairedCIFAR100

def build_classnames(data_root: str):
    tmp = PairedCIFAR100(root=data_root, train=True, size=10, augment=False)
    return tmp.class_names

def build_loader(cfg, hard: bool):
    ds = PairedCIFAR100(root=cfg.data_root, train=True, size=20000,
                        different_superclass=hard, augment=bool(cfg.use_augs))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    return ds, loader

def steps_per_epoch_estimate(cfg):
    ds = PairedCIFAR100(root=cfg.data_root, train=True, size=20000,
                        different_superclass=True, augment=bool(cfg.use_augs))
    return (len(ds) + cfg.batch_size - 1) // cfg.batch_size

def train_one_epoch(img_enc, caption_bank_cpu, loader, device, optimizer, scaler,
                    logit_scale, sched, cfg, logger=None, writer=None, epoch: int = 1):
    img_enc.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    for it, (imgs, cL, cR) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)

        # map (left,right) -> flat caption index
        idx = (cL.cpu() * 100 + cR.cpu())
        tz = caption_bank_cpu[idx].to(device, non_blocking=True)  # (B, D)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            iz = img_enc(imgs)
            loss = contrastive_loss_with_logit_scale(iz, tz, logit_scale)

        scaler.scale(loss / cfg.accum_steps).backward()

        if it % cfg.accum_steps == 0:
            if cfg.amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(img_enc.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            sched.step()

        running += loss.item()

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
