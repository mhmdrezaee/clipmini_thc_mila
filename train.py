import argparse, math, time, json, csv, sys, os, random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from pairedclip.config import TrainConfig
from pairedclip.data import PairedCIFAR100  # uses augment=... (see note below)
from pairedclip.models import ViTImageEncoder, CLIPTextEncoderHF
from pairedclip.losses import contrastive_loss_with_logit_scale

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===== Prompt ensembling (training-time only; eval still matches official template) =====
TEMPLATES = [
    "The photo on the left is {L}, the photo on the right is {R}.",   # official
    "Left: {L}. Right: {R}.",
    "On the left we see {L}, on the right we see {R}.",
]

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def build_caption_bank(txt_enc, class_names, device="cpu", batch_size=128, templates=TEMPLATES):
    """Average a few equivalent phrasings for robustness; re-normalize."""
    banks = []
    for tmpl in templates:
        prompts = [tmpl.format(L=class_names[i], R=class_names[j]) for i in range(100) for j in range(100)]
        E = txt_enc.encode_prompts(prompts, device=device, batch_size=batch_size)  # (10000, D) on device
        E = F.normalize(E, dim=-1).cpu()
        banks.append(E)
    bank = torch.stack(banks, dim=0).mean(dim=0)
    return F.normalize(bank, dim=-1).cpu()   # (10000, D), unit-norm

# ===== Half-Mixup (keeps left/right semantics) =====
def half_mixup(imgs, idx_perm, lam):
    # imgs: (B,3,32,64) → split halves along width
    left, right = imgs[..., :32], imgs[..., 32:]
    left_m = lam * left + (1.0 - lam) * left[idx_perm]
    right_m = lam * right + (1.0 - lam) * right[idx_perm]
    return torch.cat([left_m, right_m], dim=-1)

def main():
    p = argparse.ArgumentParser()
    # basics
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./runs/exp_aug")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_size", type=int, default=None)
    # model/optim
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--emb_dim", type=int, default=None)
    p.add_argument("--vit_dim", type=int, default=None)
    p.add_argument("--vit_depth", type=int, default=None)
    p.add_argument("--vit_heads", type=int, default=None)
    p.add_argument("--vit_patch", type=int, default=None)
    # training tricks
    p.add_argument("--amp", action="store_true")
    p.add_argument("--accum_steps", type=int, default=None)
    p.add_argument("--min_lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--curriculum_epochs", type=int, default=None)
    # augs / mixup
    p.add_argument("--use_augs", type=int, default=1)
    p.add_argument("--mixup", type=int, default=1)
    p.add_argument("--mixup_alpha", type=float, default=0.4)
    p.add_argument("--mixup_start_epoch", type=int, default=2)
    args = p.parse_args()

    cfg = TrainConfig()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # — outputs —
    run_dir = Path(cfg.output_dir); run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f: json.dump(cfg.__dict__, f, indent=2)

    # — data for class names —
    cold_ds = PairedCIFAR100(root=cfg.data_root, train=True, size=10, augment=False)
    class_names = cold_ds.class_names

    # — models —
    txt_enc = CLIPTextEncoderHF("openai/clip-vit-base-patch32")  # frozen
    img_enc = ViTImageEncoder(emb_dim=cfg.emb_dim, dim=cfg.vit_dim,
                              depth=cfg.vit_depth, heads=cfg.vit_heads,
                              patch=cfg.vit_patch).to(device)

    # — caption bank (prompt ensemble) —
    caption_bank = build_caption_bank(txt_enc, class_names, device="cpu", batch_size=128, templates=TEMPLATES)

    # — learnable logit scale —
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1.0 / cfg.temperature), device=device))

    opt = torch.optim.AdamW([
        {"params": img_enc.parameters(), "weight_decay": cfg.weight_decay, "lr": cfg.lr},
        {"params": [logit_scale],        "weight_decay": 0.0,            "lr": cfg.lr},
    ])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # — scheduler warmup → cosine —
    # estimate steps/epoch from a hard curriculum pass
    tmp_ds = PairedCIFAR100(root=cfg.data_root, train=True, size=20000, different_superclass=True, augment=bool(cfg.use_augs))
    steps_per_epoch = (len(tmp_ds) + cfg.batch_size - 1) // cfg.batch_size
    total_updates = cfg.epochs * max(1, steps_per_epoch // max(1, cfg.accum_steps))

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        prog = (step - cfg.warmup_steps) / max(1, total_updates - cfg.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * prog))
        return (cfg.min_lr / cfg.lr) + (1 - (cfg.min_lr / cfg.lr)) * cosine

    sched = LambdaLR(opt, lr_lambda)
    global_step = 0

    # — training loop —
    for epoch in range(1, cfg.epochs + 1):
        hard = (epoch <= cfg.curriculum_epochs)
        print(f"Starting epoch {epoch} (curriculum: different_superclass={hard}, augs={bool(cfg.use_augs)})", flush=True)

        ds = PairedCIFAR100(root=cfg.data_root, train=True, size=20000,
                            different_superclass=hard, augment=bool(cfg.use_augs))
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

        img_enc.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        t0 = time.time()

        for it, (imgs, cL, cR) in enumerate(loader, start=1):
            imgs = imgs.to(device, non_blocking=True)

            # Optional Half-Mixup (from a later epoch so training is stable)
            if cfg.mixup and epoch >= cfg.mixup_start_epoch:
                B = imgs.size(0)
                idx_perm = torch.randperm(B, device=device)
                lam = float(torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample())
                imgs = half_mixup(imgs, idx_perm, lam)

                # mix text targets the same way (on CPU then move)
                idx_a = (cL.cpu() * 100 + cR.cpu())
                idx_b = (cL[idx_perm.cpu()].cpu() * 100 + cR[idx_perm.cpu()].cpu())
                tz = lam * caption_bank[idx_a] + (1.0 - lam) * caption_bank[idx_b]
                tz = F.normalize(tz, dim=-1).to(device, non_blocking=True)
            else:
                # plain targets
                gt_idx = (cL.cpu() * 100 + cR.cpu())
                tz = caption_bank[gt_idx].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                iz = img_enc(imgs)  # (B, D)
                loss = contrastive_loss_with_logit_scale(iz, tz, logit_scale)

            scaler.scale(loss / cfg.accum_steps).backward()

            if it % cfg.accum_steps == 0:
                if cfg.amp:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(img_enc.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step(); global_step += 1

            running += loss.item()

            # heartbeat every ~100 iters
            if it % 100 == 0 or it == 1:
                ls_val = float(logit_scale.detach().exp().clamp(max=100))
                last_lr = sched.get_last_lr()[0]
                msg = f"ep {epoch} it {it}/{len(loader)} | loss {loss.item():.4f} | lr {last_lr:.3e} | logit_scale {ls_val:.2f}"
                if device == "cuda":
                    msg += f" | cuda_mem {torch.cuda.memory_allocated()/1024**2:.0f}MB"
                print(msg, flush=True)

        # EVAL (reuse evaluator; it will encode text itself — or pass the bank to avoid re-encoding)
        img_enc.eval()
        from pairedclip.eval import evaluate_topk
        all_txt = caption_bank.to(device)  # reuse bank for speed
        metrics = evaluate_topk(img_enc, txt_enc, device, class_names,
                                root=cfg.data_root, size=cfg.eval_size,
                                batch_prompts=128,  # ignored because we pass all_txt
                                )
        # NOTE: if your evaluate_topk supports an 'all_txt' argument, pass it explicitly:
        # metrics = evaluate_topk(img_enc, txt_enc, device, class_names, root=cfg.data_root,
        #                         size=cfg.eval_size, all_txt=all_txt)

        sec = time.time() - t0
        print(f"Epoch {epoch}: time={sec:.1f}s avg_loss={running/len(loader):.4f} | {metrics}", flush=True)

if __name__ == "__main__":
    main()
