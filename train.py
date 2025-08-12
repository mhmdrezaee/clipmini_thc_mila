import argparse, math, torch, random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from pairedclip.config import TrainConfig
from pairedclip.data import PairedCIFAR100
from pairedclip.models import ViTImageEncoder, CLIPTextEncoderHF
from pairedclip.losses import contrastive_loss_with_logit_scale
import os, sys, time, json, csv, logging
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--emb_dim", type=int, default=None)
    p.add_argument("--vit_dim", type=int, default=None)
    p.add_argument("--vit_depth", type=int, default=None)
    p.add_argument("--vit_heads", type=int, default=None)
    p.add_argument("--vit_patch", type=int, default=None)
    p.add_argument("--eval_size", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--accum_steps", type=int, default=None)
    p.add_argument("--min_lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--curriculum_epochs", type=int, default=None)
    p.add_argument("--use_augs", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="./runs/exp1")
    p.add_argument("--run_name", type=str, default=None)

    args = p.parse_args()

    cfg = TrainConfig()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)

    # Related to logging
    # ── outputs ──────────────────────────────────────────────────────────────────
    run_dir = Path(cfg.output_dir)
    if args.run_name:
        run_dir = run_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    def setup_logger(log_path: Path):
        logger = logging.getLogger("train")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stderr);
        sh.setFormatter(fmt)
        fh = logging.FileHandler(log_path, encoding="utf-8");
        fh.setFormatter(fmt)
        logger.addHandler(sh);
        logger.addHandler(fh)
        return logger

    logger = setup_logger(run_dir / "train.log")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg.__dict__}")

    # Save a JSON copy of the run config
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Optional: TensorBoard (won't crash if not installed)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir))
        logger.info("TensorBoard logging enabled.")
    except Exception as e:
        logger.info(f"TensorBoard not available ({e}); skipping.")

    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "avg_loss", "top1", "top10", "top100", "lr", "logit_scale"])

    def log_row(epoch, avg_loss, metrics, lr, logit_scale_val):
        with open(metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, f"{avg_loss:.6f}",
                f"{metrics['top-1']:.6f}", f"{metrics['top-10']:.6f}", f"{metrics['top-100']:.6f}",
                f"{lr:.8f}", f"{logit_scale_val:.6f}"
            ])

    # Data
    ds = PairedCIFAR100(root=cfg.data_root, train=True, size=20000)
    class_names = ds.class_names
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True)

    # Models
    txt_enc = CLIPTextEncoderHF("openai/clip-vit-base-patch32")  # frozen
    img_enc = ViTImageEncoder(emb_dim=cfg.emb_dim, dim=cfg.vit_dim,
                              depth=cfg.vit_depth, heads=cfg.vit_heads,
                              patch=cfg.vit_patch).to(device)

    def build_caption_bank(txt_enc, class_names, device="cpu", batch_size=128):
        prompts = [
            f"The photo on the left is {class_names[i]}, the photo on the right is {class_names[j]}."
            for i in range(100) for j in range(100)
        ]
        # run text on CPU, return CPU tensor; ~10k x 512 (~20MB fp32)
        bank = txt_enc.encode_prompts(prompts, device=device, batch_size=batch_size)
        return bank.cpu()  # keep on CPU

    caption_bank = build_caption_bank(txt_enc, class_names, device="cpu", batch_size=128)

    # Learnable logit scale (init from temperature τ)
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1.0 / cfg.temperature), device=device))

    # Two param groups: no weight decay on the scalar
    opt = torch.optim.AdamW([
        {"params": img_enc.parameters(), "weight_decay": cfg.weight_decay, "lr": cfg.lr},
        {"params": [logit_scale], "weight_decay": 0.0, "lr": cfg.lr},
    ])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # we re-create the loader each epoch (for curriculum), so compute total steps from the 1st epoch size
    tmp_ds = PairedCIFAR100(root=cfg.data_root, train=True, size=20000, different_superclass=True)
    steps_per_epoch = math.ceil(len(tmp_ds) / cfg.batch_size)
    total_updates = cfg.epochs * max(1, steps_per_epoch // max(1, cfg.accum_steps))

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        # cosine to min_lr
        progress = (step - cfg.warmup_steps) / max(1, total_updates - cfg.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (cfg.min_lr / cfg.lr) + (1 - (cfg.min_lr / cfg.lr)) * cosine

    sched = LambdaLR(opt, lr_lambda)
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        epoch_t0 = time.time()

        # curriculum: early epochs enforce different superclasses (harder negatives)
        hard = (epoch <= cfg.curriculum_epochs)
        logger.info(f"Starting epoch {epoch} (hard={hard})")

        ds = PairedCIFAR100(root=cfg.data_root, train=True, size=20000, different_superclass=hard, augment=cfg.use_augs)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

        img_enc.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)

        for it, (imgs, cL, cR) in enumerate(loader, start=1):
            imgs = imgs.to(device)

            # Build prompts (on CPU), run CLIP text on CPU, return embeddings on GPU (safe & light)
            prompts = [f"The photo on the left is {ds.class_names[l]}, the photo on the right is {ds.class_names[r]}."
                       for l, r in zip(cL.tolist(), cR.tolist())]

            gt_idx = (cL.cpu() * 100 + cR.cpu())  # shape (B,)
            tz = caption_bank[gt_idx].to(device, non_blocking=True)  # (B,512) on GPU
            # tz = txt_enc.encode_prompts(prompts, device="cpu", batch_size=128)  # run text on CPU
            # tz = tz.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                iz = img_enc(imgs)  # (B,512)
                loss = contrastive_loss_with_logit_scale(iz, tz, logit_scale)

                # Logging
                if it % 100 == 0 or it == 1:
                    last_lr = sched.get_last_lr()[0]
                    ls_val = float(logit_scale.detach().exp().clamp(max=100))
                    msg = f"ep {epoch} it {it}/{len(loader)} | loss {loss.item():.4f} | lr {last_lr:.6e} | logit_scale {ls_val:.2f}"
                    if device == "cuda":
                        mem = torch.cuda.memory_allocated() / (1024 ** 2)
                        msg += f" | cuda_mem {mem:.0f}MB"
                    logger.info(msg)
                    if writer:
                        step = (epoch - 1) * len(loader) + it
                        writer.add_scalar("train/loss_step", float(loss.item()), step)
                        writer.add_scalar("train/lr", last_lr, step)
                        writer.add_scalar("train/logit_scale", ls_val, step)

            scaler.scale(loss / cfg.accum_steps).backward()

            if it % cfg.accum_steps == 0:
                if cfg.amp:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(img_enc.parameters(), 1.0)
                scaler.step(opt);
                scaler.update()
                opt.zero_grad(set_to_none=True)

                # scheduler after an optimizer step
                sched.step();
                global_step += 1

            running += loss.item()

        # EVAL (unchanged, but keep text on CPU to avoid OOM)
        img_enc.eval()
        from pairedclip.eval import evaluate_topk
        metrics = evaluate_topk(img_enc, txt_enc, device, ds.class_names,
                                root=cfg.data_root, size=cfg.eval_size, batch_prompts=128)

        #Logging
        epoch_secs = time.time() - epoch_t0
        last_lr = sched.get_last_lr()[0]
        ls_val = float(logit_scale.detach().exp().clamp(max=100))
        logger.info(f"Epoch {epoch} done in {epoch_secs:.1f}s | avg_loss={running / len(loader):.4f} | "
                    f"top1={metrics['top-1']:.4f} top10={metrics['top-10']:.4f} top100={metrics['top-100']:.4f} | "
                    f"lr={last_lr:.6e} | logit_scale={ls_val:.2f}")

        log_row(epoch, running / len(loader), metrics, last_lr, ls_val)

        if writer:
            writer.add_scalar("train/avg_loss", running / len(loader), epoch)
            writer.add_scalar("eval/top1", metrics["top-1"], epoch)
            writer.add_scalar("eval/top10", metrics["top-10"], epoch)
            writer.add_scalar("eval/top100", metrics["top-100"], epoch)
            writer.add_scalar("train/lr_epoch", last_lr, epoch)
            writer.add_scalar("train/logit_scale_epoch", ls_val, epoch)

    ckpt = {
        "img_enc": img_enc.state_dict(),
        "logit_scale": logit_scale.detach().cpu(),
        "config": cfg.__dict__,
        "epoch": epoch,
    }
    torch.save(ckpt, run_dir / f"epoch_{epoch:03d}.pt")

    # Track best by top-100
    best_path = run_dir / "best.pt"
    best_score_path = run_dir / "best_score.txt"
    cur_score = metrics["top-100"]
    prev_best = -1.0
    if best_score_path.exists():
        prev_best = float(best_score_path.read_text().strip())
    if cur_score > prev_best:
        torch.save(ckpt, best_path)
        best_score_path.write_text(str(cur_score))
        logger.info(f"New best (top-100={cur_score:.4f}) → saved to {best_path}")


if __name__ == "__main__":
    main()
