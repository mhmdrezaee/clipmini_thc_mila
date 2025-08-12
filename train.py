import argparse, math, os, random, torch
from pathlib import Path

from pairedclip.config import TrainConfig
from pairedclip.models import ViTImageEncoder, CLIPTextEncoderHF
from pairedclip.utils.logging_utils import (
    setup_logger, setup_run_dir, save_config,
    init_tensorboard, init_metrics_csv, append_metrics_row
)
from pairedclip.utils.caption_bank import build_caption_bank
from pairedclip.utils.schedules import build_warmup_cosine
from pairedclip.trainer import (
    build_classnames, build_loader, steps_per_epoch_estimate,
    train_one_epoch, evaluate, save_checkpoint
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def parse_args():
    p = argparse.ArgumentParser()
    # data & run
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./runs/exp1")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--eval_size", type=int, default=None)
    # model & optim
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
    # tricks
    p.add_argument("--amp", action="store_true")
    p.add_argument("--accum_steps", type=int, default=None)
    p.add_argument("--min_lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--curriculum_epochs", type=int, default=None)
    p.add_argument("--use_augs", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = TrainConfig()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- logging / outputs
    run_dir = setup_run_dir(cfg.output_dir, args.run_name)
    logger = setup_logger(run_dir / "train.log")
    writer = init_tensorboard(run_dir)
    save_config(run_dir, cfg.__dict__)
    metrics_csv = init_metrics_csv(run_dir)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg.__dict__}")

    # ---- models
    class_names = build_classnames(cfg.data_root)
    txt_enc = CLIPTextEncoderHF("openai/clip-vit-base-patch32")  # frozen
    img_enc = ViTImageEncoder(emb_dim=cfg.emb_dim, dim=cfg.vit_dim,
                              depth=cfg.vit_depth, heads=cfg.vit_heads,
                              patch=cfg.vit_patch).to(device)
    # learnable logit scale
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1.0 / cfg.temperature), device=device))

    opt = torch.optim.AdamW([
        {"params": img_enc.parameters(), "weight_decay": cfg.weight_decay, "lr": cfg.lr},
        {"params": [logit_scale],        "weight_decay": 0.0,            "lr": cfg.lr},
    ])
    scaler = torch.amp.GradScaler(enabled=cfg.amp)

    # ---- caption bank (prompt ensemble), kept on CPU
    caption_bank = build_caption_bank(txt_enc, class_names, device="cpu", batch_size=128)

    # ---- scheduler (warmup → cosine)
    steps_est = steps_per_epoch_estimate(cfg)
    sched = build_warmup_cosine(opt, cfg.epochs, steps_est, cfg.accum_steps,
                                base_lr=cfg.lr, min_lr=cfg.min_lr, warmup_steps=cfg.warmup_steps)

    # ---- epochs
    for epoch in range(1, cfg.epochs + 1):
        hard = (epoch <= cfg.curriculum_epochs)
        logger.info(f"Starting epoch {epoch} (different_superclass={hard}, augs={bool(cfg.use_augs)})")

        ds, loader = build_loader(cfg, hard)
        avg_loss, secs = train_one_epoch(
            img_enc, caption_bank, loader, device, opt, scaler,
            logit_scale, sched, cfg, logger=logger, writer=writer, epoch=epoch
        )

        metrics = evaluate(img_enc, txt_enc, device, class_names, cfg)

        # log end-of-epoch
        last_lr = sched.get_last_lr()[0]
        ls_val = float(logit_scale.detach().exp().clamp(max=100))
        logger.info(f"Epoch {epoch} done in {secs:.1f}s | avg_loss={avg_loss:.4f} | "
                    f"top1={metrics['top-1']:.4f} top10={metrics['top-10']:.4f} top100={metrics['top-100']:.4f} | "
                    f"lr={last_lr:.6e} | logit_scale={ls_val:.2f}")

        append_metrics_row(metrics_csv, epoch, avg_loss, metrics, last_lr, ls_val)
        if writer:
            writer.add_scalar("train/avg_loss", avg_loss, epoch)
            writer.add_scalar("eval/top1", metrics["top-1"], epoch)
            writer.add_scalar("eval/top10", metrics["top-10"], epoch)
            writer.add_scalar("eval/top100", metrics["top-100"], epoch)
            writer.add_scalar("train/lr_epoch", last_lr, epoch)
            writer.add_scalar("train/logit_scale_epoch", ls_val, epoch)

        save_checkpoint(run_dir, epoch, img_enc, logit_scale, cfg, metrics, logger=logger)

if __name__ == "__main__":
    main()
