import argparse, os, csv
import torch
from torch.utils.data import DataLoader
from clipmini.config import TrainConfig
from clipmini.utils import set_seed, get_device, ensure_dir, save_checkpoint, write_json, hash_config
from clipmini.data import PairedCIFAR100, train_transform
from clipmini.models.image_encoder import TinyImageEncoder, MicroViT
from clipmini.models.text_encoder import TinyTextEncoder, OpenClipTextEncoder
from clipmini.losses import contrastive_loss
from clipmini.eval import evaluate_topk

def build_models(cfg: TrainConfig, device, class_names):
    # text encoder
    if cfg.text_encoder.lower() == "openclip":
        txt = OpenClipTextEncoder(cfg.openclip_model, cfg.openclip_pretrained, device, class_names=class_names)
    else:
        txt = TinyTextEncoder(num_classes=len(class_names), d=cfg.emb_dim, seed=0)
    txt = txt.to(device).eval()

    # image encoder
    img = MicroViT(d=cfg.emb_dim) if cfg.use_micro_vit else TinyImageEncoder(d=cfg.emb_dim)
    img = img.to(device)
    return img, txt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--emb_dim", type=int, default=None)
    p.add_argument("--use_micro_vit", action="store_true")
    p.add_argument("--text_encoder", choices=["tiny","openclip"], default=None)
    p.add_argument("--openclip_model", type=str, default=None)
    p.add_argument("--openclip_pretrained", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--accum_steps", type=int, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--train_size", type=int, default=None)
    p.add_argument("--eval_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    cfg = TrainConfig()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    set_seed(cfg.seed)
    device = get_device()
    ensure_dir(cfg.output_dir)

    # data
    ds = PairedCIFAR100(root=cfg.data_root, train=True, size=cfg.train_size, transform=train_transform(augment=True))
    class_names = ds.class_names

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    # models
    img_enc, txt_enc = build_models(cfg, device, class_names)

    # optim
    opt = torch.optim.AdamW(img_enc.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # logging
    run_hash = hash_config(cfg.__dict__)
    meta_path = os.path.join(cfg.output_dir, f"meta_{run_hash}.json")
    write_json(meta_path, {**cfg.__dict__, "device": device, "run_hash": run_hash})

    metrics_csv = os.path.join(cfg.output_dir, f"metrics_{run_hash}.csv")
    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "top-1", "top-10", "top-100"])

    global_step = 0
    best_t100 = 0.0

    for epoch in range(1, cfg.epochs + 1):
        img_enc.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)

        for it, (imgs, cL, cR) in enumerate(loader, start=1):
            imgs, cL, cR = imgs.to(device), cL.to(device), cR.to(device)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                iz = img_enc(imgs)  # on GPU
                tz = txt_enc(cL.cpu(), cR.cpu())  # text encoding on CPU
                tz = tz.to(iz.device, non_blocking=True)
                loss = contrastive_loss(iz, tz, temperature=cfg.temperature)

            scaler.scale(loss / cfg.accum_steps).backward()
            if it % cfg.accum_steps == 0:
                if cfg.grad_clip:
                    # Unscale grads before clipping when AMP is on
                    if cfg.amp:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(img_enc.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += loss.item()
            global_step += 1

            # warmup (linear)
            if global_step <= cfg.warmup_steps:
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * global_step / max(1, cfg.warmup_steps)

        img_enc.eval()
        metrics = evaluate_topk(img_enc, txt_enc, device,
                                root=cfg.data_root, size=cfg.eval_size, num_workers=cfg.num_workers)
        print(f"Epoch {epoch}: loss={running/len(loader):.4f} | {metrics}")

        # log
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, running/len(loader), metrics["top-1"], metrics["top-10"], metrics["top-100"]])

        # save
        if (epoch % cfg.save_every) == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, img_enc, opt, epoch, cfg)

        # track best by top-100
        if metrics["top-100"] > best_t100:
            best_t100 = metrics["top-100"]
            best_path = os.path.join(cfg.output_dir, "best.pt")
            save_checkpoint(best_path, img_enc, opt, epoch, cfg, extra={"top-100": best_t100})

if __name__ == "__main__":
    main()
