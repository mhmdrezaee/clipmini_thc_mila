import argparse, math, torch, random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from pairedclip.config import TrainConfig
from pairedclip.data import PairedCIFAR100
from pairedclip.models import ViTImageEncoder, CLIPTextEncoderHF
from pairedclip.losses import contrastive_loss_with_logit_scale
import os
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
    args = p.parse_args()

    cfg = TrainConfig()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        # curriculum: early epochs enforce different superclasses (harder negatives)
        hard = (epoch <= cfg.curriculum_epochs)
        ds = PairedCIFAR100(root=cfg.data_root, train=True, size=20000, different_superclass=hard)
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
            tz = txt_enc.encode_prompts(prompts, run_on="cpu", target_device=device, batch_size=128)  # (B,512)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                iz = img_enc(imgs)  # (B,512)
                loss = contrastive_loss_with_logit_scale(iz, tz, logit_scale)

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
        print(f"Epoch {epoch}: loss={running / len(loader):.4f} | {metrics}")


if __name__ == "__main__":
    main()
