import argparse, torch, random
from torch.utils.data import DataLoader
from pairedclip.config import TrainConfig
from pairedclip.data import PairedCIFAR100
from pairedclip.models import ViTImageEncoder, CLIPTextEncoderHF
from pairedclip.losses import contrastive_loss
from pairedclip.eval import evaluate_topk

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
    opt = torch.optim.AdamW(img_enc.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Train
    for epoch in range(1, cfg.epochs + 1):
        img_enc.train()
        running = 0.0
        for imgs, cL, cR in loader:
            imgs = imgs.to(device)
            # Build batch captions
            prompts = [f"The photo on the left is {class_names[l]}, the photo on the right is {class_names[r]}."
                       for l, r in zip(cL.tolist(), cR.tolist())]
            # Encode text (frozen)
            tz = txt_enc.encode_prompts(prompts, device=device)   # (B, 512)
            iz = img_enc(imgs)                                    # (B, 512)
            loss = contrastive_loss(iz, tz, temperature=cfg.temperature)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        # Eval (quick)
        img_enc.eval()
        metrics = evaluate_topk(img_enc, txt_enc, device, class_names,
                                root=cfg.data_root, size=cfg.eval_size)
        print(f"Epoch {epoch}: loss={running/len(loader):.4f} | {metrics}")

if __name__ == "__main__":
    main()
