import argparse, torch, os, json
from clipmini.utils import load_checkpoint, get_device
from clipmini.models.image_encoder import TinyImageEncoder, MicroViT
from clipmini.models.text_encoder import TinyTextEncoder, OpenClipTextEncoder
from clipmini.eval import evaluate_topk
from clipmini.data import PairedCIFAR100

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--text_encoder", choices=["tiny","openclip"], default="tiny")
    p.add_argument("--openclip_model", type=str, default="ViT-B-32")
    p.add_argument("--openclip_pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--eval_size", type=int, default=5000)   # larger for more stable metrics
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    obj = load_checkpoint(args.checkpoint)
    cfg = obj.get("config", {})
    emb_dim = cfg.get("emb_dim", 256)
    use_micro_vit = cfg.get("use_micro_vit", False)

    device = get_device()
    # dataset (for class names)
    ds = PairedCIFAR100(root=args.data_root, train=True, size=10, augment=False)
    class_names = ds.class_names

    # build image model
    img = MicroViT(d=emb_dim) if use_micro_vit else TinyImageEncoder(d=emb_dim)
    img.to(device)
    load_checkpoint(args.checkpoint, model=img)

    # build text model
    if args.text_encoder == "openclip":
        txt = OpenClipTextEncoder(args.openclip_model, args.openclip_pretrained, device, class_names=class_names).to(device).eval()
    else:
        txt = TinyTextEncoder(num_classes=len(class_names), d=emb_dim, seed=0).to(device).eval()

    # evaluate
    metrics = evaluate_topk(img.eval(), txt.eval(), device, root=args.data_root,
                            size=args.eval_size, num_workers=args.num_workers)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
