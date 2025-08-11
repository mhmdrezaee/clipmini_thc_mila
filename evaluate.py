import argparse, torch, json
from pairedclip.models import ViTImageEncoder, CLIPTextEncoderHF
from pairedclip.eval import evaluate_topk
from pairedclip.data import PairedCIFAR100

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=False)  # (optional: you can add saving in train.py)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--emb_dim", type=int, default=512)
    p.add_argument("--vit_dim", type=int, default=256)
    p.add_argument("--vit_depth", type=int, default=6)
    p.add_argument("--vit_heads", type=int, default=4)
    p.add_argument("--vit_patch", type=int, default=4)
    p.add_argument("--eval_size", type=int, default=5000)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset (for class names)
    ds = PairedCIFAR100(root=args.data_root, train=True, size=10)
    class_names = ds.class_names

    # Build models (load weights if you saved them)
    img_enc = ViTImageEncoder(args.emb_dim, args.vit_dim, args.vit_depth, args.vit_heads, args.vit_patch).to(device).eval()
    # If you saved state_dict in train.py, load it here:
    # img_enc.load_state_dict(torch.load(args.checkpoint, map_location=device)["state_dict"])

    txt_enc = CLIPTextEncoderHF("openai/clip-vit-base-patch32")

    metrics = evaluate_topk(img_enc, txt_enc, device, class_names,
                            root=args.data_root, size=args.eval_size)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
