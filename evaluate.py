import argparse
import torch
import json
from pathlib import Path
from pairedclip.models import ViTImageEncoder, CLIPTextEncoderHF
from pairedclip.eval import evaluate_topk
from pairedclip.data import PairedCIFAR100
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    p.add_argument("--data_root", type=str, default="./data", help="Root directory for dataset")
    p.add_argument("--emb_dim", type=int, default=512, help="Embedding dimension for the model")
    p.add_argument("--vit_dim", type=int, default=256, help="Dimension for ViT model")
    p.add_argument("--vit_depth", type=int, default=6, help="Number of transformer blocks for ViT")
    p.add_argument("--vit_heads", type=int, default=4, help="Number of attention heads in ViT")
    p.add_argument("--vit_patch", type=int, default=4, help="Patch size for ViT")
    p.add_argument("--eval_size", type=int, default=5000, help="Number of samples to evaluate")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset (for class names)
    ds = PairedCIFAR100(root=args.data_root, train=True, size=10)  # Using size=10 for class names, adjust if necessary
    class_names = ds.class_names

    # Build models (load weights if you saved them)
    img_enc = ViTImageEncoder(args.emb_dim, args.vit_dim, args.vit_depth, args.vit_heads, args.vit_patch).to(device)
    txt_enc = CLIPTextEncoderHF("openai/clip-vit-base-patch32").to(device)

    # Load the checkpoint if it's available
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        img_enc.load_state_dict(checkpoint["img_enc"])
        img_enc.eval()
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Run evaluation
    metrics = evaluate_topk(img_enc, txt_enc, device, class_names,
                            root=args.data_root, size=args.eval_size)

    # Output evaluation results
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
