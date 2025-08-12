import torch
import torch.nn.functional as F
from transformers import CLIPModel, AutoTokenizer
from transformers import CLIPModel, AutoTokenizer


class CLIPTextEncoderHF(torch.nn.Module):
    """
    Frozen CLIP text encoder from Hugging Face.
    Uses CLIPModel.get_text_features() to produce 512-d embeddings (ViT-B/32).
    """
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # ⬇️ force safetensors so no torch.load(.bin) happens
        self.model = CLIPModel.from_pretrained(
            model_id,
            use_safetensors=True,   # <— important
            torch_dtype=torch.float32,
            local_files_only=False
        ).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_prompts(self, prompts, device, batch_size: int = 256):
        # make sure the HF CLIP model lives on the same device as the tokens
        self.model = self.model.to(device).eval()

        zs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            toks = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            z = self.model.get_text_features(**toks)  # (B, 512) on the same device
            zs.append(F.normalize(z, dim=-1))
        return torch.cat(zs, dim=0)
