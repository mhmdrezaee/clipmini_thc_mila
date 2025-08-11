import torch
import torch.nn.functional as F
from transformers import CLIPModel, AutoTokenizer

class CLIPTextEncoderHF(torch.nn.Module):
    """
    Frozen CLIP text encoder from Hugging Face.
    Uses CLIPModel.get_text_features() to produce 512-d embeddings (ViT-B/32).
    """
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def encode_prompts(self, prompts, device, batch_size: int = 256):
        zs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            toks = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            z = self.model.get_text_features(**toks)  # (B, 512)
            zs.append(F.normalize(z, dim=-1))
        return torch.cat(zs, dim=0)  # (N, 512)
