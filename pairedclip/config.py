from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_root: str = "./data"
    batch_size: int = 256
    epochs: int = 10
    lr: float = 5e-4
    weight_decay: float = 0.05
    temperature: float = 0.07
    seed: int = 42
    num_workers: int = 4

    # model
    emb_dim: int = 512  # CLIP ViT-B/32 text dim = 512
    vit_dim: int = 256
    vit_depth: int = 6
    vit_heads: int = 4
    vit_patch: int = 4

    # eval
    eval_size: int = 2000     # quick pass while iterating
    eval_batches: int | None = None  # None = all
