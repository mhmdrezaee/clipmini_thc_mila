from dataclasses import dataclass

@dataclass
class TrainConfig:
    # data
    data_root: str = "./data"
    train_size: int = 20000         # random pairs per epoch
    eval_size: int = 1000           # quick eval sample size
    num_workers: int = 4

    # model
    emb_dim: int = 256
    use_micro_vit: bool = False
    text_encoder: str = "tiny"      # "tiny" or "openclip"
    openclip_model: str = "ViT-B-32"
    openclip_pretrained: str = "laion2b_s34b_b79k"

    # optimization
    epochs: int = 10
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.05
    temperature: float = 0.07
    grad_clip: float = 1.0
    amp: bool = True
    accum_steps: int = 1
    warmup_steps: int = 500

    # reproducibility & logging
    seed: int = 42
    output_dir: str = "./runs/default"
    save_every: int = 1
