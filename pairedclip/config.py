from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_root: str = "./data"
    output_dir: str = "./output"
    run_name: str = "Adding_logging"
    batch_size: int = 256
    epochs: int = 20
    lr: float = 5e-4
    weight_decay: float = 0.05
    temperature: float = 0.07
    seed: int = 42
    num_workers: int = 4
    amp: bool = True
    accum_steps: int = 4  # bigger effective batch
    min_lr: float = 1e-6  # cosine floor
    warmup_steps: int = 500  # linear warmup
    curriculum_epochs: int = 10  # early epochs: harder negatives (diff superclasses)
    use_augs: bool = True
    aug_policy: str = "none"
    mixup: int = 1
    mixup_alpha: float = 0.4
    mixup_start_epoch: int = 2

    # model
    emb_dim: int = 512  # CLIP ViT-B/32 text dim = 512
    vit_dim: int = 256
    vit_depth: int = 6
    vit_heads: int = 4
    vit_patch: int = 4

    # eval
    eval_size: int = 2000     # quick pass while iterating
    eval_batches: int | None = None  # None = all

    # ==== loss tuning ====
    use_swap_margin: bool = True
    swap_margin: float = 0.10  # hinge margin between pos vs. swapped
    swap_weight: float = 0.5  # weight for swap hinge term

    use_partial_softmax: bool = True
    partial_m: int = 8  # semi-hard negatives per sample
    partial_weight: float = 1.0  # weight for a partial softmax term

    label_smoothing_eps: float = 0.0  # 0.05 is a good start; 0.0 disables

    reg_logit_scale_weight: float = 0.0  # e.g., 1e-4 to gently center logit_scale
    reg_logit_scale_tau: float = 0.07  # center at log(1/τ)
