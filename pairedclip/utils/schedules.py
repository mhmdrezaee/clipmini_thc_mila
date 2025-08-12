import math
from torch.optim.lr_scheduler import LambdaLR

def build_warmup_cosine(optimizer, epochs: int, steps_per_epoch: int,
                        accum_steps: int, base_lr: float, min_lr: float,
                        warmup_steps: int):
    total_updates = epochs * max(1, steps_per_epoch // max(1, accum_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * prog))
        return (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cosine

    return LambdaLR(optimizer, lr_lambda)
