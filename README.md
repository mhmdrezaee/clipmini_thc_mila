# clipmini_thc_mila

Minimal CLIP-like image-caption alignment on paired CIFAR-100.

## Install
```bash
poetry install
# If you want OpenCLIP text encoder:
poetry install -E clip
```

## Train (GPU)
```bash
python train.py \
   --data_root ./data  \
  --output_dir ./runs/name   \
  --run_name name_for_this_run   \
  --epochs 150   \
  --batch_size 128 \
  --accum_steps 2 \
  --amp   \
  --lr 8e-4 \
  --min_lr 1e-6 \
  --warmup_steps 700   \
  --weight_decay 0.05   \
  --curriculum_epochs 3  \
   --use_augs 1 \
   --aug_policy none   \
   --mixup 0   \
   --use_swap_margin 1 \
   --swap_margin 0.05 \
   --swap_weight 0.20   \
   --eval_size 2000

```

## Evaluate
```bash
python evaluate.py \
  --checkpoint ./runs/final_150e/best_config/best.pt \
  --data_root ./data \
  --eval_size 10

```

## Notes
- Image encoder is trained from scratch; text side is frozen.
- Contrastive loss is bidirectional (image↔text).
- Augmentations apply before concatenating the two images.
- AMP and gradient accumulation are available for larger effective batches.
