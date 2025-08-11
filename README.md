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
poetry run python train.py \
  --data_root ./data \
  --epochs 10 \
  --batch_size 256 \
  --emb_dim 256 \
  --use_micro_vit \
  --text_encoder tiny \
  --lr 5e-4 --weight_decay 0.05 --temperature 0.07 \
  --amp --accum_steps 1 --warmup_steps 500 \
  --train_size 50000 --eval_size 2000 \
  --output_dir ./runs/exp1
```

## Evaluate
```bash
poetry run python evaluate.py \
  --checkpoint ./runs/exp1/best.pt \
  --data_root ./data \
  --text_encoder tiny \
  --eval_size 5000
```

## Notes
- Image encoder is trained from scratch; text side is frozen.
- Contrastive loss is bidirectional (image↔text).
- Augmentations apply before concatenating the two images.
- AMP and gradient accumulation are available for larger effective batches.


