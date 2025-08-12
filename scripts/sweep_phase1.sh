#!/usr/bin/env bash
set -euo pipefail

# ---- Choose Python runner (poetry > python3 > python)
if command -v poetry >/dev/null 2>&1 && poetry run python - <<<'print(123)' >/dev/null 2>&1; then
  PY="poetry run python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "No Python interpreter found (poetry/python3/python). Aborting." >&2
  exit 1
fi

# --- Common knobs ---
EPOCHS=4
BATCH=32
ACCUM=8                 # effective batch = 32*8=256
EVAL_SIZE=2000
DATA_ROOT=./data
OUT_ROOT=./runs/phase1

mkdir -p "$OUT_ROOT"

BASE="$PY train.py --epochs $EPOCHS --batch_size $BATCH --accum_steps $ACCUM --amp --eval_size $EVAL_SIZE --data_root $DATA_ROOT"

echo "=== Phase 1A: LR Sweep ==="
for LR in 3e-4 5e-4 8e-4 1e-3; do
  RUN_DIR="$OUT_ROOT/lr_$LR"
  echo "-> LR=$LR  -> $RUN_DIR"
  $BASE --lr $LR --output_dir "$RUN_DIR"
done

echo "=== Phase 1B: Aug/Mixup Variants ==="
LR_FOR_AUG=5e-4

$BASE --lr $LR_FOR_AUG --use_augs 1 \
      --output_dir "$OUT_ROOT/aug_base"

$BASE --lr $LR_FOR_AUG --use_augs 1 \
      --run_name trivial --output_dir "$OUT_ROOT/aug_trivial"

$BASE --lr $LR_FOR_AUG --use_augs 1 --mixup 1 --mixup_alpha 0.4 --mixup_start_epoch 2 \
      --output_dir "$OUT_ROOT/aug_mixup"

$BASE --lr $LR_FOR_AUG --use_augs 1 --mixup 1 --mixup_alpha 0.4 --mixup_start_epoch 2 \
      --run_name trivial --output_dir "$OUT_ROOT/aug_mixup_trivial"

echo "=== Done. Now summarize ==="
$PY summarize_runs.py --root "$OUT_ROOT" --out "$OUT_ROOT/summary.csv"
