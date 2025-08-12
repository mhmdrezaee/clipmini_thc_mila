#!/bin/sh
set -eu

# pick python runner
if command -v poetry >/dev/null 2>&1; then
  PY="poetry run python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

# Common knobs (fixed for isolation)
EPOCHS=10
BATCH=32
ACCUM=8                  # eff batch 256
LR=8e-4
WARMUP=300               # < total updates; exits warmup
CURRICULUM=3             # gentle early help, consistent across runs
EVAL_SIZE=2000           # fast proxy; do full eval later
DATA_ROOT=./data
OUT_ROOT=./runs/phase2_augs

mkdir -p "$OUT_ROOT"

BASE="$PY train.py \
  --epochs $EPOCHS --batch_size $BATCH --accum_steps $ACCUM --amp \
  --data_root $DATA_ROOT --eval_size $EVAL_SIZE \
  --lr $LR --warmup_steps $WARMUP --curriculum_epochs $CURRICULUM \
  --use_augs 1 --mixup 0 \
  --emb_dim 512 --vit_dim 256 --vit_depth 6 --vit_heads 4 --vit_patch 4 \
  --output_dir $OUT_ROOT"

run() {
  NAME="$1"; POLICY="$2"
  echo ">> $NAME  (policy=$POLICY)"
  $BASE --aug_policy "$POLICY" --run_name "$NAME"
}

echo "=== Phase-2: augmentation policies (10 epochs, all else fixed) ==="

run "aug_none"       "none"
run "aug_light"      "light_basic"
run "aug_color"      "light_color"
run "aug_blur"       "light_blur"
run "aug_erase"      "light_erase"
run "aug_all"        "light_all"
run "aug_rrc"        "light_rrc"

echo "=== Summarize ==="
$PY summarize_runs.py --root "$OUT_ROOT" --out "$OUT_ROOT/summary.csv"
echo "Summary written to $OUT_ROOT/summary.csv"
