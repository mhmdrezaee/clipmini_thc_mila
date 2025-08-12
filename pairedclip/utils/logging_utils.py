import sys, csv, json, logging
from pathlib import Path
from typing import Tuple, Optional

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr); sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger

def setup_run_dir(output_dir: str, run_name: Optional[str]) -> Path:
    run_dir = Path(output_dir)
    if run_name: run_dir /= run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_config(run_dir: Path, cfg: dict) -> None:
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

def init_tensorboard(run_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(run_dir))
    except Exception:
        return None

def init_metrics_csv(run_dir: Path) -> Path:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "avg_loss", "top1", "top10", "top100", "lr", "logit_scale"]
            )
    return csv_path

def append_metrics_row(csv_path: Path, epoch: int, avg_loss: float, metrics: dict,
                       lr: float, logit_scale: float) -> None:
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch, f"{avg_loss:.6f}",
            f"{metrics['top-1']:.6f}", f"{metrics['top-10']:.6f}", f"{metrics['top-100']:.6f}",
            f"{lr:.8f}", f"{logit_scale:.6f}",
        ])
