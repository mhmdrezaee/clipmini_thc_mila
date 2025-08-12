#!/usr/bin/env python
import argparse, csv, glob, os
from pathlib import Path

def read_metrics_csv(path: Path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # coerce to floats
            row["epoch"] = int(row["epoch"])
            row["avg_loss"] = float(row["avg_loss"])
            row["top1"] = float(row["top1"])
            row["top10"] = float(row["top10"])
            row["top100"] = float(row["top100"])
            row["lr"] = float(row["lr"])
            row["logit_scale"] = float(row["logit_scale"])
            rows.append(row)
    return rows

def summarize_run(run_dir: Path):
    mpath = run_dir / "metrics.csv"
    if not mpath.exists():
        return None
    rows = read_metrics_csv(mpath)
    if not rows:
        return None
    # best-by-top10 and last-epoch
    best = max(rows, key=lambda r: r["top10"])
    last = rows[-1]
    return {
        "run": str(run_dir),
        "best_epoch": best["epoch"],
        "best_top1": best["top1"],
        "best_top10": best["top10"],
        "best_top100": best["top100"],
        "last_epoch": last["epoch"],
        "last_top1": last["top1"],
        "last_top10": last["top10"],
        "last_top100": last["top100"],
        "last_loss": last["avg_loss"],
        "last_lr": last["lr"],
        "last_logit_scale": last["logit_scale"],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root folder containing run subdirs")
    ap.add_argument("--out", required=True, help="path to write summary CSV")
    args = ap.parse_args()

    root = Path(args.root)
    run_dirs = sorted([p for p in root.glob("**/") if (p / "metrics.csv").exists()])
    results = []
    for rd in run_dirs:
        s = summarize_run(rd)
        if s: results.append(s)

    # sort by best_top10 desc
    results.sort(key=lambda d: d["best_top10"], reverse=True)

    # write CSV
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run","best_epoch","best_top1","best_top10","best_top100",
            "last_epoch","last_top1","last_top10","last_top100",
            "last_loss","last_lr","last_logit_scale"
        ])
        for r in results:
            w.writerow([
                r["run"], r["best_epoch"],
                f"{r['best_top1']:.6f}", f"{r['best_top10']:.6f}", f"{r['best_top100']:.6f}",
                r["last_epoch"],
                f"{r['last_top1']:.6f}", f"{r['last_top10']:.6f}", f"{r['last_top100']:.6f}",
                f"{r['last_loss']:.6f}", f"{r['last_lr']:.8f}", f"{r['last_logit_scale']:.6f}",
            ])

    # also print a quick table
    print("\n=== Phase-1 summary (sorted by best top-10) ===")
    for r in results:
        print(f"{r['run']} | best@{r['best_epoch']} "
              f"t1={r['best_top1']:.4f} t10={r['best_top10']:.4f} t100={r['best_top100']:.4f} "
              f"| last t10={r['last_top10']:.4f} loss={r['last_loss']:.4f}")

if __name__ == "__main__":
    main()
