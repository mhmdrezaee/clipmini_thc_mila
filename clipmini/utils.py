import os, json, hashlib, time
import torch
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # perf

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(path, model, optimizer, epoch, cfg, extra=None):
    obj = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.__dict__,
        "extra": extra or {},
        "timestamp": time.time(),
    }
    torch.save(obj, path)

def load_checkpoint(path, model=None, optimizer=None):
    obj = torch.load(path, map_location="cpu")
    if model: model.load_state_dict(obj["state_dict"])
    if optimizer and "optimizer" in obj: optimizer.load_state_dict(obj["optimizer"])
    return obj

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def hash_config(cfg_dict):
    s = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]
