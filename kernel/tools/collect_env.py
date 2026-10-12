from __future__ import annotations
import argparse, json, os, platform, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

def cmd(args, cwd=None):
    try: return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None

def collect(root: Path, touch_cuda: bool=False):
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": cmd(["git","rev-parse","HEAD"], cwd=root),
        "git_branch": cmd(["git","branch","--show-current"], cwd=root),
        "git_status_porcelain": cmd(["git","status","--porcelain"], cwd=root) or "",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": cmd(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"]),
    }
    data["git_available"] = bool(data.get("git_commit"))
    data["git_dirty"] = bool(data["git_status_porcelain"].strip()) if data["git_available"] else None
    data["claim_eligible"] = bool(data["git_available"]) and data["git_dirty"] is False
    try:
        import torch
        data["torch"] = torch.__version__
        data["torch_cuda"] = torch.version.cuda
        if touch_cuda and torch.cuda.is_available():
            data["gpu0"] = torch.cuda.get_device_name(0)
    except Exception as e:
        data["torch_error"] = repr(e)
    try:
        import triton
        data["triton"] = triton.__version__
    except Exception as e:
        data["triton_error"] = repr(e)
    return data

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root', default='.'); ap.add_argument('--touch-cuda', action='store_true'); ap.add_argument('--out')
    a=ap.parse_args(); d=collect(Path(a.root).resolve(), a.touch_cuda)
    txt=json.dumps(d, indent=2, sort_keys=True); print(txt)
    if a.out: Path(a.out).write_text(txt+'\n')
if __name__=='__main__': main()
