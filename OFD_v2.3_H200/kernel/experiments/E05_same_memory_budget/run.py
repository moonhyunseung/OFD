from pathlib import Path
import subprocess, sys
ROOT = Path(__file__).resolve().parents[2]
cmd = [sys.executable, str(ROOT / "tools" / "run_experiment.py"), "--experiment", "E05_same_memory_budget", *sys.argv[1:]]
raise SystemExit(subprocess.call(cmd, cwd=ROOT.parent))
