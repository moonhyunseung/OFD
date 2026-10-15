from pathlib import Path
import subprocess, sys
ROOT = Path(__file__).resolve().parents[2]
cmd = [sys.executable, str(ROOT / "tools" / "run_experiment.py"), "--experiment", "E04b_end_to_end", *sys.argv[1:]]
raise SystemExit(subprocess.call(cmd, cwd=ROOT.parent))
