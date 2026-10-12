from __future__ import annotations
import argparse, json, os, shlex, subprocess, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

KERNEL = Path(__file__).resolve().parents[1]
REPO = KERNEL.parent
EXPS = KERNEL / "experiments"
RESULTS = KERNEL / "results"


def load_cfg(eid):
    return json.loads((EXPS / eid / "config.json").read_text())


def list_ids():
    return sorted(p.name for p in EXPS.iterdir() if p.is_dir() and (p / "config.json").exists())


def latest_manifest(eid):
    root = RESULTS / eid
    if not root.exists():
        return None
    for mp in sorted(root.glob("*/MANIFEST.json"), reverse=True):
        try:
            return json.loads(mp.read_text())
        except Exception:
            pass
    return None


def dependency_satisfied(eid):
    cfg = load_cfg(eid)
    m = latest_manifest(eid)
    if not m:
        return False, "NO_MANIFEST"
    gate_type = cfg.get("gate_type", "pass_fail")
    if gate_type == "pass_fail":
        ok = m.get("execution_status") == "COMPLETED" and m.get("gate_status") == "PASS"
    else:
        ok = m.get("execution_status") == "COMPLETED"
    return ok, f"execution={m.get('execution_status')} gate={m.get('gate_status')}"


def dependencies_missing(cfg):
    missing = []
    for dep in [d for d in cfg.get("depends_on", []) if d != "none"]:
        ok, why = dependency_satisfied(dep)
        if not ok:
            missing.append((dep, why))
    return missing


def collect_env():
    txt = subprocess.check_output(
        [sys.executable, str(KERNEL / "tools/collect_env.py"), "--root", str(REPO)], text=True
    )
    return json.loads(txt)


def manifest(cfg, run_id, commands, env):
    return {
        "schema_version": 2,
        "experiment_id": cfg["id"],
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_status": cfg["status"],
        "gate_type": cfg.get("gate_type", "pass_fail"),
        "claim_eligible_protocol": bool(cfg.get("claim_eligible", False)),
        "requires_clean_git": bool(cfg.get("requires_clean_git", False)),
        "git_commit": env.get("git_commit"),
        "git_branch": env.get("git_branch"),
        "git_available": env.get("git_available"),
        "git_dirty": env.get("git_dirty"),
        "git_status_porcelain": env.get("git_status_porcelain", ""),
        "claim_eligible_run": bool(env.get("claim_eligible")) and bool(cfg.get("claim_eligible", False)),
        "environment": env,
        "config": cfg,
        "commands": commands,
        "artifacts": [],
        "notes": [],
        "execution_status": "PLANNED",
        "gate_status": "PENDING",
        "return_code": None,
    }


def commands_for(eid, outdir):
    py = sys.executable
    if eid == "E00_cpu_reference":
        return [
            [py, "-m", "unittest", "discover", "-s", "kernel/tests/cpu", "-p", "test_*.py", "-v"],
            [py, "kernel/tools/check_cpu.py"],
        ]
    if eid == "E01_gpu_compile":
        return [[py, "kernel/tools/bench_gpu.py", "--mode", "compile", "--seed", "1337"]]
    if eid == "E02_correctness":
        return [[py, "kernel/tools/bench_gpu.py", "--mode", "correctness", "--correctness-threshold", "0.002", "--seed", "1337"]]
    if eid == "E03_split_scaling":
        cmd = [py, "kernel/tools/bench_e03.py", "--out", str(outdir / "split_scaling.csv")]
        if os.environ.get("OFD_SMOKE") == "1":
            cmd.append("--smoke")
        return [cmd]
    if eid == "E04a_core_attention":
        return [[py, "kernel/tools/bench_gpu.py", "--mode", "core-variants", "--out", str(outdir / "core_variants.csv"), "--kv-plain-max-ratio", "1.05", "--seed", "1337"]]
    if eid == "E07_long_context":
        return [[py, "kernel/tools/bench_gpu.py", "--mode", "long-core", "--out", str(outdir / "long_context_core.csv"), "--seed", "1337"]]
    return []


def gate_from_success(cfg):
    return "PASS" if cfg.get("gate_type", "pass_fail") == "pass_fail" else "NOT_APPLICABLE"


def execute(eid, dry):
    cfg = load_cfg(eid)
    print(f"\n=== {eid}: {cfg['title']} ===")
    print("status:", cfg["status"])
    print("gate type:", cfg.get("gate_type", "pass_fail"))
    print("gate:", cfg["pass_fail"])
    cmds = commands_for(eid, Path("/tmp/placeholder"))
    if dry:
        print("DRY-RUN commands:", json.dumps(cmds, indent=2))
        return 0
    if cfg["status"].startswith("BLOCKED"):
        print("BLOCKED by protocol; no execution started.")
        return 2
    missing = dependencies_missing(cfg)
    if missing:
        print("DEPENDENCY GATE BLOCKED:", missing)
        return 2

    env_info = collect_env()
    if cfg.get("requires_clean_git", False) and not env_info.get("claim_eligible", False):
        print("CLEAN-GIT GATE BLOCKED: claim-eligible experiment requires a real Git commit and a clean worktree.")
        print(env_info.get("git_status_porcelain", ""))
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    outdir = RESULTS / eid / run_id
    outdir.mkdir(parents=True)
    cmds = commands_for(eid, outdir)
    m = manifest(cfg, run_id, cmds, env_info)
    (outdir / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
    if not cmds:
        print("No executable runner is intentionally defined for this protocol yet.")
        m["execution_status"] = "BLOCKED_PROTOCOL"
        (outdir / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(KERNEL / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log = (outdir / "run.log").open("w")
    rc = 0
    for cmd in cmds:
        print("+", shlex.join(cmd))
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        proc.wait()
        rc = proc.returncode
        if rc:
            break
    log.close()

    m["return_code"] = rc
    if rc == 0:
        m["execution_status"] = "COMPLETED"
        m["gate_status"] = gate_from_success(cfg)
    elif rc == 3:
        m["execution_status"] = "BLOCKED_ENV"
        m["gate_status"] = "PENDING"
    else:
        m["execution_status"] = "FAILED"
        m["gate_status"] = "FAIL" if cfg.get("gate_type", "pass_fail") == "pass_fail" else "NOT_APPLICABLE"
    m["artifacts"] = sorted(
        str(x.relative_to(outdir)) for x in outdir.rglob("*") if x.is_file() and x.name != "MANIFEST.json"
    )
    (outdir / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
    print("result dir:", outdir)
    print("return code:", rc)
    print("execution status:", m["execution_status"])
    print("gate status:", m["gate_status"])
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=list_ids())
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ids = list_ids() if args.all else [args.experiment] if args.experiment else []
    if not ids:
        ap.error("use --experiment ID or --all")
    rc = 0
    for eid in ids:
        rc = max(rc, execute(eid, args.dry_run))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
