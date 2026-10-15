#!/usr/bin/env bash
REPO="${OFD_REPO:-$HOME/OFD}"
WORKTREE="${OFD_KERNEL_WORKTREE:-$HOME/OFD-kernel-v2}"
BACKUP_ROOT="${OFD_KERNEL_BACKUP_ROOT:-$HOME/OFD-kernel-backups}"
BRANCH="kernel/triton-v2"
BASE="2c6e180"
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " OFD KERNEL v2.3 — GPU-FREE PACKAGING INSTALL"
echo "============================================================"
echo "repo     : $REPO"
echo "worktree : $WORKTREE"
echo "base     : $BASE"
echo

if [ ! -d "$REPO/.git" ]; then
  echo "ERROR: $REPO is not a git repository. Nothing changed."
elif ! git -C "$REPO" rev-parse --verify "$BASE^{commit}" >/dev/null 2>&1; then
  echo "ERROR: base commit $BASE not found. Nothing changed."
else
  if git -C "$WORKTREE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    DIRTY="$(git -C "$WORKTREE" status --porcelain)"
    if [ -n "$DIRTY" ]; then
      echo "ERROR: existing kernel worktree is dirty; refusing to overwrite."
      echo "$DIRTY"
      echo "Commit/stash/remove those changes first."
    else
      READY=1
    fi
  elif git -C "$REPO" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git -C "$REPO" worktree add "$WORKTREE" "$BRANCH" && READY=1
  else
    git -C "$REPO" worktree add -b "$BRANCH" "$WORKTREE" "$BASE" && READY=1
  fi

  if [ "${READY:-0}" -eq 1 ]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    if [ -e "$WORKTREE/kernel" ]; then
      mkdir -p "$BACKUP_ROOT/$STAMP"
      cp -a "$WORKTREE/kernel" "$BACKUP_ROOT/$STAMP/kernel"
      rm -rf "$WORKTREE/kernel"
      echo "Previous kernel snapshot backed up outside worktree: $BACKUP_ROOT/$STAMP/kernel"
    fi
    cp -a "$SOURCE/kernel" "$WORKTREE/kernel"

    echo; echo "===== SYNTAX ====="
    PYTHONPATH="$WORKTREE/kernel/src" python -m compileall -q "$WORKTREE/kernel"; S1=$?
    echo; echo "===== CPU TESTS — CUDA HIDDEN ====="
    (cd "$WORKTREE" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WORKTREE/kernel/src" python -m unittest discover -s kernel/tests/cpu -p 'test_*.py' -v); S2=$?
    echo; echo "===== CPU ORACLE CHECK — CUDA HIDDEN ====="
    (cd "$WORKTREE" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WORKTREE/kernel/src" python kernel/tools/check_cpu.py); S3=$?
    echo; echo "===== EXPERIMENT DRY RUN ====="
    (cd "$WORKTREE" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WORKTREE/kernel/src" python kernel/tools/run_experiment.py --all --dry-run); S4=$?

    if [ "$S1" -eq 0 ] && [ "$S2" -eq 0 ] && [ "$S3" -eq 0 ] && [ "$S4" -eq 0 ]; then
      git -C "$WORKTREE" add kernel
      git -C "$WORKTREE" commit \
        -m "Package OFD Triton v2.3 measurement-safe experiment workspace" \
        -m "Remove timed-path split-count synchronization when a workspace is supplied; separate core-attention and end-to-end OFD gates; add clean-Git provenance, debug paged-pointer validation, frozen seeds, observational gate semantics, and manifest schema v2."
      echo; git -C "$WORKTREE" log -2 --oneline --decorate
      echo "LOCAL COMMIT ONLY. NOT PUSHED. NO GPU KERNEL WAS EXECUTED."
    else
      echo "Validation failed; files remain in worktree and are NOT committed."
      echo "syntax=$S1 tests=$S2 oracle=$S3 dryrun=$S4"
    fi
  fi
fi

echo "SSH session remains open."
