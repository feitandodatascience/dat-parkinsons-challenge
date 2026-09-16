#!/usr/bin/env python3
"""Stage one or more runs' fold checkpoints into the submission assets directory.

Why this is a script and not three copy commands
------------------------------------------------
`submission_src/main.py` globs `assets/fold*.pt` and averages every checkpoint it finds, which is
what makes shipping several runs together need no inference-code change. It is also what makes
staging them by hand dangerous, in two specific ways this script exists to prevent:

1. **The naming collision.** Every run writes `fold0.pt` .. `fold4.pt`, so a second run's
   checkpoints cannot simply be copied in beside the first's — they have to be renumbered. The
   experiment log records this as the reason a measured ensemble was not shipped: "both runs name
   their checkpoints fold0..fold4 so staging ten needs a manual rename".

2. **The stale checkpoint.** A leftover `fold*.pt` from a previous packaging round is silently
   averaged into the ensemble by that same glob. Nothing errors, nothing is logged, and the
   shipped model is quietly not the model that was measured. This script clears the directory
   first, for the same reason `scripts/09_blend.py` clears stale boosters before writing new ones.

Both failures produce a submission that runs cleanly and scores wrong, which is the worst kind.

What it does NOT do
-------------------
It does not choose the weighting. `main.py` averages uniformly, so the only shippable ensemble is
an equal-weight one, and staging run A twice to fake a 2:1 weighting is not supported — measure a
weighted ensemble with `scripts/12_ensemble_runs.py --weights` first, and if the weighting matters
enough to ship, that needs a change to `main.py`, not to a staging script.

Usage:
    python scripts/13_stage_checkpoints.py --runs baseline
    python scripts/13_stage_checkpoints.py --runs baseline aug_rich
    python scripts/13_stage_checkpoints.py --runs baseline aug_rich --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.config import load_config  # noqa: E402

ASSETS = REPO_ROOT / "submission_src" / "assets"

# Resume checkpoints are dicts of {model, optimizer, scheduler, ...} rather than state dicts, and
# shipping one produces a load failure at inference rather than a wrong score. Excluded by name,
# matching how the packager already filters them.
RESUME_SUFFIX = "_resume.pt"


def fold_checkpoints(run_dir: Path) -> list[Path]:
    """This run's fold checkpoints, in fold order, resume files excluded."""
    return sorted(
        p for p in run_dir.glob("fold*.pt") if not p.name.endswith(RESUME_SUFFIX)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run directories, in the order their checkpoints should be numbered")
    parser.add_argument("--config", default="conf/baseline.yaml",
                        help="config whose architecture every checkpoint must load into")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be staged, write nothing")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip loading each checkpoint into the architecture")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    models_root = REPO_ROOT / args.models_dir

    # Collect first, write second: a run named wrongly should fail before the assets directory
    # has been cleared, not halfway through repopulating it.
    staged: list[tuple[Path, str]] = []
    for run in args.runs:
        run_dir = models_root / run
        if not run_dir.is_dir():
            raise SystemExit(f"No such run directory: {args.models_dir}/{run}")
        checkpoints = fold_checkpoints(run_dir)
        if not checkpoints:
            raise SystemExit(f"No fold checkpoints in {args.models_dir}/{run}")
        for path in checkpoints:
            staged.append((path, f"fold{len(staged)}.pt"))
        print(f"  {run}: {len(checkpoints)} checkpoints")

    print(f"\n{len(staged)} checkpoints from {len(args.runs)} run(s) "
          f"-> {ASSETS.relative_to(REPO_ROOT)}/fold0.pt .. fold{len(staged) - 1}.pt")

    if args.dry_run:
        print("\nDRY RUN: nothing written.")
        return 0

    ASSETS.mkdir(parents=True, exist_ok=True)
    removed = 0
    for stale in ASSETS.glob("fold*.pt"):
        stale.unlink()
        removed += 1
    if removed:
        print(f"Cleared {removed} pre-existing checkpoint(s) — the glob would have averaged them in.")

    for source, name in staged:
        shutil.copy2(source, ASSETS / name)
    print(f"Staged {len(staged)} checkpoints.")

    if args.no_verify:
        print("\nSkipped verification (--no-verify). The packager checks them again anyway.")
        return 0

    # The same check the packager runs, done here so a mis-staged file is caught now rather than
    # after a build. A shape mismatch raises no error at predict time — adaptive pooling hides it
    # — so loading into the real architecture is the only thing that catches it.
    import torch

    from datpd.models.resnet3d import ResNet3D

    failures = []
    for _, name in staged:
        try:
            state = torch.load(ASSETS / name, map_location="cpu", weights_only=True)
            ResNet3D(widths=tuple(cfg.model.widths), dropout=cfg.model.dropout).load_state_dict(
                state
            )
        except Exception as exc:  # noqa: BLE001 - the message is the useful part
            failures.append(f"  {name}: {type(exc).__name__}: {exc}")

    if failures:
        print(f"\n{len(failures)} checkpoint(s) did not load into "
              f"ResNet3D(widths={tuple(cfg.model.widths)}):")
        print("\n".join(failures))
        return 1

    print(f"All {len(staged)} checkpoints load into ResNet3D(widths={tuple(cfg.model.widths)}).")
    print("\nNext: python scripts/05_package_submission.py --config <cfg> ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
