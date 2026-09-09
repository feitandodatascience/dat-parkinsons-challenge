"""Cross-validated training loop.

Structurally the same loop you wrote for MNIST — forward, loss, backward, step, validate — with
four changes that matter for this competition:

1. **Early stopping and model selection use validation LOG LOSS**, because that is the metric.
   Not accuracy, not AUROC.
2. **Out-of-fold predictions are saved**, because calibration must be fitted on OOF data
   (see `datpd.calibrate`) and that is worth real leaderboard points.
3. **AMP by default.** 3D activations are large; mixed precision roughly halves memory.
4. **Both fold columns supported** (`fold_strat`, `fold_group`) so the same code produces the
   optimistic and the honest estimate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import DaTScanDataset
from .models.factory import build_model
from .utils.metrics import summarize
from .utils.seed import set_seed


def _make_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps: int):
    """Cosine annealing or ReduceLROnPlateau, per config.

    Cosine is the default for short runs (no patience to tune). ReduceLROnPlateau is offered
    because it is a familiar, perfectly reasonable alternative — just make sure it watches
    validation log loss, which is what `train_fold` passes to `.step()`.
    """
    if cfg.train.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.train.plateau_factor,
            patience=cfg.train.plateau_patience,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return probabilities for a loader. Always in eval mode (see CLAUDE.md 1.1)."""
    model.eval()
    probs: list[np.ndarray] = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs)


def train_fold(
    cfg: Config, df: pd.DataFrame, fold: int, device: torch.device
) -> tuple[nn.Module, np.ndarray, np.ndarray, dict[str, float]]:
    """Train one fold. Returns (best model, val indices, val probabilities, metrics)."""
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_col = cfg.train.fold_column
    train_df = df[df[fold_col] != fold]
    val_df = df[df[fold_col] == fold]

    train_ds = DaTScanDataset(train_df, cfg.data, cfg.aug, train=True)
    val_ds = DaTScanDataset(val_df, cfg.data, cfg.aug, train=False)

    # Pinned memory only helps when there is a device to copy to asynchronously; on CPU it is
    # pure overhead and torch warns once per loader, which buries the loss curve in a long run.
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=pin,
    )

    model = build_model(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = _make_scheduler(optimizer, cfg, cfg.train.epochs)
    # BCEWithLogitsLoss, not sigmoid + BCE: it fuses the two and is numerically stable.
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.train.amp and device.type == "cuda")

    y_val = val_df["is_pathologic"].to_numpy()
    best_loss, best_state, best_probs, patience = float("inf"), None, None, 0
    start_epoch = 0

    # The per-epoch curve, banked to disk as it goes. Previously this existed only as stdout,
    # so once a terminal was closed the run could no longer answer questions about itself -
    # "did early stopping fire with the loss still falling?", "how far did the cosine schedule
    # actually get before it stopped?" - and those are exactly the questions asked afterwards.
    # `lr` is recorded because with `T_max = cfg.train.epochs`, early stopping can end a run
    # while the annealing is still far from complete, and nothing else would show that.
    history: list[dict[str, float]] = []
    history_path = out_dir / f"history_fold{fold}.json"

    # Per-epoch checkpointing. On a GPU a fold is minutes and this would be noise; on CPU a fold
    # is ~100 minutes, and without it any interruption - sleep, a Windows update, a closed
    # terminal - discards the entire fold. The unit of loss should be one epoch, not one fold.
    resume_path = out_dir / f"fold{fold}_last.pt"
    if cfg.train.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state["best_loss"])
        best_state = state["best_state"]
        best_probs = state["best_probs"]
        patience = int(state["patience"])
        history = list(state.get("history", []))
        print(
            f"  fold {fold}: resuming at epoch {start_epoch} "
            f"(best val_logloss so far {best_loss:.4f})",
            flush=True,
        )

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        start = time.time()
        running = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=cfg.train.amp and device.type == "cuda"):
                loss = criterion(model(x), y)

            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        probs = predict(model, val_loader, device)
        metrics = summarize(y_val, probs)

        # Read before stepping: this is the rate the epoch just above was actually trained at,
        # not the one the next epoch will use.
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        if cfg.train.scheduler == "plateau":
            scheduler.step(metrics["logloss"])
        else:
            scheduler.step()

        train_loss = running / max(len(train_loader), 1)
        elapsed = time.time() - start
        print(
            f"  fold {fold} epoch {epoch:3d}  "
            f"train {train_loss:.4f}  "
            f"val_logloss {metrics['logloss']:.4f}  auroc {metrics['auroc']:.4f}  "
            f"lr {epoch_lr:.2e}  {elapsed:.0f}s",
            flush=True,
        )

        history.append(
            {
                "epoch": int(epoch),
                "lr": epoch_lr,
                "train_loss": float(train_loss),
                "val_logloss": float(metrics["logloss"]),
                "auroc": float(metrics["auroc"]),
                "seconds": float(elapsed),
            }
        )
        # Written every epoch rather than at the end of the fold: on CPU a fold is ~100 minutes
        # and interruptions are expected, and a curve that only survives a completed fold would
        # be missing precisely when it is most wanted.
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        # Model selection on validation LOG LOSS - the competition metric.
        stop = False
        if metrics["logloss"] < best_loss:
            best_loss = metrics["logloss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_probs = probs
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.early_stopping_patience:
                # Say what the stop cost, not just that it happened. `T_max` is
                # cfg.train.epochs, so stopping early leaves the cosine schedule unfinished and
                # selects weights trained at a rate well above the annealed one - visible here
                # as an lr that is still a large fraction of cfg.train.lr.
                best_epoch = min(history, key=lambda h: h["val_logloss"])["epoch"]
                print(
                    f"  fold {fold}: early stop at epoch {epoch} "
                    f"(best epoch {best_epoch}, val_logloss {best_loss:.4f}); "
                    f"lr {epoch_lr:.2e} of {cfg.train.lr:.2e}, "
                    f"{epoch + 1}/{cfg.train.epochs} of the schedule run",
                    flush=True,
                )
                stop = True

        if cfg.train.resume:
            # Written after model selection so the checkpoint carries the *best* weights too,
            # not just the latest. Written to a temporary name and renamed, because a process
            # killed mid-write would otherwise leave a truncated file that resumes into a
            # crash - which is precisely the situation this exists to survive.
            tmp_path = out_dir / f"fold{fold}_last.tmp"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_loss": best_loss,
                    "best_state": best_state,
                    "best_probs": best_probs,
                    "patience": patience,
                    "history": history,
                },
                tmp_path,
            )
            tmp_path.replace(resume_path)

        if stop:
            break

    assert best_state is not None and best_probs is not None
    model.load_state_dict(best_state)

    # The fold is complete; its resume checkpoint is now dead weight (~20 MB) and would make a
    # rerun of this fold silently skip straight to the end.
    resume_path.unlink(missing_ok=True)

    return model, val_df.index.to_numpy(), best_probs, summarize(y_val, best_probs)


def assemble_oof(out_dir: Path) -> pd.DataFrame:
    """Concatenate every per-fold prediction file present in `out_dir`.

    Public because `scripts/04_evaluate.py` uses it to recover from an interrupted run: a run
    that trained three folds and died leaves three fold files and no `oof.csv`, and there is no
    reason to retrain to get a score out of them.

    Reads from disk rather than memory so that folds trained in *earlier* invocations are
    included — which is what makes `--folds` resumable across sessions.

    NOTE: these files are not stamped with the config that produced them. After changing
    preprocessing or architecture, delete `oof_fold*.csv` before retraining, or the assembled
    OOF will silently mix two models.
    """
    fold_files = sorted(out_dir.glob("oof_fold*.csv"))
    if not fold_files:
        raise FileNotFoundError(f"no per-fold predictions in {out_dir}; did any fold finish?")

    frames = [pd.read_csv(path) for path in fold_files]
    result = pd.concat(frames, ignore_index=True)

    # A uid must be predicted out-of-fold exactly once. Duplicates mean overlapping folds,
    # which would make the OOF score optimistic.
    duplicated = int(result["uid"].duplicated().sum())
    if duplicated:
        raise ValueError(
            f"{duplicated} uids appear in more than one fold's predictions; "
            "delete oof_fold*.csv and retrain rather than scoring a leaky OOF set"
        )

    if len(fold_files) > 1:
        print(f"Assembled OOF from {len(fold_files)} fold files ({len(result)} rows)")
    return result


def run(cfg: Config) -> pd.DataFrame:
    """Train every fold, save checkpoints, and write out-of-fold predictions.

    The OOF frame is the input to calibration and to every honest comparison between
    experiments. It contains uids, so it is written under `models/` which is git-ignored and
    Claude-blocked.
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seed: {cfg.seed} | fold column: {cfg.train.fold_column}")

    df = pd.read_csv(cfg.data.folds_csv)
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in cfg.train.folds:
        print(f"\n=== fold {fold} ===", flush=True)
        model, val_idx, probs, metrics = train_fold(cfg, df, fold, device)
        torch.save(model.state_dict(), out_dir / f"fold{fold}.pt")

        # Persist each fold's predictions the moment it finishes, and assemble `oof.csv` from
        # those files rather than from this process's memory.
        #
        # Without this, a run resumed as `--folds 2 3 4` writes an oof.csv containing only
        # folds 2-4, and every number computed from it - calibration temperature, the clip,
        # the experiment log row - is silently based on 60% of the data. On a CPU where five
        # folds span a whole night and interruptions are expected, that is not a corner case.
        fold_frame = df.loc[val_idx, ["uid", "is_pathologic"]].copy()
        fold_frame["oof_pred"] = probs
        fold_frame["fold"] = fold
        fold_frame.to_csv(out_dir / f"oof_fold{fold}.csv", index=False)

    result = assemble_oof(out_dir)
    result.to_csv(out_dir / "oof.csv", index=False)

    per_fold = [
        {"fold": int(fold), **summarize(grp["is_pathologic"].to_numpy(), grp["oof_pred"].to_numpy())}
        for fold, grp in result.groupby("fold")
    ]
    overall = summarize(result["is_pathologic"].to_numpy(), result["oof_pred"].to_numpy())

    # Assembled from the per-fold files rather than from this process's memory, for the same
    # reason oof.csv is: a run resumed as `--folds 2 3 4` must still produce a metrics.json
    # describing all five folds, not the three this process happened to train.
    history: dict[str, list[dict[str, float]]] = {}
    schedule: list[dict[str, float]] = []
    for path in sorted(out_dir.glob("history_fold*.json")):
        fold_history = json.loads(path.read_text(encoding="utf-8"))
        if not fold_history:
            continue
        key = path.stem.removeprefix("history_fold")
        history[key] = fold_history
        best = min(fold_history, key=lambda h: h["val_logloss"])
        last = fold_history[-1]
        schedule.append(
            {
                "fold": int(key),
                "best_epoch": best["epoch"],
                "best_val_logloss": best["val_logloss"],
                "last_epoch": last["epoch"],
                "epochs_configured": int(cfg.train.epochs),
                "stopped_early": last["epoch"] + 1 < cfg.train.epochs,
                # < 1 means the cosine anneal never finished. Weights selected at a high
                # fraction of the peak rate have not had the low-rate endgame the schedule
                # was written to give them.
                "lr_at_best_over_peak": best["lr"] / cfg.train.lr if cfg.train.lr else 0.0,
            }
        )

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "overall": overall,
                "per_fold": per_fold,
                "schedule": schedule,
                "history": history,
                "config": cfg.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== OOF ===")
    print(f"  logloss          {overall['logloss']:.4f}")
    print(f"  prior baseline   {overall['baseline_logloss']:.4f}   <- must beat this")
    print(f"  auroc            {overall['auroc']:.4f}")
    print(f"  ece              {overall['ece']:.4f}")
    print(f"\nNext: python scripts/04_evaluate.py --run {cfg.run_name}")
    return result
