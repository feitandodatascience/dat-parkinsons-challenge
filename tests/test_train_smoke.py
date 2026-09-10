"""End-to-end training smoke test on synthetic volumes.

Exercises the whole loop — Dataset construction, the preprocessing cache, augmentation, the
model forward/backward, the AMP path, the scheduler, model selection on validation log loss, and
the OOF/metrics/checkpoint outputs — in a few seconds, on CPU.

The point is timing. A full run on this project starts by preprocessing every volume into the
cache; a crash in the training loop surfaces *after* that, ten minutes in. This test puts the
same failures a few seconds away instead.

It also pins the CPU-specific path deliberately. `train.py` builds `torch.amp.GradScaler` and
`torch.autocast` unconditionally and disables them off-GPU, and "disabled" GradScaler methods
(`unscale_`, `step`) are easy to get subtly wrong. With no CUDA available in CI or on this
machine, this is the only coverage that path gets.

Synthetic NIfTI only — never point this at `data/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from datpd.config import AugConfig, Config, DataConfig, ModelConfig, TrainConfig

N_SAMPLES = 20


@pytest.fixture(scope="module")
def synthetic_project(tmp_path_factory) -> dict:
    """A miniature copy of the real project layout, built from noise."""
    root = tmp_path_factory.mktemp("proj")
    nifti_dir = root / "niftis"
    nifti_dir.mkdir()

    rng = np.random.default_rng(0)
    rows = []
    for i in range(N_SAMPLES):
        pathologic = i % 2
        # Vary shape and spacing, as the real data does, so resampling and cropping both run.
        shape = (24, 24, 20) if i % 3 else (20, 20, 24)
        spacing = (4.0, 4.0, 4.0) if i % 2 else (3.0, 3.0, 3.0)

        array = rng.normal(200, 20, size=shape).astype(np.float32).clip(0)
        zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
        centre = [s // 2 for s in shape]
        dist2 = (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2
        array += (1500.0 * (1.0 if pathologic else 0.3) * np.exp(-dist2 / 8.0)).astype(np.float32)

        affine = np.diag([*spacing, 1.0])
        uid = f"synth{i:03d}"
        nib.save(nib.Nifti1Image(array, affine), str(nifti_dir / f"{uid}.nii.gz"))

        rows.append(
            {
                "uid": uid,
                "is_pathologic": pathologic,
                "acquisition_group": f"g{i % 3}",
                # Fold 0 is validation and deliberately holds both classes, or AUROC is undefined.
                "fold_strat": 0 if i < 8 else 1,
                "fold_group": 0 if i < 8 else 1,
            }
        )

    folds_csv = root / "folds.csv"
    pd.DataFrame(rows).to_csv(folds_csv, index=False)
    return {"root": root, "nifti_dir": nifti_dir, "folds_csv": folds_csv}


def _config(synthetic_project: dict) -> Config:
    root: Path = synthetic_project["root"]
    return Config(
        run_name="smoke",
        seed=0,
        output_dir=str(root / "models"),
        data=DataConfig(
            root=str(root),
            labels_csv=str(root / "labels.csv"),
            nifti_dir=str(synthetic_project["nifti_dir"]),
            folds_csv=str(synthetic_project["folds_csv"]),
            target_spacing_mm=(4.0, 4.0, 4.0),
            target_shape=(16, 16, 16),
            normalization="reference_region",
            crop="com",
            cache_dir=str(root / "cache"),
        ),
        model=ModelConfig(name="resnet3d", widths=(4, 8), dropout=0.1, in_channels=1),
        train=TrainConfig(
            epochs=2,
            batch_size=2,
            lr=1e-3,
            amp=True,  # exercises the disabled-GradScaler path on CPU
            num_workers=0,  # in-process, so a failure surfaces as a real traceback
            early_stopping_patience=5,
            fold_column="fold_strat",
            folds=(0,),
        ),
        aug=AugConfig(rotate_degrees=5.0, translate_mm=2.0, scale_range=(0.95, 1.05)),
    )


class TestTrainingLoop:
    def test_run_produces_oof_metrics_and_a_checkpoint(self, synthetic_project: dict) -> None:
        from datpd.train import run

        cfg = _config(synthetic_project)
        oof = run(cfg)

        out_dir = Path(cfg.output_dir) / cfg.run_name
        assert (out_dir / "fold0.pt").exists(), "no checkpoint written"
        assert (out_dir / "oof.csv").exists()
        assert (out_dir / "metrics.json").exists()

        # One fold trained, so OOF covers exactly that fold's rows.
        assert len(oof) == 8
        assert oof["oof_pred"].between(0.0, 1.0).all()
        assert np.isfinite(oof["oof_pred"]).all()

        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        assert set(metrics) == {"overall", "per_fold", "schedule", "history", "config"}
        assert np.isfinite(metrics["overall"]["logloss"])


class TestTrainingHistory:
    """The per-epoch curve must outlive the terminal that printed it.

    Run 1 could not answer "did early stopping fire while the loss was still falling?" because
    the only record was stdout. These pin the fix.
    """

    def test_history_is_banked_per_epoch_to_its_own_file(self, synthetic_project: dict) -> None:
        from datpd.train import run

        cfg = _config(synthetic_project)
        run(cfg)
        out_dir = Path(cfg.output_dir) / cfg.run_name

        history = json.loads((out_dir / "history_fold0.json").read_text(encoding="utf-8"))
        assert len(history) == cfg.train.epochs, "one row per epoch trained"
        for row in history:
            assert set(row) == {
                "epoch",
                "lr",
                "train_loss",
                "val_logloss",
                "auroc",
                "seconds",
            }
            assert np.isfinite(row["val_logloss"])

        # The rate must be the one the epoch was trained at, so it is read before the
        # scheduler steps: epoch 0 therefore still carries the configured peak.
        assert history[0]["lr"] == pytest.approx(cfg.train.lr)
        # Cosine anneals monotonically, so a later epoch must never train faster.
        assert history[-1]["lr"] <= history[0]["lr"]

    def test_metrics_json_carries_the_schedule_summary(self, synthetic_project: dict) -> None:
        from datpd.train import run

        cfg = _config(synthetic_project)
        run(cfg)
        out_dir = Path(cfg.output_dir) / cfg.run_name

        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        assert set(metrics["history"]) == {"0"}

        (row,) = metrics["schedule"]
        assert row["fold"] == 0
        assert row["epochs_configured"] == cfg.train.epochs
        # patience (5) exceeds epochs (2), so this run cannot stop early.
        assert row["stopped_early"] is False
        assert row["last_epoch"] == cfg.train.epochs - 1
        assert 0.0 < row["lr_at_best_over_peak"] <= 1.0

    def test_schedule_section_renders_only_when_history_exists(
        self, synthetic_project: dict, tmp_path: Path
    ) -> None:
        """Older runs have no history_fold*.json; the report must degrade, not raise."""
        from datpd.evaluate import _schedule_section
        from datpd.train import run

        assert _schedule_section(tmp_path) == [], "no metrics.json at all"

        (tmp_path / "metrics.json").write_text(json.dumps({"overall": {}}), encoding="utf-8")
        assert _schedule_section(tmp_path) == [], "metrics.json predating the schedule block"

        cfg = _config(synthetic_project)
        run(cfg)
        lines = _schedule_section(Path(cfg.output_dir) / cfg.run_name)
        assert any("Training schedule" in line for line in lines)
        assert any("lr at best / peak" in line for line in lines)

    def test_schedule_section_survives_a_run_that_never_finished(
        self, synthetic_project: dict
    ) -> None:
        """The case the per-epoch banking exists for, and the one it first failed on.

        `run_cv` writes metrics.json only after the whole fold loop, so a run interrupted
        mid-fold has every curve on disk and nothing to describe them. Reading the schedule out
        of metrics.json therefore went blind exactly when the history mattered most - which is
        what happened to `baseline_anneal`, whose fold 1 died four epochs in.
        """
        from datpd.evaluate import _schedule_section
        from datpd.train import run

        cfg = _config(synthetic_project)
        run(cfg)
        run_dir = Path(cfg.output_dir) / cfg.run_name

        (run_dir / "metrics.json").unlink()
        assert (run_dir / "history_fold0.json").exists(), "the curve must outlive metrics.json"

        lines = _schedule_section(run_dir)
        assert any("Training schedule" in line for line in lines), "went blind without config"
        # Unknowable without the config, and reported as unknown rather than guessed.
        assert any("| ? | ? |" in line for line in lines)
        # A single-fold run must say so: its numbers are not a five-fold OOF.
        assert any("Only fold 0 trained" in line for line in lines)

    def test_the_cache_is_written_and_reused(self, synthetic_project: dict) -> None:
        """Second construction must hit the cache rather than re-resampling every volume."""
        from datpd.data.dataset import DaTScanDataset

        cfg = _config(synthetic_project)
        df = pd.read_csv(cfg.data.folds_csv)

        ds = DaTScanDataset(df, cfg.data, cfg.aug, train=False)
        _ = ds[0]

        cached = list(Path(cfg.data.cache_dir).rglob("*.prep"))
        assert cached, "preprocessing cache was never populated"

        # Loading again must return an identical array, not a re-augmented or re-resampled one.
        first = ds[0][0].numpy()
        second = ds[0][0].numpy()
        np.testing.assert_array_equal(first, second)

    def test_eval_pass_is_deterministic(self, synthetic_project: dict) -> None:
        """OOF predictions must be a fixed number, not a sample — `train=False` disables aug."""
        from datpd.data.dataset import DaTScanDataset

        cfg = _config(synthetic_project)
        df = pd.read_csv(cfg.data.folds_csv)
        ds = DaTScanDataset(df, cfg.data, cfg.aug, train=False)

        np.testing.assert_array_equal(ds[3][0].numpy(), ds[3][0].numpy())
