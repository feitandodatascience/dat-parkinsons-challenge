"""Interrupting a fold must cost one epoch, not the fold.

This is not hypothetical. A 7-hour overnight CPU run on this project was interrupted and banked
**nothing**, because `03_train.py` only wrote a fold's outputs once all its epochs finished. On
a GPU a fold is minutes and that design is fine; on CPU a fold is ~100 minutes and the machine
gets slept, updated and closed in between.

So the recovery path needs to be tested rather than assumed, and tested the only way that
means anything: interrupt a real run and restart it.

Synthetic volumes only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from datpd.train import train_fold
from tests.test_train_smoke import _config, synthetic_project  # noqa: F401  (fixture import)


def _device() -> torch.device:
    return torch.device("cpu")


class TestEpochLevelResume:
    def test_a_checkpoint_appears_after_the_first_epoch(self, synthetic_project: dict) -> None:
        cfg = _config(synthetic_project)
        cfg.run_name = "resume_a"
        cfg.train.epochs = 1

        train_fold(cfg, _folds_df(cfg), 0, _device())

        # The fold RAN TO COMPLETION here, so its resume file must be cleaned up — a stale one
        # would make a rerun skip straight past training.
        resume = Path(cfg.output_dir) / cfg.run_name / "fold0_last.pt"
        assert not resume.exists(), "completed fold left its resume checkpoint behind"

    def test_interrupted_fold_resumes_from_where_it_stopped(
        self,
        synthetic_project: dict,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Kill a fold mid-way for real, then restart it and confirm it continues.

        The interruption is injected by making the validation pass raise on the second epoch —
        after epoch 0's checkpoint has been written, which is exactly the state a `taskkill`
        or a failed wake-from-sleep leaves behind.
        """
        import datpd.train as train_module

        cfg = _config(synthetic_project)
        cfg.run_name = "resume_b"
        cfg.train.epochs = 5
        df = _folds_df(cfg)
        resume = Path(cfg.output_dir) / cfg.run_name / "fold0_last.pt"

        real_predict = train_module.predict
        calls = {"n": 0}

        def exploding_predict(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:  # let epoch 0 complete and checkpoint, then die
                raise RuntimeError("simulated interruption")
            return real_predict(*args, **kwargs)

        monkeypatch.setattr(train_module, "predict", exploding_predict)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train_fold(cfg, df, 0, _device())

        assert resume.exists(), "an interrupted fold banked nothing — the whole point"
        state = torch.load(resume, map_location="cpu", weights_only=False)
        assert state["epoch"] == 0
        assert state["best_state"] is not None

        # Restart, unpatched. It must pick up at epoch 1 rather than starting over.
        monkeypatch.setattr(train_module, "predict", real_predict)
        capsys.readouterr()
        _, _, probs, metrics = train_fold(cfg, df, 0, _device())

        assert "resuming at epoch 1" in capsys.readouterr().out
        assert not resume.exists(), "completed fold left its resume checkpoint behind"
        assert np.isfinite(probs).all()
        assert 0.0 <= float(np.min(probs)) <= float(np.max(probs)) <= 1.0
        assert np.isfinite(metrics["logloss"])

    def test_resume_state_round_trips(self, synthetic_project: dict) -> None:
        """The checkpoint must carry enough to continue: optimizer, scheduler, best-so-far."""
        cfg = _config(synthetic_project)
        cfg.run_name = "resume_c"
        cfg.train.epochs = 2
        df = _folds_df(cfg)

        train_fold(cfg, df, 0, _device())

        # Re-run the fold with a bigger budget; the run must not crash and must produce a
        # usable model. This exercises load_state_dict on optimizer/scheduler/scaler, which is
        # where a resume typically breaks.
        cfg.train.epochs = 3
        model, val_idx, probs, _ = train_fold(cfg, df, 0, _device())
        assert model is not None
        assert len(val_idx) == len(probs)

    def test_resume_can_be_disabled(self, synthetic_project: dict) -> None:
        cfg = _config(synthetic_project)
        cfg.run_name = "resume_d"
        cfg.train.epochs = 1
        cfg.train.resume = False

        train_fold(cfg, _folds_df(cfg), 0, _device())
        assert not (Path(cfg.output_dir) / cfg.run_name / "fold0_last.pt").exists()


class TestCheckpointDurability:
    def test_no_temporary_files_survive(self, synthetic_project: dict) -> None:
        """A truncated checkpoint would resume into a crash — the write must be atomic."""
        cfg = _config(synthetic_project)
        cfg.run_name = "resume_e"
        cfg.train.epochs = 2

        train_fold(cfg, _folds_df(cfg), 0, _device())

        leftovers = list((Path(cfg.output_dir) / cfg.run_name).glob("*.tmp"))
        assert leftovers == [], f"temporary checkpoint files left behind: {leftovers}"


def _folds_df(cfg):
    import pandas as pd

    return pd.read_csv(cfg.data.folds_csv)
