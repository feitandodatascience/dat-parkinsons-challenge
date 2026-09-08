"""Pipeline tests on SYNTHETIC volumes only.

These exist so the preprocessing and model code can be exercised, and their output read by
Claude, without any competition data being involved. Never point these at `data/`.

The synthetic volumes imitate the properties that matter: variable shape, variable spacing,
uint16 range, and a bright bilateral blob standing in for striatal uptake.
"""

from __future__ import annotations

import numpy as np
import pytest

from datpd.calibrate import TemperatureScaler, choose_calibrator
from datpd.data.io import Volume
from datpd.data.preprocess import (
    center_crop_or_pad,
    center_of_mass_crop,
    normalize_intensity,
    preprocess,
    resample_to_spacing,
)
from datpd.data.splits import acquisition_signature, add_folds
from datpd.utils.metrics import baseline_logloss, competition_logloss, summarize


def make_synthetic_volume(
    shape: tuple[int, int, int] = (142, 142, 112),
    spacing: tuple[float, float, float] = (2.46, 2.46, 2.46),
    pathologic: bool = False,
    seed: int = 0,
) -> Volume:
    """A fake DaT-like scan: noisy background plus two bright putamen-ish blobs."""
    rng = np.random.default_rng(seed)
    array = rng.normal(200, 30, size=shape).astype(np.float32).clip(0)

    cz, cy, cx = (s // 2 for s in shape)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for offset, strength in ((-12, 1.0), (12, 0.35 if pathologic else 1.0)):
        blob = np.exp(
            -(((zz - cz) ** 2 + (yy - cy - offset) ** 2 + (xx - cx) ** 2) / (2 * 6.0**2))
        )
        array += (2000 * strength * blob).astype(np.float32)

    return Volume(array=array, spacing_mm=spacing, affine=np.eye(4), uid=f"synth{seed}")


class TestPreprocess:
    def test_resample_changes_extent_not_physical_size(self) -> None:
        vol = make_synthetic_volume(shape=(64, 64, 64), spacing=(4.0, 4.0, 4.0))
        out = resample_to_spacing(vol, (2.0, 2.0, 2.0))
        assert out.shape == (128, 128, 128)

    def test_resample_is_noop_at_matching_spacing(self) -> None:
        vol = make_synthetic_volume(shape=(32, 32, 32), spacing=(2.0, 2.0, 2.0))
        out = resample_to_spacing(vol, (2.0, 2.0, 2.0))
        assert out is vol.array

    @pytest.mark.parametrize(
        "shape", [(142, 142, 112), (128, 128, 128), (64, 64, 40), (200, 200, 200)]
    )
    def test_crop_or_pad_always_reaches_target(self, shape: tuple[int, int, int]) -> None:
        array = np.zeros(shape, dtype=np.float32)
        assert center_crop_or_pad(array, (96, 96, 96)).shape == (96, 96, 96)

    def test_com_crop_reaches_target(self) -> None:
        vol = make_synthetic_volume(shape=(120, 120, 100), spacing=(2.0, 2.0, 2.0))
        assert center_of_mass_crop(vol.array, (96, 96, 96)).shape == (96, 96, 96)

    @pytest.mark.parametrize("method", ["reference_region", "zscore_brain", "max"])
    def test_normalisation_is_finite_and_scale_invariant(self, method: str) -> None:
        vol = make_synthetic_volume(seed=1)
        a = normalize_intensity(vol.array, method)
        b = normalize_intensity(vol.array * 7.3, method)  # simulate scanner gain
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "shape,spacing",
        [((142, 142, 112), (2.46, 2.46, 2.46)), ((128, 128, 128), (3.895, 3.895, 3.895))],
    )
    def test_full_pipeline_yields_fixed_shape(self, shape, spacing) -> None:
        """The whole point: heterogeneous inputs, one tensor shape out."""
        vol = make_synthetic_volume(shape=shape, spacing=spacing)
        out = preprocess(vol, (2.0, 2.0, 2.0), (96, 96, 96))
        assert out.shape == (96, 96, 96)
        assert np.isfinite(out).all()


class TestSplits:
    def test_signature_is_stable_under_float_noise(self) -> None:
        """Header floats carry representation noise; the fingerprint must ignore it."""
        a = acquisition_signature((128, 128, 128), (3.895, 3.895, 3.895))
        b = acquisition_signature((128, 128, 128), (3.8950001, 3.8949998, 3.895))
        assert a == b

    def test_signature_separates_genuinely_different_protocols(self) -> None:
        a = acquisition_signature((142, 142, 112), (2.46, 2.46, 2.46))
        b = acquisition_signature((128, 128, 128), (3.895, 3.895, 3.895))
        assert a != b

    def test_signature_ignores_slice_count(self) -> None:
        """Slice count follows head coverage, which is per-patient, not per-centre.

        On the real data one 3.9 mm scanner emitted z = 33, 35, 36 ... 50. Including z turned
        that single centre into twenty "centres", and grouped CV silently degenerated into
        stratified CV — protection that looks applied but isn't.
        """
        signatures = {
            acquisition_signature((128, 128, z), (3.895, 3.895, 3.9))
            for z in (33, 36, 41, 45, 50)
        }
        assert len(signatures) == 1

    def test_signature_absorbs_reconstruction_zoom_jitter(self) -> None:
        """Same family, spacing jittering in the third decimal, must be one group."""
        signatures = {
            acquisition_signature((128, 128, 45), (xy, xy, 3.0))
            for xy in (1.503, 1.509, 1.523, 1.530, 1.537)
        }
        assert len(signatures) == 1

    def test_signature_keeps_distinct_matrices_apart(self) -> None:
        """Same voxel size, different reconstruction matrix, is a different protocol."""
        assert acquisition_signature((256, 256, 256), (2.46, 2.46, 2.46)) != (
            acquisition_signature((128, 128, 128), (2.46, 2.46, 2.46))
        )

    def test_grouped_folds_tolerate_one_dominant_group(self) -> None:
        """A third of the real data shares one signature; it must land whole in one fold.

        Uneven fold sizes are the honest consequence of not splitting a centre. What must not
        happen is an empty fold, or that group being spread across folds.
        """
        import pandas as pd

        rng = np.random.default_rng(0)
        groups = ["big"] * 450 + list(rng.choice([f"g{i}" for i in range(11)], 900))
        df = pd.DataFrame(
            {
                "uid": [f"u{i}" for i in range(len(groups))],
                "is_pathologic": rng.integers(0, 2, len(groups)),
                "acquisition_group": groups,
            }
        )
        out = add_folds(df, n_splits=5)
        assert set(out["fold_group"].unique()) == {0, 1, 2, 3, 4}
        assert (out.groupby("acquisition_group")["fold_group"].nunique() == 1).all()

    def test_grouped_folds_do_not_leak_groups(self) -> None:
        import pandas as pd

        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "uid": [f"u{i}" for i in range(300)],
                "is_pathologic": rng.integers(0, 2, 300),
                "acquisition_group": rng.choice(["a", "b", "c", "d", "e", "f"], 300),
            }
        )
        out = add_folds(df, n_splits=5)
        assert set(out["fold_group"].unique()) == {0, 1, 2, 3, 4}
        # No acquisition group may appear in more than one fold.
        per_group_folds = out.groupby("acquisition_group")["fold_group"].nunique()
        assert (per_group_folds == 1).all()


class TestMetrics:
    def test_prior_is_the_baseline_to_beat(self) -> None:
        y = np.array([0] * 700 + [1] * 300)
        assert baseline_logloss(y) == pytest.approx(0.6109, abs=1e-3)

    def test_confident_and_wrong_is_expensive(self) -> None:
        """The core intuition of log loss: one confident error outweighs a hundred good ones.

        100 correct predictions at p=0.9 cost 0.105 each. Adding a SINGLE confidently-wrong
        prediction (p=0.999999 on a negative) more than doubles the mean loss, because that
        one sample contributes ~13.8 on its own. This is why clipping matters.
        """
        y = np.array([1] * 100)
        good = competition_logloss(y, np.full(100, 0.9))
        with_one_disaster = competition_logloss(
            np.append(y, 0), np.append(np.full(100, 0.9), 0.999999)
        )
        assert with_one_disaster > 2 * good

    def test_clipping_caps_the_damage(self) -> None:
        """A tighter clip trades a small tax on good predictions for a bounded worst case."""
        y = np.array([0])
        assert competition_logloss(y, np.array([0.999999]), eps=1e-6) > 13.0
        assert competition_logloss(y, np.array([0.999999]), eps=0.02) < 4.0

    def test_summarize_reports_the_full_block(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        p = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 200), 0.01, 0.99)
        out = summarize(y, p)
        assert set(out) == {
            "logloss", "auroc", "ece", "baseline_logloss", "n", "positive_rate"
        }


class TestCalibration:
    def test_temperature_scaling_fixes_overconfidence(self) -> None:
        """An overconfident-but-well-ranked model should gain from calibration."""
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 2000)
        logits = (y * 2.0 - 1.0) * rng.uniform(0.5, 2.0, 2000) + rng.normal(0, 0.7, 2000)
        overconfident = 1 / (1 + np.exp(-logits * 4.0))  # blown up on purpose

        before = competition_logloss(y, overconfident)
        scaler = TemperatureScaler().fit(y, overconfident)
        after = competition_logloss(y, scaler.transform(overconfident))

        assert after < before
        assert scaler.temperature > 1.0  # softening, as expected

    def test_choose_calibrator_never_picks_a_worse_option(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 1000)
        p = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 1000), 0.01, 0.99)
        name, _, scores = choose_calibrator(y, p)
        assert scores[name] == min(scores.values())


class TestModel:
    def test_forward_produces_one_logit_per_sample(self) -> None:
        torch = pytest.importorskip("torch")
        from datpd.models.resnet3d import ResNet3D

        model = ResNet3D()
        out = model(torch.randn(2, 1, 96, 96, 96))
        assert out.shape == (2,)

    def test_eval_mode_is_batch_size_independent(self) -> None:
        """GroupNorm means a sample's prediction cannot depend on its batch-mates.

        This is what makes per-sample independence (a competition rule) structurally true
        rather than a promise. If this test ever fails, the submission is invalid.
        """
        torch = pytest.importorskip("torch")
        from datpd.models.resnet3d import ResNet3D

        model = ResNet3D().eval()
        x = torch.randn(4, 1, 64, 64, 64)
        with torch.no_grad():
            batched = model(x)
            individually = torch.cat([model(x[i : i + 1]) for i in range(4)])
        torch.testing.assert_close(batched, individually, rtol=1e-4, atol=1e-4)
