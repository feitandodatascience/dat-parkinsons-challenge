"""Tests for scripts/07_tta.py, on SYNTHETIC volumes only.

The properties worth pinning are geometric, so none of this needs a trained model or a real
scan. The one that matters most is `test_no_view_flips_laterality`: the whole reason this TTA
set is hand-written rather than "the usual eight flips" is that a left-right flip would destroy
diagnostic asymmetry, and a regression there would be silent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("tta", REPO_ROOT / "scripts" / "07_tta.py")
assert _spec is not None and _spec.loader is not None
tta = importlib.util.module_from_spec(_spec)
sys.modules["tta"] = tta
_spec.loader.exec_module(tta)

IDENTITY = (0.0, (1, 2), (0, 0, 0), 1.0)


def asymmetric_volume(shape: tuple[int, int, int] = (32, 32, 32)) -> np.ndarray:
    """Two blobs along axis 0, the one on the low side brighter.

    Stands in for a DaT scan with a more-affected side. Any transform that mirrors axis 0 will
    move the bright blob to the other end, which is what the laterality test detects.
    """
    array = np.full(shape, 0.1, dtype=np.float32)
    cx, cy, cz = (s // 2 for s in shape)
    xx, yy, zz = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for offset, strength in ((-6, 1.0), (+6, 0.4)):
        blob = np.exp(
            -(((xx - cx - offset) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / (2 * 3.0**2))
        )
        array += (strength * blob).astype(np.float32)
    return array


def lateral_balance(array: np.ndarray) -> float:
    """Signed mass difference between the two halves of axis 0."""
    half = array.shape[0] // 2
    return float(array[:half].sum() - array[half:].sum())


class TestMakeView:
    def test_identity_is_exactly_a_no_op(self) -> None:
        """Bit-for-bit, not merely close.

        This is the path `submission_src/main.py` takes when TTA is switched off, so it must
        reproduce the un-augmented predictions exactly — otherwise "TTA disabled" would still
        perturb the shipped model, and a packaged submission would not match its own CV.
        """
        array = asymmetric_volume()
        out = tta.make_view(array, *IDENTITY)
        assert out.shape == array.shape
        assert np.array_equal(out, array)

    def test_identity_short_circuit_covers_every_no_op_spelling(self) -> None:
        for axes in ((0, 1), (0, 2), (1, 2)):
            array = asymmetric_volume()
            assert np.array_equal(tta.make_view(array, 0.0, axes, (0, 0, 0), 1.0), array)

    def test_shape_is_always_preserved(self) -> None:
        array = asymmetric_volume((24, 28, 32))
        for view in tta.VIEW_PRESETS["default"]:
            assert tta.make_view(array, *view).shape == array.shape

    def test_shift_moves_content_by_the_requested_amount(self) -> None:
        array = asymmetric_volume()
        shifted = tta.make_view(array, 0.0, (1, 2), (0, 3, 0), 1.0)
        # Content moves +3 along axis 1, so the interior should match the original offset back.
        np.testing.assert_allclose(shifted[:, 6:20, :], array[:, 3:17, :], atol=2e-2)

    def test_rotation_is_approximately_invertible(self) -> None:
        array = asymmetric_volume()
        there = tta.make_view(array, +6.0, (1, 2), (0, 0, 0), 1.0)
        back = tta.make_view(there, -6.0, (1, 2), (0, 0, 0), 1.0)
        # Interpolation blurs, so compare the interior loosely rather than exactly.
        assert np.corrcoef(back[8:24, 8:24, 8:24].ravel(), array[8:24, 8:24, 8:24].ravel())[
            0, 1
        ] > 0.99

    @pytest.mark.parametrize(
        "scale,expected_radius",
        [(0.87, 5), (1.0, 6), (1.15, 7)],
    )
    def test_scale_moves_a_feature_toward_or_away_from_the_centre(
        self, scale: float, expected_radius: int
    ) -> None:
        """Magnification pushes structure outward; minification pulls it in.

        Tracked as the displacement of a single delta, which states the geometry directly. An
        intensity-weighted moment over the whole volume does NOT work here and the reason is
        worth recording: the background occupies most of the voxels, so such a metric mostly
        measures how much bright mass exists relative to it, and magnifying a centrally-placed
        blob moves that metric the wrong way.
        """
        array = np.zeros((33, 33, 33), dtype=np.float32)
        array[16, 22, 16] = 1.0  # 6 voxels from the centre along axis 1

        out = tta.make_view(array, 0.0, (1, 2), (0, 0, 0), scale)
        peak = np.unravel_index(int(np.argmax(out)), out.shape)
        assert peak[1] - 16 == expected_radius

    def test_rejects_a_degenerate_rotation_plane(self) -> None:
        array = asymmetric_volume()
        with pytest.raises(ValueError, match="rotation plane"):
            tta.make_view(array, 5.0, (1, 1), (0, 0, 0), 1.0)

    def test_rejects_non_3d_input(self) -> None:
        with pytest.raises(ValueError, match="3D volume"):
            tta.make_view(np.zeros((8, 8)), *IDENTITY)


class TestLaterality:
    def test_no_view_flips_laterality(self) -> None:
        """Every shipped view must keep the brighter side on the side it started."""
        array = asymmetric_volume()
        reference = lateral_balance(array)
        assert reference > 0, "fixture should start with the low side brighter"

        for preset, views in tta.VIEW_PRESETS.items():
            for view in views:
                out = tta.make_view(array, *view)
                assert lateral_balance(out) > 0, f"{preset} view {view} mirrored the volume"

    def test_a_flip_would_be_caught(self) -> None:
        """Guards the guard: the detector must actually notice a mirrored volume."""
        array = asymmetric_volume()
        assert lateral_balance(np.flip(array, axis=0).copy()) < 0

    def test_views_preserve_asymmetry_magnitude(self) -> None:
        array = asymmetric_volume()
        reference = lateral_balance(array)
        for view in tta.VIEW_PRESETS["default"]:
            out = tta.make_view(array, *view)
            # Small perturbations should not wash out the asymmetry the model reads.
            assert lateral_balance(out) > 0.5 * reference


class TestPresets:
    def test_identity_view_is_first_in_every_preset(self) -> None:
        """`--views identity` must reproduce the original single-view predictions exactly."""
        for preset, views in tta.VIEW_PRESETS.items():
            assert views[0] == IDENTITY, f"{preset} does not start from the identity view"

    def test_presets_have_no_duplicate_views(self) -> None:
        for preset, views in tta.VIEW_PRESETS.items():
            assert len(set(views)) == len(views), f"{preset} repeats a view"

    def test_rotations_come_in_signed_pairs(self) -> None:
        """An unbalanced rotation set biases the average toward one orientation."""
        for preset, views in tta.VIEW_PRESETS.items():
            rotations = sorted(
                (deg, axes) for deg, axes, _, _ in views if deg
            )
            for deg, axes in rotations:
                assert (-deg, axes) in rotations, f"{preset} lacks the mirror of {deg} in {axes}"


class TestSmokeSampling:
    """`--limit` must draw a representative sample, not the first N rows.

    The first version took `.head(N)`. folds.csv inherits the label file's row order, so if that
    order carries any structure, each fold's first N rows are a systematic slice — and the smoke
    run's per-fold scores then differ for reasons unrelated to the model. That is exactly what
    a smoke run must not do, because it invites a real decision from a fake signal.
    """

    @staticmethod
    def _ordered_fold(n: int = 200) -> pd.DataFrame:
        """A fold whose row order is perfectly confounded with the label."""
        return pd.DataFrame(
            {
                "uid": [f"u{i:04d}" for i in range(n)],
                "is_pathologic": (np.arange(n) >= n // 2).astype(int),
            }
        )

    def test_head_would_have_been_unrepresentative(self) -> None:
        """Guards the guard: the fixture must actually expose the flaw being fixed."""
        fold = self._ordered_fold()
        assert fold.head(40)["is_pathologic"].mean() == 0.0
        assert fold["is_pathologic"].mean() == pytest.approx(0.5)

    def test_sampling_recovers_the_base_rate(self) -> None:
        fold = self._ordered_fold()
        drawn = fold.sample(n=40, random_state=42)
        assert 0.25 < drawn["is_pathologic"].mean() < 0.75

    def test_sampling_is_reproducible_and_differs_per_fold(self) -> None:
        """Same seed and fold reproduces; different folds must not draw identical positions."""
        fold = self._ordered_fold()
        first = fold.sample(n=40, random_state=42 + 0)
        again = fold.sample(n=40, random_state=42 + 0)
        other = fold.sample(n=40, random_state=42 + 1)
        assert list(first["uid"]) == list(again["uid"])
        assert list(first["uid"]) != list(other["uid"])

    def test_limit_larger_than_the_fold_is_safe(self) -> None:
        fold = self._ordered_fold(n=30)
        drawn = fold.sample(n=min(40, len(fold)), random_state=42)
        assert len(drawn) == 30


class TestShippedImplementationMatches:
    """`submission_src/main.py` must transform volumes identically to the experiment script.

    The OOF number that justifies shipping TTA is measured by scripts/07_tta.py. If main.py's
    copy of the view maths drifts from it by even a little, the shipped model is not the one
    that was validated — and nothing would fail loudly, the leaderboard score would just be
    worse than CV for no visible reason.
    """

    @staticmethod
    def _main_module():
        spec = importlib.util.spec_from_file_location(
            "submission_main", REPO_ROOT / "submission_src" / "main.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["submission_main"] = module
        spec.loader.exec_module(module)
        return module

    def test_view_maths_is_identical(self) -> None:
        main = self._main_module()
        array = asymmetric_volume()
        for view in tta.VIEW_PRESETS["default"]:
            np.testing.assert_array_equal(
                main._make_view(array, *view),
                tta.make_view(array, *view),
                err_msg=f"main.py and 07_tta.py disagree on view {view}",
            )

    def test_tta_is_off_by_default(self) -> None:
        """A packaged submission must not gain TTA unless it was asked for."""
        main = self._main_module()
        assert main.TTA_VIEWS == (IDENTITY,)

    def test_default_view_leaves_the_volume_untouched(self) -> None:
        main = self._main_module()
        array = asymmetric_volume()
        assert np.array_equal(main._make_view(array, *main.TTA_VIEWS[0]), array)


class TestPredictViews:
    def test_averages_in_logit_space(self) -> None:
        """A model returning a constant logit must yield exactly that logit, not its mean prob."""

        class ConstantLogit(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.full((x.shape[0], 1), 2.0)

        array = asymmetric_volume()
        out = tta.predict_views(
            ConstantLogit(), array, tta.VIEW_PRESETS["default"], torch.device("cpu")
        )
        assert out == pytest.approx(2.0)

    def test_one_forward_pass_per_view(self) -> None:
        class Counting(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seen = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.seen += x.shape[0]
                return torch.zeros((x.shape[0], 1))

        model = Counting()
        views = tta.VIEW_PRESETS["default"]
        tta.predict_views(model, asymmetric_volume(), views, torch.device("cpu"))
        assert model.seen == len(views)

    def test_identity_preset_matches_a_plain_forward_pass(self) -> None:
        class Sum(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.flatten(1).sum(dim=1, keepdim=True) / 1000.0

        array = asymmetric_volume()
        model = Sum()
        via_tta = tta.predict_views(
            model, array, tta.VIEW_PRESETS["identity"], torch.device("cpu")
        )
        direct = float(model(torch.from_numpy(array).float()[None, None]).item())
        assert via_tta == pytest.approx(direct, rel=1e-5)
