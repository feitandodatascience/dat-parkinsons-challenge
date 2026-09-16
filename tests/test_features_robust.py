"""Tests for the features added to `datpd.features`, on SYNTHETIC volumes only.

Two things are worth pinning here, and they are different things.

First, **finiteness**. `feature_names()` derives the column order by extracting features from an
all-zero volume, and `submission_src/main.py` refuses to load the boosters if that order does not
match the one they were fitted on. A feature that returns nan on a degenerate region does not
degrade the model — it stops the submission. The degenerate cases are all-zero, constant
positive, and constant NEGATIVE: the volume is reference-normalised and therefore signed, and a
constant negative volume makes `threshold * box_max` larger than every voxel, so every
supra-threshold mask comes out empty. That path has to be exercised, not assumed.

Second, **direction**. A feature that does not respond to the anatomy it claims to measure is
worse than no feature, because it looks like evidence. Each phantom below perturbs exactly one
thing — laterality, the anterior-posterior gradient, overall uptake — and the assertions say
which features must move and which must stay put.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd import features as ft  # noqa: E402

SHAPE = (96, 96, 96)


def phantom(
    right_strength: float = 1.0,
    elongation: float = 3.0,
    gradient: float = 0.0,
    uptake: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    """A DaT-like volume: two lobes either side of centre on axis 0, on a NEGATIVE background.

    The background is negative on purpose. `preprocess(normalization="reference_region")`
    produces a signed, roughly zero-centred volume — background below zero, striatum above — and
    that sign change is precisely what breaks the ratio features this module had to repair. A
    phantom with a positive background would not exercise the bug.

    `right_strength` scales the second lobe (asymmetric dopaminergic loss). `elongation` stretches
    both lobes along axis 1 (the comma that degrades to a dot). `gradient` tilts intensity along
    axis 1, positive meaning the low-index — "anterior", by convention here — end stays brighter,
    which is the caudate-sparing pattern. `uptake` scales both lobes together.
    """
    rng = np.random.default_rng(seed)
    array = rng.normal(-0.3, 0.16, size=SHAPE).astype(np.float32)

    cx, cy, cz = (s // 2 for s in SHAPE)
    xx, yy, zz = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    tilt = 1.0 - gradient * ((np.arange(SHAPE[1]) - cy) / float(cy)).reshape(1, -1, 1)
    for offset, strength in ((-7, 1.0), (+7, right_strength)):
        blob = np.exp(
            -(
                ((xx - cx - offset) ** 2) / (2 * 3.0**2)
                + ((yy - cy) ** 2) / (2 * (3.0 * elongation) ** 2)
                + ((zz - cz) ** 2) / (2 * 3.0**2)
            )
        )
        array += (strength * uptake * blob * tilt).astype(np.float32)
    return array


DEGENERATE = {
    "zeros": np.zeros(SHAPE, dtype=np.float32),
    "constant_positive": np.full(SHAPE, 2.0, dtype=np.float32),
    "constant_negative": np.full(SHAPE, -1.0, dtype=np.float32),
    "constant_tiny": np.full(SHAPE, 1e-9, dtype=np.float32),
}


class TestFinite:
    @pytest.mark.parametrize("name", sorted(DEGENERATE))
    def test_every_feature_is_finite_on_a_degenerate_volume(self, name: str) -> None:
        values = ft.extract_features(DEGENERATE[name])
        bad = [key for key, value in values.items() if not np.isfinite(value)]
        assert not bad, f"{name}: {bad}"

    def test_every_feature_is_finite_on_a_phantom(self) -> None:
        values = ft.extract_features(phantom(right_strength=0.25, gradient=0.6))
        bad = [key for key, value in values.items() if not np.isfinite(value)]
        assert not bad

    def test_a_constant_negative_volume_leaves_every_mask_empty(self) -> None:
        # The case the guards exist for: the cut sits above every voxel, so component counts,
        # surfaces and centroids are all computed on nothing.
        values = ft.extract_features(DEGENERATE["constant_negative"])
        assert values["t50_cc_count"] == 0.0
        assert values["t50_sphericity"] == 0.0
        assert values["mcent01_mean"] == 0.0

    def test_the_column_order_is_the_same_for_every_volume(self) -> None:
        names = ft.feature_names()
        for volume in (*DEGENERATE.values(), phantom(), phantom(right_strength=0.1)):
            assert list(ft.extract_features(volume)) == names


class TestContract:
    def test_vector_matches_names_in_length_and_order(self) -> None:
        volume = phantom(right_strength=0.4, seed=7)
        names = ft.feature_names()
        values = ft.extract_features(volume)
        vector = ft.feature_vector(volume)
        assert len(vector) == len(names)
        assert np.allclose(vector, [values[name] for name in names])

    def test_feature_names_is_stable(self) -> None:
        assert ft.feature_names() == ft.feature_names()

    def test_extraction_is_deterministic_and_stateless(self) -> None:
        volume = phantom(right_strength=0.3, seed=11)
        first = ft.extract_features(volume)
        ft.extract_features(phantom(right_strength=0.9, seed=12))  # nothing may carry over
        assert ft.extract_features(volume) == first

    def test_the_original_columns_are_still_a_prefix(self) -> None:
        # Fitted artifacts index a column ORDER. New work appends; it must never insert.
        names = ft.feature_names()
        assert names[0] == "global_mean"
        assert names[60] == "grad21_diff"
        assert names.index("shell_p50") == 61


class TestRepairedContrasts:
    """The point of the repair: rank order in uptake, without a sign flip."""

    def test_striatal_contrast_falls_as_uptake_falls(self) -> None:
        values = [ft.extract_features(phantom(uptake=u))["sbr_z_p99"] for u in (1.0, 2.5, 5.0)]
        assert values == sorted(values)
        assert all(v > 0 for v in values)

    def test_the_difference_form_is_ordered_too(self) -> None:
        values = [ft.extract_features(phantom(uptake=u))["sbr_diff_peak"] for u in (1.0, 2.5, 5.0)]
        assert values == sorted(values)

    def test_the_robust_denominator_is_strictly_positive(self) -> None:
        for volume in (*DEGENERATE.values(), phantom()):
            assert ft.extract_features(volume)["shell_iqr"] > 0.0

    def test_robust_spread_never_returns_zero_or_negative(self) -> None:
        assert ft.robust_spread(np.array([])) > 0.0
        assert ft.robust_spread(np.zeros(100)) > 0.0
        assert ft.robust_spread(np.full(100, -5.0)) > 0.0


class TestLaterality:
    def test_a_symmetric_phantom_scores_near_zero(self) -> None:
        values = ft.extract_features(phantom(right_strength=1.0))
        assert abs(values["rasym0_p99"]) < 0.5
        assert abs(values["rasym0_p95"]) < 0.5

    def test_losing_the_right_lobe_drives_asymmetry_toward_the_left(self) -> None:
        symmetric = ft.extract_features(phantom(right_strength=1.0))
        lopsided = ft.extract_features(phantom(right_strength=0.25))
        # Low index on axis 0 is the surviving side, so the signed difference must go positive.
        assert lopsided["rasym0_p99"] > symmetric["rasym0_p99"] + 1.0
        assert lopsided["rasym0_abs"] > symmetric["rasym0_abs"]

    def test_the_sign_follows_the_side_that_is_lost(self) -> None:
        right_lost = ft.extract_features(phantom(right_strength=0.25))
        left_lost = ft.extract_features(phantom(right_strength=0.25)[::-1].copy())
        assert right_lost["rasym0_p99"] > 0.0
        assert left_lost["rasym0_p99"] < 0.0
        assert right_lost["rasym0_abs"] == pytest.approx(left_lost["rasym0_abs"], abs=0.5)

    def test_the_worse_hemisphere_is_what_hemi_min_reports(self) -> None:
        symmetric = ft.extract_features(phantom(right_strength=1.0))
        lopsided = ft.extract_features(phantom(right_strength=0.25))
        assert lopsided["hemi0_min_p99"] < symmetric["hemi0_min_p99"]

    def test_asymmetry_does_not_leak_into_the_other_axes(self) -> None:
        lopsided = ft.extract_features(phantom(right_strength=0.25))
        assert abs(lopsided["rasym1_p99"]) < abs(lopsided["rasym0_p99"])
        assert abs(lopsided["rasym2_p99"]) < abs(lopsided["rasym0_p99"])


class TestGradient:
    def test_the_centroid_slides_toward_the_brighter_end(self) -> None:
        anterior = ft.extract_features(phantom(gradient=0.6))["mcent01_mean"]
        flat = ft.extract_features(phantom(gradient=0.0))["mcent01_mean"]
        posterior = ft.extract_features(phantom(gradient=-0.6))["mcent01_mean"]
        assert anterior < flat < posterior

    def test_the_supra_threshold_extent_follows_the_gradient(self) -> None:
        anterior = ft.extract_features(phantom(gradient=0.6))["msupra01_mean"]
        posterior = ft.extract_features(phantom(gradient=-0.6))["msupra01_mean"]
        assert anterior > 0.05 > -0.05 > posterior

    def test_the_quantile_contrast_follows_the_gradient(self) -> None:
        anterior = ft.extract_features(phantom(gradient=0.6))["mgrad01_mean"]
        flat = ft.extract_features(phantom(gradient=0.0))["mgrad01_mean"]
        posterior = ft.extract_features(phantom(gradient=-0.6))["mgrad01_mean"]
        assert anterior > flat > posterior
        assert abs(flat) < 0.1

    def test_a_flat_phantom_leaves_the_gradient_features_near_zero(self) -> None:
        values = ft.extract_features(phantom(gradient=0.0))
        assert abs(values["mcent01_mean"]) < 0.05
        assert abs(values["msupra01_mean"]) < 0.05


class TestTopology:
    def test_two_lobes_are_found_on_the_lateral_axis_only(self) -> None:
        volume = phantom()
        assert ft.two_lobe_score(volume, 0) == 2
        assert ft.two_lobe_score(volume, 1) == 1
        assert ft.two_lobe_score(volume, 2) == 1

    def test_the_lobe_count_is_a_feature_too(self) -> None:
        values = ft.extract_features(phantom())
        assert values["lobes0"] == 2.0
        assert values["lobes1"] == 1.0

    def test_losing_a_lobe_collapses_the_count_and_the_dip(self) -> None:
        both = ft.extract_features(phantom(right_strength=1.0))
        one = ft.extract_features(phantom(right_strength=0.2))
        assert both["lobes0"] == 2.0 and one["lobes0"] == 1.0
        assert both["lobedip0"] > one["lobedip0"]

    def test_the_second_component_vanishes_with_the_second_striatum(self) -> None:
        both = ft.extract_features(phantom(right_strength=1.0))
        one = ft.extract_features(phantom(right_strength=0.2))
        assert both["t50_cc_second_ratio"] > 0.5
        assert one["t50_cc_second_ratio"] < 0.1

    def test_a_dot_is_more_spherical_than_a_comma(self) -> None:
        comma = ft.extract_features(phantom(elongation=3.0))
        dot = ft.extract_features(phantom(elongation=1.0))
        assert dot["t50_sphericity"] > comma["t50_sphericity"]
        assert dot["t50_surface_ratio"] > comma["t50_surface_ratio"]

    def test_an_empty_mask_has_no_components_and_no_surface(self) -> None:
        empty = np.zeros((10, 10, 10), dtype=bool)
        assert ft.mask_components(empty) == (0.0, 0.0, 0.0)
        assert ft.mask_surface(empty) == (0.0, 0.0)

    def test_two_separated_blocks_are_two_components_of_equal_size(self) -> None:
        mask = np.zeros((20, 10, 10), dtype=bool)
        mask[2:6] = True
        mask[14:18] = True
        count, top_share, second_ratio = ft.mask_components(mask)
        assert count == 2.0
        assert top_share == pytest.approx(0.5)
        assert second_ratio == pytest.approx(1.0)


class TestExtentCurve:
    def test_the_bright_region_shrinks_as_the_cut_rises(self) -> None:
        values = ft.extract_features(phantom())
        assert values["t40_frac"] > values["t50_frac"] > values["t70_frac"] > values["t80_frac"]

    def test_the_decay_slope_is_negative_for_a_real_striatum(self) -> None:
        values = ft.extract_features(phantom())
        assert values["extent_slope"] < 0.0
        assert values["extent_log_40_80"] > 0.0


class TestMoments:
    def test_a_bright_striatum_puts_a_positive_tail_on_the_box_histogram(self) -> None:
        assert ft.extract_features(phantom())["box_skew"] > 0.5

    def test_moments_of_a_constant_region_are_zero_rather_than_nan(self) -> None:
        assert ft.standardised_moments(np.full(1000, 3.0)) == (0.0, 0.0)
        assert ft.standardised_moments(np.array([1.0, 2.0])) == (0.0, 0.0)


class TestHelpers:
    def test_split_centre_gives_two_halves_of_equal_size(self) -> None:
        for size in (32, 33):
            near, far = ft.split_centre(np.zeros((size, 4, 4)), 0)
            assert near.shape == far.shape == (size // 2, 4, 4)

    def test_split_centre_drops_only_the_middle_slice(self) -> None:
        region = np.arange(5).reshape(5, 1, 1).astype(float)
        near, far = ft.split_centre(region, 0)
        assert near.ravel().tolist() == [0.0, 1.0]
        assert far.ravel().tolist() == [3.0, 4.0]

    def test_safe_ratio_survives_a_zero_denominator(self) -> None:
        assert ft.safe_ratio(0.0, 0.0) == 0.0
        assert np.isfinite(ft.safe_ratio(1.0, 0.0))
        assert abs(ft.safe_ratio(1e30, 0.0)) <= ft.RATIO_CLIP

    def test_peak_distance_is_finite_on_a_flat_volume(self) -> None:
        euclidean, along = ft.peak_distance(np.zeros((33, 33, 33), dtype=np.float32), 0)
        assert np.isfinite(euclidean) and np.isfinite(along)
