"""Preprocessing must work at both extremes of the real field-of-view range.

The dataset spans 176 mm to 630 mm per axis. A head is roughly 180 mm across, so the fraction of
the array that is anatomy runs from nearly all of it down to about **one percent** — and the
630 mm acquisitions are a third of the dataset, not an edge case.

The failure this guards against is silent: with a percentile-based foreground test, the centre of
mass of a wide-field scan lands in the middle of the array (because the percentile selects air),
`crop: com` degenerates into `crop: center`, and an off-centre head gets its striatum clipped on
one side. Nothing errors; the ablation just reports that COM cropping "doesn't help".

Synthetic volumes only — no competition data.
"""

from __future__ import annotations

import numpy as np
import pytest

from datpd.data.io import Volume
from datpd.data.preprocess import center_of_mass_crop, preprocess


def wide_field_volume(
    shape: tuple[int, int, int] = (256, 256, 256),
    head_radius_vox: int = 36,
    offset: tuple[int, int, int] = (0, 0, 0),
    spacing: tuple[float, float, float] = (2.46, 2.46, 2.46),
    seed: int = 0,
) -> Volume:
    """A small bright head floating in a large, nearly empty field of view.

    At the default size the head is ~1% of the array, which is the real 630 mm case.
    """
    rng = np.random.default_rng(seed)
    array = rng.normal(2.0, 0.5, size=shape).astype(np.float32).clip(0)  # near-air background

    centre = [s // 2 + o for s, o in zip(shape, offset)]
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist2 = (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2

    # A hard-edged head rather than a Gaussian blob: a Gaussian's tail reaches into a centred
    # window even when its peak is far outside, which would blur the very distinction these
    # tests exist to make.
    array += (800.0 * (dist2 <= head_radius_vox**2)).astype(np.float32)
    return Volume(array=array, spacing_mm=spacing, affine=np.eye(4), uid="wide")


class TestCentreOfMassOnWideFields:
    def test_head_occupies_about_one_percent(self) -> None:
        """Guard the premise of these tests, so they stay meaningful if the fixture changes."""
        vol = wide_field_volume()
        bright = (vol.array > 0.1 * np.percentile(vol.array, 99.5)).mean()
        assert bright < 0.05

    @pytest.mark.parametrize("offset", [(0, 0, 0), (40, -30, 25), (-55, 45, -20)])
    def test_crop_follows_an_off_centre_head(self, offset: tuple[int, int, int]) -> None:
        """The cropped window must actually contain the head, wherever it sits."""
        vol = wide_field_volume(offset=offset)
        out = center_of_mass_crop(vol.array, (96, 96, 96))

        assert out.shape == (96, 96, 96)
        # The head is far brighter than the background, so if the crop found it the window's
        # peak is near the head peak rather than at background level.
        assert out.max() > 0.5 * float(vol.array.max())

    def test_centre_crop_would_have_missed_it(self) -> None:
        """Establishes that COM cropping is doing real work, not agreeing with `center`."""
        from datpd.data.preprocess import center_crop_or_pad

        # Far enough out that the head lies entirely outside the central 96^3 window.
        vol = wide_field_volume(offset=(90, -90, 88))
        centred = center_crop_or_pad(vol.array, (96, 96, 96))
        com = center_of_mass_crop(vol.array, (96, 96, 96))

        assert centred.max() < 50.0, "fixture broken: head leaked into the centred window"
        assert com.max() > 10.0 * max(centred.max(), 1e-6)

    def test_full_pipeline_is_finite_on_a_wide_field(self) -> None:
        vol = wide_field_volume(offset=(30, 30, -30))
        out = preprocess(vol, (2.0, 2.0, 2.0), (96, 96, 96), "reference_region", "com")
        assert out.shape == (96, 96, 96)
        assert np.isfinite(out).all()


class TestNarrowFields:
    def test_thin_slab_pads_to_target(self) -> None:
        """z ranges down to 108 mm on the real data; the pipeline must pad, not fail."""
        vol = wide_field_volume(shape=(128, 128, 36), head_radius_vox=30, spacing=(3.9, 3.9, 3.9))
        out = preprocess(vol, (2.0, 2.0, 2.0), (96, 96, 96), "reference_region", "com")
        assert out.shape == (96, 96, 96)
        assert np.isfinite(out).all()
