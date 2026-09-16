"""Rich augmentation, on synthetic volumes only.

The properties worth pinning are the ones whose failure would be silent: an augmentation that
quietly does nothing, one that mirrors laterality, one that produces a label outside [0, 1], or
one that costs so much per sample that a 16-hour run becomes a 24-hour one. Each of those is a
test below rather than something noticed afterwards in a training curve.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from datpd.augment import (
    RichAugDataset,
    count_noise,
    elastic_warp,
    psf_blur,
    signed_gamma,
    wrap_if_enabled,
)
from datpd.config import AugConfig

SHAPE = (96, 96, 96)


def striatum_phantom(asymmetric: bool = False) -> np.ndarray:
    """A signed, reference-region-normalised volume with two bright blobs.

    Matches the real preprocessed distribution closely enough for these tests: background below
    zero, striatum above it, peak around 6.
    """
    rng = np.random.default_rng(0)
    volume = rng.normal(-0.5, 0.3, size=SHAPE).astype(np.float32)
    zz, yy, xx = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    for side, amplitude in ((-9, 6.5), (+9, 3.0 if asymmetric else 6.5)):
        d2 = (zz - (48 + side)) ** 2 + (yy - 48) ** 2 + (xx - 48) ** 2
        volume += (amplitude * np.exp(-d2 / 40.0)).astype(np.float32)
    return volume


class StubDataset:
    """Stands in for DaTScanDataset, with the contract observed from the real one."""

    def __init__(self, n: int = 8, labels: list[float] | None = None) -> None:
        self.n = n
        self.labels = labels if labels is not None else [float(i % 2) for i in range(n)]
        self._volume = striatum_phantom()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self._volume.copy())[None],
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


class TestOpsAreIdentityWhenOff:
    """Every knob at its default must be an exact no-op, or the defaults are not defaults."""

    def test_elastic_zero_alpha(self) -> None:
        volume = striatum_phantom()
        out = elastic_warp(volume, 0.0, np.random.default_rng(0))
        assert out is volume

    def test_blur_zero_sigma(self) -> None:
        volume = striatum_phantom()
        assert psf_blur(volume, 0.0) is volume

    def test_gamma_unity(self) -> None:
        volume = striatum_phantom()
        assert signed_gamma(volume, 1.0) is volume

    def test_count_noise_zero(self) -> None:
        volume = striatum_phantom()
        assert count_noise(volume, 0.0, np.random.default_rng(0)) is volume

    def test_wrapper_returns_base_untouched(self) -> None:
        base = StubDataset()
        assert wrap_if_enabled(base, AugConfig()) is base


class TestOpsDoSomething:
    """The mirror image of the above: switched on, each must actually change the volume."""

    @pytest.mark.parametrize(
        "apply",
        [
            lambda v: elastic_warp(v, 3.0, np.random.default_rng(1)),
            lambda v: psf_blur(v, 1.0),
            lambda v: signed_gamma(v, 1.2),
            lambda v: count_noise(v, 0.1, np.random.default_rng(1)),
        ],
    )
    def test_changes_the_volume_and_stays_finite(self, apply) -> None:
        volume = striatum_phantom()
        out = apply(volume)
        assert out.shape == volume.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert not np.allclose(out, volume)


class TestSignedGamma:
    def test_is_odd_and_fixes_zero(self) -> None:
        x = np.array([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=np.float32)
        out = signed_gamma(x, 1.3)
        assert out[2] == 0.0
        assert np.allclose(out, -out[::-1])

    def test_is_monotone(self) -> None:
        x = np.linspace(-5.0, 7.0, 501, dtype=np.float32)
        for gamma in (0.7, 1.4):
            assert np.all(np.diff(signed_gamma(x, gamma)) > 0)

    def test_never_produces_nan_on_the_negative_half(self) -> None:
        """A plain x ** gamma would return NaN here. That is the bug this function exists for."""
        x = np.linspace(-3.0, -0.01, 50, dtype=np.float32)
        assert np.isfinite(signed_gamma(x, 1.15)).all()


class TestLateralityIsPreserved:
    """No augmentation may act like a left-right flip. Asymmetry is the diagnostic signal."""

    @pytest.mark.parametrize(
        "apply",
        [
            lambda v: elastic_warp(v, 3.0, np.random.default_rng(2)),
            lambda v: psf_blur(v, 1.0),
            lambda v: signed_gamma(v, 1.2),
            lambda v: count_noise(v, 0.05, np.random.default_rng(2)),
        ],
    )
    def test_output_resembles_the_input_not_its_mirror(self, apply) -> None:
        volume = striatum_phantom(asymmetric=True)
        out = apply(volume)
        mirrored = volume[::-1]

        def corr(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

        assert corr(out, volume) > corr(out, mirrored)

    def test_asymmetry_survives_the_full_pipeline(self) -> None:
        """The bright-side difference must not be washed out by the perturbations."""
        cfg = AugConfig(elastic_alpha=3.0, blur_sigma_max=1.0, gamma_range=(0.9, 1.1),
                        count_noise=0.05, rich_prob=1.0)
        ds = RichAugDataset(StubDataset(), cfg, seed=0)
        ds.base._volume = striatum_phantom(asymmetric=True)  # type: ignore[attr-defined]
        out = ds[0][0].numpy()[0]
        left, right = out[:48].max(), out[48:].max()
        assert left > right * 1.3, "the asymmetric phantom stopped being asymmetric"


class TestWrapperContract:
    def test_shape_dtype_and_label(self) -> None:
        cfg = AugConfig(elastic_alpha=2.0, blur_sigma_max=0.8, rich_prob=1.0)
        ds = RichAugDataset(StubDataset(), cfg, seed=0)
        x, y = ds[0]
        assert x.shape == (1, *SHAPE)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
        assert y.ndim == 0
        assert torch.isfinite(x).all()

    def test_length_passes_through(self) -> None:
        ds = RichAugDataset(StubDataset(n=13), AugConfig(elastic_alpha=1.0))
        assert len(ds) == 13

    def test_successive_reads_differ(self) -> None:
        """Seeding per index instead of per worker would make this pass identical volumes."""
        ds = RichAugDataset(StubDataset(), AugConfig(elastic_alpha=3.0, rich_prob=1.0), seed=0)
        assert not torch.allclose(ds[0][0], ds[0][0])

    def test_same_seed_reproduces(self) -> None:
        cfg = AugConfig(elastic_alpha=3.0, blur_sigma_max=1.0, rich_prob=1.0)
        a = RichAugDataset(StubDataset(), cfg, seed=7)[0][0]
        b = RichAugDataset(StubDataset(), cfg, seed=7)[0][0]
        assert torch.allclose(a, b)


class TestMixup:
    def test_label_lands_between_the_two_and_stays_in_range(self) -> None:
        cfg = AugConfig(mixup_alpha=0.4, mixup_prob=1.0)
        ds = RichAugDataset(StubDataset(labels=[0.0] * 4 + [1.0] * 4), cfg, seed=3)
        labels = [float(ds[i][1]) for i in range(8)]
        assert all(0.0 <= v <= 1.0 for v in labels)
        assert any(0.0 < v < 1.0 for v in labels), "mixup never actually mixed"

    def test_off_by_default_keeps_labels_hard(self) -> None:
        ds = RichAugDataset(StubDataset(), AugConfig(elastic_alpha=2.0), seed=3)
        assert all(float(ds[i][1]) in (0.0, 1.0) for i in range(8))

    def test_partner_comes_from_the_wrapped_dataset_only(self) -> None:
        """Mixup must not reach outside the training split. The partner index is bounded by it.

        Recorded via a subclass rather than by patching the instance: Python resolves dunder
        methods on the type, so assigning `base.__getitem__` would never be consulted and the
        test would pass while observing nothing.
        """

        class RecordingDataset(StubDataset):
            def __init__(self, n: int) -> None:
                super().__init__(n=n)
                self.seen: list[int] = []

            def __getitem__(self, index: int):
                self.seen.append(index)
                return super().__getitem__(index)

        base = RecordingDataset(n=5)
        ds = RichAugDataset(base, AugConfig(mixup_alpha=0.4, mixup_prob=1.0), seed=1)
        for i in range(5):
            ds[i]
        assert base.seen, "the wrapper never read the wrapped dataset"
        assert all(0 <= i < len(base) for i in base.seen)
        assert len(base.seen) == 10, "every sample should have drawn exactly one partner"


class TestEnableDetection:
    @pytest.mark.parametrize(
        "cfg",
        [
            AugConfig(elastic_alpha=1.0),
            AugConfig(blur_sigma_max=0.5),
            AugConfig(count_noise=0.05),
            AugConfig(mixup_alpha=0.2),
            AugConfig(gamma_range=(0.9, 1.1)),
        ],
    )
    def test_each_knob_switches_the_wrapper_on(self, cfg: AugConfig) -> None:
        assert isinstance(wrap_if_enabled(StubDataset(), cfg), RichAugDataset)


class TestCost:
    def test_per_sample_cost_is_affordable(self) -> None:
        """A budget, not a benchmark.

        Measured at rich_prob=1.0, so this is the worst case rather than the average; at the
        configured 0.5 the expected cost is about half.

        An epoch is 399 s of main-process compute over ~1090 samples — 366 ms per sample — and
        augmentation runs in dataloader workers in parallel with it. With `num_workers` of 4 the
        budget per sample is therefore ~1.4 s before augmentation becomes the bottleneck rather
        than being hidden entirely. The bound below sits well inside that while still catching an
        accidental order of magnitude: the first implementation of the displacement field
        upsampled coarse noise with a cubic spline and cost 3.5 s per sample, which would have
        turned a 16.6 h run into a week. Whether the cost is genuinely hidden is settled by
        timing a real epoch, not here.
        """
        cfg = AugConfig(elastic_alpha=3.0, blur_sigma_max=1.0, gamma_range=(0.9, 1.1),
                        count_noise=0.05, rich_prob=1.0)
        ds = RichAugDataset(StubDataset(), cfg, seed=0)
        ds[0]  # warm any lazy allocation

        start = time.perf_counter()
        for i in range(5):
            ds[i]
        per_sample = (time.perf_counter() - start) / 5

        print(f"\nrich augmentation: {per_sample * 1000:.0f} ms/sample at 96^3")
        assert per_sample < 0.60, f"{per_sample:.3f}s per sample is too slow for a 5-fold run"
