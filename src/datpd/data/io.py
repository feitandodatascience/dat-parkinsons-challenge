"""NIfTI loading, orientation handling and header inspection.

VENDORED INTO submission.zip. Keep the dependency surface at numpy + nibabel, and keep it free
of imports from the rest of `datpd` other than siblings that are also vendored.

The two entry points serve different needs:

- `read_header_only` reads metadata *without* touching voxels. EDA over thousands of scans and
  fold construction both need only geometry; decompressing every volume for that would waste
  minutes of every run.
- `load_volume` reads voxels, reorienting to canonical RAS first.

**Why reorient at all.** NIfTI stores an affine mapping voxel indices to physical space, and
two scanners can store the same anatomy with axes permuted or flipped. A CNN has no way to know
that; it would see two mirror populations and spend capacity undoing the difference (docs/02
Delta 1, step 1). `as_closest_canonical` permutes and flips the array so the axis order is
always (R, A, S)-ish, which makes "the left putamen" a consistent place in the tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes, io_orientation


@dataclass
class Volume:
    """One examination: voxels plus the physical geometry needed to interpret them.

    `spacing_mm` is the crucial field, and the reason a bare ndarray is not enough - without it
    "96 voxels" has no physical meaning and volumes from different scanners are not comparable.
    """

    array: np.ndarray
    spacing_mm: tuple[float, float, float]
    affine: np.ndarray
    uid: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.array.shape)  # type: ignore[return-value]

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        """Physical size of the field of view. Two volumes can share a shape and no extent."""
        return tuple(  # type: ignore[return-value]
            float(n * s) for n, s in zip(self.array.shape, self.spacing_mm)
        )


def uid_from_path(path: str | Path) -> str:
    """Recover the examination uid from a volume filename.

    Splits at the first dot rather than matching a suffix list, so single- and double-extension
    NIfTI files are handled identically. Uids in this competition are dot-free tokens, which is
    what makes that safe.
    """
    return Path(path).name.split(".")[0]


def build_volume_index(nifti_dir: str | Path) -> dict[str, Path]:
    """Map uid -> file path with a single directory scan.

    Resolving a uid by globbing costs a directory walk per sample; over thousands of samples per
    epoch that dominates. Scanning once up front is O(n) total, and it also makes a missing
    volume a clear KeyError at startup instead of an I/O error deep inside a DataLoader worker.
    """
    index: dict[str, Path] = {}
    for path in sorted(Path(nifti_dir).iterdir()):
        if path.is_file():
            index.setdefault(uid_from_path(path), path)
    return index


def _canonical_axis_permutation(affine: np.ndarray) -> list[int]:
    """Which stored axis becomes each canonical axis, per `nibabel.as_closest_canonical`.

    Lets `read_header_only` report post-reorientation geometry without loading any voxels, so
    the acquisition signature used for grouped CV matches what training actually sees.
    """
    return [int(row[0]) for row in io_orientation(affine)]


def read_header_only(path: str | Path) -> dict[str, Any]:
    """Read geometry without decompressing the voxel data.

    Reports shape and spacing **after** canonical reorientation, so the values line up with
    `load_volume`, alongside the *stored* orientation code - which is what reveals whether
    centres disagree about axis order.

    Returns keys: `shape`, `spacing_mm`, `dtype`, `orientation`, `stored_shape`.
    """
    img = nib.load(str(path))
    stored_shape = tuple(int(n) for n in img.header.get_data_shape()[:3])
    stored_zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    perm = _canonical_axis_permutation(img.affine)

    return {
        "shape": tuple(stored_shape[i] for i in perm),
        "spacing_mm": tuple(stored_zooms[i] for i in perm),
        "dtype": str(img.header.get_data_dtype()),
        "orientation": "".join(aff2axcodes(img.affine)),
        "stored_shape": stored_shape,
    }


def load_volume(path: str | Path) -> Volume:
    """Load a volume, reoriented to canonical RAS and cast to float32.

    float32 rather than the stored uint16 because everything downstream (resampling,
    normalisation) is floating point anyway; doing the cast once here keeps integer truncation
    out of the pipeline.
    """
    path = Path(path)
    img = nib.as_closest_canonical(nib.load(str(path)))
    array = np.asanyarray(img.dataobj, dtype=np.float32)

    if array.ndim > 3:
        # Some exports carry a trailing singleton time/frame axis. Drop it rather than letting
        # a (D, H, W, 1) array silently become a 4D tensor at the model.
        array = array.reshape(array.shape[:3])

    return Volume(
        array=array,
        spacing_mm=tuple(float(z) for z in img.header.get_zooms()[:3]),
        affine=np.asarray(img.affine, dtype=np.float64),
        uid=uid_from_path(path),
    )

