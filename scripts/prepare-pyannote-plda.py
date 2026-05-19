#!/usr/bin/env python3
"""Create plda_prepared.npz from upstream Pyannote PLDA files.

The public Pyannote Community-1 repository provides plda.npz and
xvec_transform.npz. The Android runtime uses a prepared NPZ so it does not need
to run SciPy linear algebra on device.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

try:
    from scipy.linalg import eigh
except ImportError as exc:
    raise SystemExit(
        "scipy is required to prepare Pyannote PLDA assets. Install it with:\n"
        "  python -m pip install numpy scipy"
    ) from exc


def prepare(plda_dir: Path, output: Path) -> None:
    xvec_path = plda_dir / "xvec_transform.npz"
    plda_path = plda_dir / "plda.npz"
    if not xvec_path.is_file():
        raise FileNotFoundError(f"Missing {xvec_path}")
    if not plda_path.is_file():
        raise FileNotFoundError(f"Missing {plda_path}")

    xvec = np.load(xvec_path, allow_pickle=False)
    plda = np.load(plda_path, allow_pickle=False)

    plda_tr = plda["tr"]
    plda_psi = plda["psi"]
    w = np.linalg.inv(plda_tr.T @ plda_tr)
    b = np.linalg.inv((plda_tr.T / plda_psi) @ plda_tr)
    acvar, wccn = eigh(b, w)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        mean1=xvec["mean1"],
        mean2=xvec["mean2"],
        lda=xvec["lda"],
        mu=plda["mu"],
        plda_tr=wccn.T[::-1],
        plda_psi=acvar[::-1],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plda-dir",
        required=True,
        type=Path,
        help="Directory containing plda.npz and xvec_transform.npz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output plda_prepared.npz path. Defaults to <plda-dir>/plda_prepared.npz.",
    )
    args = parser.parse_args()

    plda_dir = args.plda_dir.resolve()
    output = args.output.resolve() if args.output else plda_dir / "plda_prepared.npz"
    prepare(plda_dir, output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
