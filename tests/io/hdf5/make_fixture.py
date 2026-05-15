"""Build a tiny synthetic .h5 file with the LeWM PushT dataset layout.

The real ``pusht_expert_train.h5`` (13 GB compressed) is too large to use
during FFI development. This fixture replicates the on-disk shape:
top-level datasets ``ep_len`` / ``ep_offset`` / ``pixels`` / ``action``
/ ``proprio`` / ``state``, with per-frame chunking, exactly as
``stable_worldmodel.data.formats.hdf5.HDF5Writer`` produces.

The values are deterministic so the Mojo test can assert exact equality.

Usage:
    pixi run python tests/io/hdf5/make_fixture.py [output_path]

If ``output_path`` is omitted, writes to ``/tmp/mojo_rl_hdf5_fixture.h5``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


# Tiny but multi-episode so episode-boundary logic gets exercised.
EP_LENGTHS = [4, 3, 5]                       # 3 episodes, 12 frames total
H, W = 8, 6                                  # tiny image
ACTION_DIM = 2
PROPRIO_DIM = 2
STATE_DIM = 5


def build(path: Path) -> None:
    n_total = sum(EP_LENGTHS)
    offsets = np.cumsum([0, *EP_LENGTHS[:-1]], dtype=np.int64)

    # Deterministic per-frame values.
    pixels = np.zeros((n_total, H, W, 3), dtype=np.uint8)
    action = np.zeros((n_total, ACTION_DIM), dtype=np.float32)
    proprio = np.zeros((n_total, PROPRIO_DIM), dtype=np.float32)
    state = np.zeros((n_total, STATE_DIM), dtype=np.float32)
    for t in range(n_total):
        pixels[t] = (t * 7) % 256                          # uniform plane per t
        action[t] = [float(t), float(t) + 0.5]
        proprio[t] = [float(t) * 0.1, float(t) * 0.2]
        state[t] = [
            float(t) * 1.0,
            float(t) * 2.0,
            float(t) * 3.0,
            float(t) * 4.0,
            float(t) * 5.0,
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", libver="latest") as f:
        # Match HDF5Writer schema exactly:
        # per-row chunks, resizable, ep_len=int32, ep_offset=int64.
        f.create_dataset(
            "ep_len",
            data=np.asarray(EP_LENGTHS, dtype=np.int32),
            maxshape=(None,),
            dtype=np.int32,
        )
        f.create_dataset(
            "ep_offset",
            data=offsets,
            maxshape=(None,),
            dtype=np.int64,
        )
        f.create_dataset(
            "pixels", data=pixels, maxshape=(None, H, W, 3),
            chunks=(1, H, W, 3),
        )
        f.create_dataset(
            "action", data=action, maxshape=(None, ACTION_DIM),
            chunks=(1, ACTION_DIM),
        )
        f.create_dataset(
            "proprio", data=proprio, maxshape=(None, PROPRIO_DIM),
            chunks=(1, PROPRIO_DIM),
        )
        f.create_dataset(
            "state", data=state, maxshape=(None, STATE_DIM),
            chunks=(1, STATE_DIM),
        )

    print(f"[fixture] wrote {path}")
    print(f"          ep_lengths={EP_LENGTHS}  ep_offsets={offsets.tolist()}")
    print(f"          n_total={n_total}  pixels=({n_total},{H},{W},3) uint8")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/tmp/mojo_rl_hdf5_fixture.h5"
    )
    build(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
