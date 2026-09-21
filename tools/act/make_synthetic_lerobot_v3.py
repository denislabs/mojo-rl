#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | A tiny synthetic LeRobot v3.0 dataset + the reference store for it
# +--------------------------------------------------------------------------+ #
"""Builds a complete v3.0 dataset small enough to gate the Mojo importer on.

    pixi run python tools/act/make_synthetic_lerobot_v3.py --out /tmp/lerobot_synth
    pixi run mojo run -I . tests/data/test_lerobot_import.mojo /tmp/lerobot_synth

Writes `<out>/dataset/` (a real v3.0 tree) and `<out>/reference.h5`, the store
that `tools/act/lerobot_v3_to_store.py` — the trusted Python converter — makes
from it. The Mojo gate imports the same tree and compares the two stores.

⚠ WHAT THIS GATES IS THE WIRING, NOT THE DECODERS. Parquet, the resize and the
JSON reader each have their own bit-exact gate against pyarrow / Pillow. What
no unit test covers is the part of the import that is pure bookkeeping, and
that is what every field below is shaped to break if it is wrong:

  * **unequal episode lengths** (7, 4, 9, 5) — equal lengths make an
    off-by-one in `ep_offset` invisible, since every wrong answer is also a
    valid one.
  * **episodes written OUT OF ORDER** in the episode parquet (2, 0, 3, 1) —
    the importer sorts by `episode_index`; reading the file order instead
    still produces a well-formed store of shuffled episodes.
  * **two video files per camera**, with the episode boundary NOT on the file
    boundary, so the streaming decoder has to roll over mid-sequence.
  * **a gap between episodes inside a file** (`from_timestamp` skips two
    frames) so "decode forward and discard" is actually exercised; without a
    gap, ignoring `from_timestamp` entirely would pass.
  * **two cameras with different content**, so a swapped camera slot shows up.
  * **frame content that encodes (camera, frame index) in the pixels**, so a
    mis-routed frame is a wrong number rather than a similar-looking image.
  * **`libx264rgb` at `-qp 0`** — mathematically lossless RGB, so the pixels
    that come back out are the pixels that went in and any difference is the
    importer's.

The videos are 64x48 and resized to 12x16 by the import, which is a
non-integer downscale in both axes — the case where the filter windows differ
per output pixel.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 10
SRC_H, SRC_W = 48, 64
OUT_H, OUT_W = 12, 16
STATE_DIM = 4
ACTION_DIM = 3
CAMERAS = ["observation.images.left", "observation.images.top"]

LENGTHS = [7, 4, 9, 5]
# Episode -> (video file ordinal, first frame in that file). The gap at
# episode 1 and the file rollover at episode 2 are the point.
PLACEMENT = [(0, 0), (0, 9), (1, 0), (1, 11)]
FILE_ORDER = [2, 0, 3, 1]  # the order rows appear in the episode parquet


def frame(cam_slot: int, file_ord: int, k: int) -> np.ndarray:
    """A frame whose bytes identify (camera, file, index) unambiguously."""
    img = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    yy = np.arange(SRC_H)[:, None]
    xx = np.arange(SRC_W)[None, :]
    img[..., 0] = ((yy * 5 + xx * 3 + k * 11 + cam_slot * 71) % 256).astype(np.uint8)
    img[..., 1] = np.uint8((k * 23 + cam_slot * 97 + file_ord * 41) % 256)
    img[..., 2] = ((xx * 7 - yy * 2 + k * 5) % 256).astype(np.uint8)
    return img


def encode(path: Path, frames: list[np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{SRC_W}x{SRC_H}", "-r", str(FPS), "-i", "-",
            # Lossless RGB: no colour conversion, no quantisation. A yuv420p
            # encode would round-trip within a few LSB and turn every pixel
            # comparison in the gate into a tolerance argument.
            "-c:v", "libx264rgb", "-qp", "0", "-pix_fmt", "rgb24",
            str(path),
        ],
        input=b"".join(f.tobytes() for f in frames),
        capture_output=True,
    )
    if p.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {path}: {p.stderr.decode()[:400]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/lerobot_synth")
    args = ap.parse_args()
    out = Path(args.out)
    root = out / "dataset"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    n_ep = len(LENGTHS)
    n_rows = sum(LENGTHS)
    from_index = np.cumsum([0] + LENGTHS[:-1]).astype(np.int64)
    to_index = np.cumsum(LENGTHS).astype(np.int64)

    rng = np.random.default_rng(3)
    state = rng.standard_normal((n_rows, STATE_DIM)).astype(np.float32)
    action = rng.standard_normal((n_rows, ACTION_DIM)).astype(np.float32)
    ep_col = np.repeat(np.arange(n_ep, dtype=np.int64), LENGTHS)

    # ── data/ ────────────────────────────────────────────────────────────
    tbl = pa.table(
        {
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(state.reshape(-1)), STATE_DIM
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(action.reshape(-1)), ACTION_DIM
            ),
            "timestamp": pa.array(
                np.concatenate([np.arange(n) / FPS for n in LENGTHS]).astype(
                    np.float32
                )
            ),
            "frame_index": pa.array(
                np.concatenate([np.arange(n) for n in LENGTHS]).astype(np.int64)
            ),
            "episode_index": pa.array(ep_col),
            "index": pa.array(np.arange(n_rows, dtype=np.int64)),
            # ⚠⚠ MULTI-TASK, AND DELIBERATELY NOT ALL-ZERO. This column was
            # zeros until 2026-09-04 and the importer dropped it entirely, so
            # nothing in the suite could tell a store that carried task
            # identity from one that lost it. Cycling per EPISODE means the
            # gate fails if the importer drops the column, mis-orders rows, or
            # writes a per-episode value where a per-frame one belongs.
            #
            # ⚠ The real SO-101 dataset is SINGLE-TASK, so this synthetic one
            # is the only place multi-task is exercised at all.
            "task_index": pa.array(
                np.concatenate(
                    [np.full(n, e % 3, np.int64)
                     for e, n in enumerate(LENGTHS)]
                )
            ),
        }
    )
    (root / "data/chunk-000").mkdir(parents=True)
    pq.write_table(tbl, root / "data/chunk-000/file-000.parquet",
                   compression="snappy", row_group_size=9)

    # ── videos/ ──────────────────────────────────────────────────────────
    # Each file holds the episodes placed in it, padded so that every
    # episode starts exactly at its `from_timestamp`.
    for slot, cam in enumerate(CAMERAS):
        for file_ord in (0, 1):
            eps = [e for e in range(n_ep) if PLACEMENT[e][0] == file_ord]
            n_frames = max(PLACEMENT[e][1] + LENGTHS[e] for e in eps)
            frames = [frame(slot, file_ord, 1000 + i) for i in range(n_frames)]
            for e in eps:
                start = PLACEMENT[e][1]
                for k in range(LENGTHS[e]):
                    frames[start + k] = frame(slot, file_ord, e * 100 + k)
            encode(
                root / f"videos/{cam}/chunk-000/file-{file_ord:03d}.mp4", frames
            )

    # ── meta/episodes/ ───────────────────────────────────────────────────
    cols = {
        "episode_index": pa.array(np.array(FILE_ORDER, np.int64)),
        "length": pa.array(np.array([LENGTHS[e] for e in FILE_ORDER], np.int64)),
        "dataset_from_index": pa.array(
            np.array([from_index[e] for e in FILE_ORDER], np.int64)
        ),
        "dataset_to_index": pa.array(
            np.array([to_index[e] for e in FILE_ORDER], np.int64)
        ),
        "data/chunk_index": pa.array(np.zeros(n_ep, np.int64)),
        "data/file_index": pa.array(np.zeros(n_ep, np.int64)),
    }
    for cam in CAMERAS:
        cols[f"videos/{cam}/chunk_index"] = pa.array(np.zeros(n_ep, np.int64))
        cols[f"videos/{cam}/file_index"] = pa.array(
            np.array([PLACEMENT[e][0] for e in FILE_ORDER], np.int64)
        )
        cols[f"videos/{cam}/from_timestamp"] = pa.array(
            np.array([PLACEMENT[e][1] / FPS for e in FILE_ORDER], np.float64)
        )
        cols[f"videos/{cam}/to_timestamp"] = pa.array(
            np.array(
                [(PLACEMENT[e][1] + LENGTHS[e]) / FPS for e in FILE_ORDER],
                np.float64,
            )
        )
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table(cols), root / "meta/episodes/chunk-000/file-000.parquet",
        compression="snappy",
    )

    # ── meta/tasks.parquet ───────────────────────────────────────────────
    #
    # ⚠⚠ THE STRINGS ARE HOSTILE ON PURPOSE. The store carries this text
    # BYTE-EXACT because a consumer tokenises it, and the manifest it lands in
    # is a `key=value` LINE format whose reader STRIPS values — so a trailing
    # space and a newline are the two things most likely to be silently eaten
    # between parquet and store. Non-ASCII is here because a per-byte `chr`
    # round trip re-encodes anything above 127, which is a bug this tree
    # actually had in `manifest._split`.
    #
    # If these ever look like gratuitous awkwardness: an ordinary instruction
    # round-trips through a broken escaper too.
    (root / "meta").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array(np.arange(3, dtype=np.int64)),
                "task": pa.array(
                    [
                        "Grab the green cube",
                        "Place it on the shelf ",      # TRAILING SPACE
                        "Push the ünïcøde block ✓",    # multi-byte
                    ]
                ),
            }
        ),
        root / "meta/tasks.parquet",
        compression="snappy",
    )

    # ── meta/info.json ───────────────────────────────────────────────────
    info = {
        "codebase_version": "v3.0",
        "robot_type": "synthetic",
        "fps": FPS,
        "total_episodes": n_ep,
        "total_frames": n_rows,
        "features": {
            "action": {"dtype": "float32", "shape": [ACTION_DIM],
                       "names": [f"a{i}" for i in range(ACTION_DIM)]},
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM],
                                  "names": [f"s{i}" for i in range(STATE_DIM)]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    for cam in CAMERAS:
        info["features"][cam] = {
            "dtype": "video",
            "shape": [SRC_H, SRC_W, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": FPS, "video.codec": "h264"},
        }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta/info.json").write_text(json.dumps(info, indent=2))

    # ── the reference store, from the trusted Python converter ───────────
    ref = out / "reference.h5"
    if ref.exists():
        ref.unlink()
    conv = Path(__file__).with_name("lerobot_v3_to_store.py")
    p = subprocess.run(
        [sys.executable, str(conv), "--out", str(ref), "--force",
         "--height", str(OUT_H), "--width", str(OUT_W), "--root", str(root)],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise SystemExit(
            "the reference converter failed:\n" + p.stdout + p.stderr
        )
    (out / "case.txt").write_text(
        f"{root}\n{ref}\n{OUT_H}\n{OUT_W}\n"
    )
    print(p.stdout.strip())
    print(f"\nsynthetic dataset -> {root}")
    print(f"reference store   -> {ref}")


if __name__ == "__main__":
    main()
