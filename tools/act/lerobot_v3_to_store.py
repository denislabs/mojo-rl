#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | LeRobot v3.0 dataset  ->  mojo_rl TrajectoryStore (.h5)
# +--------------------------------------------------------------------------+ #
"""Convert a LeRobot v3.0 HuggingFace dataset into one `TrajectoryStore` file.

Run ONCE per (repo, resolution); everything after that is pure Mojo through
`mojo_rl/io/hdf5` + `mojo_rl.data.TrajectoryStore`.

    pixi run python tools/act/lerobot_v3_to_store.py \
        --repo DenisLabs/record-test_20260825_094319 --height 240 --width 320

⚠ This script must run in its OWN process, never inside Mojo's embedded Python
interpreter: `libarrow` (pyarrow) SIGABRTs at static-destructor teardown when
loaded there. Mojo callers shell out with `subprocess.run([sys.executable, ...])`.

## What LeRobot v3.0 looks like

    meta/info.json                       features, fps, total_{episodes,frames}
    meta/episodes/chunk-*/file-*.parquet  per-episode index + per-camera video map
    data/chunk-*/file-*.parquet          action / observation.state / timestamps
    videos/<key>/chunk-*/file-*.mp4      MANY episodes concatenated per file

The episode index carries `dataset_from_index` / `dataset_to_index` (the flat
row range, i.e. exactly `ep_offset` / `ep_len`) and, per camera, the
`(chunk_index, file_index, from_timestamp)` locating that episode inside the
packed mp4. `round(from_timestamp * fps)` is the episode's first frame index
within its video file, so each file is decoded ONCE, sequentially, and its
frames are routed to the right flat rows. No seeking.

## Output layout

A single `.h5` in `TrajectoryStoreWriter`'s format — rank-2 columns with the
true trailing shape carried in the manifest text (`data/store.mojo`
`_create_for`, `data/manifest.mojo`):

    __manifest__  uint8   (M,)          key=value block
    ep_len        int64   (E,)
    ep_offset     int64   (E,)
    qpos          float32 (N, S)        observation.state
    action        float32 (N, A)
    images        uint8   (N, C*3*H*W)  C cameras, CHW each, sorted by name

plus `norm_{qpos,action}_{mean,std}` float32 (S,)/(A,) — the reference's
`get_norm_stats` (`references/act-main/utils.py:78`), std clipped below at 1e-2.
Those four are NOT manifest columns (they have E-independent length); a reader
that has a manifest ignores undeclared datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
STD_FLOOR = 1e-2  # references/act-main/utils.py:96 `torch.clip(std, 1e-2, inf)`


# ── manifest -----------------------------------------------------------------


def escape_task_text(s: str) -> str:
    """`manifest.escape_task_text`, in Python. Byte-exact and quoted.

    ⚠ OPERATES ON BYTES, not on str characters. The Mojo side escapes five
    BYTE values and copies the rest verbatim; doing this per-character would
    diverge the moment a task string is non-ASCII, and the two writers would
    disagree on exactly the input nobody tests.
    """
    b = s.encode("utf-8")
    out = bytearray(b'"')
    for c in b:
        if c == 0x5C:
            out += b"\\\\"
        elif c == 0x22:
            out += b'\\"'
        elif c == 0x0A:
            out += b"\\n"
        elif c == 0x0D:
            out += b"\\r"
        elif c == 0x09:
            out += b"\\t"
        else:
            out.append(c)
    out += b'"'
    return out.decode("utf-8", errors="surrogateescape")


def read_task_table(root: Path):
    """`meta/tasks.parquet` -> `[(task_index, text)]`, text unchanged."""
    import pyarrow.parquet as pq

    t = pq.read_table(root / "meta" / "tasks.parquet")
    idx = [int(v) for v in t.column("task_index")]
    txt = [str(v) for v in t.column("task")]
    return list(zip(idx, txt))


def encode_manifest(
    env_id, n_rows, n_episodes, seed, source_commit, columns, tasks=()
):
    """`data/manifest.mojo::Manifest.encode` — byte-for-byte the same format.

    `columns` is a list of `(name, dtype_name, trailing_shape_tuple)`.
    `tasks` is `[(task_index, text)]` and is written AFTER the columns, in the
    order given — the Mojo encoder does the same and the two must agree byte
    for byte.

    ⚠ THE TEXT IS QUOTED AND ESCAPED, mirroring `manifest.escape_task_text`.
    The manifest is `key=value` LINES and its reader strips every value, so a
    newline would split the record and edge whitespace would vanish. The text
    must survive BYTE-EXACT because a consumer tokenises it.
    """
    lines = [
        f"schema_version={SCHEMA_VERSION}",
        f"env_id={env_id}",
        f"n_rows={n_rows}",
        f"n_episodes={n_episodes}",
        f"seed={seed}",
        f"source_commit={source_commit}",
    ]
    for name, dt, shape in columns:
        spec = f"{name}:{dt}"
        if shape:
            spec += ":" + ",".join(str(d) for d in shape)
        lines.append(f"column={spec}")
    for index, text in tasks:
        lines.append(f"task={index}\t{escape_task_text(text)}")
    return "\n".join(lines) + "\n"


# ── hub ----------------------------------------------------------------------


def snapshot(repo: str, revision: str | None):
    """Pull the whole dataset repo (metadata + parquet + mp4) into the HF cache."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo, repo_type="dataset", revision=revision)
    )


def sorted_parquets(root: Path, subdir: str):
    """`chunk-XXX/file-YYY.parquet` under `subdir`, in (chunk, file) order."""
    base = root / subdir
    if not base.is_dir():
        raise SystemExit(f"missing {subdir}/ in the dataset repo")
    out = []
    for p in base.rglob("*.parquet"):
        rel = p.relative_to(base).as_posix()
        m = re.search(r"chunk-(\d+)/file-(\d+)\.parquet$", rel)
        key = (int(m.group(1)), int(m.group(2))) if m else (1 << 30, 1 << 30)
        out.append((key, p))
    out.sort()
    return [p for _, p in out]


# ── reading ------------------------------------------------------------------


def read_frames(root: Path):
    """The flat per-frame table: action, observation.state, episode_index."""
    import pyarrow.parquet as pq

    tables = [pq.read_table(p) for p in sorted_parquets(root, "data")]
    if not tables:
        raise SystemExit("no data/*.parquet found")
    import pyarrow as pa

    t = pa.concat_tables(tables)

    def fixed_list(col):
        """fixed_size_list<float>[k] -> (N, k) float32."""
        arr = t.column(col).combine_chunks()
        width = arr.type.list_size
        flat = np.asarray(arr.flatten(), dtype=np.float32)
        return flat.reshape(len(arr), width)

    return {
        "action": fixed_list("action"),
        "qpos": fixed_list("observation.state"),
        "episode_index": np.asarray(t.column("episode_index"), dtype=np.int64),
        "task_index": np.asarray(t.column("task_index"), dtype=np.int32),
        "n_rows": t.num_rows,
    }


def read_episodes(root: Path, cameras):
    """The per-episode index, sorted by `episode_index`.

    Returns `(lengths, from_index, video_map)` where `video_map[cam]` is a list
    of `(chunk_index, file_index, from_timestamp)`, one per episode.
    """
    import pyarrow.parquet as pq
    import pyarrow as pa

    tables = [pq.read_table(p) for p in sorted_parquets(root, "meta/episodes")]
    if not tables:
        raise SystemExit("no meta/episodes/*.parquet found")
    # The stats/* columns are wide and unused here; drop them before concat so
    # a schema mismatch between files cannot fail on a column we never read.
    keep = [n for n in tables[0].schema.names if not n.startswith("stats/")]
    t = pa.concat_tables([tb.select(keep) for tb in tables])
    d = t.to_pydict()

    order = np.argsort(np.asarray(d["episode_index"], dtype=np.int64))
    lengths = np.asarray(d["length"], dtype=np.int64)[order]
    from_index = np.asarray(d["dataset_from_index"], dtype=np.int64)[order]
    to_index = np.asarray(d["dataset_to_index"], dtype=np.int64)[order]

    if not np.array_equal(to_index - from_index, lengths):
        raise SystemExit("episode index: dataset_to - dataset_from != length")
    if from_index[0] != 0 or not np.array_equal(from_index[1:], to_index[:-1]):
        raise SystemExit("episode index: flat row ranges are not contiguous")

    video_map = {}
    for cam in cameras:
        ci = np.asarray(d[f"videos/{cam}/chunk_index"], dtype=np.int64)[order]
        fi = np.asarray(d[f"videos/{cam}/file_index"], dtype=np.int64)[order]
        ts = np.asarray(
            d[f"videos/{cam}/from_timestamp"], dtype=np.float64
        )[order]
        video_map[cam] = list(zip(ci.tolist(), fi.tolist(), ts.tolist()))

    return lengths, from_index, video_map


# ── video --------------------------------------------------------------------


def decode_camera(root, cam, video_map, lengths, from_index, fps, h, w, out):
    """Decode one camera's mp4 files, writing each frame as CHW uint8.

    `out` takes `out[row] = chw_frame` and is a `_CamRows` view onto this
    camera's slice of the on-disk images dataset — NOT a numpy array. Frames
    go straight to the file: see `_CamRows` for why.
    """
    import imageio.v3 as iio
    from PIL import Image

    # Group episodes by the video file they live in.
    by_file = {}
    for ep, (ci, fi, ts) in enumerate(video_map[cam]):
        by_file.setdefault((ci, fi), []).append((ep, ts))

    filled = 0
    for (ci, fi), eps in sorted(by_file.items()):
        path = root / "videos" / cam / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
        if not path.is_file():
            raise SystemExit(f"missing video file {path}")

        # src frame index -> flat dataset row, for every frame this file owns.
        route = {}
        for ep, ts in eps:
            src0 = int(round(ts * fps))
            dst0 = int(from_index[ep])
            for k in range(int(lengths[ep])):
                route[src0 + k] = dst0 + k

        n_src = 0
        for i, frame in enumerate(iio.imiter(path, plugin="FFMPEG")):
            n_src += 1
            dst = route.get(i)
            if dst is None:
                continue
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = np.asarray(
                    Image.fromarray(frame).resize((w, h), Image.BILINEAR)
                )
            # HWC -> CHW
            out[dst] = np.transpose(frame, (2, 0, 1))
            filled += 1

        missing = [s for s in route if s >= n_src]
        if missing:
            raise SystemExit(
                f"{cam} {path.name}: {len(missing)} routed frames past the end"
                f" of the file ({n_src} frames decoded); the episode index and"
                f" the video disagree"
            )
        print(f"    {cam} chunk-{ci:03d}/file-{fi:03d}: {n_src} frames decoded")

    if filled != int(lengths.sum()):
        raise SystemExit(
            f"{cam}: filled {filled} rows, expected {int(lengths.sum())}"
        )


class _CamRows:
    """`out[row] = chw` onto ONE camera's slice of the flat images dataset.

    ⚠ THIS EXISTS TO KEEP THE CONVERTER'S MEMORY BOUNDED. It used to decode
    into `np.zeros((n_rows, n_cams, 3, h, w))` and hand that to h5py at the
    end — **7.12 GB resident** for 50 episodes at 240x320, on top of the 7.12 GB
    it then writes. That is a machine-killer on a 16 GB laptop and it scales
    with the recording, so the next dataset would be worse. Decoding is already
    strictly row-at-a-time; the array was pure accumulation.

    The images dataset is chunked `(1, cam_elems)` rather than one chunk per
    full row, so each write here lands on exactly ONE chunk. Chunked at the
    full row, every camera write would be half a chunk and HDF5 would
    read-modify-write it — twice the write traffic for no gain. A reader taking
    a whole row now touches two chunks instead of one, and they are adjacent on
    disk, which is why that side does not care.
    """

    def __init__(self, dset, slot, cam_elems):
        self._d = dset
        self._o = slot * cam_elems
        self._n = cam_elems

    def __setitem__(self, row, chw):
        self._d[row, self._o : self._o + self._n] = chw.reshape(-1)


# ── stats --------------------------------------------------------------------


def norm_stats(x: np.ndarray):
    """`get_norm_stats` (references/act-main/utils.py:78), generalized.

    The reference `torch.stack`s per-episode arrays and reduces over dims
    [0, 1], which requires equal-length episodes; over the flat row axis is the
    same quantity whenever they ARE equal, and the natural reading when they
    are not. `std` is clipped below at 1e-2 exactly as the reference does.

    ⚠ `ddof=1`. `torch.std` is UNBIASED by default; `np.std` is not. The gap is
    only sqrt(N/(N-1)) — 1.00025 at N=1997, invisible in any single check — but
    it is a systematic offset that would sit under every sim-to-reference
    comparison afterwards, so it is matched rather than waved off.

    ⚠ ACCUMULATE IN FLOAT64. The columns are float32 and `ndarray.mean` uses
    the array's own dtype as the accumulator, so reducing them in place drifts
    with the row count: at 1997 rows the mean was 2.4e-4 off the exact value,
    at 15447 rows it was **2.3e-3** — and `ACTDataset._moments`, which sums in
    Float64, was the side that was right. The gate that compares the two saw a
    growing gap and had no way to say which implementation it was accusing.
    The cast costs one temporary over a (N, 6) table.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    return mean.astype(np.float32), np.clip(std, STD_FLOOR, np.inf).astype(
        np.float32
    )


def refresh_stats(out: Path) -> int:
    """Recompute `norm_*` in place from the store's OWN qpos/action columns.

    The statistics are four 6-vectors derived from two tiny columns, but they
    live in a file whose bulk is a multi-GB image column that costs a download
    and a full video decode to reproduce. When the definition of the statistic
    changes — as it did when `norm_stats` moved to a float64 accumulator —
    rebuilding the whole store to update 48 floats is the wrong trade. This
    recomputes them from what the store already holds, so the refreshed values
    are derived from exactly the rows the store serves.

    The `.json` sidecar is patched key-by-key rather than rewritten, so a field
    this function does not know about survives.
    """
    if not out.exists():
        raise SystemExit(f"--refresh-stats: no store at {out}")
    import h5py  # local, matching the write path below

    with h5py.File(out, "r+") as f:
        q_mean, q_std = norm_stats(f["qpos"][:])
        a_mean, a_std = norm_stats(f["action"][:])
        for name, val in (
            ("norm_qpos_mean", q_mean),
            ("norm_qpos_std", q_std),
            ("norm_action_mean", a_mean),
            ("norm_action_std", a_std),
        ):
            before = f[name][:]
            f[name][...] = val
            print(
                f"  {name:18s} max|delta| ="
                f" {np.abs(before - val).max():.3e}"
            )
    side = out.with_suffix(".json")
    if side.exists():
        meta = json.loads(side.read_text())
        meta["qpos_mean"] = q_mean.tolist()
        meta["qpos_std"] = q_std.tolist()
        meta["action_mean"] = a_mean.tolist()
        meta["action_std"] = a_std.tolist()
        side.write_text(json.dumps(meta, indent=2))
    print(f"refreshed norm_* in {out}")
    return 0


# ── main ---------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        default=None,
        help="HF dataset repo id (optional with --refresh-stats + --out)",
    )
    ap.add_argument("--revision", default=None)
    ap.add_argument(
        "--root",
        default=None,
        help="a local v3.0 dataset directory, instead of downloading --repo."
        " Used by tools/act/make_synthetic_lerobot_v3.py to produce the"
        " reference store the Mojo importer is gated against",
    )
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument(
        "--cameras",
        default=None,
        help="comma-separated video feature keys; default = all, sorted",
    )
    ap.add_argument("--out", default=None, help="output .h5 (default: cache)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--refresh-stats",
        action="store_true",
        help="recompute norm_* in an EXISTING store from its own qpos/action"
        " columns and exit — no download, no video decode",
    )
    args = ap.parse_args()

    h, w = args.height, args.width

    if args.out:
        out = Path(args.out)
    elif args.repo:
        slug = args.repo.replace("/", "__")
        out = Path.home() / ".cache/mojo_rl/act_so101" / f"{slug}_{h}x{w}.h5"
    else:
        raise SystemExit("need --repo (or --out with --refresh-stats)")
    if args.refresh_stats:
        return refresh_stats(out)
    if not args.repo and not args.root:
        raise SystemExit("--repo or --root is required to build a store")
    if out.exists() and not args.force:
        print(f"already present: {out}  (pass --force to rebuild)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.root:
        print(f"[1/5] using local dataset {args.root} ...")
        root = Path(args.root)
    else:
        print(f"[1/5] downloading {args.repo} ...")
        root = snapshot(args.repo, args.revision)
    info = json.loads((root / "meta/info.json").read_text())

    version = info.get("codebase_version", "?")
    if not str(version).startswith("v3"):
        raise SystemExit(
            f"this converter targets LeRobot v3.x; info.json says {version!r}"
        )

    fps = int(info["fps"])
    feats = info["features"]
    all_cams = sorted(k for k, v in feats.items() if v.get("dtype") == "video")
    cams = args.cameras.split(",") if args.cameras else all_cams
    for c in cams:
        if c not in all_cams:
            raise SystemExit(f"no video feature {c!r}; have {all_cams}")

    s_dim = int(feats["observation.state"]["shape"][0])
    a_dim = int(feats["action"]["shape"][0])
    print(
        f"      v{version}  fps={fps}  state={s_dim}  action={a_dim}\n"
        f"      cameras={cams}  -> {h}x{w}"
    )

    print("[2/5] reading parquet ...")
    fr = read_frames(root)
    lengths, from_index, video_map = read_episodes(root, cams)
    n_rows, n_ep = fr["n_rows"], len(lengths)

    if n_rows != int(info["total_frames"]):
        raise SystemExit(
            f"data has {n_rows} rows, info.json says {info['total_frames']}"
        )
    if n_ep != int(info["total_episodes"]):
        raise SystemExit(
            f"{n_ep} episodes in the index, info.json says"
            f" {info['total_episodes']}"
        )
    if int(lengths.sum()) != n_rows:
        raise SystemExit(
            f"episode lengths sum to {int(lengths.sum())}, data has {n_rows}"
        )
    # The flat table must already be in episode order for `ep_offset` to mean
    # anything. Check rather than assume.
    expect_ep = np.repeat(np.arange(n_ep, dtype=np.int64), lengths)
    if not np.array_equal(fr["episode_index"], expect_ep):
        raise SystemExit(
            "data rows are not grouped by episode in index order; this"
            " converter's flat ep_offset/ep_len would be wrong"
        )
    print(f"      {n_rows} frames over {n_ep} episodes: {lengths.tolist()}")

    import h5py

    columns = [
        ("qpos", "float32", (s_dim,)),
        ("action", "float32", (a_dim,)),
        # ⚠ int32, matching the Mojo importer: the parquet is i64 and this
        # narrows deliberately — the column is per FRAME, so its width is paid
        # on every row, and no dataset has 2^31 tasks.
        #
        # ⚠⚠ EMPTY SHAPE, NOT `(1,)`. `ColumnSpec.__init__(name, dtype,
        # row_dim)` documents `row_dim=1` as a SCALAR COLUMN: it stores NO
        # trailing shape, so the manifest reads `column=task_index:int32` and
        # the dataset is RANK-1. Writing `(1,)` here produced
        # `column=task_index:int32:1` and a `(N, 1)` dataset — a 2-byte
        # manifest difference and a rank difference, both caught by the
        # byte-identity gate. The FORMAT is defined by the Mojo writer; this
        # file mirrors it.
        ("task_index", "int32", ()),
        ("images", "uint8", (len(cams), 3, h, w)),
    ]
    tasks = read_task_table(root)
    manifest = encode_manifest(
        env_id=f"lerobot/{args.repo}" if args.repo else f"lerobot/{root}",
        n_rows=n_rows,
        n_episodes=n_ep,
        seed=0,
        source_commit=args.revision or "",
        columns=columns,
        tasks=tasks,
    )

    # The output file is opened BEFORE the decode, so frames go straight into
    # it — see `_CamRows`. `.h5.tmp` + `os.replace` at the end still means an
    # interrupted run leaves no half-written store at the real path.
    cam_elems = 3 * h * w
    tmp = out.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        # Columns are rank-2 `[N, row_dim]`; the manifest carries the true
        # trailing shape (`TrajectoryStoreWriter._create_for`).
        f.create_dataset("qpos", data=fr["qpos"], dtype="f4")
        f.create_dataset("action", data=fr["action"], dtype="f4")
        # ⚠ RANK-1, matching a SCALAR column — see the ColumnSpec note above.
        f.create_dataset("task_index", data=fr["task_index"], dtype="i4")
        images = f.create_dataset(
            "images",
            shape=(n_rows, len(cams) * cam_elems),
            dtype="u1",
            chunks=(1, cam_elems),
        )

        print(f"[3/5] decoding video ({len(cams)} camera(s)) -> {tmp} ...")
        for slot, cam in enumerate(cams):
            decode_camera(
                root, cam, video_map, lengths, from_index, fps, h, w,
                _CamRows(images, slot, cam_elems),
            )

        print("[4/5] computing norm stats ...")
        q_mean, q_std = norm_stats(fr["qpos"])
        a_mean, a_std = norm_stats(fr["action"])
        print(f"      qpos   mean={np.round(q_mean, 3).tolist()}")
        print(f"      qpos   std ={np.round(q_std, 3).tolist()}")
        print(f"      action mean={np.round(a_mean, 3).tolist()}")
        print(f"      action std ={np.round(a_std, 3).tolist()}")

        print(f"[5/5] finishing {out} ...")
        f.create_dataset("ep_len", data=lengths.astype(np.int64))
        f.create_dataset("ep_offset", data=from_index.astype(np.int64))
        f.create_dataset("norm_qpos_mean", data=q_mean)
        f.create_dataset("norm_qpos_std", data=q_std)
        f.create_dataset("norm_action_mean", data=a_mean)
        f.create_dataset("norm_action_std", data=a_std)
        f.create_dataset(
            "__manifest__",
            data=np.frombuffer(manifest.encode("utf-8"), dtype=np.uint8),
        )
    os.replace(tmp, out)

    # Hashed in blocks. `read_bytes()` pulls the WHOLE store into one bytes
    # object — 7.12 GB for 50 episodes, and it was the last remaining place
    # this converter's memory scaled with the recording. Measured: RSS is flat
    # at 0.17 GB through the entire decode and spiked only here.
    _h = hashlib.sha256()
    with open(out, "rb") as _fh:
        for _block in iter(lambda: _fh.read(8 << 20), b""):
            _h.update(_block)
    sha = _h.hexdigest()[:16]
    print(
        f"\nwrote {out}\n"
        f"  {out.stat().st_size / 1e9:.2f} GB   sha256:{sha}\n"
        f"  images column: {len(cams)}*3*{h}*{w} = {len(cams) * 3 * h * w}"
        f" bytes/row"
    )
    (out.with_suffix(".json")).write_text(
        json.dumps(
            {
                "repo": args.repo,
                "cameras": cams,
                "height": h,
                "width": w,
                "fps": fps,
                "state_dim": s_dim,
                "action_dim": a_dim,
                "n_rows": n_rows,
                "episode_lengths": lengths.tolist(),
                "qpos_mean": q_mean.tolist(),
                "qpos_std": q_std.tolist(),
                "action_mean": a_mean.tolist(),
                "action_std": a_std.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
