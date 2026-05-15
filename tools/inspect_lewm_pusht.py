"""Print the schema of the cached ``pusht_expert_train.h5``.

Triggers the same download + decompress flow as ``LewmPushTExpert`` (no
network call if the file is already cached), then walks the HDF5 file
and prints every dataset's dtype, shape, chunks, and compression filter.
Useful for confirming the dataset's structure (and the specific
``hdf5plugin`` filter used) before pointing the Mojo loader at it.

Usage:
    pixi run python tools/inspect_lewm_pusht.py
"""

from __future__ import annotations

import pathlib
import sys

import h5py
import hdf5plugin  # noqa: F401  — registers Blosc/LZ4/ZSTD/... filters at import
from huggingface_hub import HfFileSystem
import zstandard


HF_REPO = "quentinll/lewm-pusht"
HF_FILE = "pusht_expert_train.h5.zst"
CACHE_DIR = pathlib.Path.home() / ".cache/mojo_rl/lewm_pusht"


def ensure_cached() -> pathlib.Path:
    """Stream-download + decompress in one pass so the compressed ``.zst``
    never lands on disk. Peak usage ≈ final ``.h5`` size (~15-25 GB), not
    the ~28-38 GB you'd hit if both files coexisted."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h5_path = CACHE_DIR / "pusht_expert_train.h5"
    if h5_path.exists():
        print(f"[cache] {h5_path} already present")
        return h5_path

    tmp_path = h5_path.with_suffix(h5_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[stream] {HF_REPO}/{HF_FILE}  →  {h5_path}")
    print("[stream] no .zst written; decompressing on the fly ...")
    fs = HfFileSystem()
    hf_uri = f"datasets/{HF_REPO}/{HF_FILE}"
    try:
        with fs.open(hf_uri, "rb") as f_in, open(tmp_path, "wb") as f_out:
            zstandard.ZstdDecompressor().copy_stream(f_in, f_out)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    tmp_path.rename(h5_path)
    return h5_path


def describe(h5_path: pathlib.Path) -> None:
    size_gb = h5_path.stat().st_size / 1024 ** 3
    print(f"\n[file]  {h5_path}  ({size_gb:.2f} GB)")
    with h5py.File(h5_path, "r") as f:
        names = sorted(f.keys())
        col_w = max(len(n) for n in names)
        print(
            f"\n  {'name'.ljust(col_w)}  {'dtype':<8}  {'shape':<28}  "
            f"{'chunks':<22}  filter"
        )
        print("  " + "-" * (col_w + 2 + 8 + 2 + 28 + 2 + 22 + 2 + 20))
        for name in names:
            obj = f[name]
            if not isinstance(obj, h5py.Dataset):
                print(f"  {name.ljust(col_w)}  (group)")
                continue
            shape = str(tuple(obj.shape))
            chunks = str(obj.chunks) if obj.chunks else "(contiguous)"
            # Compression filter — could be a string ("gzip", "lzf") or
            # a numeric plugin id (when hdf5plugin is involved).
            if obj.compression is not None:
                filt = f"{obj.compression}"
                if obj.compression_opts is not None:
                    filt += f" opts={obj.compression_opts}"
            else:
                # check for raw filter IDs (hdf5plugin compressors)
                ids = [obj.id.get_create_plist().get_filter(i)[0]
                       for i in range(obj.id.get_create_plist().get_nfilters())]
                filt = "ids=" + ",".join(str(i) for i in ids) if ids else "none"
            print(
                f"  {name.ljust(col_w)}  {str(obj.dtype):<8}  {shape:<28}  "
                f"{chunks:<22}  {filt}"
            )

        # Quick stats from ep_len / ep_offset
        if "ep_len" in f and "ep_offset" in f:
            ep_len = f["ep_len"][:]
            print(
                f"\n  episodes: {len(ep_len)}, "
                f"total frames: {int(ep_len.sum())}, "
                f"min/median/max len: "
                f"{int(ep_len.min())} / {int(sorted(ep_len)[len(ep_len)//2])} / "
                f"{int(ep_len.max())}"
            )


def main() -> int:
    try:
        path = ensure_cached()
    except Exception as exc:  # noqa: BLE001
        print(f"[error] cache/download failed: {exc}", file=sys.stderr)
        return 1
    describe(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
