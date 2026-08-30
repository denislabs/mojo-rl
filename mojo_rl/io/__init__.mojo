# +--------------------------------------------------------------------------+ #
# | mojo-rl I/O bindings
# +--------------------------------------------------------------------------+ #
"""I/O backends for mojo-rl datasets.

- ``hdf5``    — a Mojo FFI over libhdf5; the `TrajectoryStore` file format.
- ``parquet`` — a native Parquet reader (thrift + snappy + RLE + PLAIN).
- ``video``   — sequential H.264 decode by piping `ffmpeg`.
- ``json``    — a small JSON reader.
- ``image``   — Pillow-compatible bilinear resize.
- ``proc``    — `popen`/`fread`/`pclose`, since the stdlib has no subprocess.
- ``fetch``   — resumable HTTP download to a local cache.
- ``serial``  — raw-mode tty over libc (the SO-101 Feetech bus).

⚠ THE LAST FOUR EXIST TO GET PYTHON OUT OF THE DATA PATH. Importing a LeRobot
v3 dataset used to need `huggingface_hub`, `pyarrow`, `imageio`, `Pillow`,
`numpy` and `h5py` in a separate process; `mojo_rl/data/lerobot.mojo` now does
it with these plus `curl` and `ffmpeg` on PATH, and produces a byte-identical
store (`tests/data/test_lerobot_import.mojo`).
"""
