"""LeWM PushT expert-trajectory dataset loader.

Loads ``quentinll/lewm-pusht`` from HuggingFace as used by the LeWorldModel
paper (arXiv:2603.19312). First-run downloads ``pusht_expert_train.h5.zst``
(~13 GB) via ``huggingface_hub`` and decompresses it to
``~/.cache/mojo_rl/lewm_pusht/pusht_expert_train.h5`` (~20–40 GB). All
subsequent loads are pure-Mojo via the libhdf5 FFI in ``mojo_rl/io/hdf5``.

Schema (from ``stable_worldmodel.data.formats.hdf5``):

Schema as MEASURED from the cached `pusht_expert_train.h5` (2026-08-05):
N_total = 2_336_736 frames over N_ep = 18_685 episodes.

================ ===========  ============================  ==========================
Dataset          Dtype        Shape                         Notes
================ ===========  ============================  ==========================
``ep_len``       int32        ``(N_ep,)``                   episode lengths
``ep_offset``    int64        ``(N_ep,)``                   start index in flat storage
``pixels``       uint8        ``(N_total, 224, 224, 3)``    HWC; we deliver permuted CHW
``action``       float32      ``(N_total, 2)``
``proprio``      float32      ``(N_total, 4)``
``state``        float32      ``(N_total, 7)``
``episode_idx``  int64        ``(N_total,)``                unused here
``step_idx``     int64        ``(N_total,)``                unused here
================ ===========  ============================  ==========================

⚠ An earlier version of this table claimed P=2 / S=5 and omitted
``episode_idx`` / ``step_idx``. The loader was never wrong — it reads every
dimension from the file (see ``__init__``: ``dims[1]`` for action/proprio/state)
— but the documented shapes were. Corrected after ``mojo_rl.data``'s foreign
ingest enumerated the real file.

Sampling matches ``stable_worldmodel.data.dataset.Dataset.__getitem__``
exactly:

- A clip is ``(ep_idx, start)`` with ``start ∈ [0, ep_len[ep] - span]`` and
  ``span = num_steps * frameskip``.
- For each clip, read columns over ``[g_start, g_start+span)`` where
  ``g_start = ep_offset[ep] + start``.
- Subsample non-action columns by ``frameskip`` → ``num_steps`` rows.
- Action stays dense — reshape to ``(num_steps, frameskip * A)`` so the
  predictor's effective action dim is ``frameskip * raw_action_dim`` (matches
  ``effective_act_dim`` in ``references/le-wm-main/train.py``).
- Permute pixels HWC → CHW.

Typical usage::

    var dataset = LewmPushTExpert(frameskip=5, num_steps=10)
    var buf = LewmPushTWindow(dataset)
    for i in range(len(dataset)):
        dataset.sample_window(i, buf)
        # buf.pixels  → [num_steps, 3, H, W] uint8
        # buf.action  → [num_steps, frameskip * action_dim] f32
        # buf.proprio → [num_steps, proprio_dim] f32
        # buf.state   → [num_steps, state_dim] f32
"""

from std.memory import alloc, unsafe_memcpy
from std.python import Python, PythonObject

from mojo_rl.io.hdf5 import (
    H5File,
    H5Dataset,
    H5T_INTEGER,
    H5T_FLOAT,
    H5T_SGN_NONE,
    H5T_SGN_2,
    hsize_t,
)
from mojo_rl.nn.core.ptr import mptr, untracked


comptime _HF_REPO = "quentinll/lewm-pusht"
comptime _HF_FILE = "pusht_expert_train.h5.zst"
comptime _CACHE_SUBDIR = ".cache/mojo_rl/lewm_pusht"

# The streaming download loop runs ENTIRELY inside Python (one exec + one
# call from Mojo). Driving it chunk-by-chunk from Mojo leaked every large
# per-iteration PythonObject (read chunks + decompress outputs) until the
# function returned: RSS stayed flat (macOS swapped the cold pages), but
# the process's physical footprint grew ~1:1 with the DECOMPRESSED volume
# — 23 GB footprint / 20.7 GB of swapped-out MALLOC_LARGE at 34% of the
# download, filling swap. In-Python, each chunk is freed per iteration.
# Features: in-place progress bar (compressed bytes, avg MB/s, ETA),
# reconnect-and-seek resume into the SAME decompressor (30 retries),
# F_NOCACHE (macOS) / periodic fsync+fadvise DONTNEED (Linux) so the
# multi-GB write also stays out of the OS page cache.
comptime _DL_HELPER_PY: StaticString = """
def stream_download(uri, tmp_path):
    import gc, os, time
    import zstandard
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    total = fs.info(uri)['size']
    f_out = open(tmp_path, 'wb')
    try:
        import fcntl
        fcntl.fcntl(f_out.fileno(), fcntl.F_NOCACHE, 1)
    except Exception:
        pass  # non-macOS: fsync+fadvise below covers it
    dobj = zstandard.ZstdDecompressor().decompressobj()
    CHUNK = 16 * 1024 * 1024
    MAX_RETRIES = 30
    f_in = fs.open(uri, 'rb')
    offset = 0
    retries = 0
    n_chunks = 0
    t0 = time.monotonic()
    while offset < total:
        try:
            chunk = f_in.read(CHUNK)
        except Exception as e:
            retries += 1
            if retries > MAX_RETRIES:
                raise RuntimeError(
                    '  [lewm_pusht] download failed after %d reconnects'
                    ' (last: %s); rerun to retry' % (MAX_RETRIES, e))
            print()
            print('  [lewm_pusht] read error at %d/%d bytes - reconnecting'
                  ' (retry %d/%d): %s' % (offset, total, retries,
                                          MAX_RETRIES, e))
            time.sleep(2.0)
            try:
                f_in.close()
            except Exception:
                pass
            f_in = fs.open(uri, 'rb')
            f_in.seek(offset)
            continue
        if not chunk:
            break
        offset += len(chunk)
        f_out.write(dobj.decompress(chunk))
        chunk = None
        n_chunks += 1
        if n_chunks % 32 == 0:  # every ~512 MB compressed
            f_out.flush()
            os.fsync(f_out.fileno())
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(f_out.fileno(), 0, 0,
                                 os.POSIX_FADV_DONTNEED)
            gc.collect()
        el = time.monotonic() - t0
        mbs = offset / 1e6 / el if el > 0 else 0.0
        eta = int((total - offset) / 1e6 / mbs) if mbs > 0 else 0
        pct = offset * 100 // total
        filled = offset * 30 // total
        bar = '#' * filled + '-' * (30 - filled)
        print('\\r  [lewm_pusht] [%s] %d%% | %.1f/%.1f GB | %.1f MB/s |'
              ' ETA %dm%02ds   ' % (bar, pct, offset / 1e9, total / 1e9,
                                    mbs, eta // 60, eta % 60),
              end='', flush=True)
    print()
    f_in.close()
    f_out.close()
"""


def _ensure_dataset_cached() raises -> String:
    """Resolve the cached ``.h5`` path, downloading + decompressing if needed.

    Streams the HF blob through zstd directly to disk — the compressed
    ``.zst`` never lands locally, so peak disk usage equals the final
    ``.h5`` size (~47 GB) rather than ~60 GB. Writes to a ``.tmp``
    file and renames on success, so a mid-stream failure does not leave
    a partial file masquerading as a valid cache hit.

    Returns the absolute path to ``pusht_expert_train.h5`` on disk.
    """
    var os = Python.import_module("os")
    var home = String(os.path.expanduser(PythonObject("~")))
    var cache = home + "/" + _CACHE_SUBDIR
    _ = os.makedirs(PythonObject(cache), exist_ok=True)

    var h5_path = cache + "/pusht_expert_train.h5"
    if Bool(os.path.exists(PythonObject(h5_path))):
        return h5_path

    var tmp_path = h5_path + ".tmp"
    if Bool(os.path.exists(PythonObject(tmp_path))):
        _ = os.remove(PythonObject(tmp_path))

    print("  [lewm_pusht] no cache hit; streaming", _HF_FILE)
    print("  [lewm_pusht] (~13 GB over HTTP, decompressing to ~47 GB on disk)")

    # One exec + one call: the chunk loop lives in Python (see the
    # _DL_HELPER_PY note — Mojo-driven per-chunk PythonObjects leaked the
    # whole decompressed volume into swap until the function returned).
    var builtins = Python.import_module("builtins")
    var hf_uri = String("datasets/") + _HF_REPO + "/" + _HF_FILE
    var ns = builtins.dict()
    _ = builtins.exec(PythonObject(_DL_HELPER_PY), ns)
    _ = ns[PythonObject("stream_download")](
        PythonObject(hf_uri), PythonObject(tmp_path)
    )

    _ = os.rename(PythonObject(tmp_path), PythonObject(h5_path))
    print("  [lewm_pusht] cached → ", h5_path)
    return h5_path


struct LewmPushTWindow(Movable):
    """Pre-allocated buffers for one sample window — reuse across calls.

    Construct once per training loop, pass into ``sample_window`` repeatedly.
    The dataset writes into these buffers; the caller reads from them.

    Buffer shapes:

    - ``pixels``  → ``[num_steps, H, W, 3]`` uint8 (HWC, no permute)
    - ``action``  → ``[num_steps, frameskip * action_dim]`` float32
    - ``proprio`` → ``[num_steps, proprio_dim]`` float32
    - ``state``   → ``[num_steps, state_dim]`` float32

    Pixels are kept in the HWC layout that HDF5 stores natively — the
    HWC→CHW permute + uint8→fp32 normalize happens on the GPU via
    ``pixels_uint8_to_fp32_kernel`` after the host→device DMA.
    """

    var num_steps: Int
    var frameskip: Int
    var pixel_h: Int
    var pixel_w: Int
    var action_dim: Int
    var proprio_dim: Int
    var state_dim: Int

    var pixels: Pointer[Scalar[DType.uint8], MutUntrackedOrigin]
    """``[num_steps, H, W, 3]`` — native HDF5 layout; HWC."""
    var pixels_dense: Pointer[Scalar[DType.uint8], MutUntrackedOrigin]
    """``[num_steps * frameskip, H, W, 3]`` — scratch buffer for one
    dense HDF5 read. ``H5Sselect_hyperslab`` with ``stride>1`` is
    pathologically slow (~15× a contiguous read of the same chunk),
    so ``sample_window`` reads the dense span into this buffer and
    memcpys every ``frameskip``-th frame into ``pixels``.
    """
    var action: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    """``[num_steps, frameskip * action_dim]`` — dense actions, reshaped."""
    var proprio: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    """``[num_steps, proprio_dim]`` — subsampled by frameskip."""
    var state: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    """``[num_steps, state_dim]`` — subsampled by frameskip."""

    def __init__(
        out self,
        *,
        num_steps: Int,
        frameskip: Int,
        pixel_h: Int,
        pixel_w: Int,
        action_dim: Int,
        proprio_dim: Int,
        state_dim: Int,
    ):
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.pixel_h = pixel_h
        self.pixel_w = pixel_w
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.state_dim = state_dim

        var n_pixels = num_steps * 3 * pixel_h * pixel_w
        self.pixels = untracked(alloc[Scalar[DType.uint8]](n_pixels))
        var n_pixels_dense = num_steps * frameskip * 3 * pixel_h * pixel_w
        self.pixels_dense = untracked(alloc[Scalar[DType.uint8]](n_pixels_dense))
        self.action = untracked(alloc[Scalar[DType.float32]](
            num_steps * frameskip * action_dim
        ))
        self.proprio = untracked(alloc[Scalar[DType.float32]](
            num_steps * proprio_dim
        ))
        self.state = untracked(alloc[Scalar[DType.float32]](num_steps * state_dim))

    def __deinit__(deinit self):
        self.pixels.unsafe_free()
        self.pixels_dense.unsafe_free()
        self.action.unsafe_free()
        self.proprio.unsafe_free()
        self.state.unsafe_free()


struct LewmPushTExpert(Movable, Sized):
    """Read-only dataset over ``pusht_expert_train.h5``.

    All small columns (``ep_len``, ``ep_offset``, ``action``, ``proprio``,
    ``state``) are slurped into host RAM on construction — total well under
    100 MB. Pixels stay on disk and are read per-sample via strided HDF5
    hyperslab reads.
    """

    var _file: H5File
    var _dset_pixels: H5Dataset
    var _dset_action: H5Dataset
    var _dset_proprio: H5Dataset
    var _dset_state: H5Dataset

    var ep_len: List[Int32]
    """``[n_episodes]`` — per-episode length."""
    var ep_offset: List[Int64]
    """``[n_episodes]`` — flat-storage start offset per episode."""
    var action_flat: List[Float32]
    """Flat ``[n_total_frames, action_dim]`` — row-major."""
    var proprio_flat: List[Float32]
    """Flat ``[n_total_frames, proprio_dim]``."""
    var state_flat: List[Float32]
    """Flat ``[n_total_frames, state_dim]``."""

    var n_episodes: Int
    var n_total_frames: Int
    var pixel_h: Int
    var pixel_w: Int
    var action_dim: Int
    var proprio_dim: Int
    var state_dim: Int

    var frameskip: Int
    var num_steps: Int
    var span: Int
    """``num_steps * frameskip`` — window length on the dense frame axis."""

    var clip_ep_idx: List[Int32]
    """Clip index → episode id (matches Python ``Dataset.clip_indices[0]``)."""
    var clip_start: List[Int32]
    """Clip index → start offset within episode (matches ``Dataset.clip_indices[1]``)."""

    def __init__(
        out self,
        *,
        frameskip: Int = 5,
        num_steps: Int = 1,
        var path: String = String(""),
    ) raises:
        """Open the dataset.

        Args:
            frameskip: Stride between observation samples (matches LeWM's
                ``frameskip`` config; PushT uses 5).
            num_steps: Number of observation steps per sample (history +
                predictions).
            path: Optional override of the cached H5 path. Empty string
                triggers the standard HF download flow into
                ``~/.cache/mojo_rl/lewm_pusht/``. Used by tests to point
                at a synthetic fixture.
        """
        var h5_path: String
        if path.byte_length() > 0:
            h5_path = path^
        else:
            h5_path = _ensure_dataset_cached()
        self._file = H5File(h5_path)
        self._dset_pixels = self._file.open_dataset(String("pixels"))
        self._dset_action = self._file.open_dataset(String("action"))
        self._dset_proprio = self._file.open_dataset(String("proprio"))
        self._dset_state = self._file.open_dataset(String("state"))

        # ── shape introspection ───────────────────────────────────────
        if self._dset_pixels.ndim() != 4:
            raise Error("pixels: expected rank 4, got " + String(self._dset_pixels.ndim()))
        if self._dset_action.ndim() != 2:
            raise Error("action: expected rank 2")
        if self._dset_proprio.ndim() != 2:
            raise Error("proprio: expected rank 2")
        if self._dset_state.ndim() != 2:
            raise Error("state: expected rank 2")
        if self._dset_pixels.dtype_class != H5T_INTEGER \
            or self._dset_pixels.elem_size != 1:
            raise Error("pixels: expected uint8")
        if self._dset_action.dtype_class != H5T_FLOAT \
            or self._dset_action.elem_size != 4:
            raise Error("action: expected float32")
        if self._dset_proprio.dtype_class != H5T_FLOAT \
            or self._dset_proprio.elem_size != 4:
            raise Error("proprio: expected float32")
        if self._dset_state.dtype_class != H5T_FLOAT \
            or self._dset_state.elem_size != 4:
            raise Error("state: expected float32")

        self.n_total_frames = Int(self._dset_pixels.dims[0])
        self.pixel_h = Int(self._dset_pixels.dims[1])
        self.pixel_w = Int(self._dset_pixels.dims[2])
        if Int(self._dset_pixels.dims[3]) != 3:
            raise Error(
                "pixels: expected 3 channels, got "
                + String(Int(self._dset_pixels.dims[3]))
            )
        self.action_dim = Int(self._dset_action.dims[1])
        self.proprio_dim = Int(self._dset_proprio.dims[1])
        self.state_dim = Int(self._dset_state.dims[1])

        # ── slurp small columns ───────────────────────────────────────
        # ep_len (int32)
        var ep_len_ds = self._file.open_dataset(String("ep_len"))
        if ep_len_ds.dtype_class != H5T_INTEGER \
            or ep_len_ds.elem_size != 4 \
            or ep_len_ds.signedness != H5T_SGN_2:
            raise Error("ep_len: expected int32")
        self.n_episodes = Int(ep_len_ds.dims[0])
        var ep_len_buf = mptr(alloc[Scalar[DType.int32]](self.n_episodes))
        ep_len_ds.read_all[DType.int32](ep_len_buf)
        self.ep_len = List[Int32](capacity=self.n_episodes)
        for i in range(self.n_episodes):
            self.ep_len.append(Int32(ep_len_buf[unsafe_offset=i]))
        ep_len_buf.unsafe_free()

        # ep_offset (int64)
        var ep_off_ds = self._file.open_dataset(String("ep_offset"))
        if ep_off_ds.dtype_class != H5T_INTEGER \
            or ep_off_ds.elem_size != 8 \
            or ep_off_ds.signedness != H5T_SGN_2:
            raise Error("ep_offset: expected int64")
        if Int(ep_off_ds.dims[0]) != self.n_episodes:
            raise Error("ep_offset / ep_len length mismatch")
        var ep_off_buf = mptr(alloc[Scalar[DType.int64]](self.n_episodes))
        ep_off_ds.read_all[DType.int64](ep_off_buf)
        self.ep_offset = List[Int64](capacity=self.n_episodes)
        for i in range(self.n_episodes):
            self.ep_offset.append(Int64(ep_off_buf[unsafe_offset=i]))
        ep_off_buf.unsafe_free()

        # action (flat)
        var n_act = self.n_total_frames * self.action_dim
        var act_buf = mptr(alloc[Scalar[DType.float32]](n_act))
        self._dset_action.read_all[DType.float32](act_buf)
        self.action_flat = List[Float32](capacity=n_act)
        for i in range(n_act):
            self.action_flat.append(Float32(act_buf[unsafe_offset=i]))
        act_buf.unsafe_free()

        # proprio (flat)
        var n_pro = self.n_total_frames * self.proprio_dim
        var pro_buf = mptr(alloc[Scalar[DType.float32]](n_pro))
        self._dset_proprio.read_all[DType.float32](pro_buf)
        self.proprio_flat = List[Float32](capacity=n_pro)
        for i in range(n_pro):
            self.proprio_flat.append(Float32(pro_buf[unsafe_offset=i]))
        pro_buf.unsafe_free()

        # state (flat)
        var n_st = self.n_total_frames * self.state_dim
        var st_buf = mptr(alloc[Scalar[DType.float32]](n_st))
        self._dset_state.read_all[DType.float32](st_buf)
        self.state_flat = List[Float32](capacity=n_st)
        for i in range(n_st):
            self.state_flat.append(Float32(st_buf[unsafe_offset=i]))
        st_buf.unsafe_free()

        # ── sampling params + clip index ──────────────────────────────
        if frameskip <= 0:
            raise Error("frameskip must be positive")
        if num_steps <= 0:
            raise Error("num_steps must be positive")
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip

        var ep_idx_list = List[Int32]()
        var start_list = List[Int32]()
        for ep in range(self.n_episodes):
            var length = Int(self.ep_len[ep])
            if length >= self.span:
                for start in range(length - self.span + 1):
                    ep_idx_list.append(Int32(ep))
                    start_list.append(Int32(start))
        self.clip_ep_idx = ep_idx_list^
        self.clip_start = start_list^

    def __len__(self) -> Int:
        """Number of valid clip windows."""
        return len(self.clip_ep_idx)

    def make_window(self) raises -> LewmPushTWindow:
        """Allocate a sample-window buffer sized for this dataset.

        Convenience wrapper so callers don't need to plumb dims manually.
        """
        return LewmPushTWindow(
            num_steps=self.num_steps,
            frameskip=self.frameskip,
            pixel_h=self.pixel_h,
            pixel_w=self.pixel_w,
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            state_dim=self.state_dim,
        )

    def sample_window(
        self, idx: Int, mut into: LewmPushTWindow
    ) raises:
        """Fill ``into`` with the ``idx``-th clip window.

        Matches ``stable_worldmodel.data.dataset.Dataset.__getitem__`` byte
        for byte (modulo the action reshape, which is deferred — actions
        live in ``into.action`` as ``[num_steps, frameskip*action_dim]``
        already).
        """
        if idx < 0 or idx >= len(self.clip_ep_idx):
            raise Error("sample_window: idx out of range")
        if into.num_steps != self.num_steps \
            or into.frameskip != self.frameskip \
            or into.pixel_h != self.pixel_h \
            or into.pixel_w != self.pixel_w \
            or into.action_dim != self.action_dim \
            or into.proprio_dim != self.proprio_dim \
            or into.state_dim != self.state_dim:
            raise Error("sample_window: buffer shape mismatch")

        var ep_idx = Int(self.clip_ep_idx[idx])
        var start_in_ep = Int(self.clip_start[idx])
        var g_start = Int(self.ep_offset[ep_idx]) + start_in_ep

        # ── pixels: dense read then strided unsafe_memcpy ───────────────────────
        # `H5Sselect_hyperslab` with stride>1 is pathologically slow in
        # libhdf5 (~15× the cost of a contiguous read of the same chunk
        # range, measured on the PushT 100-frame chunks). So we read the
        # full ``num_steps * frameskip`` dense span into a scratch buffer
        # and copy every ``frameskip``-th frame into ``into.pixels``.
        # Output layout is HWC (num_steps, H, W, 3) — native HDF5 layout.
        # The HWC→CHW permute + uint8→fp32 normalize is deferred to a GPU
        # kernel (see `pixels_uint8_to_fp32_kernel`).
        self._dset_pixels.read_range[DType.uint8](
            g_start, g_start + self.span, into.pixels_dense.as_unsafe_any_origin()
        )
        var pix_per_frame = self.pixel_h * self.pixel_w * 3
        for k in range(self.num_steps):
            unsafe_memcpy(
                dest=into.pixels.unsafe_offset(k * pix_per_frame),
                src=into.pixels_dense.unsafe_offset(k * self.frameskip * pix_per_frame),
                count=pix_per_frame,
            )

        # ── action: copy DENSE span from flat host buffer ──────────────
        # Output shape: (num_steps, frameskip * action_dim) — same data
        # as a (span, action_dim) block, just reinterpreted.
        var act_total = self.span * self.action_dim
        for i in range(act_total):
            into.action[unsafe_offset=i] = self.action_flat[g_start * self.action_dim + i]

        # ── proprio: subsample by frameskip from flat host buffer ──────
        for n in range(self.num_steps):
            var src_row = g_start + n * self.frameskip
            var src_base = src_row * self.proprio_dim
            var dst_base = n * self.proprio_dim
            for j in range(self.proprio_dim):
                into.proprio[unsafe_offset=dst_base + j] = self.proprio_flat[
                    src_base + j
                ]

        # ── state: subsample by frameskip from flat host buffer ────────
        for n in range(self.num_steps):
            var src_row = g_start + n * self.frameskip
            var src_base = src_row * self.state_dim
            var dst_base = n * self.state_dim
            for j in range(self.state_dim):
                into.state[unsafe_offset=dst_base + j] = self.state_flat[src_base + j]

    def sample_clip_pixels_uint8(
        self,
        idx: Int,
        pixels_dst: Pointer[Scalar[DType.uint8], MutAnyOrigin],
        actions_dst: Pointer[Scalar[DType.float32], MutAnyOrigin],
        dense_scratch: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    ) raises:
        """Hot-path sample for the trainer's batch loop.

        Unlike ``sample_window`` this skips proprio/state and the
        ``LewmPushTWindow.pixels`` intermediate: pixels stream from
        libhdf5 into ``dense_scratch`` then strided-unsafe_memcpy directly
        into the caller's batch slot in ``pixels_dst``; actions
        unsafe_memcpy in one shot from the already-slurped flat host buffer
        into ``actions_dst``. Used by ``PushTOfflineSampler``.

        Args:
            idx: Clip index, ``0 <= idx < len(self.clip_ep_idx)``.
            pixels_dst: ``[num_steps, H, W, 3]`` uint8, caller-owned.
            actions_dst: ``[num_steps, frameskip * action_dim]`` fp32.
            dense_scratch: ``[span, H, W, 3]`` uint8 scratch
                (``span = num_steps * frameskip``). Reused across calls;
                typically the ``pixels_dense`` field of a window.
        """
        if idx < 0 or idx >= len(self.clip_ep_idx):
            raise Error("sample_clip_pixels_uint8: idx out of range")

        var ep_idx = Int(self.clip_ep_idx[idx])
        var start_in_ep = Int(self.clip_start[idx])
        var g_start = Int(self.ep_offset[ep_idx]) + start_in_ep

        # Pixels: one dense hyperslab read, then 4 strided memcpys.
        self._dset_pixels.read_range[DType.uint8](
            g_start, g_start + self.span, dense_scratch
        )
        var pix_per_frame = self.pixel_h * self.pixel_w * 3
        for k in range(self.num_steps):
            unsafe_memcpy(
                dest=pixels_dst.unsafe_offset(k * pix_per_frame),
                src=dense_scratch.unsafe_offset(k * self.frameskip * pix_per_frame),
                count=pix_per_frame,
            )

        # Actions: contiguous fp32 unsafe_memcpy from slurped host buffer.
        var act_total = self.span * self.action_dim
        unsafe_memcpy(
            dest=actions_dst,
            src=self.action_flat.unsafe_ptr().unsafe_offset(g_start * self.action_dim),
            count=act_total,
        )
