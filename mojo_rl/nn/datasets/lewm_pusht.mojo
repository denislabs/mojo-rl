"""LeWM PushT expert-trajectory dataset loader.

Loads ``quentinll/lewm-pusht`` from HuggingFace as used by the LeWorldModel
paper (arXiv:2603.19312). First-run downloads ``pusht_expert_train.h5.zst``
(~13 GB) via ``huggingface_hub`` and decompresses it to
``~/.cache/mojo_rl/lewm_pusht/pusht_expert_train.h5`` (~20–40 GB). All
subsequent loads are pure-Mojo via the libhdf5 FFI in ``mojo_rl/io/hdf5``.

Schema (from ``stable_worldmodel.data.formats.hdf5``):

================ ===========  =========================  =============================
Dataset          Dtype        Shape                      Notes
================ ===========  =========================  =============================
``ep_len``       int32        ``(N_ep,)``                episode lengths
``ep_offset``    int64        ``(N_ep,)``                start index in flat storage
``pixels``       uint8        ``(N_total, H, W, 3)``     HWC; we deliver permuted CHW
``action``       float32      ``(N_total, A)``           A=2 for PushT
``proprio``      float32      ``(N_total, P)``           P=2 for PushT (agent xy)
``state``        float32      ``(N_total, S)``           S=5 for PushT
================ ===========  =========================  =============================

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

from std.memory import alloc, memcpy
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
from mojo_rl.nn.core.module import mptr


comptime _HF_REPO = "quentinll/lewm-pusht"
comptime _HF_FILE = "pusht_expert_train.h5.zst"
comptime _CACHE_SUBDIR = ".cache/mojo_rl/lewm_pusht"


def _ensure_dataset_cached() raises -> String:
    """Resolve the cached ``.h5`` path, downloading + decompressing if needed.

    Streams the HF blob through zstd directly to disk — the compressed
    ``.zst`` never lands locally, so peak disk usage equals the final
    ``.h5`` size (~15–25 GB) rather than ~28–38 GB. Writes to a ``.tmp``
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
    print("  [lewm_pusht] (~13 GB over HTTP, decompressing to ~15–25 GB on disk)")

    var hf = Python.import_module("huggingface_hub")
    var zstandard = Python.import_module("zstandard")
    var builtins = Python.import_module("builtins")

    var fs = hf.HfFileSystem()
    var hf_uri = String("datasets/") + _HF_REPO + "/" + _HF_FILE
    var f_in = fs.open(PythonObject(hf_uri), PythonObject("rb"))
    var f_out = builtins.open(PythonObject(tmp_path), PythonObject("wb"))
    var dctx = zstandard.ZstdDecompressor()
    _ = dctx.copy_stream(f_in, f_out)
    _ = f_in.close()
    _ = f_out.close()

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

    var pixels: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin]
    """``[num_steps, H, W, 3]`` — native HDF5 layout; HWC."""
    var pixels_dense: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin]
    """``[num_steps * frameskip, H, W, 3]`` — scratch buffer for one
    dense HDF5 read. ``H5Sselect_hyperslab`` with ``stride>1`` is
    pathologically slow (~15× a contiguous read of the same chunk),
    so ``sample_window`` reads the dense span into this buffer and
    memcpys every ``frameskip``-th frame into ``pixels``.
    """
    var action: UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    """``[num_steps, frameskip * action_dim]`` — dense actions, reshaped."""
    var proprio: UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    """``[num_steps, proprio_dim]`` — subsampled by frameskip."""
    var state: UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
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
        self.pixels = mptr(alloc[Scalar[DType.uint8]](n_pixels))
        var n_pixels_dense = num_steps * frameskip * 3 * pixel_h * pixel_w
        self.pixels_dense = mptr(alloc[Scalar[DType.uint8]](n_pixels_dense))
        self.action = mptr(alloc[Scalar[DType.float32]](
            num_steps * frameskip * action_dim
        ))
        self.proprio = mptr(alloc[Scalar[DType.float32]](
            num_steps * proprio_dim
        ))
        self.state = mptr(alloc[Scalar[DType.float32]](num_steps * state_dim))

    def __del__(deinit self):
        self.pixels.free()
        self.pixels_dense.free()
        self.action.free()
        self.proprio.free()
        self.state.free()


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
            self.ep_len.append(Int32(ep_len_buf[i]))
        ep_len_buf.free()

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
            self.ep_offset.append(Int64(ep_off_buf[i]))
        ep_off_buf.free()

        # action (flat)
        var n_act = self.n_total_frames * self.action_dim
        var act_buf = mptr(alloc[Scalar[DType.float32]](n_act))
        self._dset_action.read_all[DType.float32](act_buf)
        self.action_flat = List[Float32](capacity=n_act)
        for i in range(n_act):
            self.action_flat.append(Float32(act_buf[i]))
        act_buf.free()

        # proprio (flat)
        var n_pro = self.n_total_frames * self.proprio_dim
        var pro_buf = mptr(alloc[Scalar[DType.float32]](n_pro))
        self._dset_proprio.read_all[DType.float32](pro_buf)
        self.proprio_flat = List[Float32](capacity=n_pro)
        for i in range(n_pro):
            self.proprio_flat.append(Float32(pro_buf[i]))
        pro_buf.free()

        # state (flat)
        var n_st = self.n_total_frames * self.state_dim
        var st_buf = mptr(alloc[Scalar[DType.float32]](n_st))
        self._dset_state.read_all[DType.float32](st_buf)
        self.state_flat = List[Float32](capacity=n_st)
        for i in range(n_st):
            self.state_flat.append(Float32(st_buf[i]))
        st_buf.free()

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

        # ── pixels: dense read then strided memcpy ───────────────────────
        # `H5Sselect_hyperslab` with stride>1 is pathologically slow in
        # libhdf5 (~15× the cost of a contiguous read of the same chunk
        # range, measured on the PushT 100-frame chunks). So we read the
        # full ``num_steps * frameskip`` dense span into a scratch buffer
        # and copy every ``frameskip``-th frame into ``into.pixels``.
        # Output layout is HWC (num_steps, H, W, 3) — native HDF5 layout.
        # The HWC→CHW permute + uint8→fp32 normalize is deferred to a GPU
        # kernel (see `pixels_uint8_to_fp32_kernel`).
        self._dset_pixels.read_range[DType.uint8](
            g_start, g_start + self.span, into.pixels_dense
        )
        var pix_per_frame = self.pixel_h * self.pixel_w * 3
        for k in range(self.num_steps):
            memcpy(
                dest=into.pixels + k * pix_per_frame,
                src=into.pixels_dense + k * self.frameskip * pix_per_frame,
                count=pix_per_frame,
            )

        # ── action: copy DENSE span from flat host buffer ──────────────
        # Output shape: (num_steps, frameskip * action_dim) — same data
        # as a (span, action_dim) block, just reinterpreted.
        var act_total = self.span * self.action_dim
        for i in range(act_total):
            into.action[i] = self.action_flat[g_start * self.action_dim + i]

        # ── proprio: subsample by frameskip from flat host buffer ──────
        for n in range(self.num_steps):
            var src_row = g_start + n * self.frameskip
            var src_base = src_row * self.proprio_dim
            var dst_base = n * self.proprio_dim
            for j in range(self.proprio_dim):
                into.proprio[dst_base + j] = self.proprio_flat[
                    src_base + j
                ]

        # ── state: subsample by frameskip from flat host buffer ────────
        for n in range(self.num_steps):
            var src_row = g_start + n * self.frameskip
            var src_base = src_row * self.state_dim
            var dst_base = n * self.state_dim
            for j in range(self.state_dim):
                into.state[dst_base + j] = self.state_flat[src_base + j]

    def sample_clip_pixels_uint8(
        self,
        idx: Int,
        pixels_dst: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
        actions_dst: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
        dense_scratch: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
    ) raises:
        """Hot-path sample for the trainer's batch loop.

        Unlike ``sample_window`` this skips proprio/state and the
        ``LewmPushTWindow.pixels`` intermediate: pixels stream from
        libhdf5 into ``dense_scratch`` then strided-memcpy directly
        into the caller's batch slot in ``pixels_dst``; actions
        memcpy in one shot from the already-slurped flat host buffer
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
            memcpy(
                dest=pixels_dst + k * pix_per_frame,
                src=dense_scratch + k * self.frameskip * pix_per_frame,
                count=pix_per_frame,
            )

        # Actions: contiguous fp32 memcpy from slurped host buffer.
        var act_total = self.span * self.action_dim
        memcpy(
            dest=actions_dst,
            src=self.action_flat.unsafe_ptr() + g_start * self.action_dim,
            count=act_total,
        )
