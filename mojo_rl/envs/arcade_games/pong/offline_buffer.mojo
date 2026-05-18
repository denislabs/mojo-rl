"""Pong pixel-obs offline trajectory buffer.

Stores Pong pixel observations as flat uint8 sequences with episode markers.
At sample time, returns a contiguous (B, T) window of frames + one-hot actions
as fp32 in [0, 1].

Implements ``mojo_rl.core.offline_buffer.OfflineBuffer`` so the LeWM offline
trainer (or any other consumer of pixel-obs offline data) can plug Pong in
generically.

File format LWMP v1 (little-endian):
    Bytes [0..3]:    "LWMP" magic
    Bytes [4..7]:    uint32 version (1)
    Bytes [8..15]:   uint64 n_frames
    Bytes [16..19]:  uint32 obs_c  (=4)
    Bytes [20..23]:  uint32 obs_h  (=84)
    Bytes [24..27]:  uint32 obs_w  (=84)
    Bytes [28..31]:  uint32 num_actions  (=3)
    Bytes [32..63]:  reserved (zero)

    Then in order:
      n_frames * obs_c * obs_h * obs_w bytes  — pixel data, layout (N, C, H, W).
      n_frames bytes                          — action index in {0, 1, 2}.
      n_frames bytes                          — done flag in {0, 1}.

Sampling protocol: pick start_t ∈ [0, n_frames - T]. Reject windows where any
of dones[start_t : start_t + T - 1] is set (episode boundary mid-window). The
final frame may be done (it's a valid prediction target).

Layout in fp32 sample tensors:
    pixels:  (B, T, C * H * W)  in [0, 1] (cast from uint8 / 255).
    actions: (B, T, num_actions) one-hot fp32.

Originally lived at ``mojo_rl/experimental/lewm/pong_buffer.mojo`` as
``PongBuffer``; that module is now a re-export shim.
"""

from std.memory import alloc
from std.random import random_float64

from mojo_rl.core.offline_buffer import OfflineBuffer

comptime PONG_OBS_C: Int = 4
comptime PONG_OBS_H: Int = 84
comptime PONG_OBS_W: Int = 84
comptime PONG_NUM_ACTIONS: Int = 3
comptime PONG_FRAME_BYTES: Int = PONG_OBS_C * PONG_OBS_H * PONG_OBS_W  # 28224
comptime PONG_OBS_DIM: Int = PONG_FRAME_BYTES

comptime PONG_BUFFER_MAGIC: String = "LWMP"
comptime PONG_BUFFER_VERSION: UInt32 = 1
comptime PONG_BUFFER_HEADER_BYTES: Int = 64


# ============================================================================
# Endianness helpers
# ============================================================================


@always_inline
def _write_uint32_le(mut buf: List[UInt8], value: UInt32):
    buf.append(UInt8(value & 0xFF))
    buf.append(UInt8((value >> 8) & 0xFF))
    buf.append(UInt8((value >> 16) & 0xFF))
    buf.append(UInt8((value >> 24) & 0xFF))


@always_inline
def _write_uint64_le(mut buf: List[UInt8], value: UInt64):
    for i in range(8):
        buf.append(UInt8((value >> UInt64(8 * i)) & 0xFF))


@always_inline
def _read_uint32_le(data: List[UInt8], pos: Int) -> UInt32:
    return (
        UInt32(data[pos])
        | (UInt32(data[pos + 1]) << 8)
        | (UInt32(data[pos + 2]) << 16)
        | (UInt32(data[pos + 3]) << 24)
    )


@always_inline
def _read_uint64_le(data: List[UInt8], pos: Int) -> UInt64:
    var v: UInt64 = 0
    for i in range(8):
        v |= UInt64(data[pos + i]) << UInt64(8 * i)
    return v


# ============================================================================
# PongOfflineBuffer
# ============================================================================


struct PongOfflineBuffer(Movable, OfflineBuffer):
    """Flat offline-RL buffer for Pong pixel trajectories.

    Owns three parallel uint8 arrays:
      - frames:  capacity * PONG_FRAME_BYTES bytes
      - actions: capacity bytes
      - dones:   capacity bytes

    Frames are stored in [0, 255]. Pre-allocate `capacity` frames; `n_frames`
    tracks the actual count of appended steps.

    Not thread-safe. Single-producer, single-consumer (collection then training).
    """

    # Frames are stored as (C, H, W) so we deliver CHW uint8 to the GPU
    # conversion kernel; the kernel does the layout-aware indexing.
    comptime INPUT_LAYOUT_HWC: Bool = False

    var capacity: Int
    var n_frames: Int
    var frames: UnsafePointer[UInt8, MutAnyOrigin]
    var actions: UnsafePointer[UInt8, MutAnyOrigin]
    var dones: UnsafePointer[UInt8, MutAnyOrigin]

    def __init__(out self, capacity: Int):
        self.capacity = capacity
        self.n_frames = 0
        self.frames = alloc[UInt8](capacity * PONG_FRAME_BYTES)
        self.actions = alloc[UInt8](capacity)
        self.dones = alloc[UInt8](capacity)
        # Zero-fill so partial writes don't leave garbage.
        for i in range(capacity * PONG_FRAME_BYTES):
            self.frames[i] = 0
        for i in range(capacity):
            self.actions[i] = 0
            self.dones[i] = 0

    def __init__(out self, *, deinit take: Self):
        self.capacity = take.capacity
        self.n_frames = take.n_frames
        self.frames = take.frames
        self.actions = take.actions
        self.dones = take.dones
        # `deinit` skips take.__del__, so the buffers won't be double-freed.

    def __del__(deinit self):
        if Int(self.frames) != 0:
            self.frames.free()
        if Int(self.actions) != 0:
            self.actions.free()
        if Int(self.dones) != 0:
            self.dones.free()

    # ------------------------------------------------------------------
    # Append a step
    # ------------------------------------------------------------------

    def add_step_fp32(
        mut self,
        obs: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
        action: Int,
        done: Bool,
    ):
        """Append one step. obs is the 4×84×84 frame stack in [0, 1] fp32.

        Quantizes to uint8 by `clamp(round(x * 255), 0, 255)`.
        """
        if self.n_frames >= self.capacity:
            return  # silently drop — caller should check capacity
        var base = self.n_frames * PONG_FRAME_BYTES
        for i in range(PONG_FRAME_BYTES):
            var v = obs[i] * 255.0 + 0.5
            if v < 0.0:
                v = 0.0
            elif v > 255.0:
                v = 255.0
            self.frames[base + i] = UInt8(Int(v))
        self.actions[self.n_frames] = UInt8(action)
        self.dones[self.n_frames] = UInt8(1) if done else UInt8(0)
        self.n_frames += 1

    def add_step_fp32_list(
        mut self,
        obs: List[Scalar[DType.float32]],
        action: Int,
        done: Bool,
    ):
        """List-based wrapper for environments that return List[Scalar]."""
        if self.n_frames >= self.capacity:
            return
        var base = self.n_frames * PONG_FRAME_BYTES
        for i in range(PONG_FRAME_BYTES):
            var v = obs[i] * 255.0 + 0.5
            if v < 0.0:
                v = 0.0
            elif v > 255.0:
                v = 255.0
            self.frames[base + i] = UInt8(Int(v))
        self.actions[self.n_frames] = UInt8(action)
        self.dones[self.n_frames] = UInt8(1) if done else UInt8(0)
        self.n_frames += 1

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _window_is_valid(self, start: Int, T: Int) -> Bool:
        # The window covers frames [start, start + T - 1].
        # Reject if any of dones[start .. start + T - 2] is set
        # (episode boundary mid-window). The last frame may be done.
        for i in range(start, start + T - 1):
            if self.dones[i] != 0:
                return False
        return True

    def sample_batch_uint8(
        mut self,
        B: Int,
        T: Int,
        pixels_out: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
        actions_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ) raises:
        """Sample B contiguous-T windows into a uint8 pixel buffer.

        Output layouts:
          pixels_out:  (B, T, PONG_FRAME_BYTES) uint8 in [0, 255] — CHW per frame.
          actions_out: (B, T, PONG_NUM_ACTIONS) fp32 one-hot.

        The fp32 normalize + (no-op) permute happens on GPU via
        `pixels_uint8_to_fp32_kernel` after the uint8 DMA.

        Raises if buffer doesn't contain at least one valid window.
        """
        if self.n_frames < T:
            raise Error("PongOfflineBuffer.sample_batch_uint8: buffer too small")

        var sample_pix_stride = T * PONG_FRAME_BYTES
        var sample_act_stride = T * PONG_NUM_ACTIONS
        var max_start = self.n_frames - T

        for b in range(B):
            var start: Int = -1
            # Try up to 64 random starts, fall back to linear scan.
            for _ in range(64):
                var r = random_float64() * Float64(max_start + 1)
                var cand = Int(r)
                if cand > max_start:
                    cand = max_start
                if self._window_is_valid(cand, T):
                    start = cand
                    break
            if start < 0:
                # Linear scan as fallback.
                for s in range(max_start + 1):
                    if self._window_is_valid(s, T):
                        start = s
                        break
            if start < 0:
                raise Error(
                    "PongOfflineBuffer.sample_batch_uint8: no valid windows"
                )

            # Bulk uint8 copy of (T, C, H, W) frames for this window.
            var pix_dst = pixels_out + b * sample_pix_stride
            var pix_src_base = start * PONG_FRAME_BYTES
            for t in range(T):
                var src_off = pix_src_base + t * PONG_FRAME_BYTES
                var dst_off = t * PONG_FRAME_BYTES
                for i in range(PONG_FRAME_BYTES):
                    pix_dst[dst_off + i] = self.frames[src_off + i]

            # Expand actions to one-hot for this window.
            var act_dst = actions_out + b * sample_act_stride
            for t in range(T):
                var a = Int(self.actions[start + t])
                for k in range(PONG_NUM_ACTIONS):
                    act_dst[t * PONG_NUM_ACTIONS + k] = 0.0
                if a >= 0 and a < PONG_NUM_ACTIONS:
                    act_dst[t * PONG_NUM_ACTIONS + a] = 1.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: String) raises:
        """Write buffer to disk in LWMP v1 format."""
        var total = (
            PONG_BUFFER_HEADER_BYTES
            + self.n_frames * PONG_FRAME_BYTES
            + 2 * self.n_frames
        )
        var data = List[UInt8](capacity=total)

        # Header (64 bytes)
        for i in range(4):
            data.append(UInt8(ord(PONG_BUFFER_MAGIC[byte=i])))
        _write_uint32_le(data, PONG_BUFFER_VERSION)
        _write_uint64_le(data, UInt64(self.n_frames))
        _write_uint32_le(data, UInt32(PONG_OBS_C))
        _write_uint32_le(data, UInt32(PONG_OBS_H))
        _write_uint32_le(data, UInt32(PONG_OBS_W))
        _write_uint32_le(data, UInt32(PONG_NUM_ACTIONS))
        # Padding to 64 bytes.
        while len(data) < PONG_BUFFER_HEADER_BYTES:
            data.append(UInt8(0))

        # Frames
        var frame_bytes = self.n_frames * PONG_FRAME_BYTES
        for i in range(frame_bytes):
            data.append(self.frames[i])

        # Actions
        for i in range(self.n_frames):
            data.append(self.actions[i])

        # Dones
        for i in range(self.n_frames):
            data.append(self.dones[i])

        with open(path, "w") as f:
            f.write_bytes(data)

    @staticmethod
    def load(path: String) raises -> PongOfflineBuffer:
        """Load buffer from disk. Validates magic + dimensions."""
        var data: List[UInt8]
        with open(path, "r") as f:
            data = f.read_bytes()

        if len(data) < PONG_BUFFER_HEADER_BYTES:
            raise Error("PongOfflineBuffer.load: file too small for header")

        # Magic
        for i in range(4):
            if data[i] != UInt8(ord(PONG_BUFFER_MAGIC[byte=i])):
                raise Error("PongOfflineBuffer.load: bad magic")

        var version = _read_uint32_le(data, 4)
        if version != PONG_BUFFER_VERSION:
            raise Error("PongOfflineBuffer.load: unsupported version")

        var n_frames = Int(_read_uint64_le(data, 8))
        var obs_c = Int(_read_uint32_le(data, 16))
        var obs_h = Int(_read_uint32_le(data, 20))
        var obs_w = Int(_read_uint32_le(data, 24))
        var num_actions = Int(_read_uint32_le(data, 28))

        if (
            obs_c != PONG_OBS_C
            or obs_h != PONG_OBS_H
            or obs_w != PONG_OBS_W
            or num_actions != PONG_NUM_ACTIONS
        ):
            raise Error(
                "PongOfflineBuffer.load: shape mismatch — expected"
                " 4×84×84 with 3 actions"
            )

        var expected_size = (
            PONG_BUFFER_HEADER_BYTES
            + n_frames * PONG_FRAME_BYTES
            + 2 * n_frames
        )
        if len(data) < expected_size:
            raise Error("PongOfflineBuffer.load: file truncated")

        var buf = PongOfflineBuffer(capacity=n_frames)
        var off = PONG_BUFFER_HEADER_BYTES
        for i in range(n_frames * PONG_FRAME_BYTES):
            buf.frames[i] = data[off + i]
        off += n_frames * PONG_FRAME_BYTES
        for i in range(n_frames):
            buf.actions[i] = data[off + i]
        off += n_frames
        for i in range(n_frames):
            buf.dones[i] = data[off + i]
        buf.n_frames = n_frames
        return buf^
