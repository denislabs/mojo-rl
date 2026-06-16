"""OfflineWindowBuffer — minimal in-memory offline trajectory buffer.

Holds N_TRAJ trajectories of fp32 pixel frames + actions, and samples B
length-T windows into caller-provided `(B, T·IMG_DIM)` / `(B, T·ACT)` host
buffers. Self-contained (no legacy / env deps) — the LeWM trainer's data
path. Real pixel data (uint8 + a HWC→CHW/÷255 kernel) can replace the
synthetic fill; the sampling protocol (random traj_id, start_t → slice T
frames) is the same.
"""

from std.memory import alloc

from ...nn.constants import DT
from .pixel_convert import u8_hwc_to_chw_norm


struct OfflineWindowBuffer[IMG_DIM: Int, ACT: Int, T: Int](
    Movable & ImplicitlyDeletable
):
    var n_traj: Int
    var traj_len: Int
    var frames: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [N, L, IMG_DIM]
    var actions: UnsafePointer[Scalar[DT], MutAnyOrigin]  # [N, L, ACT]
    var rng: UInt64

    def __init__(out self, n_traj: Int, traj_len: Int, seed: UInt64 = 12345):
        self.n_traj = n_traj
        self.traj_len = traj_len
        self.frames = alloc[Scalar[DT]](n_traj * traj_len * Self.IMG_DIM)
        self.actions = alloc[Scalar[DT]](n_traj * traj_len * Self.ACT)
        self.rng = seed

    def __del__(deinit self):
        self.frames.free()
        self.actions.free()

    @staticmethod
    def _det(i: Int, scale: Float64) -> Scalar[DT]:
        var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
        return Scalar[DT](v * scale)

    def fill_synthetic(mut self):
        """Deterministic synthetic trajectories with mild temporal +
        action structure (enough for the model to fit a small buffer)."""
        for n in range(self.n_traj):
            for t in range(self.traj_len):
                var a_base = (n * self.traj_len + t)
                for j in range(Self.ACT):
                    self.actions[(n * self.traj_len + t) * Self.ACT + j] = (
                        Self._det(a_base * 3 + j + 1, 1.0)
                    )
                for d in range(Self.IMG_DIM):
                    # frame depends on (traj, time, dim) — smooth in t.
                    var idx = (n * self.traj_len + t) * Self.IMG_DIM + d
                    self.frames[idx] = Self._det(
                        n * 131 + t * 7 + d + 1, 1.0
                    )

    def set_frame_u8_hwc[
        C: Int, FH: Int, FW: Int,
    ](
        mut self, traj: Int, t: Int,
        hwc: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
    ) raises:
        """Ingest one real `uint8` HWC frame at `(traj, t)`, converting to
        CHW/÷255 fp32 in place (host path of `u8_hwc_to_chw_norm`). Asserts
        `C·FH·FW == IMG_DIM`. The sampling protocol is unchanged — once
        frames are stored, `sample_into` slices windows as before."""
        comptime assert C * FH * FW == Self.IMG_DIM, (
            "set_frame_u8_hwc: C·FH·FW must equal IMG_DIM"
        )
        var dst = self.frames + (traj * self.traj_len + t) * Self.IMG_DIM
        u8_hwc_to_chw_norm["cpu", C, FH, FW, 1](hwc, dst)

    def _next(mut self) -> UInt64:
        # xorshift64*
        var x = self.rng
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.rng = x
        return x * 0x2545F4914F6CDD1D

    def sample_into(
        mut self,
        pix: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],
        batch: Int,
    ):
        """Fill pix (batch, T·IMG_DIM) and act (batch, T·ACT) with random
        length-T windows."""
        var max_start = self.traj_len - Self.T
        for b in range(batch):
            var traj = Int(self._next() % UInt64(self.n_traj))
            var start = Int(self._next() % UInt64(max_start + 1))
            for t in range(Self.T):
                var src_f = (
                    (traj * self.traj_len + start + t) * Self.IMG_DIM
                )
                var dst_f = (b * Self.T + t) * Self.IMG_DIM
                for d in range(Self.IMG_DIM):
                    pix[dst_f + d] = self.frames[src_f + d]
                var src_a = (traj * self.traj_len + start + t) * Self.ACT
                var dst_a = (b * Self.T + t) * Self.ACT
                for j in range(Self.ACT):
                    act[dst_a + j] = self.actions[src_a + j]
