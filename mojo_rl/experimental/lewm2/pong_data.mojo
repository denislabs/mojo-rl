"""PongWindowSource — bridge a `PongOfflineBuffer` to the LeWM nn2 trainer.

The Pong data pipeline already exists and is reusable as-is:
  - collection: `examples/lewm/lewm_pong_collect_buffer.mojo` (scripted +
    random follow-the-ball policy over `PongPixelEnv`) → `PongOfflineBuffer`
    (uint8 CHW [N,4,84,84] + action idx + done markers) saved as LWMP v1.
This struct is the only missing piece: it samples length-T windows from a
loaded buffer and hands the trainer exactly what `train_step` consumes —
fp32 pixels `(B, T·IMG_DIM)` (CHW, ÷255) and fp32 one-hot actions
`(B, T·ACT)` — on the chosen target.

Frames are stored CHW (`PongOfflineBuffer.INPUT_LAYOUT_HWC == False`), the
same channel-major layout `PatchEmbed`'s `Conv2D` expects, so the bridge
only normalises (`u8_to_fp32_norm`, layout-preserving) — no permute.

Flow per batch:
  buf.sample_batch_uint8 → host uint8 pix + host fp32 one-hot act
  CPU:  convert uint8→fp32 (host)              → pix/act host ptrs
  GPU:  H2D uint8 + act, convert uint8→fp32 dev → pix/act device ptrs
The trainer wraps `pix_ptr`/`act_ptr` in `TileTensor`s and calls train_step.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer

from ...nn2.constants import DT
from .pixel_convert import u8_to_fp32_norm
from mojo_rl.envs.arcade_games.pong.offline_buffer import (
    PongOfflineBuffer,
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)


struct PongWindowSource[
    IMG_DIM: Int, ACT: Int, T: Int, B: Int, target: StaticString = "cpu",
](Movable & ImplicitlyDestructible):
    comptime NPIX = Self.B * Self.T * Self.IMG_DIM
    comptime NACT = Self.B * Self.T * Self.ACT

    var buf: PongOfflineBuffer
    # Host staging (always): sampled uint8 pixels + fp32 one-hot actions.
    var pix_u8_host: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin]
    var act_host: UnsafePointer[Scalar[DT], MutAnyOrigin]
    # CPU output: converted fp32 pixels (host). GPU: device buffers below.
    var pix_fp32_host: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var pix_u8_dev: Optional[DeviceBuffer[DType.uint8]]
    var pix_fp32_dev: Optional[DeviceBuffer[DT]]
    var act_dev: Optional[DeviceBuffer[DT]]
    var ctx: Optional[DeviceContext]

    def __init__(out self, var buf: PongOfflineBuffer):
        comptime assert Self.IMG_DIM == PONG_FRAME_BYTES, (
            "PongWindowSource: IMG_DIM must equal PONG_FRAME_BYTES (28224)"
        )
        comptime assert Self.ACT == PONG_NUM_ACTIONS, (
            "PongWindowSource: ACT must equal PONG_NUM_ACTIONS (3)"
        )
        self.buf = buf^
        self.pix_u8_host = alloc[Scalar[DType.uint8]](Self.NPIX)
        self.act_host = alloc[Scalar[DT]](Self.NACT)
        self.pix_fp32_host = alloc[Scalar[DT]](Self.NPIX)
        self.pix_u8_dev = None
        self.pix_fp32_dev = None
        self.act_dev = None
        self.ctx = None

    def __del__(deinit self):
        self.pix_u8_host.free()
        self.act_host.free()
        self.pix_fp32_host.free()

    @staticmethod
    def make(
        var buf: PongOfflineBuffer,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        var s = Self(buf^)
        s.ctx = ctx
        comptime if Self.target == "gpu":
            if not ctx:
                raise Error("PongWindowSource.make[gpu]: ctx required")
            var c = ctx.value()
            s.pix_u8_dev = c.enqueue_create_buffer[DType.uint8](Self.NPIX)
            s.pix_fp32_dev = c.enqueue_create_buffer[DT](Self.NPIX)
            s.act_dev = c.enqueue_create_buffer[DT](Self.NACT)
        return s^

    def next_batch(mut self) raises:
        """Sample one (B, T) window batch and populate the target buffers.
        After this call `pix_ptr()` / `act_ptr()` are valid for one
        `train_step` / `eval_loss`."""
        self.buf.sample_batch_uint8(
            Self.B, Self.T, self.pix_u8_host, self.act_host
        )
        comptime if Self.target == "cpu":
            u8_to_fp32_norm["cpu", Self.NPIX](
                self.pix_u8_host, self.pix_fp32_host
            )
        else:
            var c = self.ctx.value()
            c.enqueue_copy(self.pix_u8_dev.value(), self.pix_u8_host)
            c.enqueue_copy(self.act_dev.value(), self.act_host)
            u8_to_fp32_norm["gpu", Self.NPIX](
                rebind[UnsafePointer[Scalar[DType.uint8], MutAnyOrigin]](
                    self.pix_u8_dev.value().unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    self.pix_fp32_dev.value().unsafe_ptr()
                ),
                ctx=self.ctx,
            )

    def pix_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        comptime if Self.target == "cpu":
            return self.pix_fp32_host
        else:
            return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.pix_fp32_dev.value().unsafe_ptr()
            )

    def act_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        comptime if Self.target == "cpu":
            return self.act_host
        else:
            return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.act_dev.value().unsafe_ptr()
            )
