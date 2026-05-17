"""LeWM buffer trait — uint8 sampling contract.

`LeWMTrainer` is generic on its data source (`BUF: LeWMBuffer`); concrete
implementations are `PongBuffer` (in-memory uint8 CHW frames) and
`LewmPushTSampler` (HDF5-backed expert clips in HWC uint8). The trait
exposes a uint8 sampling path so the heavy host work (HWC→CHW permute +
uint8→fp32 normalize) can be lifted into a GPU kernel — see
`pixels_uint8_to_fp32_kernel` in `offline_trainer.mojo`.

Conformant implementations declare their pixel layout via the comptime
field `INPUT_LAYOUT_HWC`:
  - `False`: pixels delivered as ``(B, T, C, H, W)`` uint8 — Pong/CHW.
  - `True`:  pixels delivered as ``(B, T, H, W, C)`` uint8 — PushT/HWC.
The conversion kernel uses this flag (also comptime) to pick the right
source indexing.

Actions stay on the fp32 path (negligible per-step cost; both envs need
the same downstream layout anyway).
"""


trait LeWMBuffer(Movable, ImplicitlyDestructible):
    """Minimal contract for a buffer/sampler consumed by `LeWMTrainer`.

    A conformant type must fill ``pixels_out`` (uint8, layout per
    ``INPUT_LAYOUT_HWC``) and ``actions_out`` (fp32) with a batch of
    ``B`` clips of length ``T``:

    - ``pixels_out``: ``(B, T, ...)`` uint8 in [0, 255]; total
      ``IN_CH * IMG * IMG`` elements per frame regardless of layout.
    - ``actions_out``: ``(B, T, ACT)`` fp32.

    The ``B``, ``T``, ``IN_CH``, ``IMG``, and ``ACT`` shapes are owned
    by the caller — the buffer is responsible only for delivering bytes
    in the agreed-on layout.
    """

    comptime INPUT_LAYOUT_HWC: Bool

    def sample_batch_uint8(
        mut self,
        B: Int,
        T: Int,
        pixels_out: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
        actions_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ) raises:
        ...
