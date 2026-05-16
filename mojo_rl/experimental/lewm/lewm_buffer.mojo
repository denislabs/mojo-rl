"""LeWM buffer trait — minimal contract the trainer relies on.

`LeWMTrainer` is generic on its data source (`BUF: LeWMBuffer`); concrete
implementations are `PongBuffer` (in-memory uint8 replay of Pong frames)
and `LewmPushTSampler` (HDF5-backed expert clips for the LeWM paper recipe).

The trait declares only the sampling contract; both adapters log their own
frame counts at construction, so we don't ask the trainer to print one.
"""


trait LeWMBuffer(Movable, ImplicitlyDestructible):
    """Minimal contract for a buffer/sampler consumed by `LeWMTrainer`.

    A conformant type must fill ``pixels_out`` and ``actions_out`` with a
    batch of ``B`` clips of length ``T``:

    - ``pixels_out``: ``(B, T, IN_CH * IMG * IMG)`` fp32 in ``[0, 1]``.
    - ``actions_out``: ``(B, T, ACT)`` fp32 (one-hot for discrete envs,
      or a dense ``frameskip * action_dim`` vector for continuous envs).

    The ``B``, ``T``, ``IN_CH``, ``IMG``, and ``ACT`` shapes are owned by
    the caller — the buffer is responsible only for delivering bytes in
    the agreed-on layout.
    """

    def sample_batch_fp32(
        mut self,
        B: Int,
        T: Int,
        pixels_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
        actions_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ) raises:
        ...
