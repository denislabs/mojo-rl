"""OfflineBuffer trait — uint8 pixel + fp32 action sampling contract.

Generic contract for pixel-observation offline RL buffers: a single
``sample_batch_uint8(B, T, pixels_out, actions_out)`` call delivers a
``(B, T)`` batch of clips in uint8 (pixels) + fp32 (actions). The heavy
host work (HWC↔CHW permute + uint8→fp32 normalize) is deferred to GPU.

Conformant implementations declare their pixel layout via the comptime
field ``INPUT_LAYOUT_HWC``:

- ``False``: pixels delivered as ``(B, T, C, H, W)`` uint8 — e.g. Pong/CHW.
- ``True``:  pixels delivered as ``(B, T, H, W, C)`` uint8 — e.g. PushT/HWC.

The companion GPU conversion kernel uses this flag (also comptime) to
pick the right source indexing. Actions stay on the fp32 path
(negligible per-step cost; downstream layout is uniform anyway).

Concrete implementations:
  - ``mojo_rl.envs.arcade_games.pong.offline_buffer.PongOfflineBuffer``
  - ``mojo_rl.envs.pusht.offline_sampler.PushTOfflineSampler``
"""


trait OfflineBuffer(Movable, ImplicitlyDeletable):
    """Minimal contract for a pixel-obs offline trajectory data source.

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
