"""LeWM PushT sampler — adapts `LewmPushTExpert` to the PongBuffer batch API.

The LeWM trainer (`trainer_struct.mojo`) is generic on its buffer type and only
requires:

    .sample_batch_fp32(B, T, pixels_out, actions_out) raises
    .n_frames: Int

This module wraps the HDF5-backed `LewmPushTExpert` (which delivers one clip
per `sample_window` call) into that shape:

- Random clip indices each batch row (uniform over the dataset's clip list).
- `pixels` uint8 (T, 3, H, W) -> fp32 (B, T, 3*H*W) in [0, 1].
- `action` f32 (T, frameskip * action_dim) -> copied verbatim into
  (B, T, ACT) — caller guarantees `ACT == frameskip * action_dim` at the
  comptime level when instantiating the trainer.

Constructor takes the same dataset args as `LewmPushTExpert` (`frameskip`,
`num_steps`, optional `path` for fixture tests). Owns a single
`LewmPushTWindow` scratch buffer that's reused across all batch elements.

Typical usage::

    var sampler = LewmPushTSampler(frameskip=5, num_steps=6)
    sampler.sample_batch_fp32(BATCH=16, T=6, pixels_out, actions_out)
"""

from std.random import random_float64

from mojo_rl.nn.datasets.lewm_pusht import LewmPushTExpert, LewmPushTWindow
from .lewm_buffer import LeWMBuffer


struct LewmPushTSampler(Movable, LeWMBuffer):
    """Batch sampler for LeWM PushT clips, matching PongBuffer's contract."""

    var dataset: LewmPushTExpert
    var window: LewmPushTWindow
    var n_frames: Int
    """Total dense frames across all episodes (cosmetic; matches Pong API)."""

    def __init__(
        out self,
        *,
        frameskip: Int = 5,
        num_steps: Int = 6,
        var path: String = String(""),
    ) raises:
        """Open the HDF5 dataset and pre-allocate one window buffer.

        Args:
            frameskip: Stride between observation samples (PushT paper: 5).
            num_steps: Window length on the observation axis — must match
                the trainer's `T` comptime param.
            path: Optional override of the cached `.h5` path. Empty string
                triggers HF auto-download to `~/.cache/mojo_rl/lewm_pusht/`.
        """
        self.dataset = LewmPushTExpert(
            frameskip=frameskip, num_steps=num_steps, path=path^
        )
        self.window = self.dataset.make_window()
        self.n_frames = self.dataset.n_total_frames
        print(
            "  [lewm_pusht_sampler] dataset:",
            len(self.dataset),
            "clips,",
            self.dataset.n_episodes,
            "episodes,",
            self.dataset.n_total_frames,
            "frames; H=",
            self.dataset.pixel_h,
            "W=",
            self.dataset.pixel_w,
            "act_dim=",
            self.dataset.action_dim,
            "frameskip=",
            self.dataset.frameskip,
            "num_steps=",
            self.dataset.num_steps,
        )

    def __init__(out self, *, deinit take: Self):
        self.dataset = take.dataset^
        self.window = take.window^
        self.n_frames = take.n_frames

    def sample_batch_fp32(
        mut self,
        B: Int,
        T: Int,
        pixels_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
        actions_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ) raises:
        """Fill `pixels_out` / `actions_out` with B random clips.

        Output layouts (must match the trainer's expectations):

        - ``pixels_out``: ``(B, T, 3 * H * W)`` fp32 in ``[0, 1]``.
        - ``actions_out``: ``(B, T, frameskip * action_dim)`` fp32 (verbatim).

        The trainer treats ``ACT = frameskip * action_dim`` as a single
        per-step action vector, matching the LeWM paper's
        ``effective_act_dim`` reshape.

        Raises if ``T`` doesn't match the dataset's ``num_steps`` (the
        window buffer is sized for one specific T).
        """
        if T != self.dataset.num_steps:
            raise Error(
                "LewmPushTSampler.sample_batch_fp32: T="
                + String(T)
                + " doesn't match dataset.num_steps="
                + String(self.dataset.num_steps)
                + ". Reconstruct the sampler with matching num_steps."
            )
        var n_clips = len(self.dataset)
        if n_clips <= 0:
            raise Error(
                "LewmPushTSampler.sample_batch_fp32: dataset has zero"
                " valid clips — check frameskip/num_steps vs episode"
                " lengths."
            )

        var H = self.dataset.pixel_h
        var W = self.dataset.pixel_w
        var pix_per_step = 3 * H * W
        var pix_per_sample = T * pix_per_step
        var act_per_step = self.dataset.frameskip * self.dataset.action_dim
        var act_per_sample = T * act_per_step
        var inv_255 = Float32(1.0 / 255.0)

        for b in range(B):
            var r = random_float64() * Float64(n_clips)
            var clip_idx = Int(r)
            if clip_idx >= n_clips:
                clip_idx = n_clips - 1
            if clip_idx < 0:
                clip_idx = 0

            self.dataset.sample_window(clip_idx, self.window)

            # pixels uint8 -> fp32 [0, 1]
            var pix_dst = pixels_out + b * pix_per_sample
            for i in range(pix_per_sample):
                pix_dst[i] = (
                    Float32(Int(self.window.pixels[i])) * inv_255
                )

            # actions: dense f32 copy
            var act_dst = actions_out + b * act_per_sample
            for i in range(act_per_sample):
                act_dst[i] = self.window.action[i]
