"""LeWM PushT sampler — adapts `LewmPushTExpert` to the LeWMBuffer uint8 API.

The LeWM trainer (`trainer_struct.mojo`) is generic on its buffer type
and requires (per `lewm_buffer.LeWMBuffer`):

    .INPUT_LAYOUT_HWC: Bool   (comptime)
    .sample_batch_uint8(B, T, pixels_out_u8, actions_out_fp32) raises

This module wraps the HDF5-backed `LewmPushTExpert` (which delivers one
clip per `sample_window` call as HWC uint8) into the trainer's batch
shape. The HWC→CHW permute + uint8→fp32 normalize is deferred to a GPU
kernel (`pixels_uint8_to_fp32_kernel`), so this sampler is essentially a
bulk uint8 memcpy per batch element.

Constructor takes the same dataset args as `LewmPushTExpert`
(`frameskip`, `num_steps`, optional `path` for fixture tests). Owns a
single `LewmPushTWindow` scratch buffer reused across all batch elements.

Typical usage::

    var sampler = LewmPushTSampler(frameskip=5, num_steps=4)
    sampler.sample_batch_uint8(BATCH=16, T=4, pixels_u8_out, actions_fp32_out)
"""

from std.random import random_float64

from mojo_rl.nn.datasets.lewm_pusht import LewmPushTExpert, LewmPushTWindow
from .lewm_buffer import LeWMBuffer


struct LewmPushTSampler(Movable, LeWMBuffer):
    """Batch sampler for LeWM PushT clips, conforming to the LeWMBuffer trait."""

    # HDF5 stores PushT pixels as (H, W, C) — deliver them as-is to the
    # GPU conversion kernel, which handles the permute on-device.
    comptime INPUT_LAYOUT_HWC: Bool = True

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

    def sample_batch_uint8(
        mut self,
        B: Int,
        T: Int,
        pixels_out: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
        actions_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ) raises:
        """Fill `pixels_out` (uint8 HWC) / `actions_out` (fp32) with B clips.

        Output layouts:

        - ``pixels_out``: ``(B, T, H, W, 3)`` uint8 in ``[0, 255]``.
          The GPU conversion kernel does the HWC→CHW permute + /255.
        - ``actions_out``: ``(B, T, frameskip * action_dim)`` fp32.

        Raises if ``T`` doesn't match the dataset's ``num_steps`` (the
        window buffer is sized for one specific T).
        """
        if T != self.dataset.num_steps:
            raise Error(
                "LewmPushTSampler.sample_batch_uint8: T="
                + String(T)
                + " doesn't match dataset.num_steps="
                + String(self.dataset.num_steps)
                + ". Reconstruct the sampler with matching num_steps."
            )
        var n_clips = len(self.dataset)
        if n_clips <= 0:
            raise Error(
                "LewmPushTSampler.sample_batch_uint8: dataset has zero"
                " valid clips — check frameskip/num_steps vs episode"
                " lengths."
            )

        var H = self.dataset.pixel_h
        var W = self.dataset.pixel_w
        var pix_per_step = H * W * 3
        var pix_per_sample = T * pix_per_step
        var act_per_step = self.dataset.frameskip * self.dataset.action_dim
        var act_per_sample = T * act_per_step

        for b in range(B):
            var r = random_float64() * Float64(n_clips)
            var clip_idx = Int(r)
            if clip_idx >= n_clips:
                clip_idx = n_clips - 1
            if clip_idx < 0:
                clip_idx = 0

            self.dataset.sample_window(clip_idx, self.window)

            # Bulk uint8 HWC copy (window.pixels is already HWC).
            var pix_dst = pixels_out + b * pix_per_sample
            for i in range(pix_per_sample):
                pix_dst[i] = self.window.pixels[i]

            # Actions: dense f32 copy.
            var act_dst = actions_out + b * act_per_sample
            for i in range(act_per_sample):
                act_dst[i] = self.window.action[i]
