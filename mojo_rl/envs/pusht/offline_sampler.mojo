"""PushT offline sampler — adapts ``LewmPushTExpert`` to the OfflineBuffer API.

The LeWM offline trainer (and anything else built on
``mojo_rl.core.offline_buffer.OfflineBuffer``) is generic on its buffer
type and requires:

    .INPUT_LAYOUT_HWC: Bool   (comptime)
    .sample_batch_uint8(B, T, pixels_out_u8, actions_out_fp32) raises

This module wraps the HDF5-backed ``LewmPushTExpert`` (which delivers one
clip per ``sample_window`` call as HWC uint8) into the trainer's batch
shape. The HWC→CHW permute + uint8→fp32 normalize is deferred to a GPU
kernel (``pixels_uint8_to_fp32_kernel``), so this sampler is essentially
a bulk uint8 memcpy per batch element.

Constructor takes the same dataset args as ``LewmPushTExpert``
(``frameskip``, ``num_steps``, optional ``path`` for fixture tests).
Owns a single ``LewmPushTWindow`` scratch buffer reused across all batch
elements.

Typical usage::

    var sampler = PushTOfflineSampler(frameskip=5, num_steps=4)
    sampler.sample_batch_uint8(BATCH=16, T=4, pixels_u8_out, actions_fp32_out)
"""

from std.math import isnan, sqrt
from std.memory import memcpy
from std.random import random_float64

from mojo_rl.core.offline_buffer import OfflineBuffer
from mojo_rl.nn.datasets.lewm_pusht import LewmPushTExpert, LewmPushTWindow


struct PushTOfflineSampler(Movable, OfflineBuffer):
    """Batch sampler for PushT expert clips, conforming to OfflineBuffer."""

    # HDF5 stores PushT pixels as (H, W, C) — deliver them as-is to the
    # GPU conversion kernel, which handles the permute on-device.
    comptime INPUT_LAYOUT_HWC: Bool = True

    var dataset: LewmPushTExpert
    var window: LewmPushTWindow
    var n_frames: Int
    """Total dense frames across all episodes (cosmetic; matches Pong API)."""
    var normalize_actions: Bool
    """Z-score actions per raw dim with dataset mean/std (the reference's
    `get_column_normalizer`). Off by default — existing raw-action
    checkpoints stay valid."""
    var act_mean: List[Float64]
    """Per-raw-dim action mean (len action_dim; 0s when not normalizing)."""
    var act_std: List[Float64]
    """Per-raw-dim action std (len action_dim; 1s when not normalizing)."""

    def __init__(
        out self,
        *,
        frameskip: Int = 5,
        num_steps: Int = 6,
        var path: String = String(""),
        normalize_actions: Bool = False,
    ) raises:
        """Open the HDF5 dataset and pre-allocate one window buffer.

        Args:
            frameskip: Stride between observation samples (PushT paper: 5).
            num_steps: Window length on the observation axis — must match
                the trainer's `T` comptime param.
            path: Optional override of the cached `.h5` path. Empty string
                triggers HF auto-download to `~/.cache/mojo_rl/lewm_pusht/`.
            normalize_actions: Z-score actions per raw dim with the dataset
                mean/std at sample time (reference training pipeline).
                Read the stats back via `action_mean(d)` / `action_std(d)`
                — planning must de-normalize: raw = z·std + mean.
        """
        self.dataset = LewmPushTExpert(
            frameskip=frameskip, num_steps=num_steps, path=path^
        )
        self.window = self.dataset.make_window()
        self.n_frames = self.dataset.n_total_frames
        self.normalize_actions = normalize_actions
        var adim = self.dataset.action_dim
        self.act_mean = List[Float64](length=adim, fill=0.0)
        self.act_std = List[Float64](length=adim, fill=1.0)
        if normalize_actions:
            # Per-dim mean/std over the full action column, skipping NaN
            # rows (episode-boundary padding) — matches the reference's
            # StandardScaler / get_column_normalizer fit.
            var s1 = List[Float64](length=adim, fill=0.0)
            var s2 = List[Float64](length=adim, fill=0.0)
            var cnt = List[Float64](length=adim, fill=0.0)
            for i in range(self.dataset.n_total_frames):
                for d in range(adim):
                    var v = Float64(self.dataset.action_flat[i * adim + d])
                    if isnan(v):
                        continue
                    s1[d] += v
                    s2[d] += v * v
                    cnt[d] += 1.0
            for d in range(adim):
                if cnt[d] > 1.0:
                    var mu = s1[d] / cnt[d]
                    var var_ = s2[d] / cnt[d] - mu * mu
                    self.act_mean[d] = mu
                    self.act_std[d] = sqrt(var_) if var_ > 1e-12 else 1.0
            print(
                "  [pusht_offline_sampler] action z-score: mean=(",
                self.act_mean[0],
                ",",
                self.act_mean[1],
                ") std=(",
                self.act_std[0],
                ",",
                self.act_std[1],
                ")",
            )
        print(
            "  [pusht_offline_sampler] dataset:",
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
        self.normalize_actions = take.normalize_actions
        self.act_mean = take.act_mean^
        self.act_std = take.act_std^

    def action_mean(self, d: Int) -> Float64:
        return self.act_mean[d]

    def action_std(self, d: Int) -> Float64:
        return self.act_std[d]

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
                "PushTOfflineSampler.sample_batch_uint8: T="
                + String(T)
                + " doesn't match dataset.num_steps="
                + String(self.dataset.num_steps)
                + ". Reconstruct the sampler with matching num_steps."
            )
        var n_clips = len(self.dataset)
        if n_clips <= 0:
            raise Error(
                "PushTOfflineSampler.sample_batch_uint8: dataset has zero"
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

            # Fast path: dense HDF5 read → strided memcpy directly into
            # the batch's slot, skipping the LewmPushTWindow.pixels
            # intermediate. The window only contributes its ``pixels_dense``
            # buffer as a shared scratch — proprio/state are unused here.
            self.dataset.sample_clip_pixels_uint8(
                clip_idx,
                pixels_out + b * pix_per_sample,
                actions_out + b * act_per_sample,
                self.window.pixels_dense.as_unsafe_any_origin(),
            )
            if self.normalize_actions:
                # z-score per RAW action dim (stored layout interleaves
                # frameskip sub-steps: [x0,y0,x1,y1,...] → dim = j % 2).
                # NaN (boundary padding) → 0 (reference nan_to_num).
                var adim = self.dataset.action_dim
                var base = actions_out + b * act_per_sample
                for j in range(act_per_sample):
                    var d = j % adim
                    var v = Float64(base[j])
                    if isnan(v):
                        base[j] = Scalar[DType.float32](0.0)
                    else:
                        base[j] = Scalar[DType.float32](
                            (v - self.act_mean[d]) / self.act_std[d]
                        )
