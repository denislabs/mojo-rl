"""LR schedulers for the Trainer.

Vendored verbatim from the legacy `mojo_rl.nn.training.scheduler` module during
the nn re-architecture so that `mojo_rl.experimental.pcn` carries no dependency
on legacy `nn`. The original only imports from `std.*`, so the schedule logic
and struct names are unchanged; the `Scheduler` trait is defined locally here
to keep the file self-contained.

Pure host-side: each schedule maps (epoch, total_epochs) to a Float64
multiplier that the Trainer feeds to `state.set_lr_scale()` once per epoch.
Per-epoch granularity matches `train_gpu_minibatch`'s one-graph-per-epoch
capture, so the host scalar makes it into the captured kernels without any
device-buffer plumbing.

Built-ins:
    ConstantSchedule          — always 1.0 (default)
    LinearWarmupSchedule[W]   — linear 0→1 over W epochs, then 1.0
    CosineWarmupSchedule[W,M] — linear warmup over W, then cosine decay to M
    OneCycleSchedule[P,I,F]   — Smith 1cycle: ramp up to peak, cosine decay down

Usage:
    Trainer[...].train_gpu_minibatch_full[
        ...,
        SCHEDULER = CosineWarmupSchedule[WARMUP_EPOCHS=5, MIN_SCALE=0.1],
    ](...)
"""
from std.math import cos, pi


trait Scheduler(ImplicitlyCopyable):
    """Maps (epoch, total_epochs) to an LR multiplier."""

    @staticmethod
    def lr_scale_at(epoch: Int, total_epochs: Int) -> Float64:
        """Multiplier applied to the optimizer's compile-time base LR.

        Args:
            epoch: 0-indexed epoch about to run.
            total_epochs: Total number of epochs in the training run.

        Returns:
            LR multiplier (typically in [0, 1]).
        """
        ...


struct ConstantSchedule(Scheduler):
    """No schedule — always returns 1.0."""

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def lr_scale_at(epoch: Int, total_epochs: Int) -> Float64:
        return 1.0


struct LinearWarmupSchedule[WARMUP_EPOCHS: Int](Scheduler):
    """Linear warmup from 0 to 1 over WARMUP_EPOCHS, then constant 1.0.

    Parameters:
        WARMUP_EPOCHS: Number of warmup epochs. Must be >= 1.
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def lr_scale_at(epoch: Int, total_epochs: Int) -> Float64:
        comptime assert Self.WARMUP_EPOCHS >= 1, "WARMUP_EPOCHS must be >= 1"
        if epoch < Self.WARMUP_EPOCHS:
            return Float64(epoch + 1) / Float64(Self.WARMUP_EPOCHS)
        return 1.0


struct OneCycleSchedule[
    PEAK_FRAC: Float64 = 0.3,
    INITIAL_FACTOR: Float64 = 0.1,
    FINAL_FACTOR: Float64 = 0.01,
](Scheduler):
    """Smith 1cycle LR schedule (per-step granularity).

    Linear ramp from INITIAL_FACTOR up to 1.0 over PEAK_FRAC of total_epochs,
    then cosine anneal from 1.0 down to FINAL_FACTOR over the remainder.
    `epoch` is reused as a generic step index here — call with (step, total_steps).

    Designed to lock in late-cycle steps near the optimum (small LR) after
    spending the early/middle of the cycle searching aggressively (max LR).

    Parameters:
        PEAK_FRAC: Fraction of total_epochs at which scale reaches 1.0
                   (default 0.3 ≈ Smith's recommendation).
        INITIAL_FACTOR: Start scale (default 0.1 = 10% of base LR).
        FINAL_FACTOR: End scale (default 0.01 = 1% of base LR).
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def lr_scale_at(epoch: Int, total_epochs: Int) -> Float64:
        if total_epochs <= 1:
            return 1.0
        var progress = Float64(epoch) / Float64(total_epochs - 1)
        if progress < 0.0:
            progress = 0.0
        if progress > 1.0:
            progress = 1.0
        if progress <= Self.PEAK_FRAC:
            var t = progress / Self.PEAK_FRAC
            return Self.INITIAL_FACTOR + (1.0 - Self.INITIAL_FACTOR) * t
        var t = (progress - Self.PEAK_FRAC) / (1.0 - Self.PEAK_FRAC)
        var c = 0.5 * (1.0 + cos(t * pi))
        return Self.FINAL_FACTOR + (1.0 - Self.FINAL_FACTOR) * c


struct CosineWarmupSchedule[WARMUP_EPOCHS: Int, MIN_SCALE: Float64 = 0.1](
    Scheduler
):
    """Linear warmup over WARMUP_EPOCHS, then cosine decay to MIN_SCALE.

    Matches the standard ViT / ResNet recipe: scale rises linearly to 1.0
    during warmup, then follows a half-cosine to MIN_SCALE by the final
    epoch.

    Parameters:
        WARMUP_EPOCHS: Number of warmup epochs. Must be >= 1.
        MIN_SCALE: Minimum scale at the end of decay (default 0.1).
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def lr_scale_at(epoch: Int, total_epochs: Int) -> Float64:
        comptime assert Self.WARMUP_EPOCHS >= 1, "WARMUP_EPOCHS must be >= 1"
        if epoch < Self.WARMUP_EPOCHS:
            return Float64(epoch + 1) / Float64(Self.WARMUP_EPOCHS)
        var decay_epochs = total_epochs - Self.WARMUP_EPOCHS
        if decay_epochs <= 0:
            return 1.0
        var progress = Float64(epoch - Self.WARMUP_EPOCHS) / Float64(
            decay_epochs
        )
        if progress > 1.0:
            progress = 1.0
        var c = 0.5 * (1.0 + cos(progress * pi))
        return Self.MIN_SCALE + (1.0 - Self.MIN_SCALE) * c
