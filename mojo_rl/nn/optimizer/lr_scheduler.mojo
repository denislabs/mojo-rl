"""Epoch-based learning-rate schedules for the storage Trainer.

A `Scheduler` is a compile-time hook: `Trainer.train_gpu` takes a `SCHEDULER`
parameter (default `ConstantSchedule`) and, when it isn't constant, multiplies
the optimizer's base LR by `SCHEDULER.lr_scale_at(epoch, total_epochs)` at the
start of each epoch (via `opt.set_lr`). The base LR is whatever the caller set
on the optimizer before calling `train_gpu`.

`IS_CONSTANT` lets the trainer skip all LR-poking for the default case
(preserving the exact user-set LR).

Built-ins:
  - `ConstantSchedule`                          scale ≡ 1.0
  - `LinearWarmupSchedule[WARMUP]`              linear 0 → 1, then constant 1.0
  - `CosineSchedule[MIN_SCALE]`                 cosine 1 → MIN_SCALE
  - `WarmupCosineSchedule[WARMUP, MIN_SCALE]`   linear warmup then cosine
  - `StepSchedule[DROP_EVERY, GAMMA]`           ×GAMMA every DROP_EVERY epochs

Pure host math (no Tensor / device state) — ported verbatim from the legacy
`nn/training/lr_scheduler.mojo`. NOTE: this is the comptime `Scheduler`-trait
family (epoch → scale), distinct from `optimizer/schedules.LinearWarmupSchedule`
(a runtime `lr_at(step)` struct used by DreamerV3/TD-MPC2).

`epoch`/`total_epochs` are a generic (index, horizon) pair — a step-granularity
loop (e.g. RL self-play) can call `lr_scale_at(train_step, total_train_steps)`.
"""

from std.math import cos

comptime _PI: Float64 = 3.141592653589793


trait Scheduler(Copyable & Movable & ImplicitlyDeletable):
    """LR-scale schedule over epochs."""

    comptime IS_CONSTANT: Bool

    @staticmethod
    def lr_scale_at(epoch: Int, total_epochs: Int) -> Float64:
        """Multiplier on the base LR for 0-indexed `epoch` (∈ [0, ~1])."""
        ...


@fieldwise_init
struct ConstantSchedule(Scheduler):
    """No schedule — the optimizer's LR is left untouched."""

    comptime IS_CONSTANT: Bool = True

    @staticmethod
    def lr_scale_at(epoch: Int, total_epochs: Int) -> Float64:
        return 1.0


@fieldwise_init
struct LinearWarmupSchedule[WARMUP_EPOCHS: Int = 5](Scheduler):
    """Linear warmup 0 → 1 over WARMUP_EPOCHS, then constant 1.0 (no decay)."""

    comptime IS_CONSTANT: Bool = False

    @staticmethod
    def lr_scale_at(epoch: Int, total_epochs: Int) -> Float64:
        comptime assert Self.WARMUP_EPOCHS >= 1, "WARMUP_EPOCHS must be >= 1"
        if epoch < Self.WARMUP_EPOCHS:
            return Float64(epoch + 1) / Float64(Self.WARMUP_EPOCHS)
        return 1.0


@fieldwise_init
struct CosineSchedule[MIN_SCALE: Float64 = 0.0](Scheduler):
    """Cosine decay from 1.0 (epoch 0) to MIN_SCALE (final epoch)."""

    comptime IS_CONSTANT: Bool = False

    @staticmethod
    def lr_scale_at(epoch: Int, total_epochs: Int) -> Float64:
        var denom = total_epochs - 1
        if denom < 1:
            denom = 1
        var progress = Float64(epoch) / Float64(denom)
        if progress > 1.0:
            progress = 1.0
        var c = 0.5 * (1.0 + cos(_PI * progress))
        return Self.MIN_SCALE + (1.0 - Self.MIN_SCALE) * c


@fieldwise_init
struct WarmupCosineSchedule[
    WARMUP_EPOCHS: Int = 5, MIN_SCALE: Float64 = 0.0
](Scheduler):
    """Linear warmup 0 → 1 over WARMUP_EPOCHS, then cosine 1 → MIN_SCALE."""

    comptime IS_CONSTANT: Bool = False

    @staticmethod
    def lr_scale_at(epoch: Int, total_epochs: Int) -> Float64:
        if epoch < Self.WARMUP_EPOCHS:
            return Float64(epoch + 1) / Float64(Self.WARMUP_EPOCHS)
        var denom = total_epochs - Self.WARMUP_EPOCHS
        if denom < 1:
            denom = 1
        var progress = Float64(epoch - Self.WARMUP_EPOCHS) / Float64(denom)
        if progress > 1.0:
            progress = 1.0
        var c = 0.5 * (1.0 + cos(_PI * progress))
        return Self.MIN_SCALE + (1.0 - Self.MIN_SCALE) * c


@fieldwise_init
struct StepSchedule[DROP_EVERY: Int = 20, GAMMA: Float64 = 0.1](Scheduler):
    """Multiply the LR by GAMMA every DROP_EVERY epochs (staircase)."""

    comptime IS_CONSTANT: Bool = False

    @staticmethod
    def lr_scale_at(epoch: Int, total_epochs: Int) -> Float64:
        var drops = epoch // Self.DROP_EVERY
        var scale = 1.0
        for _ in range(drops):
            scale *= Self.GAMMA
        return scale
