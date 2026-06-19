"""Learning-rate schedules (storage surface).

`LinearWarmupSchedule` reproduces the DreamerV3 reference schedule
(`schedule='const'` with `warmup`): a linear ramp `0 → target_lr` over
`warmup_steps`, then constant. Bit-matches
`optax.join_schedules([linear_schedule(0, lr, warmup), constant_schedule(lr)],
[warmup])`.

    lr_at(t) = lr · t / warmup    for 0 < t < warmup   (step 0 → 0.0)
    lr_at(t) = lr                 for t >= warmup

Drive it from the trainer (the 0-indexed update count):

    opt.lr = sched.lr_at(update_index)   # update_index starts at 0
    opt.step[target, M](model)

Pure host math — no Tensor / device state — so the storage port is the legacy
struct verbatim (import path only).
"""

from mojo_rl.nn.constants import DT


@fieldwise_init
struct LinearWarmupSchedule(Copyable & Movable & ImplicitlyDeletable):
    """Linear ramp `0 → target_lr` over `warmup_steps`, then constant.
    `warmup_steps == 0` collapses to a constant schedule (always target_lr)."""

    var target_lr: Scalar[DT]
    var warmup_steps: Int

    @staticmethod
    def make(target_lr: Scalar[DT], warmup_steps: Int = 0) -> Self:
        return Self(target_lr=target_lr, warmup_steps=warmup_steps)

    def lr_at(self, step: Int) -> Scalar[DT]:
        """LR for the 0-indexed update `step`. `step <= 0` → 0 during warmup;
        `step >= warmup_steps` → `target_lr`."""
        if self.warmup_steps <= 0:
            return self.target_lr
        if step >= self.warmup_steps:
            return self.target_lr
        if step <= 0:
            return Scalar[DT](0.0)
        return self.target_lr * Scalar[DT](step) / Scalar[DT](
            self.warmup_steps
        )
