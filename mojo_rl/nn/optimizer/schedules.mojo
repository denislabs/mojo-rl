"""Learning-rate schedules.

`LinearWarmupSchedule` reproduces the DreamerV3 reference schedule
(`references/dreamerv3-main/dreamerv3/agent.py:_make_opt`,
`schedule='const'` branch with `warmup`):

    sched = optax.constant_schedule(lr)
    if warmup:
        ramp = optax.linear_schedule(0.0, lr, warmup)
        sched = optax.join_schedules([ramp, sched], [warmup])

`optax.linear_schedule(0, lr, warmup)(t) = lr · clip(t / warmup, 0, 1)`,
joined at `warmup` with the constant schedule. So:

    lr_at(t) = lr · t / warmup    for t < warmup    (step 0 → 0.0)
    lr_at(t) = lr                 for t >= warmup

The step index is the optimizer's 0-indexed update count: the FIRST
update uses `lr_at(0) == 0` (zero learning rate on the first step), the
`warmup`-th update is the first to reach the full `lr`. Drive it from the
trainer:

    opt.lr = sched.lr_at(update_index)   # update_index starts at 0
    opt.step[target, M](model)
"""

from ..constants import DT


@fieldwise_init
struct LinearWarmupSchedule(Copyable & Movable & ImplicitlyDeletable):
    """Linear ramp `0 → target_lr` over `warmup_steps`, then constant.

    Bit-matches `optax.join_schedules([linear_schedule(0, lr, warmup),
    constant_schedule(lr)], [warmup])`. `warmup_steps == 0` collapses to a
    constant schedule (always returns `target_lr`)."""

    var target_lr: Scalar[DT]
    var warmup_steps: Int

    @staticmethod
    def make(
        target_lr: Scalar[DT], warmup_steps: Int = 0,
    ) -> Self:
        return Self(target_lr=target_lr, warmup_steps=warmup_steps)

    def lr_at(self, step: Int) -> Scalar[DT]:
        """LR for the 0-indexed update `step`. `step <= 0` → 0 during
        warmup; `step >= warmup_steps` → `target_lr`."""
        if self.warmup_steps <= 0:
            return self.target_lr
        if step >= self.warmup_steps:
            return self.target_lr
        if step <= 0:
            return Scalar[DT](0.0)
        return self.target_lr * Scalar[DT](step) / Scalar[DT](
            self.warmup_steps
        )
