"""DDPGMetrics — per-train-step bundle for the DDPG trainer.

`train_steps` is the cumulative count of trainer updates (NOT reset on
flush); `n_updates` is per-chunk."""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.metric import LogScalar


@fieldwise_init
struct DDPGMetrics(Copyable, Movable, ImplicitlyDestructible):
    var actor_loss:  LogScalar[DT]
    var critic_loss: LogScalar[DT]
    var train_steps: LogScalar[DT]
    var n_updates:   LogScalar[DT]
