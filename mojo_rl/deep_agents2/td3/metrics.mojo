"""TD3Metrics — per-train-step bundle for the TD3 trainer.

TD3 has separate actor + critic update counts due to policy delay.
Each is its own LogScalar so the logger sees both. `train_steps`
matches `n_critic_updates` cumulatively (the critic fires every
train_step). Per-chunk counts are reset on `flush_metrics`;
`train_steps` is not."""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.metric import LogScalar


@fieldwise_init
struct TD3Metrics(Copyable, Movable, ImplicitlyDestructible):
    var actor_loss:  LogScalar[DT]
    var critic_loss: LogScalar[DT]
    var train_steps:      LogScalar[DT]
    var n_actor_updates:  LogScalar[DT]
    var n_critic_updates: LogScalar[DT]
