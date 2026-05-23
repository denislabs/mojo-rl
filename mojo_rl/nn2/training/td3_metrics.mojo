"""TD3Metrics — per-train-step bundle for the TD3 trainer.

TD3 has separate actor + critic update counts due to policy delay.
Each is its own LogScalar so the logger sees both."""

from ..constants import DT
from ..core.metric import LogScalar


@fieldwise_init
struct TD3Metrics(Copyable, Movable, ImplicitlyDestructible):
    var actor_loss:  LogScalar[DT]
    var critic_loss: LogScalar[DT]
    var n_actor_updates:  LogScalar[DT]
    var n_critic_updates: LogScalar[DT]
