"""DDPGMetrics — per-train-step bundle for the DDPG trainer."""

from ..constants import DT
from ..core.metric import LogScalar


@fieldwise_init
struct DDPGMetrics(Copyable, Movable, ImplicitlyDestructible):
    var actor_loss:  LogScalar[DT]
    var critic_loss: LogScalar[DT]
    var n_updates:   LogScalar[DT]
