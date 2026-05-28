"""DQNMetrics — per-train-step bundle for the DQN trainer.

Mirrors `SACMetrics`: one @fieldwise_init struct, one LogScalar[DT] per
metric. `log_bundle[DQNMetrics, L: Logger]` walks via reflection and
emits one `log_scalar` call per field.

Fields correspond to what `DQNTrainer` accumulates in
`_loss_accum` (mean over `_update_count`), plus current epsilon
(point-in-time, not averaged)."""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.metric import LogScalar


@fieldwise_init
struct DQNMetrics(Copyable, Movable, ImplicitlyDestructible):
    var loss:      LogScalar[DT]
    var epsilon:   LogScalar[DT]
    var n_updates: LogScalar[DT]
