"""C51Metrics — per-train-step bundle for the C51 / Rainbow trainer.

Mirrors `DQNMetrics`: one @fieldwise_init struct, one LogScalar[DT] per
metric. `log_bundle[C51Metrics, L: Logger]` walks via reflection and
emits one `log_scalar` call per field.

Fields correspond to what `C51Trainer` accumulates in `_loss_accum`
(mean categorical cross-entropy over `_update_count`), plus current
epsilon (point-in-time, not averaged). `train_steps` is the cumulative
count of trainer updates (NOT reset on flush); `n_updates` is per-chunk."""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.metric import LogScalar


@fieldwise_init
struct C51Metrics(Copyable, Movable, ImplicitlyDestructible):
    var loss:        LogScalar[DT]
    var epsilon:     LogScalar[DT]
    var train_steps: LogScalar[DT]
    var n_updates:   LogScalar[DT]
