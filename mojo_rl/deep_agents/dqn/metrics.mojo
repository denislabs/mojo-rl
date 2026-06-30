"""DQNMetrics — per-train-step bundle for the DQN trainer.

Mirrors `SACMetrics`: one @fieldwise_init struct, one LogScalar[DT] per
metric. `log_bundle[DQNMetrics, L: Logger]` walks via reflection and
emits one `log_scalar` call per field.

Fields correspond to what `DQNTrainer` accumulates in
`_loss_accum` (mean over `_update_count`), plus current epsilon
(point-in-time, not averaged). `train_steps` is the cumulative count of
trainer updates (NOT reset on flush); `n_updates` is per-chunk.

The `mean_*` diagnostics are per-batch averages over each training
minibatch, accumulated across the chunk and averaged at flush (CPU
train_target only — GPU defers, mirroring SAC). `mean_td_error` is the
mean absolute Bellman residual |Q(s,a) − y|, the key DQN learning-health
signal."""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.metric import LogScalar


@fieldwise_init
struct DQNMetrics(Copyable, Movable, ImplicitlyDeletable):
    var loss:          LogScalar[DT]
    var epsilon:       LogScalar[DT]
    var mean_q:        LogScalar[DT]
    var mean_target:   LogScalar[DT]
    var mean_td_error: LogScalar[DT]
    var mean_reward:   LogScalar[DT]
    var mean_done:     LogScalar[DT]
    var train_steps:   LogScalar[DT]
    var n_updates:     LogScalar[DT]
