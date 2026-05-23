"""SACMetrics — per-train-step bundle for the SAC trainer.

Phase A.5. One @fieldwise_init struct, one LogScalar[DT] per metric.
`log_bundle[SACMetrics, L: Logger]` walks via reflection and emits one
log_scalar call per field.

Fields here correspond to what `SACTrainer` already accumulates in
`_actor_L_accum` / `_critic_L_accum` / `_alpha_accum`, divided by
`_update_count`. Q̄ / entropy / grad-norm are not yet accumulated —
adding them would require new reductions inside train_step and is
deferred to a polish chunk (bit-identity risk: every D2H / extra
reduce kernel could shift the SAC Pendulum baseline).
"""

from ..constants import DT
from ..core.metric import LogScalar


@fieldwise_init
struct SACMetrics(Copyable, Movable, ImplicitlyDestructible):
    var actor_loss:  LogScalar[DT]
    var critic_loss: LogScalar[DT]
    var alpha:       LogScalar[DT]
    var n_updates:   LogScalar[DT]   # cast to float for uniform Logger surface
