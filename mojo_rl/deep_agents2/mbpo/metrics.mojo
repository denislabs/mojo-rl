"""MBPOMetrics — per-train-step bundle for the MBPO trainer.

Mirrors `SACMetrics`: one @fieldwise_init struct, one LogScalar[DT] per
metric. `log_bundle[MBPOMetrics, L: Logger]` walks via reflection and
emits one `log_scalar` call per field.

Fields correspond to the SAC actor/critic updates that run inside one
MBPO train_step (against the mixed real+synthetic batch):
  * `actor_loss`  — mean SAC actor loss
  * `critic_loss` — mean SAC critic loss
  * `alpha`       — current entropy temperature (point-in-time)
  * `train_steps` — cumulative SAC mini-updates since trainer was made
                    (NOT reset on flush)
  * `n_updates`   — total SAC mini-updates this chunk
                    (typically `sac_updates_per_step`).

Dynamics-ensemble loss is not exposed here yet — the ensemble trains on a
different cadence (`model_train_freq`) and emits its own logs internally."""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.metric import LogScalar


@fieldwise_init
struct MBPOMetrics(Copyable, Movable, ImplicitlyDestructible):
    var actor_loss:  LogScalar[DT]
    var critic_loss: LogScalar[DT]
    var alpha:       LogScalar[DT]
    var train_steps: LogScalar[DT]
    var n_updates:   LogScalar[DT]
