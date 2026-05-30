"""MBPOMetrics — per-train-step bundle for the MBPO trainer.

Mirrors `SACMetrics`: one @fieldwise_init struct, one LogScalar[DT] per
metric. `log_bundle[MBPOMetrics, L: Logger]` walks via reflection and
emits one `log_scalar` call per field.

Fields correspond to the SAC actor/critic updates that run inside one
MBPO train_step (against the mixed real+synthetic batch):
  * `actor_loss`  — mean SAC actor loss
  * `critic_loss` — mean SAC critic loss
  * `alpha`       — current entropy temperature (point-in-time)
  * `mean_q`      — mean Q1(s, a) over the mixed batch (SAC critic forward)
  * `mean_reward` — mean reward of the mixed real+synthetic minibatch
  * `dyn_loss`    — mean dynamics-ensemble Gaussian-NLL loss over the
                    member-steps run since the last flush (the ensemble
                    trains on its own `model_train_freq` cadence, so this
                    is averaged independently of the SAC `n_updates`)
  * `train_steps` — cumulative SAC mini-updates since trainer was made
                    (NOT reset on flush)
  * `n_updates`   — total SAC mini-updates this chunk
                    (typically `sac_updates_per_step`).

`mean_q` / `mean_reward` mirror the SAC bundle (CPU-only diag walk);
`dyn_loss` surfaces the ensemble NLL the legacy MBPO trainer logged.
On the (unreachable) GPU train path these three read 0.0 — same
convention as DQN/C51/PPO."""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.metric import LogScalar


@fieldwise_init
struct MBPOMetrics(Copyable, Movable, ImplicitlyDestructible):
    var actor_loss:  LogScalar[DT]
    var critic_loss: LogScalar[DT]
    var alpha:       LogScalar[DT]
    var mean_q:      LogScalar[DT]
    var mean_reward: LogScalar[DT]
    var dyn_loss:    LogScalar[DT]
    var train_steps: LogScalar[DT]
    var n_updates:   LogScalar[DT]
