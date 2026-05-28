"""SACMetrics — per-train-step bundle for the SAC trainer.

One @fieldwise_init struct, one LogScalar[DT] per metric. Field names
match the legacy GPU-SAC logging convention emitted from
`mojo_rl/deep_agents/core/agents/offpolicy_agent.mojo:1979-2005` so
downstream monitoring tools see a consistent stream regardless of
whether the run came from the old `DeepSACAgent` or the new
`SACAgent`. `log_bundle[SACMetrics, L: Logger]` walks via reflection
and emits one `log_scalar(name, value, step)` per field.

Naming convention (one-to-one with legacy):
  actor_loss      — mean SAC actor loss across the chunk
  critic_loss     — mean SAC critic loss (twin-critic min) across the chunk
  alpha           — point-in-time entropy temperature (= exp(log_alpha))
  mean_target     — mean of target_y (Bellman target) across the chunk
  mean_reward     — mean of batch reward across the chunk
  mean_done       — mean of batch done across the chunk
  mean_abs_action — mean |action| across the chunk
  train_steps     — cumulative training updates so far (NOT reset on flush)
  n_updates       — training updates THIS chunk (reset on flush)

`mean_q` and `mean_next_q` from the legacy bundle are deferred — both
require exposing the Q-network's batch output from
`TwinCriticUpdateBlock`, which means an extra forward pass (or capturing
the existing pre-loss tensor) and a bit-identity re-verification. Track
post-Track-1."""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.metric import LogScalar


@fieldwise_init
struct SACMetrics(Copyable, Movable, ImplicitlyDestructible):
    var actor_loss:      LogScalar[DT]
    var critic_loss:     LogScalar[DT]
    var alpha:           LogScalar[DT]
    var mean_target:     LogScalar[DT]
    var mean_reward:     LogScalar[DT]
    var mean_done:       LogScalar[DT]
    var mean_abs_action: LogScalar[DT]
    var train_steps:     LogScalar[DT]
    var n_updates:       LogScalar[DT]
