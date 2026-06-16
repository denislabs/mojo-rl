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
  mean_q          — mean of online Q1(s, a) over the batch
  mean_target     — mean of target_y (Bellman target) across the chunk
  mean_reward     — mean of batch reward across the chunk
  mean_next_q     — mean of min(Q1_t, Q2_t)(s', a') over the batch
                    (the target-critic next-Q the TD bootstrap is built
                    from; reads the `min_q` ComputeGraph node)
  mean_done       — mean of batch done across the chunk
  mean_abs_action — mean |action| across the chunk
  train_steps     — cumulative training updates so far (NOT reset on flush)
  n_updates       — training updates THIS chunk (reset on flush)

`mean_next_q` reads the `min_q` intermediate of `TargetYBlock`'s
ComputeGraph via `node_out_ptr` (CPU diag walk; the GPU path leaves it
0.0, same convention as the other diagnostics). Wired 2026-05-30 once
the ComputeGraph node-output accessor landed."""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.metric import LogScalar


@fieldwise_init
struct SACMetrics(Copyable, Movable, ImplicitlyDestructible):
    var actor_loss:      LogScalar[DT]
    var critic_loss:     LogScalar[DT]
    var alpha:           LogScalar[DT]
    var mean_q:          LogScalar[DT]
    var mean_target:     LogScalar[DT]
    var mean_reward:     LogScalar[DT]
    var mean_next_q:     LogScalar[DT]
    var mean_done:       LogScalar[DT]
    var mean_abs_action: LogScalar[DT]
    var train_steps:     LogScalar[DT]
    var n_updates:       LogScalar[DT]
