"""PPOMetrics — per-train-step bundle for the PPO trainer.

Mirrors `SACMetrics`: one @fieldwise_init struct, one LogScalar[DT] per
metric. `log_bundle[PPOMetrics, L: Logger]` walks via reflection and
emits one `log_scalar` call per field.

Fields correspond to what `PPOTrainer` accumulates over the K-epoch
minibatch SGD inside one `train_step`:
  * `actor_loss`  — mean of PPO clipped-surrogate loss across all minibatches
  * `critic_loss` — mean of value MSE across all minibatches
  * `n_updates`   — total minibatch updates inside this train_step
                    (= N_EPOCHS * N_MINIBATCHES). Cast to Float64
                    for uniform Logger surface.

Entropy / KL-divergence are not yet captured — they require modifying
`PPOActorLoss` to return them alongside the loss. Defer to a polish chunk."""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.metric import LogScalar


@fieldwise_init
struct PPOMetrics(Copyable, Movable, ImplicitlyDestructible):
    var actor_loss:  LogScalar[DT]
    var critic_loss: LogScalar[DT]
    var n_updates:   LogScalar[DT]
