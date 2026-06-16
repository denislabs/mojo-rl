"""REDQOFEMetrics — drained-window log bundle for REDQ-OFE training.

Same shape as `REDQMetrics` plus the OFE aux loss. Accumulators on
the trainer collect per-train-step values; `flush_metrics` divides
sums by counts to produce a snapshot and resets the accumulators.
The driver / agent calls this on a `diag_every` cadence.

All fields are host scalars (REDQ-OFE doesn't capture under CUDA
graphs — host control flow with subset sampling + policy delay +
aux interleave). Counts (`n_updates`, `n_actor_updates`) are kept
so the caller can detect the no-updates-this-window case.
"""

from mojo_rl.nn.constants import DT


@fieldwise_init
struct REDQOFEMetrics(Movable & ImplicitlyDestructible):
    """Per-flush-window mean / count bundle.

    Means are computed as `sum / count`; if `count == 0` the
    corresponding mean is 0.0 (so a fresh flush right after `make`
    doesn't NaN-poison downstream loggers)."""

    var critic_loss: Scalar[DT]      # mean Σᵢ MSE(Qᵢ, y) per train_step
    var actor_loss: Scalar[DT]       # mean SAC actor loss (when fired)
    var alpha: Scalar[DT]             # mean exp(log_α) over the window
    var log_prob_mean: Scalar[DT]     # mean log_π(a|s) over actor steps
    var aux_loss: Scalar[DT]          # mean MSE(predictor, s') over window
    var n_updates: Int                # # outer train_step calls that ran
    var n_actor_updates: Int          # # times actor + α fired
