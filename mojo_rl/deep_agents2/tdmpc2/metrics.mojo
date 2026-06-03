"""TDMPC2Metrics — the diagnostic bundle drained by the agent's flush.

Per-component world-model losses (consistency / reward / value) + their sum
(wm_loss), the policy loss, and the RunningScale value (pi_scale). Mirrors
the SACMetrics role; consumed by flush_metrics / streamed via
flush_metrics_through_logger.
"""

from mojo_rl.nn2.constants import DT


@fieldwise_init
struct TDMPC2Metrics(Copyable & Movable):
    var consistency_loss: Scalar[DT]
    var reward_loss: Scalar[DT]
    var value_loss: Scalar[DT]
    var wm_loss: Scalar[DT]
    var pi_loss: Scalar[DT]
    var pi_scale: Scalar[DT]
