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
    # Termination-head BCE (item B); 0 unless bce_coef > 0 (episodic envs).
    var termination_loss: Scalar[DT]
    var wm_loss: Scalar[DT]
    var pi_loss: Scalar[DT]
    var pi_scale: Scalar[DT]
    # Q-Values group (avg-of-2 decoded Q at the policy's actions).
    var q_mean: Scalar[DT]
    var q_min: Scalar[DT]
    var q_max: Scalar[DT]
    # TD Targets group (the stop-grad value targets fed to the WM value loss).
    var td_target_mean: Scalar[DT]
    var td_target_min: Scalar[DT]
    var td_target_max: Scalar[DT]
