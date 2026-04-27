"""Smoke test: import REDQ from its new location after the move from core/.

Exercises every symbol exported via `mojo_rl.deep_agents` plus the
package-root path `mojo_rl.deep_agents.redq.*`.
"""

from mojo_rl.deep_agents import (
    REDQAgent,
    REDQConfig,
    DefaultREDQConfig,
    DefaultREDQLNConfig,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
    REDQ_TARGET_REM,
)
from mojo_rl.deep_agents.redq import REDQAgent as REDQAgent2
from mojo_rl.deep_agents.redq.kernels import redq_ensemble_target_kernel


def main() raises:
    print("REDQ_TARGET_MIN =", REDQ_TARGET_MIN)
    print("REDQ_TARGET_AVE =", REDQ_TARGET_AVE)
    print("REDQ_TARGET_REM =", REDQ_TARGET_REM)
    # Referencing compile-time members on the config struct forces trait
    # resolution (OffPolicyConfig + REDQConfig + all strategies).
    comptime C = DefaultREDQConfig[17, 6]
    print("NUM_ENSEMBLE =", C.NUM_ENSEMBLE)
    print("NUM_MIN =", C.NUM_MIN)
    print("UTD_RATIO =", C.UTD_RATIO)
    print("POLICY_DELAY =", C.POLICY_DELAY)
    print("NAME =", C.NAME)
    print("PASS: REDQ symbols resolve from new location")
