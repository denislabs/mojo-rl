"""Compile-only smoke for PCN-MBPO agent fork — Phase B2.

Just instantiates `PCNMBPOAgent` with a small config; doesn't train. The
`do_model_rollouts_gpu` body is currently stubbed (see file docstring),
so any test that triggers GPU rollouts will no-op rather than crash.
"""

from mojo_rl.deep_agents.mbpo_pcn import (
    DefaultPCNMBPOConfig,
    PCNMBPOAgent,
)
from mojo_rl.deep_agents.core.strategies.termination import NeverTerminate


comptime CFG = DefaultPCNMBPOConfig[
    OBS=3, ACT=1, HIDDEN=64, CAP=10000, SYNTH_CAP=20000, BS=64,
    NUM_ENSEMBLE=3, NUM_ELITES=2, DYN_HIDDEN=64,
    DYN_BATCH_SIZE=64, DYN_ROLLOUT_BATCH=64,
    actor_lr=0.0003, critic_lr=0.0003, model_lr=0.001,
    t_infer=10, lr_x=0.01, dyn_grad_clip=1.0,
    TFn=NeverTerminate, action_scale=2.0,
]


def main() raises:
    print("PCN-MBPO compile smoke")
    var agent = PCNMBPOAgent[CFG](
        gamma=0.99, tau=0.005, action_scale=2.0,
        alpha=0.1, auto_alpha=True, alpha_lr=0.0001,
        target_entropy=-1.0,
        model_train_freq=250,
        rollout_min_length=1, rollout_max_length=1,
        rollout_min_epoch=0, rollout_max_epoch=5,
        num_rollouts_per_step=100,
        real_ratio=0.05,
        sac_updates_per_step=10,
    )
    print("  agent.gamma =", agent.gamma)
    print("=== Compile smoke OK ===")
