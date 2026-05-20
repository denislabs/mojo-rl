"""Pendulum SAC — Phase-2b random-init (architectural control).

The architectural control for the PCN-init fine-tune. Same EncoderPrefix
4-layer architecture, but the first 2 layers are left at the agent's
default Xavier init — no encoder pretraining. SAC trains end-to-end
on raw 3D obs through the same network shape.

Pairs with `test_pendulum_sac_pcn_finetune.mojo`. The two together
isolate whether PCN pretraining gives SAC anything over plain Xavier
under the same architecture.

Run:
    pixi run mojo run -I . tests/pcn/test_pendulum_sac_random_finetune.mojo
"""

from std.time import perf_counter_ns

from mojo_rl.nn.constants import dtype
from mojo_rl.experimental.pcn import EncoderPrefixSACConfig
from mojo_rl.envs import PendulumEnv
from mojo_rl.deep_agents.core.agents import GenericOffPolicyAgent
from mojo_rl.deep_agents.core.training.offpolicy_train import (
    run_offpolicy_continuous_train,
)


comptime HIDDEN = 64
comptime ACTION_DIM = 1
comptime OBS_DIM = 3

comptime SAC_NUM_STEPS = 40_000
comptime SAC_MAX_STEPS = 200
comptime SAC_WARMUP_STEPS = 1000
comptime SAC_PRINT_EVERY = 50


def main() raises:
    print("=" * 60)
    print("Pendulum SAC — Phase-2b (random-init, architectural control)")
    print("=" * 60)
    print("  Encoder    : none (Xavier random in the prefix layers)")
    print("  SAC arch   : EncoderPrefix (LinearTanh -> Linear -> LinearReLU -> heads)")
    print("  SAC eps    :", SAC_NUM_STEPS)

    var env = PendulumEnv[dtype]()

    comptime AgentType = GenericOffPolicyAgent[
        EncoderPrefixSACConfig[
            OBS=OBS_DIM,
            ACT=ACTION_DIM,
            HIDDEN=HIDDEN,
            CAP=50000,
            BS=64,
            actor_lr=0.0003,
            critic_lr=0.0003,
            action_scale=2.0,
        ]
    ]
    var agent = AgentType(
        gamma=0.99,
        tau=0.005,
        action_scale=2.0,
        alpha=0.1,
        auto_alpha=True,
        alpha_lr=0.0001,
    )

    # Use the same external cpu_state path as the PCN-init test so the two
    # runs are directly comparable (no injection here — Xavier init only).
    var cpu_state = AgentType.CPUStateType()

    print("\n  --- SAC training (no encoder pretraining) ---")
    var t_sac0 = perf_counter_ns()
    var metrics = run_offpolicy_continuous_train(
        agent,
        cpu_state,
        env,
        num_steps=SAC_NUM_STEPS,
        max_steps_per_episode=SAC_MAX_STEPS,
        warmup_steps=SAC_WARMUP_STEPS,
        train_every=1,
        verbose=True,
        print_every=SAC_PRINT_EVERY,
        environment_name="Pendulum (random-init FT)",
    )
    var sac_train_t = Float64(perf_counter_ns() - t_sac0) / 1e9

    print("\n  === per-episode returns (CSV: ep,return,steps) ===")
    var rewards = metrics.get_rewards()
    var steps = metrics.get_steps()
    for i in range(len(rewards)):
        print("  CSV:", i, ",", rewards[i], ",", steps[i])

    print("\n  === Phase-2b summary (random-init) ===")
    print("  SAC train wall :", sac_train_t, "s")
    print("  Final α        :", String(agent.alpha)[byte=:6])
    print("  Last-20 avg    :", metrics.mean_reward_last_n(20))
    print("=== Done ===")
