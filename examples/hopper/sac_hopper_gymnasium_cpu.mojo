"""SAC on Gymnasium Hopper-v5 (CPU) — diagnostic test.

Runs our SAC implementation against the real MuJoCo Hopper via Gymnasium wrapper.
This is SLOW (CPU only, single env) but isolates whether collapse is due to
our physics engine or our SAC implementation.

Run with:
    pixi run mojo run -I . examples/hopper/sac_hopper_gymnasium_cpu.mojo
"""

from std.random import seed

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.deep_agents.core import run_offpolicy_continuous_train
from mojo_rl.envs.gymnasium import make_hopper


comptime OBS_DIM = 11
comptime ACTION_DIM = 3
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC on Gymnasium Hopper-v5 (CPU) — Diagnostic Test")
    print("=" * 70)

    var env = make_hopper()
    print("Env: " + env.get_info())

    var agent = DeepSACAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        actor_lr=0.0003,
        critic_lr=0.0003,
    ](
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=False,
        target_entropy=-3.0,
    )

    var cpu_state = agent.make_cpu_state()

    print("Training SAC on Gymnasium Hopper-v5 (CPU, single env)...")
    print("This will be slow — purely diagnostic to test SAC correctness.")
    print("-" * 70)

    var metrics = run_offpolicy_continuous_train(
        agent,
        cpu_state,
        env,
        num_episodes=2000,
        max_steps_per_episode=1000,
        warmup_steps=10_000,
        train_every=1,
        verbose=True,
        print_every=50,
    )

    print("-" * 70)
    print(
        "Final avg reward (last 100): "
        + String(metrics.mean_reward_last_n(100))[byte=:8]
    )
    print("Best episode reward: " + String(metrics.max_reward())[byte=:8])

    env.close()
