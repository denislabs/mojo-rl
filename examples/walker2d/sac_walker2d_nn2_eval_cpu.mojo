"""Evaluate + render a trained Walker2d SAC (deep_agents2) checkpoint.

Loads the one-file `nn2-ckpt v2` checkpoint written by
`sac_walker2d_nn2_agent_gpu.mojo` (GPU training) and runs deterministic
(greedy / actor-mean, no sampling) episodes with live 3D rendering via the
physics3d `RenderableEnv` interface — the same renderer the legacy
`sac_walker2d_eval_cpu.mojo` drives, but on the new `SACAgent` facade.

The agent architecture here MUST match the training script exactly (same
`ActorNet` / `CriticNet`, dims, hidden width) or `agent.load` will read a
mismatched parameter layout. In particular the layer *fusion* must match:
the checkpoint section names embed the trunk index, so a fused
`LinearReLU` (one module) and an unfused `Linear` + `ReLU` pair (two
modules) produce DIFFERENT layouts even though the math is identical. This
script mirrors the GPU trainer's fused `LinearReLU` actor/critic so its
checkpoints load directly. (The CPU trainer `sac_walker2d_nn2_agent.mojo`
still uses the unfused `Linear` + `ReLU` form — to eval one of *its*
checkpoints, switch the nets below back to that layout.)

Controls (renderer window):
  * close the window or press its quit key to stop early.

Run:
    pixi run mojo run -I . examples/walker2d/sac_walker2d_nn2_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_relu import LinearReLU
from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.sac import SACAgent
from mojo_rl.deep_agents2.training.blocks import UniformSampleCpuStep
from mojo_rl.envs.walker2d import Walker2d


# =============================================================================
# Architecture — MUST match sac_walker2d_nn2_agent.mojo
# =============================================================================

comptime EnvT = Walker2d[DT, TERMINATE_ON_UNHEALTHY=True]
comptime OBS_DIM = EnvT.OBS_DIM  # 17
comptime ACT_DIM = EnvT.ACTION_DIM  # 6
comptime HIDDEN = 256
comptime BATCH = 256
comptime REPLAY_CAPACITY = 100_000

comptime CHECKPOINT_PATH = "sac_walker2d_nn2.ckpt"

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000
comptime FRAME_DELAY_MS = 16  # ~60 FPS playback


comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    LinearReLU[OBS_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
comptime CriticNet = Sequential[
    LinearReLU[OBS_DIM + ACT_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents2) — Walker2d CPU eval + 3D rendering")
    print("=" * 70)
    print("  OBS_DIM         =", OBS_DIM)
    print("  ACT_DIM         =", ACT_DIM)
    print("  HIDDEN          =", HIDDEN)
    print("  Checkpoint      =", CHECKPOINT_PATH)
    print("  Episodes        =", NUM_EPISODES)
    print("  Max steps/ep    =", MAX_STEPS)
    print("=" * 70)

    # ─── Agent (architecture must match training) ────────────────────────
    var agent = SACAgent[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet,
        CriticNet,
    ](
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        init_alpha=0.2,
        target_entropy=-Scalar[DT](ACT_DIM),
        learning_starts=1_000,
        window_size=100,
        initial_episode_fill=0.0,
        use_ere=False,
        ere_eta=0.996,
    )

    # ─── Load checkpoint ─────────────────────────────────────────────────
    print("Loading checkpoint...")
    try:
        agent.load(CHECKPOINT_PATH)
        print("Checkpoint loaded.")
    except e:
        print("ERROR loading checkpoint:", e)
        print("Train first:")
        print(
            "  pixi run mojo run -I ."
            " examples/walker2d/sac_walker2d_nn2_agent.mojo"
        )
        return
    print()

    # ─── Env + renderer ──────────────────────────────────────────────────
    var env = EnvT()
    var have_renderer = env.init_renderer()
    if not have_renderer:
        print(
            "WARNING: renderer unavailable — running headless (reward only)."
        )

    print("-" * 70)
    var t_start = perf_counter_ns()
    var total_reward = Scalar[DT](0.0)
    var quit = False

    for ep in range(NUM_EPISODES):
        if quit:
            break
        var obs = env.reset_obs_list()
        var ep_reward = Scalar[DT](0.0)
        var ep_steps = 0
        var act = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))

        for _ in range(MAX_STEPS):
            # Deterministic (actor-mean) action.
            agent.select_greedy_action(obs, act)

            var result = env.step_continuous_vec[DT](act)
            obs = result[0].copy()
            ep_reward += result[1]
            var done = result[2]
            ep_steps += 1

            if have_renderer:
                env.render_frame()
                env.renderer_delay(FRAME_DELAY_MS)
                if env.check_renderer_quit() or not env.is_renderer_open():
                    quit = True
                    break

            if done:
                break

        total_reward += ep_reward
        print(
            "  Episode",
            ep + 1,
            "— reward =",
            ep_reward,
            " steps =",
            ep_steps,
        )

    if have_renderer:
        env.close_renderer()

    var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
    var avg_reward = Float64(total_reward) / Float64(NUM_EPISODES)

    print("-" * 70)
    print("EVAL SUMMARY — Walker2d (SAC deep_agents2)")
    print("-" * 70)
    print("  Episodes        =", NUM_EPISODES)
    print("  Average reward  =", avg_reward)
    print("  Eval time       =", elapsed_s, "s")
    if avg_reward > 4000.0:
        print("  Result: EXCELLENT — running fast.")
    elif avg_reward > 2000.0:
        print("  Result: GOOD — learned to walk.")
    elif avg_reward > 1000.0:
        print("  Result: OKAY — upright + moving.")
    elif avg_reward > 0.0:
        print("  Result: LEARNING — positive but not optimal.")
    else:
        print("  Result: POOR — needs more training.")
    print("=" * 70)
