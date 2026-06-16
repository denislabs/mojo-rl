"""Evaluate + render a trained Humanoid SAC (deep_agents) **LayerNorm-critic**
checkpoint.

Companion to `sac_humanoid_nn_eval_cpu.mojo`, but matched to the LayerNorm
critic variant of the trainer (`sac_humanoid_nn_agent_gpu.mojo`). That run
swaps the preset's plain fused-`LinearReLU` critic for a pre-activation
LayerNorm MLP (`Linear → LayerNorm → ReLU`, repeated), which changes the
critic's `PARAM_SIZE` — so its checkpoint (`sac_humanoid_nn_ln.ckpt`) is NOT
loadable by the preset-based eval. This eval rebuilds the EXACT same nets via
the `SACAgent[...]` primitive so the checkpoint's parameter layout matches.

Greedy (actor-mean, no sampling) episodes with live 3D rendering via the
physics3d `RenderableEnv` interface.

IMPORTANT: Humanoid trains with `action_scale=0.4`. The greedy action is
scaled by `action_scale` (`a = tanh(μ) · action_scale`), so this eval MUST
construct the agent with the same `action_scale=0.4` to reproduce the trained
policy's torques.

Controls (renderer window):
  * close the window or press its quit key to stop early.

Run:
    pixi run mojo run -I . examples/humanoid/sac_humanoid_nn_ln_eval_cpu.mojo
"""

from std.random import seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.primitives.layer_norm import LayerNorm
from mojo_rl.deep_agents.sac import SACAgent, SACActorNet
from mojo_rl.deep_agents.training.blocks import ReplaySampleStep
from mojo_rl.deep_agents.data.any_replay import AnyReplay
from mojo_rl.envs.humanoid import Humanoid


# =============================================================================
# Architecture — MUST match the LayerNorm-critic trainer exactly
# =============================================================================

comptime EnvT = Humanoid[DT, TERMINATE_ON_UNHEALTHY=True]
comptime OBS_DIM = EnvT.OBS_DIM  # 45
comptime ACT_DIM = EnvT.ACTION_DIM  # 17
# HIDDEN + CHECKPOINT_PATH MUST match the trainer (LayerNorm critic, HIDDEN=256).
comptime HIDDEN = 256
comptime BATCH = 256
# Replay capacity is irrelevant for greedy eval (no replay used) and is NOT in
# the checkpoint — keep it small to avoid a large CPU allocation.
comptime REPLAY_CAPACITY = 100_000

comptime CHECKPOINT_PATH = "sac_humanoid_nn_ln.ckpt"
comptime ACTION_SCALE = Scalar[DT](0.4)  # MUST match training

# ─── Nets — identical aliases to the trainer (LayerNorm critic) ────────────
comptime ActorNet = SACActorNet[OBS_DIM, ACT_DIM, HIDDEN]
comptime CriticNetLN = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    LayerNorm[HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    LayerNorm[HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime SampleT = ReplaySampleStep[
    AnyReplay["cpu", OBS_DIM, ACT_DIM, REPLAY_CAPACITY], BATCH
]

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000
comptime FRAME_DELAY_MS = 16  # ~60 FPS playback


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents) — Humanoid CPU eval + 3D rendering [LayerNorm]")
    print("=" * 70)
    print("  OBS_DIM         =", OBS_DIM)
    print("  ACT_DIM         =", ACT_DIM)
    print("  HIDDEN          =", HIDDEN)
    print("  critic          = pre-activation LayerNorm MLP")
    print("  action_scale    =", ACTION_SCALE)
    print("  Checkpoint      =", CHECKPOINT_PATH)
    print("  Episodes        =", NUM_EPISODES)
    print("  Max steps/ep    =", MAX_STEPS)
    print("=" * 70)

    # ─── Agent — built from the SACAgent primitive with the LayerNorm critic,
    # exactly as the trainer does, so the checkpoint's parameter layout matches.
    # `action_scale=0.4` is required for the greedy action to reproduce the
    # trained policy (it scales the tanh output).
    var agent = SACAgent["cpu", SampleT, ActorNet, CriticNetLN](
        action_scale=ACTION_SCALE,
    )

    # ─── Load checkpoint ─────────────────────────────────────────────────
    print("Loading checkpoint...")
    try:
        agent.load(CHECKPOINT_PATH)
        print("Checkpoint loaded.")
    except e:
        print("ERROR loading checkpoint:", e)
        print("Train first (GPU):")
        print(
            "  pixi run -e nvidia mojo run -I ."
            " examples/humanoid/sac_humanoid_nn_agent_gpu.mojo"
        )
        return
    print()

    # ─── Greedy eval + live 3D rendering ─────────────────────────────────
    # `eval_render` is the facade's render-enabled greedy eval: it owns the
    # init_renderer / per-step render+delay / quit / close handling and returns
    # the mean episode return. Falls back to headless if no renderer.
    print("-" * 70)
    var env = EnvT()
    var avg_reward = Float64(
        agent.eval_render[EnvT](
            env,
            NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS,
            frame_delay_ms=FRAME_DELAY_MS,
            verbose=True,
        )
    )

    print("-" * 70)
    print("EVAL SUMMARY — Humanoid (SAC deep_agents, LayerNorm critic)")
    print("-" * 70)
    print("  Episodes        =", NUM_EPISODES)
    print("  Average reward  =", avg_reward)
    if avg_reward > 5000.0:
        print("  Result: EXCELLENT — walking (mean > 5000).")
    elif avg_reward > 2000.0:
        print("  Result: STRONG — sustained upright locomotion (mean > 2000).")
    elif avg_reward > 1000.0:
        print("  Result: PROGRESS — staying upright (mean > 1000).")
    elif avg_reward > 0.0:
        print("  Result: LEARNING — positive return (mean > 0).")
    else:
        print("  Result: EARLY — still exploring (mean < 0).")
    print("=" * 70)
