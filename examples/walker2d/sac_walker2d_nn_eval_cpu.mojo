"""Evaluate + render a trained Walker2d SAC (deep_agents) checkpoint.

Loads the one-file `nn-ckpt v2` checkpoint written by either Walker2d SAC
trainer (`sac_walker2d_training.mojo` CPU or `sac_walker2d_training_gpu.mojo`
GPU) and runs deterministic (greedy / actor-mean, no sampling) episodes with
live 3D rendering via the physics3d `RenderableEnv` interface — the same
renderer the legacy `sac_walker2d_eval_cpu.mojo` drives, but on the new
`SACAgent` facade.

All three scripts now build the agent through the shared `SAC[...]` preset
(`deep_agents.sac`), so the actor/critic architecture — and therefore the
checkpoint's parameter layout — is identical across CPU training, GPU
training, and this eval. (The checkpoint section names embed each module's
trunk index, so the layer *fusion* must match: the preset's fused
`LinearReLU` hidden layers are one module each, vs two for an unfused
`Linear` + `ReLU` pair. Sharing the preset removes that footgun.)

Controls (renderer window):
  * close the window or press its quit key to stop early.

Run:
    pixi run mojo run -I . examples/walker2d/sac_walker2d_nn_eval_cpu.mojo
"""

from std.random import seed

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac import SAC
from mojo_rl.envs.walker2d import Walker2d


# =============================================================================
# Architecture — comes from the `SAC[...]` preset (matches both trainers)
# =============================================================================

comptime EnvT = Walker2d[DT, TERMINATE_ON_UNHEALTHY=True]
comptime OBS_DIM = EnvT.OBS_DIM  # 17
comptime ACT_DIM = EnvT.ACTION_DIM  # 6
comptime HIDDEN = 256
comptime BATCH = 256
comptime REPLAY_CAPACITY = 100_000

comptime CHECKPOINT_PATH = "sac_walker2d_nn.ckpt"

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000
comptime FRAME_DELAY_MS = 16  # ~60 FPS playback


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents) — Walker2d CPU eval + 3D rendering")
    print("=" * 70)
    print("  OBS_DIM         =", OBS_DIM)
    print("  ACT_DIM         =", ACT_DIM)
    print("  HIDDEN          =", HIDDEN)
    print("  Checkpoint      =", CHECKPOINT_PATH)
    print("  Episodes        =", NUM_EPISODES)
    print("  Max steps/ep    =", MAX_STEPS)
    print("=" * 70)

    # ─── Agent (architecture comes from the SAC preset — matches training) ─
    # Same `SAC[...]` preset the trainers use, so the loaded checkpoint's
    # parameter layout matches exactly. Hyperparameters are irrelevant for
    # greedy eval; we just need the net shapes, which the preset fixes.
    var agent = SAC[
        "cpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN
    ]()

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
            " examples/walker2d/sac_walker2d_training.mojo"
        )
        return
    print()

    # ─── Greedy eval + live 3D rendering ─────────────────────────────────
    # `eval_render` is the facade's render-enabled greedy eval: it owns the
    # init_renderer / per-step render+delay / quit / close handling (the
    # `RenderableEnv` loop that used to be inlined here) and returns the mean
    # episode return. Falls back to headless if no renderer is available.
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
    print("EVAL SUMMARY — Walker2d (SAC deep_agents)")
    print("-" * 70)
    print("  Episodes        =", NUM_EPISODES)
    print("  Average reward  =", avg_reward)
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
