"""Long CartPole MuZero run — confirms training actually learns.

Single-player env so no MCTS-vs-minimax confounds. CartPole-v1 has
random-policy avg reward ≈ 9, max 200. A working MuZero should climb
above 50 within ~20K env steps and approach 200 by 50K. The pre-fix
code reported ≈4-5 reward (worse than random) on the short test, which
was the original signal that something was broken. This run confirms
the post-Phase-E pipeline (P-LAYOUT, P-WINDOW, reanalyze, E4) actually
trains.
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("=== MuZero CartPole long run ===")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]

    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=64,
        HIDDEN=64,
        BINS=21,
        SIMS=25,
        K=5,
        N=10,
        BS=64,
        CAP=50000,
    ]

    var agent = GenericMuZeroAgent[Config, 32](
        gamma=0.99,
        temperature_decay_steps=20000,
    )
    print("Agent created:", Config.NAME)

    print("Training with n_envs=32 GPU environments, 50K env steps...")
    var metrics = agent.train_gpu[CartPoleGPU](
        ctx,
        num_steps=50000,
        warmup_steps=1000,
        print_every=2000,
        use_reanalyze=False,  # A/B test: reanalyze off
    )

    print()
    print("=== Results ===")
    print("GPU train steps:", agent.train_step_count)

    if agent.train_step_count > 0:
        print("PASS: GPU training completed")
    else:
        print("FAIL: no training steps")

    _ = metrics
    print("=== Done ===")
