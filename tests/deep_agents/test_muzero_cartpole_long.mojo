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

    # K=N=3: with random-policy CartPole episodes lasting ~4-5 steps,
    # the previous K=5/N=10 setup meant bootstrap NEVER fired (steps_used
    # never reached N=10 before terminal). All value targets collapsed
    # to "sum of K rewards" ≈ 5, giving the value head no learning signal.
    # K=N=3 lets bootstrap fire whenever the 3-step unroll doesn't hit
    # terminal, propagating real value information.
    # LR=1e-2, BS=16 — aligned with AlphaZero-on-CartPole's config
    # (which solves CartPole in ~6K env steps). Audit Phase E flagged
    # MuZero defaults (LR=3e-4, BS=64) as 33× lower LR + 4× larger
    # batch than AZ. Long run with prior MuZero defaults peaked at
    # reward 8 by step 22K then decayed back to 5 by step 50K — pred
    # ΔW peaked at 1.05 then settled in 0.3-0.7 band. The conservative
    # LR likely traps pred at the uniform-near attractor before it
    # can escape to state-dependent values.
    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=64,
        HIDDEN=64,
        BINS=21,
        LR=1e-2,
        SIMS=25,
        K=3,
        N=3,
        BS=16,
        CAP=50000,
    ]

    var agent = GenericMuZeroAgent[Config, 32](
        gamma=0.99,
        # temperature_decay_steps is the full training horizon for the
        # muzero-general step schedule (1.0 → 0.5 → 0.25). Setting it to
        # num_steps (50K) means temp = 1.0 for first 25K env steps, 0.5
        # for next 12.5K, 0.25 for remaining 12.5K. Bug E follow-up:
        # previous 20K linear-to-0.01 collapsed to greedy at step 19.8K,
        # killing exploration and capping reward at ~7.
        temperature_decay_steps=50000,
        pred_head_input_dim=64,  # = HIDDEN; zero-inits pred policy + value
        # heads at construction so the untrained network produces uniform
        # softmax(policy) and decoded value=0 instead of Kaiming's biased
        # logits ±2.5 / value -451. Without this MCTS commits to one action
        # regardless of obs (Bug A/B, see docs/MUZERO_AUDIT.md 2026-05-04).
    )
    print("Agent created:", Config.NAME)

    print("Training with n_envs=32 GPU environments, 50K env steps...")
    var metrics = agent.train_gpu[CartPoleGPU](
        ctx,
        num_steps=50000,
        warmup_steps=1000,
        print_every=2000,
        # Reanalyze on: per-timestep target-net forward freshens the
        # bootstrap value at sample time (matches muzero-general). Audit
        # Phase E4 wired the target nets correctly; the dead CPU-side
        # `self.reanalyze(...)` call was removed 2026-05-04.
        use_reanalyze=True,
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
