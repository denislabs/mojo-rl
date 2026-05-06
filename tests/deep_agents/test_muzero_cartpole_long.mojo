"""Single-phase MuZero CartPole — diagnose at start + end, continuous training in between.

Prior multi-phase version (5 train_gpu calls of 10K each, with mid-training
diagnose between phases) proved actively harmful: each `train_gpu` allocates a
fresh GPU replay buffer, so 50K env steps' worth of trajectories got thrown
away 4 times. The reference (muzero-general) uses one continuous training loop
with persistent replay. With LayerNorm landed in RepModel and the rest of the
reference recipe matched, the multi-phase split was the last hurdle.

Diagnostic visibility comes from:
  - 3 canonical CartPole obs (LEFT/CENTER/RIGHT) diagnosed BEFORE training
  - The `print_every=2000` trace through training (25 prints over 50K)
  - 3 canonical obs diagnosed AFTER training
This sacrifices mid-training diagnose snapshots for replay continuity.
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("=== MuZero CartPole single-phase run ===")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]

    # Phase H12 (2026-05-05): full alignment with muzero-general's CartPole
    # config (references/muzero-general-master/games/cartpole.py), excluding
    # PER (requires implementation). Their config is RADICALLY smaller and
    # uses LONGER N-step horizons:
    #
    #   muzero-general              ours (was)        ours (now, aligned)
    #   ─────────────────────────   ──────────        ────────────────────
    #   encoding_size = 8           LATENT=64         LATENT=8
    #   fc_*_layers = [16]          HIDDEN=64         HIDDEN=16
    #   support_size = 10           v_min/max=±22     v_min/max=±10
    #   num_unroll_steps = 10       K=5               K=10
    #   td_steps = 50               N=10              N=50
    #   batch_size = 128            BS=64             BS=128
    #   root_exploration_fr = 0.25  Dirichlet fr=0.5  Dirichlet fr=0.25
    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=8,
        HIDDEN=16,
        BINS=21,
        LR=2e-2,
        SIMS=50,
        K=10,
        N=50,
        BS=128,
        CAP=50000,
    ]
    #
    # Why each matters:
    # - Smaller networks (LATENT 8, HIDDEN 16): less capacity to memorize
    #   biased data → forced to find generalizing features. The bias-
    #   amplification loop we saw locks in faster with bigger nets.
    # - Tighter support (±10 vs ±22): bin width = 1.0 in encoded space
    #   (vs 2.2). Reward r=0 vs r=1 differ by 41% mass in bin 11 (vs 19%
    #   with ±22) — 2× larger gradient on the only state-conditioned
    #   reward signal CartPole offers. Values >100 raw get clipped to
    #   encoded ±10, but that's a trade muzero-general accepts.
    # - K=10, N=50 (longest leverage): with N=50 the value target is
    #   dominated by 50 steps of ACTUAL observed reward, not by the
    #   undertrained value head's bootstrap. Most CartPole episodes are
    #   <50 steps so MOST sample positions get exact MC-style targets.
    #   This is our biggest single deviation from reference and likely
    #   the main reason we plateau where they converge.
    # - BS=128: 2× gradient variance reduction.
    # - Dirichlet fraction 0.25 (reverted from 0.5): the 0.5 bump was a
    #   band-aid that didn't help; reverting to align with reference.
    var agent = GenericMuZeroAgent[Config, 32](
        gamma=0.99,
        temperature_decay_steps=50000,
        pred_head_input_dim=16,
        v_min=-10.0,
        v_max=10.0,
    )

    # Canonical CartPole obs covering tilt directions:
    var obs_left = List[Float64]()
    obs_left.append(0.0)
    obs_left.append(0.0)
    obs_left.append(-0.1)
    obs_left.append(0.0)

    var obs_center = List[Float64]()
    obs_center.append(0.0)
    obs_center.append(0.0)
    obs_center.append(0.0)
    obs_center.append(0.0)

    var obs_right = List[Float64]()
    obs_right.append(0.0)
    obs_right.append(0.0)
    obs_right.append(0.1)
    obs_right.append(0.0)

    print()
    print("########## PRE-TRAINING diagnose (post-init) ##########")
    agent.diagnose_init_state(ctx, obs_left, String("pre-train LEFT"))
    agent.diagnose_init_state(ctx, obs_center, String("pre-train CENTER"))
    agent.diagnose_init_state(ctx, obs_right, String("pre-train RIGHT"))

    print()
    print("########## SINGLE TRAINING CALL: 50K env steps ##########")
    var metrics = agent.train_gpu[CartPoleGPU](
        ctx,
        num_steps=50000,
        warmup_steps=1000,
        print_every=2000,
        use_reanalyze=True,
        lr_decay_rate=0.8,
        lr_decay_steps=1000,
        use_per=True,
    )
    _ = metrics

    print()
    print("########## POST-TRAINING diagnose ##########")
    agent.diagnose_init_state(ctx, obs_left, String("post-train LEFT"))
    agent.diagnose_init_state(ctx, obs_center, String("post-train CENTER"))
    agent.diagnose_init_state(ctx, obs_right, String("post-train RIGHT"))

    print()
    print("=== Done ===")
    print("Total GPU train steps:", agent.train_step_count)
    print("Total env steps:", agent.total_steps)
