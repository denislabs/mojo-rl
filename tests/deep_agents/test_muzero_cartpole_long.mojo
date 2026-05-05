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

    # K=5, BS=64 — match muzero-general's CartPole config exactly. Phase G
    # post-mortem 2026-05-05: with K=3 + BS=16, |dyn| stayed flat at 2.86
    # the entire run because the value-supervision unroll was too shallow
    # to teach dyn pole physics, and small batches produced too-noisy
    # gradients. Reference's K=5 gives 5 dyn applications per sample
    # (vs our 3) so value targets at k=5 diverge more across states,
    # and BS=64 reduces gradient variance.
    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=64,
        HIDDEN=64,
        BINS=21,
        LR=2e-2,
        SIMS=50,
        K=5,
        N=10,
        BS=64,
        CAP=50000,
    ]

    # Pred head zero-init RE-ENABLED (2026-05-05). The "removal" experiment
    # showed Kaiming-init pred heads produce logits ~[+2.4, -0.9] (softmax
    # 96/4) and value ~-440 at init; MCTS visits become identical [32,16]
    # across all obs from step 1, locking the network into a state-blind
    # attractor that survives even with all reference hyperparams matched
    # (LR=0.02, K=5, BS=64, MinMaxNorm-in-autograd, Adam-L2-WD=1e-4) — final
    # state had bit-identical hidden/policy/value across LEFT/CENTER/RIGHT.
    # Zero-init breaks the early-bias attractor; combined with Dirichlet
    # fraction=0.5 (bumped from 0.25 in the MLP config) for stronger root
    # exploration to escape the symmetric fixed point.
    var agent = GenericMuZeroAgent[Config, 32](
        gamma=0.99,
        temperature_decay_steps=50000,
        pred_head_input_dim=64,
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
