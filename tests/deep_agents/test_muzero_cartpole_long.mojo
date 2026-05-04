"""Long CartPole MuZero run with mid-training diagnostics.

5 phases × 10K env steps = 50K total. After each phase (including pre-init),
dump `diagnose_init_state` for 3 canonical CartPole obs (LEFT, CENTER, RIGHT)
to expose:
  1. Whether pred policy/value learn state-dependence (compare across the 3 obs)
  2. How that state-dependence evolves over training phases
  3. Whether MCTS visit counts diverge from uniform once pred is informed

If pred outputs are identical across the 3 obs in late phases, pred is state-blind
(representation collapse) regardless of training progress. If they differ but
reward still stuck at random, the policy/value targets are wrong upstream.

Caveats of the multi-call structure:
  - GPU replay buffer is freshly allocated each `train_gpu` call (lost between
    phases). Network weights persist via `download_to(agent.state)`.
  - `self.total_steps` accumulates across calls → temperature schedule is
    continuous (1.0 first 25K, 0.5 next 12.5K, 0.25 last 12.5K of total_steps).
  - Each phase does 1K warmup with random actions, so 5K total warmup mixed in
    (the agent's network ignores warmup actions for training; replay still
    stores them with uniform policy targets — minor noise).
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("=== MuZero CartPole long run with mid-training diagnostics ===")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]

    # LR=2e-2 + exponential decay (rate 0.8 per 1000 train steps) — matches
    # muzero-general/games/cartpole.py exactly. Phase G post-mortem
    # (2026-05-04) showed constant LR + AdamW WD=1e-4 or 1e-2 left rep
    # weights drifting unbounded; reference solves it via aggressive LR
    # decay, not via stronger WD. By train_step 10K, LR drops to ~0.0021;
    # by 20K, ~2.3e-4; by 50K, ~1.4e-7. So most weight movement happens
    # in first 5-10K train steps when the network is escaping the uniform
    # attractor; later updates are tiny and decay caps |W|.
    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=64,
        HIDDEN=64,
        BINS=21,
        LR=2e-2,
        SIMS=25,
        K=3,
        N=3,
        BS=16,
        CAP=50000,
    ]

    var agent = GenericMuZeroAgent[Config, 32](
        gamma=0.99,
        temperature_decay_steps=50000,
        pred_head_input_dim=64,
    )

    # Three canonical CartPole obs covering tilt directions:
    # obs format: [x, x_dot, theta, theta_dot]
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

    # Phase 0: post-init (network is just-Kaiming-then-zero-head, untrained).
    print()
    print("########## PHASE 0: post-init (no training yet) ##########")
    agent.diagnose_init_state(ctx, obs_left, String("phase=0 LEFT"))
    agent.diagnose_init_state(ctx, obs_center, String("phase=0 CENTER"))
    agent.diagnose_init_state(ctx, obs_right, String("phase=0 RIGHT"))

    # 5 training phases × 10K env steps each.
    comptime PHASES = 5
    comptime STEPS_PER_PHASE = 10000

    for phase in range(PHASES):
        print()
        print(
            "########## PHASE "
            + String(phase + 1)
            + ": training "
            + String(STEPS_PER_PHASE)
            + " env steps ##########"
        )
        var metrics = agent.train_gpu[CartPoleGPU](
            ctx,
            num_steps=STEPS_PER_PHASE,
            warmup_steps=1000 if phase == 0 else 0,
            print_every=2000,
            use_reanalyze=True,
            lr_decay_rate=0.8,
            lr_decay_steps=1000,
        )
        _ = metrics

        print()
        print(
            "########## PHASE "
            + String(phase + 1)
            + ": post-training diagnose ##########"
        )
        var lbl = String(" phase=") + String(phase + 1)
        agent.diagnose_init_state(ctx, obs_left, String("LEFT") + lbl)
        agent.diagnose_init_state(ctx, obs_center, String("CENTER") + lbl)
        agent.diagnose_init_state(ctx, obs_right, String("RIGHT") + lbl)

    print()
    print("=== Done ===")
    print("Total GPU train steps:", agent.train_step_count)
    print("Total env steps:", agent.total_steps)
