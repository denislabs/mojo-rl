"""CartPole with MuZero — Fully GPU (env + MCTS + training).

Uses Gumbel-MuZero (mctx-style) for action selection: Gumbel-Top-k
root sampling + Sequential Halving + deterministic σ(Q)-N/(1+ΣN)
interior selection + improved-policy training target. Provably
better policy improvement at low simulation budgets than PUCT —
see ``docs/mctx-main/`` and ``docs/mcts_gpu.pdf``.

Usage (Apple Silicon):
    pixi run -e apple mojo run -I . examples/cartpole/cartpole_muzero_gpu.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.deep_agents.muzero.policy_mode import GumbelMuZeroPolicy
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("MuZero on CartPole (Fully GPU, Gumbel-MuZero)")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]

    # MAX_K=2 = CartPole's action count. Sequential Halving collapses
    # to a single phase (log2(2)=1) of 12 sims per action — the SIMS=24
    # budget chosen to be cleanly divisible (was 25 with PUCT).
    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=128,
        HIDDEN=128,
        BINS=51,
        SIMS=24,
        POLICY=GumbelMuZeroPolicy[2],
    ]

    var agent = GenericMuZeroAgent[Config, 32](  # 32 parallel GPU envs
        gamma=0.997,
        v_min=-100.0,
        v_max=100.0,
        temperature=1.0,
        temperature_decay_steps=50000,
    )

    _ = agent.train_gpu[CartPoleGPU](
        ctx,
        num_steps=50000,
        warmup_steps=1000,
        print_every=1000,
    )
    print("Done! Train steps:", agent.train_step_count)
