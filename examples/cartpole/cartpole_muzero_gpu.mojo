"""CartPole with MuZero — Fully GPU (env + MCTS + training).

Default PUCT path. Gumbel-MuZero is available via
``POLICY=GumbelMuZeroPolicy[K]`` but for CartPole's 2-action space
Sequential Halving has no halving to do, so Gumbel offers no
discrimination advantage over PUCT.

NOTE: GPU MuZero has historically not converged on CartPole — CPU
``train()`` is the working reference. The Phase-K action-encoding bug
fix (2026-05-22) unblocked dynamics-network learning (action input
was being read from a misaligned scalar buffer as if it were one-hot),
but the policy still doesn't break out of random-play baseline. See
``docs/MUZERO_GPU_AUDIT_2026-05-22.md`` for the current state of the
investigation.

Usage (Apple Silicon):
    pixi run -e apple mojo run -I . examples/cartpole/cartpole_muzero_gpu.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("MuZero on CartPole (Fully GPU)")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]

    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=128,
        HIDDEN=128,
        BINS=51,
        SIMS=25,
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
