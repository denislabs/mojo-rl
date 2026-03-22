"""CartPole with MuZero — Fully GPU (env + MCTS + training).

Usage (Apple Silicon):
    pixi run -e apple mojo run -I . examples/cartpole/cartpole_muzero_gpu.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.envs.cartpole import CartPoleEnv


fn main() raises:
    print("MuZero on CartPole (Fully GPU)")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]

    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM, CartPoleGPU.NUM_ACTIONS,
        LATENT=128, HIDDEN=128, BINS=51, SIMS=25,
    ]

    var agent = GenericMuZeroAgent[Config, 32](  # 32 parallel GPU envs
        gamma=0.997, v_min=-100.0, v_max=100.0,
        temperature=1.0, temperature_decay_steps=50000,
    )

    _ = agent.train_gpu[CartPoleGPU](
        ctx, num_steps=50000, warmup_steps=1000,
        print_every=10000,
    )
    print("Done! Train steps:", agent.train_step_count)
