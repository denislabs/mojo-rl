"""MuZero init-state diagnostic — what does the untrained network do?

Calls `agent.diagnose_init_state(ctx, obs)` for two canonical CartPole obs
(pole tilting RIGHT then LEFT). Pure forward + MCTS, no training. The
agent's method handles all the buffer setup; this test just provides the
canonical inputs and labels.

Reads we're looking for:
  - If softmax(policy) is wildly skewed (e.g. [0.99, 0.01]) before any
    training, pred init bias is the root cause.
  - If softmax(policy) is balanced but visits skew, the dynamics-net
    Q-delta drives the bias.
  - If post-min-max hidden states are identical across the two obs,
    the rep/min-max collapse hypothesis is confirmed.
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("=== MuZero init-state diagnostic ===")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]
    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=64,
        HIDDEN=64,
        BINS=21,
        SIMS=8,  # one round — visit_count[root] should sum to 8 after Bug D fix
        K=3,
        N=3,
        BS=64,
        CAP=50000,
    ]
    comptime N_ENVS = 32

    # Pole tilting RIGHT and LEFT
    var obs_right = List[Float64]()
    obs_right.append(0.0)
    obs_right.append(0.0)
    obs_right.append(0.1)
    obs_right.append(0.0)
    var obs_left = List[Float64]()
    obs_left.append(0.0)
    obs_left.append(0.0)
    obs_left.append(-0.1)
    obs_left.append(0.0)

    print()
    print(
        "*** Run A: Kaiming init for ALL layers (the broken default) ***"
    )
    var agent_kaiming = GenericMuZeroAgent[Config, N_ENVS](
        gamma=0.99, temperature_decay_steps=20000
    )
    agent_kaiming.diagnose_init_state(
        ctx, obs_right, String("obs RIGHT (theta=+0.1) [Kaiming]")
    )
    agent_kaiming.diagnose_init_state(
        ctx, obs_left, String("obs LEFT  (theta=-0.1) [Kaiming]")
    )

    print()
    print(
        "*** Run B: zero-init pred heads (HIDDEN=64) — the proposed fix ***"
    )
    var agent_zero = GenericMuZeroAgent[Config, N_ENVS](
        gamma=0.99,
        temperature_decay_steps=20000,
        pred_head_input_dim=64,
    )
    agent_zero.diagnose_init_state(
        ctx, obs_right, String("obs RIGHT (theta=+0.1) [zero-head]")
    )
    agent_zero.diagnose_init_state(
        ctx, obs_left, String("obs LEFT  (theta=-0.1) [zero-head]")
    )

    print()
    print("=== Done ===")
