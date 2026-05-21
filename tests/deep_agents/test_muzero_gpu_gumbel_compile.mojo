"""Smoke test: MuZero CartPole GPU agent compiles with GumbelMuZeroPolicy.

Validates Phase 5 wiring — the agent dispatches
``Self.Config.PolicyMode.IS_GUMBEL`` into ``GumbelGPUMCTS.search_gpu``
instead of ``GenericGPUMCTS.search_gpu``. We do not exercise the
training loop here (kernel-launch parity vs the PUCT path is left to a
separate run); this just confirms the comptime dispatch + orchestrator
template instantiations resolve.

Usage:
    pixi run -e apple mojo run -I . tests/deep_agents/test_muzero_gpu_gumbel_compile.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.deep_agents.muzero.policy_mode import GumbelMuZeroPolicy
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("=== MuZero CartPole GPU — GumbelMuZeroPolicy compile smoke ===")
    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]

    # CartPole has ACT=2 → MAX_K=2 (1 halving phase of 2 candidates,
    # SIMS=8 → 4 sims per slot). Just enough to trigger Gumbel-Top-k +
    # Sequential Halving on the GPU code paths.
    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=128,
        HIDDEN=128,
        BINS=51,
        SIMS=8,
        POLICY=GumbelMuZeroPolicy[2],  # MAX_K=2 for CartPole's 2 actions
    ]
    print("Config.PolicyMode.IS_GUMBEL =", Config.PolicyMode.IS_GUMBEL)
    print("Config.PolicyMode.MAX_K     =", Config.PolicyMode.MAX_K)

    # Build the agent + call ``train_gpu`` so the Gumbel branch of
    # the per-step MCTS block is actually instantiated by the
    # compiler. We run just a few env steps; the goal is compile +
    # one round-trip of the kernels, not training convergence.
    var agent = GenericMuZeroAgent[Config, 4]()
    print("Gumbel-mode agent constructed; entering train_gpu...")
    _ = agent.train_gpu[CartPoleGPU](
        ctx,
        num_steps=128,
        warmup_steps=32,
        print_every=64,
    )
    print("Train steps completed:", agent.train_step_count)
    print("=== Gumbel-MuZero smoke OK ===")
