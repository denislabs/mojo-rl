"""CartPole with Gumbel-MuZero — Fully GPU (env + MCTS + training).

Companion to ``cartpole_muzero_gpu.mojo`` (PUCT). Same agent, same
config, only the MCTS policy mode differs:

  * PUCT  → ``MuZeroPUCTPolicy`` (default in ``MuZeroMLPConfig``)
  * Gumbel → ``GumbelMuZeroPolicy[MAX_K]`` (this script)

Gumbel-MuZero (Danihelka et al. 2022) replaces PUCT + Dirichlet root
noise with Gumbel-Top-k root sampling + Sequential Halving across
``log2(MAX_K)`` phases + deterministic interior σ(Q) − N/(1+ΣN)
selection. It has stronger theoretical guarantees at low simulation
budgets — but CartPole has only 2 actions, so ``MAX_K=2`` means a
single halving phase with nothing to halve, and Gumbel collapses to
a sampling rule with no discrimination advantage over PUCT.

Empirically (user memory) this configuration **does not converge** on
CartPole. This script exists so that regressions in the PUCT path
(which DOES converge — see 2026-05-23 fix in
``docs/MUZERO_GPU_AUDIT_2026-05-22.md``) can be diagnosed without
having to rewire the Gumbel orchestrator each time, and so the
non-convergence symptom is reproducible on demand.

Note: the 2026-05-23 root-hidden-scatter fix lived in the PUCT
orchestrator (``mcts_gpu_orchestrator.mojo``). The Gumbel
orchestrator (``mcts_gumbel_orchestrator.mojo``) already had its own
``gz_scatter_root_hidden_kernel`` and was confirmed **not** affected
by that bug — so failures observed here are a separate issue.

Usage (Apple Silicon):
    pixi run -e apple mojo run -I . examples/cartpole/cartpole_muzero_gumbel_gpu.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.deep_agents.muzero.policy_mode import GumbelMuZeroPolicy
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("MuZero on CartPole (Fully GPU, Gumbel policy)")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]

    # MAX_K must be a power of two ≤ ACT (= 2 for CartPole), so MAX_K=2.
    # That gives a single Sequential-Halving phase (log2(2) = 1) with
    # both actions kept — i.e. effectively no halving discrimination,
    # but the Gumbel orchestrator + improved-policy training target
    # still apply.
    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=128,
        HIDDEN=128,
        BINS=51,
        SIMS=25,
        POLICY=GumbelMuZeroPolicy[2],
    ]
    print("Config.PolicyMode.IS_GUMBEL =", Config.PolicyMode.IS_GUMBEL)
    print("Config.PolicyMode.MAX_K     =", Config.PolicyMode.MAX_K)

    var agent = GenericMuZeroAgent[Config, 8](
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
