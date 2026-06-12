"""MuZero CartPole convergence run (v2, CPU) — the single-player lighthouse.

Driven through the `MuZeroAgent` facade on the ``"cpu"`` target, whose `train`
wires `run_muzero_selfplay_cpu` with hyperparameters mirroring the legacy
`MuZeroMLPConfig` CartPole setup (LATENT=128/HIDDEN=128, BINS=51, K=5, N=10,
25 sims, lr 3e-4, gamma 0.997, value support [-20,20] h-space, value_coef 1.0,
visit-sampling temperature 1.0→0.5→0.25 over the run, batched MCTS 8 /
virtual-loss 3 to counter the spiky Dirichlet root prior). ``max_grad_norm=10.0``
is the legacy "clip 10".

This is the convergence/tuning harness for Phase B #28 — NOT a smoke. Random
CartPole returns ~22; "solving" is ~195+. Watch `avg_return(10)` climb.

Run (no GPU):
    pixi run mojo run -I . examples/cartpole/muzero_cartpole_v2_cpu.mojo
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.muzero import MuZeroMLPConfig, MuZeroAgent
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime Env = CartPoleEnv[DType.float64]
    # LATENT=128/HIDDEN=128, BINS=51 — legacy MuZeroMLPConfig CartPole parity.
    comptime Cfg = MuZeroMLPConfig[OBS=4, ACT=2, LATENT=128, HIDDEN=128, BINS=51]
    comptime NUM_SIMS = 25
    comptime Agent = MuZeroAgent[
        "cpu", Env,
        Cfg.Rep, Cfg.Dyn, Cfg.Pred,
        Cfg.OBS, Cfg.ACT, Cfg.LATENT, Cfg.BINS,
        NUM_SIMS=NUM_SIMS, MAX_NODES=128, CAP=50000, B=128, K=5, N=10,
    ]

    var env = Env()
    var agent = Agent(
        ctx=None,
        lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.997),
        # h-space support. ±10 saturates on CartPole: h⁻¹(10) ≈ 117 raw, but
        # V(s) at γ=0.997 reaches ~259 for a full 500-step episode
        # (h(259) ≈ 15.4) — every surviving state past ~145 steps encoded to
        # the same clipped target, capping greedy eval at ~200-330. ±20 covers
        # raw ±~424 with headroom (legacy example used ±100).
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](1.0),
        # legacy clips the global grad norm at 10 — without it the loss drifts
        # up late in the run (14.2 → 16.5 observed) as the value targets grow.
        max_grad_norm=Scalar[DT](10.0),
    )

    print("MuZero CartPole convergence (v2, CPU)")
    print("  LATENT", Cfg.LATENT, "H", Cfg.HIDDEN, "BINS", Cfg.BINS,
          "sims", NUM_SIMS, "K", 5, "N", 10, "B", 128, "lr 3e-4 clip 10")

    var loss = agent.train(
        env,
        # Round-4 (30k) was still climbing monotonically at cutoff (training
        # return 170→242 over the last 8k); legacy's sustained-500 point was
        # ~32k env steps into a 50k run. Give it the same room.
        iterations=60000,
        learning_starts=500,
        train_per_iter=1,
        # Refresh one stored (policy, root value) per iter with a fresh
        # search — n-step targets bootstrap from stored values, which go
        # stale as the net improves (legacy ran use_reanalyze=True). At
        # every=2 most of the buffer stayed stale; every=1 ≈ 2× search cost.
        reanalyze_every=1,
        temperature_decay_steps=60000,
        eval_every=2000,
        eval_episodes=5,
        seed=42,
        verbose=True,
    )

    print("final loss:", loss)
