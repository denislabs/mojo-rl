"""MuZero CartPole convergence run (v2, CPU) — the single-player lighthouse.

Drives `run_muzero_selfplay_cpu` with hyperparameters mirroring the legacy
`MuZeroMLPConfig` CartPole setup (LATENT=128/HIDDEN=128, BINS=51, K=5, N=10,
25 sims, lr 3e-4, gamma 0.997, value support [-20,20] h-space, value_coef 1.0,
visit-sampling temperature 1.0→0.5→0.25 over the run, batched MCTS 8 /
virtual-loss 3 to counter the spiky Dirichlet root prior).

This is the convergence/tuning harness for Phase B #28 — NOT a smoke. Random
CartPole returns ~22; "solving" is ~195+. Watch `avg_return(10)` climb.

Run (no GPU):
    pixi run mojo run -I . examples/cartpole/muzero_cartpole_v2_cpu.mojo
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents2.muzero.selfplay_cpu import run_muzero_selfplay_cpu
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 128   # legacy MuZeroMLPConfig CartPole parity
    comptime BINS = 51
    comptime H = 128
    comptime NUM_SIMS = 25
    comptime MAX_NODES = 128
    comptime CAP = 50000
    comptime B = 128        # legacy batch_size parity
    comptime K = 5
    comptime N = 10

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var env = CartPoleEnv[DType.float64]()
    var rep = Rep.make["cpu", INIT=Kaiming]()
    var dyn = Dyn.make["cpu", INIT=Kaiming]()
    var pred = Pred.make["cpu", INIT=Kaiming]()
    var orep = Adam.make["cpu", M=Rep](rep)
    var odyn = Adam.make["cpu", M=Dyn](dyn)
    var opred = Adam.make["cpu", M=Pred](pred)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)
    # legacy clips the global grad norm at 10 — without it the loss drifts
    # up late in the run (14.2 → 16.5 observed) as the value targets grow.
    orep.max_grad_norm = Scalar[DT](10.0)
    odyn.max_grad_norm = Scalar[DT](10.0)
    opred.max_grad_norm = Scalar[DT](10.0)

    print("MuZero CartPole convergence (v2, CPU)")
    print("  LATENT", LATENT, "H", H, "BINS", BINS, "sims", NUM_SIMS,
          "K", K, "N", N, "B", B, "lr 3e-4 clip 10")

    var loss = run_muzero_selfplay_cpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
    ](
        env, rep, dyn, pred, orep, odyn, opred,
        iterations=30000,
        learning_starts=500,
        train_per_iter=1,
        gamma=Scalar[DT](0.997),
        # h-space support. ±10 saturates on CartPole: h⁻¹(10) ≈ 117 raw, but
        # V(s) at γ=0.997 reaches ~259 for a full 500-step episode
        # (h(259) ≈ 15.4) — every surviving state past ~145 steps encoded to
        # the same clipped target, capping greedy eval at ~200-330. ±20 covers
        # raw ±~424 with headroom (legacy example used ±100).
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](1.0),
        temperature_decay_steps=30000,
        # Refresh one stored (policy, root value) per 2 iters with a fresh
        # search — n-step targets bootstrap from stored values, which go
        # stale as the net improves (legacy ran use_reanalyze=True).
        reanalyze_every=2,
        eval_every=2000,
        eval_episodes=5,
        seed=42,
        verbose=True,
    )

    print("final loss:", loss)
