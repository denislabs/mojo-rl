"""MuZero CartPole convergence run (v2, GPU) — fully on-device search + train.

The GPU twin of `muzero_cartpole_v2_cpu`, driven through
`run_muzero_selfplay_gpu_device`: the MCTS search runs on the GPU
(`GenericGPUMCTS` over the resident h/g/f nets via the MuZero GPU adapters)
and the K-step BPTT unroll trains the same nets in place — no CPU mirror, no
per-step checkpoint sync (that was the old `run_muzero_selfplay_gpu` hybrid,
still available for reference). Search is SERIAL (BATCH_SIMS=1): the batched
GPU leaf path is value-biased vs the CPU search (driver docstring has the
story); serial is bit-near-identical to the converged CPU recipe. Same
convergence target (random ~22, "solving" ~195+, sustained greedy 500 ~52k).

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/muzero_cartpole_v2_gpu.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents.muzero.selfplay_gpu_device import (
    run_muzero_selfplay_gpu_device,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 128   # legacy MuZeroMLPConfig CartPole parity
    comptime BINS = 51
    comptime H = 128
    # 25 sims, SERIAL search (driver default BATCH_SIMS=1/VLOSS=0): the GPU
    # batched-leaf path is value-biased vs the converged CPU search — see
    # the driver docstring + test_mz_search_gpu_cpu_parity.
    comptime NUM_SIMS = 25
    comptime MAX_NODES = 128
    comptime CAP = 50000
    comptime B = 128        # legacy batch_size parity
    comptime K = 5
    comptime N = 10

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var ctx = DeviceContext()
    var env = CartPoleEnv[DType.float64]()
    var rep = Rep.make["gpu", INIT=Kaiming](ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx)
    var orep = Adam.make["gpu", M=Rep](rep, ctx)
    var odyn = Adam.make["gpu", M=Dyn](dyn, ctx)
    var opred = Adam.make["gpu", M=Pred](pred, ctx)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)
    orep.max_grad_norm = Scalar[DT](10.0)
    odyn.max_grad_norm = Scalar[DT](10.0)
    opred.max_grad_norm = Scalar[DT](10.0)

    print("MuZero CartPole convergence (v2, GPU — fully on-device search+train)")
    print("  LATENT", LATENT, "H", H, "BINS", BINS, "sims", NUM_SIMS,
          "K", K, "N", N, "B", B, "lr 3e-4 clip 10")

    # The proven CPU-lighthouse recipe (see muzero_cartpole_v2_cpu.mojo for the
    # rationale on each knob): ±20 h-space support (±10 saturates at raw ~117 <
    # CartPole V≈259), value_coef 1.0, temp 1.0→0.5→0.25, reanalyze every iter.
    var loss = run_muzero_selfplay_gpu_device[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
    ](
        ctx, env, rep, dyn, pred, orep, odyn, opred,
        iterations=60000,
        learning_starts=500,
        train_per_iter=1,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](1.0),
        temperature_decay_steps=60000,
        reanalyze_every=1,
        eval_every=2000,
        eval_episodes=5,
        seed=42,
        verbose=True,
    )

    print("final loss:", loss)
