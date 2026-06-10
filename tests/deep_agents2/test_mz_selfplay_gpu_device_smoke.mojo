"""Smoke: MuZero fully-on-device GPU self-play drivers (search + train on GPU).

Drives BOTH `run_muzero_selfplay_gpu_device` (vanilla PUCT + Dirichlet, plus a
NoNoise eval planner) and `run_muzero_gumbel_selfplay_gpu` (Gumbel MuZero) for
a few hundred iterations each on CartPole at tiny sizes — asserting the full
env → GPU search → replay → GPU BPTT pipeline connects, trains, and returns a
finite loss. Exercises the temperature schedule, truncation flag, reanalyze,
and the greedy-eval path. Convergence is the example's job, not this smoke's.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents2/test_mz_selfplay_gpu_device_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents2.muzero.selfplay_gpu_device import (
    run_muzero_selfplay_gpu_device,
    run_muzero_gumbel_selfplay_gpu,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 16
    comptime BINS = 51
    comptime H = 32
    comptime NUM_SIMS = 16
    comptime MAX_NODES = 64
    comptime MAX_K = 2          # Gumbel root candidates (ACT=2)
    comptime CAP = 4000
    comptime B = 16
    comptime K = 3
    comptime N = 5

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var ctx = DeviceContext()

    # ── A: vanilla device-search MuZero ──
    var env = CartPoleEnv[DType.float64]()
    var rep = Rep.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx=ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx=ctx)
    var orep = Adam.make["gpu", M=Rep](rep, ctx)
    var odyn = Adam.make["gpu", M=Dyn](dyn, ctx)
    var opred = Adam.make["gpu", M=Pred](pred, ctx)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)

    var loss_a = run_muzero_selfplay_gpu_device[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
    ](
        ctx, env, rep, dyn, pred, orep, odyn, opred,
        iterations=400,
        learning_starts=100,
        temperature_decay_steps=400,
        reanalyze_every=4,
        eval_every=200,
        eval_episodes=1,
        seed=7,
        verbose=True,
    )
    assert_true(loss_a == loss_a, "device-search loss NaN")
    assert_true(loss_a > 0.0 and loss_a < 1e6, "device-search loss not finite")
    print("vanilla device-search loss:", loss_a)

    # ── B: Gumbel MuZero ──
    var env2 = CartPoleEnv[DType.float64]()
    var rep2 = Rep.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn2 = Dyn.make["gpu", INIT=Kaiming](ctx=ctx)
    var pred2 = Pred.make["gpu", INIT=Kaiming](ctx=ctx)
    var orep2 = Adam.make["gpu", M=Rep](rep2, ctx)
    var odyn2 = Adam.make["gpu", M=Dyn](dyn2, ctx)
    var opred2 = Adam.make["gpu", M=Pred](pred2, ctx)
    orep2.lr = Scalar[DT](3e-4)
    odyn2.lr = Scalar[DT](3e-4)
    opred2.lr = Scalar[DT](3e-4)

    var loss_b = run_muzero_gumbel_selfplay_gpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
    ](
        ctx, env2, rep2, dyn2, pred2, orep2, odyn2, opred2,
        iterations=400,
        learning_starts=100,
        temperature_decay_steps=400,
        reanalyze_every=4,
        eval_every=200,
        eval_episodes=1,
        seed=7,
        verbose=True,
    )
    assert_true(loss_b == loss_b, "gumbel loss NaN")
    assert_true(loss_b > 0.0 and loss_b < 1e6, "gumbel loss not finite")
    print("gumbel loss:", loss_b)

    print("MuZero fully-on-device GPU self-play smoke: OK")
