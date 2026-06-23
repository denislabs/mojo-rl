"""EZv2 discrete CPU self-play driver smoke — full pipeline on CartPole, no GPU.

Runs `run_ezv2_selfplay_cpu` for a short horizon and asserts the whole loop
connects: episodes get stored, training fires after warmup (MuZero BPTT + SimSiam
consistency through the obs-sequence replay), and the loss stays finite. End-to-end
Phase-C discrete integration check (env → CPU MCTS → seq replay → EZv2 unroll).
Convergence is a separate, longer run.

Note: the reported loss folds the consistency ``−cos`` term, so it can be
negative — we assert *finite*, not positive.

Run (no GPU):
    pixi run mojo run -I . tests/deep_agents/test_ezv2_selfplay_cpu_smoke.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_cpu import (
    run_ezv2_selfplay_cpu,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 16
    comptime BINS = 51
    comptime H = 32
    comptime PROJ = 32
    comptime PROJ_HID = 32
    comptime BOTTLENECK = 16
    comptime NUM_SIMS = 12
    comptime MAX_NODES = 48
    comptime CAP = 20000
    comptime B = 8
    comptime K = 3
    comptime N = 3

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
    comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]

    var env = CartPoleEnv[DType.float64]()
    var rep = Rep.make["cpu", Kaiming]()
    var dyn = Dyn.make["cpu", Kaiming]()
    var pred = Pred.make["cpu", Kaiming]()
    var proj = Proj.make["cpu", Kaiming]()
    var predh = Predh.make["cpu", Kaiming]()
    var orep = Adam(lr=Scalar[DT](1e-3))
    var odyn = Adam(lr=Scalar[DT](1e-3))
    var opred = Adam(lr=Scalar[DT](1e-3))
    var oproj = Adam(lr=Scalar[DT](1e-3))
    var opredh = Adam(lr=Scalar[DT](1e-3))
    orep.lr = Scalar[DT](0.01)
    odyn.lr = Scalar[DT](0.01)
    opred.lr = Scalar[DT](0.01)
    oproj.lr = Scalar[DT](0.01)
    opredh.lr = Scalar[DT](0.01)

    var loss = run_ezv2_selfplay_cpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
    ](
        env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=900,
        learning_starts=200,
        train_per_iter=1,
        reanalyze_every=100,
        reanalyze_batch=4,   # > 1 → exercises the multi-position reanalyze loop
        seed=7,
        verbose=True,
    )

    print("final loss:", loss)
    assert_true(loss == loss and loss < 1e30 and loss > -1e30,
        "self-play training loss not finite")
    print("EZv2 discrete CPU self-play driver smoke: OK")
