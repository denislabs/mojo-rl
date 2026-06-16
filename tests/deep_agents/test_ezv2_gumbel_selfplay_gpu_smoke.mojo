"""EZv2 GPU Gumbel self-play driver smoke (Apple/CUDA).

Runs `run_ezv2_gumbel_selfplay_gpu` for a few hundred iterations on CartPole with
tiny nets: GPU Gumbel search (via the MZ GPU adapters) drives data collection and
`ezv2_unroll_train_step_gpu` trains the resident GPU nets. Not a convergence test
— the budget is too small to expect learning. Asserts the loop completes and the
returned last loss is finite. The GPU↔CPU correctness of the unroll itself is
covered by `test_ezv2_unroll_gpu_parity.mojo`.

Run (GPU env required):
    pixi run -e apple mojo run -I . \\
        tests/deep_agents/test_ezv2_gumbel_selfplay_gpu_smoke.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_gpu import (
    run_ezv2_gumbel_selfplay_gpu,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("test_ezv2_gumbel_selfplay_gpu_smoke ...")
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 16
    comptime BINS = 21
    comptime H = 32
    comptime PROJ = 32
    comptime PROJ_HID = 32
    comptime BOTTLENECK = 16
    comptime NUM_SIMS = 8
    comptime MAX_NODES = 32
    comptime MAX_K = 2
    comptime CAP = 5000
    comptime B = 16
    comptime K = 3
    comptime N = 5

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
    comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]

    var ctx = DeviceContext()
    var env = CartPoleEnv[DType.float32]()
    var rep = Rep.make["gpu", INIT=Kaiming](ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx)
    var proj = Proj.make["gpu", INIT=Kaiming](ctx)
    var predh = Predh.make["gpu", INIT=Kaiming](ctx)
    var orep = Adam.make["gpu", M=Rep](rep, ctx)
    var odyn = Adam.make["gpu", M=Dyn](dyn, ctx)
    var opred = Adam.make["gpu", M=Pred](pred, ctx)
    var oproj = Adam.make["gpu", M=Proj](proj, ctx)
    var opredh = Adam.make["gpu", M=Predh](predh, ctx)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)
    oproj.lr = Scalar[DT](3e-4)
    opredh.lr = Scalar[DT](3e-4)

    var loss = run_ezv2_gumbel_selfplay_gpu[
        CartPoleEnv[DType.float32], Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=400,
        learning_starts=100,
        train_per_iter=1,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-10.0),
        v_max=Scalar[DT](10.0),
        value_coef=Scalar[DT](0.25),
        consistency_coef=Scalar[DT](2.0),
        reanalyze_every=50,
        reanalyze_batch=4,   # > 1 → exercises the multi-position reanalyze loop
        seed=42,
        verbose=True,
    )

    print("  final loss:", loss)
    assert_true(loss == loss and loss < 1e30 and loss > -1e30,
                "driver loss non-finite")
    print("PASS: EZv2 GPU Gumbel self-play smoke")
