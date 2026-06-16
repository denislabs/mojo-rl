"""EZv2 GPU sampled-Gumbel continuous self-play driver smoke (Apple/CUDA).

Runs `run_ezv2_sampled_selfplay_gpu` for a few hundred iterations on Pendulum with
tiny nets: GPU sampled-Gumbel search (via the MZ GPU adapters + continuous
prediction head `MZContPredGPU`) drives data collection and
`ezv2_unroll_train_step_continuous_gpu` trains the resident GPU nets. Not a
convergence test — the budget is too small to expect learning. Asserts the loop
completes and the returned last loss is finite. The GPU↔CPU correctness of the
continuous unroll itself is covered by
`test_ezv2_unroll_continuous_gpu_parity.mojo`.

Run (GPU env required):
    pixi run -e apple mojo run -I . \\
        tests/deep_agents/test_ezv2_sampled_selfplay_gpu_smoke.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents.efficient_zero_v2.nets_continuous import EZContPredNet
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_gpu_continuous import (
    run_ezv2_sampled_selfplay_gpu,
)
from mojo_rl.envs.pendulum import PendulumEnv


def main() raises:
    print("test_ezv2_sampled_selfplay_gpu_smoke ...")
    comptime OBS = 3
    comptime ACT_DIM = 1
    comptime LATENT = 16
    comptime BINS = 21
    comptime H = 32
    comptime PROJ = 32
    comptime PROJ_HID = 32
    comptime BOTTLENECK = 16
    comptime NUM_SIMS = 8
    comptime MAX_NODES = 32
    comptime K_ROOT = 4
    comptime K_NON_ROOT = 2
    comptime CAP = 5000
    comptime B = 16
    comptime K = 3
    comptime N = 5

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT_DIM, BINS, H]
    comptime Pred = EZContPredNet[LATENT, ACT_DIM, BINS, H]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
    comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]

    var ctx = DeviceContext()
    var env = PendulumEnv[DType.float32]()
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

    var loss = run_ezv2_sampled_selfplay_gpu[
        PendulumEnv[DType.float32], Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT_DIM, LATENT, BINS, NUM_SIMS, MAX_NODES, K_ROOT, K_NON_ROOT,
        CAP, B, K, N,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=400,
        learning_starts=100,
        train_per_iter=1,
        gamma=Scalar[DT](0.99),
        v_min=Scalar[DT](-50.0),
        v_max=Scalar[DT](2.0),
        max_action=Scalar[DT](2.0),
        min_std=Scalar[DT](0.5),
        seed=42,
        max_ep_steps=200,
        value_coef=Scalar[DT](0.25),
        consistency_coef=Scalar[DT](2.0),
        ent_scale=Scalar[DT](0.05),
        # exercise the target-net sync + reanalyze paths within the budget
        target_sync_interval=100,
        reanalyze_interval=50,
        reanalyze_warmup=120,
        reanalyze_batch=2,
        verbose=True,
    )

    print("  final loss:", loss)
    assert_true(loss == loss and loss < 1e30 and loss > -1e30,
                "driver loss non-finite")
    print("PASS: EZv2 GPU sampled-Gumbel continuous self-play smoke")
