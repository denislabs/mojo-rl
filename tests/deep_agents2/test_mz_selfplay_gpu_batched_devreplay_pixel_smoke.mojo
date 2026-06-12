"""Smoke: MuZero batched GPU self-play on PIXEL Pong with the DEVICE replay.

Phase 2 capstone of `docs/MUZERO_PIXEL_PONG_PLAN.md`: the device-obs path —
`run_muzero_gumbel_selfplay_gpu_batched_devreplay` keeps the obs ring on the GPU
(`GPUMCTSSequenceReplay`, uint8), so no full `[N_ENVS, OBS]` pixel observation
ever crosses the bus on the collection path; the training obs slab is gathered
device→device into the train step's buffer (`obs_on_device=True`). Tiny config
(N_ENVS=2, NUM_SIMS=4, short episodes; CAP ≥ N_ENVS·max_ep_steps) — proves the
device-replay pipeline connects, the CNN rep trains through it, and the loss is
finite. Convergence is Phase 3's job.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents2/test_mz_selfplay_gpu_batched_devreplay_pixel_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.muzero.config import MuZeroCNNConfig
from mojo_rl.deep_agents2.muzero.selfplay_gpu_batched import (
    run_muzero_gumbel_selfplay_gpu_batched_devreplay,
)
from mojo_rl.deep_agents2.training import BatchedGpuDiscreteEnv
from mojo_rl.envs.arcade_games.pong import PongPixelEnv


def main() raises:
    comptime FRAMES = 4
    comptime ACT = 3
    comptime N_ENVS = 2
    comptime LATENT = 64
    comptime HIDDEN = 128
    comptime BINS = 51
    comptime NUM_SIMS = 4
    comptime MAX_NODES = 16
    comptime MAX_K = 3
    comptime MAX_EP = 10
    comptime CAP = 600         # ≥ N_ENVS·MAX_EP and a multiple of N_ENVS
    comptime B = 8
    comptime K = 3
    comptime N = 3

    comptime Cfg = MuZeroCNNConfig[FRAMES, ACT, LATENT, HIDDEN, BINS]
    comptime OBS = Cfg.OBS   # 4*84*84 = 28224
    comptime Rep = Cfg.Rep
    comptime Dyn = Cfg.Dyn
    comptime Pred = Cfg.Pred
    comptime BatchedEnvT = BatchedGpuDiscreteEnv[
        PongPixelEnv[DT], N_ENVS, OBS, 1
    ]

    var ctx = DeviceContext()
    var env = BatchedEnvT(ctx)

    var rep = Rep.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx=ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx=ctx)
    var orep = Adam.make["gpu", M=Rep](rep, ctx)
    var odyn = Adam.make["gpu", M=Dyn](dyn, ctx)
    var opred = Adam.make["gpu", M=Pred](pred, ctx)
    orep.lr = Scalar[DT](1e-4)
    odyn.lr = Scalar[DT](1e-4)
    opred.lr = Scalar[DT](1e-4)

    var loss = run_muzero_gumbel_selfplay_gpu_batched_devreplay[
        BatchedEnvT, Rep, Dyn, Pred,
        N_ENVS, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        OBS_STORE_DT = DType.uint8,
    ](
        ctx, env, rep, dyn, pred, orep, odyn, opred,
        iterations=30,
        learning_starts=20,
        max_ep_steps=MAX_EP,
        v_min=Scalar[DT](-2.0),
        v_max=Scalar[DT](2.0),
        gamma=Scalar[DT](0.99),
        temperature_decay_steps=30,
        reanalyze_every=10,
        seed=7,
        verbose=True,
    )

    assert_true(loss == loss, "devreplay pixel MuZero loss NaN")
    assert_true(loss > 0.0 and loss < 1e6, "devreplay pixel loss not finite")
    print("devreplay pixel batched loss:", loss)
    print("MuZero batched GPU self-play DEVICE-REPLAY pixel smoke: OK")
