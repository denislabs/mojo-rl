"""Smoke: MuZero batched GPU self-play on PIXEL Pong (the full Phase 1 path).

Phase 1 capstone of `docs/MUZERO_PIXEL_PONG_PLAN.md`: the first end-to-end run of
the pixel stack — `MuZeroCNNConfig` (Nature-CNN `MZRepNetCNN`) + a GPU-batched
`PongPixelEnv` (4×84×84 obs) + `run_muzero_gumbel_selfplay_gpu_batched` with the
**uint8** host replay ring (``OBS_STORE_DT = DType.uint8``). Tiny everything
(N_ENVS=2, NUM_SIMS=4, short episodes) so it merely proves the pipeline connects,
the CNN rep runs through the GPU search + BPTT unroll, and the loss is finite.
Convergence is Phase 3's job, not this smoke's.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_mz_selfplay_gpu_batched_pixel_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.muzero.config import MuZeroCNNConfig
from mojo_rl.deep_agents.muzero.selfplay_gpu_batched import (
    run_muzero_gumbel_selfplay_gpu_batched,
)
from mojo_rl.deep_agents.training import BatchedGpuDiscreteEnv
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
    comptime CAP = 600
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

    var rep = Rep.make["gpu", Kaiming](Optional(ctx))
    var dyn = Dyn.make["gpu", Kaiming](Optional(ctx))
    var pred = Pred.make["gpu", Kaiming](Optional(ctx))
    var orep = Adam(lr=Scalar[DT](1e-3))
    var odyn = Adam(lr=Scalar[DT](1e-3))
    var opred = Adam(lr=Scalar[DT](1e-3))
    orep.lr = Scalar[DT](1e-4)
    odyn.lr = Scalar[DT](1e-4)
    opred.lr = Scalar[DT](1e-4)

    var loss = run_muzero_gumbel_selfplay_gpu_batched[
        BatchedEnvT, Rep, Dyn, Pred,
        N_ENVS, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        OBS_STORE_DT = DType.uint8,
    ](
        ctx, env, rep, dyn, pred, orep, odyn, opred,
        iterations=30,
        learning_starts=20,
        max_ep_steps=10,
        v_min=Scalar[DT](-2.0),
        v_max=Scalar[DT](2.0),
        gamma=Scalar[DT](0.99),
        temperature_decay_steps=30,
        seed=7,
        verbose=True,
    )

    assert_true(loss == loss, "pixel batched MuZero loss NaN")
    assert_true(loss > 0.0 and loss < 1e6, "pixel batched MuZero loss not finite")
    print("pixel batched loss:", loss)
    print("MuZero batched GPU self-play PIXEL smoke: OK")
