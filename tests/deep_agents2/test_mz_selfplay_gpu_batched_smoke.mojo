"""Smoke: MuZero BATCHED GPU self-play driver (N envs + batched search).

Phase 1 of `docs/MUZERO_PIXEL_PONG_PLAN.md`: drives
`run_muzero_gumbel_selfplay_gpu_batched` over ``N_ENVS=2`` parallel clean-obs
Pong envs (GPU-batched, OBS=6 so the MLP rep keeps the smoke fast — the pixel
CNN rep is covered by `test_mz_rep_cnn_smoke.mojo`). Asserts the full batched
pipeline connects and trains to a finite loss:

  batched env reset/step/selective_reset → one Gumbel search over [N_ENVS, OBS]
  → per-env episode accumulation → host MCTSSequenceReplay store_episode
  (truncation path) → GPU BPTT unroll. Also exercises batched reanalyze and the
  fixed-horizon batched greedy eval. ``max_ep_steps=30`` forces quick truncation
  so episodes store and training starts inside the smoke's iteration budget.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents2/test_mz_selfplay_gpu_batched_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.muzero.config import MuZeroMLPConfig
from mojo_rl.deep_agents2.muzero.selfplay_gpu_batched import (
    run_muzero_gumbel_selfplay_gpu_batched,
)
from mojo_rl.deep_agents2.training import BatchedGpuDiscreteEnv
from mojo_rl.envs.arcade_games.pong import PongEnv


def main() raises:
    comptime OBS = 6        # clean Pong obs
    comptime ACT = 3        # NOOP / UP / DOWN
    comptime N_ENVS = 2
    comptime LATENT = 16
    comptime HIDDEN = 32
    comptime BINS = 51
    comptime NUM_SIMS = 8
    comptime MAX_NODES = 32
    comptime MAX_K = 2
    comptime CAP = 4000
    comptime B = 16
    comptime K = 3
    comptime N = 5

    comptime Cfg = MuZeroMLPConfig[OBS, ACT, LATENT, HIDDEN, BINS]
    comptime Rep = Cfg.Rep
    comptime Dyn = Cfg.Dyn
    comptime Pred = Cfg.Pred
    comptime BatchedEnvT = BatchedGpuDiscreteEnv[PongEnv[DT], N_ENVS, OBS, 1]

    var ctx = DeviceContext()

    var env = BatchedEnvT(ctx)
    var eval_env = BatchedEnvT(ctx)

    var rep = Rep.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx=ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx=ctx)
    var orep = Adam.make["gpu", M=Rep](rep, ctx)
    var odyn = Adam.make["gpu", M=Dyn](dyn, ctx)
    var opred = Adam.make["gpu", M=Pred](pred, ctx)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)

    var loss = run_muzero_gumbel_selfplay_gpu_batched[
        BatchedEnvT, Rep, Dyn, Pred,
        N_ENVS, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
    ](
        ctx, env, rep, dyn, pred, orep, odyn, opred,
        iterations=300,
        learning_starts=60,
        max_ep_steps=30,
        temperature_decay_steps=300,
        reanalyze_every=50,
        reanalyze_batch=B,   # > N_ENVS → exercises the multi-chunk reanalyze loop
        eval_every=150,
        eval_horizon=30,
        eval_env=UnsafePointer(to=eval_env),
        seed=7,
        verbose=True,
    )

    assert_true(loss == loss, "batched MuZero loss NaN")
    assert_true(loss > 0.0 and loss < 1e6, "batched MuZero loss not finite")
    print("batched loss:", loss)
    print("MuZero batched GPU self-play smoke: OK")
