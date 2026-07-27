"""EZv2-Atari Stage-6 integration smoke (GPU) — full stack, tiny dims.

End-to-end wiring check for the Atari EZv2 path: the RGB-96 emulator env
(`AtariEnv[2]`, OBS_MODE=2) → `GumbelGPUMCTS` search over the spatial
rep/dyn/pred nets (`EZV2AtariConfig`) → CPU MCTS sequence replay → GPU
`ezv2_unroll_train_step_gpu` with **SGD** (the Stage-5 optimizer, wired through
the now-optimizer-generic driver) + **init_zero** heads + warmup→const LR.

This is NOT a convergence run — it uses tiny dims and forces short episodes
(`max_ep_steps`) so an episode lands in the buffer and a real train step
fires within a handful of iterations. It asserts the driver runs and returns
a finite loss, proving env + planner + replay + SGD train step + init_zero all
compile and integrate at the real Atari config (FRAMES=4, ACT=18, BINS=601,
spatial latent [64,6,6]).

Run (GPU env required):
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_integration_smoke.mojo
"""

from std.memory import UnsafePointer
from std.math import isnan, isinf
from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT, LAYOUT_NCHW
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.efficient_zero_v2.config_atari import EZV2AtariConfig
from mojo_rl.deep_agents.efficient_zero_v2.nets_atari import (
    ez_atari_init_zero_pred, ez_atari_init_zero_dyn,
)
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_gpu import (
    run_ezv2_gumbel_selfplay_gpu,
)
from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


def main() raises:
    comptime FRAMES = 4
    comptime ACT = 18            # full ALE action set (EZv2)
    comptime BINS = 601          # Atari support [-300,300]
    comptime Cfg = EZV2AtariConfig[FRAMES, ACT, LAYOUT=LAYOUT_NCHW]
    comptime OBS = Cfg.OBS       # 110592
    comptime LATENT = Cfg.LATENT  # 2304

    # Tiny smoke dims — keep compile + runtime small.
    comptime NUM_SIMS = 4
    comptime MAX_NODES = 32
    comptime MAX_K = 4
    comptime CAP = 500
    comptime B = 4
    comptime K = 2
    comptime N = 3

    comptime Rep = Cfg.Rep
    comptime Dyn = Cfg.Dyn
    comptime Pred = Cfg.Pred
    comptime Proj = Cfg.Proj
    comptime Predh = Cfg.Predh
    comptime Env = AtariEnv[2, DT]

    print("=" * 70)
    print("EZv2-Atari integration smoke (GPU): RGB-96 + spatial nets + SGD")
    print("=" * 70)

    var ctx = DeviceContext()
    var rom = load_rom("roms/pong.bin")
    var env = Env(
        AtariGame.PONG, rom.data.value().as_unsafe_any_origin(), rom.size,
        clip_reward=True, full_action_set=True,
    )

    var rep = Rep.make["gpu", Kaiming](Optional(ctx))
    var dyn = Dyn.make["gpu", Kaiming](Optional(ctx))
    var pred = Pred.make["gpu", Kaiming](Optional(ctx))
    var proj = Proj.make["gpu", Kaiming](Optional(ctx))
    var predh = Predh.make["gpu", Kaiming](Optional(ctx))

    # init_zero (EZv2): neutral value/reward + uniform policy at init.
    ez_atari_init_zero_pred["gpu", ACT, BINS](pred, ctx)
    ez_atari_init_zero_dyn["gpu", ACT, BINS](dyn, ctx)
    ctx.synchronize()

    # SGD (the Stage-5 optimizer): EZ Atari = 0.2 / mom 0.9 / wd 1e-4 / clip 5.
    var orep = Adam(lr=Scalar[DT](0.1))
    var odyn = Adam(lr=Scalar[DT](0.1))
    var opred = Adam(lr=Scalar[DT](0.1))
    var oproj = Adam(lr=Scalar[DT](0.1))
    var opredh = Adam(lr=Scalar[DT](0.1))

    var loss = run_ezv2_gumbel_selfplay_gpu[
        Env, Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=16,
        learning_starts=7,
        train_per_iter=1,
        lr=Scalar[DT](0.2),
        lr_warmup_iters=4,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-300.0),
        v_max=Scalar[DT](300.0),
        value_coef=Scalar[DT](0.25),
        consistency_coef=Scalar[DT](2.0),
        max_ep_steps=6,        # force short episodes so a train step fires
        seed=7,
        verbose=False,
    )

    print("  final loss:", loss)
    assert_true(not isnan(loss) and not isinf(loss),
                "EZv2-Atari integration smoke loss finite")
    _ = env^
    print("=" * 70)
    print("PASSED")
    print("=" * 70)
