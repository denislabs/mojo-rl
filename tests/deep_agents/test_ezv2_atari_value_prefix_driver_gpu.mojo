"""EZv2-Atari value-prefix DRIVER smoke (GPU) — run_ezv2_gumbel_selfplay_gpu_vp.

End-to-end check of the dedicated value-prefix driver (`selfplay_gpu_vp.mojo`):
the RGB-96 emulator env → `GumbelGPUMCTS` search through the fused
`EZDynVPNetAtari` → MCTS sequence replay → `ezv2_unroll_train_step_gpu_vp` on
`dyn.dynz`/`dyn.rew` (6 optimizers) with cumulative value-prefix targets +
init_zero heads + warmup LR. Tiny dims + forced short episodes so a real train
step fires in a few iters. Asserts the driver runs and returns a finite loss —
the same shape of check as `test_ezv2_atari_integration_smoke` for the non-VP
driver, proving the value-prefix path integrates at the real Atari config.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_value_prefix_driver_gpu.mojo
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
    EZDynVPNetAtari, ez_atari_init_zero_pred, ez_atari_init_zero_reward,
    EZ_LSTM_HORIZON,
)
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_gpu_vp import (
    run_ezv2_gumbel_selfplay_gpu_vp,
)
from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


def main() raises:
    comptime FRAMES = 4
    comptime ACT = 18
    comptime BINS = 601
    comptime Cfg = EZV2AtariConfig[FRAMES, ACT, LAYOUT=LAYOUT_NCHW]
    comptime OBS = Cfg.OBS
    comptime LATENT = Cfg.LATENT
    comptime HORIZON = EZ_LSTM_HORIZON

    comptime NUM_SIMS = 4
    comptime MAX_NODES = 32
    comptime MAX_K = 4
    comptime CAP = 500
    comptime B = 4
    comptime K = 2
    comptime N = 3
    comptime Env = AtariEnv[2, DT]
    comptime VPDyn = EZDynVPNetAtari[ACT, BINS]

    print("=" * 70)
    print("EZv2-Atari value-prefix DRIVER smoke (GPU)")
    print("=" * 70)

    var ctx = DeviceContext()
    var rom = load_rom("roms/pong.bin")
    var env = Env(
        AtariGame.PONG, rom.data.value().as_unsafe_any_origin(), rom.size,
        clip_reward=True, full_action_set=True,
    )

    var rep = Cfg.Rep.make["gpu", Kaiming](Optional(ctx))
    var dyn = VPDyn.make["gpu", Kaiming](Optional(ctx))
    var pred = Cfg.Pred.make["gpu", Kaiming](Optional(ctx))
    var proj = Cfg.Proj.make["gpu", Kaiming](Optional(ctx))
    var predh = Cfg.Predh.make["gpu", Kaiming](Optional(ctx))
    ez_atari_init_zero_pred["gpu", ACT, BINS](pred, ctx)
    ez_atari_init_zero_reward["gpu", BINS](dyn.rew, ctx)
    ctx.synchronize()

    var orep = Adam(lr=Scalar[DT](0.1))
    var odyn = Adam(lr=Scalar[DT](0.1))
    var orew = Adam(lr=Scalar[DT](0.1))
    var opred = Adam(lr=Scalar[DT](0.1))
    var oproj = Adam(lr=Scalar[DT](0.1))
    var opredh = Adam(lr=Scalar[DT](0.1))

    var loss = run_ezv2_gumbel_selfplay_gpu_vp[
        Env, Cfg.Rep, Cfg.Pred, Cfg.Proj, Cfg.Predh,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        HORIZON=HORIZON,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, orew, opred, oproj, opredh,
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
        max_ep_steps=6,
        seed=7,
        verbose=False,
    )

    print("  final loss:", loss)
    assert_true(not isnan(loss) and not isinf(loss),
                "value-prefix driver loss finite")
    _ = env^
    print("=" * 70)
    print("PASSED — value-prefix driver runs end-to-end")
    print("=" * 70)
