"""EZv2-Atari BATCHED value-prefix driver smoke (GPU, tiny dims).

End-to-end check for `run_ezv2_gumbel_selfplay_gpu_batched_vp` (the batched
value-prefix driver): N_ENVS RGB-96 `AtariEnv[2]` CPU envs → batched Gumbel
search through the fused `EZDynVPNetAtari` → **prioritized** device-obs-ring
replay → **IS-weighted** `ezv2_unroll_train_step_gpu_vp` on `dyn.dynz`/`dyn.rew`
(cumulative value-prefix targets) + init_zero + warmup LR + **wide reanalyze** +
UTD 1:1. Tiny dims + short forced episodes so prioritized, weighted train steps
and reanalyze fire in a handful of iters. Asserts the driver runs and returns a
finite loss — proving the full batched VP pipeline (PER + device-obs + reanalyze)
integrates on-device.

Run (GPU env required):
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_value_prefix_batched_gpu.mojo
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
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_gpu_batched_vp import (
    run_ezv2_gumbel_selfplay_gpu_batched_vp,
)
from mojo_rl.deep_agents.training.batched_env import BatchedCpuDiscreteEnv
from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


comptime FRAMES = 4
comptime ACT = 18
comptime BINS = 601
comptime Cfg = EZV2AtariConfig[FRAMES, ACT, LAYOUT=LAYOUT_NCHW]
comptime OBS = Cfg.OBS
comptime LATENT = Cfg.LATENT
comptime HORIZON = EZ_LSTM_HORIZON

comptime N_ENVS = 2
comptime NUM_SIMS = 4
comptime MAX_NODES = 32
comptime MAX_K = 4
comptime CAP = 600
comptime B = 4
comptime K = 2
comptime N = 3

comptime AtariPong = AtariEnv[2, DT]
comptime BatchedPong = BatchedCpuDiscreteEnv[AtariPong, N_ENVS, OBS]
comptime VPDyn = EZDynVPNetAtari[ACT, BINS]


def _make_envs(
    rom: UnsafePointer[UInt8, MutAnyOrigin], rom_size: Int
) raises -> List[AtariPong]:
    var envs = List[AtariPong]()
    for _ in range(N_ENVS):
        envs.append(
            AtariPong(
                AtariGame.PONG, rom, rom_size,
                clip_reward=True, full_action_set=True,
            )
        )
    return envs^


def main() raises:
    print("=" * 70)
    print("EZv2-Atari BATCHED value-prefix driver smoke: PER + device-obs + VP")
    print("=" * 70)

    var ctx = DeviceContext()
    var rom = load_rom("roms/pong.bin")
    var env = BatchedPong(_make_envs(rom.data.value().as_unsafe_any_origin(), rom.size), noop_max=4)

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

    var loss = run_ezv2_gumbel_selfplay_gpu_batched_vp[
        BatchedPong, Cfg.Rep, Cfg.Pred, Cfg.Proj, Cfg.Predh,
        N_ENVS, OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        REANA_W=4,                  # > N_ENVS=2: exercise the wide-reanalyze path
        HORIZON=HORIZON,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, orew, opred, oproj, opredh,
        iterations=18,
        learning_starts=6,
        train_per_iter=N_ENVS,
        lr=Scalar[DT](0.2),
        lr_warmup_iters=4,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-300.0),
        v_max=Scalar[DT](300.0),
        value_coef=Scalar[DT](0.25),
        consistency_coef=Scalar[DT](2.0),
        max_ep_steps=6,
        reanalyze_every=2,
        reanalyze_batch=B,
        seed=7,
        verbose=False,
    )

    print("  final loss:", loss)
    assert_true(not isnan(loss) and not isinf(loss),
                "EZv2-Atari batched VP smoke loss finite")
    _ = env^
    print("=" * 70)
    print("PASSED — batched value-prefix driver runs end-to-end")
    print("=" * 70)
