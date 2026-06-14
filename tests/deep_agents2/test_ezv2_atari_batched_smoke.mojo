"""EZv2-Atari BATCHED driver smoke (Stage 6c-3) — GPU, tiny dims.

End-to-end check for `run_ezv2_gumbel_selfplay_gpu_batched`: N_ENVS RGB-96
`AtariEnv[2]` CPU envs (full 18-action, reward-clip) stepped in lockstep →
batched Gumbel search over the spatial nets → **prioritized** sequence replay →
**IS-weighted** EZv2 SGD train step + init_zero + warmup LR + reanalyze
(ratio-1.0 regime) + UTD 1:1. Tiny dims + short forced episodes so an episode
lands in the buffer and real (prioritized, weighted) train steps + reanalyze
fire within a handful of iterations. Asserts the driver runs and returns a
finite loss.

Run (GPU env required):
    pixi run -e apple mojo run -I . tests/deep_agents2/test_ezv2_atari_batched_smoke.mojo
"""

from std.memory import UnsafePointer
from std.math import isnan, isinf
from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.sgd import SGD
from mojo_rl.deep_agents2.efficient_zero_v2.config_atari import EZV2AtariConfig
from mojo_rl.deep_agents2.efficient_zero_v2.nets_atari import (
    ez_atari_init_zero_pred, ez_atari_init_zero_dyn,
)
from mojo_rl.deep_agents2.efficient_zero_v2.selfplay_gpu_batched import (
    run_ezv2_gumbel_selfplay_gpu_batched,
)
from mojo_rl.deep_agents2.training.batched_env import BatchedCpuDiscreteEnv
from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


comptime FRAMES = 4
comptime ACT = 18
comptime BINS = 601
comptime Cfg = EZV2AtariConfig[FRAMES, ACT]
comptime OBS = Cfg.OBS
comptime LATENT = Cfg.LATENT

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
    print("EZv2-Atari BATCHED driver smoke (6c-3): N_ENVS RGB-96 + PER + SGD")
    print("=" * 70)

    var ctx = DeviceContext()
    var rom = load_rom("roms/pong.bin")
    var env = BatchedPong(_make_envs(rom.data.value(), rom.size), noop_max=4)

    var rep = Cfg.Rep.make["gpu", INIT=Kaiming](ctx)
    var dyn = Cfg.Dyn.make["gpu", INIT=Kaiming](ctx)
    var pred = Cfg.Pred.make["gpu", INIT=Kaiming](ctx)
    var proj = Cfg.Proj.make["gpu", INIT=Kaiming](ctx)
    var predh = Cfg.Predh.make["gpu", INIT=Kaiming](ctx)

    ez_atari_init_zero_pred["gpu", ACT, BINS](pred, ctx)
    ez_atari_init_zero_dyn["gpu", ACT, BINS](dyn, ctx)
    ctx.synchronize()

    var orep = SGD.make["gpu", M = Cfg.Rep](rep, ctx)
    var odyn = SGD.make["gpu", M = Cfg.Dyn](dyn, ctx)
    var opred = SGD.make["gpu", M = Cfg.Pred](pred, ctx)
    var oproj = SGD.make["gpu", M = Cfg.Proj](proj, ctx)
    var opredh = SGD.make["gpu", M = Cfg.Predh](predh, ctx)
    orep.momentum = Scalar[DT](0.9); orep.weight_decay = Scalar[DT](1e-4)
    odyn.momentum = Scalar[DT](0.9); odyn.weight_decay = Scalar[DT](1e-4)
    opred.momentum = Scalar[DT](0.9); opred.weight_decay = Scalar[DT](1e-4)
    oproj.momentum = Scalar[DT](0.9); oproj.weight_decay = Scalar[DT](1e-4)
    opredh.momentum = Scalar[DT](0.9); opredh.weight_decay = Scalar[DT](1e-4)
    orep.max_grad_norm = Scalar[DT](5.0); odyn.max_grad_norm = Scalar[DT](5.0)
    opred.max_grad_norm = Scalar[DT](5.0); oproj.max_grad_norm = Scalar[DT](5.0)
    opredh.max_grad_norm = Scalar[DT](5.0)

    var loss = run_ezv2_gumbel_selfplay_gpu_batched[
        BatchedPong, Cfg.Rep, Cfg.Dyn, Cfg.Pred, Cfg.Proj, Cfg.Predh,
        N_ENVS, OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        REANA_W=4,                  # > N_ENVS=2: exercise the wide-reanalyze path
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=18,
        learning_starts=6,          # stored steps
        train_per_iter=N_ENVS,      # UTD 1:1
        lr=Scalar[DT](0.2),
        lr_warmup_iters=4,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-300.0),
        v_max=Scalar[DT](300.0),
        value_coef=Scalar[DT](0.25),
        consistency_coef=Scalar[DT](2.0),
        max_ep_steps=6,             # force short episodes
        reanalyze_every=2,
        reanalyze_batch=B,          # ≈ ratio-1.0 coverage
        seed=7,
        verbose=False,
    )

    print("  final loss:", loss)
    assert_true(not isnan(loss) and not isinf(loss),
                "EZv2-Atari batched smoke loss finite")
    _ = env^
    print("=" * 70)
    print("PASSED")
    print("=" * 70)
