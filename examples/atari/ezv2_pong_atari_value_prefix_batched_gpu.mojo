"""EfficientZeroV2 + **value prefix** on Atari Pong — BATCHED driver (GPU).

The throughput path: the value-prefix sibling of `ezv2_pong_atari_gpu.mojo`,
wired through the BATCHED driver `run_ezv2_gumbel_selfplay_gpu_batched_vp`
(N_ENVS parallel CPU envs + on-device nets + PER + device-obs ring + wide
reanalyze). This removes the two big single-env bottlenecks:
  * collection runs N_ENVS=4 emulators in parallel (vs one),
  * reanalyze uses ONE wide REANA_W-root search instead of `reanalyze_batch`
    separate 1-root searches,
  * the obs slab lives in a device ring (no ~680 MB host→device copy per step).

⚠️ This batched driver currently ICEs the **Apple/Metal** backend (a codegen-size
limit, not a logic bug — every component is verified green; see the driver
docstring). The CUDA backend differs, so try it here first:

    pixi run -e nvidia mojo run -I . examples/atari/ezv2_pong_atari_value_prefix_batched_gpu.mojo

If it ICEs on NVIDIA too, fall back to the single-env
`ezv2_pong_atari_value_prefix_gpu.mojo` (slower) and ping for the function-split
refactor. Recipe matches `atari.yaml` (B=256, K=5, N=5, sims 16/top 4, γ 0.997,
support 601 over ±300, value-prefix horizon 5).

Requires the Pong ROM at `roms/pong.bin`.
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
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
comptime ACT = 18                    # full ALE action set
comptime BINS = 601                  # support over [-300, 300]
comptime Cfg = EZV2AtariConfig[FRAMES, ACT]
comptime OBS = Cfg.OBS               # 110592 = 12·96·96
comptime LATENT = Cfg.LATENT         # 2304 = [64,6,6]
comptime HORIZON = EZ_LSTM_HORIZON   # 5  (lstm_horizon_len)

comptime N_ENVS = 4                  # data.num_envs (parallel CPU emulators)
comptime NUM_SIMS = 16               # mcts.num_simulations
comptime MAX_NODES = 128
comptime MAX_K = 4                   # mcts.num_top_actions
comptime CAP = 100000                # ≥ total transitions (no eviction)
comptime B = 256                     # train.batch_size
comptime K = 5                       # rl.unroll_steps
comptime N = 5                       # rl.td_steps
comptime OBS_STORE = DType.uint8     # lossless k/255 pixel storage (4× capacity)
comptime REANA_W = 64                # reanalyze search width (one wide search)

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
    print("EZv2 + value prefix — Atari Pong (GPU, BATCHED driver)")
    print("=" * 70)

    var rom = load_rom("roms/pong.bin")
    print("ROM loaded:", rom.size, "bytes")

    with DeviceContext() as ctx:
        var env = BatchedPong(_make_envs(rom.data.value(), rom.size), noop_max=30)
        var eval_env = BatchedPong(
            _make_envs(rom.data.value(), rom.size), noop_max=30
        )

        var rep = Cfg.Rep.make["gpu", Kaiming](Optional(ctx))
        var dyn = VPDyn.make["gpu", Kaiming](Optional(ctx))
        var pred = Cfg.Pred.make["gpu", Kaiming](Optional(ctx))
        var proj = Cfg.Proj.make["gpu", Kaiming](Optional(ctx))
        var predh = Cfg.Predh.make["gpu", Kaiming](Optional(ctx))

        # init_zero (EZ value_prefix=True): neutral value/value-prefix + uniform
        # policy at init. The value-prefix head lives at dyn.rew.
        ez_atari_init_zero_pred["gpu", ACT, BINS](pred, ctx)
        ez_atari_init_zero_reward["gpu", BINS](dyn.rew, ctx)
        ctx.synchronize()

        # SGD is the paper optimizer; Adam(0.2)+warmup mirrors the sibling
        # example's placeholder (swap to nn SGD for a faithful run). 6 optimizers:
        # odyn steps dyn.dynz, orew the LSTM value-prefix head dyn.rew.
        var orep = Adam(lr=Scalar[DT](0.2))
        var odyn = Adam(lr=Scalar[DT](0.2))
        var orew = Adam(lr=Scalar[DT](0.2))
        var opred = Adam(lr=Scalar[DT](0.2))
        var oproj = Adam(lr=Scalar[DT](0.2))
        var opredh = Adam(lr=Scalar[DT](0.2))

        var env_vars = load_dotenv()
        var logger = RemoteLogger(
            server_url=env_vars.get("RL_MONITOR_URL", ""),
            run_name="EZv2 ValuePrefix Atari Pong GPU (batched)",
            buffer_size=64,
            api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
        )
        logger.set_config("agent", "EZv2 + value-prefix (batched)")
        logger.set_config("env", "Atari Pong (RGB 96, OBS_MODE=2)")
        logger.set_config("value_prefix", "True")
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("B", String(B)); logger.set_config("K", String(K))

        print("  N_ENVS", N_ENVS, "B", B, "K", K, "N", N, "sims", NUM_SIMS,
              "top", MAX_K, "REANA_W", REANA_W, "horizon", HORIZON,
              "BINS", BINS, "v±300")
        print("  budget: 25000 iters ×", N_ENVS, "= 100k env transitions")

        var loss = run_ezv2_gumbel_selfplay_gpu_batched_vp[
            BatchedPong, Cfg.Rep, Cfg.Pred, Cfg.Proj, Cfg.Predh,
            N_ENVS, OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
            REANA_W=REANA_W,
            OBS_STORE_DT=OBS_STORE,
            HORIZON=HORIZON,
            L=RemoteLogger,
        ](
            ctx, env, rep, dyn, pred, proj, predh,
            orep, odyn, orew, opred, oproj, opredh,
            iterations=25000,               # × N_ENVS = 100k env transitions
            learning_starts=2000,           # start_transitions (stored steps)
            train_per_iter=N_ENVS,          # UTD 1:1
            lr=Scalar[DT](0.2),
            lr_warmup_iters=1000,
            gamma=Scalar[DT](0.997),
            v_min=Scalar[DT](-300.0),
            v_max=Scalar[DT](300.0),
            value_coef=Scalar[DT](0.5),
            consistency_coef=Scalar[DT](5.0),
            temperature_decay_steps=25000,
            reanalyze_every=1,
            reanalyze_batch=B,              # ratio≈1.0 → ceil(B/REANA_W)=4 wide searches/iter
            eval_every=5000,
            eval_episodes=10,
            eval_horizon=10000,
            eval_env=UnsafePointer(to=eval_env),
            diag_every=200,
            report_every=500,
            logger=UnsafePointer(to=logger),
            seed=42,
            verbose=True,
        )
        logger.close()
        _ = env^
        _ = eval_env^
        print("final loss:", loss)
