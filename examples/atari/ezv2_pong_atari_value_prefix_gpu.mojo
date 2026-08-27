"""EfficientZeroV2 + **value prefix** on Atari 2600 Pong (GPU).

The value-prefix sibling of `ezv2_pong_atari_gpu.mojo` (decision B1 / B1.1, see
`docs/EZV2_ATARI_PARITY.md`). Same EZv2 Atari recipe and spatial model, but the
reward head is the stateful **value-prefix LSTM** (`EZRewardLSTMAtari`): training
predicts cumulative within-window reward sums (reset every `lstm_horizon_len`)
through `(h,c)` BPTT, and the fused `EZDynVPNetAtari` serves the search as a
drop-in `[z|act]→[z'|vp]` dynamics (the LSTM head runs with zero (h,c) per node —
the deferred search-side carry is decision B1.1).

Wired through the dedicated **value-prefix driver** `run_ezv2_gumbel_selfplay_
gpu_vp` (single CPU env / GPU nets), which trains via `ezv2_unroll_train_step_
gpu_vp` on `dyn.dynz` + `dyn.rew` with a 6th (reward-head) optimizer and
`value_prefix_from_rewards` targets.

    obs       RGB 96×96, 4-frame stack → [12,96,96]   (AtariEnv[2], OBS_MODE=2)
    actions   full 18-action ALE set                  (full_action_set)
    reward    sign-clipped (inert on Pong)            (clip_reward)
    model     spatial latent [64,6,6]=2304, init_zero (EZV2AtariConfig + VP head)
    support   601 atoms over [-300, 300]              (BINS=601, v±300)
    discount  0.997   unroll K=5   td N=5   value-prefix horizon 5

This is the SINGLE-env driver (collection is one emulator) — correct + runnable,
but slower than the batched `ezv2_pong_atari_gpu.mojo`. A batched value-prefix
driver is the natural perf follow-up. Compare the greedy-eval curve against the
Rainbow-Pong-pixel baseline.

Requires the Pong ROM at `roms/pong.bin`.

Run:
    pixi run -e apple  mojo run -I . examples/atari/ezv2_pong_atari_value_prefix_gpu.mojo  # compile/smoke
    pixi run -e nvidia mojo run -I . examples/atari/ezv2_pong_atari_value_prefix_gpu.mojo  # training
"""

from std.memory import Pointer
from max.gpu.host import DeviceContext

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
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_gpu_vp import (
    run_ezv2_gumbel_selfplay_gpu_vp,
)
from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


comptime FRAMES = 4
comptime ACT = 18                    # full ALE action set
comptime BINS = 601                  # support over [-300, 300]
comptime Cfg = EZV2AtariConfig[FRAMES, ACT]
comptime OBS = Cfg.OBS               # 110592 = 12·96·96
comptime LATENT = Cfg.LATENT         # 2304 = [64,6,6]
comptime HORIZON = EZ_LSTM_HORIZON   # 5  (lstm_horizon_len)

comptime NUM_SIMS = 16               # mcts.num_simulations
comptime MAX_NODES = 128
comptime MAX_K = 4                   # mcts.num_top_actions
comptime CAP = 100000                # ≥ total transitions (no eviction)
comptime B = 256                     # train.batch_size
comptime K = 5                       # rl.unroll_steps
comptime N = 5                       # rl.td_steps

comptime VPDyn = EZDynVPNetAtari[ACT, BINS]   # fused value-prefix dynamics


def main() raises:
    print("=" * 70)
    print("EZv2 + value prefix — Atari Pong (GPU, single-env driver)")
    print("=" * 70)

    var rom = load_rom("roms/pong.bin")
    print("ROM loaded:", rom.size, "bytes")

    with DeviceContext() as ctx:
        var env = AtariEnv[2, DT, Cfg.LAYOUT](
            AtariGame.PONG, rom.data.value(), rom.size,
            clip_reward=True, full_action_set=True,
        )

        # nets: spatial rep/pred/proj/predh from the config; fused VP dynamics.
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

        # Adam @ lr 1e-3 (the driver's `lr` arg below overrides this each train
        # step via warmup→const). NOTE: the EZv2 paper uses SGD 0.2/mom0.9/wd1e-4;
        # Adam@0.2 diverges (loss oscillated 20→289→72). Adam@1e-3 is the sane fix
        # without making the EZv2 train steps Optimizer-generic (SGD = follow-up).
        # Tune down (3e-4) if loss is unstable with consistency_coef=5. 6 opts:
        # odyn steps dyn.dynz, orew the LSTM value-prefix head dyn.rew.
        var orep = Adam(lr=Scalar[DT](1e-3))
        var odyn = Adam(lr=Scalar[DT](1e-3))
        var orew = Adam(lr=Scalar[DT](1e-3))
        var opred = Adam(lr=Scalar[DT](1e-3))
        var oproj = Adam(lr=Scalar[DT](1e-3))
        var opredh = Adam(lr=Scalar[DT](1e-3))

        var env_vars = load_dotenv()
        var logger = RemoteLogger(
            server_url=env_vars.get("RL_MONITOR_URL", ""),
            run_name="EZv2 ValuePrefix Atari Pong GPU",
            buffer_size=64,
            api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
        )
        logger.set_config("agent", "EZv2 + value-prefix (LSTM reward head)")
        logger.set_config("env", "Atari Pong (RGB 96, OBS_MODE=2)")
        logger.set_config("value_prefix", "True")
        logger.set_config("lstm_horizon", String(HORIZON))
        logger.set_config("B", String(B)); logger.set_config("K", String(K))

        print("  value_prefix=True  horizon", HORIZON,
              "B", B, "K", K, "N", N, "sims", NUM_SIMS, "top", MAX_K,
              "BINS", BINS, "v±300")
        print("  budget: 100k env transitions (single env)")

        var loss = run_ezv2_gumbel_selfplay_gpu_vp[
            AtariEnv[2, DT, Cfg.LAYOUT], Cfg.Rep, Cfg.Pred, Cfg.Proj, Cfg.Predh,
            OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
            HORIZON=HORIZON,
            L=RemoteLogger,
        ](
            ctx, env, rep, dyn, pred, proj, predh,
            orep, odyn, orew, opred, oproj, opredh,
            iterations=100000,              # 1 env step / iter → 100k transitions
            learning_starts=2000,           # start_transitions
            train_per_iter=1,               # UTD 1:1
            lr=Scalar[DT](1e-3),            # Adam (NOT 0.2 — that diverged)
            lr_warmup_iters=1000,
            gamma=Scalar[DT](0.997),
            v_min=Scalar[DT](-300.0),
            v_max=Scalar[DT](300.0),
            value_coef=Scalar[DT](0.5),     # value_loss_coeff
            consistency_coef=Scalar[DT](5.0),  # consistency_coeff
            temperature_decay_steps=100000,
            reanalyze_every=1,
            reanalyze_batch=8,              # refresh 8 stored positions / iter
            eval_every=20000,
            eval_episodes=10,
            diag_every=200,
            report_every=500,
            logger=Pointer(to=logger).as_unsafe_any_origin(),
            seed=42,
            verbose=True,
        )
        logger.close()
        _ = env^
        print("final loss:", loss)
