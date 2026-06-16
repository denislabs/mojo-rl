"""EfficientZero-V2 Atari-100k Pong run (GPU) — Stage 7, atari.yaml-faithful.

The full-parity reproduction target: EZv2 on the **Atari emulator Pong** (NOT the
custom arcade Pong), so a divergence from the published curve is a debuggable
signal. Wires every hyperparameter from
`references/EfficientZeroV2-main/ez/config/exp/atari.yaml` into the batched EZv2
driver (`run_ezv2_gumbel_selfplay_gpu_batched`):

    obs            RGB 96×96, 4-frame stack → [12,96,96]   (AtariEnv[2], OBS_MODE=2)
    actions        full 18-action ALE set                  (full_action_set)
    reward         sign-clipped                             (clip_reward; inert on Pong)
    model          spatial latent [64,6,6]=2304, init_zero  (EZV2AtariConfig)
    support        601 atoms over [-300,300]                (BINS=601, v±300)
    discount       0.997     unroll K=5   td N=5            (yaml rl:)
    optimizer      SGD 0.2 / mom 0.9 / wd 1e-4 / warmup 1%  (yaml optimizer:)
    grad clip      5.0
    loss coeffs    value 0.5, reward 1.0, policy 1.0, consistency 5.0  (yaml train:)
    PER            α=1, β=1                                  (yaml priority:)
    MCTS           Gumbel 16 sims / 4 top                   (yaml mcts:)
    data           4 envs, UTD 1:1, reanalyze ratio 1.0     (yaml data:/train:)
    budget         100k env transitions  (25000 iters × 4)  (total_transitions)
    start          train after 2000 stored steps           (start_transitions)
    eval           10 greedy episodes every 10k env steps   (eval_n_episode/interval)

MEMORY NOTE (host RAM): the replay stores the STACKED [12,96,96] obs per step in
`uint8` (lossless k/255). CAP=100000 ⇒ ~11 GB host (4× the EZ reference, which
stores single frames and stacks on read — a documented memory deviation, not a
learning one). Lower CAP / B if you OOM. Reanalyze runs on a separate **wide**
planner of `REANA_W` roots/search, so the ratio-1.0 re-target of `reanalyze_batch`
positions costs `ceil(reanalyze_batch/REANA_W)` wide searches/iter (here 256/64 =
4), not `reanalyze_batch/N_ENVS` = 64 narrow 4-root searches + 64 syncs (the old
bottleneck). `REANA_W` trades device memory (≈ REANA_W·MAX_NODES·LATENT·4B tree)
for fewer searches; lower it (or `reanalyze_batch`) if VRAM-bound.

Watch the greedy ``[eval]`` return: random Pong ≈ −21, "solved" ≈ +21. The
published EZv2 Atari-100k Pong score is well above random within the 100k budget.

Run (NVIDIA — long run; set RL_MONITOR_URL in .env for live metrics):
    pixi run -e nvidia mojo run -I . examples/atari/ezv2_pong_atari_gpu.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.optimizer.sgd import SGD
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.efficient_zero_v2.config_atari import EZV2AtariConfig
from mojo_rl.deep_agents.efficient_zero_v2.nets_atari import (
    ez_atari_init_zero_pred, ez_atari_init_zero_dyn,
)
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_gpu_batched import (
    run_ezv2_gumbel_selfplay_gpu_batched,
)
from mojo_rl.deep_agents.training.batched_env import BatchedCpuDiscreteEnv
from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


# ── atari.yaml-faithful compile-time config ──
comptime FRAMES = 4
comptime ACT = 18                    # full ALE action set
comptime BINS = 601                  # support over [-300, 300]
comptime Cfg = EZV2AtariConfig[FRAMES, ACT]
comptime OBS = Cfg.OBS               # 110592 = 12·96·96
comptime LATENT = Cfg.LATENT         # 2304 = [64,6,6]

comptime N_ENVS = 4                  # data.num_envs
comptime NUM_SIMS = 16               # mcts.num_simulations
comptime MAX_NODES = 128
comptime MAX_K = 4                   # mcts.num_top_actions
comptime CAP = 100000                # ≥ total_transitions (no eviction); uint8 ring
comptime B = 256                     # train.batch_size
comptime K = 5                       # rl.unroll_steps
comptime N = 5                       # rl.td_steps
comptime OBS_STORE = DType.uint8     # lossless k/255 pixel storage (4× capacity)
comptime REANA_W = 64                # reanalyze search width (roots/search): one
                                     # 64-root search replaces 16 narrow 4-root
                                     # ones — ratio-1.0 re-target is B/REANA_W=4
                                     # wide searches/iter, not B/N_ENVS=64 narrow.
                                     # Tree hidden ≈ REANA_W·MAX_NODES·LATENT·4B
                                     # ≈ 75 MB device; raise toward B for fewer
                                     # searches if you have the VRAM.

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
    var ctx = DeviceContext()
    var rom = load_rom("roms/pong.bin")

    # noop_max=30 (yaml NoopResetEnv) on every (selective) reset.
    var env = BatchedPong(_make_envs(rom.data.value(), rom.size), noop_max=30)
    var eval_env = BatchedPong(
        _make_envs(rom.data.value(), rom.size), noop_max=30
    )

    var rep = Cfg.Rep.make["gpu", INIT=Kaiming](ctx)
    var dyn = Cfg.Dyn.make["gpu", INIT=Kaiming](ctx)
    var pred = Cfg.Pred.make["gpu", INIT=Kaiming](ctx)
    var proj = Cfg.Proj.make["gpu", INIT=Kaiming](ctx)
    var predh = Cfg.Predh.make["gpu", INIT=Kaiming](ctx)

    # init_zero=True: neutral value/reward + uniform policy prior at init.
    ez_atari_init_zero_pred["gpu", ACT, BINS](pred, ctx)
    ez_atari_init_zero_dyn["gpu", ACT, BINS](dyn, ctx)
    ctx.synchronize()

    # SGD 0.2 / mom 0.9 / wd 1e-4 / clip 5 (the warmup→const LR is driven by the
    # driver's lr/lr_warmup_iters; lr here is overwritten each train step).
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

    # ── metrics logger (silent no-op without RL_MONITOR_URL in env/.env) ──
    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("RL_MONITOR_URL", ""),
        run_name="EZv2 Atari Pong (100k, GPU)",
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("agent", "EZv2")
    logger.set_config("env", "AtariPong(emulator)")
    logger.set_config("framework", "deep_agents/nn")

    print("EZv2 Atari-100k Pong (GPU, atari.yaml-faithful)")
    print("  N_ENVS", N_ENVS, "B", B, "K", K, "N", N, "sims", NUM_SIMS,
          "top", MAX_K, "BINS", BINS, "v±300 SGD0.2 PER(1,1) reanalyze1.0")
    print("  budget: 25000 iters ×", N_ENVS, "= 100k env transitions")

    var loss = run_ezv2_gumbel_selfplay_gpu_batched[
        BatchedPong, Cfg.Rep, Cfg.Dyn, Cfg.Pred, Cfg.Proj, Cfg.Predh,
        N_ENVS, OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        REANA_W=REANA_W,
        OBS_STORE_DT=OBS_STORE,
        L=RemoteLogger,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=25000,               # × N_ENVS = 100k env transitions
        learning_starts=2000,           # start_transitions (stored steps)
        train_per_iter=N_ENVS,          # UTD 1:1
        lr=Scalar[DT](0.2),
        lr_warmup_iters=1000,           # lr_warm_up 0.01 × 100k grad steps
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-300.0),
        v_max=Scalar[DT](300.0),
        value_coef=Scalar[DT](0.5),     # value_loss_coeff
        consistency_coef=Scalar[DT](5.0),  # consistency_coeff (NOT 2.0)
        temperature_decay_steps=25000,
        reanalyze_every=1,
        reanalyze_batch=B,              # ratio 1.0; with the wide planner this is
                                        #   ceil(B/REANA_W)=4 wide 64-root searches/
                                        #   iter (was 64 narrow 4-root ones + 64
                                        #   syncs — the bottleneck). Lower REANA_W
                                        #   or reanalyze_batch only if VRAM-bound.
        eval_every=5000,                # every 5k iters (20k env steps) — halved
                                        #   from 2500 to cut the eval-stall frequency
        eval_episodes=10,               # eval_n_episode
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
