"""EZv2 Atari Pong — TRAINING-PHASE profiling harness (GPU).

The plain `ezv2_pong_atari_gpu.mojo` run spends its first ~850 iterations in
warmup self-play (one episode must complete before training can start), so an
`nsys` capture of it is dominated by the cheap collection phase and barely shows
the train/reanalyze cost. This harness reaches the training phase in ~40
iterations by truncating episodes short (`max_ep_steps`) and lowering
`learning_starts`, then runs a short, **training-dominated** window — so the
capture reflects the real per-iteration GPU work.

CRITICAL: the model + planner + replay dims that determine per-kernel cost are
IDENTICAL to the real run (B=256, K=5, N=5, 16 sims / 4 top, latent [64,6,6],
BINS=601, REANA_W=64, reanalyze ratio 1.0). Only the things that don't change
kernel shape are shrunk for a fast, training-heavy capture:
  • max_ep_steps=40   → episodes truncate fast → buffer fills in ~40 iters
  • learning_starts=160 → training starts ~iter 40 (160 = 4 envs × 40 steps)
  • iterations=200    → ~160 training iters captured (warmup is only ~20%)
  • CAP=4096          → ~450 MB device ring (vs 11 GB) → fast startup, low VRAM
  • eval off          → no multi-minute eval stall in the window
The train step does the SAME work regardless of episode *content*, so these only
change WHEN training runs, not its cost.

Run + read (NVIDIA):
    pixi run -e nvidia nsys profile --trace=cuda --stats=true \\
        -o ezv2_train_profile \\
        mojo run -I . examples/atari/ezv2_pong_atari_profile_gpu.mojo

Then interpret:
  1. The per-section `[time s]` line at iter 200 (verbose) = WALL split across
     search / collect / env / store / train / reana for the training window.
  2. nsys "CUDA GPU Kernel Summary" → sum the Total-Time column = GPU-BUSY time.
     Compare to the run's wall (≈ the timers' sum, or `nsys stats` process
     duration). GPU-busy ≪ wall ⇒ launch/host-bound (the hypothesis); GPU-busy ≈
     wall ⇒ real compute.
  3. For the idle gaps directly: `nsys stats --report cuda_gpu_trace \\
     ezv2_train_profile.nsys-rep` (per-kernel timestamps → gaps = GPU idle), or
     open the .nsys-rep timeline in the Nsight Systems GUI and read GPU row
     utilization.
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

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


# ── real per-kernel dims (unchanged from the production run) ──
comptime FRAMES = 4
comptime ACT = 18
comptime BINS = 601
comptime Cfg = EZV2AtariConfig[FRAMES, ACT]
comptime OBS = Cfg.OBS
comptime LATENT = Cfg.LATENT

comptime N_ENVS = 4
comptime NUM_SIMS = 16
comptime MAX_NODES = 128
comptime MAX_K = 4
comptime B = 256
comptime K = 5
comptime N = 5
comptime OBS_STORE = DType.uint8
comptime REANA_W = 64

# ── profiling-only knobs (don't affect kernel shape) ──
comptime CAP = 4096                  # small device ring → fast startup/low VRAM


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
    var env = BatchedPong(_make_envs(rom.data.value(), rom.size), noop_max=30)

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

    print("EZv2 Atari Pong — TRAINING-PHASE profile (GPU)")
    print("  real dims: B", B, "K", K, "N", N, "sims", NUM_SIMS, "REANA_W",
          REANA_W, "| profiling: CAP", CAP, "max_ep 40, train@~iter40")

    var loss = run_ezv2_gumbel_selfplay_gpu_batched[
        BatchedPong, Cfg.Rep, Cfg.Dyn, Cfg.Pred, Cfg.Proj, Cfg.Predh,
        N_ENVS, OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        REANA_W=REANA_W,
        OBS_STORE_DT=OBS_STORE,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=200,                 # ~160 training iters captured
        learning_starts=160,            # 4 envs × 40 steps → train @ ~iter 40
        train_per_iter=N_ENVS,          # UTD 1:1 (real)
        lr=Scalar[DT](0.2),
        lr_warmup_iters=50,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-300.0),
        v_max=Scalar[DT](300.0),
        value_coef=Scalar[DT](0.5),
        consistency_coef=Scalar[DT](5.0),
        max_ep_steps=40,                # truncate episodes → fast buffer fill
        reanalyze_every=1,
        reanalyze_batch=B,              # ratio 1.0 (real) → 4 wide searches/iter
        eval_every=0,                   # eval off (no stall in the window)
        seed=42,
        verbose=True,                   # per-section [time s] every 100 iters
    )
    _ = env^
    print("final loss:", loss)
