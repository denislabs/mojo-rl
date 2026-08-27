"""EZv2 + value-prefix Atari Pong — TRAINING-PHASE profiling harness (GPU).

Value-prefix sibling of `ezv2_pong_atari_profile_gpu.mojo`, wired through the
BATCHED value-prefix driver `run_ezv2_gumbel_selfplay_gpu_batched_vp` (fused
EZDynVPNetAtari + LSTM reward head + PER + device-obs ring + wide reanalyze).

Like the non-VP profiler: reaches the training phase in ~10 iters by truncating
episodes short + lowering learning_starts, then runs a short, **training-
dominated** window so an `nsys` capture reflects real per-iteration GPU work.

CRITICAL: the per-kernel-cost dims are IDENTICAL to the real run (B=256, K=5,
N=5, 16 sims / 4 top, latent [64,6,6], BINS=601, REANA_W=64, reanalyze ratio
1.0, value-prefix horizon 5). Only WHEN training runs is shrunk:
  • max_ep_steps=10     → episodes truncate fast → buffer fills in ~10 iters
  • learning_starts=40  → training starts ~iter 10 (40 = 4 envs × 10 steps)
  • iterations=60       → ~50 training iters captured
  • CAP=4096            → ~450 MB device ring (vs 11 GB) → fast startup / low VRAM
  • eval off            → no eval stall in the window

⚠️ NOTE vs the non-VP profiler: the VP train step (`ezv2_unroll_train_step_gpu_vp`)
does NOT populate the driver's `phase_ns` array, so the printed "step phases" /
"rev calls" breakdown lines will read 0.0. The per-section `[time s]` line
(search / collect / env / store / train / reana) DOES work, and nsys captures
kernel-level detail directly — which is the point of this harness. (If you want
the in-train-step phase split too, ask for `phase_ns` instrumentation in
`_gpu_vp`.)

⚠️ The batched VP driver ICEs the Apple/Metal backend (codegen-size limit); run
this on NVIDIA.

Run + read (NVIDIA):
    pixi run -e nvidia nsys profile --trace=cuda --stats=true \\
        -o ezv2_vp_train_profile \\
        mojo run -I . examples/atari/ezv2_pong_atari_value_prefix_profile_gpu.mojo

Then interpret:
  1. The per-section `[time s]` line (verbose) = WALL split across search /
     collect / env / store / train / reana for the training window. From your
     run, train dominates (~77%), reana ~19%.
  2. nsys "CUDA GPU Kernel Summary" → sum the Total-Time column = GPU-BUSY time.
     Compare to wall (≈ the timers' sum). GPU-busy ≪ wall ⇒ launch/host-bound
     (many tiny sequential kernels in the K-step unroll + LSTM (h,c) ops);
     GPU-busy ≈ wall ⇒ real compute (the rep ResNet forwards dominate). This is
     the key question for what to optimize.
  3. Per-kernel gaps: `nsys stats --report cuda_gpu_trace ezv2_vp_train_profile.nsys-rep`
     (kernel timestamps → gaps = GPU idle), or open the .nsys-rep in the Nsight
     Systems GUI and read the GPU row utilization. Look for: how much time is in
     conv (rep/dyn/pred) vs LSTM GEMMs vs the ~hundreds of tiny element-wise +
     (h,c) sub-buffer-copy kernels per train step.
"""

from std.memory import Pointer
from max.gpu.host import DeviceContext

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


# ── real per-kernel dims (unchanged from the production run) ──
comptime FRAMES = 4
comptime ACT = 18
comptime BINS = 601
comptime Cfg = EZV2AtariConfig[FRAMES, ACT]
comptime OBS = Cfg.OBS
comptime LATENT = Cfg.LATENT
comptime HORIZON = EZ_LSTM_HORIZON

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
comptime CAP = 4096                  # small device ring → fast startup / low VRAM


comptime AtariPong = AtariEnv[2, DT, Cfg.LAYOUT]
comptime BatchedPong = BatchedCpuDiscreteEnv[AtariPong, N_ENVS, OBS]
comptime VPDyn = EZDynVPNetAtari[ACT, BINS]


def _make_envs(
    rom: Pointer[UInt8, MutUntrackedOrigin], rom_size: Int
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

    var rep = Cfg.Rep.make["gpu", Kaiming](Optional(ctx))
    var dyn = VPDyn.make["gpu", Kaiming](Optional(ctx))
    var pred = Cfg.Pred.make["gpu", Kaiming](Optional(ctx))
    var proj = Cfg.Proj.make["gpu", Kaiming](Optional(ctx))
    var predh = Cfg.Predh.make["gpu", Kaiming](Optional(ctx))
    ez_atari_init_zero_pred["gpu", ACT, BINS](pred, ctx)
    ez_atari_init_zero_reward["gpu", BINS](dyn.rew, ctx)
    ctx.synchronize()

    # Adam @ lr 1e-3 (driver lr/lr_warmup_iters override each train step). The
    # optimizer choice/lr is immaterial to the profile (same compute graph).
    var orep = Adam(lr=Scalar[DT](1e-3))
    var odyn = Adam(lr=Scalar[DT](1e-3))
    var orew = Adam(lr=Scalar[DT](1e-3))
    var opred = Adam(lr=Scalar[DT](1e-3))
    var oproj = Adam(lr=Scalar[DT](1e-3))
    var opredh = Adam(lr=Scalar[DT](1e-3))

    print("EZv2+VP Atari Pong — TRAINING-PHASE profile (GPU, batched)")
    print("  real dims: B", B, "K", K, "N", N, "sims", NUM_SIMS, "REANA_W",
          REANA_W, "horizon", HORIZON,
          "| profiling: CAP", CAP, "max_ep 10, train@~iter10")

    var loss = run_ezv2_gumbel_selfplay_gpu_batched_vp[
        BatchedPong, Cfg.Rep, Cfg.Pred, Cfg.Proj, Cfg.Predh,
        N_ENVS, OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        REANA_W=REANA_W,
        OBS_STORE_DT=OBS_STORE,
        HORIZON=HORIZON,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, orew, opred, oproj, opredh,
        iterations=60,                  # short training-dominated window
        learning_starts=40,             # 4 envs × 10 steps → train @ ~iter 10
        train_per_iter=N_ENVS,          # UTD 1:1 (real)
        lr=Scalar[DT](1e-3),
        lr_warmup_iters=50,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-300.0),
        v_max=Scalar[DT](300.0),
        value_coef=Scalar[DT](0.5),
        consistency_coef=Scalar[DT](5.0),
        max_ep_steps=10,                # truncate episodes → fast buffer fill
        reanalyze_every=1,
        reanalyze_batch=B,              # ratio 1.0 (real) → 4 wide searches/iter
        eval_every=0,                   # eval off (no stall in the window)
        seed=42,
        verbose=True,                   # per-section [time s] every 100 iters
        diag_sync=True,                 # accepted by the driver (no-op for VP train)
    )
    _ = env^
    print("final loss:", loss)
