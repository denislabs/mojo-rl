"""EZv2 + value-prefix Atari Pong — NCHW-vs-NHWC layout A/B profiler (GPU).

Layout-selectable twin of `ezv2_pong_atari_value_prefix_profile_gpu.mojo` for the
channels_last migration. Flip `USE_NHWC` and rebuild to profile the same
training-dominated window under each layout, then diff the two nsys captures —
the difference isolates the REPRESENTATION-net conv + BN cost (the only thing the
layout changes; Dyn/Pred/Proj/Predh consume a canonical NCHW latent via the
`ToNCHW` adapter, so they're identical across both runs).

What the layout flips (and ONLY this):
  • the rep tower's internal memory layout (NHWC = channels-last im2col + the
    coalesced BN transposed reduction), and
  • the Atari env's RGB-96 obs layout (`AtariEnv[..., LAYOUT]`), so the env emits
    the obs the rep net expects — no runtime transpose.
Everything downstream of the latent (dynamics, prediction, projector, predictor,
LSTM reward, MCTS, replay) is byte-for-byte identical.

Expected from the A/B (NVIDIA): the rep-net forward im2col is ~1.65× on the hot
48×48 conv (neutral elsewhere); BN is ~parity (coalesced both ways); backward
col2im already 2.7–5.4×. So NHWC should shrink the conv-kernel slice of the train
+ reanalyze + self-play sections with no convergence change (verify separately).

CRITICAL: per-kernel dims are IDENTICAL to the real run (B=256, K=5, N=5, 16 sims
/ 4 top, latent [64,6,6], BINS=601, REANA_W=64, reanalyze 1.0, VP horizon 5).
Only WHEN training runs is shrunk (max_ep_steps=10, learning_starts=40,
iterations=60, CAP=4096, eval off) so the capture is training-dominated.

⚠️ The batched VP driver ICEs the Apple/Metal backend (codegen-size limit); run
this on NVIDIA.

Run + read (NVIDIA) — two captures, distinct -o:
    # NCHW (USE_NHWC=False, default)
    pixi run -e nvidia nsys profile --trace=cuda --stats=true \\
        -o ezv2_vp_nchw_profile \\
        mojo run -I . examples/atari/ezv2_pong_atari_value_prefix_profile_layout_gpu.mojo
    # NHWC: set USE_NHWC=True below, then
    pixi run -e nvidia nsys profile --trace=cuda --stats=true \\
        -o ezv2_vp_nhwc_profile \\
        mojo run -I . examples/atari/ezv2_pong_atari_value_prefix_profile_layout_gpu.mojo

Then diff the "CUDA GPU Kernel Summary" of the two .nsys-rep files — focus on the
conv2d im2col/col2im + bn2d kernels (the rep-net cost). The per-section
`[time s]` line (search / collect / env / store / train / reana) also shows the
wall split per layout.
"""

from std.memory import Pointer
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.efficient_zero_v2.config_atari import EZV2AtariConfig
from mojo_rl.deep_agents.efficient_zero_v2.nets_atari import (
    EZDynVPNetAtari,
    ez_atari_init_zero_pred,
    ez_atari_init_zero_reward,
    EZ_LSTM_HORIZON,
)
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_gpu_batched_vp import (
    run_ezv2_gumbel_selfplay_gpu_batched_vp,
)
from mojo_rl.deep_agents.training.batched_env import BatchedCpuDiscreteEnv
from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


# ═══════════════════════════════════════════════════════════════════════════
# A/B TOGGLE — flip to True + rebuild for the channels-last (NHWC) capture.
# ═══════════════════════════════════════════════════════════════════════════
comptime USE_NHWC = True
comptime LAYOUT = LAYOUT_NHWC if USE_NHWC else LAYOUT_NCHW


# ── real per-kernel dims (unchanged from the production run) ──
comptime FRAMES = 4
comptime ACT = 18
comptime BINS = 601
comptime Cfg = EZV2AtariConfig[FRAMES, ACT, LAYOUT=LAYOUT]
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
comptime CAP = 4096  # small device ring → fast startup / low VRAM


# Env obs layout MUST match the rep-net layout (the rep consumes what the env
# emits — no transpose). Default NCHW keeps every other agent untouched.
comptime AtariPong = AtariEnv[2, DT, LAYOUT]
comptime BatchedPong = BatchedCpuDiscreteEnv[AtariPong, N_ENVS, OBS]
comptime VPDyn = EZDynVPNetAtari[ACT, BINS]


def _make_envs(
    rom: Pointer[UInt8, MutAnyOrigin], rom_size: Int
) raises -> List[AtariPong]:
    var envs = List[AtariPong]()
    for _ in range(N_ENVS):
        envs.append(
            AtariPong(
                AtariGame.PONG,
                rom,
                rom_size,
                clip_reward=True,
                full_action_set=True,
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

    print("EZv2+VP Atari Pong — LAYOUT A/B profile (GPU, batched)")
    print(
        "  LAYOUT =", "NHWC (channels-last)" if USE_NHWC else "NCHW (default)"
    )
    print(
        "  real dims: B",
        B,
        "K",
        K,
        "N",
        N,
        "sims",
        NUM_SIMS,
        "REANA_W",
        REANA_W,
        "horizon",
        HORIZON,
        "| profiling: CAP",
        CAP,
        "max_ep 10, train@~iter10",
    )

    var loss = run_ezv2_gumbel_selfplay_gpu_batched_vp[
        BatchedPong,
        Cfg.Rep,
        Cfg.Pred,
        Cfg.Proj,
        Cfg.Predh,
        N_ENVS,
        OBS,
        ACT,
        LATENT,
        BINS,
        NUM_SIMS,
        MAX_NODES,
        MAX_K,
        CAP,
        B,
        K,
        N,
        REANA_W=REANA_W,
        OBS_STORE_DT=OBS_STORE,
        HORIZON=HORIZON,
    ](
        ctx,
        env,
        rep,
        dyn,
        pred,
        proj,
        predh,
        orep,
        odyn,
        orew,
        opred,
        oproj,
        opredh,
        iterations=60,  # short training-dominated window
        learning_starts=40,  # 4 envs × 10 steps → train @ ~iter 10
        train_per_iter=N_ENVS,  # UTD 1:1 (real)
        lr=Scalar[DT](1e-3),
        lr_warmup_iters=50,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-300.0),
        v_max=Scalar[DT](300.0),
        value_coef=Scalar[DT](0.5),
        consistency_coef=Scalar[DT](5.0),
        max_ep_steps=10,  # truncate episodes → fast buffer fill
        reanalyze_every=1,
        reanalyze_batch=B,  # ratio 1.0 (real) → 4 wide searches/iter
        eval_every=0,  # eval off (no stall in the window)
        seed=42,
        verbose=True,  # per-section [time s] every 100 iters
        diag_sync=True,  # accepted by the driver (no-op for VP train)
    )
    _ = env^
    print("final loss:", loss)
