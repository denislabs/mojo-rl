"""DreamerV3 CartPole — TRAINING-PHASE profiling harness (GPU).

Profiling sibling of `cartpole_dreamerv3_training_gpu.mojo`, in the spirit of
`examples/atari/ezv2_pong_atari_value_prefix_profile_gpu.mojo`: it reaches the
training phase fast (tiny warmup) and then runs a short, **train_step-dominated**
window with global wall timers so an `nsys` capture reflects real per-train_step
GPU work — without the 150k-step full run.

CRITICAL: every per-train_step kernel dimension is IDENTICAL to the real GPU run
(`cartpole_dreamerv3_training_gpu.mojo`): B=16, T=16, T_IMAG=15, DETER=128, H=32,
STOCH=16, CLASSES=4, BLOCKS=4, BINS=51. Only the things that don't change kernel
shape are shrunk for a fast, profile-friendly startup:
  • CAP=8192          → tiny device replay ring → fast startup / low VRAM
  • LEARN_START small → buffer fills in a few hundred env steps
  • no env loop / no eval / no logger in the timed window

What it times (global only, per the request — kernel-level detail comes from nsys):
  1. FAST path: N × `train_step(want_diag=False)` — the real training-throughput
     path (WM-BPTT `_wm_gpu` + device-resident imagination AC `_ac_gpu_disc`,
     no host diagnostic readout). This is the window to nsys.
  2. CAPTURED path: N × `train_step_captured(want_diag=False)` — the SAME work
     replayed from a single captured CUDA graph (Stage 3 P5). The eager prologue
     (device replay sample + noise) still runs per step; only the WM+AC
     device-kernel sequence is one `cuGraphLaunch` instead of ~7700 launches.
     `fast_ms / captured_ms` = the launch-bound speedup. On non-NVIDIA the
     capture is a no-op (closure runs eagerly) → captured ≈ fast (transparency).
  3. DIAG path: a few × `train_step(want_diag=True)` — adds the host download of
     the imagination histories (only done at log/eval cadence in the real run);
     timed separately so you can see the readout's per-step host cost. Capture
     does NOT apply here (want_diag steps stay eager — the readout can't be
     captured).

How to read it WITH nsys:
  • The printed `fast path: ... ms/train_step` = WALL per train_step.
  • nsys "CUDA GPU Kernel Summary" → sum Total-Time = GPU-BUSY time over the
    window. GPU-busy ≈ wall ⇒ real compute (the RSSM/dec/rew/con + AC forwards/
    vjps dominate — optimize kernels). GPU-busy ≪ wall ⇒ launch/host-bound (many
    tiny sequential kernels across the T=16 BPTT scan + T_IMAG=15 imagination
    scan — the thing Stages 1–2 attacked; remaining gap ⇒ CUDA-graph capture).
  • `nsys stats --report cuda_gpu_trace <rep>` → kernel gaps = GPU idle between
    launches (host-bound signal).

Run + capture (NVIDIA):
    pixi run -e nvidia nsys profile --trace=cuda,osrt --stats=true \\
        -o dreamerv3_cartpole_train_profile \\
        mojo run -I . examples/cartpole/cartpole_dreamerv3_profile_gpu.mojo

Apple (parity / local wall only; nsys is NVIDIA):
    pixi run -e apple mojo run -I . examples/cartpole/cartpole_dreamerv3_profile_gpu.mojo
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.envs.cartpole import CartPoleEnv


# ── real per-train_step dims (UNCHANGED from cartpole_dreamerv3_training_gpu) ──
comptime EnvT = CartPoleEnv[DT]
comptime OBS = 4
comptime ACT = 2
comptime DETER = 128
comptime H = 32
comptime STOCH = 16
comptime CLASSES = 4
comptime BLOCKS = 4
comptime TOKEN = 32
comptime DEC_U = 32
comptime HU = 32
comptime VU = 32
comptime PU = 32
comptime BINS = 51
comptime B = 16
comptime T = 16
comptime T_IMAG = 15

# ── profiling-only knobs (don't affect per-train_step kernel shape) ──
comptime CAP = 8192            # tiny device replay ring → fast startup / low VRAM
comptime LEARN_START = 256     # buffer fills in ~LEARN_START env steps
comptime WARMUP_ENV = 512      # env steps to record before timing (> LEARN_START)
comptime FAST_STEPS = 300      # timed train_step(want_diag=False) calls (nsys this)
comptime CAP_WARMUP = 5        # train_step_captured calls to settle+capture (untimed)
comptime DIAG_STEPS = 30       # timed train_step(want_diag=True) calls

comptime Ag = DreamerV3Agent[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, True,   # DISCRETE=True, train_target="gpu"
]


def _argmax(a: UnsafePointer[Scalar[DT], MutAnyOrigin]) -> Int:
    var k = 0
    var best = a[0]
    for i in range(1, ACT):
        if a[i] > best:
            best = a[i]
            k = i
    return k


def main() raises:
    seed(42)
    print("=" * 70)
    print("DreamerV3 CartPole — TRAINING-PHASE profile (GPU)")
    print("=" * 70)
    print("  real dims: B", B, "T", T, "T_IMAG", T_IMAG, "DETER", DETER,
          "STOCH", STOCH, "CLASSES", CLASSES, "BINS", BINS)
    print("  profiling: CAP", CAP, "| LEARN_START", LEARN_START,
          "| FAST_STEPS", FAST_STEPS, "| CAP_WARMUP", CAP_WARMUP,
          "| DIAG_STEPS", DIAG_STEPS)
    print("=" * 70)

    # bare ctx (not `with`) — avoids benign teardown crash-reporter noise.
    var ctx = DeviceContext()
    var env = EnvT()
    var ag = Ag.make(
        ctx=ctx,
        lr=Scalar[DT](1.5e-4), learning_starts=LEARN_START,
        warmup_steps=0, out_init_scale=Scalar[DT](1.0),
    )

    var obsbuf = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var actbuf = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()
    var obs = env.reset_obs_list()
    ag.reset_belief()

    # ─── Warmup (UNTIMED): fill the replay ring with random one-hot rollouts ──
    print("Warmup:", WARMUP_ENV, "env steps (untimed) to fill the buffer...")
    for _step in range(WARMUP_ENV):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        var idx = Int(random_float64() * Float64(ACT))
        if idx >= ACT:
            idx = ACT - 1
        for a in range(ACT):
            actbuf[a] = Scalar[DT](1.0) if a == idx else Scalar[DT](0.0)
        var res = env.step_obs(idx)
        ag.record(obsbuf, actbuf, res[1], Scalar[DT](1.0) if res[2] else Scalar[DT](0.0))
        obs = res[0].copy()
        if res[2]:
            for i in range(OBS):
                obsbuf[i] = res[0][i]
            ag.record_terminal(obsbuf)
            obs = env.reset_obs_list()
            ag.reset_belief()
    ctx.synchronize()

    # ─── FAST path (TIMED): pure train_step(want_diag=False) — nsys THIS ─────
    print("-" * 70)
    print("FAST path:", FAST_STEPS, "x train_step(want_diag=False) ...")
    var t0 = perf_counter_ns()
    for _i in range(FAST_STEPS):
        _ = ag.train_step(want_diag=False)
    ctx.synchronize()
    var fast_s = Float64(perf_counter_ns() - t0) / 1e9
    var fast_ms = fast_s * 1e3 / Float64(FAST_STEPS)
    print("  total", fast_s, "s  ->", fast_ms, "ms/train_step  (",
          Float64(FAST_STEPS) / fast_s, "steps/s )")

    # ─── CAPTURED path (TIMED): train_step_captured — CUDA-graph replay ──────
    # First CAP_WARMUP calls capture the WM+AC device-kernel sequence (settle +
    # record, slower) — UNTIMED; the timed window is pure replay. On non-NVIDIA
    # `maybe_capture_replay` runs the closure eagerly (no-op capture).
    print("-" * 70)
    print("Capture warmup:", CAP_WARMUP, "x train_step_captured (untimed)...")
    for _i in range(CAP_WARMUP):
        _ = ag.trainer.train_step_captured(want_diag=False)
    ctx.synchronize()
    print("CAPTURED path:", FAST_STEPS,
          "x train_step_captured(want_diag=False) [replay] ...")
    var tc = perf_counter_ns()
    for _i in range(FAST_STEPS):
        _ = ag.trainer.train_step_captured(want_diag=False)
    ctx.synchronize()
    var cap_s = Float64(perf_counter_ns() - tc) / 1e9
    var cap_ms = cap_s * 1e3 / Float64(FAST_STEPS)
    print("  total", cap_s, "s  ->", cap_ms, "ms/train_step  (",
          Float64(FAST_STEPS) / cap_s, "steps/s )")

    # ─── DIAG path (TIMED): train_step(want_diag=True) — adds host readout ───
    print("-" * 70)
    print("DIAG path:", DIAG_STEPS, "x train_step(want_diag=True) ...")
    var t1 = perf_counter_ns()
    for _i in range(DIAG_STEPS):
        _ = ag.train_step(want_diag=True)
    ctx.synchronize()
    var diag_s = Float64(perf_counter_ns() - t1) / 1e9
    var diag_ms = diag_s * 1e3 / Float64(DIAG_STEPS)
    print("  total", diag_s, "s  ->", diag_ms, "ms/train_step  (",
          Float64(DIAG_STEPS) / diag_s, "steps/s )")

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("-" * 70)
    print("=" * 70)
    print("Profile complete")
    print("  fast      (eager,   want_diag=False) =", fast_ms, "ms/train_step")
    print("  captured  (replay,  want_diag=False) =", cap_ms, "ms/train_step")
    var speedup = fast_ms / cap_ms if cap_ms > 0.0 else 0.0
    print("  capture speedup (fast/captured)      =", speedup, "x")
    print("  diag      (eager,   want_diag=True ) =", diag_ms, "ms/train_step")
    print("  diag readout overhead                =", diag_ms - fast_ms, "ms/train_step")
    print("  last WM / AC loss                    =", ag.last_wm_loss(), "/", ag.last_ac_loss())
    print("=" * 70)
    obsbuf.free(); actbuf.free()
    _ = env^
    _ = ag^
