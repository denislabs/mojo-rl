"""DreamerV3 pixel-CarRacing GPU train-step PROFILING harness (nsys).

Isolates the GPU train_step (WM-BPTT + imagination AC) so nsys can chase the
heavy kernels / the bottleneck — WITHOUT the slow CPU-env warmup. Instead of
stepping the real env, it synthetically PRE-FILLS the device replay (one obs
pattern reused — kernel timings don't depend on obs values), then hammers
`train_step` in a tight loop. The kernel mix of a DreamerV3 train step is
step-invariant, so a few hundred steps profile the SAME bottlenecks you'd see at
step 500k–1000k.

Arch config mirrors `dreamerv3_car_racing_pixel_training.mojo` (BASE/DETER/B/T/
T_IMAG/OBS) so the profiled kernels match production; only CAP is smaller (replay
size doesn't affect kernel shapes) and there's no LR warmup (`warmup_steps=0`) so
the captured-graph path engages immediately.

`USE_TRAIN_CUDA_GRAPH` (comptime):
  * False (default): EAGER — every WM/AC kernel shows up individually in the nsys
    timeline → use this to find the heavy kernels / the bottleneck.
  * True: capture-replay the WM+AC sequence (one CUDA graph) → use this to measure
    the launch-overhead win of capture vs eager.

Profile (NVIDIA):
    nsys profile -o dv3_carracing_profile --trace=cuda,osrt \\
        --capture-range=cudaProfilerApi --capture-range-end=stop \\
        pixi run -e nvidia mojo run -I . \\
        examples/car_racing/dreamerv3_car_racing_pixel_profile.mojo
  (Or without --capture-range to profile the whole short run; the SKIP warmup
   train steps below run before the timed region so first-step lazy allocs /
   graph capture don't pollute the measured kernels.)
  Then: nsys stats dv3_carracing_profile.nsys-rep   # top CUDA kernels by time
"""

from std.memory import alloc
from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.dreamerv3.trainer import DreamerV3Trainer
from mojo_rl.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNN,
    DreamerDecoderCNN,
)

# ── arch (mirrors the training example) ──
comptime C = 4
comptime IMG = 96
comptime BASE = 48
comptime OBS = C * IMG * IMG  # 36864
comptime ACT = 3
comptime DETER = 2048  # mirror the training config (representative profiling)
comptime H = 256
comptime STOCH = 32
comptime CLASSES = 32
comptime BLOCKS = 8
comptime TOKEN = 1024
comptime DEC_U = 1024
comptime HU = 256
comptime VU = 256
comptime PU = 256
comptime BINS = 255
comptime B = 16
comptime T = 16
comptime T_IMAG = 15
comptime CAP = 4096  # replay size irrelevant to kernel shapes

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNN[C, IMG, IMG, BASE, TOKEN, SwishOp]
comptime DEC = DreamerDecoderCNN[FEATIN, C, IMG, IMG, BASE, SwishOp]

comptime Tr = DreamerV3Trainer[
    "gpu",
    OBS,
    ACT,
    DETER,
    H,
    STOCH,
    CLASSES,
    BLOCKS,
    TOKEN,
    DEC_U,
    HU,
    VU,
    PU,
    BINS,
    B,
    T,
    T_IMAG,
    CAP,
    False,
    ENC,
    DEC,  # DISCRETE=False (continuous)
]

# flip to True to profile the captured-graph path (vs per-kernel eager).
comptime USE_TRAIN_CUDA_GRAPH = True

comptime PREFILL = 1024  # synthetic transitions → replay sampleable
comptime LEARN_START = 512
comptime SKIP = 20  # warmup train steps BEFORE the timed region
comptime PROFILE_STEPS = 200  # measured / profiled hot loop


def main() raises:
    print("=" * 70)
    print(
        "DreamerV3 pixel-CarRacing train-step PROFILE — OBS",
        OBS,
        " B",
        B,
        " T",
        T,
        " T_IMAG",
        T_IMAG,
        " BASE",
        BASE,
    )
    print(
        "  USE_TRAIN_CUDA_GRAPH =",
        USE_TRAIN_CUDA_GRAPH,
        "  SKIP",
        SKIP,
        " PROFILE_STEPS",
        PROFILE_STEPS,
    )
    print("=" * 70)
    seed(42)
    var ctx = DeviceContext()
    var tr = Tr.make(
        ctx=ctx,
        lr=Scalar[DT](4e-5),
        learning_starts=LEARN_START,
        warmup_steps=0,  # no LR warmup → capture engages from the first step
    )

    # ── synthetic replay prefill (no env, no render) ──
    # One fixed obs pattern reused for every record (kernel timings are
    # value-independent); only done flags vary to create episode boundaries.
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    for k in range(OBS):
        obsbuf[k] = Scalar[DT](Float64(k % 17) / 17.0)
    for k in range(ACT):
        actbuf[k] = Scalar[DT](0.1)
    print("  prefilling", PREFILL, "synthetic transitions...")
    for t in range(PREFILL):
        var done = Scalar[DT](1.0) if (t + 1) % 128 == 0 else Scalar[DT](0.0)
        tr.record(obsbuf, actbuf, Scalar[DT](0.05), done)
    if not tr.can_train():
        raise Error("replay not trainable after prefill — raise PREFILL")

    # ── SKIP warmup train steps (lazy alloc + graph capture) — NOT profiled ──
    print("  warming up", SKIP, "train steps (lazy alloc + capture)...")
    for _i in range(SKIP):
        comptime if USE_TRAIN_CUDA_GRAPH:
            _ = tr.train_step_captured(want_diag=False)
        else:
            _ = tr.train_step(want_diag=False)
    ctx.synchronize()

    # ── PROFILED REGION: tight train_step loop (want_diag=False = hot path) ──
    print("  profiling", PROFILE_STEPS, "train steps...")
    var t0 = perf_counter_ns()
    for _i in range(PROFILE_STEPS):
        comptime if USE_TRAIN_CUDA_GRAPH:
            _ = tr.train_step_captured(want_diag=False)
        else:
            _ = tr.train_step(want_diag=False)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9

    obsbuf.free()
    actbuf.free()
    print("=" * 70)
    print(
        "  ",
        PROFILE_STEPS,
        "train steps in",
        dt,
        "s  →",
        Float64(PROFILE_STEPS) / dt,
        "train_steps/s",
        " (",
        dt / Float64(PROFILE_STEPS) * 1e3,
        "ms/step )",
    )
    print("=" * 70)
