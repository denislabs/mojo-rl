# +--------------------------------------------------------------------------+ #
# | ACT on the SO-ARM101 — short GPU run for nsys profiling
# +--------------------------------------------------------------------------+ #
"""Why does one ACT training step cost ~176 ms on a 5090?

Same model, batch and data as `act_so101_train_gpu.mojo` — 60 steps instead of
100,000, no validation, no checkpoints, no logger, so an nsys timeline is the
training step and nothing else.

    pixi run -e nvidia nsys profile --stats=true mojo run -I . \\
        examples/so101/act_so101_profile_gpu.mojo

⚠ Run from the project root: `mojo_rl/io/hdf5` resolves libhdf5 relative to the
working directory. `ACT_STORE` selects the dataset, as everywhere else.

## What is already known, so nobody re-measures it

    total step (RTX 5090, K=60 dim=256 ff=1024 enc=4 batch 16)   ~176 ms
    sample_batch, host side, measured with no model in-process    ~19 ms

`sample_batch` is single-threaded, serial with the device, and converts 7.4M
uint8 to float32 one element at a time — the obvious suspect, and it is **11%**.
The other ~90% is device compute. Do not start by optimizing the data path.

## The knobs, and what each one isolates

Each is a comptime flag; flip ONE and re-profile. The script prints a host-side
breakdown at the end whatever nsys does, so a first pass needs no profiler at
all.

`SKIP_RESAMPLE` — reuse one batch for every step instead of drawing a new one.
    The difference is `sample_batch` exactly: HDF5 row reads plus the
    normalization. Confirms the 11% above on the box that matters, since that
    number came from an M1 Pro.

`STUB_BACKBONE` — swap ResNet18 (20 Conv2D + 20 BatchNorm2D) for five stride-2
    convs that reproduce its 32x downsampling EXACTLY, so the transformer sees
    an identical 162-token memory and the only thing that changes is the
    convolution work. Splits the step into vision and everything-else. If the
    remainder is still large, the transformer stacks and the optimizer are
    where the time is; if it collapses, the backbone is, and
    `project_conv2d_kernel_optimization` is the thread to pull (hand GEMMs lose
    to `max_matmul`, and Apple's answer INVERTS NVIDIA's).

    ⚠ A stub with the WRONG downsampling factor does not isolate anything. The
    2-conv version from `test_act_gpu_vs_cpu.mojo` downsamples by 4, which at
    240x320 is 9602 memory tokens instead of 162 and asks for **47.2 GB** of
    attention scores — refused by Metal, and it would be refused by a 32 GB
    5090 too. A startup check now raises before anything is allocated if the
    stub's feature-map size ever stops matching ResNet18's.

`CLIP_NORM` — non-zero re-enables gradient clipping, which walks every gradient
    slab through the host (`trainer.mojo:_SumSq`): a device sync per parameter.
    Training runs it at 0.0; this measures what that decision is worth.

## What the host breakdown can and cannot tell you

`eval_step` is forward-only and `train_step` is forward + backward + optimizer,
on the same graph and the same batch. Their difference is therefore backward +
optimizer, which is the one decomposition available without instrumenting the
trainer. It is a HOST wall-clock difference over device-synchronous calls, so
treat it as an attribution hint and let nsys settle the per-kernel truth.

Two performance gaps are already documented in `docs/ACT_PORT.md` and are worth
looking for in the timeline before hunting anything new:

  * **Adam's grouped arena is not engaged.** `opt.adopt` requires a `Module` and
    a `ComputeGraph` is not one, so the optimizer walks `for_each_param` with
    per-parameter kernels — hundreds of tiny launches per step.
  * **Conv kernel shape on NVIDIA.**
"""

from std.time import perf_counter_ns
from std.python import Python, PythonObject

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn.models.resnet18 import (
    RESNET18_OUT_CH,
    ResNet18Backbone,
    ResNet18OutH,
    ResNet18OutW,
)
from mojo_rl.deep_agents.act.config import (
    SO101_ADIM,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from mojo_rl.deep_agents.act.data import ACTDataset
from mojo_rl.deep_agents.act.trainer import ACTTrainer


# ─── Profiling knobs ──────────────────────────────────────────────────────
comptime SKIP_RESAMPLE = False
comptime STUB_BACKBONE = False
comptime CLIP_NORM = 0.0

comptime WARMUP_STEPS = 5
"""Enough to get past first-launch kernel compilation and the initial H2D, and
few enough that the profile is dominated by steady state. The training example
reported 1.38 s for its first 'step' precisely because it counted this."""
comptime PROFILE_STEPS = 60

# ─── Sizing (mirrors act_so101_train_gpu.mojo exactly) ────────────────────
comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W

comptime K = 60
comptime DIM = 256
comptime HEADS = 8
comptime FF = 1024
comptime LATENT = 32
comptime N_ENC = 4
comptime N_DEC = 1
comptime BATCH = 16
comptime LR = 1e-4
comptime KL_WEIGHT = 10.0

comptime IMG_ELEMS = N_CAM * 3 * IMG_H * IMG_W

# ⚠ THE STUB MUST PRESERVE THE TOKEN COUNT, or it does not isolate the
# backbone — it changes the transformer underneath it.
#
# ResNet18 downsamples by 32: 240x320 -> 8x10, so 80 tokens per camera and 162
# memory tokens. `test_act_gpu_vs_cpu.mojo`'s stub is TWO stride-2 convs, a
# factor of 4, and it is correct there only because that gate runs at 64x64.
# Reused here it gives 60x80 = 4800 tokens per camera, 9602 memory tokens, and
# self-attention is O(N^2):
#
#     9602^2 x 8 heads x 16 batch x 4 B = 47.2 GB of attention scores
#
# which is exactly the allocation Metal refused. It would have refused on a
# 32 GB 5090 too, one profiling session in.
#
# FIVE stride-2 convs reproduce ResNet18's factor of 32 exactly, so the
# transformer sees an identical memory and the only thing the knob changes is
# the convolution work. `main` checks that before allocating anything, so this
# is enforced rather than merely written down.
comptime STUB_CH = 8
comptime _C[H: Int] = (H + 2 * 1 - 3) // 2 + 1
comptime _C2[H: Int] = _C[_C[H]]
comptime _C4[H: Int] = _C2[_C2[H]]
comptime _C5[H: Int] = _C[_C4[H]]

comptime Stub = Sequential[
    Conv2DBatchNormReLU[3, STUB_CH, 3, 2, 1, IMG_H, IMG_W],
    Conv2DBatchNormReLU[
        STUB_CH, STUB_CH, 3, 2, 1, _C[IMG_H], _C[IMG_W]
    ],
    Conv2DBatchNormReLU[
        STUB_CH, STUB_CH, 3, 2, 1, _C2[IMG_H], _C2[IMG_W]
    ],
    Conv2DBatchNormReLU[
        STUB_CH, STUB_CH, 3, 2, 1, _C[_C2[IMG_H]], _C[_C2[IMG_W]]
    ],
    Conv2DBatchNormReLU[
        STUB_CH, STUB_CH, 3, 2, 1, _C4[IMG_H], _C4[IMG_W]
    ],
]

comptime FEAT_CH = STUB_CH if STUB_BACKBONE else RESNET18_OUT_CH
comptime OH = _C5[IMG_H] if STUB_BACKBONE else ResNet18OutH[IMG_H]
comptime OW = _C5[IMG_W] if STUB_BACKBONE else ResNet18OutW[IMG_W]
comptime BACKBONE = Stub if STUB_BACKBONE else ResNet18Backbone[
    3, IMG_H, IMG_W
]

comptime T = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH, 0.1, "gpu", FEAT_CH, OH, OW, BACKBONE,
]


def store_path() raises -> String:
    var os = Python.import_module("os")
    var env = String(
        os.environ.get(PythonObject("ACT_STORE"), PythonObject(""))
    )
    if env.byte_length() > 0:
        return env
    var home = String(os.path.expanduser(PythonObject("~")))
    return (
        home
        + "/.cache/mojo_rl/act_so101/"
        + "DenisLabs__record-test_20260828_092736_"
        + String(IMG_H) + "x" + String(IMG_W) + ".h5"
    )


def main() raises:
    var path = store_path()
    var os = Python.import_module("os")
    if not Bool(os.path.exists(PythonObject(path))):
        print("MISSING STORE: " + path)
        raise Error("store not found")

    # Checked before anything is allocated. `constrained` does not exist in
    # this Mojo, so this is a startup raise rather than a build failure — it
    # still fires in the first millisecond, which is what matters for a knob
    # whose failure mode is a 47 GB allocation.
    comptime if STUB_BACKBONE:
        if (
            OH != ResNet18OutH[IMG_H] or OW != ResNet18OutW[IMG_W]
        ):
            raise Error(
                "STUB_BACKBONE produces "
                + String(OH) + "x" + String(OW) + " but ResNet18 produces "
                + String(ResNet18OutH[IMG_H]) + "x"
                + String(ResNet18OutW[IMG_W])
                + " — the knob would change the transformer's O(N^2) memory"
                " instead of isolating the backbone. See the comment at `Stub`."
            )

    var ctx = DeviceContext()
    print("=== ACT SO-ARM101 nsys profile ===")
    print("  device            " + String(ctx.name()))
    print(
        "  model             K=" + String(K) + " dim=" + String(DIM)
        + " ff=" + String(FF) + " enc=" + String(N_ENC)
        + " dec=" + String(N_DEC) + " batch=" + String(BATCH)
    )
    comptime if STUB_BACKBONE:
        print("  backbone          STUB (5 strided convs, ResNet18 token count)")
    else:
        print("  backbone          ResNet18")
    print("  SKIP_RESAMPLE     " + String(SKIP_RESAMPLE))
    print("  CLIP_NORM         " + String(CLIP_NORM))
    print(
        "  steps             " + String(PROFILE_STEPS) + " after "
        + String(WARMUP_STEPS) + " warmup"
    )

    var ds = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W](String(path), seed=7)
    print(
        "  images            "
        + ("resident" if ds.images_resident else "streamed from HDF5")
    )
    print("")

    var tr = T.make(
        lr=Scalar[DT](LR),
        kl_weight=Scalar[DT](KL_WEIGHT),
        max_grad_norm=Scalar[DT](CLIP_NORM),
        ctx=ctx,
    )

    var qpos = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var images = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var actions = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var valid = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)

    # One batch drawn up front: under SKIP_RESAMPLE it is the only one, and
    # otherwise it still gives the warmup something to run on.
    ds.sample_batch[K, BATCH](False, qpos, images, actions, valid)

    for _ in range(WARMUP_STEPS):
        _ = tr.train_step(qpos, images, actions, valid)

    var data_ns = 0
    var train_ns = 0
    var eval_ns = 0

    for _ in range(PROFILE_STEPS):
        var t0 = perf_counter_ns()
        comptime if not SKIP_RESAMPLE:
            ds.sample_batch[K, BATCH](False, qpos, images, actions, valid)
        var t1 = perf_counter_ns()
        _ = tr.train_step(qpos, images, actions, valid)
        var t2 = perf_counter_ns()
        # Forward-only on the SAME batch, so the difference from `train_step`
        # is backward + optimizer. Its own cost is counted separately and is
        # not part of the step total below.
        _ = tr.eval_step(qpos, images, actions, valid)
        var t3 = perf_counter_ns()

        data_ns += t1 - t0
        train_ns += t2 - t1
        eval_ns += t3 - t2

    var n = Float64(PROFILE_STEPS)
    var data_ms = Float64(data_ns) / 1e6 / n
    var train_ms = Float64(train_ns) / 1e6 / n
    var eval_ms = Float64(eval_ns) / 1e6 / n
    var step_ms = data_ms + train_ms

    print("  per step, mean over " + String(PROFILE_STEPS) + ":")
    print(
        "    sample_batch    " + String(data_ms) + " ms  ("
        + String(Int(100.0 * data_ms / (step_ms + 1e-12))) + "%)"
    )
    print(
        "    train_step      " + String(train_ms) + " ms  ("
        + String(Int(100.0 * train_ms / (step_ms + 1e-12))) + "%)"
    )
    print("    ---- step       " + String(step_ms) + " ms")
    print("")
    print(
        "    eval_step       " + String(eval_ms)
        + " ms   (forward only, same batch)"
    )
    print(
        "    bwd + optimizer " + String(train_ms - eval_ms)
        + " ms   (train_step - eval_step; a host-side"
    )
    print(
        "                     difference over device-synchronous calls — an"
        " attribution"
    )
    print("                     hint, not a kernel measurement)")
    print("")
    print("=== Done ===")
