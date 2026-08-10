"""FB CUDA-graph capture safety — the RNG stream must ADVANCE.

A captured train step replays a fixed kernel sequence with its scalar arguments
frozen at capture time. Every host value passed by value into a kernel is
therefore baked in, and the one that matters here is the Philox offset for the
TD3 target-smoothing noise: with `gaussian_t`'s `offset: UInt64` the graph would
redraw the IDENTICAL noise on every replay, for the rest of the run.

⚠⚠ That failure is SILENT. Frozen target-smoothing noise is still a valid (if
degenerate) regulariser, so the loss curve descends and `|B|` stays pinned —
nothing raises, nothing looks wrong. It is the same shape as the
`USE_ENV_CUDA_GRAPH` regression that captured a physics step and left every GPU
example training against a stopped simulator, undetected because the reward
curve still moved.

So this gates the mechanism directly rather than trusting it:

  [1] Two successive device-offset draws DIFFER. If the kernel that bumps the
      offset were missing or ran before the draw, they would be identical.
  [2] The device offset buffer actually advances, by the exact amount the host
      path used (`N + N % 2`) — so a captured run and an eager run walk the SAME
      Philox stream instead of diverging by a half-pair per step.
  [3] `train_device_kernels` moves the weights on every call, and moves them
      DIFFERENTLY on consecutive calls. A step whose second call reproduced the
      first would mean some per-step state stopped advancing.

⚠ What this CANNOT gate on Apple: the capture itself. `CUDAGraph` is a
compile-time no-op off NVIDIA, so `maybe_capture_replay` here simply runs the
body. This file proves the body is capture-SAFE; proving the capture works
needs an NVIDIA run of `examples/fb/fb_train_profile_gpu.mojo`.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_fb_cuda_graph_safety.mojo
"""

from std.math import abs
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh
from mojo_rl.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from mojo_rl.deep_agents.fb.trainer import FBTrainer
from mojo_rl.deep_agents.fb.kernels import gaussian_dev_t


comptime OBS = 8
comptime ACT = 3
comptime D = 16
comptime BATCH = 32
comptime HID = 32
comptime SEED = 20260810

comptime FNet = Sequential[Linear[OBS + ACT + D, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, HID], ReLU[HID], Linear[HID, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[OBS + D, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "gpu"]

comptime NOISE_N = 64


def _rt(n: Int, ctx: DeviceContext) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    t.upload(ctx)
    return t^


def test_device_offset_draws_differ() raises:
    """[1] + [2] — successive draws differ, and the offset advances exactly."""
    print("[1] two device-offset draws must DIFFER ...")
    var ctx = DeviceContext()

    var off = ctx.enqueue_create_buffer[DType.uint64](1)
    var oh = ctx.enqueue_create_host_buffer[DType.uint64](1)
    oh[0] = UInt64(0)
    ctx.enqueue_copy(off, oh)
    ctx.synchronize()

    var a = Tensor()
    var b = Tensor()
    gaussian_dev_t["gpu", NOISE_N](a, UInt64(SEED), off, Optional(ctx))
    a.download(ctx)
    gaussian_dev_t["gpu", NOISE_N](b, UInt64(SEED), off, Optional(ctx))
    b.download(ctx)

    var same = 0
    var worst = Float64(0)
    for i in range(NOISE_N):
        var d = abs(Float64(a.data[i]) - Float64(b.data[i]))
        if d < 1e-12:
            same += 1
        if d > worst:
            worst = d
    print("      identical elements:", same, "/", NOISE_N, " worst |Δ| =", worst)
    assert_true(
        same < NOISE_N // 4,
        "the two draws are largely IDENTICAL (" + String(same) + "/"
        + String(NOISE_N) + " elements match) — the Philox offset is not"
        " advancing, so a captured replay would redraw the same noise forever",
    )

    print("[2] the device offset advanced by exactly N + N%2 ...")
    ctx.enqueue_copy(oh, off)
    ctx.synchronize()
    comptime WANT = UInt64(2 * (NOISE_N + (NOISE_N % 2)))
    print("      offset =", oh[0], " want", WANT)
    assert_true(
        oh[0] == WANT,
        "offset is " + String(oh[0]) + " after two draws, expected "
        + String(WANT) + " — a captured run would then walk a DIFFERENT Philox"
        " stream than an eager one",
    )
    print("      OK")


def test_device_step_advances() raises:
    """[3] `train_device_kernels` moves weights, and differently each call."""
    print("[3] train_device_kernels advances state on every call ...")
    seed(SEED)
    var ctx = DeviceContext()
    var t = Trainer.make[Xavier](
        lr=1e-3, ctx=Optional(ctx), max_grad_norm=1.0, bc_weight=1.0
    )
    var s = _rt(BATCH * OBS, ctx)
    var a = _rt(BATCH * ACT, ctx)
    var sn = _rt(BATCH * OBS, ctx)
    var sp = _rt(BATCH * OBS, ctx)
    var z = _rt(BATCH * D, ctx)
    t.load_batch(s, a, sn, sp, z)

    var probe = _rt(BATCH * OBS, ctx)
    # ⚠ `backward_embed` computes into `dst` ON DEVICE and does not download.
    # Comparing `dst.data` without this reads a never-written host buffer, so
    # every diff is 0.0 and the test passes/fails for the wrong reason — it
    # FAILED that way first, which is the only reason it is called out here.
    var b0 = Tensor()
    t.backward_embed[BATCH](probe, b0)
    b0.download(ctx)

    t.train_device_kernels()
    var b1 = Tensor()
    t.backward_embed[BATCH](probe, b1)
    b1.download(ctx)

    t.train_device_kernels()
    var b2 = Tensor()
    t.backward_embed[BATCH](probe, b2)
    b2.download(ctx)

    var d01 = Float64(0)
    var d12 = Float64(0)
    for i in range(BATCH * D):
        var x = abs(Float64(b0.data[i]) - Float64(b1.data[i]))
        var y = abs(Float64(b1.data[i]) - Float64(b2.data[i]))
        if x > d01:
            d01 = x
        if y > d12:
            d12 = y
    print("      |B| moved: step1", d01, " step2", d12)
    assert_true(d01 > 1e-7, "first device step did not move B (" + String(d01) + ")")
    assert_true(
        d12 > 1e-7,
        "second device step did not move B (" + String(d12) + ") — per-step"
        " state has stopped advancing, which is what a frozen capture looks"
        " like",
    )
    print("      OK")


def main() raises:
    print("=" * 70)
    print("FB CUDA-graph capture safety")
    print("=" * 70)
    test_device_offset_draws_differ()
    test_device_step_advances()
    print("\n[PASS] FB capture safety")
    print("⚠ Capture/replay ITSELF is unverified here — CUDAGraph is a no-op")
    print("  off NVIDIA. Run fb_train_profile_gpu.mojo on the 4090 to gate it.")
