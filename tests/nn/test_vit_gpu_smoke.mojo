"""GPU forward + backward + optimizer-step smoke test for the ViT composite.

Tiny config — fits on M1 Pro. Validates that the full ViT GPU pipeline
(Conv2D → Transpose2DOp → BiasAdd → TransformerBlock × N → LayerNorm →
TokenMean → Linear) runs without crashing and produces non-NaN outputs.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_vit_gpu_smoke.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.random import seed
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.composites import ViT
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import AdamW
from mojo_rl.nn.initializer import Xavier
from layout import Layout, LayoutTensor


def check(cond: Bool, msg: String, mut fails: Int):
    if cond:
        print("  PASS: " + msg)
    else:
        print("  FAIL: " + msg)
        fails += 1


def main() raises:
    seed(7)
    var fails = 0

    # Tiny ViT: 8×8 image, patch=4, 4 patches, dim=16, 1 layer, 5 classes.
    comptime IC = 3
    comptime IMG = 8
    comptime PATCH = 4
    comptime D = 16
    comptime H = 4
    comptime N = 1
    comptime NP = (IMG // PATCH) * (IMG // PATCH)  # 4
    comptime NCLS = 5
    comptime BATCH = 2
    comptime Model = ViT[IC, IMG, IMG, PATCH, D, H, N, NP, NCLS]
    comptime Opt = AdamW[3e-4, 0.9, 0.999, 1e-8, 0.05]

    print("=" * 70)
    print("ViT GPU smoke: forward + backward + optimizer step")
    print("=" * 70)
    print("  IC=" + String(IC) + " IMG=" + String(IMG) + " PATCH=" + String(PATCH) + " NP=" + String(NP))
    print("  D=" + String(D) + " H=" + String(H) + " N=" + String(N) + " NCLS=" + String(NCLS))
    print("  PARAM_SIZE=" + String(Model.PARAM_SIZE) + " CACHE/sample=" + String(Model.CACHE_SIZE) + " WS/sample=" + String(Model.WORKSPACE_SIZE_PER_SAMPLE))

    var ctx = DeviceContext()
    var state = GPUNetworkState[Model, Opt](ctx)
    var cpu = NetworkState[Model, Opt]()
    cpu.initialize[Xavier[]]()
    state.upload_from(cpu, ctx)

    # ---------- Random input + one-hot target ----------
    var inp_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Model.IN_DIM)
    for i in range(BATCH * Model.IN_DIM):
        inp_host[i] = Scalar[dtype](Float32(0.1))   # cheap placeholder data
    var tgt_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NCLS)
    for i in range(BATCH * NCLS):
        tgt_host[i] = 0
    # First sample: class 0; second: class 2.
    tgt_host[0 * NCLS + 0] = 1
    tgt_host[1 * NCLS + 2] = 1

    var inp_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.IN_DIM)
    var tgt_dev = ctx.enqueue_create_buffer[dtype](BATCH * NCLS)
    ctx.enqueue_copy(inp_dev, inp_host)
    ctx.enqueue_copy(tgt_dev, tgt_host)

    var out_dev = ctx.enqueue_create_buffer[dtype](BATCH * NCLS)
    var cache_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.CACHE_SIZE)
    var gin_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.IN_DIM)
    var gout_dev = ctx.enqueue_create_buffer[dtype](BATCH * NCLS)
    var ws_dev = ctx.enqueue_create_buffer[dtype](
        max(1, BATCH * Model.WORKSPACE_SIZE_PER_SAMPLE)
    )

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](inp_dev.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, NCLS), MutAnyOrigin
    ](out_dev.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
    ](cache_dev.unsafe_ptr())
    var gin_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](gin_dev.unsafe_ptr())
    var gout_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, NCLS), MutAnyOrigin
    ](gout_dev.unsafe_ptr())

    # ---------- Forward ----------
    print("\n[1/3] forward_gpu...")
    var p_view = state.params_view()
    var s_view = state.model_state_view()
    Model.forward_gpu[BATCH, dtype](
        ctx, out_t, inp_t, p_view, s_view, cache_t, ws_dev
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NCLS)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var has_nan = False
    var any_nonzero = False
    for i in range(BATCH * NCLS):
        var v = Float64(out_host[i])
        if v != v:
            has_nan = True
        if math_abs(v) > 1e-9:
            any_nonzero = True
    check(not has_nan, "GPU forward produced no NaN", fails)
    check(any_nonzero, "GPU forward output is non-trivial", fails)

    # ---------- Mock CE-style grad: out - target ----------
    var tgt_h2 = ctx.enqueue_create_host_buffer[dtype](BATCH * NCLS)
    var gout_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NCLS)
    ctx.enqueue_copy(tgt_h2, tgt_dev)
    ctx.synchronize()
    for i in range(BATCH * NCLS):
        gout_host[i] = out_host[i] - tgt_h2[i]
    ctx.enqueue_copy(gout_dev, gout_host)
    ctx.synchronize()

    # ---------- Backward ----------
    print("\n[2/3] backward_gpu...")
    state.zero_grads(ctx)
    var p_view2 = state.params_view()
    var s_view2 = state.model_state_view()
    var grads_view = state.grads_view()
    Model.backward_gpu[BATCH, dtype](
        ctx, gin_t, gout_t, p_view2, s_view2, cache_t, grads_view, ws_dev
    )
    ctx.synchronize()

    var gp_host = ctx.enqueue_create_host_buffer[dtype](Model.PARAM_SIZE)
    ctx.enqueue_copy(gp_host, state.grads_buf)
    ctx.synchronize()
    var gp_nan = False
    var gp_nonzero = False
    for i in range(Model.PARAM_SIZE):
        var v = Float64(gp_host[i])
        if v != v:
            gp_nan = True
        if math_abs(v) > 1e-9:
            gp_nonzero = True
    check(not gp_nan, "param gradients contain no NaN", fails)
    check(gp_nonzero, "param gradients are non-trivial", fails)

    # ---------- Optimizer step ----------
    print("\n[3/3] optimizer_step...")
    state.optimizer_step(ctx)
    ctx.synchronize()

    var p_after = ctx.enqueue_create_host_buffer[dtype](Model.PARAM_SIZE)
    ctx.enqueue_copy(p_after, state.params_buf)
    ctx.synchronize()
    var p_before = ctx.enqueue_create_host_buffer[dtype](Model.PARAM_SIZE)
    cpu.initialize[Xavier[]]()
    for i in range(Model.PARAM_SIZE):
        p_before[i] = (cpu.params + i)[]
    var max_dp: Float64 = 0
    for i in range(Model.PARAM_SIZE):
        var d = math_abs(Float64(p_after[i]) - Float64(p_before[i]))
        if d > max_dp:
            max_dp = d
    check(max_dp > 1e-7, "params changed after optimizer step (max |Δp| = " + String(max_dp) + ")", fails)

    print("\n" + "=" * 70)
    if fails == 0:
        print("ALL VIT GPU SMOKE TESTS PASSED")
    else:
        print("FAILED: " + String(fails) + " checks")
    print("=" * 70)
