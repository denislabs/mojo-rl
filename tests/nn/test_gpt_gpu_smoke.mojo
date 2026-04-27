"""GPU forward + backward smoke test for the GPT composite.

Goal: verify the existing forward_gpu / backward_gpu paths through every
component of the GPT (Embedding → BiasAdd → TransformerBlock × N → LayerNorm
→ Linear) wire up without crashing, and that the result is non-NaN.

Note: the current ScaledDotProductAttention.eval_gpu / vjp_gpu fall back to
the CPU implementation. On Apple Silicon (unified memory) this works
transparently with device pointers; on NVIDIA it would crash. This test is
the early-warning that flags either case before we invest in a full GPU
training script.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_gpt_gpu_smoke.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.random import seed, random_float64
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.composites import GPT
from mojo_rl.nn.training import GPUNetworkState
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

    # Tiny GPT — V=8, S=4, D=8, H=2, N=1.
    comptime V = 8
    comptime S = 4
    comptime D = 8
    comptime H = 2
    comptime N = 1
    comptime BATCH = 2
    comptime Model = GPT[V, S, D, H, N]
    comptime Opt = AdamW[3e-4, 0.9, 0.95, 1e-8, 0.1]

    print("=" * 70)
    print("GPT GPU smoke: forward + backward + optimizer step")
    print("=" * 70)
    print("  V=" + String(V) + " S=" + String(S) + " D=" + String(D) + " H=" + String(H) + " N=" + String(N))
    print("  PARAM_SIZE=" + String(Model.PARAM_SIZE) + " CACHE_SIZE/sample=" + String(Model.CACHE_SIZE) + " WS/sample=" + String(Model.WORKSPACE_SIZE_PER_SAMPLE))

    var ctx = DeviceContext()
    var state = GPUNetworkState[Model, Opt](ctx)

    # Initialize via a transient CPU NetworkState then upload (matches Trainer.init_state_gpu).
    from mojo_rl.nn.training import NetworkState
    var cpu = NetworkState[Model, Opt]()
    cpu.initialize[Xavier[]]()
    state.upload_from(cpu, ctx)

    # ---------- Build a fake one-hot input on device ----------
    # Pick token 0 at every position (simplest valid input).
    var inp_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Model.IN_DIM)
    for i in range(BATCH * Model.IN_DIM):
        inp_host[i] = 0
    for b in range(BATCH):
        for t in range(S):
            inp_host[b * S * V + t * V + 0] = 1
    var inp_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.IN_DIM)
    ctx.enqueue_copy(inp_dev, inp_host)

    # Target = same as input (just a smoke check).
    var tgt_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.OUT_DIM)
    ctx.enqueue_copy(tgt_dev, inp_host)

    # Output, cache, grad_in, grad_out, workspace
    var out_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.OUT_DIM)
    var cache_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.CACHE_SIZE)
    var gin_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.IN_DIM)
    var gout_dev = ctx.enqueue_create_buffer[dtype](BATCH * Model.OUT_DIM)
    var ws_size = max(1, BATCH * Model.WORKSPACE_SIZE_PER_SAMPLE)
    var ws_dev = ctx.enqueue_create_buffer[dtype](ws_size)
    ctx.enqueue_memset(out_dev, 0)
    ctx.enqueue_memset(cache_dev, 0)
    ctx.enqueue_memset(gin_dev, 0)
    ctx.enqueue_memset(gout_dev, 0)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](inp_dev.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](out_dev.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
    ](cache_dev.unsafe_ptr())
    var gin_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](gin_dev.unsafe_ptr())
    var gout_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](gout_dev.unsafe_ptr())

    # ---------- Forward ----------
    print("\n[1/3] forward_gpu...")
    var p_view = state.params_view()
    var s_view = state.model_state_view()
    Model.forward_gpu[BATCH, dtype](
        ctx, out_t, inp_t, p_view, s_view, cache_t, ws_dev
    )
    ctx.synchronize()

    # Copy output back, check no NaN.
    var out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Model.OUT_DIM)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var has_nan = False
    var any_nonzero = False
    for i in range(BATCH * Model.OUT_DIM):
        var v = Float64(out_host[i])
        if v != v:
            has_nan = True
        if math_abs(v) > 1e-9:
            any_nonzero = True
    check(not has_nan, "GPU forward produced no NaN", fails)
    check(any_nonzero, "GPU forward output is non-trivial", fails)

    # ---------- Set grad_output to (out - target) (mock CE-style grad) ----------
    var gout_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Model.OUT_DIM)
    var tgt_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Model.OUT_DIM)
    ctx.enqueue_copy(tgt_host, tgt_dev)
    ctx.synchronize()
    for i in range(BATCH * Model.OUT_DIM):
        gout_host[i] = out_host[i] - tgt_host[i]
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

    # Sample some param gradients — should be non-trivial.
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

    # Verify params changed by a tiny amount (LR=3e-4 * grad).
    var p_host_after = ctx.enqueue_create_host_buffer[dtype](Model.PARAM_SIZE)
    ctx.enqueue_copy(p_host_after, state.params_buf)
    ctx.synchronize()

    var p_host_before = ctx.enqueue_create_host_buffer[dtype](Model.PARAM_SIZE)
    cpu.initialize[Xavier[]]()  # re-initialize same seed → same init values
    for i in range(Model.PARAM_SIZE):
        p_host_before[i] = (cpu.params + i)[]

    var max_dp: Float64 = 0.0
    for i in range(Model.PARAM_SIZE):
        var d = math_abs(Float64(p_host_after[i]) - Float64(p_host_before[i]))
        if d > max_dp:
            max_dp = d
    check(
        max_dp > 1e-7,
        "params changed after optimizer step (max |Δp| = " + String(max_dp) + ")",
        fails,
    )

    print("\n" + "=" * 70)
    if fails == 0:
        print("ALL GPT GPU SMOKE TESTS PASSED")
    else:
        print("FAILED: " + String(fails) + " checks")
    print("=" * 70)
