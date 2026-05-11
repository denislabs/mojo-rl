"""Gradient check, CPU/GPU parity, and BPTT accumulation for LSTMCell.

Tests:
  1. CPU finite-difference gradient check vs analytical backward for all
     inputs (x, h_prev, c_prev) and params.
  2. CPU vs GPU forward parity.
  3. CPU vs GPU backward parity (dx, dh_prev, dc_prev, grads).
  4. BPTT accumulation: 3-step unroll on CPU, manually compose backward
     across time steps, verify param grads accumulate (vs single-step
     overwrites) and equal the FD-derived gradient of the multi-step
     loss.

Usage:
    pixi run -e apple mojo run -I . tests/nn/test_lstm.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_lstm.mojo
"""

from std.memory import alloc, memset, UnsafePointer
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import LSTMCell


# =============================================================================
# Test 1: CPU gradient check
# =============================================================================


def test_cpu_gradcheck() raises:
    """Finite-difference vs analytical for x, h_prev, c_prev, params."""
    print("=" * 60)
    print("TEST: LSTMCell CPU gradient check (in=4, hidden=3, batch=2)")
    print("=" * 60)

    comptime IN = 4
    comptime H = 3
    comptime BATCH = 2
    comptime LC = LSTMCell[IN, H]
    comptime PS = LC.PARAM_SIZE
    comptime CS = LC.CACHE_SIZE

    # --- Inputs (deterministic) ---
    var x_data = alloc[Scalar[dtype]](BATCH * IN)
    for i in range(BATCH * IN):
        x_data[i] = Scalar[dtype](Float64(i % 5) * 0.3 - 0.6)
    var hp_data = alloc[Scalar[dtype]](BATCH * H)
    for i in range(BATCH * H):
        hp_data[i] = Scalar[dtype](Float64(i % 3) * 0.2 - 0.3)
    var cp_data = alloc[Scalar[dtype]](BATCH * H)
    for i in range(BATCH * H):
        cp_data[i] = Scalar[dtype](Float64(i % 7) * 0.15 - 0.5)

    # --- Params (deterministic, mid-magnitude) ---
    var p_data = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        # Spread weights in [-0.4, 0.4], biases small.
        var v = (Float64(i % 17) - 8.0) * 0.05
        p_data[i] = Scalar[dtype](v)

    var x_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        x_data
    )
    var hp_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        hp_data
    )
    var cp_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        cp_data
    )
    var p_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](p_data)

    # --- Forward ---
    var ht_data = alloc[Scalar[dtype]](BATCH * H)
    var ct_data = alloc[Scalar[dtype]](BATCH * H)
    var cache_data = alloc[Scalar[dtype]](BATCH * CS)
    memset(ht_data, 0, BATCH * H)
    memset(ct_data, 0, BATCH * H)
    memset(cache_data, 0, BATCH * CS)

    var ht_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        ht_data
    )
    var ct_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        ct_data
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CS), MutAnyOrigin
    ](cache_data)

    LC.step_forward[BATCH](x_t, hp_t, cp_t, p_t, ht_t, ct_t, cache_t)

    # --- Loss = sum(h_t) + 0.7 * sum(c_t)  ---
    # so dh = 1 everywhere, dc = 0.7 everywhere.
    var dh_data = alloc[Scalar[dtype]](BATCH * H)
    var dc_data = alloc[Scalar[dtype]](BATCH * H)
    for i in range(BATCH * H):
        dh_data[i] = Scalar[dtype](1.0)
        dc_data[i] = Scalar[dtype](0.7)

    var dh_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        dh_data
    )
    var dc_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        dc_data
    )

    var dx_data = alloc[Scalar[dtype]](BATCH * IN)
    var dhp_data = alloc[Scalar[dtype]](BATCH * H)
    var dcp_data = alloc[Scalar[dtype]](BATCH * H)
    var grads_data = alloc[Scalar[dtype]](PS)
    memset(dx_data, 0, BATCH * IN)
    memset(dhp_data, 0, BATCH * H)
    memset(dcp_data, 0, BATCH * H)
    memset(grads_data, 0, PS)

    var dx_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        dx_data
    )
    var dhp_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        dhp_data
    )
    var dcp_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        dcp_data
    )
    var grads_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        grads_data
    )

    LC.step_backward[BATCH](
        dh_t, dc_t, x_t, hp_t, cp_t, p_t, cache_t,
        dx_t, dhp_t, dcp_t, grads_t,
    )

    # ---------------------------------------------------------------------
    # Helper: compute scalar loss for current inputs/params via no-cache fwd
    # (inlined below as a @parameter closure that captures the LayoutTensors
    # by reference; no nested fn def to avoid capture-convention errors).
    # ---------------------------------------------------------------------
    var eps_fd = Float64(1e-3)

    @parameter
    @always_inline
    def _loss_eval() raises -> Float64:
        var ho = alloc[Scalar[dtype]](BATCH * H)
        var co = alloc[Scalar[dtype]](BATCH * H)
        memset(ho, 0, BATCH * H)
        memset(co, 0, BATCH * H)
        var ho_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](ho)
        var co_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](co)
        LC.step_forward_no_cache[BATCH](
            x_t, hp_t, cp_t, p_t, ho_t, co_t
        )
        var l = Float64(0.0)
        for i in range(BATCH * H):
            l += Float64(ho[i]) + 0.7 * Float64(co[i])
        ho.free()
        co.free()
        return l

    # --- FD wrt x ---
    var max_dx: Float64 = 0.0
    for idx in range(BATCH * IN):
        var orig = Float64(x_data[idx])
        x_data[idx] = Scalar[dtype](orig + eps_fd)
        var lp = _loss_eval()
        x_data[idx] = Scalar[dtype](orig - eps_fd)
        var lm = _loss_eval()
        x_data[idx] = Scalar[dtype](orig)
        var fd = (lp - lm) / (2.0 * eps_fd)
        var an = Float64(dx_data[idx])
        var d = fd - an
        if d < 0:
            d = -d
        if d > max_dx:
            max_dx = d
    print("Max |fd - analytical| dx:", max_dx)
    if max_dx < 0.01:
        print("PASS: dx")
    else:
        print("FAIL: dx (threshold 0.01)")

    # --- FD wrt h_prev ---
    var max_dhp: Float64 = 0.0
    for idx in range(BATCH * H):
        var orig = Float64(hp_data[idx])
        hp_data[idx] = Scalar[dtype](orig + eps_fd)
        var lp = _loss_eval()
        hp_data[idx] = Scalar[dtype](orig - eps_fd)
        var lm = _loss_eval()
        hp_data[idx] = Scalar[dtype](orig)
        var fd = (lp - lm) / (2.0 * eps_fd)
        var an = Float64(dhp_data[idx])
        var d = fd - an
        if d < 0:
            d = -d
        if d > max_dhp:
            max_dhp = d
    print("Max |fd - analytical| dh_prev:", max_dhp)
    if max_dhp < 0.01:
        print("PASS: dh_prev")
    else:
        print("FAIL: dh_prev (threshold 0.01)")

    # --- FD wrt c_prev ---
    var max_dcp: Float64 = 0.0
    for idx in range(BATCH * H):
        var orig = Float64(cp_data[idx])
        cp_data[idx] = Scalar[dtype](orig + eps_fd)
        var lp = _loss_eval()
        cp_data[idx] = Scalar[dtype](orig - eps_fd)
        var lm = _loss_eval()
        cp_data[idx] = Scalar[dtype](orig)
        var fd = (lp - lm) / (2.0 * eps_fd)
        var an = Float64(dcp_data[idx])
        var d = fd - an
        if d < 0:
            d = -d
        if d > max_dcp:
            max_dcp = d
    print("Max |fd - analytical| dc_prev:", max_dcp)
    if max_dcp < 0.01:
        print("PASS: dc_prev")
    else:
        print("FAIL: dc_prev (threshold 0.01)")

    # --- FD wrt params ---
    var max_dp: Float64 = 0.0
    for pidx in range(PS):
        var orig = Float64(p_data[pidx])
        p_data[pidx] = Scalar[dtype](orig + eps_fd)
        var lp = _loss_eval()
        p_data[pidx] = Scalar[dtype](orig - eps_fd)
        var lm = _loss_eval()
        p_data[pidx] = Scalar[dtype](orig)
        var fd = (lp - lm) / (2.0 * eps_fd)
        var an = Float64(grads_data[pidx])
        var d = fd - an
        if d < 0:
            d = -d
        if d > max_dp:
            max_dp = d
    print("Max |fd - analytical| dparams:", max_dp)
    if max_dp < 0.01:
        print("PASS: dparams")
    else:
        print("FAIL: dparams (threshold 0.01)")

    x_data.free()
    hp_data.free()
    cp_data.free()
    p_data.free()
    ht_data.free()
    ct_data.free()
    cache_data.free()
    dh_data.free()
    dc_data.free()
    dx_data.free()
    dhp_data.free()
    dcp_data.free()
    grads_data.free()


# =============================================================================
# Test 2: CPU vs GPU forward + backward parity
# =============================================================================


def test_cpu_vs_gpu() raises:
    """Forward + backward parity: CPU vs GPU on a 16x16 cell."""
    print()
    print("=" * 60)
    print("TEST: LSTMCell CPU vs GPU (in=16, hidden=16, batch=8)")
    print("=" * 60)

    var ctx = DeviceContext()

    comptime IN = 16
    comptime H = 16
    comptime BATCH = 8
    comptime LC = LSTMCell[IN, H]
    comptime PS = LC.PARAM_SIZE
    comptime CS = LC.CACHE_SIZE

    # Deterministic inputs
    var x_data = alloc[Scalar[dtype]](BATCH * IN)
    for i in range(BATCH * IN):
        x_data[i] = Scalar[dtype](Float64(i % 11) * 0.15 - 0.7)
    var hp_data = alloc[Scalar[dtype]](BATCH * H)
    for i in range(BATCH * H):
        hp_data[i] = Scalar[dtype](Float64(i % 7) * 0.1 - 0.3)
    var cp_data = alloc[Scalar[dtype]](BATCH * H)
    for i in range(BATCH * H):
        cp_data[i] = Scalar[dtype](Float64(i % 5) * 0.12 - 0.3)

    # Deterministic params
    var p_init = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        p_init[i] = Scalar[dtype]((Float64(i % 23) - 11.0) * 0.03)

    # ---------------- CPU forward ----------------
    var cpu_h = alloc[Scalar[dtype]](BATCH * H)
    var cpu_c = alloc[Scalar[dtype]](BATCH * H)
    var cpu_cache = alloc[Scalar[dtype]](BATCH * CS)
    memset(cpu_h, 0, BATCH * H)
    memset(cpu_c, 0, BATCH * H)
    memset(cpu_cache, 0, BATCH * CS)

    var x_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        x_data
    )
    var hp_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        hp_data
    )
    var cp_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        cp_data
    )
    var p_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](p_init)
    var hcpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](cpu_h)
    var ccpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](cpu_c)
    var cache_cpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CS), MutAnyOrigin
    ](cpu_cache)

    LC.step_forward[BATCH](
        x_t, hp_t, cp_t, p_t, hcpu_t, ccpu_t, cache_cpu_t
    )

    # ---------------- GPU forward ----------------
    var gpu_x = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    var gpu_hp = ctx.enqueue_create_buffer[dtype](BATCH * H)
    var gpu_cp = ctx.enqueue_create_buffer[dtype](BATCH * H)
    var gpu_p = ctx.enqueue_create_buffer[dtype](PS)
    var gpu_h = ctx.enqueue_create_buffer[dtype](BATCH * H)
    var gpu_c = ctx.enqueue_create_buffer[dtype](BATCH * H)
    var gpu_cache = ctx.enqueue_create_buffer[dtype](BATCH * CS)

    ctx.enqueue_copy(gpu_x, x_data)
    ctx.enqueue_copy(gpu_hp, hp_data)
    ctx.enqueue_copy(gpu_cp, cp_data)
    ctx.enqueue_copy(gpu_p, p_init)
    ctx.enqueue_memset(gpu_h, Scalar[dtype](0.0))
    ctx.enqueue_memset(gpu_c, Scalar[dtype](0.0))
    ctx.enqueue_memset(gpu_cache, Scalar[dtype](0.0))

    var gx_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        gpu_x.unsafe_ptr()
    )
    var ghp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](gpu_hp.unsafe_ptr())
    var gcp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](gpu_cp.unsafe_ptr())
    var gp_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        gpu_p.unsafe_ptr()
    )
    var gh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](gpu_h.unsafe_ptr())
    var gc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](gpu_c.unsafe_ptr())
    var gcache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CS), MutAnyOrigin
    ](gpu_cache.unsafe_ptr())

    LC.step_forward_gpu[BATCH](
        ctx, gx_t, ghp_t, gcp_t, gp_t, gh_t, gc_t, gcache_t
    )
    ctx.synchronize()

    var gpu_h_h = alloc[Scalar[dtype]](BATCH * H)
    var gpu_c_h = alloc[Scalar[dtype]](BATCH * H)
    ctx.enqueue_copy(gpu_h_h, gpu_h)
    ctx.enqueue_copy(gpu_c_h, gpu_c)
    ctx.synchronize()

    var max_h_diff: Float64 = 0.0
    var max_c_diff: Float64 = 0.0
    for i in range(BATCH * H):
        var d_h = Float64(cpu_h[i]) - Float64(gpu_h_h[i])
        if d_h < 0:
            d_h = -d_h
        if d_h > max_h_diff:
            max_h_diff = d_h
        var d_c = Float64(cpu_c[i]) - Float64(gpu_c_h[i])
        if d_c < 0:
            d_c = -d_c
        if d_c > max_c_diff:
            max_c_diff = d_c
    print("Max |cpu - gpu| h_t:", max_h_diff)
    print("Max |cpu - gpu| c_t:", max_c_diff)
    if max_h_diff < 1e-4 and max_c_diff < 1e-4:
        print("PASS: Forward parity")
    else:
        print("FAIL: Forward parity (threshold 1e-4)")

    # ---------------- Backward (CPU + GPU) ----------------
    var dh_data = alloc[Scalar[dtype]](BATCH * H)
    var dc_data = alloc[Scalar[dtype]](BATCH * H)
    for i in range(BATCH * H):
        dh_data[i] = Scalar[dtype](0.5 + Float64(i % 5) * 0.13)
        dc_data[i] = Scalar[dtype](0.3 + Float64(i % 3) * 0.21)

    # CPU backward
    var cpu_dx = alloc[Scalar[dtype]](BATCH * IN)
    var cpu_dhp = alloc[Scalar[dtype]](BATCH * H)
    var cpu_dcp = alloc[Scalar[dtype]](BATCH * H)
    var cpu_grads = alloc[Scalar[dtype]](PS)
    memset(cpu_dx, 0, BATCH * IN)
    memset(cpu_dhp, 0, BATCH * H)
    memset(cpu_dcp, 0, BATCH * H)
    memset(cpu_grads, 0, PS)

    var dh_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        dh_data
    )
    var dc_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        dc_data
    )
    var dxc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](cpu_dx)
    var dhpc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](cpu_dhp)
    var dcpc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](cpu_dcp)
    var grc_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        cpu_grads
    )

    LC.step_backward[BATCH](
        dh_t, dc_t, x_t, hp_t, cp_t, p_t, cache_cpu_t,
        dxc_t, dhpc_t, dcpc_t, grc_t,
    )

    # GPU backward (reuses gpu_cache + gpu_p from forward)
    var gpu_dh = ctx.enqueue_create_buffer[dtype](BATCH * H)
    var gpu_dc = ctx.enqueue_create_buffer[dtype](BATCH * H)
    var gpu_dx = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    var gpu_dhp = ctx.enqueue_create_buffer[dtype](BATCH * H)
    var gpu_dcp = ctx.enqueue_create_buffer[dtype](BATCH * H)
    var gpu_grads = ctx.enqueue_create_buffer[dtype](PS)
    var gpu_dcomb = ctx.enqueue_create_buffer[dtype](BATCH * 4 * H)

    ctx.enqueue_copy(gpu_dh, dh_data)
    ctx.enqueue_copy(gpu_dc, dc_data)
    ctx.enqueue_memset(gpu_dx, Scalar[dtype](0.0))
    ctx.enqueue_memset(gpu_dhp, Scalar[dtype](0.0))
    ctx.enqueue_memset(gpu_dcp, Scalar[dtype](0.0))
    ctx.enqueue_memset(gpu_grads, Scalar[dtype](0.0))

    var gdh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](gpu_dh.unsafe_ptr())
    var gdc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](gpu_dc.unsafe_ptr())
    var gdx_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](gpu_dx.unsafe_ptr())
    var gdhp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](gpu_dhp.unsafe_ptr())
    var gdcp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](gpu_dcp.unsafe_ptr())
    var ggr_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        gpu_grads.unsafe_ptr()
    )
    var gdcomb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 4 * H), MutAnyOrigin
    ](gpu_dcomb.unsafe_ptr())

    LC.step_backward_gpu[BATCH](
        ctx,
        gdh_t, gdc_t, gx_t, ghp_t, gcp_t, gp_t, gcache_t,
        gdx_t, gdhp_t, gdcp_t, ggr_t, gdcomb_t,
    )
    ctx.synchronize()

    var gpu_dx_h = alloc[Scalar[dtype]](BATCH * IN)
    var gpu_dhp_h = alloc[Scalar[dtype]](BATCH * H)
    var gpu_dcp_h = alloc[Scalar[dtype]](BATCH * H)
    var gpu_gr_h = alloc[Scalar[dtype]](PS)
    ctx.enqueue_copy(gpu_dx_h, gpu_dx)
    ctx.enqueue_copy(gpu_dhp_h, gpu_dhp)
    ctx.enqueue_copy(gpu_dcp_h, gpu_dcp)
    ctx.enqueue_copy(gpu_gr_h, gpu_grads)
    ctx.synchronize()

    var max_dx_diff: Float64 = 0.0
    for i in range(BATCH * IN):
        var d = Float64(cpu_dx[i]) - Float64(gpu_dx_h[i])
        if d < 0:
            d = -d
        if d > max_dx_diff:
            max_dx_diff = d
    var max_dhp_diff: Float64 = 0.0
    for i in range(BATCH * H):
        var d = Float64(cpu_dhp[i]) - Float64(gpu_dhp_h[i])
        if d < 0:
            d = -d
        if d > max_dhp_diff:
            max_dhp_diff = d
    var max_dcp_diff: Float64 = 0.0
    for i in range(BATCH * H):
        var d = Float64(cpu_dcp[i]) - Float64(gpu_dcp_h[i])
        if d < 0:
            d = -d
        if d > max_dcp_diff:
            max_dcp_diff = d
    var max_gr_diff: Float64 = 0.0
    for i in range(PS):
        var d = Float64(cpu_grads[i]) - Float64(gpu_gr_h[i])
        if d < 0:
            d = -d
        if d > max_gr_diff:
            max_gr_diff = d

    print("Max |cpu - gpu| dx:", max_dx_diff)
    print("Max |cpu - gpu| dh_prev:", max_dhp_diff)
    print("Max |cpu - gpu| dc_prev:", max_dcp_diff)
    print("Max |cpu - gpu| dparams:", max_gr_diff)
    if (
        max_dx_diff < 1e-3
        and max_dhp_diff < 1e-3
        and max_dcp_diff < 1e-3
        and max_gr_diff < 1e-3
    ):
        print("PASS: Backward parity")
    else:
        print("FAIL: Backward parity (threshold 1e-3)")

    x_data.free()
    hp_data.free()
    cp_data.free()
    p_init.free()
    cpu_h.free()
    cpu_c.free()
    cpu_cache.free()
    gpu_h_h.free()
    gpu_c_h.free()
    dh_data.free()
    dc_data.free()
    cpu_dx.free()
    cpu_dhp.free()
    cpu_dcp.free()
    cpu_grads.free()
    gpu_dx_h.free()
    gpu_dhp_h.free()
    gpu_dcp_h.free()
    gpu_gr_h.free()


# =============================================================================
# Test 3: BPTT — multi-step gradient accumulation
# =============================================================================


def test_bptt_accumulation() raises:
    """3-step unroll: verify backward param grads accumulate across time
    and equal the FD gradient of L = sum(h_3) (with h_0=c_0=0)."""
    print()
    print("=" * 60)
    print("TEST: LSTMCell BPTT 3-step unroll (in=2, hidden=2, batch=2)")
    print("=" * 60)

    comptime IN = 2
    comptime H = 2
    comptime BATCH = 2
    comptime T = 3
    comptime LC = LSTMCell[IN, H]
    comptime PS = LC.PARAM_SIZE
    comptime CS = LC.CACHE_SIZE

    # Inputs per time step (deterministic)
    var x_data = alloc[Scalar[dtype]](T * BATCH * IN)
    for i in range(T * BATCH * IN):
        x_data[i] = Scalar[dtype](Float64(i % 9) * 0.18 - 0.7)
    var p_data = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        p_data[i] = Scalar[dtype]((Float64(i % 19) - 9.0) * 0.04)

    var p_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](p_data)

    # Per-time buffers for h, c, cache
    # h[0..T] (T+1 slots: h_0..h_T), same for c
    var h_data = alloc[Scalar[dtype]]((T + 1) * BATCH * H)
    var c_data = alloc[Scalar[dtype]]((T + 1) * BATCH * H)
    var cache_data = alloc[Scalar[dtype]](T * BATCH * CS)
    memset(h_data, 0, (T + 1) * BATCH * H)
    memset(c_data, 0, (T + 1) * BATCH * H)
    memset(cache_data, 0, T * BATCH * CS)

    # Forward 3 steps. View each time slice.
    for t in range(T):
        var xt = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
        ](x_data + t * BATCH * IN)
        var hp = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](h_data + t * BATCH * H)
        var cp = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](c_data + t * BATCH * H)
        var ht = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](h_data + (t + 1) * BATCH * H)
        var ct = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](c_data + (t + 1) * BATCH * H)
        var cc = LayoutTensor[
            dtype, Layout.row_major(BATCH, CS), MutAnyOrigin
        ](cache_data + t * BATCH * CS)
        LC.step_forward[BATCH](xt, hp, cp, p_t, ht, ct, cc)

    # Loss = sum(h_T)
    # dh_T = 1, dc_T = 0
    var dh = alloc[Scalar[dtype]](BATCH * H)
    var dc = alloc[Scalar[dtype]](BATCH * H)
    for i in range(BATCH * H):
        dh[i] = Scalar[dtype](1.0)
        dc[i] = Scalar[dtype](0.0)

    # Backward 3 steps, threading dh/dc back. Param grads accumulate.
    var grads_data = alloc[Scalar[dtype]](PS)
    memset(grads_data, 0, PS)
    var grads_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        grads_data
    )

    var dx = alloc[Scalar[dtype]](BATCH * IN)
    var dhp = alloc[Scalar[dtype]](BATCH * H)
    var dcp = alloc[Scalar[dtype]](BATCH * H)

    for tt in range(T):
        var t = T - 1 - tt  # backward in time
        var xt = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
        ](x_data + t * BATCH * IN)
        var hp = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](h_data + t * BATCH * H)
        var cp = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](c_data + t * BATCH * H)
        var cc = LayoutTensor[
            dtype, Layout.row_major(BATCH, CS), MutAnyOrigin
        ](cache_data + t * BATCH * CS)

        var dh_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](dh)
        var dc_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](dc)
        var dx_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
        ](dx)
        var dhp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](dhp)
        var dcp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, H), MutAnyOrigin
        ](dcp)

        memset(dx, 0, BATCH * IN)
        memset(dhp, 0, BATCH * H)
        memset(dcp, 0, BATCH * H)

        LC.step_backward[BATCH](
            dh_t, dc_t, xt, hp, cp, p_t, cc,
            dx_t, dhp_t, dcp_t, grads_t,
        )

        # Thread dh, dc back: at step t-1, dh_T (from step t-1's forward
        # output) = dhp from step t. Same for dc.
        for i in range(BATCH * H):
            dh[i] = dhp[i]
            dc[i] = dcp[i]

    # FD check: gradient of L = sum(h_T) wrt one param.
    # Compare to grads_data[].
    @parameter
    @always_inline
    def _loss_eval_t() raises -> Float64:
        var hh = alloc[Scalar[dtype]]((T + 1) * BATCH * H)
        var cc2 = alloc[Scalar[dtype]]((T + 1) * BATCH * H)
        memset(hh, 0, (T + 1) * BATCH * H)
        memset(cc2, 0, (T + 1) * BATCH * H)
        for t in range(T):
            var xt = LayoutTensor[
                dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
            ](x_data + t * BATCH * IN)
            var hp = LayoutTensor[
                dtype, Layout.row_major(BATCH, H), MutAnyOrigin
            ](hh + t * BATCH * H)
            var cp = LayoutTensor[
                dtype, Layout.row_major(BATCH, H), MutAnyOrigin
            ](cc2 + t * BATCH * H)
            var ht = LayoutTensor[
                dtype, Layout.row_major(BATCH, H), MutAnyOrigin
            ](hh + (t + 1) * BATCH * H)
            var ct = LayoutTensor[
                dtype, Layout.row_major(BATCH, H), MutAnyOrigin
            ](cc2 + (t + 1) * BATCH * H)
            LC.step_forward_no_cache[BATCH](xt, hp, cp, p_t, ht, ct)
        var l = Float64(0.0)
        for i in range(BATCH * H):
            l += Float64(hh[T * BATCH * H + i])
        hh.free()
        cc2.free()
        return l

    var eps_fd = Float64(1e-3)
    var max_diff: Float64 = 0.0
    var first_few = 0
    for pidx in range(PS):
        var orig = Float64(p_data[pidx])
        p_data[pidx] = Scalar[dtype](orig + eps_fd)
        var lp = _loss_eval_t()
        p_data[pidx] = Scalar[dtype](orig - eps_fd)
        var lm = _loss_eval_t()
        p_data[pidx] = Scalar[dtype](orig)
        var fd = (lp - lm) / (2.0 * eps_fd)
        var an = Float64(grads_data[pidx])
        var d = fd - an
        if d < 0:
            d = -d
        if d > max_diff:
            max_diff = d
        if first_few < 3:
            print(
                "  pidx=",
                pidx,
                "fd=",
                fd,
                " analytical=",
                an,
                " |diff|=",
                d,
            )
            first_few += 1
    print("Max |fd - analytical| dparams (3-step BPTT):", max_diff)
    if max_diff < 0.01:
        print("PASS: BPTT param gradient accumulation")
    else:
        print("FAIL: BPTT param gradient accumulation (threshold 0.01)")

    x_data.free()
    p_data.free()
    h_data.free()
    c_data.free()
    cache_data.free()
    dh.free()
    dc.free()
    dx.free()
    dhp.free()
    dcp.free()
    grads_data.free()


def main() raises:
    test_cpu_gradcheck()
    test_cpu_vs_gpu()
    test_bptt_accumulation()
    print()
    print("All LSTMCell tests done.")
