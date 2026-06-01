"""ConvPCBlock parity spike (P0) — see docs/PCN_CONV_DESIGN.md.

Validates the convolutional PC math mapping by checking ConvPCBlock's three
core ops against the nn Conv2D autodiff primitive as an INDEPENDENT oracle
(two implementations agreeing). Per Salvatori et al. 2021, PC inference on a
conv graph reproduces backprop gradients exactly, so Conv2D.vjp is an exact
oracle (modulo the PC sign convention and the activation-first ordering):

  predict   μ          == Conv2D.eval(a_below)           (a_below = ACT(x_below))
  pull_back z_below    == Conv2D.vjp grad_input          (w.r.t a_below)
  weight_grad W-part   == −(Conv2D.vjp grad_W)           (PC bakes the −sign)
  weight_grad b-part   == −(Conv2D.vjp grad_b)

Conv2D is instantiated with USE_MAX_KERNELS=False so both eval and vjp take the
fully-naive deterministic CPU path — a clean oracle independent of BLAS.

Run:
    pixi run mojo run -I . tests/pcn/test_conv_pc_block_parity.mojo
"""

from std.memory import alloc
from std.math import sin
from layout import Layout, LayoutTensor

from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.autodiff.primitives.conv2d import Conv2D
from mojo_rl.experimental.pcn.pc_conv_block import ConvPCBlock
from mojo_rl.experimental.pcn import PCReLU

comptime dtype = DType.float32
comptime TOL: Float32 = 1e-4


def _max_abs_diff(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
) -> Float32:
    var m: Float32 = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        var ad = d if d >= 0 else -d
        if ad > m:
            m = ad
    return m


def run_parity[
    IC: Int,
    OC: Int,
    K: Int,
    S: Int,
    P: Int,
    H: Int,
    W: Int,
    BATCH: Int,
](label: String) -> Bool:
    comptime CB = ConvPCBlock[IC, OC, K, S, P, H, W, PCReLU]
    comptime C = Conv2D[IC, OC, K, S, P, H, W, False]

    comptime IN = CB.IN_DIM
    comptime OUT = CB.OUT_DIM
    comptime PSZ = CB.PARAM_SIZE
    comptime W_SIZE = OC * CB.col_size

    comptime assert IN == C.IN_DIM, "IN_DIM mismatch"
    comptime assert OUT == C.OUT_DIM, "OUT_DIM mismatch"
    comptime assert PSZ == C.PARAM_SIZE, "PARAM_SIZE mismatch"

    # ── Buffers ──────────────────────────────────────────────────────────────
    var x_buf = alloc[Scalar[dtype]](BATCH * IN)
    var a_buf = alloc[Scalar[dtype]](BATCH * IN)
    var mu_buf = alloc[Scalar[dtype]](BATCH * OUT)
    var eps_buf = alloc[Scalar[dtype]](BATCH * OUT)
    var params_buf = alloc[Scalar[dtype]](PSZ)
    var z_buf = alloc[Scalar[dtype]](BATCH * IN)
    var grads_buf = alloc[Scalar[dtype]](PSZ)

    var mu_oracle_buf = alloc[Scalar[dtype]](BATCH * OUT)
    var cache_buf = alloc[Scalar[dtype]](BATCH * C.CACHE_SIZE)
    var gi_buf = alloc[Scalar[dtype]](BATCH * IN)
    var gp_buf = alloc[Scalar[dtype]](PSZ)

    var x = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](x_buf)
    var a = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](a_buf)
    var mu = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](
        mu_buf
    )
    var eps = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](
        eps_buf
    )
    var params = LayoutTensor[dtype, Layout.row_major(PSZ), MutAnyOrigin](
        params_buf
    )
    var z = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](z_buf)
    var grads = LayoutTensor[dtype, Layout.row_major(PSZ), MutAnyOrigin](
        grads_buf
    )
    var mu_oracle = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](mu_oracle_buf)
    var cache = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
    ](cache_buf)
    var gi = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        gi_buf
    )
    var gp = LayoutTensor[dtype, Layout.row_major(PSZ), MutAnyOrigin](gp_buf)

    # ── Deterministic inputs (varied sign so ReLU is exercised) ───────────────
    for i in range(BATCH * IN):
        x_buf[i] = Scalar[dtype](sin(Float32(i) * 0.7 + 0.3) * 1.5)
    for i in range(BATCH * OUT):
        eps_buf[i] = Scalar[dtype](sin(Float32(i) * 1.1 + 1.7))
    CB.initialize_params[Xavier[42], dtype](params)

    # ── ConvPCBlock ops ───────────────────────────────────────────────────────
    CB.predict[BATCH, dtype](x, params, mu, a)
    CB.pull_back[BATCH, dtype](eps, params, z)
    CB.weight_grad[BATCH, dtype](eps, a, grads)

    # ── Oracle: Conv2D.eval(a_below) and Conv2D.vjp(grad_output=eps) ──────────
    C.eval[BATCH, dtype](a, mu_oracle, params, cache)
    for i in range(PSZ):
        gp_buf[i] = Scalar[dtype](0)
    C.vjp[BATCH, dtype](eps, gi, params, cache, gp)

    # ── Compare ───────────────────────────────────────────────────────────────
    var d_predict = _max_abs_diff(mu_buf, mu_oracle_buf, BATCH * OUT)
    var d_pullback = _max_abs_diff(z_buf, gi_buf, BATCH * IN)

    # weight_grad parity: ConvPCBlock grad == −(Conv2D grad). Negate oracle.
    var neg_gp = alloc[Scalar[dtype]](PSZ)
    for i in range(PSZ):
        neg_gp[i] = -gp_buf[i]
    var d_wgrad_W = _max_abs_diff(grads_buf, neg_gp, W_SIZE)
    var d_wgrad_b = _max_abs_diff(grads_buf + W_SIZE, neg_gp + W_SIZE, OC)
    neg_gp.free()

    print("── " + label + " ──")
    print("  IN=" + String(IN) + " OUT=" + String(OUT) + " out_h=" + String(
        CB.out_h
    ) + " out_w=" + String(CB.out_w))
    print("  predict   max|Δ| =", d_predict)
    print("  pull_back max|Δ| =", d_pullback)
    print("  weight_gW max|Δ| =", d_wgrad_W)
    print("  weight_gb max|Δ| =", d_wgrad_b)

    var ok = (
        d_predict < TOL
        and d_pullback < TOL
        and d_wgrad_W < TOL
        and d_wgrad_b < TOL
    )

    x_buf.free()
    a_buf.free()
    mu_buf.free()
    eps_buf.free()
    params_buf.free()
    z_buf.free()
    grads_buf.free()
    mu_oracle_buf.free()
    cache_buf.free()
    gi_buf.free()
    gp_buf.free()
    return ok


def main() raises:
    print("ConvPCBlock ↔ Conv2D parity spike\n")
    var all_ok = True

    # Config A: same-size (stride 1, pad 1) — boundary padding exercised.
    all_ok = run_parity[2, 3, 3, 1, 1, 4, 4, 2]("A  stride=1 pad=1 4x4") and all_ok

    # Config B: downsampling (stride 2, pad 1) — strided transposed conv.
    all_ok = run_parity[2, 3, 3, 2, 1, 5, 5, 2]("B  stride=2 pad=1 5x5") and all_ok

    # Config C: no pad, 1 channel→multi, batch 3.
    all_ok = run_parity[1, 4, 3, 1, 0, 6, 6, 3]("C  stride=1 pad=0 6x6") and all_ok

    print("")
    if all_ok:
        print("✅ PASS — ConvPCBlock matches Conv2D oracle within", TOL)
    else:
        print("❌ FAIL — parity mismatch")
        raise Error("ConvPCBlock parity failed")
