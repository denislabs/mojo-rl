"""`PairwiseDot` / `RowDot` gate — closed form, finite differences, GPU parity.

These are the two primitives the FB loss rests on
(`docs/BFM_ZERO_SHOT_RL.md` §6, "Missing #1"), and they are the first modules
in `nn` whose BOTH inputs are activations carrying gradient. That asymmetry is
where the bugs live, so the gate is built around it:

  [1] forward against an independently written triple loop;
  [2] the vjp against CENTRAL finite differences of a scalar loss;
  [3] **the transpose check** — `dA` contracts G over its rows and `dC` over its
      columns. Swapping them produces a matrix of exactly the right shape, and
      for a symmetric probe it produces the right VALUES too. So the probe here
      is deliberately asymmetric in every way it can be: A != C, and the
      upstream gradient G is non-symmetric with distinct row and column sums.
      A gate built on `A == C` or on `G == G^T` would pass with the two kernels
      exchanged;
  [4] CPU vs GPU, both directions.

⚠ On the finite-difference tolerance: fp32 activations with D=8, B=6 keep the
condition number low, so 1e-2 relative is comfortable here. Do NOT copy that
tolerance to a deep chain — the project's own note is that FD through many
layers needs eps ~1e-2 and much looser bounds. This is a single bilinear op;
if it needed a loose tolerance, something would be wrong.

Run:
    pixi run mojo run -I . tests/nn/test_pairwise_dot.mojo
"""

from max.gpu.host import DeviceContext
from std.math import abs, sqrt
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.pairwise_dot import PairwiseDot, RowDot


comptime B = 6
comptime D = 8
comptime TOL = Float64(1e-5)


def _fill_a(mut t: Tensor):
    for i in range(B):
        for k in range(D):
            t.data[i * D + k] = Scalar[DT](
                0.31 * Float64(i) - 0.17 * Float64(k) + 0.05 * Float64(i * k)
            )


def _fill_c(mut t: Tensor):
    # Deliberately NOT a permutation or scaling of A: if C were A, dA and dC
    # would coincide and swapping the two kernels would go unnoticed.
    for j in range(B):
        for k in range(D):
            t.data[j * D + k] = Scalar[DT](
                -0.22 * Float64(j) + 0.41 * Float64(k) - 0.07 * Float64(j * j)
            )


def _fill_g_matrix(mut t: Tensor):
    """Upstream gradient for the [B, B] output — NON-symmetric on purpose.

    `dA = G·C` and `dC = G^T·A`. With a symmetric G the two kernels compute the
    same contraction and the transpose bug is invisible.
    """
    for i in range(B):
        for j in range(B):
            t.data[i * B + j] = Scalar[DT](
                0.13 * Float64(i + 1) - 0.29 * Float64(j + 1)
                + 0.04 * Float64(i * i)
            )


def _assert_g_asymmetric(ref t: Tensor) raises:
    var worst = Float64(0)
    for i in range(B):
        for j in range(B):
            var d = abs(Float64(t.data[i * B + j]) - Float64(t.data[j * B + i]))
            if d > worst:
                worst = d
    assert_true(
        worst > 0.1,
        "the probe gradient is (nearly) symmetric — this gate cannot"
        " distinguish dA from dC. Worst |G - G^T| = " + String(worst),
    )


def test_forward_closed_form() raises:
    print("[1] PairwiseDot forward vs an independent triple loop ...")
    var op = PairwiseDot[D, B].make["cpu", Deterministic](None)
    var ins = TensorPack[2]()
    ins[0].ensure(B * D)
    ins[1].ensure(B * D)
    _fill_a(ins[0])
    _fill_c(ins[1])
    var out = Tensor.alloc(B * B)
    op.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), out, None)

    var worst = Float64(0)
    for i in range(B):
        for j in range(B):
            var want = Float64(0)
            for k in range(D):
                want += (
                    Float64(ins[0].data[i * D + k])
                    * Float64(ins[1].data[j * D + k])
                )
            var e = abs(Float64(out.data[i * B + j]) - want)
            if e > worst:
                worst = e
    print("      worst |M - reference| =", worst)
    assert_true(worst < TOL, "PairwiseDot forward mismatch " + String(worst))

    # The result must not be symmetric — if it were, [3] below is vacuous.
    var asym = Float64(0)
    for i in range(B):
        for j in range(B):
            var d = abs(
                Float64(out.data[i * B + j]) - Float64(out.data[j * B + i])
            )
            if d > asym:
                asym = d
    assert_true(asym > 0.1, "probe produced a symmetric M — pick other inputs")


def _loss_and_grad_cpu(
    ref a: Tensor, ref c: Tensor, ref g: Tensor,
    mut ga: Tensor, mut gc: Tensor,
) raises -> Float64:
    """L = sum_{i,j} G[i,j]·M[i,j], plus the analytic dA / dC from the vjp.

    A linear functional of M is the right probe: its gradient wrt M is exactly
    G, so the vjp is exercised with a NON-trivial, non-constant upstream
    gradient rather than the all-ones a `sum(M)` loss would hand it.
    """
    var op = PairwiseDot[D, B].make["cpu", Deterministic](None)
    var ins = TensorPack[2]()
    ins[0].ensure(B * D)
    ins[1].ensure(B * D)
    for i in range(B * D):
        ins[0].data[i] = a.data[i]
        ins[1].data[i] = c.data[i]
    var out = Tensor.alloc(B * B)
    op.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), out, None)

    var loss = Float64(0)
    for i in range(B * B):
        loss += Float64(g.data[i]) * Float64(out.data[i])

    var go = Tensor.alloc(B * B)
    for i in range(B * B):
        go.data[i] = g.data[i]
    var grads = TensorPack[2]()
    op.vjp["cpu", B](
        TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](grads[0], grads[1]),
        None,
    )
    ga.ensure(B * D)
    gc.ensure(B * D)
    for i in range(B * D):
        ga.data[i] = grads[0].data[i]
        gc.data[i] = grads[1].data[i]
    return loss


def test_vjp_finite_differences() raises:
    print("[2] PairwiseDot vjp vs central finite differences ...")
    var a = Tensor.alloc(B * D)
    var c = Tensor.alloc(B * D)
    var g = Tensor.alloc(B * B)
    _fill_a(a)
    _fill_c(c)
    _fill_g_matrix(g)
    _assert_g_asymmetric(g)

    var ga = Tensor.alloc(B * D)
    var gc = Tensor.alloc(B * D)
    var sink_a = Tensor.alloc(B * D)
    var sink_c = Tensor.alloc(B * D)
    _ = _loss_and_grad_cpu(a, c, g, ga, gc)

    comptime EPS = Float64(1e-3)
    var worst_a = Float64(0)
    var worst_c = Float64(0)

    for idx in range(B * D):
        var keep = a.data[idx]
        a.data[idx] = Scalar[DT](Float64(keep) + EPS)
        var lp = _loss_and_grad_cpu(a, c, g, sink_a, sink_c)
        a.data[idx] = Scalar[DT](Float64(keep) - EPS)
        var lm = _loss_and_grad_cpu(a, c, g, sink_a, sink_c)
        a.data[idx] = keep
        var fd = (lp - lm) / (2.0 * EPS)
        var an = Float64(ga.data[idx])
        var denom = abs(an) if abs(an) > 1.0 else 1.0
        var rel = abs(fd - an) / denom
        if rel > worst_a:
            worst_a = rel

    for idx in range(B * D):
        var keep = c.data[idx]
        c.data[idx] = Scalar[DT](Float64(keep) + EPS)
        var lp = _loss_and_grad_cpu(a, c, g, sink_a, sink_c)
        c.data[idx] = Scalar[DT](Float64(keep) - EPS)
        var lm = _loss_and_grad_cpu(a, c, g, sink_a, sink_c)
        c.data[idx] = keep
        var fd = (lp - lm) / (2.0 * EPS)
        var an = Float64(gc.data[idx])
        var denom = abs(an) if abs(an) > 1.0 else 1.0
        var rel = abs(fd - an) / denom
        if rel > worst_c:
            worst_c = rel

    print("      worst relative error: dA", worst_a, " dC", worst_c)
    assert_true(worst_a < 1e-2, "dA vs FD: " + String(worst_a))
    assert_true(worst_c < 1e-2, "dC vs FD: " + String(worst_c))


def test_transpose_not_swapped() raises:
    """`dA = G·C` and `dC = G^T·A` — check each against its own contraction.

    This is the check finite differences alone cannot make cheap: FD confirms
    the pair is jointly right, but computing both closed forms here says WHICH
    is which, so a swap is reported as a swap and not as a mysterious gradient
    error.
    """
    print("[3] PairwiseDot dA/dC contract the correct axis of G ...")
    var a = Tensor.alloc(B * D)
    var c = Tensor.alloc(B * D)
    var g = Tensor.alloc(B * B)
    _fill_a(a)
    _fill_c(c)
    _fill_g_matrix(g)
    _assert_g_asymmetric(g)

    var ga = Tensor.alloc(B * D)
    var gc = Tensor.alloc(B * D)
    _ = _loss_and_grad_cpu(a, c, g, ga, gc)

    var worst_a = Float64(0)
    var worst_c = Float64(0)
    var cross_a = Float64(0)  # how wrong dA would be if it used G^T
    for i in range(B):
        for k in range(D):
            var want = Float64(0)
            var swapped = Float64(0)
            for j in range(B):
                want += Float64(g.data[i * B + j]) * Float64(c.data[j * D + k])
                swapped += (
                    Float64(g.data[j * B + i]) * Float64(c.data[j * D + k])
                )
            var e = abs(Float64(ga.data[i * D + k]) - want)
            if e > worst_a:
                worst_a = e
            var s = abs(want - swapped)
            if s > cross_a:
                cross_a = s
    for j in range(B):
        for k in range(D):
            var want = Float64(0)
            for i in range(B):
                want += Float64(g.data[i * B + j]) * Float64(a.data[i * D + k])
            var e = abs(Float64(gc.data[j * D + k]) - want)
            if e > worst_c:
                worst_c = e

    print("      dA err", worst_a, " dC err", worst_c,
          " (a transposed G would differ by", cross_a, ")")
    assert_true(
        cross_a > 0.1,
        "G is too close to symmetric for this check to detect a transpose"
        " swap — the gate is vacuous as written",
    )
    assert_true(worst_a < TOL, "dA contracted the wrong axis: " + String(worst_a))
    assert_true(worst_c < TOL, "dC contracted the wrong axis: " + String(worst_c))


def test_rowdot_is_the_diagonal() raises:
    """`RowDot` must equal `diag(PairwiseDot)` — and its gradient must not.

    Same values, different gradient: `PairwiseDot`'s vjp with a diagonal G
    produces the same dA/dC as `RowDot`'s, which is the property that lets the
    FB anchor term use the cheap one. Checking the forward alone would miss a
    RowDot vjp that forgot one of its two inputs.
    """
    print("[4] RowDot == diag(PairwiseDot), forward and backward ...")
    var a = Tensor.alloc(B * D)
    var c = Tensor.alloc(B * D)
    _fill_a(a)
    _fill_c(c)

    var pd = PairwiseDot[D, B].make["cpu", Deterministic](None)
    var rd = RowDot[D].make["cpu", Deterministic](None)

    var ins = TensorPack[2]()
    ins[0].ensure(B * D)
    ins[1].ensure(B * D)
    for i in range(B * D):
        ins[0].data[i] = a.data[i]
        ins[1].data[i] = c.data[i]

    var m = Tensor.alloc(B * B)
    pd.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), m, None)
    var r = Tensor.alloc(B)
    rd.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), r, None)

    var worst = Float64(0)
    for i in range(B):
        var e = abs(Float64(r.data[i]) - Float64(m.data[i * B + i]))
        if e > worst:
            worst = e
    assert_true(worst < TOL, "RowDot != diag(PairwiseDot): " + String(worst))

    # Backward: seed PairwiseDot with a DIAGONAL upstream gradient carrying the
    # same per-row weights RowDot gets, then the two vjps must agree.
    var w = List[Float64]()
    for i in range(B):
        w.append(0.4 * Float64(i) - 0.9)

    var go_m = Tensor.alloc(B * B)
    for i in range(B):
        for j in range(B):
            go_m.data[i * B + j] = Scalar[DT](w[i] if i == j else 0.0)
    var gpd = TensorPack[2]()
    pd.vjp["cpu", B](
        TensorRefs[2](ins[0], ins[1]), go_m,
        TensorRefs[2](gpd[0], gpd[1]), None,
    )

    var go_r = Tensor.alloc(B)
    for i in range(B):
        go_r.data[i] = Scalar[DT](w[i])
    var grd = TensorPack[2]()
    rd.vjp["cpu", B](
        TensorRefs[2](ins[0], ins[1]), go_r,
        TensorRefs[2](grd[0], grd[1]), None,
    )

    var wa = Float64(0)
    var wc = Float64(0)
    for i in range(B * D):
        var ea = abs(Float64(gpd[0].data[i]) - Float64(grd[0].data[i]))
        var ec = abs(Float64(gpd[1].data[i]) - Float64(grd[1].data[i]))
        if ea > wa:
            wa = ea
        if ec > wc:
            wc = ec
    print("      forward", worst, " grad dA", wa, " dC", wc)
    assert_true(wa < TOL and wc < TOL, "RowDot vjp != diagonal PairwiseDot vjp")


def test_gpu_parity() raises:
    print("[5] CPU vs GPU, forward and backward ...")
    var ctx = DeviceContext()
    var a = Tensor.alloc(B * D)
    var c = Tensor.alloc(B * D)
    var g = Tensor.alloc(B * B)
    _fill_a(a)
    _fill_c(c)
    _fill_g_matrix(g)

    # CPU reference
    var ga_cpu = Tensor.alloc(B * D)
    var gc_cpu = Tensor.alloc(B * D)
    _ = _loss_and_grad_cpu(a, c, g, ga_cpu, gc_cpu)
    var op_cpu = PairwiseDot[D, B].make["cpu", Deterministic](None)
    var ins_cpu = TensorPack[2]()
    ins_cpu[0].ensure(B * D)
    ins_cpu[1].ensure(B * D)
    for i in range(B * D):
        ins_cpu[0].data[i] = a.data[i]
        ins_cpu[1].data[i] = c.data[i]
    var m_cpu = Tensor.alloc(B * B)
    op_cpu.forward["cpu", B](
        TensorRefs[2](ins_cpu[0], ins_cpu[1]), m_cpu, None
    )

    # GPU
    var op_gpu = PairwiseDot[D, B].make["gpu", Deterministic](ctx)
    var ins_gpu = TensorPack[2]()
    # `ensure` (host) then `upload` — `ensure_gpu` allocates the DEVICE buffer
    # only, leaving `data` empty, and the fill below would index out of bounds.
    ins_gpu[0].ensure(B * D)
    ins_gpu[1].ensure(B * D)
    for i in range(B * D):
        ins_gpu[0].data[i] = a.data[i]
        ins_gpu[1].data[i] = c.data[i]
    ins_gpu[0].upload(ctx)
    ins_gpu[1].upload(ctx)

    var m_gpu = Tensor()
    op_gpu.forward["gpu", B](
        TensorRefs[2](ins_gpu[0], ins_gpu[1]), m_gpu, ctx
    )
    m_gpu.download(ctx)

    var wf = Float64(0)
    for i in range(B * B):
        var e = abs(Float64(m_gpu.data[i]) - Float64(m_cpu.data[i]))
        if e > wf:
            wf = e

    var go_gpu = Tensor()
    go_gpu.ensure(B * B)
    for i in range(B * B):
        go_gpu.data[i] = g.data[i]
    go_gpu.upload(ctx)
    var grads_gpu = TensorPack[2]()
    op_gpu.vjp["gpu", B](
        TensorRefs[2](ins_gpu[0], ins_gpu[1]), go_gpu,
        TensorRefs[2](grads_gpu[0], grads_gpu[1]), ctx,
    )
    grads_gpu[0].download(ctx)
    grads_gpu[1].download(ctx)

    var wa = Float64(0)
    var wc = Float64(0)
    for i in range(B * D):
        var ea = abs(Float64(grads_gpu[0].data[i]) - Float64(ga_cpu.data[i]))
        var ec = abs(Float64(grads_gpu[1].data[i]) - Float64(gc_cpu.data[i]))
        if ea > wa:
            wa = ea
        if ec > wc:
            wc = ec

    print("      forward", wf, " dA", wa, " dC", wc)
    assert_true(wf < 1e-4, "GPU forward parity: " + String(wf))
    assert_true(wa < 1e-4, "GPU dA parity: " + String(wa))
    assert_true(wc < 1e-4, "GPU dC parity: " + String(wc))


def main() raises:
    test_forward_closed_form()
    test_vjp_finite_differences()
    test_transpose_not_swapped()
    test_rowdot_is_the_diagonal()
    test_gpu_parity()
    print("\n[PASS] PairwiseDot / RowDot gate")
