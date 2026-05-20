"""GaussianHead[IN, ACT] CPU tests — Phase 6.2.

Covers:
  - forward: hand-set W/b/log_std, verify output[:, 0:ACT] = mu and
    output[:, ACT:2*ACT] = broadcast(clamp(log_std, [-5, 2]))
  - log_std clamping in forward at both bounds
  - backward grad_input + grad_w + grad_b on the mu branch
  - backward grad_log_std reduces over batch (state-indep param)
  - for_each_param yields {weight, bias, log_std} with
    apply_decay flags {True, False, False}
  - zero_grad clears all 3 accumulators
  - analytical gradcheck via finite differences on every param + input
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_equal, assert_true
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.primitives.gaussian_head import GaussianHead
from mojo_rl.nn2.initializer import Zero


# ──────────────────────────────────────────────────────────────────────────
# CountVisitor — records the param-walk for the for_each_param test.
# ──────────────────────────────────────────────────────────────────────────


struct WalkVisitor(ParamVisitor):
    var names: List[String]
    var sizes: List[Int]
    var decays: List[Bool]

    def __init__(out self):
        self.names = List[String]()
        self.sizes = List[Int]()
        self.decays = List[Bool]()

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        self.names.append(name)
        self.sizes.append(n_elems)
        self.decays.append(apply_decay)


# ──────────────────────────────────────────────────────────────────────────
# test_forward
# ──────────────────────────────────────────────────────────────────────────


def test_forward() raises:
    """W = [[1, 2], [3, 4]], b = [10, 20], log_std = [-0.5, 1.0].
    input = [[1, 1]]. Expected mu = [10 + 1 + 3, 20 + 2 + 4] = [14, 26].
    Expected log_std cols = [-0.5, 1.0] (within bounds, no clamp)."""
    comptime IN = 2
    comptime ACT = 2
    comptime BATCH = 1

    var h = GaussianHead[IN, ACT].make[target="cpu", INIT=Zero]()
    var w = TileTensor(h.weight, row_major[IN, ACT]())
    var b = TileTensor(h.bias, row_major[ACT]())
    var ls = TileTensor(h.log_std, row_major[ACT]())

    w[0, 0] = 1.0
    w[0, 1] = 2.0
    w[1, 0] = 3.0
    w[1, 1] = 4.0
    b[0] = 10.0
    b[1] = 20.0
    ls[0] = -0.5
    ls[1] = 1.0

    var in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    in_buf[0] = 1.0
    in_buf[1] = 1.0
    for k in range(BATCH * 2 * ACT):
        out_buf[k] = -999.0

    var input = TileTensor(in_buf, row_major[BATCH, IN]())
    var output = TileTensor(out_buf, row_major[BATCH, 2 * ACT]())

    h.forward["cpu", BATCH](input, output)

    # mu portion
    assert_equal(output[0, 0], 14.0)
    assert_equal(output[0, 1], 26.0)
    # log_std portion (no clamp)
    assert_equal(output[0, 2], -0.5)
    assert_equal(output[0, 3], 1.0)

    in_buf.free()
    out_buf.free()
    print("  test_forward PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_log_std_clamp
# ──────────────────────────────────────────────────────────────────────────


def test_log_std_clamp() raises:
    """Out-of-range log_std values get clamped to [-5, 2] in forward output."""
    comptime IN = 1
    comptime ACT = 2
    comptime BATCH = 1

    var h = GaussianHead[IN, ACT].make[target="cpu", INIT=Zero]()
    var ls = TileTensor(h.log_std, row_major[ACT]())
    ls[0] = -10.0  # below LOG_STD_MIN
    ls[1] = 5.0  # above LOG_STD_MAX

    var in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    in_buf[0] = 0.0
    for k in range(BATCH * 2 * ACT):
        out_buf[k] = 0.0
    var input = TileTensor(in_buf, row_major[BATCH, IN]())
    var output = TileTensor(out_buf, row_major[BATCH, 2 * ACT]())

    h.forward["cpu", BATCH](input, output)

    assert_equal(output[0, 2], -5.0)
    assert_equal(output[0, 3], 2.0)

    in_buf.free()
    out_buf.free()
    print("  test_log_std_clamp PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_backward
# ──────────────────────────────────────────────────────────────────────────


def test_backward() raises:
    """W = [[1, 2], [3, 4]], cache = [[1, 1], [2, 3]] (BATCH=2).
    grad_output[b, j] = 1 for all (b, j).

    Expected:
      grad_input = grad_mu @ W^T = [[1+2, 3+4], [1+2, 3+4]] = [[3, 7], [3, 7]]
      grad_w[i, j] = sum_b cache[b, i] * grad_out[b, j] = sum_b cache[b, i]
        = [[1+2, 1+2], [1+3, 1+3]] = [[3, 3], [4, 4]]
      grad_b[j] = sum_b grad_out[b, j] = 2
      grad_log_std[j] = sum_b grad_out[b, ACT+j] = 2 (broadcast cols too)
    """
    comptime IN = 2
    comptime ACT = 2
    comptime BATCH = 2

    var h = GaussianHead[IN, ACT].make[target="cpu", INIT=Zero]()
    var w = TileTensor(h.weight, row_major[IN, ACT]())
    w[0, 0] = 1.0
    w[0, 1] = 2.0
    w[1, 0] = 3.0
    w[1, 1] = 4.0

    # Pre-populate cache directly.
    h.cache.resize(BATCH * IN, 0.0)
    var cache_view = TileTensor(h.cache, row_major[BATCH, IN]())
    cache_view[0, 0] = 1.0
    cache_view[0, 1] = 1.0
    cache_view[1, 0] = 2.0
    cache_view[1, 1] = 3.0

    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for k in range(BATCH * 2 * ACT):
        go_buf[k] = 1.0
    for k in range(BATCH * IN):
        gi_buf[k] = -999.0

    var grad_out = TileTensor(go_buf, row_major[BATCH, 2 * ACT]())
    var grad_in = TileTensor(gi_buf, row_major[BATCH, IN]())

    h.backward["cpu", BATCH](grad_out, grad_in)

    assert_equal(grad_in[0, 0], 3.0)
    assert_equal(grad_in[0, 1], 7.0)
    assert_equal(grad_in[1, 0], 3.0)
    assert_equal(grad_in[1, 1], 7.0)

    var gw = TileTensor(h.grad_w, row_major[IN, ACT]())
    assert_equal(gw[0, 0], 3.0)
    assert_equal(gw[0, 1], 3.0)
    assert_equal(gw[1, 0], 4.0)
    assert_equal(gw[1, 1], 4.0)

    var gb = TileTensor(h.grad_b, row_major[ACT]())
    assert_equal(gb[0], 2.0)
    assert_equal(gb[1], 2.0)

    var gls = TileTensor(h.grad_ls, row_major[ACT]())
    assert_equal(gls[0], 2.0)
    assert_equal(gls[1], 2.0)

    go_buf.free()
    gi_buf.free()
    print("  test_backward PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_grad_log_std_reduces_over_batch
# ──────────────────────────────────────────────────────────────────────────


def test_grad_log_std_reduces_over_batch() raises:
    """Verify grad_log_std[j] = sum over batch of grad_output[b, ACT+j].
    Use non-uniform per-batch grads to make sure the reduction is real."""
    comptime IN = 1
    comptime ACT = 1
    comptime BATCH = 4

    var h = GaussianHead[IN, ACT].make[target="cpu", INIT=Zero]()
    h.cache.resize(BATCH * IN, 0.0)

    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    # grad_output: per-batch log_std-column values [0.1, 0.2, 0.3, 0.4]
    # mu-column zero so other accumulators stay 0.
    for b in range(BATCH):
        go_buf[b * 2 * ACT + 0] = 0.0  # mu col
        go_buf[b * 2 * ACT + ACT + 0] = Scalar[DT](0.1 * Float64(b + 1))
    for k in range(BATCH * IN):
        gi_buf[k] = 0.0

    var grad_out = TileTensor(go_buf, row_major[BATCH, 2 * ACT]())
    var grad_in = TileTensor(gi_buf, row_major[BATCH, IN]())
    h.backward["cpu", BATCH](grad_out, grad_in)

    var gls = TileTensor(h.grad_ls, row_major[ACT]())
    # Expected: 0.1 + 0.2 + 0.3 + 0.4 = 1.0
    var diff = fabs(Float64(gls[0]) - 1.0)
    assert_true(diff < 1e-5)

    go_buf.free()
    gi_buf.free()
    print("  test_grad_log_std_reduces_over_batch PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_for_each_param
# ──────────────────────────────────────────────────────────────────────────


def test_for_each_param() raises:
    """Walk yields three params: weight (decay=True), bias (decay=False),
    log_std (decay=False) with correct sizes."""
    var h = GaussianHead[4, 3].make[target="cpu", INIT=Zero]()
    var v = WalkVisitor()
    h.for_each_param["cpu"](String("head"), v)

    assert_equal(len(v.names), 3)
    assert_equal(v.names[0], String("head.weight"))
    assert_equal(v.names[1], String("head.bias"))
    assert_equal(v.names[2], String("head.log_std"))
    assert_equal(v.sizes[0], 12)  # 4 * 3
    assert_equal(v.sizes[1], 3)
    assert_equal(v.sizes[2], 3)
    assert_equal(v.decays[0], True)
    assert_equal(v.decays[1], False)
    assert_equal(v.decays[2], False)
    print("  test_for_each_param PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_zero_grad
# ──────────────────────────────────────────────────────────────────────────


def test_zero_grad() raises:
    """The zero_grad call clears all three grad buffers."""
    var h = GaussianHead[2, 2].make[target="cpu", INIT=Zero]()
    var gw = TileTensor(h.grad_w, row_major[2, 2]())
    var gb = TileTensor(h.grad_b, row_major[2]())
    var gls = TileTensor(h.grad_ls, row_major[2]())
    for i in range(2):
        for j in range(2):
            gw[i, j] = 5.0
        gb[i] = 7.0
        gls[i] = 9.0

    h.zero_grad["cpu"]()

    var gw2 = TileTensor(h.grad_w, row_major[2, 2]())
    var gb2 = TileTensor(h.grad_b, row_major[2]())
    var gls2 = TileTensor(h.grad_ls, row_major[2]())
    for i in range(2):
        for j in range(2):
            assert_equal(gw2[i, j], 0.0)
        assert_equal(gb2[i], 0.0)
        assert_equal(gls2[i], 0.0)
    print("  test_zero_grad PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_gradcheck_fd — finite-difference gradcheck on every param + input.
#
# Loss = sum_b sum_j alpha[b, j] * output[b, j]   (arbitrary linear functional)
# So d_loss/d_output[b, j] = alpha[b, j], which we set as grad_output.
# Then verify d_loss/d_param matches numerical FD on the same closed form.
# ──────────────────────────────────────────────────────────────────────────


def test_gradcheck_fd() raises:
    """FD gradcheck. log_std initialized strictly inside [-5, 2] so the
    clamp is inactive and analytical grad matches FD."""
    comptime IN = 3
    comptime ACT = 2
    comptime BATCH = 4
    comptime EPS: Scalar[DT] = 1e-2
    comptime TOL_REL: Scalar[DT] = 1e-2

    var h = GaussianHead[IN, ACT].make[target="cpu", INIT=Zero]()

    # Deterministic params (Zero init would give only-bias FD signal).
    var w = TileTensor(h.weight, row_major[IN, ACT]())
    var b = TileTensor(h.bias, row_major[ACT]())
    var ls = TileTensor(h.log_std, row_major[ACT]())
    for i in range(IN):
        for j in range(ACT):
            w[i, j] = Scalar[DT](0.1 * Float64(i * ACT + j + 1))
    for j in range(ACT):
        b[j] = Scalar[DT](0.05 * Float64(j + 1))
        ls[j] = Scalar[DT](0.5 - 0.1 * Float64(j))  # 0.5, 0.4 — inside bounds

    var in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var alpha_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * 2 * ACT
    )
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * 2 * ACT
    )
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * 2 * ACT
    )
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)

    for bi in range(BATCH):
        for i in range(IN):
            in_buf[bi * IN + i] = Scalar[DT](
                0.1 + 0.01 * Float64(bi * IN + i)
            )
        for j in range(2 * ACT):
            alpha_buf[bi * 2 * ACT + j] = Scalar[DT](
                0.2 + 0.03 * Float64(bi * 2 * ACT + j)
            )

    var input = TileTensor(in_buf, row_major[BATCH, IN]())
    var alpha = TileTensor(alpha_buf, row_major[BATCH, 2 * ACT]())
    var output = TileTensor(out_buf, row_major[BATCH, 2 * ACT]())
    var grad_out = TileTensor(go_buf, row_major[BATCH, 2 * ACT]())
    var grad_in = TileTensor(gi_buf, row_major[BATCH, IN]())

    # Analytical gradients via one forward + backward.
    h.forward["cpu", BATCH](input, output)
    for k in range(BATCH * 2 * ACT):
        go_buf[k] = alpha_buf[k]
    for k in range(BATCH * IN):
        gi_buf[k] = 0.0
    h.backward["cpu", BATCH](grad_out, grad_in)

    var gw = TileTensor(h.grad_w, row_major[IN, ACT]())
    var gb = TileTensor(h.grad_b, row_major[ACT]())
    var gls = TileTensor(h.grad_ls, row_major[ACT]())

    # Helper closure-like macro via a callable function in test scope is
    # awkward in Mojo; inline the FD loop body. Loss L = Σ alpha[b,j] * output[b,j].
    var max_rel: Scalar[DT] = 0.0

    # FD on W.
    for i in range(IN):
        for j in range(ACT):
            var saved = w[i, j]
            w[i, j] = saved + EPS
            h.forward["cpu", BATCH](input, output)
            var Lp: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for jj in range(2 * ACT):
                    Lp += alpha[b2, jj] * output[b2, jj]
            w[i, j] = saved - EPS
            h.forward["cpu", BATCH](input, output)
            var Lm: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for jj in range(2 * ACT):
                    Lm += alpha[b2, jj] * output[b2, jj]
            w[i, j] = saved
            var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
            var an = gw[i, j]
            var denom = fabs(an) + Scalar[DT](1e-6)
            var rel = fabs(fd - an) / denom
            if rel > max_rel:
                max_rel = rel

    # FD on b.
    for j in range(ACT):
        var saved = b[j]
        b[j] = saved + EPS
        h.forward["cpu", BATCH](input, output)
        var Lp: Scalar[DT] = 0.0
        for b2 in range(BATCH):
            for jj in range(2 * ACT):
                Lp += alpha[b2, jj] * output[b2, jj]
        b[j] = saved - EPS
        h.forward["cpu", BATCH](input, output)
        var Lm: Scalar[DT] = 0.0
        for b2 in range(BATCH):
            for jj in range(2 * ACT):
                Lm += alpha[b2, jj] * output[b2, jj]
        b[j] = saved
        var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
        var an = gb[j]
        var denom = fabs(an) + Scalar[DT](1e-6)
        var rel = fabs(fd - an) / denom
        if rel > max_rel:
            max_rel = rel

    # FD on log_std.
    for j in range(ACT):
        var saved = ls[j]
        ls[j] = saved + EPS
        h.forward["cpu", BATCH](input, output)
        var Lp: Scalar[DT] = 0.0
        for b2 in range(BATCH):
            for jj in range(2 * ACT):
                Lp += alpha[b2, jj] * output[b2, jj]
        ls[j] = saved - EPS
        h.forward["cpu", BATCH](input, output)
        var Lm: Scalar[DT] = 0.0
        for b2 in range(BATCH):
            for jj in range(2 * ACT):
                Lm += alpha[b2, jj] * output[b2, jj]
        ls[j] = saved
        var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
        var an = gls[j]
        var denom = fabs(an) + Scalar[DT](1e-6)
        var rel = fabs(fd - an) / denom
        if rel > max_rel:
            max_rel = rel

    # FD on input → grad_input.
    for bi in range(BATCH):
        for i in range(IN):
            var saved = in_buf[bi * IN + i]
            in_buf[bi * IN + i] = saved + EPS
            h.forward["cpu", BATCH](input, output)
            var Lp: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for jj in range(2 * ACT):
                    Lp += alpha[b2, jj] * output[b2, jj]
            in_buf[bi * IN + i] = saved - EPS
            h.forward["cpu", BATCH](input, output)
            var Lm: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for jj in range(2 * ACT):
                    Lm += alpha[b2, jj] * output[b2, jj]
            in_buf[bi * IN + i] = saved
            var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
            var an = grad_in[bi, i]
            var denom = fabs(an) + Scalar[DT](1e-6)
            var rel = fabs(fd - an) / denom
            if rel > max_rel:
                max_rel = rel

    print("  FD gradcheck max_rel = ", max_rel)
    assert_true(max_rel < TOL_REL, "gradcheck failed")

    in_buf.free()
    alpha_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_gradcheck_fd PASSED")


# ──────────────────────────────────────────────────────────────────────────
def main() raises:
    print("=" * 60)
    print("nn2 GaussianHead unit tests (CPU, Phase 6.2)")
    print("=" * 60)
    test_forward()
    test_log_std_clamp()
    test_backward()
    test_grad_log_std_reduces_over_batch()
    test_for_each_param()
    test_zero_grad()
    test_gradcheck_fd()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
