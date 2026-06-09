"""GRUCell tests (Block D-6).

Covers:
  * Forward equals hand-computed PyTorch GRU math on a small fixed config
  * FD gradcheck against analytical backward for all parameters AND
    both inputs (x and h)
  * Backward writes consistent grads (sum-loss across batch)
"""

from std.math import abs as fabs, exp, tanh, log
from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.primitives.gru_cell import GRUCell
from mojo_rl.nn2.initializer import Kaiming


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    if x >= 0:
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))
    var e = exp(x)
    return e / (Scalar[DT](1.0) + e)


def test_forward_against_manual() raises:
    """Run forward and verify against hand-computed PyTorch GRU formula.

    Uses tiny IN=2, H=2, BATCH=2 — bookkeeping is small enough to walk
    the arithmetic by hand on paper, then re-implement in this test.
    """
    seed(0)
    comptime BATCH = 2
    comptime IN_DIM = 2
    comptime H = 2
    comptime THREE_H = 3 * H
    var g = GRUCell[IN_DIM, H].make[target="cpu", INIT=Kaiming]()

    var x_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    var h_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var out_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    for k in range(BATCH * IN_DIM):
        x_p[k] = Scalar[DT](0.1 * Float64(k + 1))
    for k in range(BATCH * H):
        h_p[k] = Scalar[DT](-0.2 + 0.05 * Float64(k))

    # Snapshot pointers to params for manual recomputation.
    var W_ih = g.W_ih.value_unsafe_ptr_cpu()
    var W_hh = g.W_hh.value_unsafe_ptr_cpu()
    var b_ih = g.b_ih.value_unsafe_ptr_cpu()
    var b_hh = g.b_hh.value_unsafe_ptr_cpu()

    var x_t = TileTensor(x_p, row_major[BATCH, IN_DIM]())
    var h_t = TileTensor(h_p, row_major[BATCH, H]())
    var out_t = TileTensor(out_p, row_major[BATCH, H]())
    g.forward["cpu", BATCH](
            TensorPack[2].of(x_t, h_t), output=out_t,
        )

    var max_err: Scalar[DT] = 0.0
    for b in range(BATCH):
        for col in range(H):
            var ir: Scalar[DT] = b_ih[col]
            var hr: Scalar[DT] = b_hh[col]
            for k in range(IN_DIM):
                ir += x_p[b * IN_DIM + k] * W_ih[k * THREE_H + col]
            for k in range(H):
                hr += h_p[b * H + k] * W_hh[k * THREE_H + col]
            var iz: Scalar[DT] = b_ih[H + col]
            var hz: Scalar[DT] = b_hh[H + col]
            for k in range(IN_DIM):
                iz += x_p[b * IN_DIM + k] * W_ih[k * THREE_H + H + col]
            for k in range(H):
                hz += h_p[b * H + k] * W_hh[k * THREE_H + H + col]
            var in_pre: Scalar[DT] = b_ih[2 * H + col]
            for k in range(IN_DIM):
                in_pre += x_p[b * IN_DIM + k] * W_ih[k * THREE_H + 2 * H + col]
            var hn_pre: Scalar[DT] = b_hh[2 * H + col]
            for k in range(H):
                hn_pre += h_p[b * H + k] * W_hh[k * THREE_H + 2 * H + col]
            var rg = _sigmoid(ir + hr)
            var zg = _sigmoid(iz + hz)
            var ng = tanh(in_pre + rg * hn_pre)
            var hp = (Scalar[DT](1.0) - zg) * ng + zg * h_p[b * H + col]
            var err = fabs(out_p[b * H + col] - hp)
            if err > max_err:
                max_err = err

    print("  GRUCell forward max_err vs manual = ", max_err)
    assert_true(max_err < Scalar[DT](1e-6), "forward mismatch")

    x_p.free()
    h_p.free()
    out_p.free()
    print("  test_forward_against_manual PASSED")


def test_backward_fd_gradcheck() raises:
    """FD gradcheck of all params + both inputs against analytical backward.
    Loss = sum_b sum_c output[b, c] → grad_output = 1 everywhere.
    """
    seed(1)
    comptime BATCH = 2
    comptime IN_DIM = 3
    comptime H = 3
    comptime EPS: Scalar[DT] = 1e-3
    comptime TOL: Scalar[DT] = 1e-2

    var g = GRUCell[IN_DIM, H].make[target="cpu", INIT=Kaiming]()
    var x_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    var h_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var out_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var go_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var dx_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    var dh_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    for k in range(BATCH * IN_DIM):
        x_p[k] = Scalar[DT](-0.2 + 0.15 * Float64(k))
    for k in range(BATCH * H):
        h_p[k] = Scalar[DT](0.1 - 0.07 * Float64(k))
    for k in range(BATCH * H):
        go_p[k] = 1.0

    var x_t  = TileTensor(x_p, row_major[BATCH, IN_DIM]())
    var h_t  = TileTensor(h_p, row_major[BATCH, H]())
    var out_t = TileTensor(out_p, row_major[BATCH, H]())
    var go_t = TileTensor(go_p, row_major[BATCH, H]())
    var dx_t = TileTensor(dx_p, row_major[BATCH, IN_DIM]())
    var dh_t = TileTensor(dh_p, row_major[BATCH, H]())

    # Zero param grads (fresh start).
    g.zero_grad["cpu"]()

    g.forward["cpu", BATCH](
            TensorPack[2].of(x_t, h_t), output=out_t,
        )
    g.vjp["cpu", BATCH](go_t, TensorPack[2].of(dx_t, dh_t))

    @parameter
    def loss_with_inputs() raises -> Scalar[DT]:
        g.forward["cpu", BATCH](
            TensorPack[2].of(x_t, h_t), output=out_t,
        )
        var L: Scalar[DT] = 0.0
        for k in range(BATCH * H):
            L += out_p[k]
        return L

    # ----- FD against x -----
    var max_x_err: Scalar[DT] = 0.0
    for k in range(BATCH * IN_DIM):
        var saved = x_p[k]
        x_p[k] = saved + EPS
        var Lp = loss_with_inputs()
        x_p[k] = saved - EPS
        var Lm = loss_with_inputs()
        x_p[k] = saved
        var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
        var an = dx_p[k]
        var err = fabs(fd - an)
        if err > max_x_err:
            max_x_err = err
    print("  GRU FD gradcheck d/dx max_abs = ", max_x_err)
    assert_true(max_x_err < TOL, "d/dx FD mismatch")

    # ----- FD against h -----
    var max_h_err: Scalar[DT] = 0.0
    for k in range(BATCH * H):
        var saved = h_p[k]
        h_p[k] = saved + EPS
        var Lp = loss_with_inputs()
        h_p[k] = saved - EPS
        var Lm = loss_with_inputs()
        h_p[k] = saved
        var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
        var an = dh_p[k]
        var err = fabs(fd - an)
        if err > max_h_err:
            max_h_err = err
    print("  GRU FD gradcheck d/dh max_abs = ", max_h_err)
    assert_true(max_h_err < TOL, "d/dh FD mismatch")

    # ----- FD against W_ih -----
    var W_ih = g.W_ih.value_unsafe_ptr_cpu()
    var dW_ih = g.W_ih.grad_unsafe_ptr_cpu()
    var max_W_ih_err: Scalar[DT] = 0.0
    for k in range(g.W_IH_SIZE):
        var saved = W_ih[k]
        W_ih[k] = saved + EPS
        var Lp = loss_with_inputs()
        W_ih[k] = saved - EPS
        var Lm = loss_with_inputs()
        W_ih[k] = saved
        var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
        var an = dW_ih[k]
        var err = fabs(fd - an)
        if err > max_W_ih_err:
            max_W_ih_err = err
    print("  GRU FD gradcheck d/dW_ih max_abs = ", max_W_ih_err)
    assert_true(max_W_ih_err < TOL, "d/dW_ih FD mismatch")

    # ----- FD against W_hh -----
    var W_hh = g.W_hh.value_unsafe_ptr_cpu()
    var dW_hh = g.W_hh.grad_unsafe_ptr_cpu()
    var max_W_hh_err: Scalar[DT] = 0.0
    for k in range(g.W_HH_SIZE):
        var saved = W_hh[k]
        W_hh[k] = saved + EPS
        var Lp = loss_with_inputs()
        W_hh[k] = saved - EPS
        var Lm = loss_with_inputs()
        W_hh[k] = saved
        var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
        var an = dW_hh[k]
        var err = fabs(fd - an)
        if err > max_W_hh_err:
            max_W_hh_err = err
    print("  GRU FD gradcheck d/dW_hh max_abs = ", max_W_hh_err)
    assert_true(max_W_hh_err < TOL, "d/dW_hh FD mismatch")

    # ----- FD against b_ih -----
    var b_ih = g.b_ih.value_unsafe_ptr_cpu()
    var db_ih = g.b_ih.grad_unsafe_ptr_cpu()
    var max_b_ih_err: Scalar[DT] = 0.0
    for k in range(g.B_IH_SIZE):
        var saved = b_ih[k]
        b_ih[k] = saved + EPS
        var Lp = loss_with_inputs()
        b_ih[k] = saved - EPS
        var Lm = loss_with_inputs()
        b_ih[k] = saved
        var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
        var an = db_ih[k]
        var err = fabs(fd - an)
        if err > max_b_ih_err:
            max_b_ih_err = err
    print("  GRU FD gradcheck d/db_ih max_abs = ", max_b_ih_err)
    assert_true(max_b_ih_err < TOL, "d/db_ih FD mismatch")

    # ----- FD against b_hh -----
    var b_hh = g.b_hh.value_unsafe_ptr_cpu()
    var db_hh = g.b_hh.grad_unsafe_ptr_cpu()
    var max_b_hh_err: Scalar[DT] = 0.0
    for k in range(g.B_IH_SIZE):
        var saved = b_hh[k]
        b_hh[k] = saved + EPS
        var Lp = loss_with_inputs()
        b_hh[k] = saved - EPS
        var Lm = loss_with_inputs()
        b_hh[k] = saved
        var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
        var an = db_hh[k]
        var err = fabs(fd - an)
        if err > max_b_hh_err:
            max_b_hh_err = err
    print("  GRU FD gradcheck d/db_hh max_abs = ", max_b_hh_err)
    assert_true(max_b_hh_err < TOL, "d/db_hh FD mismatch")

    x_p.free()
    h_p.free()
    out_p.free()
    go_p.free()
    dx_p.free()
    dh_p.free()
    print("  test_backward_fd_gradcheck PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 GRUCell tests (Block D-6)")
    print("=" * 60)
    test_forward_against_manual()
    test_backward_fd_gradcheck()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
