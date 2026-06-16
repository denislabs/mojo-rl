"""LSTMCell step-API FD gradcheck (CPU).

Single-step forward/backward of the recurrent cell, with loss
L = Σ go_h⊙h_t + go_c⊙c_t (so dh=go_h, dc=go_c). Central differences
validate dx, dh_prev, dc_prev and the parameter grads (dW_ih, dW_hh, db).
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.lstm_cell import LSTMCell
from mojo_rl.nn.initializer import Xavier


comptime BATCH = 2
comptime IN_ = 3
comptime H = 4
comptime Cell = LSTMCell[IN_, H]


def _loss(
    mut cell: Cell,
    x: UnsafePointer[Scalar[DT], MutAnyOrigin],
    hp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    cp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    go_h: UnsafePointer[Scalar[DT], MutAnyOrigin],
    go_c: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises -> Scalar[DT]:
    var ht: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var ct: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var x_t = TileTensor(x, row_major[BATCH, IN_]())
    var hp_t = TileTensor(hp, row_major[BATCH, H]())
    var cp_t = TileTensor(cp, row_major[BATCH, H]())
    var ht_t = TileTensor(ht, row_major[BATCH, H]())
    var ct_t = TileTensor(ct, row_major[BATCH, H]())
    cell.step_forward_no_cache["cpu", BATCH](x_t, hp_t, cp_t, ht_t, ct_t)
    var s: Scalar[DT] = 0.0
    for i in range(BATCH * H):
        s += go_h[i] * ht[i] + go_c[i] * ct[i]
    ht.free(); ct.free()
    return s


def main() raises:
    print("test_lstm_cell (FD gradcheck) ...")
    var eps = Scalar[DT](1e-3)
    var tol = Scalar[DT](2e-2)
    var cell = Cell.make[target="cpu", INIT=Xavier]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_)
    var hp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var cp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var go_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var go_c: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    for i in range(BATCH * IN_):
        x[i] = Scalar[DT](-0.3 + 0.17 * Float64(i))
    for i in range(BATCH * H):
        hp[i] = Scalar[DT](0.1 - 0.09 * Float64(i))
        cp[i] = Scalar[DT](-0.2 + 0.05 * Float64(i))
        go_h[i] = Scalar[DT](0.4 + 0.1 * Float64(i))
        go_c[i] = Scalar[DT](0.2 - 0.06 * Float64(i))

    # Analytic backward (with cache).
    var ht: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var ct: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var cache: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * Cell.CACHE_SIZE)
    var dx: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_)
    var dhp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var dcp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)

    var x_t = TileTensor(x, row_major[BATCH, IN_]())
    var hp_t = TileTensor(hp, row_major[BATCH, H]())
    var cp_t = TileTensor(cp, row_major[BATCH, H]())
    var ht_t = TileTensor(ht, row_major[BATCH, H]())
    var ct_t = TileTensor(ct, row_major[BATCH, H]())
    var cache_t = TileTensor(cache, row_major[BATCH, Cell.CACHE_SIZE]())
    var goh_t = TileTensor(go_h, row_major[BATCH, H]())
    var goc_t = TileTensor(go_c, row_major[BATCH, H]())
    var dx_t = TileTensor(dx, row_major[BATCH, IN_]())
    var dhp_t = TileTensor(dhp, row_major[BATCH, H]())
    var dcp_t = TileTensor(dcp, row_major[BATCH, H]())

    cell.zero_grad["cpu"]()
    cell.step_forward["cpu", BATCH](x_t, hp_t, cp_t, ht_t, ct_t, cache_t)
    cell.step_backward["cpu", BATCH](
        goh_t, goc_t, x_t, hp_t, cp_t, cache_t, dx_t, dhp_t, dcp_t
    )

    # FD on x, h_prev, c_prev.
    var max_in: Scalar[DT] = 0.0
    for i in range(BATCH * IN_):
        var s = x[i]
        x[i] = s + eps
        var lp = _loss(cell, x, hp, cp, go_h, go_c)
        x[i] = s - eps
        var ln = _loss(cell, x, hp, cp, go_h, go_c)
        x[i] = s
        var fd = (lp - ln) / (Scalar[DT](2.0) * eps)
        var d = dx[i] - fd
        max_in = max(max_in, d if d >= 0 else -d)

    var max_h: Scalar[DT] = 0.0
    var max_c: Scalar[DT] = 0.0
    for i in range(BATCH * H):
        var s = hp[i]
        hp[i] = s + eps
        var lp = _loss(cell, x, hp, cp, go_h, go_c)
        hp[i] = s - eps
        var ln = _loss(cell, x, hp, cp, go_h, go_c)
        hp[i] = s
        var fd = (lp - ln) / (Scalar[DT](2.0) * eps)
        var d = dhp[i] - fd
        max_h = max(max_h, d if d >= 0 else -d)

        var sc = cp[i]
        cp[i] = sc + eps
        var lp2 = _loss(cell, x, hp, cp, go_h, go_c)
        cp[i] = sc - eps
        var ln2 = _loss(cell, x, hp, cp, go_h, go_c)
        cp[i] = sc
        var fd2 = (lp2 - ln2) / (Scalar[DT](2.0) * eps)
        var d2 = dcp[i] - fd2
        max_c = max(max_c, d2 if d2 >= 0 else -d2)

    print("  max|dx-fd| =", max_in, " max|dh_prev-fd| =", max_h, " max|dc_prev-fd| =", max_c)
    assert_true(max_in < tol, "LSTM dx FD failed")
    assert_true(max_h < tol, "LSTM dh_prev FD failed")
    assert_true(max_c < tol, "LSTM dc_prev FD failed")

    # FD on parameters (W_ih, W_hh, b) via the Param value Lists.
    var max_wih: Scalar[DT] = 0.0
    for idx in range(Cell.W_IH_SIZE):
        var s = cell.W_ih.val.cpu[idx]
        cell.W_ih.val.cpu[idx] = s + eps
        var lp = _loss(cell, x, hp, cp, go_h, go_c)
        cell.W_ih.val.cpu[idx] = s - eps
        var ln = _loss(cell, x, hp, cp, go_h, go_c)
        cell.W_ih.val.cpu[idx] = s
        var fd = (lp - ln) / (Scalar[DT](2.0) * eps)
        var d = cell.W_ih.grd.cpu[idx] - fd
        max_wih = max(max_wih, d if d >= 0 else -d)

    var max_whh: Scalar[DT] = 0.0
    for idx in range(Cell.W_HH_SIZE):
        var s = cell.W_hh.val.cpu[idx]
        cell.W_hh.val.cpu[idx] = s + eps
        var lp = _loss(cell, x, hp, cp, go_h, go_c)
        cell.W_hh.val.cpu[idx] = s - eps
        var ln = _loss(cell, x, hp, cp, go_h, go_c)
        cell.W_hh.val.cpu[idx] = s
        var fd = (lp - ln) / (Scalar[DT](2.0) * eps)
        var d = cell.W_hh.grd.cpu[idx] - fd
        max_whh = max(max_whh, d if d >= 0 else -d)

    var max_b: Scalar[DT] = 0.0
    for idx in range(Cell.B_SIZE):
        var s = cell.b.val.cpu[idx]
        cell.b.val.cpu[idx] = s + eps
        var lp = _loss(cell, x, hp, cp, go_h, go_c)
        cell.b.val.cpu[idx] = s - eps
        var ln = _loss(cell, x, hp, cp, go_h, go_c)
        cell.b.val.cpu[idx] = s
        var fd = (lp - ln) / (Scalar[DT](2.0) * eps)
        var d = cell.b.grd.cpu[idx] - fd
        max_b = max(max_b, d if d >= 0 else -d)

    print("  max|dW_ih-fd| =", max_wih, " max|dW_hh-fd| =", max_whh, " max|db-fd| =", max_b)
    assert_true(max_wih < tol, "LSTM dW_ih FD failed")
    assert_true(max_whh < tol, "LSTM dW_hh FD failed")
    assert_true(max_b < tol, "LSTM db FD failed")
    print("  ok")
