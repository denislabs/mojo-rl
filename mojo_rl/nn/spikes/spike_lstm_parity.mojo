"""LSTMCell parity (A1): step_forward (h_t, c_t, cache) + step_backward
(dx, dh_prev, dc_prev, dW_ih, dW_hh, db) vs INDEPENDENT naive oracles.
CPU + GPU. GPU is TF32-aware (gate pre-activations + backward dx/dW now go
through max_matmul → ~1e-3 on NVIDIA tensor cores).

Run: pixi run -e apple mojo run -I . mojo_rl/nn/storage/spikes/spike_lstm_parity.mojo
"""

from std.math import exp, tanh
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.primitives.lstm_cell import LSTMCell
from mojo_rl.nn.core.initializer import Deterministic


def _sig(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


def _run[
    target: StaticString, IN_: Int, H: Int, B: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](2e-4) if target == "cpu" else Scalar[DT](2e-3)
    comptime FOURH = 4 * H
    comptime CACHE = 5 * H
    var cell = LSTMCell[IN_, H].make[target, Deterministic](ctx)

    var x = Tensor.alloc(B * IN_)
    for i in range(B * IN_):
        x.data[i] = Scalar[DT]((i % 9) - 4) * 0.11
    var h = Tensor.alloc(2 * B * H)  # slab0 prev, slab1 out
    var cc = Tensor.alloc(2 * B * H)
    for i in range(B * H):
        h.data[i] = Scalar[DT]((i % 7) - 3) * 0.13
        cc.data[i] = Scalar[DT]((i % 5) - 2) * 0.17
    var cache = Tensor.alloc(B * CACHE)
    # incoming grads dh / dc (w.r.t. h_t / c_t)
    var dh = Tensor.alloc(B * H)
    var dc = Tensor.alloc(B * H)
    for i in range(B * H):
        dh.data[i] = Scalar[DT]((i % 6) - 3) * 0.19
        dc.data[i] = Scalar[DT]((i % 4) - 2) * 0.23
    # separate h_prev / c_prev for backward (= forward slab0)
    var hp0 = Tensor.alloc(B * H)
    var cp0 = Tensor.alloc(B * H)
    for i in range(B * H):
        hp0.data[i] = h.data[i]
        cp0.data[i] = cc.data[i]

    # weights snapshot
    var wih = List[Scalar[DT]](length=IN_ * FOURH, fill=Scalar[DT](0))
    var whh = List[Scalar[DT]](length=H * FOURH, fill=Scalar[DT](0))
    var bb = List[Scalar[DT]](length=FOURH, fill=Scalar[DT](0))
    for i in range(IN_ * FOURH):
        wih[i] = cell.W_ih.val.data[i]
    for i in range(H * FOURH):
        whh[i] = cell.W_hh.val.data[i]
    for i in range(FOURH):
        bb[i] = cell.b.val.data[i]

    # ---- forward oracle (also yields the gate cache for backward) ----
    var oc = List[Scalar[DT]](length=B * CACHE, fill=Scalar[DT](0))  # i,f,g,o,tc
    var r_h = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    var r_c = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    for bi in range(B):
        for j in range(H):
            var pre = InlineArray[Scalar[DT], 4](fill=Scalar[DT](0))
            for gate in range(4):
                var k = gate * H + j
                var acc = bb[k]
                for ii in range(IN_):
                    acc += x.data[bi * IN_ + ii] * wih[ii * FOURH + k]
                for jj in range(H):
                    acc += h.data[bi * H + jj] * whh[jj * FOURH + k]
                pre[gate] = acc
            var i_v = _sig(pre[0])
            var f_v = _sig(pre[1])
            var g_v = tanh(pre[2])
            var o_v = _sig(pre[3])
            var c_new = f_v * cc.data[bi * H + j] + i_v * g_v
            var tc = tanh(c_new)
            r_h[bi * H + j] = o_v * tc
            r_c[bi * H + j] = c_new
            oc[bi * CACHE + j] = i_v
            oc[bi * CACHE + H + j] = f_v
            oc[bi * CACHE + 2 * H + j] = g_v
            oc[bi * CACHE + 3 * H + j] = o_v
            oc[bi * CACHE + 4 * H + j] = tc

    # ---- backward oracle ----
    var dcomb = List[Scalar[DT]](length=B * FOURH, fill=Scalar[DT](0))
    var r_dcp = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    for bi in range(B):
        for j in range(H):
            var i_v = oc[bi * CACHE + j]
            var f_v = oc[bi * CACHE + H + j]
            var g_v = oc[bi * CACHE + 2 * H + j]
            var o_v = oc[bi * CACHE + 3 * H + j]
            var tc = oc[bi * CACHE + 4 * H + j]
            var dh_j = dh.data[bi * H + j]
            var dc_j = dc.data[bi * H + j]
            var do_post = dh_j * tc
            var dc_total = dc_j + dh_j * o_v * (Scalar[DT](1.0) - tc * tc)
            r_dcp[bi * H + j] = dc_total * f_v
            dcomb[bi * FOURH + j] = (dc_total * g_v) * i_v * (Scalar[DT](1.0) - i_v)
            dcomb[bi * FOURH + H + j] = (dc_total * cp0.data[bi * H + j]) * f_v * (
                Scalar[DT](1.0) - f_v
            )
            dcomb[bi * FOURH + 2 * H + j] = (dc_total * i_v) * (
                Scalar[DT](1.0) - g_v * g_v
            )
            dcomb[bi * FOURH + 3 * H + j] = do_post * o_v * (Scalar[DT](1.0) - o_v)
    var r_dx = List[Scalar[DT]](length=B * IN_, fill=Scalar[DT](0))
    var r_dhp = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    var r_dwih = List[Scalar[DT]](length=IN_ * FOURH, fill=Scalar[DT](0))
    var r_dwhh = List[Scalar[DT]](length=H * FOURH, fill=Scalar[DT](0))
    var r_db = List[Scalar[DT]](length=FOURH, fill=Scalar[DT](0))
    for bi in range(B):
        for ii in range(IN_):
            var acc = Scalar[DT](0)
            for k in range(FOURH):
                acc += dcomb[bi * FOURH + k] * wih[ii * FOURH + k]
            r_dx[bi * IN_ + ii] = acc
        for jj in range(H):
            var acc = Scalar[DT](0)
            for k in range(FOURH):
                acc += dcomb[bi * FOURH + k] * whh[jj * FOURH + k]
            r_dhp[bi * H + jj] = acc
    for ii in range(IN_):
        for k in range(FOURH):
            var acc = Scalar[DT](0)
            for bi in range(B):
                acc += x.data[bi * IN_ + ii] * dcomb[bi * FOURH + k]
            r_dwih[ii * FOURH + k] = acc
    for jj in range(H):
        for k in range(FOURH):
            var acc = Scalar[DT](0)
            for bi in range(B):
                acc += hp0.data[bi * H + jj] * dcomb[bi * FOURH + k]
            r_dwhh[jj * FOURH + k] = acc
    for k in range(FOURH):
        var acc = Scalar[DT](0)
        for bi in range(B):
            acc += dcomb[bi * FOURH + k]
        r_db[k] = acc

    var dx = Tensor.alloc(B * IN_)
    var dhp = Tensor.alloc(B * H)
    var dcp = Tensor.alloc(B * H)
    comptime if target == "cpu":
        cell.step_forward["cpu", B](x, h, cc, cache, None, h_t_off=B * H, c_t_off=B * H)
        cell.zero_grad["cpu"](None)
        cell.step_backward["cpu", B](
            dh, dc, x, hp0, cp0, cache, dx, dhp, dcp, None
        )
    else:
        var dctx = ctx.value()
        x.upload(dctx); h.upload(dctx); cc.upload(dctx)
        dh.upload(dctx); dc.upload(dctx); hp0.upload(dctx); cp0.upload(dctx)
        cell.step_forward["gpu", B](x, h, cc, cache, ctx, h_t_off=B * H, c_t_off=B * H)
        cell.zero_grad["gpu"](ctx)
        cell.step_backward["gpu", B](
            dh, dc, x, hp0, cp0, cache, dx, dhp, dcp, ctx
        )
        h.download(dctx); cc.download(dctx)
        dx.download(dctx); dhp.download(dctx); dcp.download(dctx)
        cell.W_ih.grd.download(dctx); cell.W_hh.grd.download(dctx)
        cell.b.grd.download(dctx)

    def md(ref a: Tensor, ref b: List[Scalar[DT]], off: Int, n: Int) -> Scalar[DT]:
        var m = Scalar[DT](0)
        for i in range(n):
            m = max(m, abs(a.data[off + i] - b[i]))
        return m

    var d_h = md(h, r_h, B * H, B * H)
    var d_c = md(cc, r_c, B * H, B * H)
    var d_dx = md(dx, r_dx, 0, B * IN_)
    var d_dhp = md(dhp, r_dhp, 0, B * H)
    var d_dcp = md(dcp, r_dcp, 0, B * H)
    var d_dwih = md(cell.W_ih.grd, r_dwih, 0, IN_ * FOURH)
    var d_dwhh = md(cell.W_hh.grd, r_dwhh, 0, H * FOURH)
    var d_db = md(cell.b.grd, r_db, 0, FOURH)
    print(
        "  LSTM[IN=", IN_, " H=", H, "]", target, " fwd(h=", d_h, " c=", d_c,
        ") bwd(dx=", d_dx, " dhp=", d_dhp, " dcp=", d_dcp, " dWih=", d_dwih,
        " dWhh=", d_dwhh, " db=", d_db, ")",
    )
    return (
        d_h <= TOL and d_c <= TOL and d_dx <= TOL and d_dhp <= TOL
        and d_dcp <= TOL and d_dwih <= TOL and d_dwhh <= TOL and d_db <= TOL
    )


def main() raises:
    var c = DeviceContext()
    var ok = True
    print("LSTMCell fwd+bwd parity (naive oracles):")
    ok = _run["cpu", 8, 6, 3](None) and ok
    ok = _run["gpu", 8, 6, 3](Optional(c)) and ok
    ok = _run["cpu", 64, 64, 16](None) and ok
    ok = _run["gpu", 64, 64, 16](Optional(c)) and ok
    print("LSTM PARITY", "OK" if ok else "FAIL")
