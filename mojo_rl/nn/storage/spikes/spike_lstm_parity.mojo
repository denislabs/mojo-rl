"""LSTMCell forward parity (A1): step_forward (h_t, c_t, cache) vs an
INDEPENDENT naive-preact oracle. CPU + GPU. GPU gate is TF32-aware (the gate
pre-activations now go through max_matmul, ~1e-3 on NVIDIA tensor cores).

Run: pixi run -e apple mojo run -I . mojo_rl/nn/storage/spikes/spike_lstm_parity.mojo
"""

from std.math import exp, tanh
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.primitives.lstm_cell import LSTMCell
from mojo_rl.nn.storage.core.initializer import Deterministic


def _sig(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


def _run[
    target: StaticString, IN_: Int, H: Int, B: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](2e-4) if target == "cpu" else Scalar[DT](2e-3)
    comptime FOURH = 4 * H
    comptime CACHE = 5 * H
    var cell = LSTMCell[IN_, H].make[target, Deterministic](ctx)

    # Deterministic inputs.
    var x = Tensor.alloc(B * IN_)
    for i in range(B * IN_):
        x.data[i] = Scalar[DT]((i % 9) - 4) * 0.11
    # h, c sized 2·B·H: slab 0 = prev, slab 1 = out.
    var h = Tensor.alloc(2 * B * H)
    var c = Tensor.alloc(2 * B * H)
    for i in range(B * H):
        h.data[i] = Scalar[DT]((i % 7) - 3) * 0.13
        c.data[i] = Scalar[DT]((i % 5) - 2) * 0.17
    var cache = Tensor.alloc(B * CACHE)

    # Snapshot weights (host copies are filled by Deterministic before upload).
    var wih = List[Scalar[DT]](length=IN_ * FOURH, fill=Scalar[DT](0))
    var whh = List[Scalar[DT]](length=H * FOURH, fill=Scalar[DT](0))
    var bb = List[Scalar[DT]](length=FOURH, fill=Scalar[DT](0))
    for i in range(IN_ * FOURH):
        wih[i] = cell.W_ih.val.data[i]
    for i in range(H * FOURH):
        whh[i] = cell.W_hh.val.data[i]
    for i in range(FOURH):
        bb[i] = cell.b.val.data[i]

    # ---- independent naive-preact oracle ----
    var r_h = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    var r_c = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    var r_cache = List[Scalar[DT]](length=B * CACHE, fill=Scalar[DT](0))
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
            var c_new = f_v * c.data[bi * H + j] + i_v * g_v
            var tc = tanh(c_new)
            r_h[bi * H + j] = o_v * tc
            r_c[bi * H + j] = c_new
            r_cache[bi * CACHE + j] = i_v
            r_cache[bi * CACHE + H + j] = f_v
            r_cache[bi * CACHE + 2 * H + j] = g_v
            r_cache[bi * CACHE + 3 * H + j] = o_v
            r_cache[bi * CACHE + 4 * H + j] = tc

    comptime if target == "cpu":
        cell.step_forward["cpu", B](
            x, h, c, cache, None,
            h_t_off=B * H, c_t_off=B * H,
        )
    else:
        var dc = ctx.value()
        x.upload(dc)
        h.upload(dc)
        c.upload(dc)
        cell.step_forward["gpu", B](
            x, h, c, cache, ctx,
            h_t_off=B * H, c_t_off=B * H,
        )
        h.download(dc)
        c.download(dc)
        cache.download(dc)

    var d_h = Scalar[DT](0)
    var d_c = Scalar[DT](0)
    for i in range(B * H):
        d_h = max(d_h, abs(h.data[B * H + i] - r_h[i]))
        d_c = max(d_c, abs(c.data[B * H + i] - r_c[i]))
    var d_cache = Scalar[DT](0)
    for i in range(B * CACHE):
        d_cache = max(d_cache, abs(cache.data[i] - r_cache[i]))
    print(
        "  LSTM[IN=", IN_, " H=", H, "]", target,
        " d_h=", d_h, " d_c=", d_c, " d_cache=", d_cache,
    )
    return d_h <= TOL and d_c <= TOL and d_cache <= TOL


def main() raises:
    var c = DeviceContext()
    var ok = True
    print("LSTMCell forward parity (naive-preact oracle):")
    ok = _run["cpu", 8, 6, 3](None) and ok
    ok = _run["gpu", 8, 6, 3](Optional(c)) and ok
    ok = _run["cpu", 64, 64, 16](None) and ok
    ok = _run["gpu", 64, 64, 16](Optional(c)) and ok
    print("LSTM FWD PARITY", "OK" if ok else "FAIL")
