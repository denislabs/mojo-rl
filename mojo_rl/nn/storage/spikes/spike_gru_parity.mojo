"""GRUCell parity (A2): forward (out) + vjp (dx, dh, dW_ih, dW_hh, db_ih, db_hh)
vs INDEPENDENT naive oracles. CPU + GPU. GPU is TF32-aware (gate pre-acts +
dx/dh/dW now go through max_matmul → ~1e-3 on NVIDIA tensor cores).

Run: pixi run -e apple mojo run -I . mojo_rl/nn/storage/spikes/spike_gru_parity.mojo
"""

from std.math import exp, tanh
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.primitives.gru_cell import GRUCell
from mojo_rl.nn.storage.core.initializer import Deterministic


def _sig(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


def _run[
    target: StaticString, IN_: Int, H: Int, B: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](2e-4) if target == "cpu" else Scalar[DT](2e-3)
    comptime TH = 3 * H
    var g = GRUCell[IN_, H].make[target, Deterministic](ctx)

    var x = Tensor.alloc(B * IN_)
    for i in range(B * IN_):
        x.data[i] = Scalar[DT]((i % 9) - 4) * 0.11
    var h = Tensor.alloc(B * H)
    for i in range(B * H):
        h.data[i] = Scalar[DT]((i % 7) - 3) * 0.13
    var go = Tensor.alloc(B * H)
    for i in range(B * H):
        go.data[i] = Scalar[DT]((i % 6) - 3) * 0.19

    var wih = List[Scalar[DT]](length=IN_ * TH, fill=Scalar[DT](0))
    var whh = List[Scalar[DT]](length=H * TH, fill=Scalar[DT](0))
    var bih = List[Scalar[DT]](length=TH, fill=Scalar[DT](0))
    var bhh = List[Scalar[DT]](length=TH, fill=Scalar[DT](0))
    for i in range(IN_ * TH):
        wih[i] = g.W_ih.val.data[i]
    for i in range(H * TH):
        whh[i] = g.W_hh.val.data[i]
    for i in range(TH):
        bih[i] = g.b_ih.val.data[i]
        bhh[i] = g.b_hh.val.data[i]

    # ---- forward oracle ----
    var r_out = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    var cr = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    var cz = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    var cn = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    var chn = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    for b in range(B):
        for j in range(H):
            var ixr = bih[j]
            var ixz = bih[H + j]
            var ixn = bih[2 * H + j]
            var hxr = bhh[j]
            var hxz = bhh[H + j]
            var hxn = bhh[2 * H + j]
            for ii in range(IN_):
                ixr += x.data[b * IN_ + ii] * wih[ii * TH + j]
                ixz += x.data[b * IN_ + ii] * wih[ii * TH + H + j]
                ixn += x.data[b * IN_ + ii] * wih[ii * TH + 2 * H + j]
            for kk in range(H):
                hxr += h.data[b * H + kk] * whh[kk * TH + j]
                hxz += h.data[b * H + kk] * whh[kk * TH + H + j]
                hxn += h.data[b * H + kk] * whh[kk * TH + 2 * H + j]
            var rg = _sig(ixr + hxr)
            var zg = _sig(ixz + hxz)
            var ng = tanh(ixn + rg * hxn)
            cr[b * H + j] = rg
            cz[b * H + j] = zg
            cn[b * H + j] = ng
            chn[b * H + j] = hxn
            r_out[b * H + j] = (Scalar[DT](1.0) - zg) * ng + zg * h.data[b * H + j]

    # ---- backward oracle ----
    var dix = List[Scalar[DT]](length=B * TH, fill=Scalar[DT](0))
    var dhx = List[Scalar[DT]](length=B * TH, fill=Scalar[DT](0))
    for b in range(B):
        for j in range(H):
            var rg = cr[b * H + j]
            var zg = cz[b * H + j]
            var ng = cn[b * H + j]
            var hn = chn[b * H + j]
            var dh_now = go.data[b * H + j]
            var dz = dh_now * (h.data[b * H + j] - ng)
            var dn = dh_now * (Scalar[DT](1.0) - zg)
            var d_pre_n = dn * (Scalar[DT](1.0) - ng * ng)
            var d_hn = d_pre_n * rg
            var d_pre_r = (d_pre_n * hn) * rg * (Scalar[DT](1.0) - rg)
            var d_pre_z = dz * zg * (Scalar[DT](1.0) - zg)
            dix[b * TH + j] = d_pre_r
            dix[b * TH + H + j] = d_pre_z
            dix[b * TH + 2 * H + j] = d_pre_n
            dhx[b * TH + j] = d_pre_r
            dhx[b * TH + H + j] = d_pre_z
            dhx[b * TH + 2 * H + j] = d_hn
    var r_dx = List[Scalar[DT]](length=B * IN_, fill=Scalar[DT](0))
    var r_dh = List[Scalar[DT]](length=B * H, fill=Scalar[DT](0))
    var r_dwih = List[Scalar[DT]](length=IN_ * TH, fill=Scalar[DT](0))
    var r_dwhh = List[Scalar[DT]](length=H * TH, fill=Scalar[DT](0))
    var r_dbih = List[Scalar[DT]](length=TH, fill=Scalar[DT](0))
    var r_dbhh = List[Scalar[DT]](length=TH, fill=Scalar[DT](0))
    for b in range(B):
        for ii in range(IN_):
            var acc = Scalar[DT](0)
            for c in range(TH):
                acc += dix[b * TH + c] * wih[ii * TH + c]
            r_dx[b * IN_ + ii] = acc
        for kk in range(H):
            var acc = Scalar[DT](0)
            for c in range(TH):
                acc += dhx[b * TH + c] * whh[kk * TH + c]
            r_dh[b * H + kk] = acc + go.data[b * H + kk] * cz[b * H + kk]
    for ii in range(IN_):
        for c in range(TH):
            var acc = Scalar[DT](0)
            for b in range(B):
                acc += x.data[b * IN_ + ii] * dix[b * TH + c]
            r_dwih[ii * TH + c] = acc
    for kk in range(H):
        for c in range(TH):
            var acc = Scalar[DT](0)
            for b in range(B):
                acc += h.data[b * H + kk] * dhx[b * TH + c]
            r_dwhh[kk * TH + c] = acc
    for c in range(TH):
        var ai = Scalar[DT](0)
        var ah = Scalar[DT](0)
        for b in range(B):
            ai += dix[b * TH + c]
            ah += dhx[b * TH + c]
        r_dbih[c] = ai
        r_dbhh[c] = ah

    var out = Tensor.alloc(B * H)
    # Inputs / grad-inputs as TensorPacks (shared origin → TensorRefs[2]).
    var ins = TensorPack[2]()
    ins[0].ensure(B * IN_)
    ins[1].ensure(B * H)
    for i in range(B * IN_):
        ins[0].data[i] = x.data[i]
    for i in range(B * H):
        ins[1].data[i] = h.data[i]
    var gins = TensorPack[2]()
    comptime if target == "cpu":
        g.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), out, None)
        g.zero_grad["cpu"](None)
        g.vjp["cpu", B](
            TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](gins[0], gins[1]), None
        )
    else:
        var dctx = ctx.value()
        ins[0].upload(dctx); ins[1].upload(dctx); go.upload(dctx)
        g.forward["gpu", B](TensorRefs[2](ins[0], ins[1]), out, ctx)
        g.zero_grad["gpu"](ctx)
        g.vjp["gpu", B](
            TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](gins[0], gins[1]), ctx
        )
        out.download(dctx); gins[0].download(dctx); gins[1].download(dctx)
        g.W_ih.grd.download(dctx); g.W_hh.grd.download(dctx)
        g.b_ih.grd.download(dctx); g.b_hh.grd.download(dctx)
    ref dx = gins[0]
    ref dh = gins[1]

    def md(ref a: Tensor, ref b: List[Scalar[DT]], n: Int) -> Scalar[DT]:
        var m = Scalar[DT](0)
        for i in range(n):
            m = max(m, abs(a.data[i] - b[i]))
        return m

    var d_out = md(out, r_out, B * H)
    var d_dx = md(dx, r_dx, B * IN_)
    var d_dh = md(dh, r_dh, B * H)
    var d_wih = md(g.W_ih.grd, r_dwih, IN_ * TH)
    var d_whh = md(g.W_hh.grd, r_dwhh, H * TH)
    var d_bih = md(g.b_ih.grd, r_dbih, TH)
    var d_bhh = md(g.b_hh.grd, r_dbhh, TH)
    print(
        "  GRU[IN=", IN_, " H=", H, "]", target, " out=", d_out, " dx=", d_dx,
        " dh=", d_dh, " dWih=", d_wih, " dWhh=", d_whh, " dbih=", d_bih,
        " dbhh=", d_bhh,
    )
    return (
        d_out <= TOL and d_dx <= TOL and d_dh <= TOL and d_wih <= TOL
        and d_whh <= TOL and d_bih <= TOL and d_bhh <= TOL
    )


def main() raises:
    var c = DeviceContext()
    var ok = True
    print("GRUCell fwd+bwd parity (naive oracles):")
    ok = _run["cpu", 8, 6, 3](None) and ok
    ok = _run["gpu", 8, 6, 3](Optional(c)) and ok
    ok = _run["cpu", 64, 64, 16](None) and ok
    ok = _run["gpu", 64, 64, 16](Optional(c)) and ok
    print("GRU PARITY", "OK" if ok else "FAIL")
