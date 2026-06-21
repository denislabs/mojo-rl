"""LayerNorm / RMSNorm parity harness: forward + vjp (d_input, d_gamma[, d_beta])
vs an INDEPENDENT scalar reference (plain mean/var loops — a different code path
than the SIMD/GPU kernels, so a true oracle, not a circular check). CPU + GPU.

Covers VEC=4 (DIM%4==0) and VEC=1 (DIM%4!=0) kernel paths.

Run: pixi run -e apple mojo run -I . mojo_rl/nn/storage/spikes/spike_norm_parity.mojo
"""

from std.math import sqrt
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.layer_norm import LayerNorm
from mojo_rl.nn.storage.primitives.rms_norm import RMSNorm
from mojo_rl.nn.storage.core.initializer import Deterministic


comptime LN_EPS_REF: Scalar[DT] = 1e-5
comptime RMS_EPS_REF: Scalar[DT] = 1e-4


def _fill_input(mut x: Tensor, n: Int):
    for i in range(n):
        x.data[i] = Scalar[DT]((i % 13) - 6) * 0.19


def _fill_go(mut g: Tensor, n: Int):
    for i in range(n):
        g.data[i] = Scalar[DT]((i % 7) - 3) * 0.23


def _max_diff(
    ref a: Tensor, ref b: List[Scalar[DT]], n: Int
) -> Scalar[DT]:
    var m = Scalar[DT](0)
    for i in range(n):
        var d = abs(a.data[i] - b[i])
        if d > m:
            m = d
    return m


# ───────────────────────── LayerNorm ─────────────────────────
def _run_ln[
    target: StaticString, DIM: Int, B: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](2e-4)
    var N = B * DIM
    var ln = LayerNorm[DIM].make[target, Deterministic](ctx)
    var gamma = List[Scalar[DT]](length=DIM, fill=Scalar[DT](0))
    var beta = List[Scalar[DT]](length=DIM, fill=Scalar[DT](0))
    for j in range(DIM):
        gamma[j] = Scalar[DT](1.0) + Scalar[DT]((j % 5) - 2) * 0.1
        beta[j] = Scalar[DT]((j % 3) - 1) * 0.05
        ln.gamma.val.data[j] = gamma[j]
        ln.beta.val.data[j] = beta[j]

    var x = Tensor.alloc(N)
    _fill_input(x, N)
    var go = Tensor.alloc(N)
    _fill_go(go, N)

    # ---- independent scalar oracle ----
    var ref_out = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    var ref_gi = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    var ref_dg = List[Scalar[DT]](length=DIM, fill=Scalar[DT](0))
    var ref_db = List[Scalar[DT]](length=DIM, fill=Scalar[DT](0))
    var xhat = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    for b in range(B):
        var row = b * DIM
        var mean = Scalar[DT](0)
        for j in range(DIM):
            mean += x.data[row + j]
        mean /= Scalar[DT](DIM)
        var var_ = Scalar[DT](0)
        for j in range(DIM):
            var d = x.data[row + j] - mean
            var_ += d * d
        var_ /= Scalar[DT](DIM)
        var inv = Scalar[DT](1.0) / sqrt(var_ + LN_EPS_REF)
        for j in range(DIM):
            var xh = (x.data[row + j] - mean) * inv
            xhat[row + j] = xh
            ref_out[row + j] = gamma[j] * xh + beta[j]
    for b in range(B):
        var row = b * DIM
        var inv = Scalar[DT](0)
        # recompute inv_std for this row
        var mean = Scalar[DT](0)
        for j in range(DIM):
            mean += x.data[row + j]
        mean /= Scalar[DT](DIM)
        var var_ = Scalar[DT](0)
        for j in range(DIM):
            var d = x.data[row + j] - mean
            var_ += d * d
        var_ /= Scalar[DT](DIM)
        inv = Scalar[DT](1.0) / sqrt(var_ + LN_EPS_REF)
        var mean_g = Scalar[DT](0)
        var mean_g_xhat = Scalar[DT](0)
        for j in range(DIM):
            var g = go.data[row + j] * gamma[j]
            mean_g += g
            mean_g_xhat += g * xhat[row + j]
        mean_g /= Scalar[DT](DIM)
        mean_g_xhat /= Scalar[DT](DIM)
        for j in range(DIM):
            var g = go.data[row + j] * gamma[j]
            ref_gi[row + j] = inv * (g - mean_g - xhat[row + j] * mean_g_xhat)
            ref_dg[j] += go.data[row + j] * xhat[row + j]
            ref_db[j] += go.data[row + j]

    # ---- run module ----
    var out = Tensor.alloc(N)
    var gi = Tensor.alloc(N)
    comptime if target == "cpu":
        ln.forward["cpu", B](TensorRefs[1](x), out, None)
        ln.zero_grad["cpu"](None)
        ln.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    else:
        var c = ctx.value()
        ln.gamma.val.upload(c)
        ln.beta.val.upload(c)
        x.upload(c)
        go.upload(c)
        ln.forward["gpu", B](TensorRefs[1](x), out, ctx)
        ln.zero_grad["gpu"](ctx)
        ln.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
        out.download(c)
        gi.download(c)
        ln.gamma.grd.download(c)
        ln.beta.grd.download(c)

    var d_out = _max_diff(out, ref_out, N)
    var d_gi = _max_diff(gi, ref_gi, N)
    var d_dg = _max_diff(ln.gamma.grd, ref_dg, DIM)
    var d_db = _max_diff(ln.beta.grd, ref_db, DIM)
    print(
        "  LN[", DIM, "]", target, " | d_out=", d_out, " d_gi=", d_gi,
        " d_dg=", d_dg, " d_db=", d_db,
    )
    return (d_out <= TOL and d_gi <= TOL and d_dg <= TOL and d_db <= TOL)


# ───────────────────────── RMSNorm ─────────────────────────
def _run_rms[
    target: StaticString, DIM: Int, B: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](2e-4)
    var N = B * DIM
    var rn = RMSNorm[DIM].make[target, Deterministic](ctx)
    var gamma = List[Scalar[DT]](length=DIM, fill=Scalar[DT](0))
    for j in range(DIM):
        gamma[j] = Scalar[DT](1.0) + Scalar[DT]((j % 5) - 2) * 0.1
        rn.gamma.val.data[j] = gamma[j]

    var x = Tensor.alloc(N)
    _fill_input(x, N)
    var go = Tensor.alloc(N)
    _fill_go(go, N)

    # ---- independent scalar oracle ----
    var ref_out = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    var ref_gi = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    var ref_dg = List[Scalar[DT]](length=DIM, fill=Scalar[DT](0))
    var norm = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    for b in range(B):
        var row = b * DIM
        var ms = Scalar[DT](0)
        for j in range(DIM):
            ms += x.data[row + j] * x.data[row + j]
        ms /= Scalar[DT](DIM)
        var inv = Scalar[DT](1.0) / sqrt(ms + RMS_EPS_REF)
        for j in range(DIM):
            var n = x.data[row + j] * inv
            norm[row + j] = n
            ref_out[row + j] = n * gamma[j]
    for b in range(B):
        var row = b * DIM
        var ms = Scalar[DT](0)
        for j in range(DIM):
            ms += x.data[row + j] * x.data[row + j]
        ms /= Scalar[DT](DIM)
        var inv = Scalar[DT](1.0) / sqrt(ms + RMS_EPS_REF)
        var R = Scalar[DT](0)
        for j in range(DIM):
            R += go.data[row + j] * gamma[j] * norm[row + j]
        for j in range(DIM):
            ref_gi[row + j] = inv * (
                go.data[row + j] * gamma[j]
                - norm[row + j] * R / Scalar[DT](DIM)
            )
            ref_dg[j] += go.data[row + j] * norm[row + j]

    var out = Tensor.alloc(N)
    var gi = Tensor.alloc(N)
    comptime if target == "cpu":
        rn.forward["cpu", B](TensorRefs[1](x), out, None)
        rn.zero_grad["cpu"](None)
        rn.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    else:
        var c = ctx.value()
        rn.gamma.val.upload(c)
        x.upload(c)
        go.upload(c)
        rn.forward["gpu", B](TensorRefs[1](x), out, ctx)
        rn.zero_grad["gpu"](ctx)
        rn.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
        out.download(c)
        gi.download(c)
        rn.gamma.grd.download(c)

    var d_out = _max_diff(out, ref_out, N)
    var d_gi = _max_diff(gi, ref_gi, N)
    var d_dg = _max_diff(rn.gamma.grd, ref_dg, DIM)
    print(
        "  RMS[", DIM, "]", target, " | d_out=", d_out, " d_gi=", d_gi,
        " d_dg=", d_dg,
    )
    return (d_out <= TOL and d_gi <= TOL and d_dg <= TOL)


def main() raises:
    var c = DeviceContext()
    var ok = True
    print("LayerNorm parity (DIM=16 → VEC=4, DIM=10 → VEC=1):")
    ok = _run_ln["cpu", 16, 3](None) and ok
    ok = _run_ln["gpu", 16, 3](Optional(c)) and ok
    ok = _run_ln["cpu", 10, 3](None) and ok
    ok = _run_ln["gpu", 10, 3](Optional(c)) and ok
    print("RMSNorm parity (DIM=16 → VEC=4, DIM=10 → VEC=1):")
    ok = _run_rms["cpu", 16, 3](None) and ok
    ok = _run_rms["gpu", 16, 3](Optional(c)) and ok
    ok = _run_rms["cpu", 10, 3](None) and ok
    ok = _run_rms["gpu", 10, 3](Optional(c)) and ok
    print("NORM PARITY", "OK" if ok else "FAIL")
