"""D1 parity harness: LayerNormNoAffine (fwd+bwd), MinMaxNorm (fwd), SimNorm
(fwd) vs INDEPENDENT scalar oracles. CPU + GPU. Covers register-cache (small
DIM/GROUP) and fallback (large DIM/GROUP) paths.

Run: pixi run -e apple mojo run -I . mojo_rl/nn/storage/spikes/spike_norm_parity_d1.mojo
"""

from std.math import sqrt, exp
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.layer_norm_no_affine import LayerNormNoAffine
from mojo_rl.nn.storage.primitives.min_max_norm import MinMaxNorm
from mojo_rl.nn.storage.primitives.sim_norm import SimNorm
from mojo_rl.nn.storage.core.initializer import Deterministic

comptime LNNA_EPS_REF: Scalar[DT] = 1e-6
comptime MMN_EPS_REF: Scalar[DT] = 1e-5


def _fill_input(mut x: Tensor, n: Int):
    for i in range(n):
        x.data[i] = Scalar[DT]((i % 13) - 6) * 0.19


def _fill_go(mut g: Tensor, n: Int):
    for i in range(n):
        g.data[i] = Scalar[DT]((i % 7) - 3) * 0.23


def _md(ref a: Tensor, ref b: List[Scalar[DT]], n: Int) -> Scalar[DT]:
    var m = Scalar[DT](0)
    for i in range(n):
        var d = abs(a.data[i] - b[i])
        if d > m:
            m = d
    return m


def _run_lnna[
    target: StaticString, DIM: Int, B: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](2e-4)
    var N = B * DIM
    var m = LayerNormNoAffine[DIM].make[target, Deterministic](ctx)
    var x = Tensor.alloc(N)
    _fill_input(x, N)
    var go = Tensor.alloc(N)
    _fill_go(go, N)

    var r_out = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    var r_gi = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    var xhat = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    for b in range(B):
        var row = b * DIM
        var mean = Scalar[DT](0)
        for j in range(DIM):
            mean += x.data[row + j]
        mean /= Scalar[DT](DIM)
        var v = Scalar[DT](0)
        for j in range(DIM):
            var d = x.data[row + j] - mean
            v += d * d
        v /= Scalar[DT](DIM)
        var inv = Scalar[DT](1.0) / sqrt(v + LNNA_EPS_REF)
        var mg = Scalar[DT](0)
        var mgx = Scalar[DT](0)
        for j in range(DIM):
            var xh = (x.data[row + j] - mean) * inv
            xhat[row + j] = xh
            r_out[row + j] = xh
            mg += go.data[row + j]
            mgx += go.data[row + j] * xh
        mg /= Scalar[DT](DIM)
        mgx /= Scalar[DT](DIM)
        for j in range(DIM):
            r_gi[row + j] = inv * (go.data[row + j] - mg - xhat[row + j] * mgx)

    var out = Tensor.alloc(N)
    var gi = Tensor.alloc(N)
    comptime if target == "cpu":
        m.forward["cpu", B](TensorRefs[1](x), out, None)
        m.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    else:
        var c = ctx.value()
        x.upload(c)
        go.upload(c)
        m.forward["gpu", B](TensorRefs[1](x), out, ctx)
        m.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
        out.download(c)
        gi.download(c)
    var d_out = _md(out, r_out, N)
    var d_gi = _md(gi, r_gi, N)
    print("  LNNA[", DIM, "]", target, " d_out=", d_out, " d_gi=", d_gi)
    return d_out <= TOL and d_gi <= TOL


def _run_mmn[
    target: StaticString, DIM: Int, B: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](2e-4)
    var N = B * DIM
    var m = MinMaxNorm[DIM].make[target, Deterministic](ctx)
    var x = Tensor.alloc(N)
    _fill_input(x, N)

    var r_out = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    for b in range(B):
        var row = b * DIM
        var mn = x.data[row]
        var mx = x.data[row]
        for j in range(1, DIM):
            var v = x.data[row + j]
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        var s = mx - mn
        if s < MMN_EPS_REF:
            s = MMN_EPS_REF
        var inv_s = Scalar[DT](1.0) / s
        for j in range(DIM):
            r_out[row + j] = (x.data[row + j] - mn) * inv_s

    var out = Tensor.alloc(N)
    comptime if target == "cpu":
        m.forward["cpu", B](TensorRefs[1](x), out, None)
    else:
        var c = ctx.value()
        x.upload(c)
        m.forward["gpu", B](TensorRefs[1](x), out, ctx)
        out.download(c)
    var d_out = _md(out, r_out, N)
    print("  MMN[", DIM, "]", target, " d_out=", d_out)
    return d_out <= TOL


def _run_sim[
    target: StaticString, DIM: Int, GROUPS: Int, B: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](2e-4)
    comptime GS = DIM // GROUPS
    var N = B * DIM
    var m = SimNorm[DIM, GROUPS].make[target, Deterministic](ctx)
    var x = Tensor.alloc(N)
    _fill_input(x, N)

    var r_out = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    for b in range(B):
        for g in range(GROUPS):
            var base = b * DIM + g * GS
            var mx = x.data[base]
            for k in range(1, GS):
                if x.data[base + k] > mx:
                    mx = x.data[base + k]
            var se = Scalar[DT](0)
            for k in range(GS):
                se += exp(x.data[base + k] - mx)
            var inv = Scalar[DT](1.0) / se
            for k in range(GS):
                r_out[base + k] = exp(x.data[base + k] - mx) * inv

    var out = Tensor.alloc(N)
    comptime if target == "cpu":
        m.forward["cpu", B](TensorRefs[1](x), out, None)
    else:
        var c = ctx.value()
        x.upload(c)
        m.forward["gpu", B](TensorRefs[1](x), out, ctx)
        out.download(c)
    var d_out = _md(out, r_out, N)
    print("  SIM[", DIM, "/", GROUPS, "]", target, " d_out=", d_out)
    return d_out <= TOL


def main() raises:
    var c = DeviceContext()
    var ok = True
    print("D1 parity (reg-cache: small; fallback: large):")
    # LayerNormNoAffine: DIM=16 reg-cache, DIM=1152 fallback (ELEMS=9)
    ok = _run_lnna["cpu", 16, 3](None) and ok
    ok = _run_lnna["gpu", 16, 3](Optional(c)) and ok
    ok = _run_lnna["cpu", 1152, 3](None) and ok
    ok = _run_lnna["gpu", 1152, 3](Optional(c)) and ok
    # MinMaxNorm: DIM=16 reg-cache, DIM=1152 fallback
    ok = _run_mmn["cpu", 16, 3](None) and ok
    ok = _run_mmn["gpu", 16, 3](Optional(c)) and ok
    ok = _run_mmn["cpu", 1152, 3](None) and ok
    ok = _run_mmn["gpu", 1152, 3](Optional(c)) and ok
    # SimNorm: GROUP_SIZE=8 cached, GROUP_SIZE=48 fallback
    ok = _run_sim["cpu", 16, 2, 3](None) and ok
    ok = _run_sim["gpu", 16, 2, 3](Optional(c)) and ok
    ok = _run_sim["cpu", 96, 2, 3](None) and ok
    ok = _run_sim["gpu", 96, 2, 3](Optional(c)) and ok
    print("D1 NORM PARITY", "OK" if ok else "FAIL")
