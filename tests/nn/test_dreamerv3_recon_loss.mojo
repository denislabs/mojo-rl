"""SymlogMSELoss recon op — both modes (symlog default + sigmoid).

Validates the new SIGMOID branch (reference pixel recon = (sigmoid(pred)-tgt)²)
and confirms the symlog default is unchanged:
  - CPU forward golden (independent formula),
  - finite-diff gradcheck (analytic backward incl. the sigmoid chain rule),
  - GPU vs CPU parity (forward + grad_pred).

Run:
  pixi run -e apple mojo run -I . tests/nn/test_dreamerv3_recon_loss.mojo
"""

from std.math import exp, log1p
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.dreamerv3.wm_loss_ops import SymlogMSELoss


comptime OBS = 5
comptime B = 3
comptime M = B * OBS


def _symlog(x: Scalar[DT]) -> Scalar[DT]:
    var s = Scalar[DT](1.0) if x >= 0 else Scalar[DT](-1.0)
    return s * log1p(x if x >= 0 else -x)


def _sig(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


def _fill_pack(mut ins: TensorPack[2]) raises:
    ins[0].ensure(M)
    ins[1].ensure(M)
    for i in range(M):
        ins[0].data[i] = Scalar[DT]((i % 7) - 3) * 0.3          # pred (logits)
        ins[1].data[i] = Scalar[DT]((i * 13 % 11)) / 11.0       # tgt in [0,1]


def test_forward_golden() raises:
    print("test_forward_golden (symlog + sigmoid) ...")
    var sym = SymlogMSELoss[OBS].make["cpu", Deterministic]()
    var sig = SymlogMSELoss[OBS, True].make["cpu", Deterministic]()
    var ins = TensorPack[2]()
    _fill_pack(ins)
    var o_sym = Tensor.alloc(B)
    var o_sig = Tensor.alloc(B)
    sym.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), o_sym, None)
    sig.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), o_sig, None)

    var ok = True
    for b in range(B):
        var es: Scalar[DT] = 0
        var eg: Scalar[DT] = 0
        for k in range(OBS):
            var pv = ins[0].data[b * OBS + k]
            var tv = ins[1].data[b * OBS + k]
            var ds = pv - _symlog(tv)
            es += ds * ds
            var dg = _sig(pv) - tv
            eg += dg * dg
        if abs(o_sym.data[b] - es) > 1e-5 or abs(o_sig.data[b] - eg) > 1e-5:
            ok = False
    assert_true(ok, "recon forward golden")
    print("  ok")


def _loss_sum(mut m: SymlogMSELoss[OBS, True], mut ins: TensorPack[2]) raises -> Scalar[DT]:
    var o = Tensor.alloc(B)
    m.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), o, None)
    var s: Scalar[DT] = 0
    for b in range(B):
        s += o.data[b]
    return s


def test_gradcheck_sigmoid() raises:
    print("test_gradcheck_sigmoid (finite-diff vs analytic, sigmoid chain) ...")
    comptime H = Scalar[DT](3e-3)
    comptime TOL = Scalar[DT](2e-2)
    var m = SymlogMSELoss[OBS, True].make["cpu", Deterministic]()
    var ins = TensorPack[2]()
    _fill_pack(ins)
    var go = Tensor.alloc(B)
    for b in range(B):
        go.data[b] = 1.0
    var g = TensorPack[2]()
    m.vjp["cpu", B](
        TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](g[0], g[1]), None
    )

    var maxerr: Scalar[DT] = 0
    for j in range(M):
        var saved = ins[0].data[j]
        ins[0].data[j] = saved + H
        var lp = _loss_sum(m, ins)
        ins[0].data[j] = saved - H
        var lm = _loss_sum(m, ins)
        ins[0].data[j] = saved
        var num = (lp - lm) / (2 * H)
        var err = abs(num - g[0].data[j])
        if err > maxerr:
            maxerr = err
    print("  max|num-analytic| grad_pred:", maxerr)
    assert_true(maxerr < TOL, "sigmoid recon gradcheck")
    print("  ok")


def _parity[SIGMOID: Bool](name: String) raises:
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = SymlogMSELoss[OBS, SIGMOID].make["cpu", Deterministic]()
    var gpu = SymlogMSELoss[OBS, SIGMOID].make["gpu", Deterministic](Optional(c))
    var ins = TensorPack[2]()
    _fill_pack(ins)
    var go = Tensor.alloc(B)
    for b in range(B):
        go.data[b] = Scalar[DT](b + 1) * 0.5

    var c_out = Tensor.alloc(B)
    var c_g = TensorPack[2]()
    cpu.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), c_out, None)
    cpu.vjp["cpu", B](
        TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](c_g[0], c_g[1]), None
    )

    var gins = TensorPack[2]()
    gins[0].ensure(M)
    gins[1].ensure(M)
    for i in range(M):
        gins[0].data[i] = ins[0].data[i]
        gins[1].data[i] = ins[1].data[i]
    gins[0].upload(c)
    gins[1].upload(c)
    var ggo = Tensor.alloc(B)
    for b in range(B):
        ggo.data[b] = go.data[b]
    ggo.upload(c)
    var g_out = Tensor.alloc(B)
    var g_g = TensorPack[2]()
    gpu.forward["gpu", B](TensorRefs[2](gins[0], gins[1]), g_out, Optional(c))
    gpu.vjp["gpu", B](
        TensorRefs[2](gins[0], gins[1]), ggo, TensorRefs[2](g_g[0], g_g[1]), Optional(c)
    )
    g_out.download(c)
    g_g[0].download(c)

    var mo: Scalar[DT] = 0
    var mg: Scalar[DT] = 0
    for b in range(B):
        if abs(g_out.data[b] - c_out.data[b]) > mo:
            mo = abs(g_out.data[b] - c_out.data[b])
    for i in range(M):
        if abs(g_g[0].data[i] - c_g[0].data[i]) > mg:
            mg = abs(g_g[0].data[i] - c_g[0].data[i])
    print("  ", name, "GPU vs CPU: out", mo, " grad_pred", mg)
    assert_true(mo < TOL and mg < TOL, "recon GPU vs CPU " + name)


def test_gpu_vs_cpu() raises:
    print("test_gpu_vs_cpu (both modes) ...")
    _parity[False]("symlog")
    _parity[True]("sigmoid")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("SymlogMSELoss recon op — symlog default + sigmoid (reference pixel)")
    print("=" * 70)
    test_forward_golden()
    test_gradcheck_sigmoid()
    test_gpu_vs_cpu()
    print("ALL PASSED")
