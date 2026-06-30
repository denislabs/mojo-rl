"""Dreamer4 storage migration — CPU↔GPU parity gate.

Validates the two custom GPU paths of the migrated package:
  • `Dreamer4Tokenizer` forward (encoder + decoder transformer on device).
  • `Dreamer4Dynamics` forward + vjp (the bespoke front-end assembly /
    agent-token / param-grad kernels + the GPU child-module forwards/vjps).

Both are built with `Deterministic` init (no RNG → CPU and GPU params are
bit-identical), run on each target, and the outputs / input-grads / a param-grad
must agree to a tight tolerance.

Run: pixi run -e apple mojo run -I . tests/nn/test_dreamer4_storage_gpu.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic

from mojo_rl.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents.dreamer4.dynamics import Dreamer4Dynamics
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao


def _absdiff(a: List[Scalar[DT]], b: List[Scalar[DT]], n: Int) -> Float64:
    var m: Float64 = 0.0
    for i in range(n):
        var d = Float64(a[i]) - Float64(b[i])
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def _tokenizer_parity(c: DeviceContext) raises -> Float64:
    comptime DP = 4
    comptime D = 8
    comptime NH = 2
    comptime T = 2
    comptime L = 2
    comptime NP = 3
    comptime D_BOT = 4
    comptime HID = 8
    comptime DEPTH = 1
    comptime BATCH = 2
    comptime N = BATCH * NP * DP

    comptime TOK = Dreamer4Tokenizer[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, 0.0, 0.0, 0, True
    ]
    var tc = TOK.make["cpu", Deterministic](None)
    var tg = TOK.make["gpu", Deterministic](Optional(c))

    var inc = Tensor.alloc(N)
    var ing = Tensor.alloc(N)
    for i in range(N):
        var v = Scalar[DT]((i % 5) - 2) * 0.1
        inc.data[i] = v
        ing.data[i] = v
    ing.upload(c)

    var oc = Tensor.alloc(N)
    tc.forward["cpu", BATCH](TensorRefs[1](inc), oc, None)
    var og = Tensor.alloc_gpu(c, N)
    tg.forward["gpu", BATCH](TensorRefs[1](ing), og, Optional(c))
    og.download(c)
    return _absdiff(oc.data, og.data, N)


def _dynamics_parity(c: DeviceContext) raises -> Tuple[Float64, Float64, Float64]:
    comptime DSP = 4
    comptime NSP = 2
    comptime D = 8
    comptime NH = 2
    comptime T = 2
    comptime NREG = 1
    comptime HID = 8
    comptime DEPTH = 1
    comptime KMAX = 4
    comptime NAGENT = 1
    comptime BATCH = 4   # = 2 sequences × T=2
    comptime ND = NSP * DSP
    comptime AGD = NAGENT * D

    comptime DYN = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX, True, 0, 0, NAGENT,
    ]
    var dc = DYN.make["cpu", Deterministic](None)
    var dg = DYN.make["gpu", Deterministic](Optional(c))

    # control inputs (host)
    var sig = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](2))
    var step = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0))
    var agin = List[Scalar[DT]](length=BATCH * AGD, fill=Scalar[DT](0))
    var gh = List[Scalar[DT]](length=BATCH * AGD, fill=Scalar[DT](0))
    for i in range(BATCH * AGD):
        agin[i] = Scalar[DT]((i % 5) - 2) * 0.1
        gh[i] = Scalar[DT]((i % 3) - 1) * 0.1

    dc.set_indices(_mao(sig.unsafe_ptr()), _mao(step.unsafe_ptr()), BATCH)
    dg.set_indices(_mao(sig.unsafe_ptr()), _mao(step.unsafe_ptr()), BATCH)
    dc.set_agent_in(_mao(agin.unsafe_ptr()), BATCH)
    dg.set_agent_in(_mao(agin.unsafe_ptr()), BATCH)

    # forward
    var inc = Tensor.alloc(BATCH * ND)
    var ing = Tensor.alloc(BATCH * ND)
    for i in range(BATCH * ND):
        var v = Scalar[DT]((i % 7) - 3) * 0.1
        inc.data[i] = v
        ing.data[i] = v
    ing.upload(c)
    var oc = Tensor.alloc(BATCH * ND)
    dc.forward["cpu", BATCH](TensorRefs[1](inc), oc, None)
    var og = Tensor.alloc_gpu(c, BATCH * ND)
    dg.forward["gpu", BATCH](TensorRefs[1](ing), og, Optional(c))
    og.download(c)
    var d_out = _absdiff(oc.data, og.data, BATCH * ND)

    # vjp
    dc.set_grad_h(_mao(gh.unsafe_ptr()), BATCH)
    dg.set_grad_h(_mao(gh.unsafe_ptr()), BATCH)
    var goc = Tensor.alloc(BATCH * ND)
    var gog = Tensor.alloc(BATCH * ND)
    for i in range(BATCH * ND):
        var v = Scalar[DT]((i % 4) - 2) * 0.1
        goc.data[i] = v
        gog.data[i] = v
    gog.upload(c)
    var gic = Tensor.alloc(BATCH * ND)
    var gig = Tensor.alloc_gpu(c, BATCH * ND)
    dc.vjp["cpu", BATCH](TensorRefs[1](inc), goc, TensorRefs[1](gic), None)
    dg.vjp["gpu", BATCH](TensorRefs[1](ing), gog, TensorRefs[1](gig), Optional(c))
    gig.download(c)
    var d_gin = _absdiff(gic.data, gig.data, BATCH * ND)

    # a param grad: action_base
    dg.action_base.grd.download(c)
    var d_gw = _absdiff(dc.action_base.grd.data, dg.action_base.grd.data, D)
    return (d_out, d_gin, d_gw)


def main() raises:
    print("Dreamer4 storage CPU↔GPU parity gate")
    var c = DeviceContext()

    var dt = _tokenizer_parity(c)
    print("  tokenizer forward  max|Δ| =", dt)

    var dd = _dynamics_parity(c)
    print("  dynamics forward   max|Δ| =", dd[0])
    print("  dynamics grad_in   max|Δ| =", dd[1])
    print("  dynamics action_base.grd max|Δ| =", dd[2])

    var tol = Scalar[DT](2e-3)
    var ok = (
        dt < Float64(tol)
        and dd[0] < Float64(tol)
        and dd[1] < Float64(tol)
        and dd[2] < Float64(tol)
    )
    assert_true(ok, "Dreamer4 storage CPU/GPU parity")
    print("DREAMER4 STORAGE GPU PARITY OK")
