"""Dreamer4Dynamics — CPU↔GPU parity (Phase 2.4).

    pixi run -e apple  mojo run -I . tests/nn2/test_dreamer4_dynamics_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn2/test_dreamer4_dynamics_gpu.mojo

Forces identical params on both targets (reseed the global RNG before each
`make`, since Xavier draws from it), runs the SAME packed input + signal/step
indices through forward + vjp on CPU and GPU, and compares the prediction,
grad_input, and the bespoke conditioning-param grads (action_base /
signal_table / step_table / register). Validates the device token-assembly
kernel + its four param-grad kernels match the CPU front-end.
"""

from std.memory import alloc
from std.math import sin
from std.random import seed
from std.testing import assert_true
from layout import TileTensor, row_major
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.deep_agents2.dreamer4.dynamics import Dreamer4Dynamics


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _maxdiff(a: UnsafePointer[Scalar[DT], MutAnyOrigin],
             b: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Float64:
    var m: Float64 = 0.0
    for i in range(n):
        var d = abs(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


def main() raises:
    print("=" * 70)
    print("Dreamer4Dynamics — CPU↔GPU parity (Phase 2.4)")
    print("=" * 70)

    comptime DSP = 4
    comptime NSP = 4
    comptime D = 8
    comptime NH = 2
    comptime T = 2
    comptime NREG = 2
    comptime HID = 16
    comptime DEPTH = 2
    comptime KMAX = 4
    comptime B = 2
    comptime BATCH = B * T
    comptime IO = NSP * DSP
    comptime N = BATCH * IO
    comptime DD = type_of(
        Dreamer4Dynamics[DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX]()
    )
    comptime NSIG = KMAX + 1
    comptime NSTEP = DD.NSTEP

    var ctx = DeviceContext()

    seed(123)
    var dcpu = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX
    ].make[target="cpu", INIT=Xavier]()
    seed(123)
    var dgpu = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX
    ].make[target="gpu", INIT=Xavier](ctx)

    # shared inputs
    var z = _alloc(N)
    var sig = _alloc(BATCH)
    var stp = _alloc(BATCH)
    var go = _alloc(N)
    for i in range(N):
        z[i] = Scalar[DT](0.3 * sin(0.5 + 0.4 * Float64(i)))
        go[i] = Scalar[DT](0.1 * sin(1.7 + 0.3 * Float64(i)))
    for bt in range(BATCH):
        sig[bt] = Scalar[DT](Float64((bt + 1) % (KMAX + 1)))
        stp[bt] = Scalar[DT](Float64(bt % 2))

    # ── CPU forward + vjp ───────────────────────────────────────────────
    var pred_c = _alloc(N)
    var gin_c = _alloc(N)
    var zt = TileTensor(z, row_major[BATCH, IO]())
    var pc = TileTensor(pred_c, row_major[BATCH, IO]())
    var gc = TileTensor(gin_c, row_major[BATCH, IO]())
    var goc = TileTensor(go, row_major[BATCH, IO]())
    dcpu.set_indices(sig, stp, BATCH)
    dcpu.zero_grad["cpu"]()
    dcpu.forward["cpu", BATCH](zt, output=pc)
    dcpu.vjp["cpu", BATCH](goc, gc)

    # ── GPU forward + vjp ───────────────────────────────────────────────
    var zd = ctx.enqueue_create_buffer[DT](N)
    var pd = ctx.enqueue_create_buffer[DT](N)
    var god = ctx.enqueue_create_buffer[DT](N)
    var gind = ctx.enqueue_create_buffer[DT](N)
    var zh = ctx.enqueue_create_host_buffer[DT](N)
    var goh = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        zh.unsafe_ptr()[i] = z[i]
        goh.unsafe_ptr()[i] = go[i]
    ctx.enqueue_copy(zd, zh)
    ctx.enqueue_copy(god, goh)
    var zdt = TileTensor(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](zd.unsafe_ptr()),
        row_major[BATCH, IO]())
    var pdt = TileTensor(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](pd.unsafe_ptr()),
        row_major[BATCH, IO]())
    var godt = TileTensor(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](god.unsafe_ptr()),
        row_major[BATCH, IO]())
    var gindt = TileTensor(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gind.unsafe_ptr()),
        row_major[BATCH, IO]())
    dgpu.set_indices(sig, stp, BATCH)
    dgpu.zero_grad["gpu"]()
    dgpu.forward["gpu", BATCH](zdt, output=pdt)
    dgpu.vjp["gpu", BATCH](godt, gindt)
    ctx.synchronize()

    # download GPU results
    var pred_g = _alloc(N)
    var gin_g = _alloc(N)
    var ph = ctx.enqueue_create_host_buffer[DT](N)
    var gih = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(ph, pd)
    ctx.enqueue_copy(gih, gind)
    ctx.synchronize()
    for i in range(N):
        pred_g[i] = ph.unsafe_ptr()[i]
        gin_g[i] = gih.unsafe_ptr()[i]

    var dpred = _maxdiff(pred_c, pred_g, N)
    var dgin = _maxdiff(gin_c, gin_g, N)
    print("  forward  max|Δ| =", dpred)
    print("  grad_in  max|Δ| =", dgin)

    # param-grad parity (download gpu grads into host buffers, compare)
    var hab = ctx.enqueue_create_host_buffer[DT](D)
    var hsg = ctx.enqueue_create_host_buffer[DT](NSIG * D)
    var hst = ctx.enqueue_create_host_buffer[DT](NSTEP * D)
    var hrg = ctx.enqueue_create_host_buffer[DT](NREG * D)
    ctx.enqueue_copy(hab, dgpu.action_base.grad_dev.value())
    ctx.enqueue_copy(hsg, dgpu.signal_table.grad_dev.value())
    ctx.enqueue_copy(hst, dgpu.step_table.grad_dev.value())
    ctx.enqueue_copy(hrg, dgpu.register.grad_dev.value())
    ctx.synchronize()
    var cab = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dcpu.action_base.grad.unsafe_ptr())
    var csg = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dcpu.signal_table.grad.unsafe_ptr())
    var cst = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dcpu.step_table.grad.unsafe_ptr())
    var crg = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dcpu.register.grad.unsafe_ptr())
    var dab = _maxdiff(cab, rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](hab.unsafe_ptr()), D)
    var dsg = _maxdiff(csg, rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](hsg.unsafe_ptr()), NSIG * D)
    var dst = _maxdiff(cst, rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](hst.unsafe_ptr()), NSTEP * D)
    var drg = _maxdiff(crg, rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](hrg.unsafe_ptr()), NREG * D)
    print("  grad action_base  max|Δ| =", dab)
    print("  grad signal_table max|Δ| =", dsg)
    print("  grad step_table   max|Δ| =", dst)
    print("  grad register     max|Δ| =", drg)

    assert_true(dpred < 2e-5, "forward parity")
    assert_true(dgin < 2e-5, "grad_input parity")
    assert_true(dab < 2e-5, "action_base grad parity")
    assert_true(dsg < 2e-5, "signal_table grad parity")
    assert_true(dst < 2e-5, "step_table grad parity")
    assert_true(drg < 2e-5, "register grad parity")
    print("=" * 70)
    print("ALL PASSED — Dreamer4Dynamics CPU↔GPU parity")
    print("=" * 70)
