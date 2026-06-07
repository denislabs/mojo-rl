"""Dreamer4Dynamics agent tokens — CPU↔GPU parity (Phase 3.3).

    pixi run -e apple  mojo run -I . tests/nn2/test_dreamer4_dynamics_agent_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn2/test_dreamer4_dynamics_agent_gpu.mojo

With NAGENT>0 the GPU forward writes the agent input into the grid agent
columns (`_dyn_set_agent_kernel`), runs the transformer, extracts h_t
(`_dyn_extract_agent_fwd_kernel`); the GPU vjp adds the h_t grad into the
transformer-out grad (`_dyn_add_agent_grad_kernel`) and extracts the agent
input grad (`_dyn_extract_agent_grad_kernel`). Reseed the global RNG before
each `make` for identical params, then check single-step parity of:
forward flow, h_t, grad_input, grad_agent_in, and a transformer param grad
(register). The act-MLP-style fp32 drift note applies — single backward only.
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
    print("Dreamer4Dynamics agent tokens — CPU↔GPU parity")
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
    comptime NAGENT = 1
    comptime B = 2
    comptime BATCH = B * T
    comptime IO = NSP * DSP
    comptime N = BATCH * IO
    comptime AGD = NAGENT * D
    comptime AGN = BATCH * AGD
    comptime RN = NREG * D                 # register param size

    var ctx = DeviceContext()

    seed(321)
    var dcpu = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX, True, 0, 0, NAGENT
    ].make[target="cpu", INIT=Xavier]()
    seed(321)
    var dgpu = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX, True, 0, 0, NAGENT
    ].make[target="gpu", INIT=Xavier](ctx)

    var z = _alloc(N)
    var sig = _alloc(BATCH)
    var stp = _alloc(BATCH)
    var go = _alloc(N)               # flow grad_output
    var agent_in = _alloc(AGN)
    var grad_h = _alloc(AGN)         # grad of h_t (set_grad_h)
    for i in range(N):
        z[i] = Scalar[DT](0.3 * sin(0.5 + 0.4 * Float64(i)))
        go[i] = Scalar[DT](0.1 * sin(1.7 + 0.3 * Float64(i)))
    for bt in range(BATCH):
        sig[bt] = Scalar[DT](Float64((bt + 1) % (KMAX + 1)))
        stp[bt] = Scalar[DT](Float64(bt % 2))
    for i in range(AGN):
        agent_in[i] = Scalar[DT](0.6 * sin(0.2 + 0.7 * Float64(i)))
        grad_h[i] = Scalar[DT](0.2 * sin(1.1 + 0.5 * Float64(i)))

    var zt = TileTensor(z, row_major[BATCH, IO]())

    # GPU device buffers for z / pred / grad_output / grad_input
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

    dcpu.set_indices(sig, stp, BATCH)
    dgpu.set_indices(sig, stp, BATCH)
    dcpu.set_agent_in(agent_in, BATCH)
    dgpu.set_agent_in(agent_in, BATCH)

    # ── forward (CPU + GPU) ─────────────────────────────────────────────
    var pred_c = _alloc(N)
    var gin_c = _alloc(N)
    var pc = TileTensor(pred_c, row_major[BATCH, IO]())
    var gc = TileTensor(gin_c, row_major[BATCH, IO]())
    var goc = TileTensor(go, row_major[BATCH, IO]())

    dcpu.zero_grad["cpu"]()
    dcpu.forward["cpu", BATCH](zt, output=pc)
    var h_c = _alloc(AGN)
    for i in range(AGN):
        h_c[i] = dcpu.agent_out_ptr_cpu()[i]
    dcpu.set_grad_h(grad_h, BATCH)
    dcpu.vjp["cpu", BATCH](goc, gc)
    var gain_c = _alloc(AGN)
    for i in range(AGN):
        gain_c[i] = dcpu.grad_agent_in_ptr_cpu()[i]

    dgpu.zero_grad["gpu"]()
    dgpu.forward["gpu", BATCH](zdt, output=pdt)
    dgpu.set_grad_h(grad_h, BATCH)
    dgpu.vjp["gpu", BATCH](godt, gindt)
    ctx.synchronize()

    # copy GPU outputs back
    var ph = ctx.enqueue_create_host_buffer[DT](N)
    var gih = ctx.enqueue_create_host_buffer[DT](N)
    var hh = ctx.enqueue_create_host_buffer[DT](AGN)
    var gainh = ctx.enqueue_create_host_buffer[DT](AGN)
    var regh = ctx.enqueue_create_host_buffer[DT](RN)
    ctx.enqueue_copy(ph, pd)
    ctx.enqueue_copy(gih, gind)
    ctx.enqueue_copy(hh, dgpu.agent_out_dev())
    ctx.enqueue_copy(gainh, dgpu.grad_agent_in_dev())
    ctx.enqueue_copy(regh, dgpu.register.grd.dev.value())
    ctx.synchronize()

    var pred_g = _alloc(N)
    var gin_g = _alloc(N)
    for i in range(N):
        pred_g[i] = ph.unsafe_ptr()[i]
        gin_g[i] = gih.unsafe_ptr()[i]

    var dpred = _maxdiff(pred_c, pred_g, N)
    var dgin = _maxdiff(gin_c, gin_g, N)
    var dh = _maxdiff(
        h_c, rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](hh.unsafe_ptr()), AGN)
    var dgain = _maxdiff(
        gain_c,
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gainh.unsafe_ptr()), AGN)
    var dreg = _maxdiff(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dcpu.register.grd.cpu.unsafe_ptr()),
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](regh.unsafe_ptr()), RN)

    print("  forward flow     max|Δ| =", dpred)
    print("  h_t              max|Δ| =", dh)
    print("  grad_input       max|Δ| =", dgin)
    print("  grad_agent_in    max|Δ| =", dgain)
    print("  grad register    max|Δ| =", dreg)

    assert_true(dpred < 2e-5, "forward flow parity")
    assert_true(dh < 2e-5, "h_t parity")
    assert_true(dgin < 2e-5, "grad_input parity")
    assert_true(dgain < 2e-5, "grad_agent_in parity")
    assert_true(dreg < 2e-5, "register grad parity")
    print("=" * 70)
    print("ALL PASSED — agent tokens CPU↔GPU parity")
    print("=" * 70)
