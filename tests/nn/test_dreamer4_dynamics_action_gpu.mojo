"""Dreamer4Dynamics action conditioning — CPU↔GPU parity.

    pixi run -e apple  mojo run -I . tests/nn/test_dreamer4_dynamics_action_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_dreamer4_dynamics_action_gpu.mojo

With ADIM>0 the GPU forward uploads actions, runs act_mlp on device, and adds
its output into the action token (`_dyn_add_act_kernel`); the GPU vjp extracts
the action-token grad (`_dyn_extract_token0_kernel`) and runs act_mlp.vjp on
device. Reseed the global RNG before each `make` for identical params, then:
  1. forward / grad_input / action_base-grad parity (single step);
  2. DIRECT act-MLP param-grad parity: the act-MLP's first Linear weight grad
     (act_mlp.children[0]) must match CPU↔GPU at a single backward — this
     isolates the device act-MLP backward (the multi-step Adam trajectory is
     NOT used as a gate: the transformer tail's matmul param grads drift at the
     fp32 level CPU↔GPU and compound over steps, independent of the act-MLP).
"""

from std.memory import alloc
from std.math import sin
from std.random import seed
from std.testing import assert_true
from layout import TileTensor, row_major
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.deep_agents.dreamer4.dynamics import Dreamer4Dynamics


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
    print("Dreamer4Dynamics action conditioning — CPU↔GPU parity")
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
    comptime ADIM = 3
    comptime AHID = 2 * D                 # act-MLP hidden (matches AHID_EFF)
    comptime AW = ADIM * AHID             # first Linear weight size
    comptime B = 2
    comptime BATCH = B * T
    comptime IO = NSP * DSP
    comptime N = BATCH * IO

    var ctx = DeviceContext()

    seed(123)
    var dcpu = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX, True, ADIM
    ].make[target="cpu", INIT=Xavier]()
    seed(123)
    var dgpu = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX, True, ADIM
    ].make[target="gpu", INIT=Xavier](ctx)

    # shared inputs (fixed grad_output drives the parity train loop)
    var z = _alloc(N)
    var sig = _alloc(BATCH)
    var stp = _alloc(BATCH)
    var go = _alloc(N)
    var actions = _alloc(BATCH * ADIM)
    var act_mask = _alloc(ADIM)
    for i in range(N):
        z[i] = Scalar[DT](0.3 * sin(0.5 + 0.4 * Float64(i)))
        go[i] = Scalar[DT](0.1 * sin(1.7 + 0.3 * Float64(i)))
    for bt in range(BATCH):
        sig[bt] = Scalar[DT](Float64((bt + 1) % (KMAX + 1)))
        stp[bt] = Scalar[DT](Float64(bt % 2))
    for a in range(ADIM):
        act_mask[a] = 1.0
    for bt in range(BATCH):
        for a in range(ADIM):
            actions[bt * ADIM + a] = Scalar[DT](
                0.7 * sin(0.2 + 0.9 * Float64(bt * ADIM + a))
            )

    var zt = TileTensor(z, row_major[BATCH, IO]())

    # ── GPU device buffers ──────────────────────────────────────────────
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
    dcpu.set_actions(actions, act_mask, BATCH)
    dgpu.set_actions(actions, act_mask, BATCH)

    # ── 1. single-step forward + grad parity ────────────────────────────
    var pred_c = _alloc(N)
    var gin_c = _alloc(N)
    var pc = TileTensor(pred_c, row_major[BATCH, IO]())
    var gc = TileTensor(gin_c, row_major[BATCH, IO]())
    var goc = TileTensor(go, row_major[BATCH, IO]())
    dcpu.zero_grad["cpu"]()
    dcpu.forward["cpu", BATCH](zt, output=pc)
    dcpu.vjp["cpu", BATCH](goc, gc)

    dgpu.zero_grad["gpu"]()
    dgpu.forward["gpu", BATCH](zdt, output=pdt)
    dgpu.vjp["gpu", BATCH](godt, gindt)
    ctx.synchronize()

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
    var hab = ctx.enqueue_create_host_buffer[DT](D)
    ctx.enqueue_copy(hab, dgpu.action_base.grd.dev.value())
    ctx.synchronize()
    var dab = _maxdiff(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dcpu.action_base.grd.cpu.unsafe_ptr()),
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](hab.unsafe_ptr()), D)
    print("  forward          max|Δ| =", dpred)
    print("  grad_input       max|Δ| =", dgin)
    print("  grad action_base max|Δ| =", dab)

    # ── 2. DIRECT act-MLP param-grad parity (first Linear weight) ────────
    var haw = ctx.enqueue_create_host_buffer[DT](AW)
    ctx.enqueue_copy(haw, dgpu.act_mlp.children[0].weight.grd.dev.value())
    ctx.synchronize()
    var daw = _maxdiff(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            dcpu.act_mlp.children[0].weight.grd.cpu.unsafe_ptr()
        ),
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](haw.unsafe_ptr()),
        AW,
    )
    print("  grad act_mlp.fc1 max|Δ| =", daw)

    assert_true(dpred < 2e-5, "forward parity")
    assert_true(dgin < 2e-5, "grad_input parity")
    assert_true(dab < 2e-5, "action_base grad parity")
    assert_true(daw < 2e-5, "act-MLP fc1 weight-grad parity")
    print("=" * 70)
    print("ALL PASSED — action conditioning CPU↔GPU parity")
    print("=" * 70)
