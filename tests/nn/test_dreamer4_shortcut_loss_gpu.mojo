"""Shortcut-forcing loss — CPU↔GPU parity (Phase 2.4).

    pixi run -e apple  mojo run -I . tests/nn/test_dreamer4_shortcut_loss_gpu.mojo

`dynamics_pretrain_loss[FWD="gpu"]` runs the three dynamics forwards on the
device (the heavy compute) while the element-wise flow/bootstrap arithmetic
stays on host. With identical params (reseeded RNG) and identical sampled
inputs, the GPU path must match the CPU path on the returned loss, grad_zhat,
and the main prediction zhat.
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
from mojo_rl.deep_agents.dreamer4.shortcut_loss import dynamics_pretrain_loss


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
    print("Dreamer4 shortcut-forcing loss — CPU↔GPU parity (Phase 2.4)")
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
    comptime B_SELF = 1
    comptime BF = B * T
    comptime ND = NSP * DSP
    comptime N = BF * ND

    var ctx = DeviceContext()
    seed(7)
    var dcpu = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX
    ].make[target="cpu", INIT=Xavier]()
    seed(7)
    var dgpu = Dreamer4Dynamics[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX
    ].make[target="gpu", INIT=Xavier](ctx)

    var z1 = _alloc(N)
    var z0 = _alloc(N)
    var sigma = _alloc(BF)
    var sidx = _alloc(BF)
    var pidx = _alloc(BF)
    for i in range(N):
        z1[i] = Scalar[DT](0.5 + 0.4 * sin(0.3 + 0.5 * Float64(i)))
        z0[i] = Scalar[DT](0.2 * sin(2.1 + 0.9 * Float64(i)))
    for t in range(T):
        sigma[0 * T + t] = 0.5
        sidx[0 * T + t] = 2.0
        pidx[0 * T + t] = 2.0
        sigma[1 * T + t] = 0.3
        sidx[1 * T + t] = 1.0
        pidx[1 * T + t] = 1.0

    # ── CPU loss ────────────────────────────────────────────────────────
    var gz_c = _alloc(N)
    var zhat_c = _alloc(N)
    var loss_c = dynamics_pretrain_loss[
        type_of(dcpu), B, T, B_SELF, NSP, DSP, KMAX
    ](dcpu, z1, z0, sigma, sidx, pidx, True, gz_c, zhat_c)

    # ── GPU loss (forwards on device, arithmetic on host) ───────────────
    var gz_g = _alloc(N)
    var zhat_g = _alloc(N)
    var dev_in = ctx.enqueue_create_buffer[DT](N)
    var dev_out = ctx.enqueue_create_buffer[DT](N)
    var h_in = ctx.enqueue_create_host_buffer[DT](N)
    var h_out = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    var loss_g = dynamics_pretrain_loss[
        type_of(dgpu), B, T, B_SELF, NSP, DSP, KMAX, FWD="gpu"
    ](
        dgpu, z1, z0, sigma, sidx, pidx, True, gz_g, zhat_g,
        ctx=ctx, dev_in=dev_in, dev_out=dev_out, h_in=h_in, h_out=h_out,
    )

    var dloss = abs(loss_c - loss_g)
    var dgz = _maxdiff(gz_c, gz_g, N)
    var dzh = _maxdiff(zhat_c, zhat_g, N)
    print("  loss    cpu =", loss_c, " gpu =", loss_g, " |Δ| =", dloss)
    print("  grad_zhat max|Δ| =", dgz)
    print("  zhat      max|Δ| =", dzh)

    assert_true(dloss < 1e-6, "loss parity")
    assert_true(dgz < 2e-5, "grad_zhat parity")
    assert_true(dzh < 2e-5, "zhat parity")
    print("=" * 70)
    print("ALL PASSED — shortcut-forcing loss CPU↔GPU parity")
    print("=" * 70)
