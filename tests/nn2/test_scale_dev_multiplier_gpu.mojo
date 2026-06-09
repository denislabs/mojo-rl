"""Scale[DIM] device-pointer multiplier (Slice 4a) — GPU.

Verifies the `multiplier_ptr` path (scale factor read from a device buffer,
CUDA-graph capturable) produces the same result as the baked-scalar
`multiplier` path, for both forward and vjp. This is the foundation for
SAC's on-device α: another kernel can update `mptr[0]` between captured
launches without re-baking a kernel arg.
"""

from std.math import abs as fabs
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.scale import Scale


def test_scale_dev_multiplier_gpu() raises:
    comptime DIM = 1
    comptime BATCH = 4
    comptime N = BATCH * DIM
    comptime M: Scalar[DT] = 3.0
    comptime TOL: Scalar[DT] = 1e-6

    var ctx = DeviceContext()
    var scale = Scale[DIM].make["gpu", INIT=Zero](ctx)

    # Input [BATCH, DIM] on device.
    var in_host = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for k in range(N):
        in_host.unsafe_ptr()[k] = Scalar[DT](0.5 + 0.25 * Float64(k))
    var in_dev = ctx.enqueue_create_buffer[DT](N)
    ctx.enqueue_copy(in_dev, in_host)
    var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        in_dev.unsafe_ptr()
    )
    var in_t = TileTensor(in_p, row_major[BATCH, DIM]())

    var out_dev = ctx.enqueue_create_buffer[DT](N)
    var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        out_dev.unsafe_ptr()
    )
    var out_t = TileTensor(out_p, row_major[BATCH, DIM]())
    var out_host = ctx.enqueue_create_host_buffer[DT](N)

    # ── Path A: baked scalar multiplier.
    scale.multiplier = M
    scale.forward["gpu", BATCH](in_t, output=out_t)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var a0 = out_host.unsafe_ptr()[0]
    var a1 = out_host.unsafe_ptr()[1]

    # ── Path B: device-pointer multiplier (same value, different source).
    var m_dev = ctx.enqueue_create_buffer[DT](1)
    var m_host = ctx.enqueue_create_host_buffer[DT](1)
    ctx.synchronize()
    m_host.unsafe_ptr()[0] = M
    ctx.enqueue_copy(m_dev, m_host)
    scale.multiplier = Scalar[DT](999.0)  # poison the baked path
    scale.set_multiplier_ptr(m_dev.unsafe_ptr())
    scale.forward["gpu", BATCH](in_t, output=out_t)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var max_diff: Scalar[DT] = 0.0
    for k in range(N):
        var expect = in_host.unsafe_ptr()[k] * M
        var got = out_host.unsafe_ptr()[k]
        var d = fabs(got - expect)
        if d > max_diff:
            max_diff = d
    print("  forward: dev-ptr vs expected max_diff =", max_diff)
    print("  (baked-path sample:", a0, a1, ")")
    assert_true(max_diff < TOL, "dev-ptr forward != input*M")

    # ── vjp: grad_in = M * grad_out, via the device-pointer path.
    var go_dev = ctx.enqueue_create_buffer[DT](N)
    var go_host = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for k in range(N):
        go_host.unsafe_ptr()[k] = Scalar[DT](1.0 + 0.1 * Float64(k))
    ctx.enqueue_copy(go_dev, go_host)
    var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        go_dev.unsafe_ptr()
    )
    var go_t = TileTensor(go_p, row_major[BATCH, DIM]())
    var gi_dev = ctx.enqueue_create_buffer[DT](N)
    var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        gi_dev.unsafe_ptr()
    )
    var gi_t = TileTensor(gi_p, row_major[BATCH, DIM]())
    scale.vjp["gpu", BATCH](go_t, gi_t)
    var gi_host = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.synchronize()
    var gmax: Scalar[DT] = 0.0
    for k in range(N):
        var expect = go_host.unsafe_ptr()[k] * M
        var d = fabs(gi_host.unsafe_ptr()[k] - expect)
        if d > gmax:
            gmax = d
    print("  vjp: dev-ptr vs expected max_diff =", gmax)
    assert_true(gmax < TOL, "dev-ptr vjp != grad_out*M")

    print("  test_scale_dev_multiplier_gpu PASSED")


def main() raises:
    print("=" * 60)
    print("Scale device-pointer multiplier (Slice 4a) — GPU")
    print("=" * 60)
    test_scale_dev_multiplier_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
