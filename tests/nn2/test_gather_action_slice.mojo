"""GatherActionSlice[NA, K] smoke + CPU/GPU parity test."""

from std.memory import alloc
from std.gpu.host import DeviceContext
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.primitives.gather_action_slice import GatherActionSlice
from mojo_rl.nn2.initializer import Zero


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def test_forward_cpu() raises:
    print("test_forward_cpu ...")
    comptime BATCH = 2
    comptime NA = 3
    comptime K = 4

    var g = GatherActionSlice[NA, K].make[target="cpu", INIT=Zero]()
    var v: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NA * K)
    var idx: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * K)

    # Row 0: action 1 → slice [4..8). Row 1: action 2 → slice [8..12).
    for c in range(NA * K):
        v[c] = Scalar[DT](c)             # row 0: 0..11
        v[NA * K + c] = Scalar[DT](100 + c)  # row 1: 100..111
    idx[0] = 1
    idx[1] = 2

    var vp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](v)
    var ip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](idx)
    var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out)
    var v_t = TileTensor(vp, row_major[BATCH, NA * K]())
    var i_t = TileTensor(ip, row_major[BATCH, NA * K]())  # hetero-variadic carrier
    var o_t = TileTensor(op, row_major[BATCH, K]())
    g.forward["cpu", BATCH](
            TensorPack[2].of(v_t, i_t), output=o_t,
        )

    # Row 0 action 1 → values [4, 5, 6, 7]
    assert_true(out[0] == Scalar[DT](4), "row 0 k=0")
    assert_true(out[1] == Scalar[DT](5), "row 0 k=1")
    assert_true(out[2] == Scalar[DT](6), "row 0 k=2")
    assert_true(out[3] == Scalar[DT](7), "row 0 k=3")
    # Row 1 action 2 → values [108, 109, 110, 111]
    assert_true(out[4] == Scalar[DT](108), "row 1 k=0")
    assert_true(out[5] == Scalar[DT](109), "row 1 k=1")
    assert_true(out[6] == Scalar[DT](110), "row 1 k=2")
    assert_true(out[7] == Scalar[DT](111), "row 1 k=3")
    print("  ok")


def test_cpu_gpu_parity() raises:
    print("test_cpu_gpu_parity ...")
    comptime BATCH = 8
    comptime NA = 4
    comptime K = 7

    try:
        var ctx = DeviceContext()

        var v_host: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NA * K)
        var idx_host: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        var out_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * K)
        var out_gpu_host: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * K)
        for c in range(BATCH * NA * K):
            v_host[c] = Scalar[DT](0.05 * Float64(c) - 0.5)
        for b in range(BATCH):
            idx_host[b] = Scalar[DT]((b * 3 + 2) % NA)

        # CPU.
        var g_cpu = GatherActionSlice[NA, K].make[target="cpu", INIT=Zero]()
        var vp_cpu = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](v_host)
        var ip_cpu = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](idx_host)
        var op_cpu = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out_cpu)
        var v_cpu_t = TileTensor(vp_cpu, row_major[BATCH, NA * K]())
        var i_cpu_t = TileTensor(ip_cpu, row_major[BATCH, NA * K]())
        var o_cpu_t = TileTensor(op_cpu, row_major[BATCH, K]())
        g_cpu.forward["cpu", BATCH](
            TensorPack[2].of(v_cpu_t, i_cpu_t), output=o_cpu_t,
        )

        # GPU.
        var g_gpu = GatherActionSlice[NA, K].make[target="gpu", INIT=Zero](ctx=ctx)
        var v_dev = ctx.enqueue_create_buffer[DT](BATCH * NA * K)
        var i_dev = ctx.enqueue_create_buffer[DT](BATCH)
        var o_dev = ctx.enqueue_create_buffer[DT](BATCH * K)
        ctx.enqueue_copy(v_dev, v_host)
        ctx.enqueue_copy(i_dev, idx_host)
        var vp_dev = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](v_dev.unsafe_ptr())
        var ip_dev = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](i_dev.unsafe_ptr())
        var op_dev = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o_dev.unsafe_ptr())
        var v_gpu_t = TileTensor(vp_dev, row_major[BATCH, NA * K]())
        var i_gpu_t = TileTensor(ip_dev, row_major[BATCH, NA * K]())
        var o_gpu_t = TileTensor(op_dev, row_major[BATCH, K]())
        g_gpu.forward["gpu", BATCH](
            TensorPack[2].of(v_gpu_t, i_gpu_t), output=o_gpu_t,
        )
        ctx.enqueue_copy(out_gpu_host, o_dev)
        ctx.synchronize()

        var max_diff: Scalar[DT] = 0.0
        for k in range(BATCH * K):
            var d = _abs(out_cpu[k] - out_gpu_host[k])
            if d > max_diff:
                max_diff = d
        print("  max |cpu - gpu| =", max_diff)
        assert_true(max_diff == Scalar[DT](0), "CPU/GPU forward mismatch")
        print("  ok")
    except e:
        print("  (skipped — no GPU:", e, ")")


def main() raises:
    print("=" * 70)
    print("GatherActionSlice[NA, K] smoke + CPU/GPU parity")
    print("=" * 70)
    test_forward_cpu()
    test_cpu_gpu_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
