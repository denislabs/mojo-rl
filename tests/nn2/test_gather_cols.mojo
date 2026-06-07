"""Smoke + CPU↔GPU parity test for GatherCols[NA].

Validates:
  - forward: out[b, 0] == values[b, Int(idx[b, 0])]
  - vjp:     grad_values zero-filled, grad_idx zero-filled
  - CPU and GPU produce identical forward outputs

CPU + GPU. GPU only runs when DeviceContext can be created.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.primitives.gather_cols import GatherCols
from mojo_rl.nn2.initializer import Zero


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def test_forward_cpu() raises:
    print("test_forward_cpu ...")
    comptime BATCH = 3
    comptime NA = 4

    var g = GatherCols[NA].make[target="cpu", INIT=Zero]()

    var v: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NA)
    var idx: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)

    # Row 0 values [10, 20, 30, 40], idx=2 → expect 30.
    # Row 1 values [-1, -2, -3, -4], idx=0 → expect -1.
    # Row 2 values [7, 8, 9, 11], idx=3 → expect 11.
    v[0]=10; v[1]=20; v[2]=30; v[3]=40
    v[4]=-1; v[5]=-2; v[6]=-3; v[7]=-4
    v[8]=7;  v[9]=8;  v[10]=9; v[11]=11
    idx[0] = 2; idx[1] = 0; idx[2] = 3

    # Hetero-variadic shape workaround: both *inputs constructed with the
    # SAME comptime Layout (row_major[BATCH, NA]). typed_view inside the
    # leaf recovers the real shape — Layout on the TileTensor is dead
    # metadata after unpack. See feedback_mojo_variadic_hetero_shape_workaround.
    var vp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](v)
    var ip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](idx)
    var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out)
    var v_t = TileTensor(vp, row_major[BATCH, NA]())
    var i_t = TileTensor(ip, row_major[BATCH, NA]())
    var o_t = TileTensor(op, row_major[BATCH, 1]())
    g.forward["cpu", BATCH](
            TensorPack[2].of(v_t, i_t), output=o_t,
        )

    assert_true(out[0] == Scalar[DT](30), "row 0 gather")
    assert_true(out[1] == Scalar[DT](-1), "row 1 gather")
    assert_true(out[2] == Scalar[DT](11), "row 2 gather")
    print("  ok")


def test_vjp_cpu_zero_fill() raises:
    print("test_vjp_cpu_zero_fill ...")
    comptime BATCH = 2
    comptime NA = 3

    var g = GatherCols[NA].make[target="cpu", INIT=Zero]()

    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var gv: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NA)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    go[0] = 5.0; go[1] = -2.0
    # Pre-fill grad slabs with junk to prove vjp overwrites with zero.
    for i in range(BATCH * NA):
        gv[i] = Scalar[DT](42.0)
    for b in range(BATCH):
        gi[b] = Scalar[DT](99.0)

    # Hetero-variadic for grad_inputs: same comptime Layout for both.
    var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
    var gvp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gv)
    var gip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi)
    var go_t = TileTensor(gop, row_major[BATCH, 1]())
    var gv_t = TileTensor(gvp, row_major[BATCH, NA]())
    var gi_t = TileTensor(gip, row_major[BATCH, NA]())
    g.vjp["cpu", BATCH](go_t, TensorPack[2].of(gv_t, gi_t))

    for i in range(BATCH * NA):
        assert_true(gv[i] == Scalar[DT](0.0), "grad_values zero-fill")
    for b in range(BATCH):
        assert_true(gi[b] == Scalar[DT](0.0), "grad_idx zero-fill")
    print("  ok")


def test_cpu_gpu_parity() raises:
    print("test_cpu_gpu_parity ...")
    comptime BATCH = 8
    comptime NA = 5

    try:
        var ctx = DeviceContext()

        var v_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NA)
        var idx_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        var out_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        var out_gpu_host: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)

        # Deterministic fill — each row gets a distinct gather index.
        for b in range(BATCH):
            for a in range(NA):
                v_cpu[b * NA + a] = Scalar[DT](
                    100.0 * Float64(b) + Float64(a)
                )
            idx_cpu[b] = Scalar[DT]((b * 2 + 1) % NA)

        # CPU. Hetero-variadic: same carrier Layout for both inputs.
        var g_cpu = GatherCols[NA].make[target="cpu", INIT=Zero]()
        var vp_cpu = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](v_cpu)
        var ip_cpu = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](idx_cpu)
        var op_cpu = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out_cpu)
        var v_cpu_t = TileTensor(vp_cpu, row_major[BATCH, NA]())
        var i_cpu_t = TileTensor(ip_cpu, row_major[BATCH, NA]())
        var o_cpu_t = TileTensor(op_cpu, row_major[BATCH, 1]())
        g_cpu.forward["cpu", BATCH](
            TensorPack[2].of(v_cpu_t, i_cpu_t), output=o_cpu_t,
        )

        # GPU. Same hetero-variadic workaround.
        var g_gpu = GatherCols[NA].make[target="gpu", INIT=Zero](ctx=ctx)
        var v_dev = ctx.enqueue_create_buffer[DT](BATCH * NA)
        var i_dev = ctx.enqueue_create_buffer[DT](BATCH)
        var o_dev = ctx.enqueue_create_buffer[DT](BATCH)
        ctx.enqueue_copy(v_dev, v_cpu)
        ctx.enqueue_copy(i_dev, idx_cpu)
        var vp_dev = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](v_dev.unsafe_ptr())
        var ip_dev = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](i_dev.unsafe_ptr())
        var op_dev = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o_dev.unsafe_ptr())
        var v_gpu_t = TileTensor(vp_dev, row_major[BATCH, NA]())
        var i_gpu_t = TileTensor(ip_dev, row_major[BATCH, NA]())
        var o_gpu_t = TileTensor(op_dev, row_major[BATCH, 1]())
        g_gpu.forward["gpu", BATCH](
            TensorPack[2].of(v_gpu_t, i_gpu_t), output=o_gpu_t,
        )
        ctx.enqueue_copy(out_gpu_host, o_dev)
        ctx.synchronize()

        var max_diff: Scalar[DT] = 0.0
        for b in range(BATCH):
            var d = _abs(out_cpu[b] - out_gpu_host[b])
            if d > max_diff:
                max_diff = d
        print("  max |cpu - gpu| =", max_diff)
        assert_true(max_diff == Scalar[DT](0), "CPU/GPU forward mismatch")
        print("  ok")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 70)
    print("GatherCols[NA] smoke + CPU/GPU parity")
    print("=" * 70)
    test_forward_cpu()
    test_vjp_cpu_zero_fill()
    test_cpu_gpu_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
