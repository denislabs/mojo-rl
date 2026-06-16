"""MSEPerSample GPU forward+backward parity vs CPU — validates the GPU
cache_diff allocation in `_ensure_gpu` (was stripped by the S5 migration)."""
from std.memory import alloc
from std.gpu.host import DeviceContext
from std.testing import assert_true
from layout import TileTensor, row_major
from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.mse_per_sample import MSEPerSample
from mojo_rl.nn.core import TensorPack
from mojo_rl.nn.initializer import Zero


def test_mse_per_sample_gpu_parity() raises:
    comptime BATCH = 6
    comptime DIM = 5
    comptime N = BATCH * DIM
    var ctx = DeviceContext()

    var ha = ctx.enqueue_create_host_buffer[DT](N)
    var hb = ctx.enqueue_create_host_buffer[DT](N)
    var hgo = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()
    for i in range(N):
        ha[i] = Scalar[DT](-1.0 + 0.13 * Float64(i))
        hb[i] = Scalar[DT](0.4 - 0.07 * Float64(i))
    for r in range(BATCH):
        hgo[r] = Scalar[DT](0.5 + 0.1 * Float64(r))

    # ---- CPU reference ----
    var cpu = MSEPerSample[DIM].make[target="cpu", INIT=Zero]()
    var a_c = alloc[Scalar[DT]](N); var b_c = alloc[Scalar[DT]](N)
    var o_c = alloc[Scalar[DT]](BATCH)
    var ga_c = alloc[Scalar[DT]](N); var gb_c = alloc[Scalar[DT]](N)
    var go_c = alloc[Scalar[DT]](BATCH)
    for i in range(N): a_c[i] = ha[i]; b_c[i] = hb[i]
    for r in range(BATCH): go_c[r] = hgo[r]
    var ap = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](a_c)
    var bp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b_c)
    var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o_c)
    var gap = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](ga_c)
    var gbp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gb_c)
    var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go_c)
    var o_cpu_t = TileTensor(op, row_major[BATCH, 1]())
    cpu.forward["cpu", BATCH](
        TensorPack[2].of(TileTensor(ap, row_major[BATCH, DIM]()),
                         TileTensor(bp, row_major[BATCH, DIM]())),
        output=o_cpu_t)
    cpu.vjp["cpu", BATCH](
        TileTensor(gop, row_major[BATCH, 1]()),
        TensorPack[2].of(TileTensor(gap, row_major[BATCH, DIM]()),
                         TileTensor(gbp, row_major[BATCH, DIM]())))

    # ---- GPU ----
    var gpu = MSEPerSample[DIM].make[target="gpu", INIT=Zero](ctx=ctx)
    var da = ctx.enqueue_create_buffer[DT](N); var db = ctx.enqueue_create_buffer[DT](N)
    var do_ = ctx.enqueue_create_buffer[DT](BATCH)
    var dga = ctx.enqueue_create_buffer[DT](N); var dgb = ctx.enqueue_create_buffer[DT](N)
    var dgo = ctx.enqueue_create_buffer[DT](BATCH)
    ctx.enqueue_copy(da, ha); ctx.enqueue_copy(db, hb); ctx.enqueue_copy(dgo, hgo)
    ctx.synchronize()
    var gap2 = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](da.unsafe_ptr())
    var gbp2 = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](db.unsafe_ptr())
    var gop2 = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](do_.unsafe_ptr())
    var o_gpu_t = TileTensor(gop2, row_major[BATCH, 1]())
    gpu.forward["gpu", BATCH](
        TensorPack[2].of(TileTensor(gap2, row_major[BATCH, DIM]()),
                         TileTensor(gbp2, row_major[BATCH, DIM]())),
        output=o_gpu_t)
    var ggo = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dgo.unsafe_ptr())
    var gga = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dga.unsafe_ptr())
    var ggb = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dgb.unsafe_ptr())
    gpu.vjp["gpu", BATCH](
        TileTensor(ggo, row_major[BATCH, 1]()),
        TensorPack[2].of(TileTensor(gga, row_major[BATCH, DIM]()),
                         TileTensor(ggb, row_major[BATCH, DIM]())))
    var ho = ctx.enqueue_create_host_buffer[DT](BATCH)
    var hga = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(ho, do_); ctx.enqueue_copy(hga, dga)
    ctx.synchronize()

    var maxf: Scalar[DT] = 0.0
    for r in range(BATCH):
        var d = ho[r] - o_c[r]; maxf = max(maxf, d if d >= 0 else -d)
    var maxg: Scalar[DT] = 0.0
    for i in range(N):
        var d = hga[i] - ga_c[i]; maxg = max(maxg, d if d >= 0 else -d)
    print("  forward max|gpu-cpu| =", maxf, "  grad_a max|gpu-cpu| =", maxg)
    assert_true(maxf < 1e-5, "MSEPerSample forward GPU/CPU mismatch")
    assert_true(maxg < 1e-5, "MSEPerSample grad_a GPU/CPU mismatch")
    a_c.free(); b_c.free(); o_c.free(); ga_c.free(); gb_c.free(); go_c.free()
    print("  ok")


def main() raises:
    print("MSEPerSample GPU/CPU parity ...")
    test_mse_per_sample_gpu_parity()
    print("ALL PASSED")
