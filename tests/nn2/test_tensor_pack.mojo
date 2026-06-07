"""De-risk test for the S2′ TensorPack core (2026-06-07).

Proves the indexable origin-erased N-pack:
  1. constructs from a homogeneous variadic TileTensor pack (`of(*views)`),
  2. exposes each view's raw pointer (`ptr[i]()`) for CPU access,
  3. rebuilds a typed GPU `LayoutTensor` (`lt[i, layout]()`) that a kernel
     can read/write, round-tripping through device memory (Apple Metal).

This is the same mechanic the trait `forward`/`vjp` surface would use
once the full rollout swaps `*inputs` → `TensorPack`. Adoption inside a
real leaf (`binary_elementwise`) is validated separately by
`test_binary_elementwise_parity`.
"""

from std.memory import alloc
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.testing import assert_equal, assert_true
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import TensorPack


def _erased(
    p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int
) -> TileTensor[
    DT, type_of(row_major[1, 1]()), MutAnyOrigin
]:
    # Helper only used to feed the pack — the layout shape is irrelevant
    # since TensorPack keeps just the base pointer.
    return TileTensor(p, row_major[1, 1]())


def test_cpu_pack_ptr() raises:
    """Build: of() from a 3-way variadic, then ptr[i]() reads back each view."""
    comptime N = 4
    var a = alloc[Scalar[DT]](N)
    var b = alloc[Scalar[DT]](N)
    var c = alloc[Scalar[DT]](N)
    for k in range(N):
        a[k] = Scalar[DT](k)
        b[k] = Scalar[DT](10 + k)
        c[k] = Scalar[DT](100 + k)
    var pa = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](a)
    var pb = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b)
    var pc = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](c)

    var pack = TensorPack[3].of(
        _erased(pa, N), _erased(pb, N), _erased(pc, N)
    )

    # ptr[i]() must alias the same storage (writes are visible).
    assert_equal(pack.ptr[0]()[2], Scalar[DT](2))
    assert_equal(pack.ptr[1]()[2], Scalar[DT](12))
    assert_equal(pack.ptr[2]()[3], Scalar[DT](103))
    pack.ptr[1]()[0] = Scalar[DT](999)
    assert_equal(b[0], Scalar[DT](999))

    a.free(); b.free(); c.free()
    print("  test_cpu_pack_ptr ok")


def _add_via_pack_kernel[N: Int](
    i0: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    i1: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        o[idx] = rebind[Scalar[DT]](i0[idx]) + rebind[Scalar[DT]](i1[idx])


def test_gpu_pack_lt() raises:
    """Build: of() on device buffers, lt[i, layout]() as kernel args, D2H check."""
    comptime N = 8
    comptime layout = Layout.row_major(N)
    var ctx = DeviceContext()

    var d0 = ctx.enqueue_create_buffer[DT](N)
    var d1 = ctx.enqueue_create_buffer[DT](N)
    var do = ctx.enqueue_create_buffer[DT](N)
    var h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for k in range(N):
        h[k] = Scalar[DT](k)
    ctx.enqueue_copy(d0, h)
    ctx.synchronize()            # let d0 copy land before reusing h
    for k in range(N):
        h[k] = Scalar[DT](2 * k)
    ctx.enqueue_copy(d1, h)
    ctx.synchronize()

    var p0 = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](d0.unsafe_ptr())
    var p1 = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](d1.unsafe_ptr())
    var po = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](do.unsafe_ptr())

    var pack = TensorPack[3].of(
        _erased(p0, N), _erased(p1, N), _erased(po, N)
    )

    comptime n_blocks = (N + TPB - 1) // TPB
    comptime kernel = _add_via_pack_kernel[N]
    ctx.enqueue_function[kernel](
        pack.lt[0, layout](), pack.lt[1, layout](), pack.lt[2, layout](),
        grid_dim=n_blocks, block_dim=TPB,
    )
    ctx.enqueue_copy(h, do)
    ctx.synchronize()
    for k in range(N):
        assert_equal(h[k], Scalar[DT](3 * k))

    print("  test_gpu_pack_lt ok")


def main() raises:
    print("=" * 60)
    print("S2′ TensorPack de-risk")
    print("=" * 60)
    test_cpu_pack_ptr()
    test_gpu_pack_lt()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
