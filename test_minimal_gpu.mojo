"""Minimal GPU test to check compilation time."""

from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

comptime dtype = DType.float32


fn simple_add_kernel(
    output: LayoutTensor[dtype, Layout.row_major(4), MutAnyOrigin],
    input_a: LayoutTensor[dtype, Layout.row_major(4), ImmutAnyOrigin],
    input_b: LayoutTensor[dtype, Layout.row_major(4), ImmutAnyOrigin],
):
    from gpu import thread_idx

    var idx = Int(thread_idx.x)
    if idx < 4:
        output[idx] = rebind[Scalar[dtype]](input_a[idx]) + rebind[Scalar[dtype]](
            input_b[idx]
        )


fn main() raises:
    print("Minimal GPU test")

    with DeviceContext() as ctx:
        var a_buf = ctx.enqueue_create_buffer[dtype](4)
        var b_buf = ctx.enqueue_create_buffer[dtype](4)
        var out_buf = ctx.enqueue_create_buffer[dtype](4)

        with a_buf.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(4):
                a_host[i] = Scalar[dtype](Float32(i))
                b_host[i] = Scalar[dtype](Float32(i * 2))

        var out = LayoutTensor[dtype, Layout.row_major(4), MutAnyOrigin](out_buf)
        var a = LayoutTensor[dtype, Layout.row_major(4), ImmutAnyOrigin](a_buf)
        var b = LayoutTensor[dtype, Layout.row_major(4), ImmutAnyOrigin](b_buf)

        ctx.enqueue_function_checked[simple_add_kernel, simple_add_kernel](
            out, a, b, grid_dim=(1, 1), block_dim=(4, 1)
        )
        ctx.synchronize()

        with out_buf.map_to_host() as out_host:
            print("Result:", out_host[0], out_host[1], out_host[2], out_host[3])
            # Expected: 0, 3, 6, 9

    print("Test passed!")
