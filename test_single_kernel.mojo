"""Test importing a single kernel from a2c_native."""

from deep_agents.gpu.a2c_native import naive_matmul_kernel
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

comptime dtype = DType.float32


fn main() raises:
    print("Testing single kernel import from a2c_native")

    comptime M = 4
    comptime K = 4
    comptime N = 4
    comptime TILE = 4

    with DeviceContext() as ctx:
        var A_buf = ctx.enqueue_create_buffer[dtype](M * K)
        var B_buf = ctx.enqueue_create_buffer[dtype](K * N)
        var C_buf = ctx.enqueue_create_buffer[dtype](M * N)

        with A_buf.map_to_host() as A_host, B_buf.map_to_host() as B_host:
            for i in range(M * K):
                A_host[i] = Scalar[dtype](1.0)
            for i in range(K * N):
                B_host[i] = Scalar[dtype](1.0)

        var C = LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin](C_buf)
        var A = LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin](A_buf)
        var B = LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin](B_buf)

        ctx.enqueue_function_checked[
            naive_matmul_kernel[M, K, N, TILE], naive_matmul_kernel[M, K, N, TILE]
        ](
            C,
            A,
            B,
            grid_dim=(1, 1),
            block_dim=(TILE, TILE),
        )
        ctx.synchronize()

        with C_buf.map_to_host() as C_host:
            print("C[0,0] =", C_host[0], "(expected 4.0)")

    print("Test passed!")
