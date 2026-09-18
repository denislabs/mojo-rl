"""`bmm_tiled` — a batched matmul for the shapes MAX's dispatch leaves on the
floor, namely attention's two products.

`bmm` (MAX's `batched_matmul`) is the right default nearly everywhere. It is
not the right default for `Q.Kt` and `A.V`: MAX's multistage GEMM wants
`n % 128 == 0` and `k >= 128`, and BOTH attention products fail that — scores
have `k = HEAD_DIM` (64), the context product has `n = HEAD_DIM` — so both
fall to the vendor path. Measured (`benchmarks/attention_matmul_bench.mojo`),
SigLIP's 1.61 GFLOP product took 18.2 ms on an M1 and 0.224 ms on a 5090:
88 and 7200 GFLOP/s, a tenth of each machine.

Two kernels, because the shapes that need this are not one family:

  * `_bmm_tile4_kernel` — a 64x64 output tile per block, 16x16 threads, 4x4
    outputs each. Each shared-memory element is reused four times from
    registers. It wants a big output: 8.4x on an M1 and 4.0x on a 5090 at
    SigLIP's shape, but 0.75-0.85x on ACT's DEPLOY shapes (M=60, N=32), where
    a 64x64 tile is mostly out-of-bounds threads.
  * `_bmm_tile1_kernel` — 16x16, one output per thread. 2.1x on an M1 at those
    small shapes, and never slower than `bmm` on a 5090.

`bmm_tiled_is_worth_it` picks between them (and defers to `bmm` when neither
fits), at compile time, from the extents alone.

⚠ THE THRESHOLDS ARE SET FROM AN M1 AND A 5090 AND NOTHING ELSE YET. The
board that actually deploys this — the Orin — has not run
`attn-matmul-bench-jetson`; nothing calls `bmm_tiled` until it has. Both
constants below are one number each, so a board run moves them without
touching a kernel.

⚠ NOT A DROP-IN FOR `bmm` EVERYWHERE. These kernels accumulate `k` in
increasing order with no split-K and no tensor cores, which is why they come
out bit-identical to `bmm` at every shape measured — the property the
attention cache gates rely on. A shape with a long `k` and a small output
would be better served by MAX's split-K, which this does not do.
"""

from max.gpu import barrier, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT


comptime MMT_T1: Int = 16
"""Edge of the one-output-per-thread tile (256 threads per block)."""

comptime MMT_TM: Int = 64
comptime MMT_TN: Int = 64
comptime MMT_TK: Int = 16
comptime MMT_RM: Int = 4
comptime MMT_RN: Int = 4
"""The 64x64 tile: 16x16 threads, 4x4 outputs each, staged 16 deep in k."""

comptime MMT_MIN_TILE4: Int = 64
"""Below this in M or N, the 64x64 tile spends most of its threads out of
bounds and loses to the 16x16 one — measured on a 5090 at ACT's deploy shapes
(0.85x and 0.75x against `bmm`, where 16x16 held 1.0-1.05x)."""

comptime MMT_MIN_WORK: Int = 1 << 20
"""Total MACs (BH*M*N*K) below which neither tiled kernel is worth choosing
over `bmm`: at ACT's decoder shapes the whole product is ~2.5 M MACs and every
variant lands within launch overhead of the others."""


@always_inline
def bmm_tiled_is_worth_it[BH: Int, M: Int, N: Int, K: Int]() -> Bool:
    """Whether `bmm_tiled` should be used instead of `bmm` at this shape."""
    return BH * M * N * K >= MMT_MIN_WORK


@always_inline
def bmm_tiled_uses_big_tile[M: Int, N: Int]() -> Bool:
    """Which of the two kernels `bmm_tiled` dispatches to."""
    return M >= MMT_MIN_TILE4 and N >= MMT_MIN_TILE4


def _bmm_tile1_kernel[BH: Int, M: Int, N: Int, K: Int](
    a: LayoutTensor[DT, Layout.row_major(BH * M * K), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(BH * K * N), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(BH * M * N), MutAnyOrigin],
):
    """One output per thread, both operands staged through shared memory."""
    var sa = LayoutTensor[
        DT, Layout.row_major(MMT_T1, MMT_T1), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var sb = LayoutTensor[
        DT, Layout.row_major(MMT_T1, MMT_T1), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var bh = Int(block_idx.z)
    var ty = Int(thread_idx.y)
    var tx = Int(thread_idx.x)
    var row = Int(block_idx.y) * MMT_T1 + ty
    var col = Int(block_idx.x) * MMT_T1 + tx
    var a_base = bh * M * K
    var b_base = bh * K * N
    var acc = Scalar[DT](0)

    for kt in range(0, K, MMT_T1):
        var ak = kt + tx
        sa[ty, tx] = (
            rebind[Scalar[DT]](a.ptr[unsafe_offset=a_base + row * K + ak])
            if (row < M and ak < K) else Scalar[DT](0)
        )
        var bk = kt + ty
        sb[ty, tx] = (
            rebind[Scalar[DT]](b.ptr[unsafe_offset=b_base + bk * N + col])
            if (bk < K and col < N) else Scalar[DT](0)
        )
        barrier()
        for kk in range(MMT_T1):
            acc += rebind[Scalar[DT]](sa[ty, kk]) * rebind[Scalar[DT]](
                sb[kk, tx]
            )
        barrier()

    if row < M and col < N:
        o.ptr[unsafe_offset=bh * M * N + row * N + col] = acc


def _bmm_tile4_kernel[BH: Int, M: Int, N: Int, K: Int](
    a: LayoutTensor[DT, Layout.row_major(BH * M * K), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(BH * K * N), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(BH * M * N), MutAnyOrigin],
):
    """4x4 outputs per thread — each shared-memory element feeds four MACs
    from registers instead of one, which is what lifts a tiled matmul off the
    shared-memory bandwidth bound."""
    var sa = LayoutTensor[
        DT, Layout.row_major(MMT_TM, MMT_TK), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var sb = LayoutTensor[
        DT, Layout.row_major(MMT_TK, MMT_TN), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var bh = Int(block_idx.z)
    var ty = Int(thread_idx.y)
    var tx = Int(thread_idx.x)
    var tid = ty * MMT_T1 + tx
    var m0 = Int(block_idx.y) * MMT_TM
    var n0 = Int(block_idx.x) * MMT_TN
    var a_base = bh * M * K
    var b_base = bh * K * N
    comptime THREADS = MMT_T1 * MMT_T1

    var acc = Array[Scalar[DT], MMT_RM * MMT_RN](fill=Scalar[DT](0))

    for kt in range(0, K, MMT_TK):
        comptime for l in range(MMT_TM * MMT_TK // THREADS):
            var idx = tid + l * THREADS
            var r = idx // MMT_TK
            var cc = idx % MMT_TK
            sa[r, cc] = (
                rebind[Scalar[DT]](
                    a.ptr[unsafe_offset=a_base + (m0 + r) * K + kt + cc]
                )
                if (m0 + r < M and kt + cc < K) else Scalar[DT](0)
            )
        comptime for l in range(MMT_TK * MMT_TN // THREADS):
            var idx = tid + l * THREADS
            var r = idx // MMT_TN
            var cc = idx % MMT_TN
            sb[r, cc] = (
                rebind[Scalar[DT]](
                    b.ptr[unsafe_offset=b_base + (kt + r) * N + n0 + cc]
                )
                if (kt + r < K and n0 + cc < N) else Scalar[DT](0)
            )
        barrier()
        for kk in range(MMT_TK):
            comptime for i in range(MMT_RM):
                var av = rebind[Scalar[DT]](sa[ty * MMT_RM + i, kk])
                comptime for j in range(MMT_RN):
                    acc[i * MMT_RN + j] += av * rebind[Scalar[DT]](
                        sb[kk, tx * MMT_RN + j]
                    )
        barrier()

    comptime for i in range(MMT_RM):
        var r = m0 + ty * MMT_RM + i
        comptime for j in range(MMT_RN):
            var cc = n0 + tx * MMT_RN + j
            if r < M and cc < N:
                o.ptr[unsafe_offset=bh * M * N + r * N + cc] = acc[
                    i * MMT_RN + j
                ]


@always_inline
def bmm_tiled[
    *, BH: Int, M: Int, N: Int, K: Int
](
    mut o: DeviceBuffer[DT],
    a: DeviceBuffer[DT],
    b: DeviceBuffer[DT],
    c: DeviceContext,
) raises:
    """`o[BH, M, N] = a[BH, M, K] @ b[BH, K, N]`, row-major, on `c`.

    `b` must already be laid out `(BH, K, N)` — there is no transposing
    variant, deliberately: the caller that needs `Kt` materialises it in one
    pass while packing (`_xa_pack_kt_kernel`), which is cheaper than either a
    separate transpose or a strided read inside the matmul."""
    comptime lay_a = Layout.row_major(BH * M * K)
    comptime lay_b = Layout.row_major(BH * K * N)
    comptime lay_o = Layout.row_major(BH * M * N)
    # The origin-linking ctor takes the buffer BY REFERENCE, so the views are
    # built from locals rather than from the (by-value) arguments.
    var ab = a
    var bb = b
    var ob = o
    var av = LayoutTensor[DT, lay_a, MutAnyOrigin](ab)
    var bv = LayoutTensor[DT, lay_b, MutAnyOrigin](bb)
    var ov = LayoutTensor[DT, lay_o, MutAnyOrigin](ob)
    comptime if bmm_tiled_uses_big_tile[M, N]():
        c.enqueue_function[_bmm_tile4_kernel[BH, M, N, K]](
            av, bv, ov,
            grid_dim=(
                (N + MMT_TN - 1) // MMT_TN, (M + MMT_TM - 1) // MMT_TM, BH
            ),
            block_dim=(MMT_T1, MMT_T1),
        )
    else:
        c.enqueue_function[_bmm_tile1_kernel[BH, M, N, K]](
            av, bv, ov,
            grid_dim=(
                (N + MMT_T1 - 1) // MMT_T1, (M + MMT_T1 - 1) // MMT_T1, BH
            ),
            block_dim=(MMT_T1, MMT_T1),
        )
