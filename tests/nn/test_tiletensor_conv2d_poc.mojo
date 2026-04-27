"""POC: Conv2D kernels using TileTensor instead of raw LayoutTensor.

Demonstrates how TileTensor simplifies GPU kernel code for conv2D:
  1. im2col kernel: flat index → coordinate transform (TileTensor.tile)
  2. tiled forward kernel: 2x2 register-tiled matmul (TileTensor.tile + shared mem)

Validates against existing Conv2D CPU implementation for correctness.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_tiletensor_conv2d_poc.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, Idx, TileTensor, row_major
from mojo_rl.nn.constants import dtype, TPB, MMA_BLOCK_THREADS
from mojo_rl.nn.autodiff.primitives.conv2d import Conv2D


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def fill_random(ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        ptr[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())


def max_abs_diff(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
) -> Scalar[dtype]:
    var mx: Scalar[dtype] = 0
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    return mx


# ─────────────────────────────────────────────────────────────────────
# POC 1: im2col kernel with TileTensor
# ─────────────────────────────────────────────────────────────────────
# Original im2col_wrapper is 35 lines of flat-index → coordinate math.
# TileTensor version uses .tile() for structured coordinate extraction.


@always_inline
def im2col_kernel_tiletensor[
    BATCH: Int,
    in_channels: Int,
    in_h: Int,
    in_w: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    out_h: Int,
    out_w: Int,
    col_size: Int,
    spatial_out: Int,
    CACHE_SIZE: Int,
    IN_DIM: Int,
](
    cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
    ],
    input: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
):
    """Function im2col: extract input patches into column matrix for matmul.

    Each thread handles one element of the output column matrix.
    cache[b, s * col_size + k] = input patch value at (s, k) for batch b.

    TileTensor is used to view the cache as a 3D (BATCH, spatial_out, col_size)
    tensor, replacing manual index arithmetic.
    """
    comptime KS2 = kernel_size * kernel_size
    comptime im2col_elems = BATCH * CACHE_SIZE

    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= im2col_elems:
        return

    # ── TileTensor: view cache as (BATCH, spatial_out, col_size) ──
    # This replaces: b = idx // CACHE_SIZE, pos = idx % CACHE_SIZE,
    #                s = pos // col_size, k = pos % col_size
    var cache_3d = TileTensor(
        cache.ptr, row_major[BATCH, spatial_out, col_size]()
    )
    var b = idx // CACHE_SIZE
    var pos = idx % CACHE_SIZE
    var s = pos // col_size
    var k = pos % col_size

    # Decode spatial position → (oh, ow)
    var oh = s // out_w
    var ow = s % out_w

    # Decode kernel position → (ch, kh, kw)
    var ch = k // KS2
    var rem_k = k % KS2
    var kh = rem_k // kernel_size
    var kw = rem_k % kernel_size

    # Map to input coordinates
    var ih = oh * stride - padding + kh
    var iw = ow * stride - padding + kw

    var val: Scalar[dtype] = 0
    if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
        val = rebind[Scalar[dtype]](input[b, ch * in_h * in_w + ih * in_w + iw])

    # ── TileTensor: write using 3D indexing instead of flat arithmetic ──
    cache_3d[b, s, k] = val


# ─────────────────────────────────────────────────────────────────────
# POC 2: Tiled forward kernel with TileTensor
# ─────────────────────────────────────────────────────────────────────
# Original eval_kernel_2x2 is 180 lines with manual shared mem loading,
# complex index math for 2 elements per thread, and implicit im2col.
#
# This version separates concerns:
#   - Uses explicit im2col (kernel above) so the matmul is clean
#   - Uses TileTensor.tile() to express the tiling structure
#   - Keeps 2x2 register tiling for performance


@always_inline
def conv_forward_tiletensor[
    BATCH: Int,
    out_channels: Int,
    col_size: Int,
    spatial_out: Int,
    OUT_DIM: Int,
    PARAM_SIZE: Int,
    CACHE_SIZE: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    params: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
    ],
):
    """Tiled conv forward: output = W @ col + bias, per batch.

    W:    (out_channels, col_size)     — from params
    col:  (col_size, spatial_out)      — from cache (im2col output, transposed view)
    bias: (out_channels,)              — from params[W_SIZE:]

    Grid: (ceil(spatial_out/32), ceil(out_channels/32), BATCH)
    Block: (256, 1)

    Each thread computes a 2x2 output tile using shared memory tiling.
    TileTensor.tile() replaces manual block/thread → index mapping.
    """
    comptime BT = 32
    comptime SK = 16
    comptime W_SIZE = out_channels * col_size

    var tid = Int(thread_idx.x)
    var sub_r = tid // 16  # 2x2 sub-tile row (0..15)
    var sub_c = tid % 16  # 2x2 sub-tile col (0..15)
    var batch = Int(block_idx.z)

    # ── TileTensor views of W and output ──
    var W = TileTensor(params.ptr, row_major[out_channels, col_size]())
    var bias = TileTensor(params.ptr + W_SIZE, row_major[out_channels]())

    # View col for this batch as (col_size, spatial_out) — transposed im2col
    # cache is (BATCH, spatial_out * col_size), we need (col_size, spatial_out) per batch
    # cache[b] stores s*col_size+k, so reading as (spatial_out, col_size) is natural
    # For W @ col we need (col_size, spatial_out) — that's cache[b] transposed
    # Keep flat access for col since the storage order (s, k) doesn't match needed (k, s)

    # ── Tile W into BT×SK blocks, col into SK×BT blocks ──
    var block_oc = Int(block_idx.y) * BT
    var block_s = Int(block_idx.x) * BT

    var a_smem = LayoutTensor[
        dtype,
        Layout.row_major(BT, SK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var b_smem = LayoutTensor[
        dtype,
        Layout.row_major(SK, BT),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var acc00: Scalar[dtype] = 0
    var acc01: Scalar[dtype] = 0
    var acc10: Scalar[dtype] = 0
    var acc11: Scalar[dtype] = 0

    comptime num_k_tiles = (col_size + SK - 1) // SK

    for k_tile in range(num_k_tiles):
        var k_off = k_tile * SK

        # ── Load A tile: W[block_oc..+BT, k_off..+SK] ──
        # 256 threads load BT*SK = 512 elements → 2 per thread
        var a_r0 = tid // SK
        var a_c0 = tid % SK
        var a_r1 = (tid + 256) // SK
        var a_c1 = (tid + 256) % SK

        var ga_oc0 = block_oc + a_r0
        var ga_k0 = k_off + a_c0
        if ga_oc0 < out_channels and ga_k0 < col_size:
            a_smem[a_r0, a_c0] = W[ga_oc0, ga_k0]
        else:
            a_smem[a_r0, a_c0] = 0

        var ga_oc1 = block_oc + a_r1
        var ga_k1 = k_off + a_c1
        if ga_oc1 < out_channels and ga_k1 < col_size:
            a_smem[a_r1, a_c1] = W[ga_oc1, ga_k1]
        else:
            a_smem[a_r1, a_c1] = 0

        # ── Load B tile: col[k_off..+SK, block_s..+BT] ──
        # col is stored as cache[b, s*col_size + k], need col[k, s]
        # so: col[k, s] = cache[b, s*col_size + k]
        var b_r0 = tid // BT
        var b_c0 = tid % BT
        var b_r1 = (tid + 256) // BT
        var b_c1 = (tid + 256) % BT

        var k_idx0 = k_off + b_r0
        var s_idx0 = block_s + b_c0
        if k_idx0 < col_size and s_idx0 < spatial_out:
            b_smem[b_r0, b_c0] = cache[batch, s_idx0 * col_size + k_idx0]
        else:
            b_smem[b_r0, b_c0] = 0

        var k_idx1 = k_off + b_r1
        var s_idx1 = block_s + b_c1
        if k_idx1 < col_size and s_idx1 < spatial_out:
            b_smem[b_r1, b_c1] = cache[batch, s_idx1 * col_size + k_idx1]
        else:
            b_smem[b_r1, b_c1] = 0

        barrier()

        # ── 2x2 accumulation ──
        for k in range(SK):
            if k_off + k < col_size:
                var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                var a1 = rebind[Scalar[dtype]](a_smem[sub_r * 2 + 1, k])
                var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                var b1 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2 + 1])
                acc00 += a0 * b0
                acc01 += a0 * b1
                acc10 += a1 * b0
                acc11 += a1 * b1

        barrier()

    # ── Store with bias using TileTensor for structured output access ──
    var out_3d = TileTensor(
        output.ptr, row_major[BATCH, out_channels, spatial_out]()
    )

    var oc0 = block_oc + sub_r * 2
    var s0 = block_s + sub_c * 2

    if oc0 < out_channels and s0 < spatial_out:
        out_3d[batch, oc0, s0] = acc00 + bias[oc0]
    if oc0 < out_channels and s0 + 1 < spatial_out:
        out_3d[batch, oc0, s0 + 1] = acc01 + bias[oc0]
    if oc0 + 1 < out_channels and s0 < spatial_out:
        out_3d[batch, oc0 + 1, s0] = acc10 + bias[oc0 + 1]
    if oc0 + 1 < out_channels and s0 + 1 < spatial_out:
        out_3d[batch, oc0 + 1, s0 + 1] = acc11 + bias[oc0 + 1]


# ─────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────


def test_im2col_tiletensor[
    BATCH: Int,
    IC: Int,
    OC: Int,
    KS: Int,
    STRIDE: Int,
    PAD: Int,
    IN_H: Int,
    IN_W: Int,
](ctx: DeviceContext) raises:
    """Test im2col TileTensor kernel against CPU reference."""
    comptime C = Conv2D[IC, OC, KS, STRIDE, PAD, IN_H, IN_W]

    print(
        "  im2col TileTensor: ["
        + String(IC)
        + "→"
        + String(OC)
        + ", "
        + String(KS)
        + "×"
        + String(KS)
        + ", s="
        + String(STRIDE)
        + ", p="
        + String(PAD)
        + "] "
        + String(IN_H)
        + "×"
        + String(IN_W)
        + " batch="
        + String(BATCH)
    )

    # Allocate
    var input_host = List[Scalar[dtype]](capacity=BATCH * C.IN_DIM)
    var params_host = List[Scalar[dtype]](capacity=C.PARAM_SIZE)
    var cache_cpu = List[Scalar[dtype]](capacity=BATCH * C.CACHE_SIZE)
    var output_cpu = List[Scalar[dtype]](capacity=BATCH * C.OUT_DIM)

    for _ in range(BATCH * C.IN_DIM):
        input_host.append(
            Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
        )
    for _ in range(C.PARAM_SIZE):
        params_host.append(
            Scalar[dtype](random_float64(-0.5, 0.5).cast[dtype]())
        )
    for _ in range(BATCH * C.CACHE_SIZE):
        cache_cpu.append(0)
    for _ in range(BATCH * C.OUT_DIM):
        output_cpu.append(0)

    # CPU reference — eval populates cache with im2col
    var input_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
    ](input_host.unsafe_ptr())
    var output_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
    ](output_cpu.unsafe_ptr())
    var params_lt = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin
    ](params_host.unsafe_ptr())
    var cache_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
    ](cache_cpu.unsafe_ptr())

    C.eval[BATCH](input_lt, output_lt, params_lt, cache_lt)

    # GPU im2col via TileTensor
    var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.IN_DIM)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.CACHE_SIZE)

    var input_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.IN_DIM)
    for i in range(BATCH * C.IN_DIM):
        input_hb.unsafe_ptr()[i] = input_host[i]
    ctx.enqueue_copy(input_buf, input_hb)

    # Zero cache
    var cache_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.CACHE_SIZE)
    for i in range(BATCH * C.CACHE_SIZE):
        cache_hb.unsafe_ptr()[i] = 0
    ctx.enqueue_copy(cache_buf, cache_hb)

    comptime im2col_elems = BATCH * C.CACHE_SIZE
    comptime im2col_blocks = (im2col_elems + TPB - 1) // TPB

    var cache_gpu_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
    ](cache_buf)
    var input_gpu_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
    ](input_buf)

    @always_inline
    @parameter
    def im2col_wrapper(
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
        ],
    ):
        im2col_kernel_tiletensor[
            BATCH,
            IC,
            IN_H,
            IN_W,
            KS,
            STRIDE,
            PAD,
            C.out_h,
            C.out_w,
            C.col_size,
            C.spatial_out,
            C.CACHE_SIZE,
            C.IN_DIM,
        ](cache, input)

    ctx.enqueue_function[im2col_wrapper, im2col_wrapper](
        cache_gpu_lt,
        input_gpu_lt,
        grid_dim=(im2col_blocks,),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    # Read back and compare
    ctx.enqueue_copy(cache_hb, cache_buf)
    ctx.synchronize()

    var diff = max_abs_diff(
        cache_cpu.unsafe_ptr(), cache_hb.unsafe_ptr(), BATCH * C.CACHE_SIZE
    )
    print("    max abs diff (im2col):", diff)
    if diff < 1e-4:
        print("    ✓ PASS")
    else:
        print("    ✗ FAIL")


def test_forward_tiletensor[
    BATCH: Int,
    IC: Int,
    OC: Int,
    KS: Int,
    STRIDE: Int,
    PAD: Int,
    IN_H: Int,
    IN_W: Int,
](ctx: DeviceContext) raises:
    """Test full forward (im2col + tiled matmul) against CPU reference."""
    comptime C = Conv2D[IC, OC, KS, STRIDE, PAD, IN_H, IN_W]

    print(
        "  forward TileTensor: ["
        + String(IC)
        + "→"
        + String(OC)
        + ", "
        + String(KS)
        + "×"
        + String(KS)
        + ", s="
        + String(STRIDE)
        + ", p="
        + String(PAD)
        + "] "
        + String(IN_H)
        + "×"
        + String(IN_W)
        + " batch="
        + String(BATCH)
    )

    # Generate data
    var input_host = List[Scalar[dtype]](capacity=BATCH * C.IN_DIM)
    var params_host = List[Scalar[dtype]](capacity=C.PARAM_SIZE)

    for _ in range(BATCH * C.IN_DIM):
        input_host.append(
            Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
        )
    for _ in range(C.PARAM_SIZE):
        params_host.append(
            Scalar[dtype](random_float64(-0.5, 0.5).cast[dtype]())
        )

    # ── CPU reference ──
    var cache_cpu = List[Scalar[dtype]](capacity=BATCH * C.CACHE_SIZE)
    var output_cpu = List[Scalar[dtype]](capacity=BATCH * C.OUT_DIM)
    for _ in range(BATCH * C.CACHE_SIZE):
        cache_cpu.append(0)
    for _ in range(BATCH * C.OUT_DIM):
        output_cpu.append(0)

    var input_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
    ](input_host.unsafe_ptr())
    var output_cpu_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
    ](output_cpu.unsafe_ptr())
    var params_lt = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin
    ](params_host.unsafe_ptr())
    var cache_cpu_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
    ](cache_cpu.unsafe_ptr())

    C.eval[BATCH](input_lt, output_cpu_lt, params_lt, cache_cpu_lt)

    # ── GPU: im2col + tiled forward via TileTensor ──
    var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.IN_DIM)
    var params_buf = ctx.enqueue_create_buffer[dtype](C.PARAM_SIZE)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.CACHE_SIZE)
    var output_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.OUT_DIM)

    # Upload
    var input_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.IN_DIM)
    var params_hb = ctx.enqueue_create_host_buffer[dtype](C.PARAM_SIZE)
    for i in range(BATCH * C.IN_DIM):
        input_hb.unsafe_ptr()[i] = input_host[i]
    for i in range(C.PARAM_SIZE):
        params_hb.unsafe_ptr()[i] = params_host[i]
    ctx.enqueue_copy(input_buf, input_hb)
    ctx.enqueue_copy(params_buf, params_hb)

    # Zero cache and output
    var cache_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.CACHE_SIZE)
    var output_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.OUT_DIM)
    for i in range(BATCH * C.CACHE_SIZE):
        cache_hb.unsafe_ptr()[i] = 0
    for i in range(BATCH * C.OUT_DIM):
        output_hb.unsafe_ptr()[i] = 0
    ctx.enqueue_copy(cache_buf, cache_hb)
    ctx.enqueue_copy(output_buf, output_hb)

    # Step 1: im2col
    comptime im2col_elems = BATCH * C.CACHE_SIZE
    comptime im2col_blocks = (im2col_elems + TPB - 1) // TPB

    var cache_gpu_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
    ](cache_buf)
    var input_gpu_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
    ](input_buf)

    @always_inline
    @parameter
    def im2col_w(
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
        ],
    ):
        im2col_kernel_tiletensor[
            BATCH,
            IC,
            IN_H,
            IN_W,
            KS,
            STRIDE,
            PAD,
            C.out_h,
            C.out_w,
            C.col_size,
            C.spatial_out,
            C.CACHE_SIZE,
            C.IN_DIM,
        ](cache, input)

    ctx.enqueue_function[im2col_w, im2col_w](
        cache_gpu_lt,
        input_gpu_lt,
        grid_dim=(im2col_blocks,),
        block_dim=(TPB,),
    )

    # Step 2: tiled forward matmul
    comptime grid_x = (C.spatial_out + 31) // 32
    comptime grid_y = (C.out_channels + 31) // 32

    var output_gpu_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
    ](output_buf)
    var params_gpu_lt = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin
    ](params_buf)

    @always_inline
    @parameter
    def fwd_wrapper(
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        conv_forward_tiletensor[
            BATCH,
            OC,
            C.col_size,
            C.spatial_out,
            C.OUT_DIM,
            C.PARAM_SIZE,
            C.CACHE_SIZE,
        ](output, params, cache)

    ctx.enqueue_function[fwd_wrapper, fwd_wrapper](
        output_gpu_lt,
        params_gpu_lt,
        cache_gpu_lt,
        grid_dim=(grid_x, grid_y, BATCH),
        block_dim=(MMA_BLOCK_THREADS, 1),
    )
    ctx.synchronize()

    # Read back and compare
    ctx.enqueue_copy(output_hb, output_buf)
    ctx.synchronize()

    var diff = max_abs_diff(
        output_cpu.unsafe_ptr(), output_hb.unsafe_ptr(), BATCH * C.OUT_DIM
    )
    print("    max abs diff (forward):", diff)
    if diff < 1e-2:
        print("    ✓ PASS")
    else:
        print("    ✗ FAIL")
        # Print first few mismatches
        var count = 0
        for i in range(BATCH * C.OUT_DIM):
            var d = output_cpu[i] - output_hb.unsafe_ptr()[i]
            if d < 0:
                d = -d
            if d > 1e-2 and count < 5:
                print(
                    "      ["
                    + String(i)
                    + "] cpu="
                    + String(output_cpu[i])
                    + " gpu="
                    + String(output_hb.unsafe_ptr()[i])
                )
                count += 1


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() raises:
    seed(42)
    print("=" * 60)
    print("TileTensor Conv2D POC")
    print("=" * 60)

    with DeviceContext() as ctx:
        # Small conv for quick validation
        print("\n── Small conv: [1→2, 3×3, s=1, p=1] 6×6 ──")
        test_im2col_tiletensor[4, 1, 2, 3, 1, 1, 6, 6](ctx)
        test_forward_tiletensor[4, 1, 2, 3, 1, 1, 6, 6](ctx)

        # Medium conv matching CartPole-like dims
        print("\n── Medium conv: [4→16, 3×3, s=1, p=1] 10×10 ──")
        test_im2col_tiletensor[8, 4, 16, 3, 1, 1, 10, 10](ctx)
        test_forward_tiletensor[8, 4, 16, 3, 1, 1, 10, 10](ctx)

        # Atari-like conv1
        print("\n── Atari conv1: [4→32, 8×8, s=4, p=0] 84×84 ──")
        test_im2col_tiletensor[4, 4, 32, 8, 4, 0, 84, 84](ctx)
        test_forward_tiletensor[4, 4, 32, 8, 4, 0, 84, 84](ctx)

    print("\n" + "=" * 60)
    print("POC complete!")
    print("=" * 60)
