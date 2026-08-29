"""Does routing `Linear`'s dW GEMM through our own split-K workspace change
the gradient?

`Linear`'s backward computes `grad_w = cacheTᵀ @ go`, where K is the FLATTENED
row count `batch * tokens`. That is long-K and skinny by construction, which is
exactly when `select_config` partitions K — and MAX's split-K allocates its
reduction workspace on every call, which costs a cuMemAlloc/cuMemFree pair and
makes the step impossible to capture into a CUDA graph
(`docs/MODULAR_MATMUL_ALLOC_REPORT.md` Measurement 5). `Linear` now runs that
branch against a workspace it owns.

This is the in-process A/B of exactly that change: two identical `Linear`s, the
same weights and the same input, one pinned to `_sk_p = 1` so it takes plain
`max_matmul`. Their `grad_w` must agree.

⚠ VACUITY IS THE FAILURE MODE HERE. On any part where the split path does not
apply — Apple, AMD, H100, sm_100 Blackwell, or simply a B under
`select_config`'s `min_k_partition = 1024` floor — both arms run the SAME code
and "0 mismatches" means "nothing was tested". So the test PRINTS the partition
count it chose and FAILS if no shape in the sweep actually split. A green run
with `split shapes: 0` is a red run.

    pixi run -e nvidia mojo run -I . tests/nn/test_linear_splitk_dw_gpu.mojo
    MOJO_RL_SPLITK=0 pixi run -e nvidia mojo run -I . tests/nn/...   # both arms plain
"""

from std.math import abs
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.splitk_gemm import splitk_path_applies
from mojo_rl.nn.primitives.linear import Linear


def check[IN: Int, OUT: Int, B: Int](
    ctx: DeviceContext, mut n_split: Int
) raises:
    """One shape. `B` is `batch * tokens`, the contraction of the dW GEMM."""
    comptime L = Linear[IN, OUT]

    var la = L.make["gpu", INIT=Kaiming](ctx=ctx)   # may split
    var lb = L.make["gpu", INIT=Kaiming](ctx=ctx)   # pinned to plain matmul
    lb._sk_p = 1

    # Identical weights: copy A's slab into B host-side, then push both down.
    la.weight.val.ensure_host(ctx, L.W_SIZE)
    lb.weight.val.ensure_host(ctx, L.W_SIZE)
    la.bias.val.ensure_host(ctx, L.B_SIZE)
    lb.bias.val.ensure_host(ctx, L.B_SIZE)
    for i in range(L.W_SIZE):
        lb.weight.val.data[i] = la.weight.val.data[i]
    for i in range(L.B_SIZE):
        lb.bias.val.data[i] = la.bias.val.data[i]
    la.weight.val.upload_resident(ctx)
    lb.weight.val.upload_resident(ctx)
    la.bias.val.upload_resident(ctx)
    lb.bias.val.upload_resident(ctx)

    var xa = Tensor.alloc(B * IN)
    var xb = Tensor.alloc(B * IN)
    for i in range(B * IN):
        var v = Scalar[DT](0.01) * Scalar[DT]((i % 37) - 18)
        xa.data[i] = v
        xb.data[i] = v
    xa.upload(ctx)
    xb.upload(ctx)

    var ya = Tensor.alloc_gpu(ctx, B * OUT)
    var yb = Tensor.alloc_gpu(ctx, B * OUT)
    la.forward["gpu", B](TensorRefs[1](xa), ya, Optional(ctx))
    lb.forward["gpu", B](TensorRefs[1](xb), yb, Optional(ctx))

    var goa = Tensor.alloc(B * OUT)
    var gob = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        var v = Scalar[DT](0.003) * Scalar[DT]((i % 23) - 11)
        goa.data[i] = v
        gob.data[i] = v
    goa.upload(ctx)
    gob.upload(ctx)

    var gia = Tensor.alloc_gpu(ctx, B * IN)
    var gib = Tensor.alloc_gpu(ctx, B * IN)
    la.vjp["gpu", B](TensorRefs[1](xa), goa, TensorRefs[1](gia), Optional(ctx))
    lb.vjp["gpu", B](TensorRefs[1](xb), gob, TensorRefs[1](gib), Optional(ctx))

    la.weight.grd.download(ctx)
    lb.weight.grd.download(ctx)
    ctx.synchronize()

    var max_abs = Float64(0)
    var mag = Float64(0)
    for i in range(L.W_SIZE):
        var a = Float64(la.weight.grd.data[i])
        var b = Float64(lb.weight.grd.data[i])
        if abs(b) > mag:
            mag = abs(b)
        var d = abs(a - b)
        if d > max_abs:
            max_abs = d
    var rel = (max_abs / mag) if mag > 0.0 else 0.0

    if la._sk_p > 1:
        n_split += 1

    print(
        "  IN=", IN, " OUT=", OUT, " B=", B,
        "   P=", la._sk_p,
        "   rows=", L.W_SIZE,
        "   max|dW_split - dW_plain|=", max_abs, "  rel=", rel,
        sep="",
    )

    # The reduce sums `P` separate fp32 accumulations, so this is not expected
    # to be bit-identical once P > 1 -- but it must stay at fp32 rounding, not
    # drift. A dropped K tail (see `partitions_legal`) shows up here as a
    # relative error equal to the fraction of K lost, i.e. 1e-3 or worse.
    if rel > 1e-4:
        raise Error("split-K dW disagrees with the plain GEMM beyond fp32 noise")


def main() raises:
    comptime if not splitk_path_applies[DeviceContext.default_device_info]():
        print(
            "split-K path does not apply on this device (Apple / AMD / H100 /"
            " sm_100 Blackwell) — Linear is unchanged here, nothing to test."
        )
        return

    with DeviceContext() as ctx:
        print("dW = cacheT^T @ go: M=IN, N=OUT, K=B (batch*tokens)")
        var n_split = 0

        print("== aligned shapes, UNPADDED dW branch (should split) ==")
        # ACT's transformer encoder: B = 16 * 162.
        check[256, 256, 2592](ctx, n_split)
        check[256, 1024, 2592](ctx, n_split)
        check[1024, 256, 2592](ctx, n_split)

        print("== unaligned shapes, PADDED dW branch (should split) ==")
        # `Linear` has TWO GPU dW sites — one behind the K/N padding and one
        # for the aligned case — and they are far apart in the file. The first
        # version of this change patched only the padded one, every shape above
        # is aligned, and the run came back `split shapes: 0`. Both branches
        # are covered from here on.
        check[518, 101, 2592](ctx, n_split)   # K_PAD=544, N_PAD=128
        check[24, 256, 2592](ctx, n_split)    # K_PAD=128, N_PAD=256

        print("== controls: below min_k_partition, must NOT split ==")
        # ACT's CVAE encoder (B = 16*62) and decoder self-attention (16*60).
        check[256, 256, 992](ctx, n_split)
        check[256, 256, 960](ctx, n_split)

        print()
        print("split shapes:", n_split, "of 7")
        if n_split == 0:
            raise Error(
                "NO shape took the split-K path — this run tested nothing."
                " Check select_config's min_k_partition and the device gate"
                " before reading the zeros above as a pass."
            )
        if n_split < 5:
            raise Error(
                "fewer split shapes than expected: both the aligned and the"
                " padded dW branch should split at B=2592, so a shortfall means"
                " one of the two sites is not routed"
            )
        print("all good")
