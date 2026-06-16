"""QKVToMajor layout-parity CPU test (audit L6).

QKVToMajor rearranges a token-major QKV projection into the qkv-major
layout `ScaledDotProductAttention` expects:

    out[g*SEQ*DIM + t*DIM + d] = in[t*3*DIM + g*DIM + d]

Feeding token-major straight into SDPA scrambles the position axis and
leaks future tokens past the causal mask (see
feedback_attention_qkv_layout_mismatch). This makes that permutation
explicit:

  (1) Forward: fill the input with unique values, recompute the expected
      qkv-major permutation by hand, assert bit-exact.
  (2) Backward is the inverse permutation, so vjp(forward(x)) must
      reproduce the original input bit-exact (round-trip identity).

Run: `pixi run mojo run -I . tests/nn/test_qkv_to_major_parity_cpu.mojo`
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.primitives.qkv_to_major import QKVToMajor


comptime SEQ = 2
comptime DIM = 3
comptime BATCH = 2
comptime W = 3 * SEQ * DIM  # 18


def main() raises:
    print("=" * 70)
    print("QKVToMajor layout parity (L6)")
    print("=" * 70)

    var q = QKVToMajor[SEQ, DIM].make[target="cpu", INIT=Zero]()

    var xin: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * W
    )
    var yout: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * W
    )
    # Unique values so any mis-permutation is detectable.
    for b in range(BATCH):
        for idx in range(W):
            xin[b * W + idx] = Scalar[DT](b * 100 + idx)

    var x_t = TileTensor(xin, row_major[BATCH, W]())
    var y_t = TileTensor(yout, row_major[BATCH, W]())
    q.forward["cpu", BATCH](x_t, output=y_t)

    # (1) Hand-compute the expected qkv-major permutation.
    comptime D3 = 3 * DIM
    comptime SD = SEQ * DIM
    var fwd_err: Scalar[DT] = 0.0
    for b in range(BATCH):
        for g in range(3):
            for t in range(SEQ):
                for d in range(DIM):
                    var got = yout[b * W + g * SD + t * DIM + d]
                    var want = xin[b * W + t * D3 + g * DIM + d]
                    var e = got - want
                    fwd_err += e if e >= Scalar[DT](0) else -e
    print("  forward |out - expected| =", fwd_err)
    assert_true(
        fwd_err == Scalar[DT](0.0),
        "forward must produce the exact qkv-major permutation",
    )

    # (2) Round-trip: vjp(forward output) == original input.
    var gin: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * W
    )
    var go_t = TileTensor(yout, row_major[BATCH, W]())
    var gi_t = TileTensor(gin, row_major[BATCH, W]())
    q.vjp["cpu", BATCH](go_t, gi_t)
    var rt_err: Scalar[DT] = 0.0
    for i in range(BATCH * W):
        var e = gin[i] - xin[i]
        rt_err += e if e >= Scalar[DT](0) else -e
    print("  round-trip |vjp(fwd(x)) - x| =", rt_err)
    assert_true(
        rt_err == Scalar[DT](0.0),
        "backward must be the exact inverse permutation",
    )

    xin.free(); yout.free(); gin.free()
    print("=" * 70)
    print("PASS — QKVToMajor permutation + inverse are exact")
    print("=" * 70)
