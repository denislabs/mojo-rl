"""Parity test: LayerNormV2 matches v1 LayerNorm bit-for-bit on CPU."""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.primitives.layer_norm import LayerNorm
from mojo_rl.nn2.primitives.layer_norm_v2 import LayerNormV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero


comptime BATCH = 4
comptime DIM = 7


def _seed(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, salt: UInt64):
    var state: UInt64 = salt
    for k in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        p[k] = (r - Scalar[DT](0.5))


def main() raises:
    var l1 = LayerNorm[DIM].make[target="cpu", INIT=Zero]()
    var l2 = LayerNormV2[DIM].make[target="cpu", INIT=Zero]()

    # Both should initialize γ=1, β=0. Skew γ/β in v1 and replicate to v2
    # to test a non-default value.
    var g1_p = l1.gamma.unsafe_ptr()
    var b1_p = l1.beta.unsafe_ptr()
    var g2_p = l2.gamma.value_unsafe_ptr_cpu()
    var b2_p = l2.beta.value_unsafe_ptr_cpu()
    var state: UInt64 = UInt64(0xFEEDFACE)
    for k in range(DIM):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        g1_p[k] = Scalar[DT](0.7) + r
        b1_p[k] = r - Scalar[DT](0.3)
        g2_p[k] = g1_p[k]
        b2_p[k] = b1_p[k]

    # ── Forward parity ───────────────────────────────────────────────
    var in_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var o1_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var o2_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    _seed(in_p, BATCH * DIM, UInt64(0xABCDEF12))

    var in_t  = TileTensor(in_p, row_major[BATCH, DIM]())
    var o1_t  = TileTensor(o1_p, row_major[BATCH, DIM]())
    var o2_t  = TileTensor(o2_p, row_major[BATCH, DIM]())
    l1.forward["cpu", BATCH](in_t, o1_t)
    l2.forward["cpu", BATCH](in_t, o2_t)

    var ok_fwd = True
    for k in range(BATCH * DIM):
        if o1_p[k] != o2_p[k]:
            ok_fwd = False
    print("forward: PASS" if ok_fwd else "forward: FAIL")

    # ── Backward(mode='all') parity ──────────────────────────────────
    var go_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi1_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi2_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    _seed(go_p, BATCH * DIM, UInt64(0x55AA55AA))
    for k in range(BATCH * DIM):
        gi1_p[k] = Scalar[DT](0.0)
        gi2_p[k] = Scalar[DT](0.0)
    var go_t  = TileTensor(go_p,  row_major[BATCH, DIM]())
    var gi1_t = TileTensor(gi1_p, row_major[BATCH, DIM]())
    var gi2_t = TileTensor(gi2_p, row_major[BATCH, DIM]())
    l1.backward["cpu", BATCH](go_t, gi1_t)
    l2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)

    var ok_gi = True
    for k in range(BATCH * DIM):
        if gi1_p[k] != gi2_p[k]:
            ok_gi = False
    var gg1_p = l1.grad_gamma.unsafe_ptr()
    var gg2_p = l2.gamma.grad_unsafe_ptr_cpu()
    var gb1_p = l1.grad_beta.unsafe_ptr()
    var gb2_p = l2.beta.grad_unsafe_ptr_cpu()
    var ok_gg = True
    var ok_gb = True
    for k in range(DIM):
        if gg1_p[k] != gg2_p[k]:
            ok_gg = False
        if gb1_p[k] != gb2_p[k]:
            ok_gb = False
    print(
        "backward(all): grad_in=", "PASS" if ok_gi else "FAIL",
        " grad_gamma=", "PASS" if ok_gg else "FAIL",
        " grad_beta=",  "PASS" if ok_gb else "FAIL",
    )

    # ── Backward(mode='input_only') parity ───────────────────────────
    var gg_pre = List[Scalar[DT]]()
    var gb_pre = List[Scalar[DT]]()
    for k in range(DIM):
        gg_pre.append(gg2_p[k])
        gb_pre.append(gb2_p[k])

    var gi1b_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi2b_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        gi1b_p[k] = Scalar[DT](0.0)
        gi2b_p[k] = Scalar[DT](0.0)
    var gi1b_t = TileTensor(gi1b_p, row_major[BATCH, DIM]())
    var gi2b_t = TileTensor(gi2b_p, row_major[BATCH, DIM]())
    l1.backward_input["cpu", BATCH](go_t, gi1b_t)
    l2.backward["cpu", BATCH, mode="input_only"](go_t, gi2b_t)

    var ok_bwi = True
    for k in range(BATCH * DIM):
        if gi1b_p[k] != gi2b_p[k]:
            ok_bwi = False
    var ok_clean = True
    for k in range(DIM):
        if gg2_p[k] != gg_pre[k] or gb2_p[k] != gb_pre[k]:
            ok_clean = False
    print(
        "backward(input_only): grad_in=", "PASS" if ok_bwi else "FAIL",
        " params_unchanged=", "PASS" if ok_clean else "FAIL",
    )

    # ── zero_grad parity ─────────────────────────────────────────────
    l2.zero_grad[target="cpu"]()
    var ok_zg = True
    for k in range(DIM):
        if gg2_p[k] != Scalar[DT](0.0) or gb2_p[k] != Scalar[DT](0.0):
            ok_zg = False
    print("zero_grad: PASS" if ok_zg else "zero_grad: FAIL")

    var all_ok = (
        ok_fwd and ok_gi and ok_gg and ok_gb
        and ok_bwi and ok_clean and ok_zg
    )
    if all_ok:
        print()
        print("PASS — LayerNormV2 is bit-identical to v1 on CPU.")
    else:
        raise Error("layer_norm_v2 parity test failed")

    in_p.free()
    o1_p.free()
    o2_p.free()
    go_p.free()
    gi1_p.free()
    gi2_p.free()
    gi1b_p.free()
    gi2b_p.free()
