"""Parity test: LinearV2 matches v1 Linear bit-for-bit on CPU.

Forward, backward(mode='all'), backward(mode='input_only'), and
zero_grad. All four exercised at NoAMP (fp32) on CPU."""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_v2 import LinearV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero, Kaiming


comptime BATCH = 4
comptime IN  = 6
comptime OUT = 5


def _seed_buf(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, salt: UInt64):
    var state: UInt64 = salt
    for k in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        p[k] = (r - Scalar[DT](0.5))


def main() raises:
    # ── Build matched layers ─────────────────────────────────────────
    var l1 = Linear[IN, OUT].make[target="cpu", INIT=Kaiming]()
    var l2 = LinearV2[IN, OUT].make[target="cpu", INIT=Zero]()

    # Copy l1's weight + bias INTO l2 so the two layers compute the
    # same function. l2 was built with INIT=Zero so we just overwrite.
    var w_size = IN * OUT
    var b_size = OUT
    var w1_p = l1.weight.unsafe_ptr()
    var b1_p = l1.bias.unsafe_ptr()
    var w2_p = l2.weight.value_unsafe_ptr_cpu()
    var b2_p = l2.bias.value_unsafe_ptr_cpu()
    for k in range(w_size):
        w2_p[k] = w1_p[k]
    for k in range(b_size):
        b2_p[k] = b1_p[k]

    # ── Forward parity ───────────────────────────────────────────────
    var in_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var o1_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var o2_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    _seed_buf(in_p, BATCH * IN, UInt64(0xDEADBEEF))

    var in_t  = TileTensor(in_p, row_major[BATCH, IN]())
    var o1_t  = TileTensor(o1_p, row_major[BATCH, OUT]())
    var o2_t  = TileTensor(o2_p, row_major[BATCH, OUT]())
    l1.forward["cpu", BATCH](in_t, o1_t)
    l2.forward["cpu", BATCH](in_t, o2_t)

    var ok_fwd = True
    for k in range(BATCH * OUT):
        if o1_p[k] != o2_p[k]:
            ok_fwd = False
    print("forward: PASS" if ok_fwd else "forward: FAIL")

    # ── Backward(mode='all') parity ──────────────────────────────────
    # NOTE on backward-order invariant: v1 writes grad_input first then
    # accumulates grad_w; v2 reverses that. Because grad_input and the
    # cache live in DIFFERENT buffers in this test (l2's cache points
    # at `in_p`; we write grad_input into a fresh `gi2_p` buffer), the
    # ordering difference is invisible — the math comes out identical.
    # The aliasing-safety property of v2's order is exercised separately
    # by the alias-safe sub-test below.
    var go_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi1_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var gi2_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    _seed_buf(go_p, BATCH * OUT, UInt64(0xCAFEF00D))
    for k in range(BATCH * IN):
        gi1_p[k] = Scalar[DT](0.0)
        gi2_p[k] = Scalar[DT](0.0)
    var go_t  = TileTensor(go_p,  row_major[BATCH, OUT]())
    var gi1_t = TileTensor(gi1_p, row_major[BATCH, IN]())
    var gi2_t = TileTensor(gi2_p, row_major[BATCH, IN]())
    l1.backward["cpu", BATCH](go_t, gi1_t)
    l2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)

    var ok_bwd_gi = True
    for k in range(BATCH * IN):
        if gi1_p[k] != gi2_p[k]:
            ok_bwd_gi = False
    var ok_bwd_gw = True
    var gw1_p = l1.grad_w.unsafe_ptr()
    var gw2_p = l2.weight.grad_unsafe_ptr_cpu()
    for k in range(IN * OUT):
        if gw1_p[k] != gw2_p[k]:
            ok_bwd_gw = False
    var ok_bwd_gb = True
    var gb1_p = l1.grad_b.unsafe_ptr()
    var gb2_p = l2.bias.grad_unsafe_ptr_cpu()
    for k in range(OUT):
        if gb1_p[k] != gb2_p[k]:
            ok_bwd_gb = False
    print(
        "backward(all): grad_in=", "PASS" if ok_bwd_gi else "FAIL",
        " grad_w=",   "PASS" if ok_bwd_gw else "FAIL",
        " grad_b=",   "PASS" if ok_bwd_gb else "FAIL",
    )

    # ── Backward(mode='input_only') parity ───────────────────────────
    # Save current grad_w / grad_b state, run input_only, check
    # grad_input matches v1's backward_input AND param grads unchanged.
    var gw2_pre = List[Scalar[DT]]()
    var gb2_pre = List[Scalar[DT]]()
    for k in range(IN * OUT):
        gw2_pre.append(gw2_p[k])
    for k in range(OUT):
        gb2_pre.append(gb2_p[k])

    var gi1b_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var gi2b_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for k in range(BATCH * IN):
        gi1b_p[k] = Scalar[DT](0.0)
        gi2b_p[k] = Scalar[DT](0.0)
    var gi1b_t = TileTensor(gi1b_p, row_major[BATCH, IN]())
    var gi2b_t = TileTensor(gi2b_p, row_major[BATCH, IN]())
    l1.backward_input["cpu", BATCH](go_t, gi1b_t)
    l2.backward["cpu", BATCH, mode="input_only"](go_t, gi2b_t)

    var ok_bwi = True
    for k in range(BATCH * IN):
        if gi1b_p[k] != gi2b_p[k]:
            ok_bwi = False
    var ok_param_clean = True
    for k in range(IN * OUT):
        if gw2_p[k] != gw2_pre[k]:
            ok_param_clean = False
    for k in range(OUT):
        if gb2_p[k] != gb2_pre[k]:
            ok_param_clean = False
    print(
        "backward(input_only): grad_in=", "PASS" if ok_bwi else "FAIL",
        " params_unchanged=", "PASS" if ok_param_clean else "FAIL",
    )

    # ── zero_grad parity ─────────────────────────────────────────────
    l2.zero_grad[target="cpu"]()
    var ok_zg = True
    for k in range(IN * OUT):
        if gw2_p[k] != Scalar[DT](0.0):
            ok_zg = False
    for k in range(OUT):
        if gb2_p[k] != Scalar[DT](0.0):
            ok_zg = False
    print("zero_grad: PASS" if ok_zg else "zero_grad: FAIL")

    # ── Alias-safe backward — _cached_input_ptr aliasing grad_input ──
    # This is the case that motivates the flipped backward order.
    # If forward saved a ptr to `in_p`, and grad_input is written to
    # `in_p` (same buffer), then v1 would clobber the cache before
    # computing grad_w. v2 must produce correct grad_w.
    var l2b = LinearV2[IN, OUT].make[target="cpu", INIT=Zero]()
    var w2b_p = l2b.weight.value_unsafe_ptr_cpu()
    var b2b_p = l2b.bias.value_unsafe_ptr_cpu()
    for k in range(w_size):
        w2b_p[k] = w1_p[k]
    for k in range(b_size):
        b2b_p[k] = b1_p[k]

    var alias_in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var alias_out_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    _seed_buf(alias_in_p, BATCH * IN, UInt64(0xC0DEBABE))
    var alias_in_t  = TileTensor(alias_in_p, row_major[BATCH, IN]())
    var alias_out_t = TileTensor(alias_out_p, row_major[BATCH, OUT]())
    l2b.forward["cpu", BATCH](alias_in_t, alias_out_t)

    # Build reference: a fresh LinearV2 with same weights, separate
    # cache + grad_input buffers (no aliasing).
    var l2c = LinearV2[IN, OUT].make[target="cpu", INIT=Zero]()
    var w2c_p = l2c.weight.value_unsafe_ptr_cpu()
    var b2c_p = l2c.bias.value_unsafe_ptr_cpu()
    for k in range(w_size):
        w2c_p[k] = w1_p[k]
    for k in range(b_size):
        b2c_p[k] = b1_p[k]
    var ref_in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var ref_out_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for k in range(BATCH * IN):
        ref_in_p[k] = alias_in_p[k]  # same input values
    var ref_in_t  = TileTensor(ref_in_p, row_major[BATCH, IN]())
    var ref_out_t = TileTensor(ref_out_p, row_major[BATCH, OUT]())
    l2c.forward["cpu", BATCH](ref_in_t, ref_out_t)

    # Backward — aliasing version writes grad_input into alias_in_p
    # (same as cache); reference version writes into a fresh ref_gi_p.
    var alias_go_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    _seed_buf(alias_go_p, BATCH * OUT, UInt64(0xBAADF00D))
    var alias_go_t = TileTensor(alias_go_p, row_major[BATCH, OUT]())

    var ref_go_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var ref_gi_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for k in range(BATCH * OUT):
        ref_go_p[k] = alias_go_p[k]
    for k in range(BATCH * IN):
        ref_gi_p[k] = Scalar[DT](0.0)
    var ref_go_t = TileTensor(ref_go_p, row_major[BATCH, OUT]())
    var ref_gi_t = TileTensor(ref_gi_p, row_major[BATCH, IN]())

    l2b.backward["cpu", BATCH, mode="all"](alias_go_t, alias_in_t)  # alias!
    l2c.backward["cpu", BATCH, mode="all"](ref_go_t, ref_gi_t)

    # Aliased grad_input and reference grad_input must match.
    var ok_alias_gi = True
    for k in range(BATCH * IN):
        if alias_in_p[k] != ref_gi_p[k]:
            ok_alias_gi = False
    # Param grads must match — proves cache wasn't clobbered before grad_w read.
    var ok_alias_gw = True
    var gw2b_p = l2b.weight.grad_unsafe_ptr_cpu()
    var gw2c_p = l2c.weight.grad_unsafe_ptr_cpu()
    for k in range(IN * OUT):
        if gw2b_p[k] != gw2c_p[k]:
            ok_alias_gw = False
    print(
        "alias-safe backward: grad_in=", "PASS" if ok_alias_gi else "FAIL",
        " grad_w=", "PASS" if ok_alias_gw else "FAIL",
    )

    var all_ok = (
        ok_fwd and ok_bwd_gi and ok_bwd_gw and ok_bwd_gb
        and ok_bwi and ok_param_clean and ok_zg
        and ok_alias_gi and ok_alias_gw
    )
    if all_ok:
        print()
        print("PASS — LinearV2 is bit-identical to v1 Linear on CPU,")
        print("       backward(input_only) matches backward_input,")
        print("       and the flipped backward order is alias-safe.")
    else:
        raise Error("linear_v2 parity test failed")

    in_p.free()
    o1_p.free()
    o2_p.free()
    go_p.free()
    gi1_p.free()
    gi2_p.free()
    gi1b_p.free()
    gi2b_p.free()
    alias_in_p.free()
    alias_out_p.free()
    alias_go_p.free()
    ref_in_p.free()
    ref_out_p.free()
    ref_go_p.free()
    ref_gi_p.free()
