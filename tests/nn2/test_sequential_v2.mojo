"""Parity test: SequentialV2[LinearV2, ReLUV2, LinearV2, TanhV2, LinearV2]
matches v1 Sequential[Linear, ReLU, Linear, Tanh, Linear] bit-for-bit on CPU.

Exercises the orchestrator-owns-slabs design end-to-end with the
flipped-backward leaves (LinearV2). Critical correctness property:
the mid-slab between Sequential children IS the slab that LinearV2's
backward writes grad_input into AND that LinearV2's forward cached
via _cached_input_ptr. The flipped backward order must keep grad_w
correct under that aliasing — proven by Linear's standalone alias
test; this test confirms it composes through SequentialV2 too.
"""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.combinators.sequential_v2 import SequentialV2
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_v2 import LinearV2
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.relu_v2 import ReLUV2
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.primitives.tanh_v2 import TanhV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero, Kaiming


comptime BATCH = 4
comptime D0 = 6
comptime D1 = 8
comptime D2 = 5
comptime D3 = 3
comptime D4 = 4  # final OUT


def _seed(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, salt: UInt64):
    var state: UInt64 = salt
    for k in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        p[k] = (r - Scalar[DT](0.5))


def main() raises:
    # ── Build matched chains: Linear → ReLU → Linear → Tanh → Linear ─
    # Use INIT=Kaiming on v1 so Linear weights are non-trivial, then
    # mirror them into v2.
    comptime V1 = Sequential[
        Linear[D0, D1], ReLU[D1],
        Linear[D1, D2], Tanh[D2],
        Linear[D2, D3],
    ]
    comptime V2 = SequentialV2[
        LinearV2[D0, D1], ReLUV2[D1],
        LinearV2[D1, D2], TanhV2[D2],
        LinearV2[D2, D3],
    ]

    var s1 = V1.make[target="cpu", INIT=Kaiming]()
    var s2 = V2.make[target="cpu", INIT=Zero]()

    # Mirror Linear weights from s1 → s2.
    # s1.children[0] : Linear[D0, D1]
    # s2.children[0] : LinearV2[D0, D1]
    var w1_0 = s1.children[0].weight.unsafe_ptr()
    var b1_0 = s1.children[0].bias.unsafe_ptr()
    var w2_0 = s2.children[0].weight.value_unsafe_ptr_cpu()
    var b2_0 = s2.children[0].bias.value_unsafe_ptr_cpu()
    for k in range(D0 * D1):
        w2_0[k] = w1_0[k]
    for k in range(D1):
        b2_0[k] = b1_0[k]
    # children[2] : Linear[D1, D2]
    var w1_2 = s1.children[2].weight.unsafe_ptr()
    var b1_2 = s1.children[2].bias.unsafe_ptr()
    var w2_2 = s2.children[2].weight.value_unsafe_ptr_cpu()
    var b2_2 = s2.children[2].bias.value_unsafe_ptr_cpu()
    for k in range(D1 * D2):
        w2_2[k] = w1_2[k]
    for k in range(D2):
        b2_2[k] = b1_2[k]
    # children[4] : Linear[D2, D3]
    var w1_4 = s1.children[4].weight.unsafe_ptr()
    var b1_4 = s1.children[4].bias.unsafe_ptr()
    var w2_4 = s2.children[4].weight.value_unsafe_ptr_cpu()
    var b2_4 = s2.children[4].bias.value_unsafe_ptr_cpu()
    for k in range(D2 * D3):
        w2_4[k] = w1_4[k]
    for k in range(D3):
        b2_4[k] = b1_4[k]

    # ── Forward parity ───────────────────────────────────────────────
    var in_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D0)
    var o1_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D3)
    var o2_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D3)
    _seed(in_p, BATCH * D0, UInt64(0xFADEC0DE))

    var in_t = TileTensor(in_p, row_major[BATCH, D0]())
    var o1_t = TileTensor(o1_p, row_major[BATCH, D3]())
    var o2_t = TileTensor(o2_p, row_major[BATCH, D3]())
    s1.forward["cpu", BATCH](in_t, o1_t)
    s2.forward["cpu", BATCH](in_t, o2_t)

    var ok_fwd = True
    for k in range(BATCH * D3):
        if o1_p[k] != o2_p[k]:
            ok_fwd = False
    print("forward: PASS" if ok_fwd else "forward: FAIL")

    # ── Backward(mode='all') parity ──────────────────────────────────
    var go_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D3)
    var gi1_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D0)
    var gi2_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D0)
    _seed(go_p, BATCH * D3, UInt64(0xC0FFEE))
    for k in range(BATCH * D0):
        gi1_p[k] = Scalar[DT](0.0)
        gi2_p[k] = Scalar[DT](0.0)
    var go_t  = TileTensor(go_p,  row_major[BATCH, D3]())
    var gi1_t = TileTensor(gi1_p, row_major[BATCH, D0]())
    var gi2_t = TileTensor(gi2_p, row_major[BATCH, D0]())
    s1.backward["cpu", BATCH](go_t, gi1_t)
    s2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)

    var ok_gi = True
    for k in range(BATCH * D0):
        if gi1_p[k] != gi2_p[k]:
            ok_gi = False

    # Compare every Linear's param grads.
    var gw1_0 = s1.children[0].grad_w.unsafe_ptr()
    var gw2_0 = s2.children[0].weight.grad_unsafe_ptr_cpu()
    var ok_gw0 = True
    for k in range(D0 * D1):
        if gw1_0[k] != gw2_0[k]:
            ok_gw0 = False
    var gw1_2 = s1.children[2].grad_w.unsafe_ptr()
    var gw2_2 = s2.children[2].weight.grad_unsafe_ptr_cpu()
    var ok_gw2 = True
    for k in range(D1 * D2):
        if gw1_2[k] != gw2_2[k]:
            ok_gw2 = False
    var gw1_4 = s1.children[4].grad_w.unsafe_ptr()
    var gw2_4 = s2.children[4].weight.grad_unsafe_ptr_cpu()
    var ok_gw4 = True
    for k in range(D2 * D3):
        if gw1_4[k] != gw2_4[k]:
            ok_gw4 = False

    var gb1_0 = s1.children[0].grad_b.unsafe_ptr()
    var gb2_0 = s2.children[0].bias.grad_unsafe_ptr_cpu()
    var ok_gb0 = True
    for k in range(D1):
        if gb1_0[k] != gb2_0[k]:
            ok_gb0 = False

    print(
        "backward(all): grad_in=", "PASS" if ok_gi else "FAIL",
        " grad_w[Linear0]=", "PASS" if ok_gw0 else "FAIL",
        " grad_w[Linear2]=", "PASS" if ok_gw2 else "FAIL",
        " grad_w[Linear4]=", "PASS" if ok_gw4 else "FAIL",
        " grad_b[Linear0]=", "PASS" if ok_gb0 else "FAIL",
    )

    # ── Backward(mode='input_only') parity ───────────────────────────
    # CAVEAT: the unified-buffer v2 design DISCARDS the forward cache
    # during backward — `_cached_input_ptr` aliases the orchestrator
    # mid-slab, which gets overwritten by grad_input. A second backward
    # without re-forwarding would read stale data. Re-forward both
    # chains here so caches are fresh; v1 doesn't strictly need this
    # (it copies its caches) but it's harmless.
    s1.forward["cpu", BATCH](in_t, o1_t)
    s2.forward["cpu", BATCH](in_t, o2_t)

    var gw_pre_0 = List[Scalar[DT]]()
    for k in range(D0 * D1):
        gw_pre_0.append(gw2_0[k])

    var gi1b_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D0)
    var gi2b_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D0)
    for k in range(BATCH * D0):
        gi1b_p[k] = Scalar[DT](0.0)
        gi2b_p[k] = Scalar[DT](0.0)
    var gi1b_t = TileTensor(gi1b_p, row_major[BATCH, D0]())
    var gi2b_t = TileTensor(gi2b_p, row_major[BATCH, D0]())
    s1.backward_input["cpu", BATCH](go_t, gi1b_t)
    s2.backward["cpu", BATCH, mode="input_only"](go_t, gi2b_t)

    var ok_bwi = True
    for k in range(BATCH * D0):
        if gi1b_p[k] != gi2b_p[k]:
            ok_bwi = False
    var ok_clean = True
    for k in range(D0 * D1):
        if gw2_0[k] != gw_pre_0[k]:
            ok_clean = False
    print(
        "backward(input_only): grad_in=", "PASS" if ok_bwi else "FAIL",
        " params_unchanged[Linear0.weight]=", "PASS" if ok_clean else "FAIL",
    )

    # ── zero_grad recurses through children ──────────────────────────
    s2.zero_grad[target="cpu"]()
    var ok_zg = True
    for k in range(D0 * D1):
        if gw2_0[k] != Scalar[DT](0.0):
            ok_zg = False
    for k in range(D1 * D2):
        if gw2_2[k] != Scalar[DT](0.0):
            ok_zg = False
    for k in range(D2 * D3):
        if gw2_4[k] != Scalar[DT](0.0):
            ok_zg = False
    print("zero_grad(recursive): PASS" if ok_zg else "zero_grad(recursive): FAIL")

    var all_ok = (
        ok_fwd and ok_gi and ok_gw0 and ok_gw2 and ok_gw4 and ok_gb0
        and ok_bwi and ok_clean and ok_zg
    )
    if all_ok:
        print()
        print("PASS — SequentialV2 5-deep chain is bit-identical to v1")
        print("       on CPU. backward(input_only) collapses backward_input.")
        print("       zero_grad recurses through every child.")
    else:
        raise Error("sequential_v2 parity test failed")

    in_p.free()
    o1_p.free()
    o2_p.free()
    go_p.free()
    gi1_p.free()
    gi2_p.free()
    gi1b_p.free()
    gi2b_p.free()
