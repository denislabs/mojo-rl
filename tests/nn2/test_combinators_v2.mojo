"""Parity tests: ResidualV2 + ParallelV2 + ConcatV2 vs v1 on CPU."""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.combinators.residual import Residual
from mojo_rl.nn2.combinators.residual_v2 import ResidualV2
from mojo_rl.nn2.combinators.parallel import Parallel
from mojo_rl.nn2.combinators.parallel_v2 import ParallelV2
from mojo_rl.nn2.combinators.concat import Concat
from mojo_rl.nn2.combinators.concat_v2 import ConcatV2
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.combinators.sequential_v2 import SequentialV2
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_v2 import LinearV2
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.relu_v2 import ReLUV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero, Kaiming


comptime BATCH = 4


def _seed(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, salt: UInt64):
    var state: UInt64 = salt
    for k in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        p[k] = (r - Scalar[DT](0.5))


def _eq(a: UnsafePointer[Scalar[DT], MutAnyOrigin],
        b: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) raises -> Bool:
    for k in range(n):
        if a[k] != b[k]:
            return False
    return True


# ──────────────────────────────────────────────────────────────────────
# ResidualV2 — Inner = Linear[D, D] with ReLU between (Sequential)
# ──────────────────────────────────────────────────────────────────────


comptime RD = 6


def _check_residual() raises -> Bool:
    comptime V1 = Residual[Sequential[Linear[RD, RD], ReLU[RD]]]
    comptime V2 = ResidualV2[SequentialV2[LinearV2[RD, RD], ReLUV2[RD]]]
    var r1 = V1.make[target="cpu", INIT=Kaiming]()
    var r2 = V2.make[target="cpu", INIT=Zero]()

    # Mirror Linear params.
    var w1 = r1.inner.children[0].weight.unsafe_ptr()
    var b1 = r1.inner.children[0].bias.unsafe_ptr()
    var w2 = r2.inner.children[0].weight.value_unsafe_ptr_cpu()
    var b2 = r2.inner.children[0].bias.value_unsafe_ptr_cpu()
    for k in range(RD * RD):
        w2[k] = w1[k]
    for k in range(RD):
        b2[k] = b1[k]

    var in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * RD)
    var o1:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * RD)
    var o2:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * RD)
    _seed(in_p, BATCH * RD, UInt64(0xAAAA0001))
    var in_t = TileTensor(in_p, row_major[BATCH, RD]())
    var o1_t = TileTensor(o1, row_major[BATCH, RD]())
    var o2_t = TileTensor(o2, row_major[BATCH, RD]())
    r1.forward["cpu", BATCH](in_t, o1_t)
    r2.forward["cpu", BATCH](in_t, o2_t)
    var ok_fwd = _eq(o1, o2, BATCH * RD)

    var go:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * RD)
    var gi1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * RD)
    var gi2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * RD)
    _seed(go, BATCH * RD, UInt64(0xAAAA0002))
    for k in range(BATCH * RD):
        gi1[k] = Scalar[DT](0.0)
        gi2[k] = Scalar[DT](0.0)
    var go_t = TileTensor(go, row_major[BATCH, RD]())
    var gi1_t = TileTensor(gi1, row_major[BATCH, RD]())
    var gi2_t = TileTensor(gi2, row_major[BATCH, RD]())
    r1.backward["cpu", BATCH](go_t, gi1_t)
    r2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)
    var ok_bwd_gi = _eq(gi1, gi2, BATCH * RD)

    # Inner Linear grad_w / grad_b should match.
    var gw1 = r1.inner.children[0].grad_w.unsafe_ptr()
    var gw2 = r2.inner.children[0].weight.grad_unsafe_ptr_cpu()
    var ok_gw = _eq(gw1, gw2, RD * RD)
    var gb1 = r1.inner.children[0].grad_b.unsafe_ptr()
    var gb2 = r2.inner.children[0].bias.grad_unsafe_ptr_cpu()
    var ok_gb = _eq(gb1, gb2, RD)

    print(
        "Residual: fwd=", "PASS" if ok_fwd else "FAIL",
        " bwd.gi=", "PASS" if ok_bwd_gi else "FAIL",
        " bwd.gw=", "PASS" if ok_gw else "FAIL",
        " bwd.gb=", "PASS" if ok_gb else "FAIL",
    )

    in_p.free(); o1.free(); o2.free()
    go.free(); gi1.free(); gi2.free()
    return ok_fwd and ok_bwd_gi and ok_gw and ok_gb


# ──────────────────────────────────────────────────────────────────────
# ParallelV2 — Linear[IN, OUT_A], Linear[IN, OUT_B]
# ──────────────────────────────────────────────────────────────────────


comptime PIN = 5
comptime POUT_A = 3
comptime POUT_B = 2
comptime POUT = POUT_A + POUT_B


def _check_parallel() raises -> Bool:
    comptime V1 = Parallel[Linear[PIN, POUT_A], Linear[PIN, POUT_B]]
    comptime V2 = ParallelV2[LinearV2[PIN, POUT_A], LinearV2[PIN, POUT_B]]
    var p1 = V1.make[target="cpu", INIT=Kaiming]()
    var p2 = V2.make[target="cpu", INIT=Zero]()

    # Mirror branch params.
    var w1a = p1.branch_a.weight.unsafe_ptr()
    var b1a = p1.branch_a.bias.unsafe_ptr()
    var w2a = p2.branch_a.weight.value_unsafe_ptr_cpu()
    var b2a = p2.branch_a.bias.value_unsafe_ptr_cpu()
    for k in range(PIN * POUT_A):
        w2a[k] = w1a[k]
    for k in range(POUT_A):
        b2a[k] = b1a[k]
    var w1b = p1.branch_b.weight.unsafe_ptr()
    var b1b = p1.branch_b.bias.unsafe_ptr()
    var w2b = p2.branch_b.weight.value_unsafe_ptr_cpu()
    var b2b = p2.branch_b.bias.value_unsafe_ptr_cpu()
    for k in range(PIN * POUT_B):
        w2b[k] = w1b[k]
    for k in range(POUT_B):
        b2b[k] = b1b[k]

    var in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * PIN)
    var o1:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * POUT)
    var o2:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * POUT)
    _seed(in_p, BATCH * PIN, UInt64(0xBBBB0001))
    var in_t = TileTensor(in_p, row_major[BATCH, PIN]())
    var o1_t = TileTensor(o1, row_major[BATCH, POUT]())
    var o2_t = TileTensor(o2, row_major[BATCH, POUT]())
    p1.forward["cpu", BATCH](in_t, o1_t)
    p2.forward["cpu", BATCH](in_t, o2_t)
    var ok_fwd = _eq(o1, o2, BATCH * POUT)

    var go:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * POUT)
    var gi1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * PIN)
    var gi2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * PIN)
    _seed(go, BATCH * POUT, UInt64(0xBBBB0002))
    for k in range(BATCH * PIN):
        gi1[k] = Scalar[DT](0.0)
        gi2[k] = Scalar[DT](0.0)
    var go_t = TileTensor(go, row_major[BATCH, POUT]())
    var gi1_t = TileTensor(gi1, row_major[BATCH, PIN]())
    var gi2_t = TileTensor(gi2, row_major[BATCH, PIN]())
    p1.backward["cpu", BATCH](go_t, gi1_t)
    p2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)
    var ok_bwd_gi = _eq(gi1, gi2, BATCH * PIN)

    var gw1a = p1.branch_a.grad_w.unsafe_ptr()
    var gw2a = p2.branch_a.weight.grad_unsafe_ptr_cpu()
    var ok_gwa = _eq(gw1a, gw2a, PIN * POUT_A)
    var gw1b = p1.branch_b.grad_w.unsafe_ptr()
    var gw2b = p2.branch_b.weight.grad_unsafe_ptr_cpu()
    var ok_gwb = _eq(gw1b, gw2b, PIN * POUT_B)

    print(
        "Parallel: fwd=", "PASS" if ok_fwd else "FAIL",
        " bwd.gi=", "PASS" if ok_bwd_gi else "FAIL",
        " bwd.gw[A]=", "PASS" if ok_gwa else "FAIL",
        " bwd.gw[B]=", "PASS" if ok_gwb else "FAIL",
    )

    in_p.free(); o1.free(); o2.free()
    go.free(); gi1.free(); gi2.free()
    return ok_fwd and ok_bwd_gi and ok_gwa and ok_gwb


# ──────────────────────────────────────────────────────────────────────
# ConcatV2 — 3 branches Linear[IN, ?]
# ──────────────────────────────────────────────────────────────────────


comptime CIN = 4
comptime COUT_0 = 2
comptime COUT_1 = 3
comptime COUT_2 = 2
comptime COUT = COUT_0 + COUT_1 + COUT_2


def _check_concat() raises -> Bool:
    comptime V1 = Concat[
        Linear[CIN, COUT_0], Linear[CIN, COUT_1], Linear[CIN, COUT_2],
    ]
    comptime V2 = ConcatV2[
        LinearV2[CIN, COUT_0], LinearV2[CIN, COUT_1], LinearV2[CIN, COUT_2],
    ]
    var c1 = V1.make[target="cpu", INIT=Kaiming]()
    var c2 = V2.make[target="cpu", INIT=Zero]()

    # Mirror Linear params, branch 0..2
    var w1_0 = c1.branches[0].weight.unsafe_ptr()
    var b1_0 = c1.branches[0].bias.unsafe_ptr()
    var w2_0 = c2.branches[0].weight.value_unsafe_ptr_cpu()
    var b2_0 = c2.branches[0].bias.value_unsafe_ptr_cpu()
    for k in range(CIN * COUT_0):
        w2_0[k] = w1_0[k]
    for k in range(COUT_0):
        b2_0[k] = b1_0[k]
    var w1_1 = c1.branches[1].weight.unsafe_ptr()
    var b1_1 = c1.branches[1].bias.unsafe_ptr()
    var w2_1 = c2.branches[1].weight.value_unsafe_ptr_cpu()
    var b2_1 = c2.branches[1].bias.value_unsafe_ptr_cpu()
    for k in range(CIN * COUT_1):
        w2_1[k] = w1_1[k]
    for k in range(COUT_1):
        b2_1[k] = b1_1[k]
    var w1_2 = c1.branches[2].weight.unsafe_ptr()
    var b1_2 = c1.branches[2].bias.unsafe_ptr()
    var w2_2 = c2.branches[2].weight.value_unsafe_ptr_cpu()
    var b2_2 = c2.branches[2].bias.value_unsafe_ptr_cpu()
    for k in range(CIN * COUT_2):
        w2_2[k] = w1_2[k]
    for k in range(COUT_2):
        b2_2[k] = b1_2[k]

    var in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * CIN)
    var o1:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * COUT)
    var o2:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * COUT)
    _seed(in_p, BATCH * CIN, UInt64(0xCCCC0001))
    var in_t = TileTensor(in_p, row_major[BATCH, CIN]())
    var o1_t = TileTensor(o1, row_major[BATCH, COUT]())
    var o2_t = TileTensor(o2, row_major[BATCH, COUT]())
    c1.forward["cpu", BATCH](in_t, o1_t)
    c2.forward["cpu", BATCH](in_t, o2_t)
    var ok_fwd = _eq(o1, o2, BATCH * COUT)

    var go:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * COUT)
    var gi1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * CIN)
    var gi2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * CIN)
    _seed(go, BATCH * COUT, UInt64(0xCCCC0002))
    for k in range(BATCH * CIN):
        gi1[k] = Scalar[DT](0.0)
        gi2[k] = Scalar[DT](0.0)
    var go_t = TileTensor(go, row_major[BATCH, COUT]())
    var gi1_t = TileTensor(gi1, row_major[BATCH, CIN]())
    var gi2_t = TileTensor(gi2, row_major[BATCH, CIN]())
    c1.backward["cpu", BATCH](go_t, gi1_t)
    c2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)
    var ok_bwd_gi = _eq(gi1, gi2, BATCH * CIN)

    var gw1_0 = c1.branches[0].grad_w.unsafe_ptr()
    var gw2_0 = c2.branches[0].weight.grad_unsafe_ptr_cpu()
    var ok_gw0 = _eq(gw1_0, gw2_0, CIN * COUT_0)
    var gw1_1 = c1.branches[1].grad_w.unsafe_ptr()
    var gw2_1 = c2.branches[1].weight.grad_unsafe_ptr_cpu()
    var ok_gw1 = _eq(gw1_1, gw2_1, CIN * COUT_1)
    var gw1_2 = c1.branches[2].grad_w.unsafe_ptr()
    var gw2_2 = c2.branches[2].weight.grad_unsafe_ptr_cpu()
    var ok_gw2 = _eq(gw1_2, gw2_2, CIN * COUT_2)

    print(
        "Concat: fwd=", "PASS" if ok_fwd else "FAIL",
        " bwd.gi=", "PASS" if ok_bwd_gi else "FAIL",
        " bwd.gw[0]=", "PASS" if ok_gw0 else "FAIL",
        " bwd.gw[1]=", "PASS" if ok_gw1 else "FAIL",
        " bwd.gw[2]=", "PASS" if ok_gw2 else "FAIL",
    )

    in_p.free(); o1.free(); o2.free()
    go.free(); gi1.free(); gi2.free()
    return ok_fwd and ok_bwd_gi and ok_gw0 and ok_gw1 and ok_gw2


def main() raises:
    var ok = True
    ok = ok and _check_residual()
    ok = ok and _check_parallel()
    ok = ok and _check_concat()
    if ok:
        print()
        print("PASS — ResidualV2 + ParallelV2 + ConcatV2 are bit-identical")
        print("       to v1 on CPU. All inner-leaf param grads match.")
    else:
        raise Error("combinators_v2 parity test failed")
