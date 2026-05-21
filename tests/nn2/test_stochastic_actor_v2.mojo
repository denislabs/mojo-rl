"""Parity test: StochasticActorV2 matches v1 StochasticActor on CPU."""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.primitives.stochastic_actor_v2 import StochasticActorV2
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_v2 import LinearV2
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.relu_v2 import ReLUV2
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.primitives.tanh_v2 import TanhV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero, Kaiming


comptime BATCH = 4
comptime OBS = 5
comptime HIDDEN = 7
comptime ACT = 2


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


def main() raises:
    # TRUNK: Linear[OBS, HIDDEN] → Tanh → Linear[HIDDEN, HIDDEN] → ReLU
    # HEADS: Parallel[Linear[HIDDEN, ACT], Linear[HIDDEN, ACT]]
    comptime V1 = StochasticActor[
        OBS, ACT,
        Linear[OBS, HIDDEN], Tanh[HIDDEN],
        Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    ]
    comptime V2 = StochasticActorV2[
        OBS, ACT,
        LinearV2[OBS, HIDDEN], TanhV2[HIDDEN],
        LinearV2[HIDDEN, HIDDEN], ReLUV2[HIDDEN],
    ]

    var a1 = V1.make[target="cpu", INIT=Kaiming]()
    var a2 = V2.make[target="cpu", INIT=Zero]()

    # Mirror trunk Linear params: TRUNK[0] (Linear[OBS, HIDDEN]), TRUNK[2]
    var w1_t0 = a1.trunk.children[0].weight.unsafe_ptr()
    var b1_t0 = a1.trunk.children[0].bias.unsafe_ptr()
    var w2_t0 = a2.trunk.children[0].weight.value_unsafe_ptr_cpu()
    var b2_t0 = a2.trunk.children[0].bias.value_unsafe_ptr_cpu()
    for k in range(OBS * HIDDEN):
        w2_t0[k] = w1_t0[k]
    for k in range(HIDDEN):
        b2_t0[k] = b1_t0[k]
    var w1_t2 = a1.trunk.children[2].weight.unsafe_ptr()
    var b1_t2 = a1.trunk.children[2].bias.unsafe_ptr()
    var w2_t2 = a2.trunk.children[2].weight.value_unsafe_ptr_cpu()
    var b2_t2 = a2.trunk.children[2].bias.value_unsafe_ptr_cpu()
    for k in range(HIDDEN * HIDDEN):
        w2_t2[k] = w1_t2[k]
    for k in range(HIDDEN):
        b2_t2[k] = b1_t2[k]
    # Mirror head Linear params.
    var w1_ha = a1.heads.branch_a.weight.unsafe_ptr()
    var b1_ha = a1.heads.branch_a.bias.unsafe_ptr()
    var w2_ha = a2.heads.branch_a.weight.value_unsafe_ptr_cpu()
    var b2_ha = a2.heads.branch_a.bias.value_unsafe_ptr_cpu()
    for k in range(HIDDEN * ACT):
        w2_ha[k] = w1_ha[k]
    for k in range(ACT):
        b2_ha[k] = b1_ha[k]
    var w1_hb = a1.heads.branch_b.weight.unsafe_ptr()
    var b1_hb = a1.heads.branch_b.bias.unsafe_ptr()
    var w2_hb = a2.heads.branch_b.weight.value_unsafe_ptr_cpu()
    var b2_hb = a2.heads.branch_b.bias.value_unsafe_ptr_cpu()
    for k in range(HIDDEN * ACT):
        w2_hb[k] = w1_hb[k]
    for k in range(ACT):
        b2_hb[k] = b1_hb[k]

    # ── Forward ──────────────────────────────────────────────────────
    var in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)
    var o1:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var o2:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    _seed(in_p, BATCH * OBS, UInt64(0xACADEDE))
    var in_t = TileTensor(in_p, row_major[BATCH, OBS]())
    var o1_t = TileTensor(o1, row_major[BATCH, 2 * ACT]())
    var o2_t = TileTensor(o2, row_major[BATCH, 2 * ACT]())
    a1.forward["cpu", BATCH](in_t, o1_t)
    a2.forward["cpu", BATCH](in_t, o2_t)
    var ok_fwd = _eq(o1, o2, BATCH * 2 * ACT)
    print("forward: PASS" if ok_fwd else "forward: FAIL")

    # ── Backward(mode='all') ─────────────────────────────────────────
    var go:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var gi1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)
    var gi2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)
    _seed(go, BATCH * 2 * ACT, UInt64(0xC0DAC0DA))
    for k in range(BATCH * OBS):
        gi1[k] = Scalar[DT](0.0)
        gi2[k] = Scalar[DT](0.0)
    var go_t = TileTensor(go, row_major[BATCH, 2 * ACT]())
    var gi1_t = TileTensor(gi1, row_major[BATCH, OBS]())
    var gi2_t = TileTensor(gi2, row_major[BATCH, OBS]())
    a1.backward["cpu", BATCH](go_t, gi1_t)
    a2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)
    var ok_gi = _eq(gi1, gi2, BATCH * OBS)

    var gw1_t0 = a1.trunk.children[0].grad_w.unsafe_ptr()
    var gw2_t0 = a2.trunk.children[0].weight.grad_unsafe_ptr_cpu()
    var ok_gw_t0 = _eq(gw1_t0, gw2_t0, OBS * HIDDEN)
    var gw1_ha = a1.heads.branch_a.grad_w.unsafe_ptr()
    var gw2_ha = a2.heads.branch_a.weight.grad_unsafe_ptr_cpu()
    var ok_gw_ha = _eq(gw1_ha, gw2_ha, HIDDEN * ACT)
    var gw1_hb = a1.heads.branch_b.grad_w.unsafe_ptr()
    var gw2_hb = a2.heads.branch_b.weight.grad_unsafe_ptr_cpu()
    var ok_gw_hb = _eq(gw1_hb, gw2_hb, HIDDEN * ACT)

    print(
        "backward(all): grad_in=", "PASS" if ok_gi else "FAIL",
        " grad_w[trunk.0]=", "PASS" if ok_gw_t0 else "FAIL",
        " grad_w[heads.a]=", "PASS" if ok_gw_ha else "FAIL",
        " grad_w[heads.b]=", "PASS" if ok_gw_hb else "FAIL",
    )

    # ── zero_grad recurses ───────────────────────────────────────────
    a2.zero_grad[target="cpu"]()
    var ok_zg = True
    for k in range(OBS * HIDDEN):
        if gw2_t0[k] != Scalar[DT](0.0):
            ok_zg = False
    for k in range(HIDDEN * ACT):
        if gw2_ha[k] != Scalar[DT](0.0):
            ok_zg = False
        if gw2_hb[k] != Scalar[DT](0.0):
            ok_zg = False
    print("zero_grad(recursive): PASS" if ok_zg else "zero_grad(recursive): FAIL")

    if ok_fwd and ok_gi and ok_gw_t0 and ok_gw_ha and ok_gw_hb and ok_zg:
        print()
        print("PASS — StochasticActorV2 matches v1 bit-for-bit on CPU.")
        print("       trunk + heads compose through SequentialV2 + ParallelV2 +")
        print("       LinearV2 + TanhV2 + ReLUV2 correctly.")
    else:
        raise Error("stochastic_actor_v2 parity test failed")

    in_p.free(); o1.free(); o2.free()
    go.free(); gi1.free(); gi2.free()
