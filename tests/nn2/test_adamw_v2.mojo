"""Parity test: AdamWV2 + LinearV2 matches v1 AdamW + v1 Linear after
multiple optimizer steps on CPU. Exercises decoupled weight decay —
Linear's weight has decay=True, bias has decay=False, so the test
catches any mismatch in the per-param decay flag handling.
"""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.optimizer.adamw import AdamW
from mojo_rl.nn2.optimizer.adamw_v2 import AdamWV2
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_v2 import LinearV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero, Kaiming


comptime BATCH = 4
comptime IN = 6
comptime OUT = 5
comptime N_STEPS = 5


def _seed(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, salt: UInt64):
    var state: UInt64 = salt
    for k in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        p[k] = (r - Scalar[DT](0.5))


def main() raises:
    var l1 = Linear[IN, OUT].make[target="cpu", INIT=Kaiming]()
    var l2 = LinearV2[IN, OUT].make[target="cpu", INIT=Zero]()
    var w1 = l1.weight.unsafe_ptr()
    var b1 = l1.bias.unsafe_ptr()
    var w2 = l2.weight.value_unsafe_ptr_cpu()
    var b2 = l2.bias.value_unsafe_ptr_cpu()
    for k in range(IN * OUT):
        w2[k] = w1[k]
    for k in range(OUT):
        b2[k] = b1[k]

    # v1 AdamW: weight_decay defaults to 0.01 from `make`; can be overridden
    # via `make_with_wd`. Here we use the default to keep things simple.
    var opt1 = AdamW.make[target="cpu", M=Linear[IN, OUT]](
        l1, lr=Scalar[DT](3e-4), beta1=Scalar[DT](0.9),
        beta2=Scalar[DT](0.999), eps=Scalar[DT](1e-8),
    )
    # v2 AdamWV2: hyperparams as mut fields.
    var opt2 = AdamWV2.make[target="cpu", M=LinearV2[IN, OUT]](l2)
    opt2.lr = Scalar[DT](3e-4)
    opt2.weight_decay = Scalar[DT](0.01)  # match v1 default

    var in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var o1:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var o2:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi1:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var gi2:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)

    var in_t = TileTensor(in_p, row_major[BATCH, IN]())
    var o1_t = TileTensor(o1, row_major[BATCH, OUT]())
    var o2_t = TileTensor(o2, row_major[BATCH, OUT]())
    var go_t = TileTensor(go, row_major[BATCH, OUT]())
    var gi1_t = TileTensor(gi1, row_major[BATCH, IN]())
    var gi2_t = TileTensor(gi2, row_major[BATCH, IN]())

    var ok = True
    for step in range(N_STEPS):
        _seed(in_p, BATCH * IN, UInt64(0xCC01) + UInt64(step))
        _seed(go, BATCH * OUT, UInt64(0xDD02) + UInt64(step))

        l1.forward["cpu", BATCH](in_t, o1_t)
        l2.forward["cpu", BATCH](in_t, o2_t)

        opt1.zero_grad["cpu", M=Linear[IN, OUT]](l1)
        opt2.zero_grad["cpu", M=LinearV2[IN, OUT]](l2)

        l1.backward["cpu", BATCH](go_t, gi1_t)
        l2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)

        opt1.step["cpu", M=Linear[IN, OUT]](l1)
        opt2.step["cpu", M=LinearV2[IN, OUT]](l2)

        for k in range(IN * OUT):
            if w1[k] != w2[k]:
                ok = False
        for k in range(OUT):
            if b1[k] != b2[k]:
                ok = False
        print("step", step, ":", "PASS" if ok else "FAIL")
        if not ok:
            break

    if ok:
        print()
        print("PASS — AdamWV2 + LinearV2 match v1 AdamW + v1 Linear bit-for-bit")
        print("       across", N_STEPS, "cycles. Per-param decay flag handled")
        print("       correctly (weight=True, bias=False on Linear).")
    else:
        raise Error("adamw_v2 parity test failed")

    in_p.free()
    o1.free()
    o2.free()
    go.free()
    gi1.free()
    gi2.free()
