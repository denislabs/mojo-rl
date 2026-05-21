"""Parity test: AdamV2 + LinearV2 matches v1 Adam + v1 Linear after
multiple optimizer steps on CPU.

Bit-identical parameter check after 5 (forward → backward → step → zero_grad)
cycles. Demonstrates the slim trait: hyperparams set as public mut fields
after `make`, not as `make` args.
"""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.optimizer.adam_v2 import AdamV2
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
    # Build matched chains + matched optimizers.
    var l1 = Linear[IN, OUT].make[target="cpu", INIT=Kaiming]()
    var l2 = LinearV2[IN, OUT].make[target="cpu", INIT=Zero]()
    # Mirror v1's params into v2.
    var w1 = l1.weight.unsafe_ptr()
    var b1 = l1.bias.unsafe_ptr()
    var w2 = l2.weight.value_unsafe_ptr_cpu()
    var b2 = l2.bias.value_unsafe_ptr_cpu()
    for k in range(IN * OUT):
        w2[k] = w1[k]
    for k in range(OUT):
        b2[k] = b1[k]

    # v1 Adam takes hyperparams in make. v2 AdamV2 takes them as mut fields.
    var opt1 = Adam.make[target="cpu", M=Linear[IN, OUT]](
        l1, lr=Scalar[DT](3e-4), beta1=Scalar[DT](0.9),
        beta2=Scalar[DT](0.999), eps=Scalar[DT](1e-8),
    )
    var opt2 = AdamV2.make[target="cpu", M=LinearV2[IN, OUT]](l2)
    opt2.lr = Scalar[DT](3e-4)
    # beta1/beta2/eps already match defaults; explicit for clarity.
    opt2.beta1 = Scalar[DT](0.9)
    opt2.beta2 = Scalar[DT](0.999)
    opt2.eps = Scalar[DT](1e-8)

    # Allocate IO buffers once.
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
        # Fresh inputs / grad_outputs each step.
        _seed(in_p, BATCH * IN, UInt64(0xAA01) + UInt64(step))
        _seed(go, BATCH * OUT, UInt64(0xBB02) + UInt64(step))

        # Forward.
        l1.forward["cpu", BATCH](in_t, o1_t)
        l2.forward["cpu", BATCH](in_t, o2_t)

        # zero_grad before backward (v1 convention).
        opt1.zero_grad["cpu", M=Linear[IN, OUT]](l1)
        opt2.zero_grad["cpu", M=LinearV2[IN, OUT]](l2)

        # Backward.
        l1.backward["cpu", BATCH](go_t, gi1_t)
        l2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)

        # Step.
        opt1.step["cpu", M=Linear[IN, OUT]](l1)
        opt2.step["cpu", M=LinearV2[IN, OUT]](l2)

        # Compare params after this step.
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
        print("PASS — AdamV2 + LinearV2 match v1 Adam + v1 Linear bit-for-bit")
        print("       across", N_STEPS, "(forward → backward → step) cycles.")
        print("       Slim trait: hyperparams set via public mut fields after make.")
    else:
        raise Error("adam_v2 parity test failed")

    in_p.free()
    o1.free()
    o2.free()
    go.free()
    gi1.free()
    gi2.free()
