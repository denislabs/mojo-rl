"""StopGradParams reuses its grad stash across backwards — the capture-blocker gate.

`_GradStash` used to do `Tensor.alloc(N)` + `ensure_gpu` into a FRESH `List` on
every `vjp`, i.e. a device alloc + free per param per backward. Allocation
inside a CUDA-graph capture region is illegal, so this wrapper aborted the
BFM-Zero G1 capture on a `request=1KB` — exactly the B-net's `RMSNorm[256]`
gamma (docs/BFM_ZERO_G1_REPRODUCTION.md §12.8).

Checked over K backwards on the same module:
1. NO-ALLOC PRECONDITION — after the first backward every stash buffer already
   has `dev` set and `n >= N`, which is exactly the condition under which
   `Tensor.ensure_gpu` returns without allocating; and the buffer count, each
   `n`, and each DEVICE POINTER are unchanged on backwards 2..K. A reallocating
   stash cannot hold a stable pointer across a free.
2. FREEZE STILL HOLDS on every backward, not just the first — the `slot`
   bookkeeping that makes the reuse possible is what could silently mis-pair a
   saved buffer with a param, and a wrong pairing would restore the WRONG grad.
3. NON-VACUITY — the walk must visit at least two params (Linear = weight +
   bias, so slot advances past 0), and a standalone Linear given the same input
   must end with a NONZERO weight grad, so "frozen" is a real observation
   rather than an empty one.

Run: pixi run -e apple mojo run -I . tests/nn/test_stop_grad_params_reuse_gpu.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.stop_grad_params import StopGradParams


comptime D = 5
comptime O = 3
comptime B = 4
comptime WSZ = D * O
comptime K = 3  # backwards
comptime SG = StopGradParams[Linear[D, O]]


def main() raises:
    var c = DeviceContext()
    print("StopGradParams stash reuse across", K, "backwards")
    comptime TOL = Scalar[DT](1e-4)

    var sg = SG.make["gpu", Deterministic](Optional(c))
    var lin = Linear[D, O].make["gpu", Deterministic](Optional(c))
    sg.zero_grad["gpu"](Optional(c))
    lin.zero_grad["gpu"](Optional(c))

    var x = Tensor.alloc(B * D)
    for i in range(B * D):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    x.upload(c)

    var ptrs = List[Int]()
    var sizes = List[Int]()
    var nbuf = 0
    var ok = True

    for step in range(K):
        var go = Tensor.alloc(B * O)
        for i in range(B * O):
            go.data[i] = Scalar[DT](((i * 3 + step) % 7) - 3) * 0.25
        go.upload(c)
        var out = Tensor.alloc(B * O)
        var gi = Tensor.alloc(B * D)
        sg.forward["gpu", B](TensorRefs[1](x), out, Optional(c))
        sg.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(c))

        # ---- 2. freeze holds on EVERY backward ----------------------------
        sg.inner.weight.grd.download(c)
        var g: Scalar[DT] = 0.0
        for k in range(WSZ):
            g += abs(sg.inner.weight.grd.data[k])
        if g > TOL:
            ok = False
            print("    FREEZE BROKEN on backward", step, " sum|dW| =", g)

        # ---- 1. the no-alloc precondition --------------------------------
        if step == 0:
            nbuf = len(sg.stash.saved)
            for i in range(nbuf):
                if not sg.stash.saved[i].dev:
                    ok = False
                    print("    stash buffer", i, "has no device buffer")
                sizes.append(sg.stash.saved[i].n)
                ptrs.append(Int(sg.stash.saved[i].dev.value().unsafe_ptr()))
        else:
            if len(sg.stash.saved) != nbuf:
                ok = False
                print("    stash GREW on backward", step, ":", nbuf, "->", len(sg.stash.saved))
            for i in range(nbuf):
                if sg.stash.saved[i].n != sizes[i]:
                    ok = False
                    print("    stash buffer", i, "resized on backward", step)
                if Int(sg.stash.saved[i].dev.value().unsafe_ptr()) != ptrs[i]:
                    ok = False
                    print("    stash buffer", i, "REALLOCATED on backward", step)

    print("  stash buffers:", nbuf, " (stable pointers across backwards 2..", K, ")")

    # ---- 3. non-vacuity ---------------------------------------------------
    assert_true(
        nbuf >= 2,
        "vacuous: the stash walk visited < 2 params, so `slot` never advanced"
        " past 0 and the reuse bookkeeping is untested",
    )
    var go2 = Tensor.alloc(B * O)
    for i in range(B * O):
        go2.data[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.25
    go2.upload(c)
    var o_l = Tensor.alloc(B * O)
    var gi_l = Tensor.alloc(B * D)
    lin.forward["gpu", B](TensorRefs[1](x), o_l, Optional(c))
    lin.vjp["gpu", B](TensorRefs[1](x), go2, TensorRefs[1](gi_l), Optional(c))
    lin.weight.grd.download(c)
    var lin_g: Scalar[DT] = 0.0
    for k in range(WSZ):
        lin_g += abs(lin.weight.grd.data[k])
    print("  standalone Linear sum|dW| =", lin_g, "(must be nonzero)")
    assert_true(
        lin_g > Scalar[DT](1e-3),
        "vacuous: the unwrapped Linear's weight grad is ~0 too, so 'frozen'"
        " observes nothing",
    )

    assert_true(ok, "StopGradParams stash reuse / freeze")
    print("STOP_GRAD_PARAMS_REUSE OK")
