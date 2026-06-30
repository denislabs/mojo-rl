"""State checkpoint round-trip (storage surface, CPU + GPU).

BatchNorm running stats are now a `State` role, walked by `for_each_state` and
persisted by save_params/load_params. This gates that:
  1. a training-mode forward evolves the running stats away from their defaults
     (running_mean 0, running_var 1), and
  2. save_params → fresh model → load_params restores those exact running stats
     (which a state-less checkpoint would have dropped).

The BatchNorm sits INSIDE a Sequential, so this also exercises the combinator
`for_each_state` recursion.

Run: pixi run -e apple mojo run -I . tests/nn/test_state_checkpoint_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.checkpoint import save_params, load_params
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.batch_norm_1d import BatchNorm1D
from mojo_rl.nn.combinators.sequential import Sequential


comptime D = 4
comptime H = 5
comptime O = 3
comptime B = 6
comptime NET = Sequential[Linear[D, H], BatchNorm1D[H], Linear[H, O]]


def _check[target: StaticString](
    ctx: Optional[DeviceContext], path: String
) raises -> Bool:
    comptime TOL = Scalar[DT](1e-5)
    var a = NET.make[target, Deterministic](ctx)

    # Evolve running stats: a few training-mode forwards with varied input.
    var out = Tensor.alloc(B * O)
    for it in range(3):
        var x = Tensor.alloc(B * D)
        for i in range(B * D):
            x.data[i] = Scalar[DT]((((i + it) % 7) - 3)) * 0.3
        comptime if target == "gpu":
            x.upload(ctx.value())
        a.forward[target, B](TensorRefs[1](x), out, ctx)

    save_params[target](a, path, ctx)

    # Fresh model — running stats at defaults (mean 0, var 1) until loaded.
    var b = NET.make[target, Deterministic](ctx)
    load_params[target](b, path, ctx)

    # After save, a's running stats are resident in .t.data (CheckpointWriter
    # downloaded on GPU); after load, b's are too (CheckpointReader wrote .t.data).
    var moved = False  # confirm stats actually evolved (test is meaningful)
    var matched = True
    for k in range(H):
        var am = a.children[1].running_mean.t.data[k]
        var bm = b.children[1].running_mean.t.data[k]
        var av = a.children[1].running_var.t.data[k]
        var bv = b.children[1].running_var.t.data[k]
        if abs(am) > Scalar[DT](1e-4) or abs(av - Scalar[DT](1.0)) > Scalar[DT](1e-4):
            moved = True
        if abs(am - bm) > TOL or abs(av - bv) > TOL:
            matched = False
    return moved and matched


def main() raises:
    print("State checkpoint round-trip (BatchNorm running stats)")
    var oc = _check["cpu"](None, String("/tmp/storage_state_ckpt_cpu.txt"))
    print("  CPU:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c), String("/tmp/storage_state_ckpt_gpu.txt"))
    print("  GPU:", "OK" if og else "FAIL")
    assert_true(oc and og, "State checkpoint round-trip")
    print("STATE CHECKPOINT OK")
