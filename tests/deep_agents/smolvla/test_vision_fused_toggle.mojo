"""The deploy's `fused_attention` switch REACHES the CrossAttention leaf inside
`SigLIPAttention` — through the ComputeGraph — and comes back off.

    pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_vision_fused_toggle.mojo

⚠ WHY A GATE FOR A ONE-LINE FORWARDER. `Module.set_attr`'s default is `pass`,
so a wrapper that forgets to forward an attr makes the caller's switch a
silent no-op: the deploy would print "fused" and run the two-pass kernel,
and nothing downstream could tell. Dropout's `training` did exactly this
once. The observable here is the leaf's own contract: after a fused forward
it REFUSES a vjp (no weights were cached), and after the switch is turned
back off the same vjp runs. A wrapper that swallowed the attr passes neither
half. Shallow shape, seeded weights, seconds.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.smolvla.vision import SigLIPAttention

comptime SEQ = 8
comptime DIM = 16
comptime HEADS = 4
comptime B = 2
comptime Attn = SigLIPAttention[SEQ, DIM, HEADS]
comptime N = B * SEQ * DIM


def _vjp_raises(mut a: Attn, mut x: Tensor, mut g: Tensor, mut gx: Tensor,
                ctx: DeviceContext) -> Bool:
    try:
        a.vjp["gpu", B](
            TensorRefs[1, MutAnyOrigin](x), g,
            TensorRefs[1, MutAnyOrigin](gx), Optional(ctx),
        )
    except:
        return True
    return False


def main() raises:
    var ctx = DeviceContext()
    print("=" * 70)
    print("SigLIPAttention: the fused_attention switch reaches the leaf")
    print("=" * 70)
    var a = Attn.make["gpu", Deterministic](Optional(ctx))
    var x = Tensor.alloc(N)
    var g = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT](((i * 7) % 11) - 5) * 0.1
        g.data[i] = Scalar[DT](((i * 3) % 5) - 2) * 0.1
    x.upload(ctx)
    g.upload(ctx)
    var out = Tensor()
    var gx = Tensor()

    # [1] off (the default): forward then vjp works.
    a.forward["gpu", B](TensorRefs[1, MutAnyOrigin](x), out, Optional(ctx))
    ctx.synchronize()
    if _vjp_raises(a, x, g, gx, ctx):
        raise Error("[1] the vjp must run with the switch off (default)")
    print("  [1] off by default: forward + vjp run")

    # [2] on, set on the WRAPPER: the leaf refuses a vjp => the attr reached.
    a.set_attr["fused_attention"](Scalar[DT](1.0))
    a.forward["gpu", B](TensorRefs[1, MutAnyOrigin](x), out, Optional(ctx))
    ctx.synchronize()
    out.download(ctx)
    for i in range(N):
        var v = Float64(out.data[i])
        if v != v:
            raise Error("[2] fused forward produced NaN")
    if not _vjp_raises(a, x, g, gx, ctx):
        raise Error(
            "[2] vjp ran after a fused forward: the switch did NOT reach the"
            " CrossAttention leaf (set_attr swallowed on the way down)"
        )
    print("  [2] on via the wrapper: fused forward finite, vjp refuses")

    # [3] off again: back to the two-pass path, vjp runs.
    a.set_attr["fused_attention"](Scalar[DT](0.0))
    a.forward["gpu", B](TensorRefs[1, MutAnyOrigin](x), out, Optional(ctx))
    ctx.synchronize()
    if _vjp_raises(a, x, g, gx, ctx):
        raise Error("[3] the vjp must run again once the switch is off")
    print("  [3] off again: vjp runs")
    print("PASSED")
