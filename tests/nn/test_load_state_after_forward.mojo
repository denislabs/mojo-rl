"""A forward BEFORE `load_state` must not shadow the loaded weights.

`Linear`/`LinearAct` cache a derived copy of the weight (`w_pad`, the
K-alignment pad; `w_bf`, the AMP bf16 recast) gated on `weight.val.version`.
Restoring a checkpoint writes `weight.val` directly, so unless the loader
advances that counter the cache keeps serving the PRE-LOAD weight and the
checkpoint is silently ignored.

The discriminating sequence is the one a viewer performs when it switches
checkpoints after it has already driven a policy:

    A: make -> load -> forward          (cache built AFTER the load)
    B: make -> forward -> load -> forward   (cache built BEFORE the load)

A and B must agree. Before the fix they did not.
"""

from std.math import abs
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.checkpoint import save_params, load_params
from mojo_rl.nn.primitives.linear import Linear

comptime IN = 518      # unaligned → the pad cache is ACTIVE
comptime OUT = 256
comptime B = 8
comptime PATH = "/tmp/_test_load_state_after_forward.ckpt"


def fill_x(mut x: Tensor):
    for i in range(B * IN):
        x.data[i] = Scalar[DT](0.011) * Scalar[DT]((i % 29) - 14)


def main() raises:
    var ctx = DeviceContext()
    print("device:", ctx.name())
    comptime L = Linear[IN, OUT]
    print("  IN=", IN, "  NEEDS_PAD=", L.NEEDS_PAD, "  K_PAD=", L.K_PAD)

    # A DIFFERENT set of weights to save, so a stale cache is detectable.
    var donor = L.make["gpu", INIT=Kaiming](ctx=ctx)
    save_params["gpu", L](donor, PATH, Optional(ctx))

    var x = Tensor.alloc(B * IN)
    fill_x(x)
    x.upload(ctx)
    var y_a = Tensor.alloc_gpu(ctx, B * OUT)
    var y_b = Tensor.alloc_gpu(ctx, B * OUT)

    # ── A: load FIRST, then forward ──────────────────────────────────────
    var a = L.make["gpu", INIT=Kaiming](ctx=ctx)
    load_params["gpu", L](a, PATH, Optional(ctx))
    a.forward["gpu", B](TensorRefs[1](x), y_a, Optional(ctx))
    y_a.download(ctx)

    # ── B: forward FIRST (builds the cache), then load, then forward ─────
    var b = L.make["gpu", INIT=Kaiming](ctx=ctx)
    var scratch = Tensor.alloc_gpu(ctx, B * OUT)
    b.forward["gpu", B](TensorRefs[1](x), scratch, Optional(ctx))
    load_params["gpu", L](b, PATH, Optional(ctx))
    b.forward["gpu", B](TensorRefs[1](x), y_b, Optional(ctx))
    y_b.download(ctx)
    ctx.synchronize()

    var max_rel = Float64(0)
    var mag = Float64(0)
    for i in range(B * OUT):
        var va = Float64(y_a.data[i])
        var vb = Float64(y_b.data[i])
        if abs(va) > mag:
            mag = abs(va)
        var denom = abs(va) if abs(va) > 1e-6 else 1e-6
        var r = abs(va - vb) / denom
        if r > max_rel:
            max_rel = r
    print("  A (load->fwd) [0] =", y_a.data[0])
    print("  B (fwd->load->fwd) [0] =", y_b.data[0])
    print("  max_rel =", max_rel, "  |A|max =", mag)
    if mag == 0.0:
        raise Error("VACUOUS: output is all zeros")
    if max_rel > 1e-5:
        raise Error(
            "STALE CACHE: a forward before load_state shadowed the checkpoint"
        )
    print("PASSED — the pre-load forward did not shadow the checkpoint")
