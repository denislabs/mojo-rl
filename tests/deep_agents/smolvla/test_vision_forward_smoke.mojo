"""The SigLIP tower runs — forward and vjp, at the checkpoint's real shapes.

A name map over a module tree that cannot execute proves very little, so this
is the companion to `test_vision_name_map.mojo`: same tower, real dimensions
(512x512 image -> 1024 tokens x 768), constructed and pushed through both
directions on the GPU.

⚠ Non-vacuity: a NaN check alone passes on an all-zero output, which is exactly
what a mis-shaped attention or a dead residual produces. So this asserts the
output is finite AND that it actually varies — a constant tensor is a failure
here, not a pass.

Run:
  pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_vision_forward_smoke.mojo
  pixi run -e nvidia mojo run -I . tests/deep_agents/smolvla/test_vision_forward_smoke.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.smolvla.vision import (
    SigLIPVisionTower, SIGLIP_IMG, SIGLIP_TOKENS, SIGLIP_DIM,
)

comptime TOWER = SigLIPVisionTower[]
comptime B = 1
comptime IN_N = 3 * SIGLIP_IMG * SIGLIP_IMG
comptime OUT_N = SIGLIP_TOKENS * SIGLIP_DIM


def main() raises:
    print("=" * 68)
    print("SmolVLA SigLIP vision tower — forward/vjp smoke (GPU)")
    print("=" * 68)
    print("  in  =", IN_N, "(3 x", SIGLIP_IMG, "x", SIGLIP_IMG, ")")
    print("  out =", OUT_N, "(", SIGLIP_TOKENS, "tokens x", SIGLIP_DIM, ")")

    var c = DeviceContext()
    var net = TOWER.make["gpu", Deterministic](Optional(c))

    var x = Tensor.alloc(B * IN_N)
    for i in range(B * IN_N):
        x.data[i] = Scalar[DT](((i * 31) % 23) - 11) * 0.03
    x.upload(c)
    var out = Tensor.alloc(B * OUT_N)

    net.forward["gpu", B](TensorRefs[1](x), out, Optional(c))
    out.download(c)

    var nan = 0
    var lo = out.data[0]
    var hi = out.data[0]
    for i in range(B * OUT_N):
        var y = out.data[i]
        if y != y:
            nan += 1
        if y < lo:
            lo = y
        if y > hi:
            hi = y
    print("  forward: compared", B * OUT_N, "elems  nan =", nan,
          " min =", lo, " max =", hi)
    assert_true(nan == 0, "forward produced NaN")
    # A dead tower emits a constant; that is the failure a NaN check misses.
    assert_true(hi - lo > 1e-6, "output is constant — the tower is not"
                                " computing anything")

    var go = Tensor.alloc(B * OUT_N)
    for i in range(B * OUT_N):
        go.data[i] = Scalar[DT](((i % 7) - 3)) * 0.001
    go.upload(c)
    var gi = Tensor.alloc(B * IN_N)
    net.zero_grad["gpu"](Optional(c))
    net.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(c))
    gi.download(c)

    var gnan = 0
    var gnz = 0
    for i in range(B * IN_N):
        var g = gi.data[i]
        if g != g:
            gnan += 1
        elif g != 0:
            gnz += 1
    print("  vjp    : compared", B * IN_N, "elems  nan =", gnan,
          " nonzero =", gnz)
    assert_true(gnan == 0, "vjp produced NaN")
    assert_true(gnz > 0, "input gradient is entirely zero — the backward did"
                         " not reach the input")

    print()
    print("PASSED")
