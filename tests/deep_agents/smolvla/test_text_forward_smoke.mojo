"""The SmolLM2 text tower runs — forward and vjp, at the checkpoint's widths.

Companion to `test_text_name_map.mojo`: the map proves every published tensor
has a slot, this proves the tower those slots live in can actually execute.
Real widths (960 hidden, 15 q heads over 5 kv, ff 2560, 16 layers); SEQ is small
so the walk and the attention stay cheap, and no shape here depends on it.

⚠ Non-vacuity: finite is not enough. A tower whose residual path dominates and
whose sublayers contribute nothing is finite and wrong, so the output is also
required to VARY, and the input gradient to be non-zero everywhere.

Run:
  pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_text_forward_smoke.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.smolvla.text import (
    SmolLMTextTower, SMOLLM_DIM, SMOLLM_HEADS, SMOLLM_KV_HEADS, SMOLLM_LAYERS,
)

comptime SEQ = 32
comptime TOWER = SmolLMTextTower[SEQ]
comptime B = 1
comptime N = SEQ * SMOLLM_DIM


def main() raises:
    print("=" * 68)
    print("SmolVLA SmolLM2 text tower — forward/vjp smoke (GPU)")
    print("=" * 68)
    print("  ", SMOLLM_LAYERS, "layers, dim", SMOLLM_DIM, ",", SMOLLM_HEADS,
          "q heads over", SMOLLM_KV_HEADS, "kv, seq", SEQ)

    var c = DeviceContext()
    var net = TOWER.make["gpu", Deterministic](Optional(c))

    var x = Tensor.alloc(B * N)
    for i in range(B * N):
        x.data[i] = Scalar[DT](((i * 31) % 23) - 11) * 0.02
    x.upload(c)
    var out = Tensor.alloc(B * N)
    net.forward["gpu", B](TensorRefs[1](x), out, Optional(c))
    out.download(c)

    var nan = 0
    var lo = out.data[0]
    var hi = out.data[0]
    for i in range(B * N):
        var y = out.data[i]
        if y != y:
            nan += 1
        if y < lo:
            lo = y
        if y > hi:
            hi = y
    print("  forward: compared", B * N, " nan =", nan, " min =", lo,
          " max =", hi)
    assert_true(nan == 0, "forward produced NaN")
    assert_true(hi - lo > 1e-6, "output is constant — the tower computes"
                                " nothing")

    var go = Tensor.alloc(B * N)
    for i in range(B * N):
        go.data[i] = Scalar[DT](((i % 7) - 3)) * 0.001
    go.upload(c)
    var gi = Tensor.alloc(B * N)
    net.zero_grad["gpu"](Optional(c))
    net.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(c))
    gi.download(c)

    var gnan = 0
    var gnz = 0
    for i in range(B * N):
        var g = gi.data[i]
        if g != g:
            gnan += 1
        elif g != 0:
            gnz += 1
    print("  vjp    : compared", B * N, " nan =", gnan, " nonzero =", gnz)
    assert_true(gnan == 0, "vjp produced NaN")
    assert_true(gnz == B * N, "input gradient has zeros — the backward did not"
                              " reach every input")

    print()
    print("PASSED")
