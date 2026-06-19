"""Grouped polyak (arena soft-update) correctness (GPU).

When both the online and target nets are arena-backed, `polyak_arenas` does the
target-net soft-update `target = (1-τ)·target + τ·online` in ONE kernel over the
value arenas (vs N per-param launches). Gated against the analytic formula:
  target_after[i] == (1-τ)·target_init[i] + τ·online_final[i]
with online drifted from target by a couple of Adam steps (so the update is
non-trivial). The online arena comes from its Adam optimizer; the target arena is
a standalone ParamArena (target nets have no optimizer).

Run: pixi run -e apple mojo run -I . tests/nn/test_grouped_polyak_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.optimizer.param_arena import ParamArena, polyak_arenas


comptime D = 4
comptime H = 6
comptime O = 3
comptime B = 5
comptime NET = Sequential[Linear[D, H], Linear[H, O]]


def main() raises:
    print("Grouped polyak (arena soft-update) correctness")
    var c = DeviceContext()
    comptime tau = Scalar[DT](0.2)

    # Online net — adopted by its Adam optimizer (→ optG.arena).
    var online = NET.make["gpu", Deterministic](Optional(c))
    var optG = Adam(lr=1e-1)
    optG.adopt["gpu"](online, Optional(c))

    # Target net — standalone ParamArena (target nets have no optimizer).
    var target = NET.make["gpu", Deterministic](Optional(c))
    var arenaTg = ParamArena()
    arenaTg.adopt["gpu"](target, Optional(c))

    # Snapshot target's init values (before polyak).
    arenaTg.val.download(c)
    var target_init = List[Scalar[DT]]()
    for i in range(arenaTg.total):
        target_init.append(arenaTg.val.data[i])

    # Drift the online net a couple of steps so online != target.
    for step in range(2):
        var x = Tensor.alloc(B * D); var go = Tensor.alloc(B * O)
        for i in range(B * D):
            x.data[i] = Scalar[DT](((i + step) % 5) - 2) * 0.3
        for i in range(B * O):
            go.data[i] = Scalar[DT](((i * 3 + step) % 7) - 3) * 0.4
        x.upload(c); go.upload(c)
        var out = Tensor.alloc(B * O); var gi = Tensor.alloc(B * D)
        optG.zero_grad["gpu"](online, Optional(c))
        online.forward["gpu", B](TensorRefs[1](x), out, Optional(c))
        online.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(c))
        optG.step["gpu"](online, Optional(c))

    # Snapshot the drifted online values.
    optG.arena.val.download(c)
    var online_final = List[Scalar[DT]]()
    for i in range(optG.arena.total):
        online_final.append(optG.arena.val.data[i])

    # Grouped soft-update, then check vs the analytic formula.
    polyak_arenas(arenaTg, optG.arena, tau, c)
    arenaTg.val.download(c)

    var max_err = Scalar[DT](0.0)
    var moved = False
    for i in range(arenaTg.total):
        var expected = (Scalar[DT](1.0) - tau) * target_init[i] + tau * online_final[i]
        var err = abs(arenaTg.val.data[i] - expected)
        if err > max_err:
            max_err = err
        if abs(target_init[i] - online_final[i]) > Scalar[DT](1e-4):
            moved = True  # online actually drifted (test is non-trivial)
    print("  total =", arenaTg.total, " max|after - expected| =", max_err, " drifted:", moved)
    assert_true(moved and max_err < Scalar[DT](1e-6), "grouped polyak")
    print("GROUPED POLYAK OK")
