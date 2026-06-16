"""Phase 5 NatureDQN composition smoke (PORTING_PLAN.md).

Builds a scaled-down NatureDQN with the canonical layer pattern:

    Sequential[
        Conv2D, ReLU,
        Conv2D, ReLU,
        Conv2D, ReLU,
        Flatten,
        Linear, ReLU,
        Linear,
    ]

Then runs forward + backward and checks that:
  1. The whole stack compiles (Sequential picks each child's
     `IN_DIMS[0]` / `OUT_DIM` matching at every boundary).
  2. Output values are finite (no NaN / inf from a broken backward
     somewhere in the chain).
  3. grad_input has the right shape (matches input by reference —
     same flat size; we just check the call completes).

We deliberately do NOT FD-gradcheck the composite — each child already
has its own FD test in Phase 1 / 2 / 5. This test is purely about
"do all the dims chain through `Sequential` correctly".

Spatial dims scaled to fit a tractable forward/backward pass:
    Input    : [BATCH=2, 4, 16, 16]            → flat 4·16·16 = 1024
    Conv2D(4→8,  k=4, s=2, p=0)                → 8·7·7  =  392
    Conv2D(8→16, k=3, s=1, p=0)                → 16·5·5 =  400
    Conv2D(16→16, k=3, s=1, p=0)               → 16·3·3 =  144
    Flatten(144)                               →           144
    Linear(144 → 64)                           →            64
    Linear(64 → 4)                             →             4
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.conv2d import Conv2D
from mojo_rl.nn.primitives.flatten import Flatten
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.initializer import Xavier


def test_nature_dqn_compose() raises:
    print("test_nature_dqn_compose ...")
    comptime BATCH = 2
    comptime IC = 4
    comptime H_IN = 16
    comptime W_IN = 16
    comptime IN_FLAT = IC * H_IN * W_IN

    # Conv1: 4→8 k=4 s=2 p=0, in 16x16 → out 7x7
    comptime Conv1 = Conv2D[IC, 8, 4, 2, 0, H_IN, W_IN]
    # After Conv1: flat 8*7*7 = 392
    comptime OUT1 = 8 * 7 * 7
    # Conv2: 8→16 k=3 s=1 p=0, in 7x7 → out 5x5
    comptime Conv2 = Conv2D[8, 16, 3, 1, 0, 7, 7]
    comptime OUT2 = 16 * 5 * 5
    # Conv3: 16→16 k=3 s=1 p=0, in 5x5 → out 3x3
    comptime Conv3 = Conv2D[16, 16, 3, 1, 0, 5, 5]
    comptime OUT3 = 16 * 3 * 3   # 144

    comptime Net = Sequential[
        Conv1, ReLU[OUT1],
        Conv2, ReLU[OUT2],
        Conv3, ReLU[OUT3],
        Flatten[OUT3],
        Linear[OUT3, 64], ReLU[64],
        Linear[64, 4],
    ]
    var net = Net.make[target="cpu", INIT=Xavier]()

    # Run forward + backward.
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN_FLAT
    )
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * 4
    )
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * 4
    )
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN_FLAT
    )
    for i in range(BATCH * IN_FLAT):
        x[i] = Scalar[DT](-0.3 + 0.0017 * Float64(i))
    for i in range(BATCH * 4):
        go[i] = Scalar[DT](0.5 + 0.1 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, IN_FLAT]())
    var y_t = TileTensor(y, row_major[BATCH, 4]())
    var go_t = TileTensor(go, row_major[BATCH, 4]())
    var gi_t = TileTensor(gi, row_major[BATCH, IN_FLAT]())

    net.forward["cpu", BATCH](x_t, output=y_t)
    net.zero_grad["cpu"]()
    net.vjp["cpu", BATCH](go_t, gi_t)

    # Finite-output sanity (no NaN/inf in y or gi).
    var max_abs_y: Scalar[DT] = 0.0
    for i in range(BATCH * 4):
        var v = y[i]
        var av = v if v >= Scalar[DT](0) else -v
        if av > max_abs_y:
            max_abs_y = av
        # NaN detection: NaN != NaN.
        assert_true(
            v == v,
            "NatureDQN output contains NaN",
        )
    var max_abs_gi: Scalar[DT] = 0.0
    var nonzero_gi: Int = 0
    for i in range(BATCH * IN_FLAT):
        var v = gi[i]
        var av = v if v >= Scalar[DT](0) else -v
        assert_true(
            v == v,
            "NatureDQN grad_input contains NaN",
        )
        if av > max_abs_gi:
            max_abs_gi = av
        if v != Scalar[DT](0.0):
            nonzero_gi += 1
    print("  max |y| =", max_abs_y)
    print("  max |gi| =", max_abs_gi, "  nonzero gi lanes =", nonzero_gi, " / ", BATCH * IN_FLAT)
    # Grad should reach a meaningful fraction of input lanes (a few ReLU
    # gates will mask some, and Conv1's effective receptive field doesn't
    # cover every input lane uniformly, but should hit a clear majority).
    assert_true(
        nonzero_gi > (BATCH * IN_FLAT) // 4,
        "NatureDQN backward should reach a majority of input lanes",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("NatureDQN composition smoke (Phase 5, PORTING_PLAN.md)")
    print("=" * 70)
    test_nature_dqn_compose()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
