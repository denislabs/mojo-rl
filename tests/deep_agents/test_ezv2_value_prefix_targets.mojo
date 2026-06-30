"""EZv2 value-prefix target transform unit test (Stage 3).

`value_prefix_from_rewards[K, HORIZON]` converts per-step reward targets ([K,B],
time-major) into cumulative within-window value prefixes that reset at every
HORIZON boundary — matching EZ `batch_worker.py:381-395` (value_prefix=True).

Two cases:
  * K=6, HORIZON=3 (two windows) → resets at k=0 and k=3, hand-computed.
  * K=5, HORIZON=5 (shipping Atari) → single window, prefix = full cumsum.

Run:
    pixi run mojo run -I . tests/deep_agents/test_ezv2_value_prefix_targets.mojo
"""

from std.math import abs
from std.testing import assert_true
from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.zero import value_prefix_from_rewards


def _close(a: Scalar[DT], b: Scalar[DT]) -> Bool:
    return abs(a - b) < Scalar[DT](1e-6)


def main() raises:
    print("=" * 70)
    print("EZv2 value-prefix target transform unit test")
    print("=" * 70)

    # ── Case 1: K=6, HORIZON=3, B=2 — two windows, reset at k=0 and k=3 ──
    comptime K1 = 6
    comptime H1 = 3
    comptime B1 = 2
    # rewards (time-major [K,B]): b0 = 1,2,3,4,5,6 ; b1 = 10,20,30,40,50,60
    var r = List[Scalar[DT]](length=K1 * B1, fill=0)
    for k in range(K1):
        r[k * B1 + 0] = Scalar[DT](k + 1)
        r[k * B1 + 1] = Scalar[DT]((k + 1) * 10)
    value_prefix_from_rewards[K1, H1](r, B1)
    # expected b0: window0 [1,3,6] window1 [4,9,15]; b1: [10,30,60] [40,90,150]
    var exp0 = List[Scalar[DT]](length=K1, fill=0)
    var exp1 = List[Scalar[DT]](length=K1, fill=0)
    exp0[0] = 1; exp0[1] = 3; exp0[2] = 6; exp0[3] = 4; exp0[4] = 9; exp0[5] = 15
    exp1[0] = 10; exp1[1] = 30; exp1[2] = 60
    exp1[3] = 40; exp1[4] = 90; exp1[5] = 150
    for k in range(K1):
        assert_true(_close(r[k * B1 + 0], exp0[k]), "b0 prefix @k")
        assert_true(_close(r[k * B1 + 1], exp1[k]), "b1 prefix @k")
    print("   K=6,H=3 two-window prefixes OK")

    # ── Case 2: K=5, HORIZON=5, B=1 — shipping Atari: full cumsum ──
    comptime K2 = 5
    comptime H2 = 5
    var r2 = List[Scalar[DT]](length=K2, fill=0)
    for k in range(K2):
        r2[k] = Scalar[DT](2)          # constant reward 2
    value_prefix_from_rewards[K2, H2](r2, 1)
    # single window → cumsum: 2,4,6,8,10
    for k in range(K2):
        assert_true(_close(r2[k], Scalar[DT](2 * (k + 1))), "single-window cumsum")
    print("   K=5,H=5 single-window cumsum OK")

    print("=" * 70)
    print("PASS — value-prefix target transform")
    print("=" * 70)
