"""`linear_sum_assignment` / `emd_uniform` against an exact brute force and scipy.

The G1 tracking eval's `emd` was the last metric that needed Python
(`scipy.optimize.linear_sum_assignment`). `core/assignment.mojo` is the native
port; this is what makes it trustworthy.

  [1] EXACT, AND INDEPENDENT OF SCIPY. For n <= 7 every assignment is
      enumerated, so the optimum is known by construction rather than by
      agreement with another implementation of the same idea. A gate that only
      compared against scipy would share whatever scipy shares with us; this
      one cannot (`_a_gate_that_shares_its_reference_implementation_is_blind`).
  [2] Rectangular n < m, where rows pick from a wider column set.
  [3] THE ORACLE, at sizes brute force cannot reach: scipy on a 120x120 cost
      matrix and on a (100, 29) EMD, the eval's actual shape. Skipped loudly if
      scipy is absent — [1] still runs, and the report says which legs ran.
  [4] NON-VACUITY. Random costs can make the identity assignment optimal, and
      then a solver that always returns the diagonal passes [1] while being
      broken. So the number of cases where the optimum strictly beats the
      diagonal is counted and asserted non-zero.

⚠ THE VALUE IS GATED, NEVER THE PERMUTATION. Tied costs admit several optimal
assignments with identical totals; comparing chosen columns would fail a
correct solver.

Run: pixi run mojo run -I . tests/core/test_assignment_emd.mojo
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.random import random_float64, seed
from std.testing import assert_true

from noeira.core.assignment import (
    linear_sum_assignment, assignment_cost, pairwise_l2, emd_uniform,
)


comptime INF: Float64 = 1e300


def _brute(cost: List[Float64], n: Int, m: Int, row: Int,
           mut used: List[Bool]) raises -> Float64:
    """Exact optimum by enumerating every assignment. Only for tiny n."""
    if row == n:
        return 0.0
    var best = INF
    for j in range(m):
        if not used[j]:
            used[j] = True
            var sub = _brute(cost, n, m, row + 1, used)
            used[j] = False
            var tot = cost[row * m + j] + sub
            if tot < best:
                best = tot
    return best


def _rand_cost(n: Int, m: Int) raises -> List[Float64]:
    var c = List[Float64](length=n * m, fill=0.0)
    for i in range(n * m):
        c[i] = random_float64() * 10.0
    return c^


def _diag_cost(cost: List[Float64], n: Int, m: Int) -> Float64:
    var t = Float64(0)
    for i in range(n):
        t += cost[i * m + i]
    return t


def main() raises:
    seed(20260911)
    print("linear_sum_assignment / emd_uniform")
    var checked = 0
    var beat_diag = 0
    var worst = Float64(0)

    # ---- [1] + [2] exact brute force, square and rectangular ------------
    for n in range(1, 8):
        for m in range(n, n + 3):
            for _ in range(12):
                var cost = _rand_cost(n, m)
                var a = linear_sum_assignment(cost, n, m)
                # a valid assignment: n distinct columns in range
                var seen = List[Bool](length=m, fill=False)
                for i in range(n):
                    var j = a[i]
                    assert_true(
                        j >= 0 and j < m and not seen[j],
                        "invalid assignment at row " + String(i),
                    )
                    seen[j] = True
                var got = assignment_cost(cost, a, m)
                var used = List[Bool](length=m, fill=False)
                var want = _brute(cost, n, m, 0, used)
                var err = abs(got - want)
                if err > worst:
                    worst = err
                assert_true(
                    err < 1e-9,
                    "n=" + String(n) + " m=" + String(m) + ": solver "
                    + String(got) + " vs exact " + String(want),
                )
                if n == m and want < _diag_cost(cost, n, m) - 1e-9:
                    beat_diag += 1
                checked += 1
    print("  [1/2] exact: ", checked, "cases n=1..7 (square + rectangular),",
          " worst |solver - brute| =", worst)

    # ---- [4] non-vacuity -------------------------------------------------
    print("  [4] cases where the optimum strictly beat the diagonal:", beat_diag,
          "of", checked)
    assert_true(checked > 0, "vacuous: no cases were checked at all")
    assert_true(
        beat_diag > 20,
        "vacuous: the optimum almost never beat the diagonal (" + String(beat_diag)
        + "), so a solver that just returns the identity would pass [1]",
    )

    # ---- [3] the oracle, at real sizes ----------------------------------
    var have_scipy = True
    var opt = PythonObject(None)
    var builtins = PythonObject(None)
    try:
        builtins = Python.import_module("builtins")
        opt = Python.import_module("scipy.optimize")
    except e:
        have_scipy = False
    if not have_scipy:
        print("  [3] SKIPPED — scipy not importable; [1] still gated the solver")
    else:
        var np = Python.import_module("numpy")
        # (a) a raw 120x120 cost matrix
        var T = 120
        var cost = _rand_cost(T, T)
        var pl = builtins.list()
        for i in range(T * T):
            _ = pl.append(cost[i])
        var arr = np.array(pl).reshape(T, T)
        var rc = opt.linear_sum_assignment(arr)
        var sp = Float64(0)
        for i in range(T):
            var r = Int(Float64(py=rc[0][i]))
            var c = Int(Float64(py=rc[1][i]))
            sp += cost[r * T + c]
        var mine = assignment_cost(cost, linear_sum_assignment(cost, T, T), T)
        print("  [3a] 120x120: ours", mine, " scipy", sp, " diff", abs(mine - sp))
        assert_true(abs(mine - sp) < 1e-9, "120x120 disagrees with scipy")

        # (b) the eval's actual shape: EMD over (100, 29) trajectories
        var t = 100
        var d = 29
        var x = List[Float64](length=t * d, fill=0.0)
        var y = List[Float64](length=t * d, fill=0.0)
        for i in range(t * d):
            x[i] = random_float64() * 2.0 - 1.0
            y[i] = random_float64() * 2.0 - 1.0
        var mine_emd = emd_uniform(x, y, t, d)
        var c2 = pairwise_l2(x, y, t, d)
        var pl2 = builtins.list()
        for i in range(t * t):
            _ = pl2.append(c2[i])
        var arr2 = np.array(pl2).reshape(t, t)
        var rc2 = opt.linear_sum_assignment(arr2)
        var sp2 = Float64(0)
        for i in range(t):
            var r = Int(Float64(py=rc2[0][i]))
            var c = Int(Float64(py=rc2[1][i]))
            sp2 += c2[r * t + c]
        sp2 = sp2 / Float64(t)
        print("  [3b] EMD (100, 29): ours", mine_emd, " scipy", sp2,
              " diff", abs(mine_emd - sp2))
        assert_true(abs(mine_emd - sp2) < 1e-9, "EMD disagrees with scipy")

    print("ASSIGNMENT_EMD OK")
