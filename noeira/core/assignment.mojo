"""Linear sum assignment, and the uniform-weight EMD built on it.

`scipy.optimize.linear_sum_assignment` was the last thing keeping Python on
the G1 tracking-eval path. It is not exotic — a pairwise cost matrix and a
shortest-augmenting-path solve — so it lives here natively, and
`tools/g1/bfm_zero_tracking_oracle.py` becomes the ORACLE it is gated against,
the same arrangement as the LeRobot and LAFAN importers.

## Why the EMD is an assignment at all

For two sets of EQUAL size with UNIFORM weights, the optimal-transport plan is
a permutation: every unit of mass moves whole. `ot.emd2` and
`linear_sum_assignment` therefore agree exactly on this input, which is what
the reference's `_emd` relies on and why no transport solver is needed.

⚠ THE VALUE IS THE OUTPUT, NOT THE PERMUTATION. With tied costs several
assignments are optimal and they all have the same total, so a gate that
compares the chosen columns against another solver's will fail on a correct
implementation. Compare the COST. See `_an_orthogonality_check_cannot_gate_a_rotation`
for the same shape of mistake.

⚠ Float64 THROUGHOUT, deliberately. The solve is combinatorial, so the answer
is exact once the costs are; float32 costs would introduce ties that are not
ties in the reference and let the two implementations pick genuinely different
totals.
"""

from std.math import sqrt


comptime _INF: Float64 = 1e300


def linear_sum_assignment(
    cost: List[Float64], n: Int, m: Int
) raises -> List[Int]:
    """Minimum-cost assignment of `n` rows to distinct columns of `m`.

    `cost` is row-major, `cost[i * m + j]`. Returns `col4row`: for each row,
    the column it was assigned. Requires `n <= m`.

    The Hungarian algorithm in its shortest-augmenting-path form with dual
    potentials (`u`, `v`) — one augmentation per row, each found by a Dijkstra
    over the reduced costs. O(n·m²); at the eval's n = m = 499 that is ~1.2e8
    operations, tens of milliseconds.
    """
    if n > m:
        raise Error(
            "linear_sum_assignment: n > m (" + String(n) + " > " + String(m)
            + "); transpose the problem"
        )
    if n <= 0:
        return List[Int]()
    if len(cost) < n * m:
        raise Error(
            "linear_sum_assignment: cost has " + String(len(cost))
            + " entries, need n*m = " + String(n * m)
        )

    # 1-based working arrays; index 0 is the algorithm's virtual start column.
    var u = List[Float64](length=n + 1, fill=0.0)
    var v = List[Float64](length=m + 1, fill=0.0)
    var p = List[Int](length=m + 1, fill=0)    # p[j] = row (1-based) on column j
    var way = List[Int](length=m + 1, fill=0)  # predecessor column on the path

    for i in range(1, n + 1):
        p[0] = i
        var j0 = 0
        var minv = List[Float64](length=m + 1, fill=_INF)
        var used = List[Bool](length=m + 1, fill=False)
        while True:
            used[j0] = True
            var i0 = p[j0]
            var delta = _INF
            var j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    var cur = cost[(i0 - 1) * m + (j - 1)] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Walk the augmenting path back, flipping the matching along it.
        while True:
            var j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    var col4row = List[Int](length=n, fill=-1)
    for j in range(1, m + 1):
        if p[j] != 0:
            col4row[p[j] - 1] = j - 1
    return col4row^


def assignment_cost(
    cost: List[Float64], col4row: List[Int], m: Int
) raises -> Float64:
    """Total cost of an assignment — the sum the solver minimised."""
    var tot = Float64(0)
    for i in range(len(col4row)):
        var j = col4row[i]
        if j < 0:
            raise Error("assignment_cost: row " + String(i) + " unassigned")
        tot += cost[i * m + j]
    return tot


def pairwise_l2(
    x: List[Float64], y: List[Float64], t: Int, d: Int
) raises -> List[Float64]:
    """`cost[i*t + j] = ||x_i - y_j||_2`, both `(t, d)` row-major.

    ⚠ `sqrt(max(s, 0))` exactly as the reference writes it — the clamp is not
    decoration, it is what keeps a -0.0 sum from producing a NaN.
    """
    var cost = List[Float64](length=t * t, fill=0.0)
    for i in range(t):
        for j in range(t):
            var s = Float64(0)
            for k in range(d):
                var e = x[i * d + k] - y[j * d + k]
                s += e * e
            cost[i * t + j] = sqrt(s if s > 0.0 else 0.0)
    return cost^


def emd_uniform(
    x: List[Float64], y: List[Float64], t: Int, d: Int
) raises -> Float64:
    """The reference's `_emd`: mean assigned pairwise distance between two
    equal-size, uniformly weighted sets of `d`-dimensional points.

    Note what it measures — rows are matched to rows REGARDLESS OF TIME ORDER,
    so this is a distance between two trajectories as SETS of poses. That is
    what makes it different from `distance` (mean per-frame error) and why a
    policy can score well on one and badly on the other.
    """
    if t <= 0:
        raise Error("emd_uniform: t must be positive")
    var cost = pairwise_l2(x, y, t, d)
    var a = linear_sum_assignment(cost, t, t)
    return assignment_cost(cost, a, t) / Float64(t)
