"""G16 — SWM Phase 6b: many cycles at once, and the reading that `det` misses.

Every gate so far has read exactly ONE holonomy, because a ring has one
fundamental cycle. This is the first setting where the machinery does what §2
actually asks: take a spanning tree, enumerate `|E| - |V| + 1` fundamental
cycles, and classify each on its own.

The base is a torus grid; the bundle is made NON-ORIENTABLE by a reflecting
x-seam. That is the structure which makes a Klein bottle non-orientable (`w1`
non-zero), on a base whose graph is rich in cycles — and it is described that
way rather than as "a Klein bottle" because the base identification here is the
torus's, not the Klein bottle's. What is under test is the multi-cycle reading,
and this provides it: on a 6x5 grid, 31 fundamental cycles of which 5 reverse
orientation, 8 are non-trivial rotations (loops crossing the seam twice at
different frames), and the rest are trivial.

Two things this gate reaches that nothing before it could:

**Cross-confirmation on REAL cycles.** `confirm_by_independent_cycles` was
gated on synthetic edge lists. Here the cycles come from an actual spanning
tree, and 4 edge-disjoint pairs of non-trivial cycles exist — the situation the
rule was written for, since one biased edge can explain any number of
overlapping cycles but not two that share nothing.

**`dim ker(H - I)`, which `det H` under-reports above 2D.** The design doc gives
the example itself: in O(3), `-I` has `det = -1` and NO fixed vector. Measured
here — O(2) reflection: `det -1`, fixed subspace 1. O(3) single reflection:
`det -1`, fixed subspace 2. O(3) `-I`: `det -1`, fixed subspace **0**. So in 2D
the two readings agree and above 2D they come apart, and a method that only ever
reads the determinant is answering a coarser question than the one asked.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_klein_multicycle.mojo
"""

from std.math import abs
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat, fixed_subspace_dim
from mojo_rl.experimental.swm.procrustes import procrustes_o_d
from mojo_rl.experimental.swm.envs.klein_grid import KleinGrid
from mojo_rl.experimental.swm.observables import (
    classify,
    class_name,
    cycles_are_edge_disjoint,
    confirm_by_independent_cycles,
    CLASS_NOMINAL,
    CLASS_OBSTRUCTION,
    CLASS_UNDECIDED,
)

comptime DT = DType.float64
comptime W = 6
comptime H = 5
comptime ANGLE_TOL = 0.2


def main() raises:
    var checks = 0

    for arm in range(2):
        var non_orientable = arm == 0
        var label = "non-orientable" if non_orientable else "ORIENTABLE control"
        var kg = KleinGrid[W, H](non_orientable)
        var g = kg.build_graph()
        var cyc = g.fundamental_cycle_edges()

        var expect = g.n_edges() - g.n_places + 1
        checks += 1
        assert_true(
            len(cyc) == expect,
            label + ": a spanning tree must leave |E|-|V|+1 = " + String(expect)
            + " fundamental cycles, got " + String(len(cyc)),
        )

        var n_obstruction = 0
        var n_undecided = 0
        var n_nominal = 0
        var nontrivial = List[Int]()
        for i in range(len(cyc)):
            var d = g.holonomy_det(cyc[i])
            var f = g.holonomy_dist_to_identity(cyc[i])
            var v = classify(0.0, 1.0, d, f, ANGLE_TOL, False)
            if v == CLASS_OBSTRUCTION:
                n_obstruction += 1
            elif v == CLASS_UNDECIDED:
                n_undecided += 1
                nontrivial.append(cyc[i])
            elif v == CLASS_NOMINAL:
                n_nominal += 1
        print(
            label, ": cycles", len(cyc), " OBSTRUCTION", n_obstruction,
            " UNDECIDED", n_undecided, " NOMINAL", n_nominal,
        )

        if non_orientable:
            checks += 2
            assert_true(
                n_obstruction > 0,
                "a non-orientable bundle must have orientation-reversing cycles",
            )
            assert_true(
                n_nominal > 0,
                "...and trivial ones too — if EVERY cycle were flagged, the "
                + "classification would not be discriminating between cycles",
            )

            # ---- cross-confirmation on cycles from a real spanning tree -----
            var sets = List[List[Int]]()
            var fros = List[Float64]()
            for i in range(len(nontrivial)):
                sets.append(g.cycle_edge_set(nontrivial[i]))
                fros.append(g.holonomy_dist_to_identity(nontrivial[i]))
            var disjoint_pairs = 0
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    if cycles_are_edge_disjoint(sets[i], sets[j]):
                        disjoint_pairs += 1
            print("   non-trivial det=+1 cycles:", len(sets),
                  " edge-disjoint pairs:", disjoint_pairs)
            checks += 2
            assert_true(
                disjoint_pairs > 0,
                "the grid must actually contain EDGE-DISJOINT non-trivial "
                + "cycles, else cross-confirmation is untested here",
            )
            assert_true(
                confirm_by_independent_cycles(sets, fros, ANGLE_TOL),
                "edge-disjoint cycles both reporting a holonomy must confirm",
            )
            # ...and a subset that overlaps must NOT confirm.
            var overlapping = List[List[Int]]()
            var over_fro = List[Float64]()
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    if not cycles_are_edge_disjoint(sets[i], sets[j]):
                        overlapping.append(sets[i].copy())
                        overlapping.append(sets[j].copy())
                        over_fro.append(fros[i])
                        over_fro.append(fros[j])
                        break
                if len(overlapping) > 0:
                    break
            checks += 1
            assert_true(
                len(overlapping) == 0
                or not confirm_by_independent_cycles(
                    overlapping, over_fro, ANGLE_TOL
                ),
                "two OVERLAPPING cycles must not confirm — one biased edge "
                + "explains both",
            )
        else:
            checks += 2
            assert_true(
                n_obstruction == 0,
                "NEGATIVE CONTROL FAILED: the orientable bundle reported "
                + String(n_obstruction) + " obstructed cycles",
            )
            assert_true(
                n_nominal == len(cyc),
                "the orientable bundle must be trivial on EVERY cycle, got "
                + String(n_nominal) + "/" + String(len(cyc)),
            )

    # =====================================================================
    # Transports RECOVERED from noisy observations must give the same reading.
    # =====================================================================
    var kg = KleinGrid[W, H](True)
    var g = kg.build_graph()
    var batches = kg.rollout_pairs(240, 40, 0.02, 7717)
    var g_fit = kg.build_graph()
    var fitted = 0
    for e in range(g.n_edges()):
        if batches[e].count() >= 8:
            g_fit.transports[e] = procrustes_o_d[2, DT](batches[e])
            fitted += 1
    g_fit.rebuild_gauge(0)
    checks += 1
    assert_true(
        fitted == g.n_edges(),
        "every edge must have been visited enough to fit: " + String(fitted)
        + "/" + String(g.n_edges()),
    )
    var cyc = g.fundamental_cycle_edges()
    var agree = 0
    for i in range(len(cyc)):
        var d_true = g.holonomy_det(cyc[i])
        var d_fit = g_fit.holonomy_det(cyc[i])
        if (d_true < 0) == (d_fit < 0):
            agree += 1
    print("cycles whose det H SIGN survives Procrustes recovery:", agree, "/",
          len(cyc))
    checks += 1
    assert_true(
        agree == len(cyc),
        "the Z/2 class of EVERY cycle must survive recovery from noisy data: "
        + String(agree) + "/" + String(len(cyc)),
    )

    # =====================================================================
    # dim ker(H - I): what det H under-reports above 2D (the doc's §2 example).
    # =====================================================================
    var refl2 = SqMat[2, DT].identity()
    refl2[1, 1] = Scalar[DT](-1)
    var refl3 = SqMat[3, DT].identity()
    refl3[2, 2] = Scalar[DT](-1)
    var neg3 = SqMat[3, DT].identity()
    for i in range(3):
        neg3[i, i] = Scalar[DT](-1)

    var k2 = fixed_subspace_dim[2, DT](refl2)
    var k3r = fixed_subspace_dim[3, DT](refl3)
    var k3n = fixed_subspace_dim[3, DT](neg3)
    print("O(2) reflection: det", refl2.det(), " fixed subspace", k2)
    print("O(3) reflection: det", refl3.det(), " fixed subspace", k3r)
    print("O(3) H = -I    : det", neg3.det(), " fixed subspace", k3n)
    checks += 5
    assert_true(k2 == 1, "an O(2) reflection fixes a line, got " + String(k2))
    assert_true(k3r == 2, "an O(3) reflection fixes a plane, got " + String(k3r))
    assert_true(
        k3n == 0,
        "O(3) H = -I must fix NOTHING — this is the doc's own example of det H "
        + "under-reporting above 2D. got " + String(k3n),
    )
    assert_true(
        abs(Float64(neg3.det()) + 1.0) < 1e-12
        and abs(Float64(refl3.det()) + 1.0) < 1e-12,
        "both O(3) cases must have det = -1 — that is the point: SAME class, "
        + "DIFFERENT fixed subspace",
    )
    assert_true(
        fixed_subspace_dim[3, DT](SqMat[3, DT].identity()) == 3,
        "the identity must fix everything",
    )

    print()
    print("grid:", W, "x", H, " assertions compared :", checks)
    print("PASS: G16 multi-cycle classification, cross-confirmation, dim ker(H-I)")
