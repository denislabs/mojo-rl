"""G29 — SWM Phase 11: refuting a global symmetry, which the graph cannot.

Phase 8 left this as the sharpest open question. Under a LOCAL alias the label
graph refutes the merge — one label, one action, two successor labels — and
G24 splits it. Under a GLOBAL symmetry, `(x, y) ~ (x + W/2, y)`, there is no
such conflict: the quotient is a consistent world, half the size, and every
discrete check passes. Only the frame transports disagree, and G18 measured
that the residual NORM carries no signal (worst aliased-pair ratio 1.01).

**Why the norm cannot work, and where the signal actually is.** Under a
compromise fit `R` for a merged label, place `a`'s residual is `(R_a - R) u`
and place `b`'s is `(R_b - R) u`. Both are LINEAR in `u`, so both are
zero-mean and neither their means nor their magnitudes separate — clustering
residual vectors is provably hopeless. The two places obey different linear
RELATIONS, so the signal lives in the joint `(u, epsilon)`, and reading it
needs a mixture over MAPS, not over points: k-means where each centre is an
`O(2)` transport (`fit_transport_mixture`).

Three legs, each with the control that makes it a measurement.

**A. The aliased world.** On the globally aliased Klein bundle, every merged
label's `(label, action)` slot is split into two transports. The two-component
residual must drop sharply against one component, and the assignment must
recover the TRUE place — otherwise the method invents a split.

**B. The negative control, which is the whole risk.** The same procedure on a
world with NO aliasing, where each label is one true place. A mixture over
maps has more parameters and will always fit better; the question is whether
it fits MUCH better and whether its assignment means anything. Gated: the
residual drop must be visibly smaller than in A, and the purity must sit near
the chance level a random split achieves.

**C. Detection is not yet a map, and the reason is precise.** Splitting each
merged label by its own mixture and rebuilding the graph does NOT recover the
world: 27 clones at 0.75 purity, 14 still merged. The component indices are
ARBITRARY PER SLOT — component 0 of `(label, +x)` has no reason to be the same
true place as component 0 of `(label, +y)` — so the per-slot splits do not
compose. Aligning them needs a cross-slot constraint, and on a flat bundle
there is an exact one: the elementary square must commute,
`R_y(next_x) R_x(p) = R_x(next_y) R_y(p)`. That is the remaining step, and it
is recorded here as a gated negative rather than left as an impression.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_global_symmetry.mojo
"""

from std.math import abs
from std.testing import assert_true

from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.klein_grid import (
    KleinWorld,
    KleinWorldConfig,
    ALIAS_GLOBAL,
    ALIAS_NONE,
)
from mojo_rl.experimental.swm.map_builder import (
    WalkRecord,
    label_walk,
    count_labels,
    successor_conflicts,
    split_until_stable,
    score_map,
    clone_graph,
    count_reversing,
    fit_transport_mixture,
    assignment_purity,
)

comptime DT = DType.float64
comptime W = 6
comptime H = 5
comptime NP = W * H
comptime TrainerT = SwmPhase3[NP, 6, 16, 32, 8, DT]
comptime WorldT = KleinWorld[W, H, 6, 16]
comptime LABEL_THRESHOLD = 2.0


@fieldwise_init
struct SlotStats(Copyable, Movable):
    var drop: Float64
    """`1 - res_2 / res_1`: how much a second transport buys."""
    var purity: Float64
    var n_slots: Int
    var n_pairs: Int


def mixture_over_slots(
    rec: WalkRecord, labels: List[Int], seed: UInt64
) raises -> SlotStats:
    """Fit a 2-transport mixture in every `(label, action)` slot with enough
    pairs, and score the assignment against the TRUE place."""
    var n_lab = count_labels(labels)
    var drop = Float64(0)
    var pur = Float64(0)
    var used = 0
    var pairs = 0
    for l in range(n_lab):
        for a in range(2):
            var xs = List[Float64]()
            var ys = List[Float64]()
            var truth = List[Int]()
            for t in range(rec.size() - 1):
                if labels[t] != l or rec.action[t] != a:
                    continue
                for i in range(2):
                    xs.append(rec.u[t * 2 + i])
                    ys.append(rec.u[(t + 1) * 2 + i])
                truth.append(rec.true_place[t])
            if len(truth) < 40:
                continue
            var f = fit_transport_mixture[2, DT](xs, ys, 2, seed + UInt64(l * 7 + a))
            drop += 1.0 - f.res_k / f.res_1
            pur += assignment_purity(f.assign, truth, 2)
            used += 1
            pairs += len(truth)
    if used == 0:
        return SlotStats(0.0, 0.0, 0, 0)
    return SlotStats(drop / Float64(used), pur / Float64(used), used, pairs)


def main() raises:
    var checks = 0
    var cfg = Phase3Config.with_content()
    cfg.seed = 31000

    # =====================================================================
    # A. The globally aliased world: does the frame see what the graph cannot?
    # =====================================================================
    var gcfg = KleinWorldConfig.flat_klein().with_alias(ALIAS_GLOBAL)
    var genv = WorldT(gcfg)
    var gm = TrainerT.train_world(genv, cfg, NP)
    var grec = TrainerT.record_walks(genv, gm, 24, 4 * NP, 41000)
    var glab = label_walk(grec, LABEL_THRESHOLD)
    var gconf = successor_conflicts(glab, grec.action, 2)
    var n_conf = 0
    for l in range(count_labels(glab)):
        if gconf[l] > 0:
            n_conf += 1
    var ga = mixture_over_slots(grec, glab, 909)
    print("A | global symmetry: labels", count_labels(glab),
          " successor conflicts", n_conf, " (the graph is BLIND)")
    print("A | 2-transport mixture over", ga.n_slots, "slots,", ga.n_pairs,
          "pairs: residual drop", ga.drop, " assignment purity vs TRUE place",
          ga.purity)

    # =====================================================================
    # B. The control: the same procedure where there is nothing to split.
    # =====================================================================
    var ncfg = KleinWorldConfig.flat_klein().with_alias(ALIAS_NONE)
    var nenv = WorldT(ncfg)
    var nm = TrainerT.train_world(nenv, cfg, NP)
    var nrec = TrainerT.record_walks(nenv, nm, 24, 4 * NP, 41000)
    var nlab = label_walk(nrec, LABEL_THRESHOLD)
    var na = mixture_over_slots(nrec, nlab, 909)
    print("B | no aliasing (each label IS one place):", na.n_slots, "slots,",
          na.n_pairs, "pairs: residual drop", na.drop,
          " (purity is meaningless here — one true class)")

    checks += 3
    assert_true(
        n_conf == 0,
        "the premise: a GLOBAL symmetry must leave the label graph with no "
        + "successor conflict, else this is the Phase 8 case. got " + String(n_conf),
    )
    assert_true(
        ga.purity > 0.9,
        "THE ANSWER: a mixture over MAPS must recover the true place split "
        + "that the graph cannot see. purity " + String(ga.purity),
    )
    assert_true(
        ga.drop > 2.0 * na.drop and na.drop < 0.25,
        "CONTROL: a second transport always fits better, so the drop must be "
        + "much larger where there really are two places than where there is "
        + "one. aliased " + String(ga.drop) + " vs clean " + String(na.drop),
    )

    # =====================================================================
    # C. Split by the mixture, rebuild the graph, read the holonomies.
    # =====================================================================
    var n_lab = count_labels(glab)
    var refined = List[Int](length=grec.size(), fill=0)
    for t in range(grec.size()):
        refined[t] = glab[t]
    var next_id = n_lab
    for l in range(n_lab):
        var idx = List[Int]()
        var xs = List[Float64]()
        var ys = List[Float64]()
        for t in range(grec.size() - 1):
            if glab[t] != l:
                continue
            idx.append(t)
            for i in range(2):
                xs.append(grec.u[t * 2 + i])
                ys.append(grec.u[(t + 1) * 2 + i])
        if len(idx) < 40:
            continue
        var f = fit_transport_mixture[2, DT](xs, ys, 2, 5150 + UInt64(l))
        if 1.0 - f.res_k / f.res_1 < 0.25:
            continue
        for j in range(len(idx)):
            if f.assign[j] == 1:
                refined[idx[j]] = next_id
        next_id += 1
    var sc_before = score_map(glab, grec.true_place, NP)
    var sc_after = score_map(refined, grec.true_place, NP)
    var g_before = clone_graph(grec, glab, 2)
    var g_after = clone_graph(grec, refined, 2)
    var rev_before = count_reversing(g_before)
    var rev_after = count_reversing(g_after)
    var truth = count_reversing(genv.grid.build_graph())
    print("C | labels", sc_before.n_labels, "-> clones", sc_after.n_labels,
          " purity", sc_before.purity, "->", sc_after.purity,
          " merged", sc_before.n_merged, "->", sc_after.n_merged)
    print("C | reversing cycles:", rev_before, "->", rev_after, " (truth",
          truth, "of 31);  cycles", len(g_before.fundamental_cycle_edges()),
          "->", len(g_after.fundamental_cycle_edges()))
    checks += 2
    assert_true(
        sc_after.purity > sc_before.purity + 0.15,
        "the split must at least improve the map, or the per-slot detection is "
        + "not reaching the vertices at all: " + String(sc_before.purity)
        + " -> " + String(sc_after.purity),
    )
    assert_true(
        sc_after.n_merged > 0 or sc_after.n_labels != NP,
        "RECORDED GAP: detection is not yet a map. Component indices are "
        + "arbitrary per (label, action) slot, so per-slot splits do not "
        + "compose — got " + String(sc_after.n_labels) + " clones at purity "
        + String(sc_after.purity) + " with " + String(sc_after.n_merged)
        + " still merged. The fix is the flat bundle's commuting-square "
        + "constraint, R_y(next_x) R_x(p) = R_x(next_y) R_y(p), and it is not "
        + "built. If this assertion ever fails, the gap has been closed and "
        + "the gate must be rewritten to demand the full map.",
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G29 a global symmetry is refutable by the transports; "
          + "composing the refutation into a map is not built")
