"""G19 — SWM Phase 7: identifications in two dimensions, no oracle in the loop.

G18 narrowed 6c: a false identification cannot manufacture a reflection the
world lacks; it mis-attributes one the world has, and the clique removes most
false closures except those whose spurious holonomy coincides with `I` or
`M`. Phase 7's plan predicted the second dimension would eliminate those
survivors. This gate asks — and the answer depends on a fact the plan did not
state: whether the bundle is FLAT.

**The world.** A `W x H` torus grid of places carrying an O(2) frame bundle
with one reversing seam (6b's construction), now in a FLAT variant: the seam is
inserted as a deck transformation, so every elementary square is trivial and
the transport between two places depends only on the homotopy class of the
path. In root gauge every loop holonomy then lies in `{I, M}`. 6b's original
bundle is curved along the seam (8 of its 31 fundamental cycles are non-trivial
rotations) and serves as the control on that assumption. Transports are
RECOVERED by Procrustes from noisy frame observations along random walks, as in
6b; no learned encoder (the frame bundle is the object under test, and 6b's
scoping note applies).

**The recogniser is modelled, not learned.** Each place has a texture label;
under aliasing two places share one. At every step the walker proposes a
closure to its most recent visit of every place carrying its current label:
one true, and one false when aliased. G18 showed why a learned content channel
cannot be used for this leg on an orientable world (it absorbs frame
information and disambiguates on its own), and the claims below are about the
holonomy of the WALK a closure spans and about the GRAPH, neither of which
depends on how the closure was proposed.

Three legs.

**A. The Z/2 clique is exact on a flat bundle and wrong on a curved one.** On
the flat Klein bundle the clique keeps >= 95 % of true closures with the
bootstrapped `M` matching the seam; on 6b's curved bundle it visibly REJECTS
true closures, because their walk holonomies include rotations that are not in
`{I, M}`. That is a stated limitation, gated so it cannot be forgotten: the
clique assumes the holonomy group is `Z/2`, which is a statement about
flatness, and a world with genuine curvature needs a different rule.

**B. Gauge coincidence is NOT resolved by the second dimension.** On a flat
bundle a false closure's holonomy is the transport between the two aliased
places, which is the same along every homotopic path and differs by `M` across
classes — so if it coincides with `{I, M}` in one class it does in all. The
surviving false fraction is reported beside the ring's (G18: 10/95) and the
plan's prediction is recorded as refuted or not by the number. The orientable
flat torus with the same modelled confusion must read `det = -1` on zero
closures — the control that a false closure never invents a reflection.

**C. What DOES refute a false identification: the graph.** A merged label with
LOCAL aliasing (one pair of places, their neighbours distinct) has two
different successor labels under one action — a contradiction a deterministic
transition model can see, and the signal a clone-splitting rule (CSCG) runs on.
Under GLOBAL aliasing (`(x, y) ~ (x + W/2, y)`, a symmetry of the texture map)
there is no such conflict: the quotient is a consistent world, and only the
frame transports disagree (G18 leg C). Gated as counts: local aliasing ->
every merged label conflicts; global and none -> zero conflicts.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_klein_recognition.mojo
"""

from std.math import abs, cos, sin
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.rng import Rng
from mojo_rl.experimental.swm.procrustes import procrustes_o_d, PairBatch
from mojo_rl.experimental.swm.envs.klein_grid import KleinGrid, ACT_X, ACT_Y
from mojo_rl.experimental.swm.observables import maximal_clique_z2, Z2Clique

comptime DT = DType.float64
comptime W = 6
comptime H = 5
comptime NP = W * H
comptime TOL = 0.3
comptime EPISODES = 24
comptime STEPS = 4 * NP
comptime NOISE = 0.02

comptime ALIAS_NONE = 0
comptime ALIAS_LOCAL = 1
comptime ALIAS_GLOBAL = 2


def labels_for(mode: Int) -> List[Int]:
    """Texture label per place. LOCAL: one pair, (1,1) ~ (4,3). GLOBAL:
    (x, y) ~ (x + W/2, y) for every place."""
    var lab = List[Int](length=NP, fill=0)
    for p in range(NP):
        lab[p] = p
    if mode == ALIAS_LOCAL:
        lab[3 * W + 4] = 1 * W + 1
    elif mode == ALIAS_GLOBAL:
        for y in range(H):
            for x in range(W):
                lab[y * W + x] = y * W + (x % (W // 2))
    return lab^


struct WalkClosures(Copyable, Movable):
    var h: List[SqMat[2, DT]]
    var correct: List[Bool]
    var n_true: Int
    var n_false: Int
    var successor_conflicts: Int
    """Labels with more than one successor label under one action."""
    var merged_labels: Int

    def __init__(out self):
        self.h = List[SqMat[2, DT]]()
        self.correct = List[Bool]()
        self.n_true = 0
        self.n_false = 0
        self.successor_conflicts = 0
        self.merged_labels = 0

    def __init__(out self, *, copy: Self):
        self.h = copy.h.copy()
        self.correct = copy.correct.copy()
        self.n_true = copy.n_true
        self.n_false = copy.n_false
        self.successor_conflicts = copy.successor_conflicts
        self.merged_labels = copy.merged_labels

    def __init__(out self, *, deinit move: Self):
        self.h = move.h^
        self.correct = move.correct^
        self.n_true = move.n_true
        self.n_false = move.n_false
        self.successor_conflicts = move.successor_conflicts
        self.merged_labels = move.merged_labels


def fit_transports(
    world: KleinGrid[W, H], seed: UInt64
) raises -> List[SqMat[2, DT]]:
    """Per-edge O(2) fits from noisy frame pairs (index `2p` = +x, `2p+1` = +y)."""
    var batches = world.rollout_pairs(48, 6 * NP, NOISE, seed)
    var out = List[SqMat[2, DT]]()
    for e in range(2 * NP):
        out.append(procrustes_o_d[2, DT](batches[e]))
    return out^


def walk(
    fit: List[SqMat[2, DT]], lab: List[Int], seed: UInt64
) raises -> WalkClosures:
    """Random +x/+y walks; closures proposed by LABEL to the most recent visit
    of every place carrying it; holonomies in root gauge `T_t^T T_s`."""
    var rng = Rng(seed)
    var out = WalkClosures()
    # successor label sets per (label, action): up to 2 stored, conflict if 2.
    var succ_a = List[Int](length=2 * NP, fill=-1)
    var succ_b = List[Int](length=2 * NP, fill=-1)
    for _ in range(EPISODES):
        var x = 0
        var y = 0
        var t_cum = SqMat[2, DT].identity()
        var last_visit_t = List[Int](length=NP, fill=-1)
        var t_at = List[SqMat[2, DT]]()
        for _ in range(STEPS):
            var p = y * W + x
            # propose closures to every place sharing p's label
            for q in range(NP):
                if lab[q] != lab[p] or last_visit_t[q] < 0:
                    continue
                var s = last_visit_t[q]
                out.h.append(t_cum.transpose() * t_at[s])
                var ok = q == p
                out.correct.append(ok)
                if ok:
                    out.n_true += 1
                else:
                    out.n_false += 1
            last_visit_t[p] = len(t_at)
            t_at.append(t_cum.copy())
            # step
            var go_x = rng.uniform() < 0.5
            var e = 2 * p if go_x else 2 * p + 1
            t_cum = fit[e] * t_cum
            var nx = (x + 1) % W if go_x else x
            var ny = y if go_x else (y + 1) % H
            var key = lab[p] * 2 + (0 if go_x else 1)
            var nl = lab[ny * W + nx]
            if succ_a[key] < 0:
                succ_a[key] = nl
            elif succ_a[key] != nl and succ_b[key] < 0:
                succ_b[key] = nl
            x = nx
            y = ny
    var label_count = List[Int](length=NP, fill=0)
    for p in range(NP):
        label_count[lab[p]] += 1
    for l in range(NP):
        if label_count[l] > 1:
            out.merged_labels += 1
    for key in range(2 * NP):
        if succ_b[key] >= 0:
            out.successor_conflicts += 1
    return out^


def seam_reflection(world: KleinGrid[W, H]) -> SqMat[2, DT]:
    """The x-loop holonomy of row 0 based at the origin — the oracle `M`."""
    var m = SqMat[2, DT].identity()
    for x in range(W):
        m = world.x_edge[x] * m
    return m^


def clique_report(
    name: String, c: WalkClosures, oracle_m: SqMat[2, DT]
) raises -> Z2Clique:
    var cl = maximal_clique_z2[2, DT](c.h, TOL, False)
    var t_in = 0
    var f_in = 0
    for i in range(len(cl.members)):
        if c.correct[cl.members[i]]:
            t_in += 1
        else:
            f_in += 1
    var neg_false = 0
    var neg_all = 0
    for i in range(len(c.h)):
        if Float64(c.h[i].det()) < 0:
            neg_all += 1
            if not c.correct[i]:
                neg_false += 1
    var m_err = 2.0
    if cl.reference >= 0:
        var m_ref = SqMat[2, DT]()
        for i in range(2):
            for j in range(2):
                m_ref[i, j] = Scalar[DT](cl.refined_reference[i * 2 + j])
        m_err = Float64((m_ref - oracle_m).frobenius_norm())
    print(name + " | closures", len(c.h), " true", c.n_true, " false",
          c.n_false, "  det=-1: all", neg_all, " false", neg_false)
    print(name + " | clique members", len(cl.members), "  true in", t_in, "/",
          c.n_true, "  false in", f_in, "/", c.n_false, "  |M - M_seam|_F",
          m_err, "  merged labels", c.merged_labels, " successor conflicts",
          c.successor_conflicts)
    return cl^


def true_kept(c: WalkClosures, cl: Z2Clique) -> Float64:
    var t_in = 0
    for i in range(len(cl.members)):
        if c.correct[cl.members[i]]:
            t_in += 1
    return Float64(t_in) / Float64(c.n_true)


def false_kept(c: WalkClosures, cl: Z2Clique) -> Int:
    var f_in = 0
    for i in range(len(cl.members)):
        if not c.correct[cl.members[i]]:
            f_in += 1
    return f_in


def neg_count(c: WalkClosures) -> Int:
    var n = 0
    for i in range(len(c.h)):
        if Float64(c.h[i].det()) < 0:
            n += 1
    return n


def worst_square(wd: KleinGrid[W, H]) -> Float64:
    """Largest elementary-square holonomy, from the PLANTED transports."""
    var worst = Float64(0)
    for y in range(H):
        for x in range(W):
            var p = y * W + x
            var q = y * W + (x + 1) % W
            var r = ((y + 1) % H) * W + x
            var sq = wd.y_edge[q] * wd.x_edge[p]
            sq = wd.x_edge[r].transpose() * sq
            sq = wd.y_edge[p].transpose() * sq
            var d = Float64(sq.dist_to_identity())
            if d > worst:
                worst = d
    return worst


def main() raises:
    var checks = 0

    # ---- worlds ------------------------------------------------------------
    var flat = KleinGrid[W, H](True, 20260904, True)
    var curved = KleinGrid[W, H](True, 20260904, False)
    var torus = KleinGrid[W, H](False, 20260904, True)
    var fit_flat = fit_transports(flat, 11)
    var fit_curved = fit_transports(curved, 11)
    var fit_torus = fit_transports(torus, 11)
    var m_flat = seam_reflection(flat)
    var m_curved = seam_reflection(curved)

    # Flatness itself, measured: every elementary square trivial on the flat
    # bundle; a seam square NON-trivial on the curved one.
    var worst_sq_flat = worst_square(flat)
    var worst_sq_curved = worst_square(curved)
    print("flatness | worst square holonomy: flat", worst_sq_flat, "  curved",
          worst_sq_curved)
    checks += 2
    assert_true(worst_sq_flat < 1e-9, "the flat bundle must have trivial squares")
    assert_true(
        worst_sq_curved > 0.1,
        "CONTROL: 6b's bundle must be visibly curved at the seam, got "
        + String(worst_sq_curved),
    )

    # ---- A. the clique on flat vs curved, no aliasing ----------------------
    var lab_none = labels_for(ALIAS_NONE)
    var c_flat = walk(fit_flat, lab_none, 7)
    var c_curved = walk(fit_curved, lab_none, 7)
    var cl_flat = clique_report("A flat  ", c_flat, m_flat)
    var cl_curved = clique_report("A curved", c_curved, m_curved)
    checks += 3
    assert_true(
        true_kept(c_flat, cl_flat) >= 0.95,
        "on a FLAT bundle every walk holonomy is in {I, M}, so the Z/2 clique "
        + "must keep >= 95% of true closures: " + String(true_kept(c_flat, cl_flat)),
    )
    var m_ref = SqMat[2, DT]()
    for i in range(2):
        for j in range(2):
            m_ref[i, j] = Scalar[DT](cl_flat.refined_reference[i * 2 + j])
    assert_true(
        Float64((m_ref - m_flat).frobenius_norm()) < 0.1,
        "the bootstrapped M must be the seam reflection",
    )
    assert_true(
        true_kept(c_curved, cl_curved) < 0.85,
        "LIMITATION, gated: on a CURVED bundle the Z/2 clique must visibly "
        + "reject true closures (walk holonomies include rotations). kept "
        + String(true_kept(c_curved, cl_curved)),
    )

    # ---- B. gauge coincidence under GLOBAL aliasing, and the orientable control
    var lab_glob = labels_for(ALIAS_GLOBAL)
    var c_glob = walk(fit_flat, lab_glob, 7)
    var cl_glob = clique_report("B global", c_glob, m_flat)
    var c_tor = walk(fit_torus, lab_glob, 7)
    var cl_tor = clique_report("B torus ", c_tor, SqMat[2, DT].identity())
    var surv = false_kept(c_glob, cl_glob)
    var surv_frac = Float64(surv) / Float64(c_glob.n_false)
    print("B | false closures surviving the clique:", surv, "/", c_glob.n_false,
          "=", surv_frac, "  (ring, G18: 10/95 = 0.105)")
    checks += 4
    assert_true(
        c_glob.n_false > 100 and c_tor.n_false > 100,
        "aliasing must offer false closures in both worlds",
    )
    assert_true(
        true_kept(c_glob, cl_glob) >= 0.95,
        "aliasing must not cost true closures: " + String(true_kept(c_glob, cl_glob)),
    )
    assert_true(
        neg_count(c_tor) == 0,
        "on the orientable torus no walk carries a reflection, so zero "
        + "closures may read det = -1: got " + String(neg_count(c_tor)),
    )
    assert_true(
        surv > 0,
        "the surviving gauge-coincident false closures must be SHOWN to exist "
        + "in 2D, else the plan's 'eliminated in 2D' would stand unmeasured. "
        + "got " + String(surv),
    )

    # ---- C. successor conflicts: the graph refutes LOCAL aliasing ------------
    var lab_loc = labels_for(ALIAS_LOCAL)
    var c_loc = walk(fit_flat, lab_loc, 7)
    var cl_loc = clique_report("C local ", c_loc, m_flat)
    print("C | successor conflicts: none", c_flat.successor_conflicts, " local",
          c_loc.successor_conflicts, " global", c_glob.successor_conflicts,
          "   merged labels: none", c_flat.merged_labels, " local",
          c_loc.merged_labels, " global", c_glob.merged_labels)
    checks += 3
    assert_true(
        c_flat.successor_conflicts == 0 and c_glob.successor_conflicts == 0,
        "no conflicts without aliasing, and none under a GLOBAL symmetry (the "
        + "quotient is a consistent world): " + String(c_flat.successor_conflicts)
        + ", " + String(c_glob.successor_conflicts),
    )
    assert_true(
        c_loc.merged_labels == 1 and c_loc.successor_conflicts >= 1,
        "LOCAL aliasing must be refuted by the graph: the merged label needs a "
        + "successor conflict under at least one action, got "
        + String(c_loc.successor_conflicts),
    )
    assert_true(
        c_loc.n_false > 20 and false_kept(c_loc, cl_loc) < c_loc.n_false,
        "local aliasing must offer false closures and the clique must reject "
        + "some (the rest is the graph's job)",
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G19 the Z/2 clique is exact on a flat bundle, 2D does not "
          + "resolve gauge coincidence, and the graph refutes local aliasing")
