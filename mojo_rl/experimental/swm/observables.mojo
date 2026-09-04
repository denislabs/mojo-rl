"""Reading the graph: residuals, confidence, and the classification table.

Three families of quantity, and NONE of them enters a loss
(docs/SHEAF_WORLD_MODELS_V2.md §4.4).

**The residual is measured before consensus.** On a frustrated cycle the
inference redistributes the disagreement — onto the identification edge when the
anchors are strong, across every edge of the cycle when they are weak (measured
in the G11 gate) — so a residual read AFTER inference has had its premise
destroyed by the act of measuring. `procrustes.mean_squared_residual` on
observed pairs is the honest reading.

**The classification table (§1.2):**

|                        | det H = -1        | det H = +1, angle > tol | trivial |
|------------------------|-------------------|-------------------------|---------|
| residuals nominal      | OBSTRUCTION       | UNDECIDED / confirmed   | NOMINAL |
| one residual high      | ABERRANT          | ABERRANT                | ABERRANT|

The asymmetry is deliberate. `det H = -1` with nominal residuals is a fact about
the world, to be recorded and handed to the planner, NOT corrected. A continuous
holonomy is not: a constant sensor bias produces one that is perfectly coherent
and indistinguishable from real curvature by a single cycle (control D'), so it
stays UNDECIDED until edge-disjoint cycles agree.
"""

from std.math import abs, sqrt, atan2

from .so_d import SqMat, fixed_subspace_dim
from .procrustes import polar_orthogonal_factor

comptime CLASS_NOMINAL: UInt8 = 0
comptime CLASS_ABERRANT: UInt8 = 1
comptime CLASS_OBSTRUCTION: UInt8 = 2
comptime CLASS_UNDECIDED: UInt8 = 3
comptime CLASS_CURVATURE_CONFIRMED: UInt8 = 4


def class_name(c: UInt8) -> String:
    if c == CLASS_NOMINAL:
        return "NOMINAL"
    if c == CLASS_ABERRANT:
        return "ABERRANT"
    if c == CLASS_OBSTRUCTION:
        return "OBSTRUCTION"
    if c == CLASS_UNDECIDED:
        return "UNDECIDED"
    return "CURVATURE_CONFIRMED"


def gnc_weight(r: Float64, mu: Float64, c_bar: Float64) -> Float64:
    """Geman-McClure confidence. `r` is the residual NORM, not its square.

    `w = (mu c^2 / (r^2 + mu c^2))^2`, so `w -> 1` for a nominal edge and
    `w -> 0` for an outlier, smoothly.
    """
    var a = mu * c_bar * c_bar
    if a <= 0:
        return 0.0
    var d = r * r + a
    var q = a / d
    return q * q


struct GncSchedule(Copyable, ImplicitlyCopyable, Movable):
    """Graduated non-convexity: `mu` large -> nearly convex, `mu -> 1` -> the
    true Geman-McClure shape.

    The numpy prototype used a FIXED `c_bar` with no schedule, which is why its
    weights sat at ~0.82 everywhere and only the injected-fault edge was
    informative. With the schedule the nominal edges land near 1.
    """

    var mu: Float64
    var factor: Float64

    def __init__(out self, mu0: Float64 = 1e4, factor: Float64 = 1.4):
        self.mu = mu0
        self.factor = factor

    def __init__(out self, *, copy: Self):
        self.mu = copy.mu
        self.factor = copy.factor

    def __init__(out self, *, deinit move: Self):
        self.mu = move.mu
        self.factor = move.factor

    def step(mut self):
        self.mu /= self.factor
        if self.mu < 1.0:
            self.mu = 1.0

    def converged(self) -> Bool:
        return self.mu <= 1.0


def estimate_c_bar(
    residuals: List[Float64], inlier_multiple: Float64 = 10.0
) -> Float64:
    """The INLIER THRESHOLD: the residual scale below which an edge is trusted.

    Not the typical residual. Geman-McClure gives `w = 0.25` exactly at
    `r = c_bar`, so setting `c_bar` to the median would score every nominal
    edge at 0.25 and leave no room to distinguish "fine" from "doubtful"
    (measured, before this was fixed). The prototype's `10 * median` puts a
    median-residual edge at ~0.98 and is kept.

    Median rather than mean: a mean is dragged up by the very outliers this
    exists to expose — on control D the faulty edge is 920x the median.
    """
    var n = len(residuals)
    if n == 0:
        return 1.0
    var v = residuals.copy()
    for i in range(1, n):
        var x = v[i]
        var j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x
    var med = v[n // 2] if n % 2 == 1 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    if med <= 1e-12:
        med = 1e-12
    return inlier_multiple * med


def gnc_weights(
    residuals: List[Float64], sweeps: Int = 40, factor: Float64 = 1.4
) -> List[Float64]:
    """Run the GNC schedule to convergence and return per-edge confidences."""
    var c_bar = estimate_c_bar(residuals)
    var sched = GncSchedule(1e4, factor)
    var w = List[Float64](length=len(residuals), fill=1.0)
    for _ in range(sweeps):
        for i in range(len(residuals)):
            w[i] = gnc_weight(residuals[i], sched.mu, c_bar)
        sched.step()
    return w^


def classify(
    r_local: Float64,
    r_nominal: Float64,
    det_h: Float64,
    fro_h: Float64,
    angle_tol: Float64,
    confirmed: Bool,
    outlier_factor: Float64 = 10.0,
) -> UInt8:
    """The §1.2 table. An outlier edge dominates: the cycle reading is not
    trustworthy while one of its edges is suspect."""
    if r_local > outlier_factor * r_nominal:
        return CLASS_ABERRANT
    if det_h < 0:
        return CLASS_OBSTRUCTION
    if fro_h >= angle_tol:
        return CLASS_CURVATURE_CONFIRMED if confirmed else CLASS_UNDECIDED
    return CLASS_NOMINAL


def cycles_are_edge_disjoint(
    a: List[Int], b: List[Int]
) -> Bool:
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                return False
    return True


def confirm_by_independent_cycles(
    cycle_edges: List[List[Int]],
    cycle_fro: List[Float64],
    angle_tol: Float64,
) -> Bool:
    """Cross-confirmation in the spirit of PCM (Mangelson et al. 2018).

    A continuous holonomy is CONFIRMED only when two cycles that share no edge
    both report one. A single biased edge cannot explain two edge-disjoint
    cycles; it can explain any number of overlapping ones. Deliberately the weak
    form of PCM — no maximal-clique search — because the decision it feeds is
    binary and the doc's claim is only that one cycle is not enough.
    """
    var n = len(cycle_edges)
    for i in range(n):
        if cycle_fro[i] < angle_tol:
            continue
        for j in range(i + 1, n):
            if cycle_fro[j] < angle_tol:
                continue
            if cycles_are_edge_disjoint(cycle_edges[i], cycle_edges[j]):
                return True
    return False


struct ClassificationLatch(Copyable, ImplicitlyCopyable, Movable):
    """Hysteresis on the verdict, so the closed loop cannot oscillate.

    The classification feeds `in_energy`, which changes the energy, which
    changes what is learned, which changes the holonomy, which changes the
    classification. A bare threshold inside that loop chatters whenever a
    residual sits near the outlier boundary — and each flip re-admits or
    re-removes a constraint, so the chatter is not cosmetic.

    Once a verdict is adopted it is held for `hold_steps` before a different one
    can replace it. It does NOT pin the verdict forever: after the hold expires
    the latch follows the evidence, so a genuinely changed world still gets
    through. `changes` is the metric G12 reads.
    """

    var current: UInt8
    var hold_remaining: Int
    var hold_steps: Int
    var changes: Int
    var updates: Int

    def __init__(out self, hold_steps: Int = 8, initial: UInt8 = CLASS_NOMINAL):
        self.current = initial
        self.hold_remaining = 0
        self.hold_steps = hold_steps
        self.changes = 0
        self.updates = 0

    def __init__(out self, *, copy: Self):
        self.current = copy.current
        self.hold_remaining = copy.hold_remaining
        self.hold_steps = copy.hold_steps
        self.changes = copy.changes
        self.updates = copy.updates

    def __init__(out self, *, deinit move: Self):
        self.current = move.current
        self.hold_remaining = move.hold_remaining
        self.hold_steps = move.hold_steps
        self.changes = move.changes
        self.updates = move.updates

    def update(mut self, proposed: UInt8) -> UInt8:
        self.updates += 1
        if self.hold_remaining > 0:
            self.hold_remaining -= 1
            return self.current
        if proposed != self.current:
            self.current = proposed
            self.changes += 1
            self.hold_remaining = self.hold_steps
        return self.current


def pairwise_consistent_in_group[
    D: Int, dtype: DType = DType.float64
](
    composition: SqMat[D, dtype],
    holonomy_group: List[SqMat[D, dtype]],
    tol: Float64,
) -> Bool:
    """PCM consistency, corrected for a world with NON-TRIVIAL holonomy.

    The textbook criterion (Mangelson et al. 2018) asks whether composing two
    loop closures around the cycle they form returns the IDENTITY. That assumes
    the world admits a global frame. On a Mobius ring it does not: the
    composition of two genuine closures at different lap parities is the
    REFLECTION, and the textbook test throws it out as inconsistent.

    The composition must be formed in ROOT GAUGE — see
    `closure_pair_composition`. Measured on E1 with texture aliasing (G17,
    353 true and 95 false closures), separating true pairs from true/false:

        criterion                       true-true      true-false   false-false
        composition = I  (textbook)      56 %            4.8 %       16 %
        composition in {I, M} (this)    100 %           12.6 %       16.5 %

    The textbook form accepts exactly the same-parity pairs (even-even plus
    odd-odd) and rejects every mixed pair — on a real system that is throwing
    away a large share of the loop closures the map depends on. The right
    question is whether the composition lies in the world's holonomy group,
    which is exactly the object this method already measures. The numbers are
    pinned by G17/G18; an earlier version of this docstring quoted 44 % / 89 %
    from a composition that mixed gauges (see `closure_pair_composition`).
    """
    var best = 1e300
    for g in range(len(holonomy_group)):
        var d = Float64((composition - holonomy_group[g]).frobenius_norm())
        if d < best:
            best = d
    return best <= tol


def closure_pair_composition[
    D: Int, dtype: DType = DType.float64
](h_a: SqMat[D, dtype], h_b: SqMat[D, dtype]) -> SqMat[D, dtype]:
    """The PCM composition of two closures, in root gauge: `H_a H_b^T`.

    PCM composes two loop closures around the 4-cycle they form with the two
    tree paths between their endpoints. In the spanning-tree gauge (`T_p`
    carries the root frame to place `p`, and every holonomy is read as
    `T_dst^T R_e T_src`) the tree is FLAT: transport along a tree path is the
    identity, so the 4-cycle reduces to the product of the two closure
    holonomies. For two genuine closures at lap counts `m_a`, `m_b` on a ring
    with monodromy `M` this is `M^(m_b - m_a)` — in the holonomy group by
    construction.

    The first version of G17 formed `T_ta^T T_tb * T_sb^T T_sa` instead, which
    treats the tree path between the two base places as a transport and mixes
    the gauges of the two closures. Measured: that version accepted 73 % of
    true closure pairs; this one accepts 100 %, with the textbook `= I`
    criterion moving from 47 % to exactly the same-parity pair count (56 %).
    """
    return h_a * h_b.transpose()


struct Z2Clique(Copyable, Movable):
    """The largest set of closures whose holonomies generate a group of order
    at most two — `{I, M}` for one shared reflection `M`.

    `reference` indexes the closure whose holonomy plays `M` (or -1 when the
    clique is identity-only). `members` is sorted ascending, so two cliques
    can be compared element-wise.
    """

    var members: List[Int]
    var reference: Int
    var n_identity: Int
    var n_reflection: Int
    var spread: Float64
    """Mean Frobenius distance of the reflection members to the refined `M`."""
    var refined_reference: List[Float64]
    """Row-major `D x D` refined `M` (empty when identity-only)."""

    def __init__(out self):
        self.members = List[Int]()
        self.reference = -1
        self.n_identity = 0
        self.n_reflection = 0
        self.spread = 0.0
        self.refined_reference = List[Float64]()

    def __init__(out self, *, copy: Self):
        self.members = copy.members.copy()
        self.reference = copy.reference
        self.n_identity = copy.n_identity
        self.n_reflection = copy.n_reflection
        self.spread = copy.spread
        self.refined_reference = copy.refined_reference.copy()

    def __init__(out self, *, deinit move: Self):
        self.members = move.members^
        self.reference = move.reference
        self.n_identity = move.n_identity
        self.n_reflection = move.n_reflection
        self.spread = move.spread
        self.refined_reference = move.refined_reference^

    def contains(self, idx: Int) -> Bool:
        for i in range(len(self.members)):
            if self.members[i] == idx:
                return True
        return False

    def same_members(self, other: Self) -> Bool:
        if len(self.members) != len(other.members):
            return False
        for i in range(len(self.members)):
            if self.members[i] != other.members[i]:
                return False
        return True


def maximal_clique_z2[
    D: Int, dtype: DType = DType.float64
](
    holonomies: List[SqMat[D, dtype]],
    tol: Float64,
    scan_reversed: Bool = False,
) raises -> Z2Clique:
    """Maximal-clique PCM for a world whose holonomy group is `Z/2`.

    The group is BOOTSTRAPPED from the closures rather than supplied: nothing
    here knows the lap length or the true monodromy. Under the root-gauge
    composition, a set of closures is pairwise consistent iff every member's
    holonomy is `I` or one shared reflection `M` — because then every pairwise
    product is `I`, `M` or `M^2 = I`. So the maximal clique is found exactly,
    without a clique search: take each `det = -1` closure in turn as the
    candidate `M`, collect everything within `tol` of `I` or of it, and keep
    the largest set (ties to the tightest candidate). A candidate sitting
    BETWEEN the true reflections and a cluster of gauge-coincident false ones
    can collect the largest set while being off-centre (measured: 0.285 from
    the true monodromy on G18's world), so the reference is then REFINED to
    the orthogonal polar factor of the members' summed holonomies, and the
    members re-collected around it once. `refined_reference` carries the
    result; `reference` still names the seed closure. `scan_reversed` exists
    so a gate can check that the answer does not depend on scan order.

    `det H` is then read from clique MEMBERS only. On a one-action ring with
    texture aliasing the false closures fragment into per-base-cell groups of
    ~16 against a true clique of 353 (G18), so the largest clique is the true
    one; what survives inside it are false closures whose spurious holonomy
    happens to coincide with `I` or `M` — a gauge coincidence no consistency
    test can see, and the reason the 2D gate (G19) exists.
    """
    var n = len(holonomies)
    var ident = SqMat[D, dtype].identity()
    var is_id = List[Bool](length=n, fill=False)
    var is_refl = List[Bool](length=n, fill=False)
    var best = Z2Clique()
    var best_spread = 1e300
    for a in range(n):
        is_id[a] = Float64((holonomies[a] - ident).frobenius_norm()) <= tol
        is_refl[a] = Float64(holonomies[a].det()) < 0 and not is_id[a]
        if is_id[a]:
            best.members.append(a)
            best.n_identity += 1

    for step in range(n):
        var cand = n - 1 - step if scan_reversed else step
        if not is_refl[cand]:
            continue
        var members = List[Int]()
        var n_refl = 0
        var spread = Float64(0)
        for a in range(n):
            if is_id[a]:
                members.append(a)
                continue
            var d = Float64(
                (holonomies[a] - holonomies[cand]).frobenius_norm()
            )
            if d <= tol:
                members.append(a)
                n_refl += 1
                spread += d
        var larger = len(members) > len(best.members)
        var tighter = len(members) == len(best.members) and spread < best_spread
        if larger or tighter:
            best.members = members^
            best.reference = cand
            best.n_reflection = n_refl
            best_spread = spread
    if best.reference < 0:
        return best^

    # Refine: polar factor of the summed reflection members, then re-collect.
    var acc = SqMat[D, dtype]()
    for i in range(len(best.members)):
        var a = best.members[i]
        if not is_id[a]:
            acc = acc + holonomies[a]
    var m_ref = polar_orthogonal_factor[D, dtype](acc)
    var refined = Z2Clique()
    refined.reference = best.reference
    var total = Float64(0)
    for a in range(n):
        if is_id[a]:
            refined.members.append(a)
            refined.n_identity += 1
            continue
        var d = Float64((holonomies[a] - m_ref).frobenius_norm())
        if d <= tol:
            refined.members.append(a)
            refined.n_reflection += 1
            total += d
    if refined.n_reflection > 0:
        refined.spread = total / Float64(refined.n_reflection)
    for i in range(D):
        for j in range(D):
            refined.refined_reference.append(Float64(m_ref[i, j]))
    return refined^


struct CycleVerdict(Copyable, ImplicitlyCopyable, Movable):
    """A cycle's classification WITH the reading `det` alone under-reports.

    `fixed_dim = dim ker(H - I)` says which latent directions still admit a
    global frame around this cycle. In 2D it is redundant with the class (a
    reflection fixes a line); above 2D it is not — an O(3) reflection fixes a
    plane, `-I` fixes nothing, and both are `det = -1` OBSTRUCTIONs (6b, G22).
    """

    var cls: UInt8
    var det: Float64
    var fro: Float64
    var fixed_dim: Int
    var dim: Int

    def __init__(
        out self, cls: UInt8, det: Float64, fro: Float64, fixed_dim: Int, dim: Int
    ):
        self.cls = cls
        self.det = det
        self.fro = fro
        self.fixed_dim = fixed_dim
        self.dim = dim

    def __init__(out self, *, copy: Self):
        self.cls = copy.cls
        self.det = copy.det
        self.fro = copy.fro
        self.fixed_dim = copy.fixed_dim
        self.dim = copy.dim

    def __init__(out self, *, deinit move: Self):
        self.cls = move.cls
        self.det = move.det
        self.fro = move.fro
        self.fixed_dim = move.fixed_dim
        self.dim = move.dim

    def describe(self) -> String:
        return (
            class_name(self.cls) + " det " + String(self.det) + " |H-I| "
            + String(self.fro) + " fixed " + String(self.fixed_dim) + "/"
            + String(self.dim)
        )


def classify_cycle[
    D: Int, dtype: DType = DType.float64
](
    r_local: Float64,
    r_nominal: Float64,
    h: SqMat[D, dtype],
    angle_tol: Float64,
    confirmed: Bool,
    outlier_factor: Float64 = 10.0,
    fixed_tol: Float64 = 1e-6,
) -> CycleVerdict:
    """`classify` applied to the holonomy itself, carrying `dim ker(H - I)`.

    The class is exactly what `classify` returns from `det H` and `|H - I|`;
    nothing about the verdict changes in 2D. What is added is the fixed
    subspace, so that a planner above 2D can be told WHICH directions the
    obstruction leaves globally representable rather than only that one
    exists. `fixed_tol` is a rank tolerance on the LEARNED holonomy, which is
    orthogonal to ~1e-12 but not exactly.
    """
    var det = Float64(h.det())
    var fro = Float64(h.dist_to_identity())
    var cls = classify(r_local, r_nominal, det, fro, angle_tol, confirmed, outlier_factor)
    var fd = fixed_subspace_dim[D, dtype](h, fixed_tol)
    return CycleVerdict(cls, det, fro, fd, D)
