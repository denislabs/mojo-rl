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
