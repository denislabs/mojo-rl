"""Planning in EDGE COORDINATES: rollouts transport the frame, they do not infer.

A rollout applies `u <- R_e u` for each edge crossed, with the per-edge transport
already learned. The inference of §4.3 runs ONCE per real step and never inside
the planner, which is what removes the design doc's own cost objection to v1
(`K * horizon * candidates`): the cost per rollout step is `D^2`, not an
inference.

**The monodromy must not be applied twice.** The reflection at the seam is
already inside the edge transports, so a full lap accumulates it by construction.
A planner that additionally multiplied by the cycle's monodromy on closing the
loop would compute `H^2 = I` and predict no reflection at all — exactly wrong,
and wrong in the direction that looks like success at even parity. The
`apply_monodromy_twice` flag exists so the G14 gate can measure that this is a
real error rather than a stylistic note.

**Backward moves need no extra learning.** The transports are orthogonal, so
crossing an edge in reverse is `R^T = R^-1` exactly. The table is trained on
forward transitions only and the reverse direction comes free — a property of
the O(D) restriction, not an implementation shortcut.

**The task is goal-conditioned, and that is what makes it gauge-free.** The
encoder learns some basis `A`, so "the landmark appears on the left" is a
direction the planner has no way to name. Handing it the GOAL's frame encoding
instead ("reach a state that looks like this") is well posed in any gauge:
matching `u` means matching `F_{cell,parity}`, which is precisely reaching the
right cell at the right PARITY. Arriving at the correct cell on the wrong lap
gives the mirrored frame and does not match.
"""

from std.math import abs, sqrt

from .so_d import SqMat
from .rng import Rng

comptime MODEL_ORTHOGONAL: Int = 0
comptime MODEL_TRANSLATION: Int = 1
comptime MODEL_PLACE_LOOKUP: Int = 2

comptime PLAN_FORWARD: Int = 0
comptime PLAN_BACKWARD: Int = 1


@fieldwise_init
struct PlannerConfig(Copyable, ImplicitlyCopyable, Movable):
    var horizon: Int
    var candidates: Int
    var iterations: Int
    var elite_frac: Float64
    var step_penalty: Float64
    """Prefers reaching the goal sooner; makes the plan a path, not a wander."""
    var trust_lambda: Float64
    """Penalty on crossing a low-confidence edge (v2 §4.7): do not build a plan
    on an identification you do not believe."""
    var explore_bonus: Float64
    """Intrinsic reward for closing a cycle whose holonomy is not yet confirmed
    — exploration directed by H^1."""

    @staticmethod
    def default() -> Self:
        return Self(26, 96, 4, 0.15, 0.01, 1.0, 0.0)


struct FrameModel[N_CELLS: Int, dtype: DType = DType.float64](
    Copyable, Movable
):
    """A one-step frame predictor on the ring, in one of three flavours.

    ORTHOGONAL is SWM-H. TRANSLATION is the constant sheaf (ablation B): a
    single global frame whose transitions are offsets — it cannot represent the
    seam. PLACE_LOOKUP is the aliasing baseline: the frame is a function of the
    CELL alone, so the two lap parities are indistinguishable to it by
    construction; it is what a model without the double cover is limited to.
    """

    var kind: Int
    var transports: List[SqMat[2, Self.dtype]]
    var translations: List[Float64]
    var lookup: List[Float64]
    var edge_w: List[Float64]
    var monodromy: SqMat[2, Self.dtype]
    var apply_monodromy_twice: Bool
    var unconfirmed_cycle: Bool
    """Whether the ring's cycle still lacks a confirmed holonomy reading."""

    def __init__(
        out self,
        kind: Int,
        var transports: List[SqMat[2, Self.dtype]],
        var translations: List[Float64],
        var lookup: List[Float64],
    ) raises:
        self.kind = kind
        self.transports = transports^
        self.translations = translations^
        self.lookup = lookup^
        self.edge_w = List[Float64](length=Self.N_CELLS, fill=1.0)
        self.monodromy = SqMat[2, Self.dtype].identity()
        self.apply_monodromy_twice = False
        self.unconfirmed_cycle = False
        if self.kind == MODEL_ORTHOGONAL:
            var h = SqMat[2, Self.dtype].identity()
            for i in range(Self.N_CELLS):
                h = self.transports[i] * h
            self.monodromy = h^

    def __init__(out self, *, copy: Self):
        self.kind = copy.kind
        self.transports = copy.transports.copy()
        self.translations = copy.translations.copy()
        self.lookup = copy.lookup.copy()
        self.edge_w = copy.edge_w.copy()
        self.monodromy = copy.monodromy.copy()
        self.apply_monodromy_twice = copy.apply_monodromy_twice
        self.unconfirmed_cycle = copy.unconfirmed_cycle

    def __init__(out self, *, deinit move: Self):
        self.kind = move.kind
        self.transports = move.transports^
        self.translations = move.translations^
        self.lookup = move.lookup^
        self.edge_w = move.edge_w^
        self.monodromy = move.monodromy^
        self.apply_monodromy_twice = move.apply_monodromy_twice
        self.unconfirmed_cycle = move.unconfirmed_cycle

    def edge_of(self, cell: Int, action: Int) -> Int:
        if action == PLAN_FORWARD:
            return cell
        return (cell - 1 + Self.N_CELLS) % Self.N_CELLS

    def next_cell(self, cell: Int, action: Int) -> Int:
        if action == PLAN_FORWARD:
            return (cell + 1) % Self.N_CELLS
        return (cell - 1 + Self.N_CELLS) % Self.N_CELLS

    def step(
        mut self, u: List[Float64], cell: Int, action: Int
    ) -> List[Float64]:
        """One imagined move. `D^2` for the frame channel — no inference."""
        var e = self.edge_of(cell, action)
        var out = List[Float64](length=2, fill=0)
        if self.kind == MODEL_ORTHOGONAL:
            var r = self.transports[e]
            for i in range(2):
                var s = Float64(0)
                for j in range(2):
                    # Reverse traversal is R^T, exact because R is orthogonal.
                    if action == PLAN_FORWARD:
                        s += Float64(r[i, j]) * u[j]
                    else:
                        s += Float64(r[j, i]) * u[j]
                out[i] = s
            if self.apply_monodromy_twice and e == Self.N_CELLS - 1:
                var v = List[Float64](length=2, fill=0)
                for i in range(2):
                    var s = Float64(0)
                    for j in range(2):
                        s += Float64(self.monodromy[i, j]) * out[j]
                    v[i] = s
                out = v^
        elif self.kind == MODEL_TRANSLATION:
            for i in range(2):
                if action == PLAN_FORWARD:
                    out[i] = u[i] + self.translations[e * 2 + i]
                else:
                    out[i] = u[i] - self.translations[e * 2 + i]
        else:
            var nc = self.next_cell(cell, action)
            for i in range(2):
                out[i] = self.lookup[nc * 2 + i]
        return out^


struct Plan(Copyable, Movable):
    """A plan that COMMITS to an arrival time, not just a direction.

    "Did the agent ever pass through the goal?" is nearly free on a small ring
    with a long horizon — a wandering policy scores well (measured: the
    parity-blind translation model hit 14/14 parity-1 goals that way, purely
    because backward moves cross the seam). Requiring the planner to say WHEN it
    will be there makes the task need the parity, which is the point.
    """

    var actions: List[Int]
    var arrival: Int
    var predicted_cost: Float64

    def __init__(
        out self, var actions: List[Int], arrival: Int, predicted_cost: Float64
    ):
        self.actions = actions^
        self.arrival = arrival
        self.predicted_cost = predicted_cost

    def __init__(out self, *, copy: Self):
        self.actions = copy.actions.copy()
        self.arrival = copy.arrival
        self.predicted_cost = copy.predicted_cost

    def __init__(out self, *, deinit move: Self):
        self.actions = move.actions^
        self.arrival = move.arrival
        self.predicted_cost = move.predicted_cost


def _dist2(a: List[Float64], b: List[Float64]) -> Float64:
    var d = Float64(0)
    for i in range(len(a)):
        var e = a[i] - b[i]
        d += e * e
    return d


def score_plan[
    N_CELLS: Int, dtype: DType = DType.float64
](
    mut model: FrameModel[N_CELLS, dtype],
    u0: List[Float64],
    cell0: Int,
    u_goal: List[Float64],
    actions: List[Int],
    cfg: PlannerConfig,
    mut arrival: Int,
) -> Float64:
    """Lower is better: closest approach to the goal frame, plus the costs.

    Writes the step at which the model BELIEVES it is closest into `arrival`.

    Confidence enters as a penalty on crossing a doubtful edge, and an
    unconfirmed cycle earns a bonus for being closed — the two ways the
    observables reach the planner (v2 §4.7).
    """
    var u = u0.copy()
    var cell = cell0
    var best = _dist2(u, u_goal)
    arrival = 0
    var cost = Float64(0)
    var crossed_seam = False
    for k in range(len(actions)):
        var a = actions[k]
        var e = model.edge_of(cell, a)
        cost += cfg.trust_lambda * (1.0 - model.edge_w[e])
        if e == N_CELLS - 1:
            crossed_seam = True
        u = model.step(u, cell, a)
        cell = model.next_cell(cell, a)
        var d = _dist2(u, u_goal) + cfg.step_penalty * Float64(k + 1)
        if d < best:
            best = d
            arrival = k + 1
    if model.unconfirmed_cycle and crossed_seam:
        cost -= cfg.explore_bonus
    return best + cost


def plan_exhaustive[
    N_CELLS: Int, dtype: DType = DType.float64
](
    mut model: FrameModel[N_CELLS, dtype],
    u0: List[Float64],
    cell0: Int,
    u_goal: List[Float64],
    cfg: PlannerConfig,
) -> Plan:
    """Optimal plan over MONOTONE walks, by exhaustive search.

    On a ring, walking forward `k` steps for `k` in `[0, 2N)` reaches every
    state of the double cover exactly once, so the optimum over monotone plans
    is a scan of `4N` rollouts. Cheap, and exact.

    This exists to separate MODEL error from SEARCH error. The CEM planner
    scores by closest approach anywhere along a trajectory, which invites
    spurious matches, and its step penalty is a path-length prior that trades
    near goals against far ones — measured: penalty 0 gives 14/14 on far
    (parity-1) goals and 16/26 on near ones, penalty 0.01 gives 11/14 and 22/26.
    Those are properties of the search, not of the world model, and a gate on
    the model should not be reading them.
    """
    var best_score = 1e300
    var best_dir = PLAN_FORWARD
    var best_k = 0
    for direction in range(2):
        var u = u0.copy()
        var cell = cell0
        var cost = Float64(0)
        var d0 = _dist2(u, u_goal)
        if d0 < best_score:
            best_score = d0
            best_dir = direction
            best_k = 0
        for k in range(2 * N_CELLS):
            var e = model.edge_of(cell, direction)
            cost += cfg.trust_lambda * (1.0 - model.edge_w[e])
            u = model.step(u, cell, direction)
            cell = model.next_cell(cell, direction)
            var d = _dist2(u, u_goal) + cost
            if d < best_score:
                best_score = d
                best_dir = direction
                best_k = k + 1
    var actions = List[Int](length=best_k, fill=best_dir)
    return Plan(actions^, best_k, best_score)


def plan[
    N_CELLS: Int, dtype: DType = DType.float64
](
    mut model: FrameModel[N_CELLS, dtype],
    u0: List[Float64],
    cell0: Int,
    u_goal: List[Float64],
    cfg: PlannerConfig,
    mut rng: Rng,
) -> Plan:
    """Discrete CEM over binary action sequences.

    Two actions, so the sampling distribution is one Bernoulli per timestep and
    the "refit" is just the elite mean. Simple on purpose: the object under test
    is the world model, and a cleverer optimiser would only blur which of the
    two is responsible for a failure.
    """
    var probs = List[Float64](length=cfg.horizon, fill=0.5)
    var n_elite = Int(Float64(cfg.candidates) * cfg.elite_frac)
    if n_elite < 2:
        n_elite = 2
    var best_seq = List[Int](length=cfg.horizon, fill=PLAN_FORWARD)
    var best_score = 1e300
    var best_arrival = 0

    for _ in range(cfg.iterations):
        var seqs = List[Int](length=cfg.candidates * cfg.horizon, fill=0)
        var scores = List[Float64](length=cfg.candidates, fill=0)
        for c in range(cfg.candidates):
            var actions = List[Int](length=cfg.horizon, fill=0)
            for k in range(cfg.horizon):
                var a = (
                    PLAN_FORWARD if rng.uniform() < probs[k] else PLAN_BACKWARD
                )
                actions[k] = a
                seqs[c * cfg.horizon + k] = a
            var arr = 0
            var s = score_plan[N_CELLS, dtype](
                model, u0, cell0, u_goal, actions, cfg, arr
            )
            scores[c] = s
            if s < best_score:
                best_score = s
                best_arrival = arr
                best_seq = actions^
        # Elites: indices of the n_elite smallest scores.
        var order = List[Int](length=cfg.candidates, fill=0)
        for i in range(cfg.candidates):
            order[i] = i
        for i in range(1, cfg.candidates):
            var v = order[i]
            var j = i - 1
            while j >= 0 and scores[order[j]] > scores[v]:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = v
        for k in range(cfg.horizon):
            var acc = Float64(0)
            for i in range(n_elite):
                if seqs[order[i] * cfg.horizon + k] == PLAN_FORWARD:
                    acc += 1.0
            # Keep it off the boundary so later iterations can still explore.
            probs[k] = 0.1 + 0.8 * (acc / Float64(n_elite))
    return Plan(best_seq^, best_arrival, best_score)
