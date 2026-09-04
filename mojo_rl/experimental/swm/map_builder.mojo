"""Oracle-free map building: labels from the content channel, clones from the
graph (Phase 8b).

Every place graph before this file took its vertices from the oracle cell.
Here the vertices come from the agent's own observations, in three steps that
each carry a measurement:

1. **Online labelling.** The content channel `h` of every visit is assigned to
   the nearest stored centroid within a threshold, or opens a new one. Under
   texture aliasing two cells land on one label; that is the input the rest
   exists for.
2. **Successor conflicts.** A label with two different successor labels under
   one action cannot be one place in a deterministic world. G19 measured that
   this — and not the transport residual — is the signal that separates a
   LOCAL false identification from a consistent quotient.
3. **Splitting by context.** Visits of a conflicting label are joined when they
   share a `(action, successor label)` or `(action, predecessor label)` key;
   the connected components are the clones. Iterated, because a split changes
   the keys of the neighbours. This is a Clone-Structured Cognitive Graph's
   clone count read off the graph instead of found by EM.

Then the transports are re-fitted per `(clone, action)` by Procrustes on the
encoded frame pairs, and the holonomies are read on the clone graph exactly as
on an oracle graph. Nothing here reads a true place except the purity
metrics, which score the result.
"""

from std.math import sqrt

from .so_d import SqMat
from .procrustes import PairBatch, procrustes_o_d
from .place_graph import PlaceGraph, Edge
from .rng import Rng


struct WalkRecord(Copyable, Movable):
    """What the agent recorded along its walks: encodings, actions, and the
    truth kept beside them for scoring only."""

    var u: List[Float64]
    """Frame channel, `2` per visit."""
    var h: List[Float64]
    """Content channel, `content_dim` per visit."""
    var action: List[Int]
    """Action taken FROM this visit, `-1` at an episode's last visit."""
    var true_place: List[Int]
    var content_dim: Int

    def __init__(out self, content_dim: Int):
        self.u = List[Float64]()
        self.h = List[Float64]()
        self.action = List[Int]()
        self.true_place = List[Int]()
        self.content_dim = content_dim

    def __init__(out self, *, copy: Self):
        self.u = copy.u.copy()
        self.h = copy.h.copy()
        self.action = copy.action.copy()
        self.true_place = copy.true_place.copy()
        self.content_dim = copy.content_dim

    def __init__(out self, *, deinit move: Self):
        self.u = move.u^
        self.h = move.h^
        self.action = move.action^
        self.true_place = move.true_place^
        self.content_dim = move.content_dim

    def size(self) -> Int:
        return len(self.action)

    def push(
        mut self,
        lat: List[Float64],
        d: Int,
        action: Int,
        true_place: Int,
    ):
        for i in range(d):
            self.u.append(lat[i])
        for i in range(self.content_dim):
            self.h.append(lat[d + i])
        self.action.append(action)
        self.true_place.append(true_place)

    def prefix(self, n_visits: Int) -> Self:
        """The first `n_visits` visits — for sample-efficiency curves."""
        var out = Self(self.content_dim)
        var n = n_visits if n_visits < self.size() else self.size()
        for t in range(n):
            for i in range(2):
                out.u.append(self.u[t * 2 + i])
            for i in range(self.content_dim):
                out.h.append(self.h[t * self.content_dim + i])
            # the cut visit becomes an episode end
            out.action.append(self.action[t] if t < n - 1 else -1)
            out.true_place.append(self.true_place[t])
        return out^


struct OnlineLabeler(Copyable, Movable):
    """Nearest-centroid assignment with a threshold; new centroid otherwise."""

    var dim: Int
    var threshold: Float64
    var centroids: List[Float64]
    var counts: List[Int]

    def __init__(out self, dim: Int, threshold: Float64):
        self.dim = dim
        self.threshold = threshold
        self.centroids = List[Float64]()
        self.counts = List[Int]()

    def __init__(out self, *, copy: Self):
        self.dim = copy.dim
        self.threshold = copy.threshold
        self.centroids = copy.centroids.copy()
        self.counts = copy.counts.copy()

    def __init__(out self, *, deinit move: Self):
        self.dim = move.dim
        self.threshold = move.threshold
        self.centroids = move.centroids^
        self.counts = move.counts^

    def n_labels(self) -> Int:
        return len(self.counts)

    def assign(mut self, h: List[Float64], offset: Int) -> Int:
        var best = -1
        var best_d = self.threshold * self.threshold
        for c in range(len(self.counts)):
            var d = Float64(0)
            for i in range(self.dim):
                var e = h[offset + i] - self.centroids[c * self.dim + i]
                d += e * e
            if d < best_d:
                best_d = d
                best = c
        if best < 0:
            for i in range(self.dim):
                self.centroids.append(h[offset + i])
            self.counts.append(1)
            return len(self.counts) - 1
        var n = Float64(self.counts[best])
        for i in range(self.dim):
            var idx = best * self.dim + i
            self.centroids[idx] += (h[offset + i] - self.centroids[idx]) / (n + 1.0)
        self.counts[best] += 1
        return best


def label_walk(rec: WalkRecord, threshold: Float64) -> List[Int]:
    """One pass of online labelling over the walk's content channel."""
    var lab = OnlineLabeler(rec.content_dim, threshold)
    var out = List[Int](length=rec.size(), fill=0)
    for t in range(rec.size()):
        out[t] = lab.assign(rec.h, t * rec.content_dim)
    return out^


def count_labels(labels: List[Int]) -> Int:
    var m = -1
    for i in range(len(labels)):
        if labels[i] > m:
            m = labels[i]
    return m + 1


def successor_conflicts(
    labels: List[Int], actions: List[Int], n_actions: Int
) -> List[Int]:
    """Per label, the number of `(action)` slots with more than one successor
    label. Zero everywhere means the label graph is a deterministic map."""
    var n = count_labels(labels)
    var first = List[Int](length=n * n_actions, fill=-1)
    var conflict = List[Bool](length=n * n_actions, fill=False)
    for t in range(len(labels) - 1):
        var a = actions[t]
        if a < 0:
            continue
        var key = labels[t] * n_actions + a
        var nxt = labels[t + 1]
        if first[key] < 0:
            first[key] = nxt
        elif first[key] != nxt:
            conflict[key] = True
    var out = List[Int](length=n, fill=0)
    for l in range(n):
        for a in range(n_actions):
            if conflict[l * n_actions + a]:
                out[l] += 1
    return out^


def _find(mut parent: List[Int], x: Int) -> Int:
    var r = x
    while parent[r] != r:
        r = parent[r]
    var c = x
    while parent[c] != r:
        var nxt = parent[c]
        parent[c] = r
        c = nxt
    return r


def _union(mut parent: List[Int], a: Int, b: Int):
    var ra = _find(parent, a)
    var rb = _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def split_by_context(
    labels: List[Int], actions: List[Int], n_actions: Int
) -> List[Int]:
    """One round of clone splitting. Visits of the same label are joined when
    they share a successor key `(action, next label)` or a predecessor key
    `(action, previous label)`; components become clones. Labels without a
    conflict keep a single component (every context agrees), so they are
    untouched. Returns the new per-visit labels, renumbered densely."""
    var n_visits = len(labels)
    var n = count_labels(labels)
    var parent = List[Int](length=n_visits, fill=0)
    for t in range(n_visits):
        parent[t] = t
    # representative visit per (label, key); keys index a flat table
    # key = kind * (n * n_actions) + neighbour_label * n_actions + action
    var n_keys = 2 * n * n_actions
    var rep = List[Int](length=n * n_keys, fill=-1)
    for t in range(n_visits):
        var l = labels[t]
        if t + 1 < n_visits and actions[t] >= 0:
            var k = 0 * (n * n_actions) + labels[t + 1] * n_actions + actions[t]
            var slot = l * n_keys + k
            if rep[slot] < 0:
                rep[slot] = t
            else:
                _union(parent, rep[slot], t)
        if t > 0 and actions[t - 1] >= 0:
            var k = 1 * (n * n_actions) + labels[t - 1] * n_actions + actions[t - 1]
            var slot = l * n_keys + k
            if rep[slot] < 0:
                rep[slot] = t
            else:
                _union(parent, rep[slot], t)
    # visits with no key at all (isolated single-step episodes) join their
    # label's first component
    var first_of_label = List[Int](length=n, fill=-1)
    for t in range(n_visits):
        var l = labels[t]
        if first_of_label[l] < 0:
            first_of_label[l] = t
    var out = List[Int](length=n_visits, fill=-1)
    var root_id = List[Int](length=n_visits, fill=-1)
    var next_id = 0
    for t in range(n_visits):
        var has_key = (t + 1 < n_visits and actions[t] >= 0) or (
            t > 0 and actions[t - 1] >= 0
        )
        var r = _find(parent, t if has_key else first_of_label[labels[t]])
        if root_id[r] < 0:
            root_id[r] = next_id
            next_id += 1
        out[t] = root_id[r]
    return out^


def split_until_stable(
    labels: List[Int], actions: List[Int], n_actions: Int, max_rounds: Int = 4
) -> List[Int]:
    var cur = labels.copy()
    var n_prev = count_labels(cur)
    for _ in range(max_rounds):
        var nxt = split_by_context(cur, actions, n_actions)
        var n_now = count_labels(nxt)
        cur = nxt^
        if n_now == n_prev:
            break
        n_prev = n_now
    return cur^


@fieldwise_init
struct MapScore(Copyable, ImplicitlyCopyable, Movable):
    var n_labels: Int
    var purity: Float64
    """Visit-weighted fraction of each label's visits that belong to its
    majority true place."""
    var n_true_split: Int
    """True places whose visits are spread over more than one label (with at
    least 5 % of their visits elsewhere) — over-splitting."""
    var n_merged: Int
    """Labels whose visits come from more than one true place (>= 5 % each) —
    under-splitting, i.e. aliasing left unresolved."""


def score_map(labels: List[Int], true_place: List[Int], n_true: Int) -> MapScore:
    var n = count_labels(labels)
    var tab = List[Int](length=n * n_true, fill=0)
    var lab_n = List[Int](length=n, fill=0)
    var true_n = List[Int](length=n_true, fill=0)
    for t in range(len(labels)):
        tab[labels[t] * n_true + true_place[t]] += 1
        lab_n[labels[t]] += 1
        true_n[true_place[t]] += 1
    var pure = 0
    var merged = 0
    for l in range(n):
        var best = 0
        var many = 0
        for p in range(n_true):
            var c = tab[l * n_true + p]
            if c > best:
                best = c
            if Float64(c) >= 0.05 * Float64(lab_n[l]) and c > 0:
                many += 1
        pure += best
        if many > 1:
            merged += 1
    var split = 0
    for p in range(n_true):
        var many = 0
        for l in range(n):
            var c = tab[l * n_true + p]
            if Float64(c) >= 0.05 * Float64(true_n[p]) and c > 0:
                many += 1
        if many > 1:
            split += 1
    var purity = Float64(pure) / Float64(len(labels)) if len(labels) > 0 else 0.0
    return MapScore(n, purity, split, merged)


def clone_graph(
    rec: WalkRecord,
    labels: List[Int],
    n_actions: Int,
    min_pairs: Int = 4,
) raises -> PlaceGraph[2, DType.float64]:
    """Places = labels; one edge per `(label, action)` with a majority
    successor and at least `min_pairs` observed transitions; transport =
    Procrustes O(2) fit on the encoded frame pairs of that slot."""
    var n = count_labels(labels)
    var batches = List[PairBatch[2, DType.float64]]()
    for _ in range(n * n_actions):
        batches.append(PairBatch[2, DType.float64]())
    var succ_count = List[Int](length=n * n_actions * n, fill=0)
    for t in range(rec.size() - 1):
        var a = rec.action[t]
        if a < 0:
            continue
        var slot = labels[t] * n_actions + a
        var x = InlineArray[Scalar[DType.float64], 2](fill=0)
        var y = InlineArray[Scalar[DType.float64], 2](fill=0)
        for i in range(2):
            x[i] = Scalar[DType.float64](rec.u[t * 2 + i])
            y[i] = Scalar[DType.float64](rec.u[(t + 1) * 2 + i])
        batches[slot].push(x, y)
        succ_count[slot * n + labels[t + 1]] += 1
    var g = PlaceGraph[2, DType.float64]()
    for _ in range(n):
        _ = g.add_place()
    for l in range(n):
        for a in range(n_actions):
            var slot = l * n_actions + a
            if batches[slot].count() < min_pairs:
                continue
            var best = -1
            var best_c = 0
            for m in range(n):
                if succ_count[slot * n + m] > best_c:
                    best_c = succ_count[slot * n + m]
                    best = m
            if best < 0:
                continue
            var r = procrustes_o_d[2, DType.float64](batches[slot])
            _ = g.add_edge(Edge.action_edge(l, best, a), r)
    g.rebuild_gauge(0)
    return g^


def count_reversing(g: PlaceGraph[2, DType.float64]) raises -> Int:
    var cyc = g.fundamental_cycle_edges()
    var n = 0
    for i in range(len(cyc)):
        if g.holonomy_det(cyc[i]) < 0:
            n += 1
    return n


@fieldwise_init
struct MixtureFit(Copyable, Movable):
    """Result of splitting one `(label, action)` slot into `K` transports."""

    var assign: List[Int]
    var res_1: Float64
    """Mean residual of the single best transport over all pairs."""
    var res_k: Float64
    """...and of the `K`-component fit. The DROP is the signal."""
    var n_pairs: Int
    var counts: List[Int]


def fit_transport_mixture[
    D: Int, dtype: DType = DType.float64
](
    xs: List[Float64],
    ys: List[Float64],
    k: Int,
    seed: UInt64,
    rounds: Int = 25,
    restarts: Int = 4,
) raises -> MixtureFit:
    """Split observed transitions into `k` transports by alternating fit and
    assignment — k-means over `O(D)` maps rather than over points.

    This is the last lead on the GLOBAL-SYMMETRY ambiguity (Phase 8): when a
    texture symmetry merges two places, the label graph has NO successor
    conflict — the quotient is a consistent world — so nothing discrete can
    refute it. Only the frame transports disagree, and G18 measured that the
    residual NORM is not bimodal (worst pair ratio 1.01).

    The norm cannot be bimodal, and that is not a limitation of the
    measurement: the residual of place `a` under a compromise fit `R` is
    `(R_a - R) u`, LINEAR in `u`, so both places' residuals are zero-mean
    ellipses and neither their means nor their magnitudes separate. The signal
    is in the JOINT `(u, epsilon)` — the two places obey different linear
    relations — which is exactly what a mixture over MAPS reads and a mixture
    over points cannot.

    Restarts because the objective is non-convex; the best final residual wins.
    """
    var n = len(xs) // D
    var best = MixtureFit(List[Int](), 0.0, 0.0, n, List[Int]())
    var best_res = 1e300
    var rng = Rng(seed)

    # one-component reference
    var one = PairBatch[D, dtype]()
    for t in range(n):
        var xa = InlineArray[Scalar[dtype], D](fill=0)
        var ya = InlineArray[Scalar[dtype], D](fill=0)
        for i in range(D):
            xa[i] = Scalar[dtype](xs[t * D + i])
            ya[i] = Scalar[dtype](ys[t * D + i])
        one.push(xa, ya)
    var r1 = procrustes_o_d[D, dtype](one)
    var res1 = Float64(0)
    for t in range(n):
        var d = Float64(0)
        for i in range(D):
            var p = Float64(0)
            for j in range(D):
                p += Float64(r1[i, j]) * xs[t * D + j]
            d += (p - ys[t * D + i]) * (p - ys[t * D + i])
        res1 += sqrt(d)
    res1 /= Float64(n)

    for _ in range(restarts):
        var assign = List[Int](length=n, fill=0)
        for t in range(n):
            assign[t] = Int(rng.uniform() * Float64(k)) % k
        var res_k = 1e300
        for _ in range(rounds):
            # fit each component
            var rs = List[SqMat[D, dtype]]()
            for c in range(k):
                var b = PairBatch[D, dtype]()
                for t in range(n):
                    if assign[t] != c:
                        continue
                    var xa = InlineArray[Scalar[dtype], D](fill=0)
                    var ya = InlineArray[Scalar[dtype], D](fill=0)
                    for i in range(D):
                        xa[i] = Scalar[dtype](xs[t * D + i])
                        ya[i] = Scalar[dtype](ys[t * D + i])
                    b.push(xa, ya)
                if b.count() < D + 1:
                    rs.append(r1.copy())
                else:
                    rs.append(procrustes_o_d[D, dtype](b))
            # reassign
            var total = Float64(0)
            for t in range(n):
                var bd = 1e300
                var ba = 0
                for c in range(k):
                    var d = Float64(0)
                    for i in range(D):
                        var p = Float64(0)
                        for j in range(D):
                            p += Float64(rs[c][i, j]) * xs[t * D + j]
                        d += (p - ys[t * D + i]) * (p - ys[t * D + i])
                    d = sqrt(d)
                    if d < bd:
                        bd = d
                        ba = c
                assign[t] = ba
                total += bd
            res_k = total / Float64(n)
        if res_k < best_res:
            best_res = res_k
            var counts = List[Int](length=k, fill=0)
            for t in range(n):
                counts[assign[t]] += 1
            best = MixtureFit(assign.copy(), res1, res_k, n, counts^)
    return best^


def assignment_purity(assign: List[Int], truth: List[Int], k: Int) -> Float64:
    """Fraction of pairs in the majority true class of their component."""
    var vals = List[Int]()
    for t in range(len(truth)):
        var seen = False
        for v in range(len(vals)):
            if vals[v] == truth[t]:
                seen = True
                break
        if not seen:
            vals.append(truth[t])
    var nt = len(vals)
    if nt == 0 or len(assign) == 0:
        return 0.0
    var tab = List[Int](length=k * nt, fill=0)
    for t in range(len(assign)):
        var ti = 0
        for v in range(nt):
            if vals[v] == truth[t]:
                ti = v
                break
        tab[assign[t] * nt + ti] += 1
    var good = 0
    for c in range(k):
        var best = 0
        for v in range(nt):
            if tab[c * nt + v] > best:
                best = tab[c * nt + v]
        good += best
    return Float64(good) / Float64(len(assign))


def align_transport_tags(
    labels: List[Int],
    actions: List[Int],
    tags: List[Int],
    n_actions: Int,
    rounds: Int = 6,
) -> List[Int]:
    """Compose per-slot transport splits into one consistent place id.

    `tags[t]` is the mixture component of visit `t`'s OUTGOING transition
    within its `(label, action)` slot, or `-1` where that slot was not split.
    Those tags separate the two places sharing a label, but the component
    indices of `(l, +x)` and `(l, +y)` have no reason to agree, which is why
    G29's per-label split did not compose.

    **Union-find is the wrong tool here, and that was measured.** The tags are
    ~92 % pure, and a union-find is MONOTONE: one false merge is permanent, so
    an 8 % error rate collapses every label back into one block (measured: 15
    clones, purity unchanged from the quotient). Composition has to average the
    noise rather than take its transitive closure.

    So: majority voting. Each label picks a REFERENCE action — the one with
    the most tagged visits — and its tag becomes the relative place id. A
    transition table `(label, rel, action) -> rel'` is then voted from the
    visits where both ends are already known, and used to fill in the visits
    that took a non-reference action. Every step is a majority over hundreds of
    visits, so an 8 % tag error changes no verdict.
    """
    var n = len(labels)
    var n_lab = count_labels(labels)

    # reference action per label: the one with the most tagged visits
    var tagged_count = List[Int](length=n_lab * n_actions, fill=0)
    for t in range(n):
        if actions[t] >= 0 and tags[t] >= 0:
            tagged_count[labels[t] * n_actions + actions[t]] += 1
    var ref_action = List[Int](length=n_lab, fill=-1)
    for l in range(n_lab):
        var best = 0
        for a in range(n_actions):
            if tagged_count[l * n_actions + a] > best:
                best = tagged_count[l * n_actions + a]
                ref_action[l] = a

    var rel = List[Int](length=n, fill=-1)
    for t in range(n):
        if (
            actions[t] >= 0
            and tags[t] >= 0
            and actions[t] == ref_action[labels[t]]
        ):
            rel[t] = tags[t]

    for _ in range(rounds):
        # vote the forward table (label, rel, action) -> rel'
        var votes = List[Int](length=n_lab * 2 * n_actions * 2, fill=0)
        for t in range(n - 1):
            if actions[t] < 0 or rel[t] < 0 or rel[t + 1] < 0:
                continue
            var key = ((labels[t] * 2 + rel[t]) * n_actions + actions[t]) * 2
            votes[key + rel[t + 1]] += 1
        var nxt = List[Int](length=n_lab * 2 * n_actions, fill=-1)
        for l in range(n_lab):
            for r in range(2):
                for a in range(n_actions):
                    var k = (l * 2 + r) * n_actions + a
                    var v0 = votes[k * 2]
                    var v1 = votes[k * 2 + 1]
                    if v0 == 0 and v1 == 0:
                        continue
                    nxt[k] = 0 if v0 >= v1 else 1

        var changed = False
        # forward fill
        for t in range(1, n):
            if rel[t] >= 0 or actions[t - 1] < 0 or rel[t - 1] < 0:
                continue
            var k = (
                (labels[t - 1] * 2 + rel[t - 1]) * n_actions + actions[t - 1]
            )
            if nxt[k] >= 0:
                rel[t] = nxt[k]
                changed = True
        # backward fill: the dynamics is injective, so the predecessor of a
        # known place under a known action is determined too
        for t in range(n - 2, -1, -1):
            if rel[t] >= 0 or actions[t] < 0 or rel[t + 1] < 0:
                continue
            var found = -1
            for r in range(2):
                var k = (labels[t] * 2 + r) * n_actions + actions[t]
                if nxt[k] == rel[t + 1]:
                    if found >= 0:
                        found = -1
                        break
                    found = r
            if found >= 0:
                rel[t] = found
                changed = True
        if not changed:
            break

    var out = List[Int](length=n, fill=0)
    for t in range(n):
        out[t] = labels[t] * 2 + (rel[t] if rel[t] >= 0 else 0)
    # renumber densely
    var seen = List[Int]()
    var dense = List[Int](length=n, fill=0)
    for t in range(n):
        var idx = -1
        for k in range(len(seen)):
            if seen[k] == out[t]:
                idx = k
                break
        if idx < 0:
            seen.append(out[t])
            idx = len(seen) - 1
        dense[t] = idx
    return dense^


def transport_tags(
    rec: WalkRecord,
    labels: List[Int],
    n_actions: Int,
    seed: UInt64,
    min_pairs: Int = 40,
    min_drop: Float64 = 0.25,
) raises -> List[Int]:
    """Per-visit mixture component of its outgoing transition, `-1` when the
    slot has too few pairs or the second transport buys too little."""
    var n = rec.size()
    var out = List[Int](length=n, fill=-1)
    var n_lab = count_labels(labels)
    for l in range(n_lab):
        for a in range(n_actions):
            var idx = List[Int]()
            var xs = List[Float64]()
            var ys = List[Float64]()
            for t in range(n - 1):
                if labels[t] != l or rec.action[t] != a:
                    continue
                idx.append(t)
                for i in range(2):
                    xs.append(rec.u[t * 2 + i])
                    ys.append(rec.u[(t + 1) * 2 + i])
            if len(idx) < min_pairs:
                continue
            var f = fit_transport_mixture[2, DType.float64](
                xs, ys, 2, seed + UInt64(l * 17 + a)
            )
            if 1.0 - f.res_k / f.res_1 < min_drop:
                continue
            for j in range(len(idx)):
                out[idx[j]] = f.assign[j]
    return out^
