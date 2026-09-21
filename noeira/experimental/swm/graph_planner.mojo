"""Planning on a learned clone graph, in the double cover (Phase 8d).

The ring planner scans monotone walks, which a ring makes exhaustive. A place
graph needs a search, and the search has to live on the DOUBLE COVER: two
paths to the same clone may arrive with frames that differ by the holonomy of
the loop they enclose, and on a non-orientable world those are different
states with different rewards. So the search state is `(clone, u)`, with at
most a few distinct `u` kept per clone (frames within `frame_tol` of a stored
one are the same state), and the frame is TRANSPORTED along every edge by the
clone graph's own re-fitted `R` — never predicted by a free map.

The goal is an observation, encoded to `(u_goal, h_goal)`. Goal CLONES are
those whose content centroid lies within `h_tol` of `h_goal` — under aliasing
that is more than one clone, and the frame decides between them exactly as it
decides the parity. Among every reached state at a goal clone, the one with
the smallest `‖u − u_goal‖` wins, and the path to it is returned.
"""

from std.math import sqrt

from .so_d import SqMat
from .place_graph import PlaceGraph, EDGE_ACTION


struct GraphPlan(Copyable, Movable):
    var actions: List[Int]
    var found: Bool
    var goal_clone: Int
    var u_err: Float64
    var u_err_runner_up: Float64
    """Smallest `‖u − u_goal‖` over the OTHER reached states at goal clones —
    typically the same cell at the other parity. `u_err_runner_up − u_err` is
    the frame's margin for the choice it made; small when the goal's
    landmark lies along the reflection axis (the holonomy's fixed
    subspace), where the two parities look alike."""
    var n_states: Int
    var n_goal_clones: Int

    def __init__(out self):
        self.actions = List[Int]()
        self.found = False
        self.goal_clone = -1
        self.u_err = 1e300
        self.u_err_runner_up = 1e300
        self.n_states = 0
        self.n_goal_clones = 0

    def __init__(out self, *, copy: Self):
        self.actions = copy.actions.copy()
        self.found = copy.found
        self.goal_clone = copy.goal_clone
        self.u_err = copy.u_err
        self.u_err_runner_up = copy.u_err_runner_up
        self.n_states = copy.n_states
        self.n_goal_clones = copy.n_goal_clones

    def __init__(out self, *, deinit move: Self):
        self.actions = move.actions^
        self.found = move.found
        self.goal_clone = move.goal_clone
        self.u_err = move.u_err
        self.u_err_runner_up = move.u_err_runner_up
        self.n_states = move.n_states
        self.n_goal_clones = move.n_goal_clones


def nearest_centroid(
    h: List[Float64], centroids: List[Float64], dim: Int, n: Int
) -> Int:
    var best = 1e300
    var arg = -1
    for c in range(n):
        var d = Float64(0)
        for i in range(dim):
            var e = centroids[c * dim + i] - h[i]
            d += e * e
        if d < best:
            best = d
            arg = c
    return arg


def plan_double_cover(
    g: PlaceGraph[2, DType.float64],
    centroids: List[Float64],
    content_dim: Int,
    start_clone: Int,
    u0: List[Float64],
    u_goal: List[Float64],
    h_goal: List[Float64],
    h_tol: Float64,
    frame_tol: Float64,
    transport_frames: Bool = True,
    max_states_per_clone: Int = 4,
) -> GraphPlan:
    """Breadth-first search over `(clone, u)`; see the module docstring.

    `transport_frames=False` is the constant-sheaf ablation: `u` is never
    moved, every clone has one state, and the search is parity-blind.
    """
    var n = g.n_places
    # adjacency by (clone, action): edge index or -1
    var n_actions = 0
    for e in range(g.n_edges()):
        if g.edges[e].action + 1 > n_actions:
            n_actions = g.edges[e].action + 1
    var adj = List[Int](length=n * n_actions, fill=-1)
    for e in range(g.n_edges()):
        var ed = g.edges[e]
        if ed.kind == EDGE_ACTION:
            adj[ed.src * n_actions + ed.action] = e

    # goal clones by content
    var is_goal = List[Bool](length=n, fill=False)
    var n_goal = 0
    for c in range(n):
        var d = Float64(0)
        for i in range(content_dim):
            var e = centroids[c * content_dim + i] - h_goal[i]
            d += e * e
        if sqrt(d) <= h_tol:
            is_goal[c] = True
            n_goal += 1

    # BFS state store
    var st_clone = List[Int]()
    var st_u = List[Float64]()
    var st_parent = List[Int]()
    var st_action = List[Int]()
    var per_clone = List[Int](length=n, fill=0)
    st_clone.append(start_clone)
    st_u.append(u0[0])
    st_u.append(u0[1])
    st_parent.append(-1)
    st_action.append(-1)
    per_clone[start_clone] += 1
    var head = 0
    while head < len(st_clone):
        var c = st_clone[head]
        for a in range(n_actions):
            var e = adj[c * n_actions + a]
            if e < 0:
                continue
            var nxt = g.edges[e].dst
            var u = List[Float64](length=2, fill=0)
            if transport_frames:
                var r = g.transports[e]
                for i in range(2):
                    var s = Float64(0)
                    for j in range(2):
                        s += Float64(r[i, j]) * st_u[head * 2 + j]
                    u[i] = s
            else:
                u[0] = st_u[head * 2]
                u[1] = st_u[head * 2 + 1]
            # seen?
            var seen = False
            for k in range(len(st_clone)):
                if st_clone[k] != nxt:
                    continue
                var d0 = st_u[k * 2] - u[0]
                var d1 = st_u[k * 2 + 1] - u[1]
                if sqrt(d0 * d0 + d1 * d1) <= frame_tol:
                    seen = True
                    break
            if seen or per_clone[nxt] >= max_states_per_clone:
                continue
            st_clone.append(nxt)
            st_u.append(u[0])
            st_u.append(u[1])
            st_parent.append(head)
            st_action.append(a)
            per_clone[nxt] += 1
        head += 1

    var out = GraphPlan()
    out.n_states = len(st_clone)
    out.n_goal_clones = n_goal
    var best = -1
    for k in range(len(st_clone)):
        if not is_goal[st_clone[k]]:
            continue
        var d0 = st_u[k * 2] - u_goal[0]
        var d1 = st_u[k * 2 + 1] - u_goal[1]
        var d = sqrt(d0 * d0 + d1 * d1)
        if d < out.u_err:
            out.u_err_runner_up = out.u_err
            out.u_err = d
            best = k
        elif d < out.u_err_runner_up:
            out.u_err_runner_up = d
    if best < 0:
        return out^
    out.found = True
    out.goal_clone = st_clone[best]
    var rev = List[Int]()
    var k = best
    while st_parent[k] >= 0:
        rev.append(st_action[k])
        k = st_parent[k]
    for i in range(len(rev)):
        out.actions.append(rev[len(rev) - 1 - i])
    return out^


def clone_centroids(
    h: List[Float64], labels: List[Int], content_dim: Int, n_labels: Int
) -> List[Float64]:
    var acc = List[Float64](length=n_labels * content_dim, fill=0)
    var cnt = List[Float64](length=n_labels, fill=0)
    for t in range(len(labels)):
        var l = labels[t]
        cnt[l] += 1
        for i in range(content_dim):
            acc[l * content_dim + i] += h[t * content_dim + i]
    for l in range(n_labels):
        if cnt[l] > 0:
            for i in range(content_dim):
                acc[l * content_dim + i] /= cnt[l]
    return acc^
