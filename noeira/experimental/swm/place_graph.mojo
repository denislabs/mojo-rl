"""The place graph: a runtime graph of places, its tree gauge, and holonomy.

Deliberately NOT expressed in the PCN's comptime chain. `PCSequential[*BLOCKS]`
is a compile-time list of levels with a flat parameter slab and a
`comptime for`-unrolled inference; this graph grows online with a runtime edge
count. Folding one into the other is how you get a compile explosion. `D` is
the only compile-time parameter here.

The gauge trick is the whole reason holonomy is cheap. Fix a spanning tree,
store for each place `p` the transport `T_p` from the root along the tree.
Then the fundamental cycle created by a non-tree edge `e = (p -> q)` has

    H_e = T_q^T R_e T_p            (T orthogonal, so T^-1 = T^T)

based at the root: three matrix products, `O(D^3)`, **independent of the cycle's
length**. This is pose-graph SLAM's gauge fixing, and it is what makes the
observable survive on long loops where the spectral gap (~1/N^2) does not.

Note the two spellings in the design doc agree: §2's general
`H_e = T_q^-1 R_e T_p` reduces to §4.4's `H_e = T_p'^-1 T_p` for an
identification edge, whose restriction is `I` on both sides. Only the general
form is implemented; §5's pseudo-code omits `R_e` and is wrong as written.
"""

from .so_d import SqMat

comptime EDGE_ACTION: UInt8 = 0
comptime EDGE_INTERMODAL: UInt8 = 1
comptime EDGE_IDENTIFICATION: UInt8 = 2


@fieldwise_init
struct Edge(Copyable, ImplicitlyCopyable, Movable):
    """One edge of the base graph. Transport lives alongside, in `PlaceGraph`."""

    var src: Int
    var dst: Int
    var kind: UInt8
    var action: Int
    var sigma: Int8
    """Orientation bit: +1 keeps R in SO(D), -1 composes the fixed reflection.

    Discrete on purpose. Neither Cayley nor exp can cross between the two
    components of O(D), so `det H` is exactly the product of these bits around
    a cycle — no continuous mechanism contributes to it.
    """
    var w: Float64
    """Confidence in [0, 1] (GNC). A switchable constraint, Sunderhauf-style."""
    var r_local: Float64
    """Pre-consensus residual — measured on observed pairs, NOT after inference."""
    var in_energy: Bool
    """False for an identification whose cycle carries a non-trivial holonomy.

    Such an edge is not a constraint, it is a monodromy: including it would
    force the frame into the cycle's fixed subspace, which is precisely the
    collapse the cocycle loss produces.
    """

    @staticmethod
    def action_edge(src: Int, dst: Int, action: Int) -> Self:
        return Self(src, dst, EDGE_ACTION, action, 1, 1.0, 0.0, True)

    @staticmethod
    def identification(src: Int, dst: Int) -> Self:
        return Self(src, dst, EDGE_IDENTIFICATION, -1, 1, 1.0, 0.0, True)


struct PlaceGraph[D: Int, dtype: DType = DType.float64](Copyable, Movable):
    """Places, edges, per-edge transports, and the tree gauge over them."""

    var n_places: Int
    var edges: List[Edge]
    var transports: List[SqMat[Self.D, Self.dtype]]
    """`R_e` per edge, indexed alike. `u_dst = R_e u_src`."""
    var parent: List[Int]
    """Spanning-tree parent of each place; -1 for the root or an unreached one."""
    var parent_edge: List[Int]
    """Edge index joining a place to its parent; -1 if none."""
    var t_root: List[SqMat[Self.D, Self.dtype]]
    """`T_p`: transport root -> p along the tree. Valid after `rebuild_gauge`."""
    var tree_valid: Bool

    def __init__(out self):
        self.n_places = 0
        self.edges = List[Edge]()
        self.transports = List[SqMat[Self.D, Self.dtype]]()
        self.parent = List[Int]()
        self.parent_edge = List[Int]()
        self.t_root = List[SqMat[Self.D, Self.dtype]]()
        self.tree_valid = False

    def __init__(out self, *, copy: Self):
        self.n_places = copy.n_places
        self.edges = copy.edges.copy()
        self.transports = copy.transports.copy()
        self.parent = copy.parent.copy()
        self.parent_edge = copy.parent_edge.copy()
        self.t_root = copy.t_root.copy()
        self.tree_valid = copy.tree_valid

    def __init__(out self, *, deinit move: Self):
        self.n_places = move.n_places
        self.edges = move.edges^
        self.transports = move.transports^
        self.parent = move.parent^
        self.parent_edge = move.parent_edge^
        self.t_root = move.t_root^
        self.tree_valid = move.tree_valid

    # -- construction ---------------------------------------------------------

    def add_place(mut self) -> Int:
        """A new place every step; recognition adds an edge, never merges."""
        var idx = self.n_places
        self.n_places += 1
        self.parent.append(-1)
        self.parent_edge.append(-1)
        self.t_root.append(SqMat[Self.D, Self.dtype].identity())
        self.tree_valid = False
        return idx

    def add_edge(
        mut self, edge: Edge, transport: SqMat[Self.D, Self.dtype]
    ) -> Int:
        var idx = len(self.edges)
        self.edges.append(edge)
        self.transports.append(transport)
        self.tree_valid = False
        return idx

    def n_edges(self) -> Int:
        return len(self.edges)

    # -- gauge ----------------------------------------------------------------

    def rebuild_gauge(mut self, root: Int = 0) raises:
        """BFS a spanning tree from `root` over ACTION edges, then fill `T_p`.

        Only action edges carry dynamics, so the tree is the exploration tree
        (v2 §4.1). Identification edges are therefore always non-tree edges,
        and each contributes exactly one fundamental cycle.
        """
        if self.n_places == 0:
            raise Error("rebuild_gauge: empty graph")
        if root < 0 or root >= self.n_places:
            raise Error("rebuild_gauge: root out of range")

        for i in range(self.n_places):
            self.parent[i] = -1
            self.parent_edge[i] = -1
            self.t_root[i] = SqMat[Self.D, Self.dtype].identity()

        # Adjacency built once per rebuild: (neighbour, edge index, forward?)
        var adj_start = List[Int](length=self.n_places + 1, fill=0)
        for e in range(len(self.edges)):
            if self.edges[e].kind != EDGE_ACTION:
                continue
            adj_start[self.edges[e].src + 1] += 1
            adj_start[self.edges[e].dst + 1] += 1
        for i in range(self.n_places):
            adj_start[i + 1] += adj_start[i]
        var total = adj_start[self.n_places]
        var adj_node = List[Int](length=total, fill=0)
        var adj_edge = List[Int](length=total, fill=0)
        var adj_fwd = List[Bool](length=total, fill=False)
        var cursor = adj_start.copy()
        for e in range(len(self.edges)):
            if self.edges[e].kind != EDGE_ACTION:
                continue
            var s = self.edges[e].src
            var d = self.edges[e].dst
            adj_node[cursor[s]] = d
            adj_edge[cursor[s]] = e
            adj_fwd[cursor[s]] = True
            cursor[s] += 1
            adj_node[cursor[d]] = s
            adj_edge[cursor[d]] = e
            adj_fwd[cursor[d]] = False
            cursor[d] += 1

        var visited = List[Bool](length=self.n_places, fill=False)
        var queue = List[Int]()
        visited[root] = True
        queue.append(root)
        var head = 0
        while head < len(queue):
            var p = queue[head]
            head += 1
            for k in range(adj_start[p], adj_start[p + 1]):
                var q = adj_node[k]
                if visited[q]:
                    continue
                var e = adj_edge[k]
                visited[q] = True
                self.parent[q] = p
                self.parent_edge[q] = e
                # Forward edge p -> q transports with R_e; traversing it
                # backwards transports with R_e^T (orthogonality is what makes
                # the reverse direction exact and free).
                if adj_fwd[k]:
                    self.t_root[q] = self.transports[e] * self.t_root[p]
                else:
                    self.t_root[q] = self.transports[e].transpose() * self.t_root[p]
                queue.append(q)

        self.tree_valid = True

    def is_tree_edge(self, e: Int) -> Bool:
        var ed = self.edges[e]
        return self.parent_edge[ed.dst] == e or self.parent_edge[ed.src] == e

    def fundamental_cycle_edges(self) -> List[Int]:
        """Non-tree edges — one fundamental cycle each, generating pi_1.

        For a non-abelian group like O(D) with D >= 2 these generators are what
        is required; an arbitrary basis of the cycle SPACE would not do.
        """
        var out = List[Int]()
        for e in range(len(self.edges)):
            if not self.is_tree_edge(e):
                out.append(e)
        return out^

    # -- the observable -------------------------------------------------------

    def holonomy(self, e: Int) raises -> SqMat[Self.D, Self.dtype]:
        """`H_e = T_dst^T R_e T_src`, based at the root. `O(D^3)`, length-free."""
        if not self.tree_valid:
            raise Error("holonomy: call rebuild_gauge() first")
        var ed = self.edges[e]
        return (
            self.t_root[ed.dst].transpose() * self.transports[e]
        ) * self.t_root[ed.src]

    def holonomy_det(self, e: Int) raises -> Float64:
        """`det H_e in {+1, -1}` — the Z/2 class, the only robust invariant.

        The continuous part of the holonomy cannot be distinguished from a
        constant sensor bias by a single cycle (control D'), so it is reported
        but never asserted on without cross-confirmation.
        """
        return Float64(self.holonomy(e).det())

    def holonomy_dist_to_identity(self, e: Int) raises -> Float64:
        return Float64(self.holonomy(e).dist_to_identity())

    def cycle_edge_set(self, e: Int) raises -> List[Int]:
        """Every edge of the fundamental cycle created by non-tree edge `e`.

        The non-tree edge plus the symmetric difference of the two tree paths to
        the root — i.e. the path through their lowest common ancestor. Needed by
        the cross-confirmation rule, which asks whether two cycles are EDGE
        DISJOINT: a single biased edge can explain any number of overlapping
        cycles but not two that share nothing.
        """
        if not self.tree_valid:
            raise Error("cycle_edge_set: call rebuild_gauge() first")
        var ed = self.edges[e]
        var seen = List[Int]()
        var depth_a = List[Int]()
        var p = ed.src
        while self.parent_edge[p] >= 0:
            depth_a.append(self.parent_edge[p])
            p = self.parent[p]
        var depth_b = List[Int]()
        p = ed.dst
        while self.parent_edge[p] >= 0:
            depth_b.append(self.parent_edge[p])
            p = self.parent[p]
        # Symmetric difference: shared ancestry cancels.
        for i in range(len(depth_a)):
            var shared = False
            for j in range(len(depth_b)):
                if depth_a[i] == depth_b[j]:
                    shared = True
                    break
            if not shared:
                seen.append(depth_a[i])
        for j in range(len(depth_b)):
            var shared = False
            for i in range(len(depth_a)):
                if depth_b[j] == depth_a[i]:
                    shared = True
                    break
            if not shared:
                seen.append(depth_b[j])
        seen.append(e)
        return seen^
