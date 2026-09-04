"""E1-2D: the frame bundle of a Klein bottle — two cycle generators, one of them
orientation-reversing.

The ring of `mobius_ring.mojo` has a single fundamental cycle, so every gate so
far has read exactly one holonomy. A torus grid has TWO independent generators,
and on a Klein bottle one of them reverses orientation while the other does not.
That is the first setting where the machinery has to do what §2 actually asks:
take a spanning tree, read `|E| - |V| + 1` fundamental cycles, and classify each
one on its own.

**Scope, stated plainly.** This builds the FRAME BUNDLE — the place graph and its
per-edge orthogonal transports — and generates frame observations, in the manner
of the numpy prototype. It is not a full 2D observation model with mixing and
texture: running learned encoders here would repeat Phase 3's question in a
bigger world, whereas what is untested is the multi-cycle reading. So the gate
covers planted transports and transports RECOVERED from noisy observations, and
does not claim anything about learned encoders in 2D.

**The world is flat except at the seam.** Per-cell frames `g(x, y)` are drawn and
the edge transports derived as `R = g_next g_prev^{-1}`, which makes every
elementary square trivial by construction and both loop holonomies trivial. Any
non-triviality then comes from the seam and nothing else — the same discipline
as the ring's "angles sum to zero".
"""

from std.math import cos, sin

from ..so_d import SqMat
from ..rng import Rng
from ..place_graph import PlaceGraph, Edge
from ..procrustes import PairBatch

comptime ACT_X: Int = 0
comptime ACT_Y: Int = 1


def _rot(t: Float64) -> SqMat[2, DType.float64]:
    var m = SqMat[2, DType.float64]()
    m[0, 0] = Scalar[DType.float64](cos(t))
    m[0, 1] = Scalar[DType.float64](-sin(t))
    m[1, 0] = Scalar[DType.float64](sin(t))
    m[1, 1] = Scalar[DType.float64](cos(t))
    return m^


struct KleinGrid[W: Int, H: Int](Copyable, Movable):
    """`W x H` torus of places whose frame bundle is a Klein bottle if `klein`.

    Place `(x, y)` has index `y * W + x`. Edges: `+x` from every cell (wrapping
    at `x = W-1`) and `+y` from every cell (wrapping at `y = H-1`). Reverse
    moves need no separate edge — the transports are orthogonal, so traversing
    backwards is `R^T` exactly.
    """

    comptime N_PLACES: Int = Self.W * Self.H
    comptime N_X_EDGES: Int = Self.W * Self.H
    comptime N_Y_EDGES: Int = Self.W * Self.H

    var x_edge: List[SqMat[2, DType.float64]]
    var y_edge: List[SqMat[2, DType.float64]]
    var klein: Bool

    def __init__(out self, klein: Bool, seed: UInt64 = 20260904):
        var rng = Rng(seed)
        # A frame per cell; edge transports are the differences, so every
        # elementary square is trivial and both loops start out trivial.
        var theta = List[Float64](length=Self.N_PLACES, fill=0)
        for i in range(Self.N_PLACES):
            theta[i] = rng.uniform_range(-0.6, 0.6)

        var refl = SqMat[2, DType.float64].identity()
        refl[1, 1] = Scalar[DType.float64](-1)

        self.x_edge = List[SqMat[2, DType.float64]]()
        self.y_edge = List[SqMat[2, DType.float64]]()
        for y in range(Self.H):
            for x in range(Self.W):
                var here = theta[y * Self.W + x]
                var nx = (x + 1) % Self.W
                var rx = _rot(theta[y * Self.W + nx] - here)
                # The seam: crossing x = W-1 -> 0 reverses orientation. This is
                # the single edge that makes the bundle a Klein bottle rather
                # than a torus.
                if klein and x == Self.W - 1:
                    rx = refl * rx
                self.x_edge.append(rx^)
                var ny = (y + 1) % Self.H
                self.y_edge.append(_rot(theta[ny * Self.W + x] - here))
        self.klein = klein

    def __init__(out self, *, copy: Self):
        self.x_edge = copy.x_edge.copy()
        self.y_edge = copy.y_edge.copy()
        self.klein = copy.klein

    def __init__(out self, *, deinit move: Self):
        self.x_edge = move.x_edge^
        self.y_edge = move.y_edge^
        self.klein = move.klein

    def place_of(self, x: Int, y: Int) -> Int:
        return y * Self.W + x

    def build_graph(self) raises -> PlaceGraph[2, DType.float64]:
        """Places + `+x` and `+y` action edges, with the spanning-tree gauge."""
        var g = PlaceGraph[2, DType.float64]()
        for _ in range(Self.N_PLACES):
            _ = g.add_place()
        for y in range(Self.H):
            for x in range(Self.W):
                var p = self.place_of(x, y)
                var qx = self.place_of((x + 1) % Self.W, y)
                _ = g.add_edge(
                    Edge.action_edge(p, qx, ACT_X), self.x_edge[p]
                )
                var qy = self.place_of(x, (y + 1) % Self.H)
                _ = g.add_edge(
                    Edge.action_edge(p, qy, ACT_Y), self.y_edge[p]
                )
        g.rebuild_gauge(0)
        return g^

    def rollout_pairs(
        self, episodes: Int, steps: Int, noise: Float64, seed: UInt64
    ) raises -> List[PairBatch[2, DType.float64]]:
        """Frame observations along random walks, one batch per graph edge.

        Batch index matches the edge index used by `build_graph`: `2*p` is the
        `+x` edge out of place `p`, `2*p + 1` the `+y` edge.
        """
        var rng = Rng(seed)
        var batches = List[PairBatch[2, DType.float64]]()
        for _ in range(2 * Self.N_PLACES):
            batches.append(PairBatch[2, DType.float64]())

        for _ in range(episodes):
            var a = rng.uniform_range(0.0, 6.283185307179586)
            var u = InlineArray[Scalar[DType.float64], 2](fill=0)
            u[0] = Scalar[DType.float64](cos(a))
            u[1] = Scalar[DType.float64](sin(a))
            var x = 0
            var y = 0
            for _ in range(steps):
                var p = self.place_of(x, y)
                var go_x = rng.uniform() < 0.5
                var r = self.x_edge[p] if go_x else self.y_edge[p]
                var edge_idx = 2 * p if go_x else 2 * p + 1
                var v = InlineArray[Scalar[DType.float64], 2](fill=0)
                for i in range(2):
                    var s = Scalar[DType.float64](0)
                    for j in range(2):
                        s += r[i, j] * u[j]
                    v[i] = s + Scalar[DType.float64](rng.normal() * noise)
                batches[edge_idx].push(u, v)
                for i in range(2):
                    u[i] = v[i]
                if go_x:
                    x = (x + 1) % Self.W
                else:
                    y = (y + 1) % Self.H
        return batches^
