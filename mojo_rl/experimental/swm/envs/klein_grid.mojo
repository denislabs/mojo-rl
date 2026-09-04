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

from std.math import abs, cos, sin

from ..so_d import SqMat
from ..rng import Rng
from ..place_graph import PlaceGraph, Edge
from ..procrustes import PairBatch
from ..world import SwmWorld

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
    var flat: Bool

    def __init__(
        out self, klein: Bool, seed: UInt64 = 20260904, flat: Bool = False
    ):
        """`flat=False` reproduces 6b's bundle, whose seam is CURVED: a square
        straddling it composes `refl r refl^-1 = r^-1`, so 8 of its 31
        fundamental cycles are non-trivial rotations and the holonomy group is
        not finite. `flat=True` inserts the reflection as a deck
        transformation, `R_x(W-1, y) = g(0, y) M g(W-1, y)^-1`, which makes
        every elementary square trivial: the connection is flat, every
        root-gauge holonomy lies in `{I, M}`, and the bundle is what a frame
        bundle over a physical Klein-like world would be. G19 uses the flat
        one and keeps the curved one as the control on the Z/2 assumption."""
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
                    if flat:
                        rx = (_rot(theta[y * Self.W + nx]) * refl) * _rot(-here)
                    else:
                        rx = refl * rx
                self.x_edge.append(rx^)
                var ny = (y + 1) % Self.H
                self.y_edge.append(_rot(theta[ny * Self.W + x] - here))
        self.klein = klein
        self.flat = flat

    def __init__(out self, *, copy: Self):
        self.x_edge = copy.x_edge.copy()
        self.y_edge = copy.y_edge.copy()
        self.klein = copy.klein
        self.flat = copy.flat

    def __init__(out self, *, deinit move: Self):
        self.x_edge = move.x_edge^
        self.y_edge = move.y_edge^
        self.klein = move.klein
        self.flat = move.flat

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


comptime ALIAS_NONE: Int = 0
comptime ALIAS_LOCAL: Int = 1
comptime ALIAS_GLOBAL: Int = 2


@fieldwise_init
struct KleinWorldConfig(Copyable, ImplicitlyCopyable, Movable):
    """Knobs of the 2D observation world (Phase 8)."""

    var klein: Bool
    var flat: Bool
    var world_seed: UInt64
    var obs_noise: Float64
    var texture_alias: Int
    """`ALIAS_NONE`, `ALIAS_LOCAL` (one pair, `(1,1) ~ (4,3)` on a 6x5 grid,
    neighbours distinct) or `ALIAS_GLOBAL` (`(x, y) ~ (x + W/2, y)`, a
    symmetry of the texture map)."""
    var labels_from_texture: Bool
    """`place_label()` returns the texture label rather than the oracle cell —
    the trainer then indexes transports by what a recogniser can deliver."""

    @staticmethod
    def flat_klein(world_seed: UInt64 = 20260904) -> Self:
        return Self(True, True, world_seed, 0.02, ALIAS_NONE, False)

    @staticmethod
    def flat_torus(world_seed: UInt64 = 20260904) -> Self:
        return Self(False, True, world_seed, 0.02, ALIAS_NONE, False)

    def with_alias(self, mode: Int, labels_from_texture: Bool = True) -> Self:
        var c = self
        c.texture_alias = mode
        c.labels_from_texture = labels_from_texture
        return c


struct KleinWorld[
    W: Int,
    H: Int,
    NUISANCE_DIM: Int,
    OBS_DIM: Int,
](SwmWorld):
    """The Klein/torus frame bundle with an OBSERVATION MODEL, so a learned
    encoder can be trained on it: the ring's recipe (transported landmark +
    per-cell texture, overcompletely mixed, plus noise) on a 2D place graph.

    Actions are `ACT_X` / `ACT_Y` (wrapping); exploration is a random walk.
    The frame starts at identity on every reset, so the frame at a cell depends
    on the homotopy class of the path taken — that is the double cover, and it
    is what the content channel must be invariant to.
    """

    comptime N_PLACES: Int = Self.W * Self.H
    comptime LATENT_DIM: Int = 2 + Self.NUISANCE_DIM
    comptime dtype: DType = DType.float64
    """`KleinGrid` is float64-only, so the world is too."""
    comptime ELEM: DType = DType.float64

    var cfg: KleinWorldConfig
    var grid: KleinGrid[Self.W, Self.H]
    var texture_of: List[Int]
    """Texture label per cell (identity without aliasing)."""
    var nuisance: List[Scalar[Self.dtype]]
    var mix: List[Scalar[Self.dtype]]
    var x: Int
    var y: Int
    var frame: SqMat[2, Self.dtype]
    var w: List[Scalar[Self.dtype]]
    var rng: Rng

    def __init__(out self, cfg: KleinWorldConfig) raises:
        comptime assert Self.W % 2 == 0, "global aliasing folds the grid in x"
        self.cfg = cfg
        self.grid = KleinGrid[Self.W, Self.H](cfg.klein, cfg.world_seed, cfg.flat)
        var wr = Rng(cfg.world_seed ^ 0x5EED_7EA7_0000_0001)

        self.texture_of = List[Int](length=Self.N_PLACES, fill=0)
        for p in range(Self.N_PLACES):
            self.texture_of[p] = p
        if cfg.texture_alias == ALIAS_LOCAL:
            self.texture_of[3 * Self.W + 4] = 1 * Self.W + 1
        elif cfg.texture_alias == ALIAS_GLOBAL:
            for yy in range(Self.H):
                for xx in range(Self.W):
                    self.texture_of[yy * Self.W + xx] = yy * Self.W + (
                        xx % (Self.W // 2)
                    )

        var tex = List[Scalar[Self.dtype]](
            length=Self.N_PLACES * Self.NUISANCE_DIM, fill=0
        )
        for p in range(Self.N_PLACES):
            for k in range(Self.NUISANCE_DIM):
                tex[p * Self.NUISANCE_DIM + k] = Scalar[Self.dtype](
                    wr.uniform_range(-1.0, 1.0)
                )
        self.nuisance = List[Scalar[Self.dtype]](
            length=Self.N_PLACES * Self.NUISANCE_DIM, fill=0
        )
        for p in range(Self.N_PLACES):
            var t = self.texture_of[p]
            for k in range(Self.NUISANCE_DIM):
                self.nuisance[p * Self.NUISANCE_DIM + k] = tex[
                    t * Self.NUISANCE_DIM + k
                ]

        self.mix = List[Scalar[Self.dtype]](
            length=Self.OBS_DIM * Self.LATENT_DIM, fill=0
        )
        for r in range(Self.OBS_DIM):
            for c in range(Self.LATENT_DIM):
                var v = wr.uniform_range(-1.0, 1.0)
                if abs(v) < 0.25:
                    v = 0.25 if v >= 0 else -0.25
                self.mix[r * Self.LATENT_DIM + c] = Scalar[Self.dtype](v)

        self.x = 0
        self.y = 0
        self.frame = SqMat[2, Self.dtype].identity()
        self.w = List[Scalar[Self.dtype]](length=2, fill=0)
        self.w[0] = 1
        self.rng = Rng(cfg.world_seed ^ 0xA5A5_A5A5_A5A5_A5A5)

    def __init__(out self, *, copy: Self):
        self.cfg = copy.cfg
        self.grid = copy.grid.copy()
        self.texture_of = copy.texture_of.copy()
        self.nuisance = copy.nuisance.copy()
        self.mix = copy.mix.copy()
        self.x = copy.x
        self.y = copy.y
        self.frame = copy.frame.copy()
        self.w = copy.w.copy()
        self.rng = copy.rng

    def __init__(out self, *, deinit move: Self):
        self.cfg = move.cfg
        self.grid = move.grid^
        self.texture_of = move.texture_of^
        self.nuisance = move.nuisance^
        self.mix = move.mix^
        self.x = move.x
        self.y = move.y
        self.frame = move.frame^
        self.w = move.w^
        self.rng = move.rng

    def reset(mut self, seed: UInt64) raises:
        self.rng = Rng(seed)
        var a = self.rng.uniform_range(0.0, 6.283185307179586)
        self.w[0] = Scalar[Self.dtype](cos(a))
        self.w[1] = Scalar[Self.dtype](sin(a))
        self.x = 0
        self.y = 0
        self.frame = SqMat[2, Self.dtype].identity()

    def step(mut self, action: Int) raises:
        var p = self.place_id()
        if action == ACT_X:
            self.frame = self.grid.x_edge[p] * self.frame
            self.x = (self.x + 1) % Self.W
        elif action == ACT_Y:
            self.frame = self.grid.y_edge[p] * self.frame
            self.y = (self.y + 1) % Self.H
        else:
            raise Error("KleinWorld.step: action must be ACT_X or ACT_Y")

    def explore_action(mut self) -> Int:
        return ACT_X if self.rng.uniform() < 0.5 else ACT_Y

    def observation(mut self) -> List[Scalar[Self.dtype]]:
        var latent = List[Scalar[Self.dtype]](length=Self.LATENT_DIM, fill=0)
        var lm = self.true_landmark()
        latent[0] = lm[0]
        latent[1] = lm[1]
        var p = self.place_id()
        for k in range(Self.NUISANCE_DIM):
            latent[2 + k] = self.nuisance[p * Self.NUISANCE_DIM + k]
        var obs = List[Scalar[Self.dtype]](length=Self.OBS_DIM, fill=0)
        for r in range(Self.OBS_DIM):
            var s = Scalar[Self.dtype](0)
            for c in range(Self.LATENT_DIM):
                s += self.mix[r * Self.LATENT_DIM + c] * latent[c]
            obs[r] = s + Scalar[Self.dtype](
                self.rng.normal() * self.cfg.obs_noise
            )
        return obs^

    # -- oracles --------------------------------------------------------------

    def place_id(self) -> Int:
        return self.y * Self.W + self.x

    def texture_label(self) -> Int:
        return self.texture_of[self.place_id()]

    def place_label(self) -> Int:
        if self.cfg.labels_from_texture:
            return self.texture_of[self.place_id()]
        return self.place_id()

    def true_landmark(self) -> InlineArray[Scalar[Self.dtype], 2]:
        var out = InlineArray[Scalar[Self.dtype], 2](fill=0)
        for i in range(2):
            var s = Scalar[Self.dtype](0)
            for j in range(2):
                s += self.frame[i, j] * self.w[j]
            out[i] = s
        return out^

    def nuisance_at(self, cell: Int) -> List[Scalar[Self.dtype]]:
        var out = List[Scalar[Self.dtype]](length=Self.NUISANCE_DIM, fill=0)
        for k in range(Self.NUISANCE_DIM):
            out[k] = self.nuisance[cell * Self.NUISANCE_DIM + k]
        return out^
