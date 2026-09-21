"""E1 in D dimensions: the Mobius ring with an `O(D)` frame (Phase 12).

G30 measured, on planted transports, that GAUGE COINCIDENCE — a false closure
whose walk holonomy lands within tolerance of `I` or of the monodromy `M` — is
a two-dimensional artefact: 59 % of candidate walks at `d = 2`, 9 % at 3, and
**0 %** at 4 and 6. That says a wider fibre would buy away the residue Phase 7
recorded as irreducible. It does not say a LEARNED wide frame keeps its
landmark, its anti-collapse or its `Z/2` class, which is what this world exists
to ask.

Same recipe as `mobius_ring.mojo` — a transported landmark and a
non-transported per-cell texture, mixed overcompletely, plus noise — with the
landmark in `R^FRAME_DIM` and the transports in `O(FRAME_DIM)`.

**The transports are built from a FRAME PER CELL**, `R_i = g_{i+1} g_i^-1`,
with the seam carrying `g_0 Q g_{N-1}^-1` for a fixed Householder reflection
`Q`. That construction telescopes to the identity exactly in any dimension.
The ring's original "draw angles that sum to zero" trick does NOT generalise:
it is abelian, and in `O(D>2)` the Baker-Campbell-Hausdorff residue is large
enough to destroy the flatness outright — measured in G30, where a first
version built that way accepted 0 of 36 TRUE closures at `D >= 3`.
"""

from std.math import abs, cos, sin, sqrt

from ..so_d import SqMat, skew_from_vector, expm_skew, householder
from ..rng import Rng
from ..world import SwmWorld

comptime ACTION_FORWARD_ND: Int = 0


@fieldwise_init
struct MobiusNDConfig(Copyable, ImplicitlyCopyable, Movable):
    var mobius: Bool
    var world_seed: UInt64
    var obs_noise: Float64
    var step: Float64
    """Frobenius norm of each cell frame's generator."""
    var texture_alias: Int

    @staticmethod
    def default_mobius(world_seed: UInt64 = 20260904) -> Self:
        return Self(True, world_seed, 0.02, 0.6, 1)

    @staticmethod
    def default_orientable(world_seed: UInt64 = 20260904) -> Self:
        return Self(False, world_seed, 0.02, 0.6, 1)

    def with_alias(self, group_size: Int) -> Self:
        var c = self
        c.texture_alias = group_size
        return c


struct MobiusRingND[
    N_CELLS: Int,
    FRAME_DIM: Int,
    NUISANCE_DIM: Int,
    OBS_DIM: Int,
](SwmWorld):
    """Ring whose fibre is `R^FRAME_DIM`, transported by `O(FRAME_DIM)`."""

    comptime LATENT_DIM: Int = Self.FRAME_DIM + Self.NUISANCE_DIM
    comptime SEAM_EDGE: Int = Self.N_CELLS - 1
    comptime NGEN: Int = Self.FRAME_DIM * (Self.FRAME_DIM - 1) // 2
    comptime dtype: DType = DType.float64
    comptime ELEM: DType = DType.float64

    var cfg: MobiusNDConfig
    var edge_rot: List[SqMat[Self.FRAME_DIM, Self.dtype]]
    var nuisance: List[Scalar[Self.dtype]]
    var mix: List[Scalar[Self.dtype]]
    var cell: Int
    var frame: SqMat[Self.FRAME_DIM, Self.dtype]
    var w: List[Scalar[Self.dtype]]
    var seam_crossings: Int
    var rng: Rng

    def __init__(out self, cfg: MobiusNDConfig) raises:
        comptime assert Self.N_CELLS >= 3, "a ring needs at least 3 cells"
        comptime assert Self.FRAME_DIM >= 2, "a frame needs at least 2 dims"
        comptime assert (
            Self.OBS_DIM >= Self.LATENT_DIM
        ), "observation must not lose the latent"
        self.cfg = cfg
        var wr = Rng(cfg.world_seed)

        # A frame per cell; edge transports are their differences, so the
        # orientable ring is EXACTLY flat in any dimension (see the header).
        var frames = List[SqMat[Self.FRAME_DIM, Self.dtype]]()
        for _ in range(Self.N_CELLS):
            var v = List[Float64](length=Self.NGEN, fill=0)
            var nrm = Float64(0)
            for k in range(Self.NGEN):
                v[k] = wr.normal()
                nrm += v[k] * v[k]
            nrm = sqrt(nrm)
            if nrm < 1e-12:
                nrm = 1.0
            var sp = List[Scalar[Self.dtype]](length=Self.NGEN, fill=0)
            for k in range(Self.NGEN):
                sp[k] = Scalar[Self.dtype](v[k] * cfg.step / nrm)
            frames.append(
                expm_skew[Self.FRAME_DIM, Self.dtype](
                    skew_from_vector[Self.FRAME_DIM, Self.dtype](Span(sp))
                )
            )

        var refl_v = List[Float64](length=Self.FRAME_DIM, fill=0)
        refl_v[0] = 1.0
        var q = householder[Self.FRAME_DIM, Self.dtype](Span(refl_v))

        self.edge_rot = List[SqMat[Self.FRAME_DIM, Self.dtype]]()
        for i in range(Self.N_CELLS):
            var nxt = (i + 1) % Self.N_CELLS
            var inv = frames[i].transpose()
            if cfg.mobius and i == Self.SEAM_EDGE:
                self.edge_rot.append((frames[nxt] * q) * inv)
            else:
                self.edge_rot.append(frames[nxt] * inv)

        # Texture per cell, aliased in groups exactly as on the 2D ring.
        var group_size = cfg.texture_alias if cfg.texture_alias >= 1 else 1
        var n_groups = Self.N_CELLS // group_size
        if n_groups < 1:
            n_groups = 1
        var group_tex = List[Scalar[Self.dtype]](
            length=n_groups * Self.NUISANCE_DIM, fill=0
        )
        for g in range(n_groups):
            for k in range(Self.NUISANCE_DIM):
                group_tex[g * Self.NUISANCE_DIM + k] = Scalar[Self.dtype](
                    wr.uniform_range(-1.0, 1.0)
                )
        self.nuisance = List[Scalar[Self.dtype]](
            length=Self.N_CELLS * Self.NUISANCE_DIM, fill=0
        )
        for c in range(Self.N_CELLS):
            var g = c % n_groups
            for k in range(Self.NUISANCE_DIM):
                self.nuisance[c * Self.NUISANCE_DIM + k] = group_tex[
                    g * Self.NUISANCE_DIM + k
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

        self.cell = 0
        self.frame = SqMat[Self.FRAME_DIM, Self.dtype].identity()
        self.w = List[Scalar[Self.dtype]](length=Self.FRAME_DIM, fill=0)
        self.w[0] = 1
        self.seam_crossings = 0
        self.rng = Rng(cfg.world_seed ^ 0xA5A5_A5A5_A5A5_A5A5)

    def __init__(out self, *, copy: Self):
        self.cfg = copy.cfg
        self.edge_rot = copy.edge_rot.copy()
        self.nuisance = copy.nuisance.copy()
        self.mix = copy.mix.copy()
        self.cell = copy.cell
        self.frame = copy.frame.copy()
        self.w = copy.w.copy()
        self.seam_crossings = copy.seam_crossings
        self.rng = copy.rng

    def __init__(out self, *, deinit move: Self):
        self.cfg = move.cfg
        self.edge_rot = move.edge_rot^
        self.nuisance = move.nuisance^
        self.mix = move.mix^
        self.cell = move.cell
        self.frame = move.frame^
        self.w = move.w^
        self.seam_crossings = move.seam_crossings
        self.rng = move.rng

    def reset(mut self, seed: UInt64) raises:
        """A fresh landmark direction, uniform on the unit sphere of `R^D`, so
        the per-edge transport is identifiable from observed pairs."""
        self.rng = Rng(seed)
        var nrm = Float64(0)
        var v = List[Float64](length=Self.FRAME_DIM, fill=0)
        for i in range(Self.FRAME_DIM):
            v[i] = self.rng.normal()
            nrm += v[i] * v[i]
        nrm = sqrt(nrm)
        if nrm < 1e-12:
            nrm = 1.0
        for i in range(Self.FRAME_DIM):
            self.w[i] = Scalar[Self.dtype](v[i] / nrm)
        self.cell = 0
        self.frame = SqMat[Self.FRAME_DIM, Self.dtype].identity()
        self.seam_crossings = 0

    def step(mut self, action: Int) raises:
        if action != ACTION_FORWARD_ND:
            raise Error("MobiusRingND.step: only forward is defined")
        var e = self.cell
        self.frame = self.edge_rot[e] * self.frame
        self.cell = (self.cell + 1) % Self.N_CELLS
        if e == Self.SEAM_EDGE:
            self.seam_crossings += 1

    def explore_action(mut self) -> Int:
        return ACTION_FORWARD_ND

    def observation(mut self) -> List[Scalar[Self.dtype]]:
        var latent = List[Scalar[Self.dtype]](length=Self.LATENT_DIM, fill=0)
        var lm = self.true_landmark()
        for i in range(Self.FRAME_DIM):
            latent[i] = lm[i]
        for k in range(Self.NUISANCE_DIM):
            latent[Self.FRAME_DIM + k] = self.nuisance[
                self.cell * Self.NUISANCE_DIM + k
            ]
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
        return self.cell

    def place_label(self) -> Int:
        return self.cell

    def lap_parity(self) -> Int:
        return self.seam_crossings % 2

    def true_landmark(self) -> List[Scalar[Self.dtype]]:
        var out = List[Scalar[Self.dtype]](length=Self.FRAME_DIM, fill=0)
        for i in range(Self.FRAME_DIM):
            var s = Scalar[Self.dtype](0)
            for j in range(Self.FRAME_DIM):
                s += self.frame[i, j] * self.w[j]
            out[i] = s
        return out^

    def nuisance_at(self, cell: Int) -> List[Scalar[Self.dtype]]:
        var out = List[Scalar[Self.dtype]](length=Self.NUISANCE_DIM, fill=0)
        for k in range(Self.NUISANCE_DIM):
            out[k] = self.nuisance[cell * Self.NUISANCE_DIM + k]
        return out^

    def edge_transport(self, edge: Int) -> SqMat[Self.FRAME_DIM, Self.dtype]:
        return self.edge_rot[edge].copy()
