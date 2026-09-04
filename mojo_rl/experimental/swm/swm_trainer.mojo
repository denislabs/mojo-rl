"""Phase 3: learn an encoder and per-edge transports, then READ the holonomy.

This is the design doc's model A with encoders in front of it. No descent-based
inference yet, no confidence weights, no classification: targets are the
encoder's own outputs (pre-consensus anchors), learning is local per edge, and
the holonomy enters no loss anywhere.

## What is actually being asked

Hypothesis 4.0 says the topologically relevant part of the state is SEPARABLE:
carried by an orthogonally transported channel `u`, with everything else in an
ordinary content channel. E1's observation mixes a transported landmark with a
non-transported per-cell texture, so the encoder has to find the transported
subspace rather than be handed it. If it does, `det H = -1` survives; if it
does not, hypothesis 4.0 is false as stated.

## The degenerate optimum this training loop had to be redesigned around

The transport-consistency loss is TRIVIALLY SATISFIABLE by a place-indexed
constant. If the encoder learns `u = f(place)`, then the transport for
(action, place) — which is itself indexed by place — only has to carry one fixed
vector `f(i)` to one fixed vector `f(i+1)`. In 2D exactly one rotation and
exactly one reflection do that, both with ~zero residual, so the orientation bit
is chosen by noise and `det H` becomes a FAIR COIN.

Measured, before the fix: landmark R^2 = 0.003, nuisance R^2 = 0.5 — the encoder
had learned the texture — and `det H = -1` came out in 2/6 Mobius seeds and 2/6
ORIENTABLE seeds. Same rate. Without the orientable twin on the same binary that
would have read as a successful P1.

The anti-collapse term cannot be measured across places, because the degenerate
solution has excellent across-place variance (different cells, different
textures) — that is what it IS. It has to be measured WITHIN a place. This is
not a trick: a frame is precisely the thing that varies while you stand still,
and a place label is precisely the thing that does not. So the hinge is applied
per place over a minibatch of episodes, where what varies at a fixed cell is the
landmark direction and the lap parity.

## The failure that would masquerade as a refutation

If `u` COLLAPSES to rank one, a reflection about its surviving axis acts exactly
like the identity, every edge fits a rotation, and `det H` comes out `+1` — a
clean-looking "no obstruction" that says nothing about the world. That is an
anti-collapse deficiency, not evidence against the hypothesis, and the two are
indistinguishable from `det H` alone. So `u_anisotropy` is reported beside every
`det H`, and a reading with a collapsed frame channel is INVALID rather than
negative.

## Why det H is invariant to whatever basis the encoder happens to pick

The encoder learns `u = A x` for some unknown `A`. Learned transports are then
`A R A^-1`; constraining them to O(D) forces `A` to be conformal, and
`det(A R A^-1) = det(R)` regardless. The class survives the gauge — which is the
whole reason it, and not the angle, is the observable.
"""

from std.math import abs, sqrt

from .so_d import SqMat
from .rng import Rng
from .mlp import Mlp
from .transport import TransportTable
from .place_graph import PlaceGraph, Edge
from .procrustes import PairBatch
from .envs.mobius_ring import MobiusRing, MobiusConfig, ACTION_FORWARD


@fieldwise_init
struct Phase3Config(Copyable, ImplicitlyCopyable, Movable):
    var epochs: Int
    var episodes_per_epoch: Int
    var batch_episodes: Int
    """Episodes pooled before an Adam step. The per-place variance hinge needs
    several visits to the same cell to have anything to measure."""
    var laps: Int
    var lr_encoder: Float64
    var lr_transport: Float64
    var warmup_epochs: Int
    """Bit flips are held until the encoder has stopped being noise.

    Not a free knob: at 8/60 epochs one seed in eight locked several edges into
    the wrong component early and never recovered (landmark R^2 0.43, eight
    reflected edges, det H = +1). At 20/80 it is 8/8. Enabling the discrete
    choice before the continuous representation has settled is a real failure
    mode, not a tuning detail."""
    var var_weight: Float64
    var cov_weight: Float64
    var seed: UInt64
    var flip_margin: Float64
    """Relative margin the losing branch must beat the incumbent by. 0 = a bare
    argmin, which is what the design doc specifies and what G7 shows chatters."""
    var min_observations: Int

    @staticmethod
    def default() -> Self:
        return Self(80, 24, 8, 3, 0.004, 0.05, 20, 1.0, 1.0, 20260904, 0.25, 64)


@fieldwise_init
struct Phase3Result(Copyable, ImplicitlyCopyable, Movable):
    var det_h: Float64
    var fro_h: Float64
    var n_reflected: Int
    var total_flips: Int
    var transport_residual: Float64
    var u_anisotropy: Float64
    """`sqrt(lambda_min / lambda_max)` of cov(u). Near 0 means a COLLAPSED frame
    channel, and any `det H` read from it is meaningless."""
    var landmark_r2: Float64
    """How much of the true landmark `u` explains. High = the frame channel
    found the transported subspace."""
    var nuisance_r2: Float64
    """How much of the texture leaked into `u`. Low = the split worked."""
    var within_place_std: Float64
    """Mean std of `u` at a FIXED place. Near 0 means the encoder found the
    degenerate place-indexed-constant solution, where the orientation bit is
    decided by noise and `det H` is a coin flip — INVALID, not negative."""


struct TrainedModel[
    OBS_DIM: Int,
    HID: Int,
    LAT: Int,
    D: Int,
    N_ACTIONS: Int,
    N_PLACES: Int,
    dtype: DType = DType.float64,
](Copyable, Movable):
    """A trained encoder plus its transport table, so gates can reuse both."""

    var enc: Mlp[Self.OBS_DIM, Self.HID, Self.LAT, Self.dtype]
    var table: TransportTable[Self.D, Self.N_ACTIONS, Self.N_PLACES, Self.dtype]

    def __init__(
        out self,
        var enc: Mlp[Self.OBS_DIM, Self.HID, Self.LAT, Self.dtype],
        var table: TransportTable[
            Self.D, Self.N_ACTIONS, Self.N_PLACES, Self.dtype
        ],
    ):
        self.enc = enc^
        self.table = table^

    def __init__(out self, *, copy: Self):
        self.enc = copy.enc.copy()
        self.table = copy.table.copy()

    def __init__(out self, *, deinit move: Self):
        self.enc = move.enc^
        self.table = move.table^


struct EncodedRollouts[dtype: DType = DType.float64](Copyable, Movable):
    """Frame-channel encodings of held-out episodes, laid out for the ablations.

    `batches[e]` holds the observed (u_src, u_dst) pairs of edge `e`; `seq_u`
    holds whole trajectories so a loop-closure error can be computed the way
    the numpy oracle computes it.
    """

    var batches: List[PairBatch[2, Self.dtype]]
    var seq_u: List[Scalar[Self.dtype]]
    var n_episodes: Int
    var n_frames: Int

    def __init__(
        out self,
        var batches: List[PairBatch[2, Self.dtype]],
        var seq_u: List[Scalar[Self.dtype]],
        n_episodes: Int,
        n_frames: Int,
    ):
        self.batches = batches^
        self.seq_u = seq_u^
        self.n_episodes = n_episodes
        self.n_frames = n_frames

    def __init__(out self, *, copy: Self):
        self.batches = copy.batches.copy()
        self.seq_u = copy.seq_u.copy()
        self.n_episodes = copy.n_episodes
        self.n_frames = copy.n_frames

    def __init__(out self, *, deinit move: Self):
        self.batches = move.batches^
        self.seq_u = move.seq_u^
        self.n_episodes = move.n_episodes
        self.n_frames = move.n_frames


struct SwmPhase3[
    N_CELLS: Int,
    NUISANCE_DIM: Int,
    OBS_DIM: Int,
    HID: Int,
    CONTENT_DIM: Int,
    dtype: DType = DType.float64,
]:
    """Encoder + transport table trained on the Mobius corridor."""

    comptime D: Int = 2
    comptime LAT: Int = Self.D + Self.CONTENT_DIM
    comptime EncoderT = Mlp[Self.OBS_DIM, Self.HID, Self.LAT, Self.dtype]
    comptime EnvT = MobiusRing[
        Self.N_CELLS, Self.NUISANCE_DIM, Self.OBS_DIM, Self.dtype
    ]
    comptime TableT = TransportTable[Self.D, 2, Self.N_CELLS, Self.dtype]
    comptime ModelT = TrainedModel[
        Self.OBS_DIM, Self.HID, Self.LAT, Self.D, 2, Self.N_CELLS, Self.dtype
    ]

    @staticmethod
    def run(env_cfg: MobiusConfig, cfg: Phase3Config) raises -> Phase3Result:
        var model = Self.train(env_cfg, cfg)
        var env = Self.EnvT(env_cfg)
        var steps = cfg.laps * Self.N_CELLS
        return Self._evaluate(env, model.enc, model.table, cfg, steps, steps + 1)

    @staticmethod
    def train(
        env_cfg: MobiusConfig, cfg: Phase3Config
    ) raises -> Self.ModelT:
        var rng = Rng(cfg.seed)
        var env = Self.EnvT(env_cfg)
        var enc = Self.EncoderT(rng)
        var table = Self.TableT(
            flip_margin=cfg.flip_margin,
            min_observations=cfg.min_observations,
        )

        var steps = cfg.laps * Self.N_CELLS
        var n_frames = steps + 1

        for epoch in range(cfg.epochs):
            var allow_flip = epoch >= cfg.warmup_epochs
            var ep = 0
            while ep < cfg.episodes_per_epoch:
                var batch = cfg.batch_episodes
                if ep + batch > cfg.episodes_per_epoch:
                    batch = cfg.episodes_per_epoch - ep
                enc.zero_grad()

                var obs_all = List[List[Scalar[Self.dtype]]]()
                var hid_all = List[List[Scalar[Self.dtype]]]()
                var lat_all = List[List[Scalar[Self.dtype]]]()
                var dlat_all = List[List[Scalar[Self.dtype]]]()
                var place_all = List[Int]()
                var is_last = List[Bool]()

                for b in range(batch):
                    env.reset(
                        cfg.seed ^ UInt64(epoch * 100000 + (ep + b) * 131 + 1)
                    )
                    for t in range(n_frames):
                        var o = env.observation()
                        var hid = List[Scalar[Self.dtype]](
                            length=Self.HID, fill=0
                        )
                        var lat = List[Scalar[Self.dtype]](
                            length=Self.LAT, fill=0
                        )
                        enc.forward(o, hid, lat)
                        obs_all.append(o^)
                        hid_all.append(hid^)
                        lat_all.append(lat^)
                        dlat_all.append(
                            List[Scalar[Self.dtype]](length=Self.LAT, fill=0)
                        )
                        place_all.append(env.place_id())
                        is_last.append(t == steps)
                        if t < steps:
                            env.step(ACTION_FORWARD)

                var n_all = len(lat_all)
                var scale = Scalar[Self.dtype](1.0 / Float64(batch * steps))
                for t in range(n_all):
                    if is_last[t]:
                        continue
                    var u_src = List[Scalar[Self.dtype]](length=Self.D, fill=0)
                    var u_dst = List[Scalar[Self.dtype]](length=Self.D, fill=0)
                    for i in range(Self.D):
                        u_src[i] = lat_all[t][i]
                        u_dst[i] = lat_all[t + 1][i]
                    var d_src = table.observe(
                        ACTION_FORWARD,
                        place_all[t],
                        u_src,
                        u_dst,
                        cfg.lr_transport,
                        allow_flip,
                    )
                    for i in range(Self.D):
                        dlat_all[t][i] += d_src[i] * scale

                Self._place_variance_grad(lat_all, place_all, dlat_all, cfg)

                var dx = List[Scalar[Self.dtype]](length=Self.OBS_DIM, fill=0)
                for t in range(n_all):
                    enc.backward(obs_all[t], hid_all[t], dlat_all[t], dx)
                enc.adam_step(cfg.lr_encoder)
                ep += batch

        return Self.ModelT(enc^, table^)

    @staticmethod
    def _place_variance_grad(
        lat_all: List[List[Scalar[Self.dtype]]],
        place_all: List[Int],
        mut dlat_all: List[List[Scalar[Self.dtype]]],
        cfg: Phase3Config,
    ):
        """Hinge on the std of `u` AT A FIXED PLACE, plus global decorrelation.

        Measured within a place and not across places, because the degenerate
        `u = f(place)` solution has excellent across-place variance — that is
        exactly what it is — while having zero within-place variance. What
        varies at a fixed cell is the landmark direction and the lap parity,
        which is precisely the content the frame channel is supposed to carry.
        """
        var n = len(lat_all)
        var counts = List[Float64](length=Self.N_CELLS, fill=0)
        var mean = List[Float64](length=Self.N_CELLS * Self.D, fill=0)
        for t in range(n):
            var p = place_all[t]
            counts[p] += 1.0
            for j in range(Self.D):
                mean[p * Self.D + j] += Float64(lat_all[t][j])
        for p in range(Self.N_CELLS):
            if counts[p] > 0:
                for j in range(Self.D):
                    mean[p * Self.D + j] /= counts[p]

        var var_pj = List[Float64](length=Self.N_CELLS * Self.D, fill=0)
        for t in range(n):
            var p = place_all[t]
            for j in range(Self.D):
                var c = Float64(lat_all[t][j]) - mean[p * Self.D + j]
                var_pj[p * Self.D + j] += c * c
        for p in range(Self.N_CELLS):
            if counts[p] > 1:
                for j in range(Self.D):
                    var_pj[p * Self.D + j] /= counts[p] - 1.0

        for t in range(n):
            var p = place_all[t]
            if counts[p] < 2:
                continue
            var denom = counts[p] - 1.0
            for j in range(Self.D):
                var sd = sqrt(var_pj[p * Self.D + j] + 1e-8)
                if sd >= 1.0:
                    continue
                var dl_dvar = -(1.0 - sd) / sd
                var c = Float64(lat_all[t][j]) - mean[p * Self.D + j]
                dlat_all[t][j] += Scalar[Self.dtype](
                    cfg.var_weight * dl_dvar * 2.0 * c / denom
                    / Float64(Self.N_CELLS)
                )

        # Global decorrelation: keeps the two frame dimensions from becoming
        # the same direction, which is rank-one collapse in disguise.
        var m0 = Float64(0)
        var m1 = Float64(0)
        for t in range(n):
            m0 += Float64(lat_all[t][0])
            m1 += Float64(lat_all[t][1])
        m0 /= Float64(n)
        m1 /= Float64(n)
        var cov01 = Float64(0)
        for t in range(n):
            cov01 += (Float64(lat_all[t][0]) - m0) * (Float64(lat_all[t][1]) - m1)
        var d = Float64(n - 1)
        cov01 /= d
        for t in range(n):
            var c0 = Float64(lat_all[t][0]) - m0
            var c1 = Float64(lat_all[t][1]) - m1
            dlat_all[t][0] += Scalar[Self.dtype](
                cfg.cov_weight * 2.0 * cov01 * c1 / d
            )
            dlat_all[t][1] += Scalar[Self.dtype](
                cfg.cov_weight * 2.0 * cov01 * c0 / d
            )

    @staticmethod
    def encode_rollouts(
        model: Self.ModelT,
        env_cfg: MobiusConfig,
        cfg: Phase3Config,
        n_episodes: Int,
        seed_salt: UInt64 = 0xAB1A_7000,
    ) raises -> EncodedRollouts[Self.dtype]:
        """Encode held-out episodes into frame-channel pairs and trajectories.

        The ablations of `ablations.mojo` are fitted on THESE, i.e. on the same
        learned representations model A uses, so the comparison isolates the
        transport model. That is also what the numpy prototype does — it fits
        every arm to one fixed set of observations.
        """
        var env = Self.EnvT(env_cfg)
        var steps = cfg.laps * Self.N_CELLS
        var n_frames = steps + 1
        var batches = List[PairBatch[2, Self.dtype]]()
        for _ in range(Self.N_CELLS):
            batches.append(PairBatch[2, Self.dtype]())
        var seq_u = List[Scalar[Self.dtype]](
            length=n_episodes * n_frames * 2, fill=0
        )

        for ep in range(n_episodes):
            env.reset(cfg.seed ^ (seed_salt + UInt64(ep)))
            var prev = List[Scalar[Self.dtype]](length=Self.D, fill=0)
            var prev_place = 0
            for t in range(n_frames):
                var o = env.observation()
                var hid = List[Scalar[Self.dtype]](length=Self.HID, fill=0)
                var lat = List[Scalar[Self.dtype]](length=Self.LAT, fill=0)
                model.enc.forward(o, hid, lat)
                for i in range(Self.D):
                    seq_u[(ep * n_frames + t) * 2 + i] = lat[i]
                if t > 0:
                    var x = InlineArray[Scalar[Self.dtype], 2](fill=0)
                    var y = InlineArray[Scalar[Self.dtype], 2](fill=0)
                    for i in range(Self.D):
                        x[i] = prev[i]
                        y[i] = lat[i]
                    batches[prev_place].push(x, y)
                for i in range(Self.D):
                    prev[i] = lat[i]
                prev_place = env.place_id()
                if t < steps:
                    env.step(ACTION_FORWARD)

        return EncodedRollouts[Self.dtype](
            batches^, seq_u^, n_episodes, n_frames
        )

    @staticmethod
    def _evaluate(
        mut env: Self.EnvT,
        enc: Self.EncoderT,
        table: Self.TableT,
        cfg: Phase3Config,
        steps: Int,
        n_frames: Int,
    ) raises -> Phase3Result:
        """Read the observables. NONE of this feeds back into any loss."""
        # ---- holonomy of the ring from the learned transports -------------
        var g = PlaceGraph[Self.D, Self.dtype]()
        for _ in range(Self.N_CELLS):
            _ = g.add_place()
        for i in range(Self.N_CELLS):
            _ = g.add_edge(
                Edge.action_edge(i, (i + 1) % Self.N_CELLS, ACTION_FORWARD),
                table.transport_for(ACTION_FORWARD, i),
            )
        g.rebuild_gauge(0)
        var cyc = g.fundamental_cycle_edges()
        var det_h = g.holonomy_det(cyc[0])
        var fro_h = g.holonomy_dist_to_identity(cyc[0])

        # ---- collect a held-out batch of encodings ------------------------
        var us = List[Float64]()
        var lms = List[Float64]()
        var nus = List[Float64]()
        var places = List[Int]()
        var resid = Float64(0)
        var n_trans = 0
        comptime EVAL_EPISODES = 16
        for ep in range(EVAL_EPISODES):
            env.reset(cfg.seed ^ UInt64(0xE7A1_0000 + ep))
            var prev_u = List[Scalar[Self.dtype]](length=Self.D, fill=0)
            var prev_place = 0
            for t in range(n_frames):
                var o = env.observation()
                var hid = List[Scalar[Self.dtype]](length=Self.HID, fill=0)
                var lat = List[Scalar[Self.dtype]](length=Self.LAT, fill=0)
                enc.forward(o, hid, lat)
                var lm = env.true_landmark()
                var nu = env.nuisance_at(env.place_id())
                for i in range(Self.D):
                    us.append(Float64(lat[i]))
                places.append(env.place_id())
                lms.append(Float64(lm[0]))
                lms.append(Float64(lm[1]))
                for k in range(Self.NUISANCE_DIM):
                    nus.append(Float64(nu[k]))
                if t > 0:
                    var r = table.transport_for(ACTION_FORWARD, prev_place)
                    for i in range(Self.D):
                        var pred = Scalar[Self.dtype](0)
                        for j in range(Self.D):
                            pred += r[i, j] * prev_u[j]
                        var e = Float64(pred - lat[i])
                        resid += e * e
                    n_trans += 1
                for i in range(Self.D):
                    prev_u[i] = lat[i]
                prev_place = env.place_id()
                if t < steps:
                    env.step(ACTION_FORWARD)
        if n_trans > 0:
            resid /= Float64(n_trans)

        var n_pts = len(us) // Self.D
        return Phase3Result(
            det_h,
            fro_h,
            table.n_reflected(),
            table.total_flips(),
            resid,
            Self._anisotropy(us, n_pts),
            Self._explained_variance(us, lms, n_pts, Self.D),
            Self._explained_variance(us, nus, n_pts, Self.NUISANCE_DIM),
            Self._within_place_std(us, places, n_pts),
        )

    @staticmethod
    def _within_place_std(us: List[Float64], places: List[Int], n: Int) -> Float64:
        """Mean std of `u` at a fixed place — the degeneracy detector.

        Near zero means the encoder found `u = f(place)`, where the transport
        constraint is vacuous and the orientation bit, hence `det H`, is decided
        by noise. A `det H` read under that condition is INVALID, not negative.
        """
        var counts = List[Float64](length=64, fill=0)
        var mean = List[Float64](length=64 * 2, fill=0)
        var maxp = 0
        for t in range(n):
            var p = places[t]
            if p >= 64:
                continue
            if p > maxp:
                maxp = p
            counts[p] += 1.0
            mean[p * 2] += us[t * 2]
            mean[p * 2 + 1] += us[t * 2 + 1]
        for p in range(maxp + 1):
            if counts[p] > 0:
                mean[p * 2] /= counts[p]
                mean[p * 2 + 1] /= counts[p]
        var acc = List[Float64](length=64 * 2, fill=0)
        for t in range(n):
            var p = places[t]
            if p >= 64:
                continue
            var a = us[t * 2] - mean[p * 2]
            var b = us[t * 2 + 1] - mean[p * 2 + 1]
            acc[p * 2] += a * a
            acc[p * 2 + 1] += b * b
        var total = Float64(0)
        var seen = 0
        for p in range(maxp + 1):
            if counts[p] < 2:
                continue
            total += sqrt(acc[p * 2] / (counts[p] - 1.0))
            total += sqrt(acc[p * 2 + 1] / (counts[p] - 1.0))
            seen += 2
        if seen == 0:
            return 0.0
        return total / Float64(seen)

    @staticmethod
    def _anisotropy(us: List[Float64], n: Int) -> Float64:
        """`sqrt(lambda_min/lambda_max)` of cov(u). 1 = isotropic, 0 = collapsed."""
        if n < 2:
            return 0.0
        var m0 = Float64(0)
        var m1 = Float64(0)
        for t in range(n):
            m0 += us[t * 2]
            m1 += us[t * 2 + 1]
        m0 /= Float64(n)
        m1 /= Float64(n)
        var c00 = Float64(0)
        var c01 = Float64(0)
        var c11 = Float64(0)
        for t in range(n):
            var a = us[t * 2] - m0
            var b = us[t * 2 + 1] - m1
            c00 += a * a
            c01 += a * b
            c11 += b * b
        var d = Float64(n - 1)
        c00 /= d
        c01 /= d
        c11 /= d
        var tr = c00 + c11
        var det = c00 * c11 - c01 * c01
        var disc = tr * tr / 4.0 - det
        if disc < 0:
            disc = 0
        var root = sqrt(disc)
        var l_max = tr / 2.0 + root
        var l_min = tr / 2.0 - root
        if l_max <= 1e-300 or l_min <= 0:
            return 0.0
        return sqrt(l_min / l_max)

    @staticmethod
    def _explained_variance(
        us: List[Float64], ys: List[Float64], n: Int, y_dim: Int
    ) -> Float64:
        """R^2 of the best LINEAR map from `u` to `y`, over all `y` dimensions.

        Linear on purpose: the question is whether the frame channel CONTAINS
        the landmark direction, not whether some further network could dig it
        out of an entangled code.
        """
        if n < 3:
            return 0.0
        # Normal equations for [u, 1] -> y, per output dimension.
        var s00 = Float64(0)
        var s01 = Float64(0)
        var s11 = Float64(0)
        var s0 = Float64(0)
        var s1 = Float64(0)
        var sn = Float64(n)
        for t in range(n):
            var a = us[t * 2]
            var b = us[t * 2 + 1]
            s00 += a * a
            s01 += a * b
            s11 += b * b
            s0 += a
            s1 += b
        var total_ss = Float64(0)
        var resid_ss = Float64(0)
        for k in range(y_dim):
            var ya = Float64(0)
            var yb = Float64(0)
            var ysum = Float64(0)
            for t in range(n):
                var y = ys[t * y_dim + k]
                ya += us[t * 2] * y
                yb += us[t * 2 + 1] * y
                ysum += y
            # Solve the 3x3 system by elimination on the normal equations.
            var a11 = s00
            var a12 = s01
            var a13 = s0
            var a22 = s11
            var a23 = s1
            var a33 = sn
            var det3 = (
                a11 * (a22 * a33 - a23 * a23)
                - a12 * (a12 * a33 - a23 * a13)
                + a13 * (a12 * a23 - a22 * a13)
            )
            if abs(det3) < 1e-18:
                continue
            var i11 = (a22 * a33 - a23 * a23) / det3
            var i12 = -(a12 * a33 - a23 * a13) / det3
            var i13 = (a12 * a23 - a22 * a13) / det3
            var i22 = (a11 * a33 - a13 * a13) / det3
            var i23 = -(a11 * a23 - a12 * a13) / det3
            var i33 = (a11 * a22 - a12 * a12) / det3
            var w0 = i11 * ya + i12 * yb + i13 * ysum
            var w1 = i12 * ya + i22 * yb + i23 * ysum
            var w2 = i13 * ya + i23 * yb + i33 * ysum
            var ymean = ysum / sn
            for t in range(n):
                var y = ys[t * y_dim + k]
                var pred = w0 * us[t * 2] + w1 * us[t * 2 + 1] + w2
                var e = y - pred
                resid_ss += e * e
                var c = y - ymean
                total_ss += c * c
        if total_ss <= 1e-300:
            return 0.0
        var r2 = 1.0 - resid_ss / total_ss
        return 0.0 if r2 < 0 else r2
