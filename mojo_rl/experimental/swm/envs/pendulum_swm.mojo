"""Pendulum — the transfer test: a PHYSICAL world, where the transport must
read the state (Phase 9).

Every world before this one was built to have the structure the method wants:
a group action on a frame, plus nuisance that does not transport. Pendulum is
not built for anything. It is picked because it is the smallest physical
system in this repo whose state contains an exact `SO(2)` action:

    theta_dot' = theta_dot + (3g/2l sin theta + 3/(m l^2) u) dt   (clipped)
    theta'     = theta + theta_dot' dt

so `(cos theta, sin theta)` is carried by `R(theta_dot' dt)` EXACTLY — an
isometry, gated to 1e-12 below — while `theta_dot` evolves nonlinearly. The
split hypothesis 4.0 asks for exists here analytically, which is what makes
the world worth the trip.

## The rule this world bends, deliberately

Design doc v3 §4.2: `R_e = rho(a, l(p), c)` must never be conditioned on the
latent COORDINATES; `c` must be exogenous or pass a severe bottleneck. Here
the rotation angle depends on `theta_dot`, which is CONTENT. So the transport
has to read the content, and the only legitimate route is the bottleneck.

The bottleneck is the place index itself. On E1 the transports were indexed by
(action, PLACE) because the ring is inhomogeneous — the same action rotates
the frame differently in each cell. The pendulum is homogeneous: a rotation is
a rotation wherever you stand, and what selects it is the velocity. So the
place index becomes a **velocity bin**, `N_BINS` of them over
`[-MAX_SPEED, MAX_SPEED]`, and everything downstream is unchanged: the
transport table is indexed by (action, bin), and the trainer's anti-collapse
hinge — applied per place — becomes per velocity bin, which is exactly the
right analogue. At a fixed velocity what varies is the angle, i.e. the frame;
that is the same sentence as "a frame is what varies while you stand still".

`log2(N_BINS)` bits is a severe bottleneck by any reading. What it costs is
what Phase 9 measures.

## What has no content here, and must not be claimed

Every transport is a rotation, so `det H = +1` around every cycle by
construction: the pendulum's frame bundle is trivial and there is NO
obstruction to detect. The `Z/2` machinery is asked for one thing only — that
it does not manufacture an obstruction on a physical system — and anything
more would be vacuous.

## Ground truth

`true_landmark` (the transported `(cos theta, sin theta)`), `nuisance_at` (the
bin's representative velocity), `theta`, `theta_dot` and `place_id` are
ORACLES, for the gates. The encoder sees only `observation()`.
"""

from std.collections import InlineArray
from std.math import abs, cos, sin, sqrt, pi

from ..so_d import SqMat
from ..rng import Rng
from ..world import SwmWorld

comptime TORQUE_LEFT: Int = 0
comptime TORQUE_NONE: Int = 1
comptime TORQUE_RIGHT: Int = 2

comptime MAX_SPEED: Float64 = 8.0
comptime MAX_TORQUE: Float64 = 2.0
comptime DT: Float64 = 0.05
comptime GRAVITY: Float64 = 10.0
comptime MASS: Float64 = 1.0
comptime LENGTH: Float64 = 1.0


@fieldwise_init
struct PendulumSwmConfig(Copyable, ImplicitlyCopyable, Movable):
    var world_seed: UInt64
    var obs_noise: Float64
    var mixed_obs: Bool
    """`True`: the observation is an overcomplete mixing of
    `(cos theta, sin theta, theta_dot / MAX_SPEED)`, so the encoder must FIND
    the transported subspace — the E1 discipline. `False`: the raw Gymnasium
    observation, which hands the split over on a plate and exists as the
    easy-mode control."""
    var start_still: Bool
    """Reset near the bottom at rest (the Gymnasium start is uniform)."""

    @staticmethod
    def default(world_seed: UInt64 = 20260904) -> Self:
        return Self(world_seed, 0.01, True, False)

    @staticmethod
    def raw_obs(world_seed: UInt64 = 20260904) -> Self:
        var c = Self.default(world_seed)
        c.mixed_obs = False
        return c


struct PendulumSwm[
    N_BINS: Int,
    OBS_DIM: Int,
](SwmWorld):
    """Gymnasium Pendulum-v1 dynamics, observed through a mixing, with the
    velocity bin as the place index."""

    comptime N_PLACES: Int = Self.N_BINS
    comptime NUISANCE_DIM: Int = 1
    comptime LATENT_DIM: Int = 3
    comptime dtype: DType = DType.float64
    comptime ELEM: DType = DType.float64

    var cfg: PendulumSwmConfig
    var mix: List[Scalar[Self.dtype]]
    var theta: Float64
    var theta_dot: Float64
    var rng: Rng

    def __init__(out self, cfg: PendulumSwmConfig) raises:
        comptime assert Self.N_BINS >= 2, "the velocity bottleneck needs bins"
        comptime assert (
            Self.OBS_DIM >= Self.LATENT_DIM
        ), "observation must not lose the latent"
        self.cfg = cfg
        var wr = Rng(cfg.world_seed)
        self.mix = List[Scalar[Self.dtype]](
            length=Self.OBS_DIM * Self.LATENT_DIM, fill=0
        )
        for r in range(Self.OBS_DIM):
            for c in range(Self.LATENT_DIM):
                var v = wr.uniform_range(-1.0, 1.0)
                if abs(v) < 0.25:
                    v = 0.25 if v >= 0 else -0.25
                self.mix[r * Self.LATENT_DIM + c] = Scalar[Self.dtype](v)
        self.theta = pi
        self.theta_dot = 0.0
        self.rng = Rng(cfg.world_seed ^ 0xA5A5_A5A5_A5A5_A5A5)

    def __init__(out self, *, copy: Self):
        self.cfg = copy.cfg
        self.mix = copy.mix.copy()
        self.theta = copy.theta
        self.theta_dot = copy.theta_dot
        self.rng = copy.rng

    def __init__(out self, *, deinit move: Self):
        self.cfg = move.cfg
        self.mix = move.mix^
        self.theta = move.theta
        self.theta_dot = move.theta_dot
        self.rng = move.rng

    # -- dynamics -------------------------------------------------------------

    def reset(mut self, seed: UInt64) raises:
        self.rng = Rng(seed)
        if self.cfg.start_still:
            self.theta = pi + self.rng.uniform_range(-0.2, 0.2)
            self.theta_dot = self.rng.uniform_range(-0.5, 0.5)
        else:
            self.theta = self.rng.uniform_range(-pi, pi)
            self.theta_dot = self.rng.uniform_range(-1.0, 1.0)

    @staticmethod
    def torque_of(action: Int) -> Float64:
        if action == TORQUE_LEFT:
            return -MAX_TORQUE
        if action == TORQUE_RIGHT:
            return MAX_TORQUE
        return 0.0

    def step(mut self, action: Int) raises:
        if action < 0 or action > 2:
            raise Error("PendulumSwm.step: action must be 0, 1 or 2")
        var u = Self.torque_of(action)
        var acc = (
            3.0 * GRAVITY / (2.0 * LENGTH) * sin(self.theta)
            + 3.0 / (MASS * LENGTH * LENGTH) * u
        )
        var nd = self.theta_dot + acc * DT
        if nd > MAX_SPEED:
            nd = MAX_SPEED
        elif nd < -MAX_SPEED:
            nd = -MAX_SPEED
        self.theta_dot = nd
        self.theta += nd * DT

    def explore_action(mut self) -> Int:
        """Uniform over the three torques: enough to sweep the state space at
        this horizon, and it makes the transport identifiable per bin."""
        var r = self.rng.uniform()
        if r < 0.3333333333333333:
            return TORQUE_LEFT
        if r < 0.6666666666666666:
            return TORQUE_NONE
        return TORQUE_RIGHT

    # -- what the agent sees --------------------------------------------------

    def observation(mut self) -> List[Scalar[Self.dtype]]:
        var latent = List[Scalar[Self.dtype]](length=Self.LATENT_DIM, fill=0)
        latent[0] = Scalar[Self.dtype](cos(self.theta))
        latent[1] = Scalar[Self.dtype](sin(self.theta))
        latent[2] = Scalar[Self.dtype](self.theta_dot / MAX_SPEED)
        var obs = List[Scalar[Self.dtype]](length=Self.OBS_DIM, fill=0)
        for r in range(Self.OBS_DIM):
            var s = Scalar[Self.dtype](0)
            if self.cfg.mixed_obs:
                for c in range(Self.LATENT_DIM):
                    s += self.mix[r * Self.LATENT_DIM + c] * latent[c]
            elif r < Self.LATENT_DIM:
                s = latent[r]
            obs[r] = s + Scalar[Self.dtype](
                self.rng.normal() * self.cfg.obs_noise
            )
        return obs^

    # -- oracles --------------------------------------------------------------

    def bin_of(self, speed: Float64) -> Int:
        var t = (speed + MAX_SPEED) / (2.0 * MAX_SPEED)
        var b = Int(t * Float64(Self.N_BINS))
        if b < 0:
            b = 0
        if b >= Self.N_BINS:
            b = Self.N_BINS - 1
        return b

    def bin_speed(self, b: Int) -> Float64:
        """Representative (centre) speed of a bin."""
        return (
            -MAX_SPEED
            + 2.0 * MAX_SPEED * (Float64(b) + 0.5) / Float64(Self.N_BINS)
        )

    def place_id(self) -> Int:
        """The velocity bin — the bottleneck the transport is allowed to read."""
        return self.bin_of(self.theta_dot)

    def place_label(self) -> Int:
        return self.place_id()

    def true_landmark(self) -> InlineArray[Scalar[Self.dtype], 2]:
        """`(cos theta, sin theta)` — the exactly transported part."""
        var out = InlineArray[Scalar[Self.dtype], 2](fill=0)
        out[0] = Scalar[Self.dtype](cos(self.theta))
        out[1] = Scalar[Self.dtype](sin(self.theta))
        return out^

    def nuisance_at(self, cell: Int) -> List[Scalar[Self.dtype]]:
        """The bin's representative speed: the quantity `u` must NOT encode.

        Quantized, so an R^2 against it slightly UNDER-states leakage; the
        gate also measures against the exact `theta_dot`.
        """
        var out = List[Scalar[Self.dtype]](length=1, fill=0)
        out[0] = Scalar[Self.dtype](self.bin_speed(cell) / MAX_SPEED)
        return out^

    def speed(self) -> Float64:
        return self.theta_dot

    def angle(self) -> Float64:
        return self.theta

    def true_transport(self) -> SqMat[2, Self.dtype]:
        """`R(theta_dot * DT)`: the rotation the NEXT step will apply, given
        the velocity already updated by that step. Gated exact."""
        var a = self.theta_dot * DT
        var m = SqMat[2, Self.dtype]()
        m[0, 0] = Scalar[Self.dtype](cos(a))
        m[0, 1] = Scalar[Self.dtype](-sin(a))
        m[1, 0] = Scalar[Self.dtype](sin(a))
        m[1, 1] = Scalar[Self.dtype](cos(a))
        return m^

    def reward(self) -> Float64:
        """Gymnasium's cost, negated: upright and slow is best."""
        var t = self.theta
        while t > pi:
            t -= 2.0 * pi
        while t < -pi:
            t += 2.0 * pi
        return -(t * t + 0.1 * self.theta_dot * self.theta_dot)
