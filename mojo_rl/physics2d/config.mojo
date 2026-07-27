"""PhysicsConfig — per-step parameters for the strided 2D physics kernels.

Extracted from the deleted `kernel.mojo` (the `PhysicsKernel` orchestrator was
test-only — every production env runs its own fused physics kernel — but this
config struct is the shared parameter carrier for those fused kernels).
"""

from .constants import (
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    DEFAULT_BAUMGARTE,
    DEFAULT_SLOP,
    DEFAULT_VELOCITY_ITERATIONS,
    DEFAULT_POSITION_ITERATIONS,
)


struct PhysicsConfig(Copyable, Movable):
    """Configuration for strided physics simulation.

    Contains all parameters needed for a physics step.
    """

    var gravity_x: Float64
    var gravity_y: Float64
    var dt: Float64
    var friction: Float64
    var restitution: Float64
    var baumgarte: Float64
    var slop: Float64
    var velocity_iterations: Int
    var position_iterations: Int

    def __init__(
        out self,
        gravity_x: Float64 = 0.0,
        gravity_y: Float64 = -10.0,
        dt: Float64 = 0.02,
        friction: Float64 = DEFAULT_FRICTION,
        restitution: Float64 = DEFAULT_RESTITUTION,
        baumgarte: Float64 = DEFAULT_BAUMGARTE,
        slop: Float64 = DEFAULT_SLOP,
        velocity_iterations: Int = DEFAULT_VELOCITY_ITERATIONS,
        position_iterations: Int = DEFAULT_POSITION_ITERATIONS,
    ):
        self.gravity_x = gravity_x
        self.gravity_y = gravity_y
        self.dt = dt
        self.friction = friction
        self.restitution = restitution
        self.baumgarte = baumgarte
        self.slop = slop
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations

    def __init__(out self, *, copy: Self):
        self.gravity_x = copy.gravity_x
        self.gravity_y = copy.gravity_y
        self.dt = copy.dt
        self.friction = copy.friction
        self.restitution = copy.restitution
        self.baumgarte = copy.baumgarte
        self.slop = copy.slop
        self.velocity_iterations = copy.velocity_iterations
        self.position_iterations = copy.position_iterations

    def __init__(out self, *, deinit move: Self):
        self.gravity_x = move.gravity_x
        self.gravity_y = move.gravity_y
        self.dt = move.dt
        self.friction = move.friction
        self.restitution = move.restitution
        self.baumgarte = move.baumgarte
        self.slop = move.slop
        self.velocity_iterations = move.velocity_iterations
        self.position_iterations = move.position_iterations
