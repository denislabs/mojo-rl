"""Motor controllers for 3D joints.

Provides PD (Proportional-Derivative) controllers for joint actuation,
commonly used in MuJoCo-style environments.
"""

from math import sqrt
from math3d import Vec3


struct PDController:
    """PD controller parameters for joint motors.

    The control law is:
        torque = kp * (target - current) - kd * velocity

    The torque is clamped to ±max_force.
    """

    var kp: Float64  # Proportional gain (stiffness)
    var kd: Float64  # Derivative gain (damping)
    var max_force: Float64  # Maximum torque magnitude

    fn __init__(
        out self,
        kp: Float64 = 100.0,
        kd: Float64 = 10.0,
        max_force: Float64 = 100.0,
    ):
        """Initialize PD controller with default gains."""
        self.kp = kp
        self.kd = kd
        self.max_force = max_force

    fn compute_torque(self, target: Float64, current: Float64, velocity: Float64) -> Float64:
        """Compute control torque.

        Args:
            target: Target joint position (angle).
            current: Current joint position.
            velocity: Current joint velocity.

        Returns:
            Clamped control torque.
        """
        var error = target - current
        var torque = self.kp * error - self.kd * velocity

        # Clamp to max force
        if torque > self.max_force:
            return self.max_force
        if torque < -self.max_force:
            return -self.max_force
        return torque


struct Motor3D:
    """General motor controller for 3D joints.

    Supports multiple control modes:
    - TORQUE: Direct torque control
    - VELOCITY: Velocity tracking with damping
    - POSITION: Position tracking with PD control
    """

    # Control modes
    comptime MODE_TORQUE: Int = 0
    comptime MODE_VELOCITY: Int = 1
    comptime MODE_POSITION: Int = 2

    var mode: Int
    var kp: Float64  # Position gain
    var kd: Float64  # Velocity/damping gain
    var max_force: Float64

    fn __init__(
        out self,
        mode: Int = 2,  # Default to position control
        kp: Float64 = 100.0,
        kd: Float64 = 10.0,
        max_force: Float64 = 100.0,
    ):
        """Initialize motor controller."""
        self.mode = mode
        self.kp = kp
        self.kd = kd
        self.max_force = max_force

    fn compute_torque(
        self,
        target: Float64,  # Target position or velocity
        current_pos: Float64,
        current_vel: Float64,
    ) -> Float64:
        """Compute control torque based on mode.

        Args:
            target: Target value (position for POSITION mode, velocity for VELOCITY mode,
                   torque for TORQUE mode).
            current_pos: Current joint position.
            current_vel: Current joint velocity.

        Returns:
            Control torque.
        """
        var torque: Float64

        if self.mode == Self.MODE_TORQUE:
            # Direct torque control
            torque = target
        elif self.mode == Self.MODE_VELOCITY:
            # Velocity tracking: torque = kd * (target_vel - current_vel)
            torque = self.kd * (target - current_vel)
        else:  # MODE_POSITION
            # PD position control
            var error = target - current_pos
            torque = self.kp * error - self.kd * current_vel

        # Clamp
        if torque > self.max_force:
            return self.max_force
        if torque < -self.max_force:
            return -self.max_force
        return torque


# =============================================================================
# MuJoCo-Style Motor Presets
# =============================================================================


fn make_hopper_motors() -> List[PDController]:
    """Create motor controllers for Hopper environment.

    Hopper has 3 motors: hip, knee, ankle
    """
    var motors = List[PDController]()
    motors.append(PDController(kp=100.0, kd=10.0, max_force=100.0))  # Hip
    motors.append(PDController(kp=100.0, kd=10.0, max_force=100.0))  # Knee
    motors.append(PDController(kp=100.0, kd=10.0, max_force=100.0))  # Ankle
    return motors


fn make_walker_motors() -> List[PDController]:
    """Create motor controllers for Walker2d environment.

    Walker has 6 motors: 3 per leg (hip, knee, ankle)
    """
    var motors = List[PDController]()
    for i in range(6):
        motors.append(PDController(kp=100.0, kd=10.0, max_force=100.0))
    return motors


fn make_cheetah_motors() -> List[PDController]:
    """Create motor controllers for HalfCheetah environment.

    Cheetah has 6 motors: 3 back leg + 3 front leg
    """
    var motors = List[PDController]()
    for i in range(6):
        motors.append(PDController(kp=120.0, kd=12.0, max_force=120.0))
    return motors


fn make_ant_motors() -> List[PDController]:
    """Create motor controllers for Ant environment.

    Ant has 8 motors: 2 per leg (hip, ankle) × 4 legs
    """
    var motors = List[PDController]()
    for i in range(8):
        motors.append(PDController(kp=100.0, kd=10.0, max_force=150.0))
    return motors


fn make_humanoid_motors() -> List[PDController]:
    """Create motor controllers for Humanoid environment.

    Humanoid has 17 motors for various joints.
    Different joints have different gains.
    """
    var motors = List[PDController]()

    # Abdomen (3 DOF)
    motors.append(PDController(kp=100.0, kd=10.0, max_force=100.0))
    motors.append(PDController(kp=100.0, kd=10.0, max_force=100.0))
    motors.append(PDController(kp=100.0, kd=10.0, max_force=100.0))

    # Legs (6 DOF each, 12 total)
    for _ in range(12):
        motors.append(PDController(kp=100.0, kd=10.0, max_force=100.0))

    # Arms (2 DOF, simplified)
    motors.append(PDController(kp=50.0, kd=5.0, max_force=50.0))
    motors.append(PDController(kp=50.0, kd=5.0, max_force=50.0))

    return motors
