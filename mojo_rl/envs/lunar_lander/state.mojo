from mojo_rl.core import State


struct LunarLanderState[dtype: DType](
    Copyable, ImplicitlyCopyable, Movable, State
):
    """Observation state for LunarLander (8D continuous observation)."""

    var x: Scalar[Self.dtype]  # Horizontal position (normalized)
    var y: Scalar[Self.dtype]  # Vertical position (normalized)
    var vx: Scalar[Self.dtype]  # Horizontal velocity (normalized)
    var vy: Scalar[Self.dtype]  # Vertical velocity (normalized)
    var angle: Scalar[Self.dtype]  # Angle (radians)
    var angular_velocity: Scalar[Self.dtype]  # Angular velocity (normalized)
    var left_leg_contact: Scalar[Self.dtype]  # 1.0 if touching, 0.0 otherwise
    var right_leg_contact: Scalar[Self.dtype]  # 1.0 if touching, 0.0 otherwise

    fn __init__(out self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.left_leg_contact = 0.0
        self.right_leg_contact = 0.0

    fn __init__(out self, *, deinit take: Self):
        self.x = take.x
        self.y = take.y
        self.vx = take.vx
        self.vy = take.vy
        self.angle = take.angle
        self.angular_velocity = take.angular_velocity
        self.left_leg_contact = take.left_leg_contact
        self.right_leg_contact = take.right_leg_contact

    fn __init__(out self, *, copy: Self):
        self.x = copy.x
        self.y = copy.y
        self.vx = copy.vx
        self.vy = copy.vy
        self.angle = copy.angle
        self.angular_velocity = copy.angular_velocity
        self.left_leg_contact = copy.left_leg_contact
        self.right_leg_contact = copy.right_leg_contact

    fn __eq__(self, other: Self) -> Bool:
        """Check equality of two states."""
        return (
            self.x == other.x
            and self.y == other.y
            and self.vx == other.vx
            and self.vy == other.vy
            and self.angle == other.angle
            and self.angular_velocity == other.angular_velocity
            and self.left_leg_contact == other.left_leg_contact
            and self.right_leg_contact == other.right_leg_contact
        )

    fn to_list(self) -> List[Scalar[Self.dtype]]:
        """Convert to list for agent interface."""
        var result = List[Scalar[Self.dtype]]()
        result.append(self.x)
        result.append(self.y)
        result.append(self.vx)
        result.append(self.vy)
        result.append(self.angle)
        result.append(self.angular_velocity)
        result.append(self.left_leg_contact)
        result.append(self.right_leg_contact)
        return result^
