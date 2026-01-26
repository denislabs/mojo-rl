from core import State
from .constants import BipedalWalkerConstants


struct BipedalWalkerState[DTYPE: DType](
    Copyable, ImplicitlyCopyable, Movable, State
):
    """Observation state for BipedalWalker (24D continuous observation)."""

    comptime NUM_LIDAR: Int = BipedalWalkerConstants[Self.DTYPE].NUM_LIDAR

    # Hull state (4)
    var hull_angle: Scalar[Self.DTYPE]
    var hull_angular_velocity: Scalar[Self.DTYPE]
    var vel_x: Scalar[Self.DTYPE]
    var vel_y: Scalar[Self.DTYPE]

    # Leg 1 state (5): hip, knee, contact
    var hip1_angle: Scalar[Self.DTYPE]
    var hip1_speed: Scalar[Self.DTYPE]
    var knee1_angle: Scalar[Self.DTYPE]
    var knee1_speed: Scalar[Self.DTYPE]
    var leg1_contact: Scalar[Self.DTYPE]

    # Leg 2 state (5)
    var hip2_angle: Scalar[Self.DTYPE]
    var hip2_speed: Scalar[Self.DTYPE]
    var knee2_angle: Scalar[Self.DTYPE]
    var knee2_speed: Scalar[Self.DTYPE]
    var leg2_contact: Scalar[Self.DTYPE]

    # Lidar (10)
    var lidar: InlineArray[Scalar[Self.DTYPE], Self.NUM_LIDAR]

    fn __init__(out self):
        self.hull_angle = 0.0
        self.hull_angular_velocity = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.hip1_angle = 0.0
        self.hip1_speed = 0.0
        self.knee1_angle = 0.0
        self.knee1_speed = 0.0
        self.leg1_contact = 0.0
        self.hip2_angle = 0.0
        self.hip2_speed = 0.0
        self.knee2_angle = 0.0
        self.knee2_speed = 0.0
        self.leg2_contact = 0.0
        self.lidar = InlineArray[Scalar[Self.DTYPE], Self.NUM_LIDAR](1.0)

    fn __copyinit__(out self, other: Self):
        self.hull_angle = other.hull_angle
        self.hull_angular_velocity = other.hull_angular_velocity
        self.vel_x = other.vel_x
        self.vel_y = other.vel_y
        self.hip1_angle = other.hip1_angle
        self.hip1_speed = other.hip1_speed
        self.knee1_angle = other.knee1_angle
        self.knee1_speed = other.knee1_speed
        self.leg1_contact = other.leg1_contact
        self.hip2_angle = other.hip2_angle
        self.hip2_speed = other.hip2_speed
        self.knee2_angle = other.knee2_angle
        self.knee2_speed = other.knee2_speed
        self.leg2_contact = other.leg2_contact
        self.lidar = InlineArray[Scalar[Self.DTYPE], Self.NUM_LIDAR](fill=Scalar[Self.DTYPE](1.0))
        for i in range(Self.NUM_LIDAR):
            self.lidar[i] = other.lidar[i]

    fn __moveinit__(out self, deinit other: Self):
        self.hull_angle = other.hull_angle
        self.hull_angular_velocity = other.hull_angular_velocity
        self.vel_x = other.vel_x
        self.vel_y = other.vel_y
        self.hip1_angle = other.hip1_angle
        self.hip1_speed = other.hip1_speed
        self.knee1_angle = other.knee1_angle
        self.knee1_speed = other.knee1_speed
        self.leg1_contact = other.leg1_contact
        self.hip2_angle = other.hip2_angle
        self.hip2_speed = other.hip2_speed
        self.knee2_angle = other.knee2_angle
        self.knee2_speed = other.knee2_speed
        self.leg2_contact = other.leg2_contact
        self.lidar = InlineArray[Scalar[Self.DTYPE], Self.NUM_LIDAR](fill=Scalar[Self.DTYPE](1.0))
        for i in range(Self.NUM_LIDAR):
            self.lidar[i] = other.lidar[i]

    fn __eq__(self, other: Self) -> Bool:
        return (
            self.hull_angle == other.hull_angle
            and self.hull_angular_velocity == other.hull_angular_velocity
            and self.vel_x == other.vel_x
            and self.vel_y == other.vel_y
        )

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to 24D list for agent interface."""
        var result = List[Scalar[Self.DTYPE]]()
        result.append(self.hull_angle)
        result.append(self.hull_angular_velocity)
        result.append(self.vel_x)
        result.append(self.vel_y)
        result.append(self.hip1_angle)
        result.append(self.hip1_speed)
        result.append(self.knee1_angle)
        result.append(self.knee1_speed)
        result.append(self.leg1_contact)
        result.append(self.hip2_angle)
        result.append(self.hip2_speed)
        result.append(self.knee2_angle)
        result.append(self.knee2_speed)
        result.append(self.leg2_contact)
        for i in range(Self.NUM_LIDAR):
            result.append(self.lidar[i])
        return result^

    fn to_list_typed[dtype: DType](self) -> List[Scalar[dtype]]:
        """Convert to 24D list with specified dtype."""
        var result = List[Scalar[dtype]]()
        result.append(Scalar[dtype](self.hull_angle))
        result.append(Scalar[dtype](self.hull_angular_velocity))
        result.append(Scalar[dtype](self.vel_x))
        result.append(Scalar[dtype](self.vel_y))
        result.append(Scalar[dtype](self.hip1_angle))
        result.append(Scalar[dtype](self.hip1_speed))
        result.append(Scalar[dtype](self.knee1_angle))
        result.append(Scalar[dtype](self.knee1_speed))
        result.append(Scalar[dtype](self.leg1_contact))
        result.append(Scalar[dtype](self.hip2_angle))
        result.append(Scalar[dtype](self.hip2_speed))
        result.append(Scalar[dtype](self.knee2_angle))
        result.append(Scalar[dtype](self.knee2_speed))
        result.append(Scalar[dtype](self.leg2_contact))
        for i in range(Self.NUM_LIDAR):
            result.append(Scalar[dtype](self.lidar[i]))
        return result^
