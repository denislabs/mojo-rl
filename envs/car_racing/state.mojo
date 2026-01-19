"""CarRacing V2 state struct.

This module defines the state representation for the GPU-enabled CarRacing environment.
"""

from math import sqrt, pi

from core import State


@fieldwise_init
struct CarRacingV2State[DTYPE: DType](
    Copyable, ImplicitlyCopyable, Movable, State
):
    """State for CarRacing V2 environment.

    Observation format (13 values, normalized for neural networks):
    - [0-1]: Position (x, y) normalized by PLAYFIELD
    - [2]: Angle normalized by pi
    - [3-4]: Velocity (vx, vy) normalized by max velocity
    - [5]: Angular velocity normalized
    - [6-7]: Front wheel steering angles (FL, FR)
    - [8-11]: Wheel angular velocities (FL, FR, RL, RR)
    - [12]: Speed indicator (magnitude of velocity)
    """

    # Hull state
    var x: Scalar[Self.DTYPE]
    var y: Scalar[Self.DTYPE]
    var angle: Scalar[Self.DTYPE]
    var vx: Scalar[Self.DTYPE]
    var vy: Scalar[Self.DTYPE]
    var angular_velocity: Scalar[Self.DTYPE]

    # Wheel states
    var wheel_angle_fl: Scalar[Self.DTYPE]
    var wheel_angle_fr: Scalar[Self.DTYPE]
    var wheel_omega_fl: Scalar[Self.DTYPE]
    var wheel_omega_fr: Scalar[Self.DTYPE]
    var wheel_omega_rl: Scalar[Self.DTYPE]
    var wheel_omega_rr: Scalar[Self.DTYPE]

    # Speed indicator
    var speed: Scalar[Self.DTYPE]

    fn __init__(out self):
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.angular_velocity = 0.0
        self.wheel_angle_fl = 0.0
        self.wheel_angle_fr = 0.0
        self.wheel_omega_fl = 0.0
        self.wheel_omega_fr = 0.0
        self.wheel_omega_rl = 0.0
        self.wheel_omega_rr = 0.0
        self.speed = 0.0

    fn __copyinit__(out self, other: Self):
        self.x = other.x
        self.y = other.y
        self.angle = other.angle
        self.vx = other.vx
        self.vy = other.vy
        self.angular_velocity = other.angular_velocity
        self.wheel_angle_fl = other.wheel_angle_fl
        self.wheel_angle_fr = other.wheel_angle_fr
        self.wheel_omega_fl = other.wheel_omega_fl
        self.wheel_omega_fr = other.wheel_omega_fr
        self.wheel_omega_rl = other.wheel_omega_rl
        self.wheel_omega_rr = other.wheel_omega_rr
        self.speed = other.speed

    fn __moveinit__(out self, deinit other: Self):
        self.x = other.x
        self.y = other.y
        self.angle = other.angle
        self.vx = other.vx
        self.vy = other.vy
        self.angular_velocity = other.angular_velocity
        self.wheel_angle_fl = other.wheel_angle_fl
        self.wheel_angle_fr = other.wheel_angle_fr
        self.wheel_omega_fl = other.wheel_omega_fl
        self.wheel_omega_fr = other.wheel_omega_fr
        self.wheel_omega_rl = other.wheel_omega_rl
        self.wheel_omega_rr = other.wheel_omega_rr
        self.speed = other.speed

    fn __eq__(self, other: Self) -> Bool:
        return self.x == other.x and self.y == other.y

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to list for agent interface."""
        var result = List[Scalar[Self.DTYPE]]()
        result.append(self.x)
        result.append(self.y)
        result.append(self.angle)
        result.append(self.vx)
        result.append(self.vy)
        result.append(self.angular_velocity)
        result.append(self.wheel_angle_fl)
        result.append(self.wheel_angle_fr)
        result.append(self.wheel_omega_fl)
        result.append(self.wheel_omega_fr)
        result.append(self.wheel_omega_rl)
        result.append(self.wheel_omega_rr)
        result.append(self.speed)
        return result^

    fn to_list_typed[dtype: DType](self) -> List[Scalar[dtype]]:
        """Convert to list with specified dtype."""
        var result = List[Scalar[dtype]]()
        result.append(Scalar[dtype](self.x))
        result.append(Scalar[dtype](self.y))
        result.append(Scalar[dtype](self.angle))
        result.append(Scalar[dtype](self.vx))
        result.append(Scalar[dtype](self.vy))
        result.append(Scalar[dtype](self.angular_velocity))
        result.append(Scalar[dtype](self.wheel_angle_fl))
        result.append(Scalar[dtype](self.wheel_angle_fr))
        result.append(Scalar[dtype](self.wheel_omega_fl))
        result.append(Scalar[dtype](self.wheel_omega_fr))
        result.append(Scalar[dtype](self.wheel_omega_rl))
        result.append(Scalar[dtype](self.wheel_omega_rr))
        result.append(Scalar[dtype](self.speed))
        return result^

    @staticmethod
    fn from_buffer[
        BATCH: Int, STATE_SIZE: Int, OBS_OFFSET: Int, OBS_DIM: Int,
    ](
        state: LayoutTensor[
            DType.float32, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
    ) -> CarRacingV2State[Self.DTYPE]:
        """Create state from GPU buffer."""
        from layout import Layout, LayoutTensor

        var result = CarRacingV2State[Self.DTYPE]()
        result.x = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 0])
        result.y = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 1])
        result.angle = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 2])
        result.vx = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 3])
        result.vy = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 4])
        result.angular_velocity = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 5])
        result.wheel_angle_fl = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 6])
        result.wheel_angle_fr = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 7])
        result.wheel_omega_fl = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 8])
        result.wheel_omega_fr = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 9])
        result.wheel_omega_rl = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 10])
        result.wheel_omega_rr = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 11])
        result.speed = Scalar[Self.DTYPE](state[env, OBS_OFFSET + 12])
        return result^
