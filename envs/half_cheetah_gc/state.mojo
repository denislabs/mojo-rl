"""Half Cheetah GC state representation.

17-dimensional observation space matching MuJoCo Half Cheetah v5:
- qpos[1:9]: 8 position values (excluding rootx for translation invariance)
  - rootz (height)
  - rooty (pitch angle)
  - bthigh, bshin, bfoot (back leg joint angles)
  - fthigh, fshin, ffoot (front leg joint angles)
- qvel[0:9]: 9 velocity values
  - rootx velocity (forward velocity)
  - rootz velocity (vertical velocity)
  - rooty velocity (pitch angular velocity)
  - bthigh, bshin, bfoot velocities
  - fthigh, fshin, ffoot velocities
"""

from core import State


struct HalfCheetahGCState(Copyable, Movable, State):
    """Half Cheetah GC observation state (17D).

    Position observations (8D) - excludes rootx:
      [0]: z_position (rootz qpos) - height of torso
      [1]: y_angle (rooty qpos) - pitch angle of torso
      [2]: bthigh_angle - back thigh joint angle
      [3]: bshin_angle - back shin joint angle
      [4]: bfoot_angle - back foot joint angle
      [5]: fthigh_angle - front thigh joint angle
      [6]: fshin_angle - front shin joint angle
      [7]: ffoot_angle - front foot joint angle

    Velocity observations (9D):
      [8]: x_velocity (rootx qvel) - forward velocity
      [9]: z_velocity (rootz qvel) - vertical velocity
      [10]: y_angular_velocity (rooty qvel) - pitch rate
      [11]: bthigh_velocity - back thigh joint velocity
      [12]: bshin_velocity - back shin joint velocity
      [13]: bfoot_velocity - back foot joint velocity
      [14]: fthigh_velocity - front thigh joint velocity
      [15]: fshin_velocity - front shin joint velocity
      [16]: ffoot_velocity - front foot joint velocity
    """

    # Position observations (8D)
    var z_position: Float64  # [0] rootz qpos
    var y_angle: Float64  # [1] rooty qpos
    var bthigh_angle: Float64  # [2] back thigh joint
    var bshin_angle: Float64  # [3] back shin joint
    var bfoot_angle: Float64  # [4] back foot joint
    var fthigh_angle: Float64  # [5] front thigh joint
    var fshin_angle: Float64  # [6] front shin joint
    var ffoot_angle: Float64  # [7] front foot joint

    # Velocity observations (9D)
    var x_velocity: Float64  # [8] rootx qvel (forward velocity)
    var z_velocity: Float64  # [9] rootz qvel (vertical velocity)
    var y_angular_velocity: Float64  # [10] rooty qvel (pitch rate)
    var bthigh_velocity: Float64  # [11] back thigh joint velocity
    var bshin_velocity: Float64  # [12] back shin joint velocity
    var bfoot_velocity: Float64  # [13] back foot joint velocity
    var fthigh_velocity: Float64  # [14] front thigh joint velocity
    var fshin_velocity: Float64  # [15] front shin joint velocity
    var ffoot_velocity: Float64  # [16] front foot joint velocity

    fn __init__(out self):
        """Initialize state with zeros."""
        self.z_position = 0.0
        self.y_angle = 0.0
        self.bthigh_angle = 0.0
        self.bshin_angle = 0.0
        self.bfoot_angle = 0.0
        self.fthigh_angle = 0.0
        self.fshin_angle = 0.0
        self.ffoot_angle = 0.0
        self.x_velocity = 0.0
        self.z_velocity = 0.0
        self.y_angular_velocity = 0.0
        self.bthigh_velocity = 0.0
        self.bshin_velocity = 0.0
        self.bfoot_velocity = 0.0
        self.fthigh_velocity = 0.0
        self.fshin_velocity = 0.0
        self.ffoot_velocity = 0.0

    fn __init__(
        out self,
        z_position: Float64,
        y_angle: Float64,
        bthigh_angle: Float64,
        bshin_angle: Float64,
        bfoot_angle: Float64,
        fthigh_angle: Float64,
        fshin_angle: Float64,
        ffoot_angle: Float64,
        x_velocity: Float64,
        z_velocity: Float64,
        y_angular_velocity: Float64,
        bthigh_velocity: Float64,
        bshin_velocity: Float64,
        bfoot_velocity: Float64,
        fthigh_velocity: Float64,
        fshin_velocity: Float64,
        ffoot_velocity: Float64,
    ):
        """Initialize state with all values."""
        self.z_position = z_position
        self.y_angle = y_angle
        self.bthigh_angle = bthigh_angle
        self.bshin_angle = bshin_angle
        self.bfoot_angle = bfoot_angle
        self.fthigh_angle = fthigh_angle
        self.fshin_angle = fshin_angle
        self.ffoot_angle = ffoot_angle
        self.x_velocity = x_velocity
        self.z_velocity = z_velocity
        self.y_angular_velocity = y_angular_velocity
        self.bthigh_velocity = bthigh_velocity
        self.bshin_velocity = bshin_velocity
        self.bfoot_velocity = bfoot_velocity
        self.fthigh_velocity = fthigh_velocity
        self.fshin_velocity = fshin_velocity
        self.ffoot_velocity = ffoot_velocity

    fn __copyinit__(out self, read other: Self):
        """Copy constructor."""
        self.z_position = other.z_position
        self.y_angle = other.y_angle
        self.bthigh_angle = other.bthigh_angle
        self.bshin_angle = other.bshin_angle
        self.bfoot_angle = other.bfoot_angle
        self.fthigh_angle = other.fthigh_angle
        self.fshin_angle = other.fshin_angle
        self.ffoot_angle = other.ffoot_angle
        self.x_velocity = other.x_velocity
        self.z_velocity = other.z_velocity
        self.y_angular_velocity = other.y_angular_velocity
        self.bthigh_velocity = other.bthigh_velocity
        self.bshin_velocity = other.bshin_velocity
        self.bfoot_velocity = other.bfoot_velocity
        self.fthigh_velocity = other.fthigh_velocity
        self.fshin_velocity = other.fshin_velocity
        self.ffoot_velocity = other.ffoot_velocity

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.z_position = other.z_position
        self.y_angle = other.y_angle
        self.bthigh_angle = other.bthigh_angle
        self.bshin_angle = other.bshin_angle
        self.bfoot_angle = other.bfoot_angle
        self.fthigh_angle = other.fthigh_angle
        self.fshin_angle = other.fshin_angle
        self.ffoot_angle = other.ffoot_angle
        self.x_velocity = other.x_velocity
        self.z_velocity = other.z_velocity
        self.y_angular_velocity = other.y_angular_velocity
        self.bthigh_velocity = other.bthigh_velocity
        self.bshin_velocity = other.bshin_velocity
        self.bfoot_velocity = other.bfoot_velocity
        self.fthigh_velocity = other.fthigh_velocity
        self.fshin_velocity = other.fshin_velocity
        self.ffoot_velocity = other.ffoot_velocity

    fn to_list(self) -> List[Float64]:
        """Convert state to a list of 17 float values."""
        var result = List[Float64](capacity=17)
        # Position observations (8D)
        result.append(self.z_position)
        result.append(self.y_angle)
        result.append(self.bthigh_angle)
        result.append(self.bshin_angle)
        result.append(self.bfoot_angle)
        result.append(self.fthigh_angle)
        result.append(self.fshin_angle)
        result.append(self.ffoot_angle)
        # Velocity observations (9D)
        result.append(self.x_velocity)
        result.append(self.z_velocity)
        result.append(self.y_angular_velocity)
        result.append(self.bthigh_velocity)
        result.append(self.bshin_velocity)
        result.append(self.bfoot_velocity)
        result.append(self.fthigh_velocity)
        result.append(self.fshin_velocity)
        result.append(self.ffoot_velocity)
        return result^

    fn __getitem__(self, idx: Int) -> Float64:
        """Access observation by index (0-16)."""
        if idx == 0:
            return self.z_position
        elif idx == 1:
            return self.y_angle
        elif idx == 2:
            return self.bthigh_angle
        elif idx == 3:
            return self.bshin_angle
        elif idx == 4:
            return self.bfoot_angle
        elif idx == 5:
            return self.fthigh_angle
        elif idx == 6:
            return self.fshin_angle
        elif idx == 7:
            return self.ffoot_angle
        elif idx == 8:
            return self.x_velocity
        elif idx == 9:
            return self.z_velocity
        elif idx == 10:
            return self.y_angular_velocity
        elif idx == 11:
            return self.bthigh_velocity
        elif idx == 12:
            return self.bshin_velocity
        elif idx == 13:
            return self.bfoot_velocity
        elif idx == 14:
            return self.fthigh_velocity
        elif idx == 15:
            return self.fshin_velocity
        elif idx == 16:
            return self.ffoot_velocity
        else:
            return 0.0

    fn __eq__(self, other: Self) -> Bool:
        """Check equality with another state."""
        return (
            self.z_position == other.z_position
            and self.y_angle == other.y_angle
            and self.bthigh_angle == other.bthigh_angle
            and self.bshin_angle == other.bshin_angle
            and self.bfoot_angle == other.bfoot_angle
            and self.fthigh_angle == other.fthigh_angle
            and self.fshin_angle == other.fshin_angle
            and self.ffoot_angle == other.ffoot_angle
            and self.x_velocity == other.x_velocity
            and self.z_velocity == other.z_velocity
            and self.y_angular_velocity == other.y_angular_velocity
            and self.bthigh_velocity == other.bthigh_velocity
            and self.bshin_velocity == other.bshin_velocity
            and self.bfoot_velocity == other.bfoot_velocity
            and self.fthigh_velocity == other.fthigh_velocity
            and self.fshin_velocity == other.fshin_velocity
            and self.ffoot_velocity == other.ffoot_velocity
        )

    fn __ne__(self, other: Self) -> Bool:
        """Check inequality with another state."""
        return not self.__eq__(other)
