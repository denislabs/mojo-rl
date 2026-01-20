"""Pendulum V2 state struct.

Observation state for the GPU-enabled Pendulum environment.
"""

from math import cos, sin

from core import State
from layout import Layout, LayoutTensor


@fieldwise_init
struct PendulumV2State[DTYPE: DType](
    Copyable, ImplicitlyCopyable, Movable, State
):
    """Observation state for Pendulum V2 (3D continuous observation).

    Observation format (normalized for neural networks):
    - [0]: cos(theta) in [-1, 1]
    - [1]: sin(theta) in [-1, 1]
    - [2]: theta_dot (angular velocity) in [-8, 8]
    """

    var cos_theta: Scalar[Self.DTYPE]
    var sin_theta: Scalar[Self.DTYPE]
    var theta_dot: Scalar[Self.DTYPE]

    fn __init__(out self):
        self.cos_theta = 1.0  # theta=0 (pointing up)
        self.sin_theta = 0.0
        self.theta_dot = 0.0

    fn __copyinit__(out self, other: Self):
        self.cos_theta = other.cos_theta
        self.sin_theta = other.sin_theta
        self.theta_dot = other.theta_dot

    fn __moveinit__(out self, deinit other: Self):
        self.cos_theta = other.cos_theta
        self.sin_theta = other.sin_theta
        self.theta_dot = other.theta_dot

    fn __eq__(self, other: Self) -> Bool:
        return (
            self.cos_theta == other.cos_theta
            and self.sin_theta == other.sin_theta
            and self.theta_dot == other.theta_dot
        )

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to list for agent interface."""
        var result = List[Scalar[Self.DTYPE]]()
        result.append(self.cos_theta)
        result.append(self.sin_theta)
        result.append(self.theta_dot)
        return result^

    fn to_list_typed[dtype: DType](self) -> List[Scalar[dtype]]:
        """Convert to list with specified dtype."""
        var result = List[Scalar[dtype]]()
        result.append(Scalar[dtype](self.cos_theta))
        result.append(Scalar[dtype](self.sin_theta))
        result.append(Scalar[dtype](self.theta_dot))
        return result^

    @staticmethod
    fn from_theta(
        theta: Scalar[Self.DTYPE], theta_dot: Scalar[Self.DTYPE]
    ) -> PendulumV2State[Self.DTYPE]:
        """Create state from raw physics values."""
        return PendulumV2State[Self.DTYPE](
            cos_theta=Scalar[Self.DTYPE](cos(Float64(theta))),
            sin_theta=Scalar[Self.DTYPE](sin(Float64(theta))),
            theta_dot=theta_dot,
        )

    @staticmethod
    fn from_buffer[
        BATCH: Int,
        STATE_SIZE: Int,
    ](
        state: LayoutTensor[
            DType.float32, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
    ) -> PendulumV2State[Self.DTYPE]:
        """Create state from GPU buffer by reading theta and computing observation.

        Args:
            state: GPU state buffer with shape [BATCH, STATE_SIZE].
            env: Environment index.

        Returns:
            PendulumV2State with observation values.
        """
        # Read raw physics values
        var theta = state[env, 0]  # THETA offset
        var theta_dot = state[env, 1]  # THETA_DOT offset

        # Compute observation
        return PendulumV2State[Self.DTYPE](
            cos_theta=Scalar[Self.DTYPE](cos(Float64(theta))),
            sin_theta=Scalar[Self.DTYPE](sin(Float64(theta))),
            theta_dot=Scalar[Self.DTYPE](theta_dot),
        )

    @staticmethod
    fn from_obs_buffer[
        BATCH: Int,
        OBS_DIM: Int,
    ](
        obs: LayoutTensor[
            DType.float32, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ) -> PendulumV2State[Self.DTYPE]:
        """Create state directly from observation buffer.

        Args:
            obs: Observation buffer with shape [BATCH, OBS_DIM].
            env: Environment index.

        Returns:
            PendulumV2State with observation values.
        """
        return PendulumV2State[Self.DTYPE](
            cos_theta=Scalar[Self.DTYPE](obs[env, 0]),
            sin_theta=Scalar[Self.DTYPE](obs[env, 1]),
            theta_dot=Scalar[Self.DTYPE](obs[env, 2]),
        )
