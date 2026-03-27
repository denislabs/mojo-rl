"""Termination functions for model-based rollouts.

Used by MBPO to determine if a model-predicted state is terminal,
since the dynamics model does not predict termination signals.
Each environment defines its own termination criteria.
"""

from mojo_rl.nn.constants import dtype


trait TerminationFn:
    """Environment-specific termination check for model rollouts."""

    @staticmethod
    def is_terminal(obs: List[Scalar[dtype]]) -> Bool:
        ...


struct NeverTerminate(TerminationFn):
    """No early termination. Use for environments without termination
    conditions (e.g., HalfCheetah, Swimmer, Walker2d)."""

    @staticmethod
    def is_terminal(obs: List[Scalar[dtype]]) -> Bool:
        return False


struct HopperTerminate(TerminationFn):
    """Hopper-v2/v4: terminate if height < 0.7 or |angle| > 0.2.

    Observation layout: [z_pos, angle, ...].
    """

    @staticmethod
    def is_terminal(obs: List[Scalar[dtype]]) -> Bool:
        var height = Float64(obs[0])
        var angle = Float64(obs[1])
        return height < 0.7 or angle > 0.2 or angle < -0.2


struct AntTerminate(TerminationFn):
    """Ant-v2/v4: terminate if height not in [0.2, 1.0].

    Observation layout: [x_pos, y_pos, z_pos, ...] (z_pos at index 0
    after removing x,y from obs in standard Gym wrapper).
    """

    @staticmethod
    def is_terminal(obs: List[Scalar[dtype]]) -> Bool:
        var height = Float64(obs[0])
        return height < 0.2 or height > 1.0


struct InvertedPendulumTerminate(TerminationFn):
    """InvertedPendulum: terminate if |angle| > 0.2.

    Observation layout: [x, x_dot, theta, theta_dot].
    """

    @staticmethod
    def is_terminal(obs: List[Scalar[dtype]]) -> Bool:
        var angle = Float64(obs[2])
        return angle > 0.2 or angle < -0.2
