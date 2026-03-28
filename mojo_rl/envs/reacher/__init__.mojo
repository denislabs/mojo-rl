"""Reacher Environment Package.

MuJoCo-style Reacher using the physics3d engine.

A two-jointed robot arm where the goal is to move the end effector (fingertip)
close to a target spawned at a random position.

Observation space (10D):
    cos(q0), cos(q1): cosine of joint angles
    sin(q0), sin(q1): sine of joint angles
    qpos[2:4]: target joint positions (target_x, target_y)
    qvel[0:2]: joint angular velocities
    delta_x, delta_y: fingertip - target world position (x, y)

Action space (2D):
    Continuous torques applied to hinge joints, clipped to [-1, 1].

Reward: -||fingertip - target||_2 - sum(action^2)
Termination: Never (truncated after 50 steps).

Example usage:
    from mojo_rl.envs.reacher import Reacher
    from mojo_rl.core import ContAction

    var env = Reacher()
    var state = env.reset()

    var action = ContAction[2]()
    var result = env.step(action)
"""

from .reacher import Reacher
from .reacher_xml import ReacherModel
from .reacher_config import ReacherConfig
