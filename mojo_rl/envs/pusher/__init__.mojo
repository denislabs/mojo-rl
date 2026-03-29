"""Pusher Environment Package.

MuJoCo-style Pusher using the physics3d engine.

A 7-DOF robotic arm that pushes a cylinder (object) to a goal position.
Zero gravity table-top manipulation task.

Observation space (23D):
    qpos[0:7]: 7 arm joint angles
    qvel[0:7]: 7 arm joint velocities
    tips_arm xpos (3D): fingertip world position
    object xpos (3D): object world position
    goal xpos (3D): goal world position

Action space (7D):
    Continuous torques applied to 7 arm joints, clipped to [-2, 2].

Reward: -||obj - goal|| - 0.1 * sum(action^2) - 0.5 * ||fingertip - obj||
Termination: Never (truncated after 100 steps).

Example usage:
    from mojo_rl.envs.pusher import Pusher
    from mojo_rl.core import ContAction

    var env = Pusher()
    var state = env.reset()

    var action = ContAction[7]()
    var result = env.step(action)
"""

from .pusher import Pusher
from .pusher_xml import PusherModel
from .pusher_config import PusherConfig
