"""Walker2d Environment Package.

MuJoCo-style Walker2d using the physics3d engine.

A bipedal walker with two legs (right and left), each with thigh, leg, and foot.
The goal is to walk forward as fast as possible without falling.

Observation space (17D):
    qpos[1:9]: rootz, rooty, thigh, leg, foot, thigh_left, leg_left, foot_left
    qvel[0:9]: all 9 generalized velocities
    (qpos[0]=rootx excluded; obs_qpos_skip=1)

Action space (6D):
    Torques for: thigh, leg, foot, thigh_left, leg_left, foot_left; clipped to [-1, 1].

Reward: x_velocity + 1.0 (if healthy) - 0.001 * sum(action²).
Termination: rootz ∉ (-0.45, 0.75) or |rooty| >= 1.0.
    (Equivalent to Gymnasium's world_z ∈ [0.8, 2.0] with zero-init qpos.)

Example usage:
    from mojo_rl.envs.walker2d import Walker2d
    from mojo_rl.core import ContAction

    var env = Walker2d()
    var state = env.reset()

    var action = ContAction[6]()
    var result = env.step(action)
"""

from .walker2d import Walker2d
