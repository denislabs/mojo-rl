"""Humanoid Environment Package.

MuJoCo-style Humanoid using the physics3d engine.

A 3D bipedal humanoid robot with 17 joints (excluding the root free joint).
The goal is to walk forward as fast as possible while staying upright.

Observation space (45D, simplified):
    qpos[2:24]: Joint positions (excluding free joint x/y translation)
    qvel[0:23]: Joint velocities

    Note: Gymnasium Humanoid-v4 uses 376D obs including cinert, cvel,
    qfrc_actuator, and cfrc_ext. This implementation uses simplified 45D obs
    for GPU training efficiency.

Action space (17D):
    Continuous joint torques.

Reward: 1.25 * x_velocity + 5.0 * is_healthy - 0.1 * ctrl_cost
Termination: torso z < 1.0 or torso z > 2.0.

Init: torso starts at z=1.4 (standing), init_qpos_gpu adds z=1.4 and quat_w=1.0
on top of reset noise.

Example usage:
    from envs.humanoid import Humanoid
    from core import ContAction

    var env = Humanoid()
    var state = env.reset()

    var action = ContAction[17]()
    var result = env.step(action)
"""

from .humanoid import Humanoid
