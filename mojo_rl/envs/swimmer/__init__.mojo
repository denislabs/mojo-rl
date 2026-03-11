"""Swimmer Environment Package.

MuJoCo-style Swimmer using the physics3d engine.

A 3-link planar swimmer that learns to move forward using joint torques.
Note: The original MuJoCo swimmer uses fluid drag (density=4000, viscosity=0.1),
which is not implemented in physics3d. The agent can still learn to swim using
joint torques; only the fluid resistance forces are absent.

Observation space (8D):
    qpos[2]: free_body_rot (hinge, z-axis)
    qpos[3]: motor1_rot
    qpos[4]: motor2_rot
    qvel[0:5]: all 5 generalized velocities
    (qpos[0]=slider1_x and qpos[1]=slider2_y are excluded; obs_qpos_skip=2)

Action space (2D):
    motor1_rot torque, motor2_rot torque; clipped to [-1, 1].

Reward: x_velocity - 0.0001 * sum(action²). Never terminates.

Example usage:
    from mojo_rl.envs.swimmer import Swimmer
    from mojo_rl.core import ContAction

    var env = Swimmer()
    var state = env.reset()

    var action = ContAction[2]()
    var result = env.step(action)
"""

from .swimmer import Swimmer
