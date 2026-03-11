"""HumanoidStandup Environment Package.

MuJoCo-style HumanoidStandup using the physics3d engine.

A 3D bipedal humanoid robot starting in a lying-down position (torso z=0.105).
The goal is to stand up by maximizing the upward velocity of the torso.

Observation space (45D, simplified):
    qpos[2:24]: Joint positions (excluding free joint x/y translation)
    qvel[0:23]: Joint velocities

    Note: Gymnasium HumanoidStandup-v4 uses 376D obs including cinert, cvel,
    qfrc_actuator, and cfrc_ext. This implementation uses simplified 45D obs
    for GPU training efficiency.

Action space (17D):
    Continuous joint torques.

Reward: qpos[2] / timestep - 0.1 * ctrl_cost + 1.0
    where qpos[2] is the torso world z (higher = better while standing up).
    No termination (episode runs until max_steps).

Init: torso starts at z=0.105 (lying on back), init_qpos_gpu adds z=0.105
and quat_w=1.0 on top of reset noise.

Example usage:
    from envs.humanoid_standup import HumanoidStandup
    from core import ContAction

    var env = HumanoidStandup()
    var state = env.reset()

    var action = ContAction[17]()
    var result = env.step(action)
"""

from .humanoid_standup import HumanoidStandup
