"""InvertedPendulum Environment Package.

MuJoCo-style Inverted Pendulum using the physics3d engine.

A cart-pole system where the goal is to balance a pole upright on a sliding cart.

Observation space (4D):
    qpos[0]: cart position (slider x)
    qpos[1]: pole angle (hinge)
    qvel[0]: cart velocity
    qvel[1]: pole angular velocity

Action space (1D):
    Continuous force applied to the cart, clipped to [-3, 3].

Reward: +1 for every step the pole remains balanced.
Termination: |cart_pos| >= 1.0 or |pole_angle| >= 0.2 radians.

Example usage:
    from envs.inverted_pendulum import InvertedPendulum
    from core import ContAction

    var env = InvertedPendulum()
    var state = env.reset()

    var action = ContAction[1]()
    var result = env.step(action)
"""

from .inverted_pendulum import InvertedPendulum
