"""InvertedDoublePendulum Environment Package.

MuJoCo-style Inverted Double Pendulum using the physics3d engine.

A cart-pole system with two links (pole and pole2) where the goal is to balance
both poles upright on a sliding cart.

Observation space (9D):
    obs[0]: cart x position
    obs[1]: sin(pole1 angle)
    obs[2]: sin(pole2 angle)
    obs[3]: cos(pole1 angle)
    obs[4]: cos(pole2 angle)
    obs[5]: cart velocity (clipped to [-10, 10])
    obs[6]: pole1 angular velocity (clipped to [-10, 10])
    obs[7]: pole2 angular velocity (clipped to [-10, 10])
    obs[8]: 0.0 (constraint force placeholder)

Action space (1D):
    Continuous force applied to the cart, clipped to [-1, 1] * gear(500).

Reward: 10 - 0.01*x_tip^2 - (z_tip-2)^2 - 1e-3*v1^2 - 5e-3*v2^2
    where x_tip and z_tip are the horizontal and vertical position of the
    tip of pole2 computed analytically from joint angles.
Termination: z_tip (tip height) <= 1.0 m.

Example usage:
    from mojo_rl.envs.inverted_double_pendulum import InvertedDoublePendulum
    from mojo_rl.core import ContAction

    var env = InvertedDoublePendulum()
    var state = env.reset()

    var action = ContAction[1]()
    var result = env.step(action)
"""

from .inverted_double_pendulum import InvertedDoublePendulum
