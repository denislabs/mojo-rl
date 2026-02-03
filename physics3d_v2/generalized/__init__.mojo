"""MuJoCo-style Generalized Coordinates Physics Engine.

This module provides a parallel physics engine based on generalized coordinates,
following MuJoCo's approach:
- State: Joint angles/positions (qpos) and velocities (qvel)
- Joints ADD DOFs (transformation chain) instead of constraining them
- Body positions computed via forward kinematics from qpos
- Dynamics computed in joint space (mass matrix, Coriolis, gravity forces)

Example usage:
    from physics3d_v2.generalized import ModelGC, DataGC
    from physics3d_v2.generalized.integrator import step_gc

    # Create a single pendulum
    var model = ModelGC[DType.float64, 1, 1, 1, 1, 5]()
    model.set_body(0, mass=1.0, inertia=(0.1, 0.1, 0.1))
    model.set_body_parent(0, -1)
    model.add_hinge_joint(body_id=0, pos=(0, 0, 1), axis=(0, 1, 0))

    var data = DataGC[DType.float64, 1, 1, 1, 1, 5]()
    data.qpos[0] = 0.5  # Initial angle

    for i in range(1000):
        step_gc(model, data)
        print(data.get_body_z(0))
"""

# Core types
from .types import ModelGC, DataGC, ContactInfoGC

# Joint definitions
from .joint_types import (
    JointDef,
    JNT_FREE,
    JNT_BALL,
    JNT_SLIDE,
    JNT_HINGE,
    FREE_QPOS_SIZE,
    FREE_QVEL_SIZE,
    BALL_QPOS_SIZE,
    BALL_QVEL_SIZE,
    SLIDE_QPOS_SIZE,
    SLIDE_QVEL_SIZE,
    HINGE_QPOS_SIZE,
    HINGE_QVEL_SIZE,
    get_joint_qpos_size,
    get_joint_qvel_size,
)
