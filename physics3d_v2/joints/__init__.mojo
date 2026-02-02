"""Physics3D v2 Joint System.

Joint constraints for articulated bodies (pendulums, chains, robots).

Currently supports:
- HingeJoint: Single-axis rotation (5 DOF constraint)

Usage:
    from physics3d_v2.joints import HingeJoint

    # Create model with joints
    var model = Model[DType.float64, 1, 5, 1](gravity_z=-9.81)  # MAX_JOINTS=1
    model.set_body(0, mass=1.0, radius=0.1)

    # Add hinge joint anchored to world
    model.add_hinge_joint(
        parent=-1,  # -1 = world anchor
        child=0,
        anchor_parent=(0.0, 0.0, 1.0),  # World anchor point
        anchor_child=(0.0, 0.0, 0.1),   # Body local anchor
        axis=(0.0, 1.0, 0.0),           # Y-axis rotation
    )
"""

from .hinge_joint import HingeJoint
from .joint_solver import (
    # Joint state sensing (Phase 7)
    get_joint_angle,
    get_joint_angular_velocity,
    # Torque actuation (Phase 7)
    apply_joint_torques,
    apply_joint_torques_gpu,
    # Constraint solving
    solve_joint_velocity_constraints,
    solve_joint_position_constraints,
    # GPU versions
    solve_joint_velocity_constraints_gpu,
    solve_joint_position_constraints_gpu,
)
