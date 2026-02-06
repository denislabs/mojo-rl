"""Physics3D v2 Joint System.

Joint constraints for articulated bodies (pendulums, chains, robots).

Currently supports:
- HingeJoint: Single-axis rotation (5 DOF constraint)
- SlideJoint: Single-axis translation (5 DOF constraint)

Usage:
    from physics3d.joints import HingeJoint, SlideJoint

    # Create model with hinge joints
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

    # Create model with slide joints
    var model2 = Model[DType.float64, 1, 5, 0, 1](gravity_z=-9.81)  # MAX_SLIDE_JOINTS=1
    model2.add_slide_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, 0.0),
        anchor_child=(0.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),  # X-axis translation
    )
"""

from .hinge_joint import HingeJoint
from .slide_joint import SlideJoint
from .joint_solver import (
    # Hinge joint state sensing (Phase 7)
    get_joint_angle,
    get_joint_angular_velocity,
    # Hinge joint torque actuation (Phase 7)
    apply_joint_torques,
    apply_joint_torques_gpu,
    # Hinge joint constraint solving
    solve_joint_velocity_constraints,
    solve_joint_position_constraints,
    # Hinge joint GPU versions
    solve_joint_velocity_constraints_gpu,
    solve_joint_position_constraints_gpu,
    # Slide joint state sensing
    get_slide_joint_position,
    get_slide_joint_velocity,
    # Slide joint force actuation
    apply_slide_joint_forces,
    # Slide joint constraint solving
    solve_slide_joint_velocity_constraints,
    solve_slide_joint_position_constraints,
    # Slide joint GPU versions
    solve_slide_joint_velocity_constraints_gpu,
    solve_slide_joint_position_constraints_gpu,
)
