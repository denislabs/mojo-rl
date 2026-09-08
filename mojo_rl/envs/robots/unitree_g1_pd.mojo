"""Unitree G1 PD controller tables — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tests/robots/g1_bake.py
CI checks it with: pixi run python tests/robots/g1_bake.py --check

Source of truth is BFM-Zero's `config/robot/g1/g1_29dof_hard_waist.yaml`,
read by the reference's own substring rule (a gain key such as
`hip_yaw` applies to every dof whose name contains it) and emitted per
dof in the model's joint order, which `test_unitree_g1_vs_mujoco`
pins against `mjModel`'s joint names.

The torque law these feed (`legged_robot_base._compute_torques`):

    a       in [-1, 1]                      (the policy's tanh output)
    a       *= NORMALIZE_TO / NORMALIZE_FROM   then clipped to +-ACTION_CLIP
    target  = a * ACTION_SCALE * effort / kp + default_pos
    tau     = kp * (target - q) - kd * qdot,  clipped to +-effort

evaluated at EVERY physics substep, and applied through the model's
`<motor>` actuators (gear 1), whose `ctrlrange` clamps it AGAIN.

!! `ctrlrange` IS NOT `effort` ON FOUR JOINTS: the sim-to-sim XML says
+-88 N m on hip pitch/roll where this table (the yaml) says 139. The
env applies both, in the reference's order; see `unitree_g1_config`.
"""

comptime G1_N_DOF: Int = 29
comptime G1_ACTION_SCALE: Float64 = 0.25
comptime G1_ACTION_CLIP: Float64 = 5.0
comptime G1_NORMALIZE_FROM: Float64 = 1.0
comptime G1_NORMALIZE_TO: Float64 = 5.0
# `init_state.pos[2]` — the pelvis height the reference resets to.
comptime G1_INIT_ROOT_Z: Float64 = 0.8
# `1 / sim.fps` and `control_decimation`, from the simulator config.
comptime G1_SIM_TIMESTEP: Float64 = 0.005
comptime G1_CONTROL_DECIMATION: Int = 4


@always_inline
def g1_dof_name(i: Int) -> StaticString:
    """`dof_names`, the reference's dof order — pinned against `mjModel` joint names."""
    if i == 0:
        return "left_hip_pitch_joint"
    elif i == 1:
        return "left_hip_roll_joint"
    elif i == 2:
        return "left_hip_yaw_joint"
    elif i == 3:
        return "left_knee_joint"
    elif i == 4:
        return "left_ankle_pitch_joint"
    elif i == 5:
        return "left_ankle_roll_joint"
    elif i == 6:
        return "right_hip_pitch_joint"
    elif i == 7:
        return "right_hip_roll_joint"
    elif i == 8:
        return "right_hip_yaw_joint"
    elif i == 9:
        return "right_knee_joint"
    elif i == 10:
        return "right_ankle_pitch_joint"
    elif i == 11:
        return "right_ankle_roll_joint"
    elif i == 12:
        return "waist_yaw_joint"
    elif i == 13:
        return "waist_roll_joint"
    elif i == 14:
        return "waist_pitch_joint"
    elif i == 15:
        return "left_shoulder_pitch_joint"
    elif i == 16:
        return "left_shoulder_roll_joint"
    elif i == 17:
        return "left_shoulder_yaw_joint"
    elif i == 18:
        return "left_elbow_joint"
    elif i == 19:
        return "left_wrist_roll_joint"
    elif i == 20:
        return "left_wrist_pitch_joint"
    elif i == 21:
        return "left_wrist_yaw_joint"
    elif i == 22:
        return "right_shoulder_pitch_joint"
    elif i == 23:
        return "right_shoulder_roll_joint"
    elif i == 24:
        return "right_shoulder_yaw_joint"
    elif i == 25:
        return "right_elbow_joint"
    elif i == 26:
        return "right_wrist_roll_joint"
    elif i == 27:
        return "right_wrist_pitch_joint"
    elif i == 28:
        return "right_wrist_yaw_joint"
    return ""


@always_inline
def g1_kp(i: Int) -> Float64:
    """`control.stiffness`, N*m/rad."""
    if i == 0:  # left_hip_pitch_joint
        return 99.09843
    elif i == 1:  # left_hip_roll_joint
        return 99.09843
    elif i == 2:  # left_hip_yaw_joint
        return 40.17924
    elif i == 3:  # left_knee_joint
        return 99.09843
    elif i == 4:  # left_ankle_pitch_joint
        return 28.50125
    elif i == 5:  # left_ankle_roll_joint
        return 28.50125
    elif i == 6:  # right_hip_pitch_joint
        return 99.09843
    elif i == 7:  # right_hip_roll_joint
        return 99.09843
    elif i == 8:  # right_hip_yaw_joint
        return 40.17924
    elif i == 9:  # right_knee_joint
        return 99.09843
    elif i == 10:  # right_ankle_pitch_joint
        return 28.50125
    elif i == 11:  # right_ankle_roll_joint
        return 28.50125
    elif i == 12:  # waist_yaw_joint
        return 300.0
    elif i == 13:  # waist_roll_joint
        return 300.0
    elif i == 14:  # waist_pitch_joint
        return 300.0
    elif i == 15:  # left_shoulder_pitch_joint
        return 14.25062
    elif i == 16:  # left_shoulder_roll_joint
        return 14.25062
    elif i == 17:  # left_shoulder_yaw_joint
        return 14.25062
    elif i == 18:  # left_elbow_joint
        return 14.25062
    elif i == 19:  # left_wrist_roll_joint
        return 14.25062
    elif i == 20:  # left_wrist_pitch_joint
        return 16.77833
    elif i == 21:  # left_wrist_yaw_joint
        return 16.77833
    elif i == 22:  # right_shoulder_pitch_joint
        return 14.25062
    elif i == 23:  # right_shoulder_roll_joint
        return 14.25062
    elif i == 24:  # right_shoulder_yaw_joint
        return 14.25062
    elif i == 25:  # right_elbow_joint
        return 14.25062
    elif i == 26:  # right_wrist_roll_joint
        return 14.25062
    elif i == 27:  # right_wrist_pitch_joint
        return 16.77833
    elif i == 28:  # right_wrist_yaw_joint
        return 16.77833
    return 0.0


@always_inline
def g1_kd(i: Int) -> Float64:
    """`control.damping`, N*m*s/rad."""
    if i == 0:  # left_hip_pitch_joint
        return 6.3088
    elif i == 1:  # left_hip_roll_joint
        return 6.3088
    elif i == 2:  # left_hip_yaw_joint
        return 2.55789
    elif i == 3:  # left_knee_joint
        return 6.3088
    elif i == 4:  # left_ankle_pitch_joint
        return 1.81445
    elif i == 5:  # left_ankle_roll_joint
        return 1.81445
    elif i == 6:  # right_hip_pitch_joint
        return 6.3088
    elif i == 7:  # right_hip_roll_joint
        return 6.3088
    elif i == 8:  # right_hip_yaw_joint
        return 2.55789
    elif i == 9:  # right_knee_joint
        return 6.3088
    elif i == 10:  # right_ankle_pitch_joint
        return 1.81445
    elif i == 11:  # right_ankle_roll_joint
        return 1.81445
    elif i == 12:  # waist_yaw_joint
        return 5.0
    elif i == 13:  # waist_roll_joint
        return 5.0
    elif i == 14:  # waist_pitch_joint
        return 5.0
    elif i == 15:  # left_shoulder_pitch_joint
        return 0.90722
    elif i == 16:  # left_shoulder_roll_joint
        return 0.90722
    elif i == 17:  # left_shoulder_yaw_joint
        return 0.90722
    elif i == 18:  # left_elbow_joint
        return 0.90722
    elif i == 19:  # left_wrist_roll_joint
        return 0.90722
    elif i == 20:  # left_wrist_pitch_joint
        return 1.06814
    elif i == 21:  # left_wrist_yaw_joint
        return 1.06814
    elif i == 22:  # right_shoulder_pitch_joint
        return 0.90722
    elif i == 23:  # right_shoulder_roll_joint
        return 0.90722
    elif i == 24:  # right_shoulder_yaw_joint
        return 0.90722
    elif i == 25:  # right_elbow_joint
        return 0.90722
    elif i == 26:  # right_wrist_roll_joint
        return 0.90722
    elif i == 27:  # right_wrist_pitch_joint
        return 1.06814
    elif i == 28:  # right_wrist_yaw_joint
        return 1.06814
    return 0.0


@always_inline
def g1_effort(i: Int) -> Float64:
    """`dof_effort_limit_list`, N*m — the torque clip AND the `<motor ctrlrange>`."""
    if i == 0:  # left_hip_pitch_joint
        return 139.0
    elif i == 1:  # left_hip_roll_joint
        return 139.0
    elif i == 2:  # left_hip_yaw_joint
        return 88.0
    elif i == 3:  # left_knee_joint
        return 139.0
    elif i == 4:  # left_ankle_pitch_joint
        return 50.0
    elif i == 5:  # left_ankle_roll_joint
        return 50.0
    elif i == 6:  # right_hip_pitch_joint
        return 139.0
    elif i == 7:  # right_hip_roll_joint
        return 139.0
    elif i == 8:  # right_hip_yaw_joint
        return 88.0
    elif i == 9:  # right_knee_joint
        return 139.0
    elif i == 10:  # right_ankle_pitch_joint
        return 50.0
    elif i == 11:  # right_ankle_roll_joint
        return 50.0
    elif i == 12:  # waist_yaw_joint
        return 88.0
    elif i == 13:  # waist_roll_joint
        return 50.0
    elif i == 14:  # waist_pitch_joint
        return 50.0
    elif i == 15:  # left_shoulder_pitch_joint
        return 25.0
    elif i == 16:  # left_shoulder_roll_joint
        return 25.0
    elif i == 17:  # left_shoulder_yaw_joint
        return 25.0
    elif i == 18:  # left_elbow_joint
        return 25.0
    elif i == 19:  # left_wrist_roll_joint
        return 25.0
    elif i == 20:  # left_wrist_pitch_joint
        return 5.0
    elif i == 21:  # left_wrist_yaw_joint
        return 5.0
    elif i == 22:  # right_shoulder_pitch_joint
        return 25.0
    elif i == 23:  # right_shoulder_roll_joint
        return 25.0
    elif i == 24:  # right_shoulder_yaw_joint
        return 25.0
    elif i == 25:  # right_elbow_joint
        return 25.0
    elif i == 26:  # right_wrist_roll_joint
        return 25.0
    elif i == 27:  # right_wrist_pitch_joint
        return 5.0
    elif i == 28:  # right_wrist_yaw_joint
        return 5.0
    return 0.0


@always_inline
def g1_default_pos(i: Int) -> Float64:
    """`init_state.default_joint_angles`, rad — the PD target at action 0."""
    if i == 0:  # left_hip_pitch_joint
        return -0.1
    elif i == 1:  # left_hip_roll_joint
        return 0.0
    elif i == 2:  # left_hip_yaw_joint
        return 0.0
    elif i == 3:  # left_knee_joint
        return 0.3
    elif i == 4:  # left_ankle_pitch_joint
        return -0.2
    elif i == 5:  # left_ankle_roll_joint
        return 0.0
    elif i == 6:  # right_hip_pitch_joint
        return -0.1
    elif i == 7:  # right_hip_roll_joint
        return 0.0
    elif i == 8:  # right_hip_yaw_joint
        return 0.0
    elif i == 9:  # right_knee_joint
        return 0.3
    elif i == 10:  # right_ankle_pitch_joint
        return -0.2
    elif i == 11:  # right_ankle_roll_joint
        return 0.0
    elif i == 12:  # waist_yaw_joint
        return 0.0
    elif i == 13:  # waist_roll_joint
        return 0.0
    elif i == 14:  # waist_pitch_joint
        return 0.0
    elif i == 15:  # left_shoulder_pitch_joint
        return 0.0
    elif i == 16:  # left_shoulder_roll_joint
        return 0.0
    elif i == 17:  # left_shoulder_yaw_joint
        return 0.0
    elif i == 18:  # left_elbow_joint
        return 0.0
    elif i == 19:  # left_wrist_roll_joint
        return 0.0
    elif i == 20:  # left_wrist_pitch_joint
        return 0.0
    elif i == 21:  # left_wrist_yaw_joint
        return 0.0
    elif i == 22:  # right_shoulder_pitch_joint
        return 0.0
    elif i == 23:  # right_shoulder_roll_joint
        return 0.0
    elif i == 24:  # right_shoulder_yaw_joint
        return 0.0
    elif i == 25:  # right_elbow_joint
        return 0.0
    elif i == 26:  # right_wrist_roll_joint
        return 0.0
    elif i == 27:  # right_wrist_pitch_joint
        return 0.0
    elif i == 28:  # right_wrist_yaw_joint
        return 0.0
    return 0.0


@always_inline
def g1_pos_lower(i: Int) -> Float64:
    """`dof_pos_lower_limit_list`, rad."""
    if i == 0:  # left_hip_pitch_joint
        return -2.5307
    elif i == 1:  # left_hip_roll_joint
        return -0.5236
    elif i == 2:  # left_hip_yaw_joint
        return -2.7576
    elif i == 3:  # left_knee_joint
        return -0.087267
    elif i == 4:  # left_ankle_pitch_joint
        return -0.87267
    elif i == 5:  # left_ankle_roll_joint
        return -0.2618
    elif i == 6:  # right_hip_pitch_joint
        return -2.5307
    elif i == 7:  # right_hip_roll_joint
        return -2.9671
    elif i == 8:  # right_hip_yaw_joint
        return -2.7576
    elif i == 9:  # right_knee_joint
        return -0.087267
    elif i == 10:  # right_ankle_pitch_joint
        return -0.87267
    elif i == 11:  # right_ankle_roll_joint
        return -0.2618
    elif i == 12:  # waist_yaw_joint
        return -2.618
    elif i == 13:  # waist_roll_joint
        return -0.52
    elif i == 14:  # waist_pitch_joint
        return -0.52
    elif i == 15:  # left_shoulder_pitch_joint
        return -3.0892
    elif i == 16:  # left_shoulder_roll_joint
        return -1.5882
    elif i == 17:  # left_shoulder_yaw_joint
        return -2.618
    elif i == 18:  # left_elbow_joint
        return -1.0472
    elif i == 19:  # left_wrist_roll_joint
        return -1.972222054
    elif i == 20:  # left_wrist_pitch_joint
        return -1.61443
    elif i == 21:  # left_wrist_yaw_joint
        return -1.61443
    elif i == 22:  # right_shoulder_pitch_joint
        return -3.0892
    elif i == 23:  # right_shoulder_roll_joint
        return -2.2515
    elif i == 24:  # right_shoulder_yaw_joint
        return -2.618
    elif i == 25:  # right_elbow_joint
        return -1.0472
    elif i == 26:  # right_wrist_roll_joint
        return -1.972222054
    elif i == 27:  # right_wrist_pitch_joint
        return -1.61443
    elif i == 28:  # right_wrist_yaw_joint
        return -1.61443
    return 0.0


@always_inline
def g1_pos_upper(i: Int) -> Float64:
    """`dof_pos_upper_limit_list`, rad."""
    if i == 0:  # left_hip_pitch_joint
        return 2.8798
    elif i == 1:  # left_hip_roll_joint
        return 2.9671
    elif i == 2:  # left_hip_yaw_joint
        return 2.7576
    elif i == 3:  # left_knee_joint
        return 2.8798
    elif i == 4:  # left_ankle_pitch_joint
        return 0.5236
    elif i == 5:  # left_ankle_roll_joint
        return 0.2618
    elif i == 6:  # right_hip_pitch_joint
        return 2.8798
    elif i == 7:  # right_hip_roll_joint
        return 0.5236
    elif i == 8:  # right_hip_yaw_joint
        return 2.7576
    elif i == 9:  # right_knee_joint
        return 2.8798
    elif i == 10:  # right_ankle_pitch_joint
        return 0.5236
    elif i == 11:  # right_ankle_roll_joint
        return 0.2618
    elif i == 12:  # waist_yaw_joint
        return 2.618
    elif i == 13:  # waist_roll_joint
        return 0.52
    elif i == 14:  # waist_pitch_joint
        return 0.52
    elif i == 15:  # left_shoulder_pitch_joint
        return 2.6704
    elif i == 16:  # left_shoulder_roll_joint
        return 2.2515
    elif i == 17:  # left_shoulder_yaw_joint
        return 2.618
    elif i == 18:  # left_elbow_joint
        return 2.0944
    elif i == 19:  # left_wrist_roll_joint
        return 1.972222054
    elif i == 20:  # left_wrist_pitch_joint
        return 1.61443
    elif i == 21:  # left_wrist_yaw_joint
        return 1.61443
    elif i == 22:  # right_shoulder_pitch_joint
        return 2.6704
    elif i == 23:  # right_shoulder_roll_joint
        return 1.5882
    elif i == 24:  # right_shoulder_yaw_joint
        return 2.618
    elif i == 25:  # right_elbow_joint
        return 2.0944
    elif i == 26:  # right_wrist_roll_joint
        return 1.972222054
    elif i == 27:  # right_wrist_pitch_joint
        return 1.61443
    elif i == 28:  # right_wrist_yaw_joint
        return 1.61443
    return 0.0


@always_inline
def g1_vel_limit(i: Int) -> Float64:
    """`dof_vel_limit_list`, rad/s."""
    if i == 0:  # left_hip_pitch_joint
        return 32.0
    elif i == 1:  # left_hip_roll_joint
        return 32.0
    elif i == 2:  # left_hip_yaw_joint
        return 32.0
    elif i == 3:  # left_knee_joint
        return 20.0
    elif i == 4:  # left_ankle_pitch_joint
        return 37.0
    elif i == 5:  # left_ankle_roll_joint
        return 37.0
    elif i == 6:  # right_hip_pitch_joint
        return 32.0
    elif i == 7:  # right_hip_roll_joint
        return 32.0
    elif i == 8:  # right_hip_yaw_joint
        return 32.0
    elif i == 9:  # right_knee_joint
        return 20.0
    elif i == 10:  # right_ankle_pitch_joint
        return 37.0
    elif i == 11:  # right_ankle_roll_joint
        return 37.0
    elif i == 12:  # waist_yaw_joint
        return 32.0
    elif i == 13:  # waist_roll_joint
        return 37.0
    elif i == 14:  # waist_pitch_joint
        return 37.0
    elif i == 15:  # left_shoulder_pitch_joint
        return 37.0
    elif i == 16:  # left_shoulder_roll_joint
        return 37.0
    elif i == 17:  # left_shoulder_yaw_joint
        return 37.0
    elif i == 18:  # left_elbow_joint
        return 37.0
    elif i == 19:  # left_wrist_roll_joint
        return 37.0
    elif i == 20:  # left_wrist_pitch_joint
        return 22.0
    elif i == 21:  # left_wrist_yaw_joint
        return 22.0
    elif i == 22:  # right_shoulder_pitch_joint
        return 37.0
    elif i == 23:  # right_shoulder_roll_joint
        return 37.0
    elif i == 24:  # right_shoulder_yaw_joint
        return 37.0
    elif i == 25:  # right_elbow_joint
        return 37.0
    elif i == 26:  # right_wrist_roll_joint
        return 37.0
    elif i == 27:  # right_wrist_pitch_joint
        return 22.0
    elif i == 28:  # right_wrist_yaw_joint
        return 22.0
    return 0.0
