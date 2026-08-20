"""MuJoCo 3.10.0's kinematic TREE, by name — GENERATED.

⚠ DO NOT EDIT. Regenerate with:
    pixi run python scripts/dump_mujoco_body_tree.py

`parents(i)` is a space-separated list of each body's PARENT name
("world" for the worldbody and for its direct children);
`geom_bodies(i)` names the body each geom belongs to.
"""


def tree_case_count() -> Int:
    return 7


def tree_path(i: Int) -> String:
    if i == 0:
        return String("references/mujoco_menagerie-main/hello_robot_stretch_3/scene.xml")
    if i == 1:
        return String("references/mujoco_menagerie-main/apptronik_apollo/scene.xml")
    if i == 2:
        return String("references/mujoco_menagerie-main/franka_fr3_v2/scene.xml")
    if i == 3:
        return String("references/mujoco_menagerie-main/aloha/scene.xml")
    if i == 4:
        return String("references/mujoco_menagerie-main/unitree_go2/scene.xml")
    if i == 5:
        return String("mojo_rl/envs/ant/assets/ant.xml")
    if i == 6:
        return String("mojo_rl/envs/humanoid/assets/humanoid.xml")
    return String("")


def tree_nbody(i: Int) -> Int:
    if i == 0:
        return 41
    if i == 1:
        return 37
    if i == 2:
        return 11
    if i == 3:
        return 21
    if i == 4:
        return 14
    if i == 5:
        return 14
    if i == 6:
        return 14
    return 0


def tree_ngeom(i: Int) -> Int:
    if i == 0:
        return 131
    if i == 1:
        return 79
    if i == 2:
        return 37
    if i == 3:
        return 95
    if i == 4:
        return 57
    if i == 5:
        return 14
    if i == 6:
        return 18
    return 0


def tree_parents(i: Int) -> String:
    if i == 0:
        return String("world world base_link base_link base_link base_link base_link base_link base_link base_link base_link link_lift link_arm_l4 link_arm_l3 link_arm_l2 link_arm_l1 link_arm_l0 link_arm_l0 link_arm_l0 link_wrist_yaw link_DW3_wrist_pitch link_SG3_gripper_body link_SG3_aruco_d405 link_d405 link_SG3_gripper_body link_SG3_gripper_body link_gripper_slider link_gripper_finger_left link_gripper_finger_left link_gripper_slider link_gripper_finger_right link_gripper_finger_right link_lift base_link link_head_pan link_head_tilt link_head_tilt link_SE3_head_nav_cam world world world")
    if i == 1:
        return String("world world base_link base_link base_link torso_roll_link torso_pitch_link torso_link neck_yaw_link neck_roll_link torso_link l_shoulder_aa_link l_shoulder_ie_link l_shoulder_fe_link l_elbow_fe_link l_wrist_roll_link l_wrist_yaw_link torso_link r_shoulder_aa_link r_shoulder_ie_link r_shoulder_fe_link r_elbow_fe_link r_wrist_roll_link r_wrist_yaw_link base_link l_hip_ie_link l_hip_aa_link l_hip_fe_link l_knee_fe_link l_ankle_ie_link base_link r_hip_ie_link r_hip_aa_link r_hip_fe_link r_knee_fe_link r_ankle_ie_link world")
    if i == 2:
        return String("world world base fr3v2_link0 fr3v2_link1 fr3v2_link2 fr3v2_link3 fr3v2_link4 fr3v2_link5 fr3v2_link6 fr3v2_link7")
    if i == 3:
        return String("world world left/base_link left/shoulder_link left/upper_arm_link left/upper_forearm_link left/lower_forearm_link left/wrist_link left/gripper_link left/gripper_base left/gripper_base world right/base_link right/shoulder_link right/upper_arm_link right/upper_forearm_link right/lower_forearm_link right/wrist_link right/gripper_link right/gripper_base right/gripper_base")
    if i == 4:
        return String("world world base FL_hip FL_thigh base FR_hip FR_thigh base RL_hip RL_thigh base RR_hip RR_thigh")
    if i == 5:
        return String("world world torso front_left_leg aux_1 torso front_right_leg aux_2 torso back_leg aux_3 torso right_back_leg aux_4")
    if i == 6:
        return String("world world torso lwaist pelvis right_thigh right_shin pelvis left_thigh left_shin torso right_upper_arm torso left_upper_arm")
    return String("")


def tree_geom_bodies(i: Int) -> String:
    if i == 0:
        return String("world base_link base_link base_link base_link base_link base_link base_link base_link base_link base_link link_mast link_mast link_head link_head link_head link_head link_head link_head link_head link_head link_head link_head link_head link_head link_head link_aruco_right_base link_aruco_right_base link_aruco_left_base link_aruco_left_base laser laser link_right_wheel link_right_wheel link_right_wheel link_left_wheel link_left_wheel link_left_wheel link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_lift link_arm_l4 link_arm_l4 link_arm_l4 link_arm_l4 link_arm_l3 link_arm_l3 link_arm_l3 link_arm_l3 link_arm_l2 link_arm_l2 link_arm_l2 link_arm_l2 link_arm_l1 link_arm_l1 link_arm_l1 link_arm_l1 link_arm_l0 link_arm_l0 link_arm_l0 link_arm_l0 link_arm_l0 link_arm_l0 link_aruco_top_wrist link_aruco_top_wrist link_aruco_inner_wrist link_aruco_inner_wrist link_wrist_yaw link_wrist_yaw link_wrist_yaw link_wrist_yaw link_DW3_wrist_pitch link_DW3_wrist_pitch link_DW3_wrist_pitch link_DW3_wrist_pitch link_SG3_gripper_body link_SG3_gripper_body link_SG3_aruco_d405 link_SG3_aruco_d405 link_d405 link_d405 link_d405 link_d405 link_gripper_slider link_gripper_finger_left link_gripper_finger_left link_gripper_finger_left link_gripper_finger_left link_gripper_finger_left rubber_tip_left rubber_tip_left link_SG3_gripper_left_finger_aruco link_SG3_gripper_left_finger_aruco link_gripper_finger_right link_gripper_finger_right link_gripper_finger_right link_gripper_finger_right link_gripper_finger_right rubber_tip_right rubber_tip_right link_SG3_gripper_right_finger_aruco link_SG3_gripper_right_finger_aruco link_aruco_shoulder link_aruco_shoulder link_head_pan link_head_pan link_head_pan link_head_pan link_head_tilt link_head_tilt link_head_tilt link_head_tilt link_SE3_head_nav_cam table object1 object2")
    if i == 1:
        return String("world base_link base_link torso_roll_link torso_pitch_link torso_link torso_link torso_link torso_link neck_yaw_link neck_pitch_link l_shoulder_aa_link l_shoulder_ie_link l_shoulder_fe_link l_shoulder_fe_link l_shoulder_fe_link l_shoulder_fe_link l_elbow_fe_link l_wrist_roll_link l_wrist_roll_link l_wrist_yaw_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link l_wrist_pitch_link r_shoulder_aa_link r_shoulder_ie_link r_shoulder_fe_link r_shoulder_fe_link r_shoulder_fe_link r_shoulder_fe_link r_elbow_fe_link r_wrist_roll_link r_wrist_roll_link r_wrist_yaw_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link r_wrist_pitch_link l_hip_ie_link l_hip_aa_link l_hip_fe_link l_hip_fe_link l_knee_fe_link l_knee_fe_link l_ankle_ie_link l_foot_link l_foot_link r_hip_ie_link r_hip_aa_link r_hip_fe_link r_hip_fe_link r_knee_fe_link r_knee_fe_link r_ankle_ie_link r_foot_link r_foot_link")
    if i == 2:
        return String("world fr3v2_link0 fr3v2_link0 fr3v2_link0 fr3v2_link0 fr3v2_link0 fr3v2_link0 fr3v2_link0 fr3v2_link1 fr3v2_link1 fr3v2_link2 fr3v2_link2 fr3v2_link3 fr3v2_link3 fr3v2_link4 fr3v2_link4 fr3v2_link5 fr3v2_link5 fr3v2_link6 fr3v2_link6 fr3v2_link6 fr3v2_link6 fr3v2_link6 fr3v2_link6 fr3v2_link6 fr3v2_link6 fr3v2_link6 fr3v2_link6 fr3v2_link6 fr3v2_link7 fr3v2_link7 fr3v2_link7 fr3v2_link7 fr3v2_link7 fr3v2_link7 fr3v2_link7 fr3v2_link7")
    if i == 3:
        return String("world world world world world world world world world world world world world world world world world world world world world world world world world world world world world world world world world world world left/base_link left/base_link left/shoulder_link left/shoulder_link left/upper_arm_link left/upper_arm_link left/upper_forearm_link left/upper_forearm_link left/lower_forearm_link left/lower_forearm_link left/wrist_link left/wrist_link left/gripper_base left/gripper_base left/gripper_base left/gripper_base left/gripper_base left/gripper_base left/gripper_base left/gripper_base left/left_finger_link left/left_finger_link left/left_finger_link left/left_finger_link left/left_finger_link left/right_finger_link left/right_finger_link left/right_finger_link left/right_finger_link left/right_finger_link right/base_link right/base_link right/shoulder_link right/shoulder_link right/upper_arm_link right/upper_arm_link right/upper_forearm_link right/upper_forearm_link right/lower_forearm_link right/lower_forearm_link right/wrist_link right/wrist_link right/gripper_base right/gripper_base right/gripper_base right/gripper_base right/gripper_base right/gripper_base right/gripper_base right/gripper_base right/left_finger_link right/left_finger_link right/left_finger_link right/left_finger_link right/left_finger_link right/right_finger_link right/right_finger_link right/right_finger_link right/right_finger_link right/right_finger_link")
    if i == 4:
        return String("world base base base base base base base base FL_hip FL_hip FL_hip FL_thigh FL_thigh FL_thigh FL_calf FL_calf FL_calf FL_calf FL_calf FL_calf FR_hip FR_hip FR_hip FR_thigh FR_thigh FR_thigh FR_calf FR_calf FR_calf FR_calf FR_calf FR_calf RL_hip RL_hip RL_hip RL_thigh RL_thigh RL_thigh RL_calf RL_calf RL_calf RL_calf RL_calf RL_calf RR_hip RR_hip RR_hip RR_thigh RR_thigh RR_thigh RR_calf RR_calf RR_calf RR_calf RR_calf RR_calf")
    if i == 5:
        return String("world torso front_left_leg aux_1 #4 front_right_leg aux_2 #7 back_leg aux_3 #10 right_back_leg aux_4 #13")
    if i == 6:
        return String("world torso torso torso lwaist pelvis right_thigh right_shin right_foot left_thigh left_shin left_foot right_upper_arm right_lower_arm right_lower_arm left_upper_arm left_lower_arm left_lower_arm")
    return String("")
