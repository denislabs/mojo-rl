"""MuJoCo 3.10.0's mass-matrix DIAGONAL BLOCKS — GENERATED.

⚠ DO NOT EDIT. Regenerate with:
    pixi run python scripts/dump_mujoco_trees.py

`blk(i)` is a space-separated `dof_adr dof_num compact` triple per
block, in MuJoCo's tree order. `compact` is 1 when M restricted to the
block is DIAGONAL. See the script header for why `adr`/`num` are gated
exactly and `compact` is gated ONE-DIRECTIONALLY.

`nC` is MuJoCo's stored entry count — the SPARSE size our dense `nv*nv`
is measured against.
"""


def blk_case_count() -> Int:
    return 28


def blk_path(i: Int) -> String:
    if i == 0:
        return String("mojo_rl/envs/robots/assets/so101_park_k0.xml")
    if i == 1:
        return String("mojo_rl/envs/robots/assets/so101_park_k3.xml")
    if i == 2:
        return String("mojo_rl/envs/robots/assets/so101_park_k6.xml")
    if i == 3:
        return String("mojo_rl/envs/robots/assets/so101_park_k9.xml")
    if i == 4:
        return String("mojo_rl/envs/robots/assets/so_arm101.xml")
    if i == 5:
        return String("mojo_rl/envs/ant/assets/ant.xml")
    if i == 6:
        return String("mojo_rl/envs/humanoid/assets/humanoid.xml")
    if i == 7:
        return String("mojo_rl/envs/dm_control/assets/acrobot.xml")
    if i == 8:
        return String("mojo_rl/envs/dm_control/assets/ball_in_cup.xml")
    if i == 9:
        return String("mojo_rl/envs/dm_control/assets/cartpole3.xml")
    if i == 10:
        return String("mojo_rl/envs/dm_control/assets/cheetah.xml")
    if i == 11:
        return String("mojo_rl/envs/dm_control/assets/finger.xml")
    if i == 12:
        return String("mojo_rl/envs/dm_control/assets/fish.xml")
    if i == 13:
        return String("mojo_rl/envs/dm_control/assets/hopper.xml")
    if i == 14:
        return String("mojo_rl/envs/dm_control/assets/humanoid_cmu.xml")
    if i == 15:
        return String("mojo_rl/envs/dm_control/assets/manipulator_bring_ball.xml")
    if i == 16:
        return String("mojo_rl/envs/dm_control/assets/point_mass.xml")
    if i == 17:
        return String("mojo_rl/envs/dm_control/assets/quadruped_escape.xml")
    if i == 18:
        return String("mojo_rl/envs/dm_control/assets/quadruped_fetch.xml")
    if i == 19:
        return String("mojo_rl/envs/dm_control/assets/reacher.xml")
    if i == 20:
        return String("mojo_rl/envs/dm_control/assets/stacker_2.xml")
    if i == 21:
        return String("mojo_rl/envs/dm_control/assets/dog_fetch.xml")
    if i == 22:
        return String("references/mujoco_menagerie-main/hello_robot_stretch_3/scene.xml")
    if i == 23:
        return String("references/mujoco_menagerie-main/apptronik_apollo/scene.xml")
    if i == 24:
        return String("references/mujoco_menagerie-main/franka_fr3_v2/scene.xml")
    if i == 25:
        return String("references/mujoco_menagerie-main/aloha/scene.xml")
    if i == 26:
        return String("references/mujoco_menagerie-main/unitree_go2/scene.xml")
    if i == 27:
        return String("references/mujoco_menagerie-main/shadow_hand/scene_right.xml")
    return String("")


def blk_nv(i: Int) -> Int:
    if i == 0:
        return 6
    if i == 1:
        return 24
    if i == 2:
        return 42
    if i == 3:
        return 60
    if i == 4:
        return 6
    if i == 5:
        return 14
    if i == 6:
        return 23
    if i == 7:
        return 2
    if i == 8:
        return 4
    if i == 9:
        return 4
    if i == 10:
        return 9
    if i == 11:
        return 3
    if i == 12:
        return 13
    if i == 13:
        return 7
    if i == 14:
        return 62
    if i == 15:
        return 11
    if i == 16:
        return 2
    if i == 17:
        return 22
    if i == 18:
        return 28
    if i == 19:
        return 2
    if i == 20:
        return 14
    if i == 21:
        return 85
    if i == 22:
        return 38
    if i == 23:
        return 38
    if i == 24:
        return 7
    if i == 25:
        return 16
    if i == 26:
        return 18
    if i == 27:
        return 30
    return 0


def blk_nC(i: Int) -> Int:
    if i == 0:
        return 21
    if i == 1:
        return 39
    if i == 2:
        return 57
    if i == 3:
        return 75
    if i == 4:
        return 21
    if i == 5:
        return 81
    if i == 6:
        return 185
    if i == 7:
        return 3
    if i == 8:
        return 5
    if i == 9:
        return 10
    if i == 10:
        return 36
    if i == 11:
        return 4
    if i == 12:
        return 75
    if i == 13:
        return 28
    if i == 14:
        return 952
    if i == 15:
        return 35
    if i == 16:
        return 2
    if i == 17:
        return 157
    if i == 18:
        return 163
    if i == 19:
        return 3
    if i == 20:
        return 38
    if i == 21:
        return 1330
    if i == 22:
        return 272
    if i == 23:
        return 374
    if i == 24:
        return 28
    if i == 25:
        return 70
    if i == 26:
        return 117
    if i == 27:
        return 113
    return 0


def blk(i: Int) -> String:
    if i == 0:
        return String("0 6 0")
    if i == 1:
        return String("0 6 0 6 6 1 12 6 1 18 6 1")
    if i == 2:
        return String("0 6 0 6 6 1 12 6 1 18 6 1 24 6 1 30 6 1 36 6 1")
    if i == 3:
        return String("0 6 0 6 6 1 12 6 1 18 6 1 24 6 1 30 6 1 36 6 1 42 6 1 48 6 1 54 6 1")
    if i == 4:
        return String("0 6 0")
    if i == 5:
        return String("0 14 0")
    if i == 6:
        return String("0 23 0")
    if i == 7:
        return String("0 2 0")
    if i == 8:
        return String("0 2 0 2 2 1")
    if i == 9:
        return String("0 4 0")
    if i == 10:
        return String("0 9 0")
    if i == 11:
        return String("0 2 0 2 1 1")
    if i == 12:
        return String("0 13 0")
    if i == 13:
        return String("0 7 0")
    if i == 14:
        return String("0 62 0")
    if i == 15:
        return String("0 8 0 8 3 1")
    if i == 16:
        return String("0 2 1")
    if i == 17:
        return String("0 22 0")
    if i == 18:
        return String("0 22 0 22 6 1")
    if i == 19:
        return String("0 2 0")
    if i == 20:
        return String("0 8 0 8 3 1 11 3 1")
    if i == 21:
        return String("0 79 0 79 6 1")
    if i == 22:
        return String("0 26 0 26 6 1 32 6 1")
    if i == 23:
        return String("0 38 0")
    if i == 24:
        return String("0 7 0")
    if i == 25:
        return String("0 8 0 8 8 0")
    if i == 26:
        return String("0 18 0")
    if i == 27:
        return String("0 24 0 24 6 1")
    return String("")
