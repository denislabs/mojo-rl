"""MuJoCo's own name tables — the golden for `FlatModelDef`'s.

Prints one line per element so a Mojo gate can diff it verbatim.
"""
import sys
import mujoco

MODELS = [
    ("walker2d", "mojo_rl/envs/walker2d/assets/walker2d.xml"),
    ("humanoid", "mojo_rl/envs/humanoid/assets/humanoid.xml"),
    ("ant", "mojo_rl/envs/ant/assets/ant.xml"),
    ("so_arm100", "mojo_rl/envs/robots/assets/so_arm100.xml"),
    # ⚠ THE SITE-BEARING MODELS ARE THE POINT. Without one, every `site` arm
    # compares 0 elements — and sites carry the ordering trap this gate exists
    # for: `<worldbody>`'s OWN sites belong to body 0 and come FIRST, ahead of
    # every site declared inside a body however early it appears in the text.
    # That is exactly the finger / manipulator / stacker divergence.
    ("quadruped_walk", "mojo_rl/envs/dm_control/assets/quadruped_walk.xml"),
]

OBJ = [
    ("body", mujoco.mjtObj.mjOBJ_BODY, "nbody"),
    ("joint", mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
    ("geom", mujoco.mjtObj.mjOBJ_GEOM, "ngeom"),
    ("site", mujoco.mjtObj.mjOBJ_SITE, "nsite"),
    ("actuator", mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"),
]

for name, path in MODELS:
    try:
        m = mujoco.MjModel.from_xml_path(path)
    except Exception as e:
        print(f"# {name}: LOAD FAILED {e}")
        continue
    for label, obj, count in OBJ:
        n = getattr(m, count)
        for i in range(n):
            nm = mujoco.mj_id2name(m, obj, i)
            print(f"{name}\t{label}\t{i}\t{nm if nm else ''}")
