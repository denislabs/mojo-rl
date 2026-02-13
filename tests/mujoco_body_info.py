"""Dump MuJoCo body info: positions, inertia, mass, parent.
Compare with our engine's forward kinematics.
"""
import numpy as np
import mujoco

np.set_printoptions(precision=8, linewidth=200, suppress=True)

model = mujoco.MjModel.from_xml_path("../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print("=== Body Info ===")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"\nBody {i}: {name}")
    print(f"  mass = {model.body_mass[i]:.6f}")
    print(f"  pos (in parent) = {model.body_pos[i]}")
    print(f"  ipos (inertia frame pos in body frame) = {model.body_ipos[i]}")
    print(f"  iquat (inertia frame quat in body frame) = {model.body_iquat[i]}")
    print(f"  inertia (diagonal in ipos/iquat frame) = {model.body_inertia[i]}")
    print(f"  parentid = {model.body_parentid[i]}")
    print(f"  xpos (world) = {data.xpos[i]}")
    print(f"  xquat (world) = {data.xquat[i]}")
    print(f"  xipos (inertia world pos) = {data.xipos[i]}")
    print(f"  ximat (inertia world rot, 3x3) =")
    mat = data.ximat[i].reshape(3,3)
    print(f"    {mat}")

print("\n\n=== Geom Info ===")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    gtype = model.geom_type[i]
    body_id = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    type_names = {0: 'plane', 3: 'capsule'}
    pos = model.geom_pos[i]
    quat = model.geom_quat[i]
    size = model.geom_size[i]
    print(f"\nGeom {i}: {name} ({type_names.get(gtype, '?')})")
    print(f"  body = {body_name} (id={body_id})")
    print(f"  pos (in body frame) = {pos}")
    print(f"  quat (in body frame) = {quat}")
    print(f"  size = {size}")
    # For capsules, xpos = world position of geom center
    print(f"  xpos (world) = {data.geom_xpos[i]}")
