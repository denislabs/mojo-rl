"""Compare: MuJoCo HalfCheetah under random torques.
Shows max penetration, max velocity, rootz range.

Run with: python tests/mujoco_random_action_test.py
"""
import numpy as np
try:
    import mujoco
except ImportError:
    print("mujoco not installed. Run: pip install mujoco")
    exit(1)

import os

# Load the HalfCheetah model
xml_path = os.path.join(os.path.dirname(__file__),
    "../../../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml")
if not os.path.exists(xml_path):
    # Try alternate path
    xml_path = os.path.expanduser("~/Documents/mojo-rl/Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml")

print(f"Loading model from: {xml_path}")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f"dt = {model.opt.timestep}")
print(f"nv = {model.nv}, nq = {model.nq}")
print(f"nu = {model.nu}")
print(f"solref = {model.opt.o_solref}")
print(f"solimp = {model.opt.o_solimp}")
print()

# Check per-geom solimp
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    print(f"  geom {i} ({name}): solref={model.geom_solref[i]}, solimp={model.geom_solimp[i]}")
print()

np.random.seed(42)
mujoco.mj_resetData(model, data)

MAX_STEPS = 1000
max_pen = 0.0
max_vel = 0.0
max_rootz = 0.0
min_rootz = 10.0
max_ncon = 0

for step in range(MAX_STEPS):
    # Random actions in [-1, 1]
    data.ctrl[:] = np.random.uniform(-1, 1, model.nu)

    mujoco.mj_step(model, data)

    rootz = data.qpos[1]  # HalfCheetah rootz is index 1

    # Check contacts
    ncon = data.ncon
    if ncon > max_ncon:
        max_ncon = ncon

    for c in range(ncon):
        contact = data.contact[c]
        pen = -contact.dist
        if pen > max_pen:
            max_pen = pen

    vel = np.max(np.abs(data.qvel))
    if vel > max_vel:
        max_vel = vel

    if rootz > max_rootz:
        max_rootz = rootz
    if rootz < min_rootz:
        min_rootz = rootz

    if (step + 1) % 100 == 0:
        pitch = data.qpos[2]
        print(f"  step {step+1}: rootz={rootz:.4f} pitch={pitch:.4f} ncon={ncon} max_pen={max_pen:.6f} max_vel={vel:.4f}")

print()
print("=" * 60)
print("MUJOCO RESULTS (HalfCheetah, random torques):")
print(f"  Max penetration:  {max_pen:.6f} m")
print(f"  Max velocity:     {max_vel:.4f}")
print(f"  Max rootz:        {max_rootz:.4f}")
print(f"  Min rootz:        {min_rootz:.4f}")
print(f"  Max contacts:     {max_ncon}")
print("=" * 60)
