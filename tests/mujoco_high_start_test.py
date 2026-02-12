"""MuJoCo HalfCheetah with high initial position (rootz=0.7) + random torques.
Tests whether MuJoCo also becomes unstable when starting from our initial height.

Run with: python tests/mujoco_high_start_test.py
"""
import numpy as np
import mujoco
import os

xml_path = os.path.expanduser("~/Documents/mojo-rl/Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f"dt = {model.opt.timestep}")
print(f"solver = {['PGS', 'CG', 'Newton'][model.opt.solver]}")

np.random.seed(42)
mujoco.mj_resetData(model, data)

# Set initial rootz = 0.7 to match our engine
data.qpos[1] = 0.7  # rootz
print(f"Initial qpos: {data.qpos}")
print()

MAX_STEPS = 1000
max_pen = 0.0
max_vel = 0.0
max_rootz = 0.7
min_rootz = 0.7

for step in range(MAX_STEPS):
    data.ctrl[:] = np.random.uniform(-1, 1, model.nu)
    mujoco.mj_step(model, data)

    rootz = data.qpos[1]
    ncon = data.ncon

    for c in range(ncon):
        pen = -data.contact[c].dist
        if pen > max_pen:
            max_pen = pen

    vel = np.max(np.abs(data.qvel))
    if vel > max_vel:
        max_vel = vel

    if rootz > max_rootz:
        max_rootz = rootz
    if rootz < min_rootz:
        min_rootz = rootz

    if (step + 1) % 100 == 0 or step < 10:
        pitch = data.qpos[2]
        print(f"  step {step+1}: rootz={rootz:.4f} pitch={pitch:.4f} ncon={ncon} max_pen={max_pen:.6f} max_vel={vel:.4f}")

print()
print("=" * 60)
print("MUJOCO RESULTS (rootz=0.7 start, random torques):")
print(f"  Max penetration:  {max_pen:.6f} m")
print(f"  Max velocity:     {max_vel:.4f}")
print(f"  Max rootz:        {max_rootz:.4f}")
print(f"  Min rootz:        {min_rootz:.4f}")
print("=" * 60)
