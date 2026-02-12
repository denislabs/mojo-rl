"""Deep debug: MuJoCo solver internals for HalfCheetah.
Shows contact forces, efc_* arrays, qacc, etc.

Run with: python tests/mujoco_solver_debug.py
"""
import numpy as np
import mujoco
import os

xml_path = os.path.expanduser("~/Documents/mojo-rl/Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f"dt = {model.opt.timestep}")
print(f"solver = {['PGS', 'CG', 'Newton'][model.opt.solver]}")
print(f"iterations = {model.opt.iterations}")
print(f"tolerance = {model.opt.tolerance}")
print(f"noslip_iterations = {model.opt.noslip_iterations}")
print(f"impratio = {model.opt.impratio}")
print(f"nv = {model.nv}")
print()

# Print geom properties
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    margin = model.geom_margin[i]
    gap = model.geom_gap[i]
    print(f"  geom {i} ({name}): type={model.geom_type[i]} margin={margin} gap={gap} size={model.geom_size[i]}")
print()

np.random.seed(42)
mujoco.mj_resetData(model, data)

# Run a few steps with verbose output
for step in range(20):
    data.ctrl[:] = np.random.uniform(-1, 1, model.nu)
    mujoco.mj_step(model, data)

    rootz = data.qpos[1]
    pitch = data.qpos[2]
    ncon = data.ncon

    if step < 5 or ncon > 0:
        print(f"Step {step+1}: rootz={rootz:.6f} pitch={pitch:.6f} ncon={ncon}")
        print(f"  qvel: {data.qvel}")
        print(f"  qacc: {data.qacc}")
        if ncon > 0:
            for c in range(ncon):
                ct = data.contact[c]
                print(f"  contact[{c}]: dist={ct.dist:.6f} pos={ct.pos} frame={ct.frame[:3]}")
                print(f"    geom1={ct.geom1} geom2={ct.geom2}")
            # Show constraint forces
            nefc = data.nefc
            print(f"  nefc={nefc}")
            if nefc > 0:
                print(f"  efc_force: {data.efc_force[:nefc]}")
                print(f"  efc_aref: {data.efc_aref[:nefc]}")
                print(f"  efc_D: {data.efc_D[:nefc]}")
                print(f"  efc_R: {data.efc_R[:nefc]}")
                print(f"  efc_diagApprox: {data.efc_diagApprox[:nefc]}")
        print()

# Now run 100 steps and show stats when contacts appear
print("=" * 60)
print("Running 1000 steps with random torques...")
max_pen = 0
max_efc_force = 0
for step in range(1000):
    data.ctrl[:] = np.random.uniform(-1, 1, model.nu)
    mujoco.mj_step(model, data)

    for c in range(data.ncon):
        pen = -data.contact[c].dist
        if pen > max_pen:
            max_pen = pen

    nefc = data.nefc
    if nefc > 0:
        max_f = np.max(np.abs(data.efc_force[:nefc]))
        if max_f > max_efc_force:
            max_efc_force = max_f

    if (step+1) % 200 == 0:
        rootz = data.qpos[1]
        print(f"  step {step+1}: rootz={rootz:.4f} ncon={data.ncon} nefc={data.nefc} max_pen={max_pen:.6f} max_efc_force={max_efc_force:.2f}")

print(f"\nFinal: max_pen={max_pen:.6f}m, max_efc_force={max_efc_force:.2f}")
