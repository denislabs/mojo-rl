"""Compare MuJoCo vs our engine: settle first, then apply actions."""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(".pixi/envs/default/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/half_cheetah.xml")
data = mujoco.MjData(model)

# Phase 1: settle with zero actions (100 steps)
for _ in range(100):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)

print("=== After settling (100 zero-action steps) ===")
print(f"rootx={data.qpos[0]:.6f} rootz={data.qpos[1]:.6f} rooty={data.qpos[2]:.6f}")
print(f"vx={data.qvel[0]:.6f} vz={data.qvel[1]:.6f} vy={data.qvel[2]:.6f}")
print(f"ncon={data.ncon}")
print()

# Phase 2: apply +1.0 actions from settled state
data.ctrl[:] = 1.0
print("=== Now applying +1.0 actions ===")
for step in range(100):
    mujoco.mj_step(model, data)
    if step < 10 or step % 10 == 0:
        print(f"Step {step+1}: rootx={data.qpos[0]:.4f} rootz={data.qpos[1]:.4f} "
              f"vx={data.qvel[0]:.4f} vz={data.qvel[1]:.4f} nc={data.ncon}")
        if step < 5:
            print(f"  qacc: {data.qacc[:9]}")
            print(f"  qfrc_constraint: {data.qfrc_constraint[:9]}")
            # Contact forces
            for c in range(data.ncon):
                ct = data.contact[c]
                # Get contact force
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, force)
                print(f"  contact {c}: body_a={ct.geom1} body_b={ct.geom2} "
                      f"dist={ct.dist:.6f} force={force[:3]}")
