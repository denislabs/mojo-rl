"""Test: does MuJoCo friction damp existing sliding velocity?"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(
    ".pixi/envs/default/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/half_cheetah.xml"
)
data = mujoco.MjData(model)

# Settle first
for _ in range(200):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)

print("After settling:")
print(f"  rootx={data.qpos[0]:.4f} rootz={data.qpos[1]:.4f} vx={data.qvel[0]:.4f} ncon={data.ncon}")

# Inject horizontal sliding velocity
data.qvel[0] = 1.0  # 1 m/s in X direction
print(f"\nInjected vx=1.0, zero actions:")

for step in range(50):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)
    if step < 10 or step % 5 == 0:
        # Get friction forces
        fx_total = 0.0
        for c in range(data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, c, force)
            fx_total += force[1]  # friction along t1
        print(f"  Step {step+1}: vx={data.qvel[0]:.4f} nc={data.ncon} "
              f"qacc_x={data.qacc[0]:.2f} qfrc_constraint_x={data.qfrc_constraint[0]:.2f}")
