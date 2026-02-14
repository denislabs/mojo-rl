"""Compare settled + actions behavior: MuJoCo vs our engine."""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(
    ".pixi/envs/default/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/half_cheetah.xml"
)
data = mujoco.MjData(model)

# Settle
for _ in range(100):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)

print(f"After settling:")
print(f"  rootx={data.qpos[0]:.6f} rootz={data.qpos[1]:.6f} vx={data.qvel[0]:.6f}")
print(f"  ncon={data.ncon}")

# Apply +1.0 actions for 100 steps
print(f"\n+1.0 actions:")
for step in range(100):
    data.ctrl[:] = 1.0
    mujoco.mj_step(model, data)
    if step < 20 or step % 10 == 0:
        print(f"  Step {step+1}: rootx={data.qpos[0]:.4f} rootz={data.qpos[1]:.4f} "
              f"vx={data.qvel[0]:.4f} vz={data.qvel[1]:.4f} ncon={data.ncon}")
