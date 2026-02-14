"""Get MuJoCo's settled state for use in our engine."""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(
    ".pixi/envs/default/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/half_cheetah.xml"
)
data = mujoco.MjData(model)

print("Initial state:")
print(f"  qpos = {list(data.qpos[:9])}")
print(f"  qvel = {list(data.qvel[:9])}")
print(f"  ncon = {data.ncon}")

# Settle with zero actions
for _ in range(200):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)

print(f"\nAfter 200 zero-action steps:")
print(f"  qpos = {list(data.qpos[:9])}")
print(f"  qvel = {list(data.qvel[:9])}")
print(f"  ncon = {data.ncon}")

# Print for easy copy-paste into Mojo
print("\n# Mojo copy-paste:")
for i, v in enumerate(data.qpos[:9]):
    print(f"    env.data.qpos[{i}] = {v:.15f}")
for i, v in enumerate(data.qvel[:9]):
    print(f"    env.data.qvel[{i}] = {v:.15f}")

# Also print body positions
print(f"\n  Body positions:")
for i in range(model.nbody):
    print(f"    body {i} ({model.body(i).name}): pos={data.xpos[i]}")

# Print initial qpos for reference
print(f"\n  Initial qpos (from XML keyframe or defaults):")
data2 = mujoco.MjData(model)
print(f"    qpos = {list(data2.qpos[:9])}")
