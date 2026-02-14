"""Test MuJoCo with elliptic cone to compare with our elliptic solver."""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(
    ".pixi/envs/default/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/half_cheetah.xml"
)
# Switch to elliptic cone
model.opt.cone = 1  # mjCONE_ELLIPTIC
data = mujoco.MjData(model)

# Settle
for _ in range(200):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)

print("Elliptic cone mode:")
print(f"  ncon = {data.ncon}, nefc = {data.nefc}")

# Inject vx=1.0
data.qvel[0] = 1.0
mujoco.mj_forward(model, data)

print(f"\nAfter vx=1.0 injection:")
print(f"  ncon = {data.ncon}, nefc = {data.nefc}")
print(f"  qacc = {data.qacc[:9]}")
print(f"  qfrc_constraint = {data.qfrc_constraint[:9]}")

# Print constraint rows
efc_J = data.efc_J.reshape(data.nefc, model.nv) if data.nefc > 0 else np.array([])
for i in range(data.nefc):
    print(f"\n  Row {i}: type={data.efc_type[i]} force={data.efc_force[i]:.4f} "
          f"R={data.efc_R[i]:.6f} aref={data.efc_aref[i]:.4f}")
    if data.nefc > 0:
        print(f"    J[:9] = {efc_J[i,:9]}")

# Now step and compare
data2 = mujoco.MjData(model)
for _ in range(200):
    data2.ctrl[:] = 0.0
    mujoco.mj_step(model, data2)

data2.qvel[0] = 1.0
print("\n=== 20 steps with vx=1.0 injected (elliptic cone) ===")
for step in range(20):
    mujoco.mj_step(model, data2)
    print(f"  Step {step+1}: vx={data2.qvel[0]:.4f} ncon={data2.ncon} "
          f"qfrc_constraint[0]={data2.qfrc_constraint[0]:.4f}")

# Compare with pyramidal
model_p = mujoco.MjModel.from_xml_path(
    ".pixi/envs/default/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/half_cheetah.xml"
)
model_p.opt.cone = 0  # pyramidal (default)
data_p = mujoco.MjData(model_p)
for _ in range(200):
    data_p.ctrl[:] = 0.0
    mujoco.mj_step(model_p, data_p)
data_p.qvel[0] = 1.0
print("\n=== 20 steps with vx=1.0 injected (pyramidal cone) ===")
for step in range(20):
    mujoco.mj_step(model_p, data_p)
    print(f"  Step {step+1}: vx={data_p.qvel[0]:.4f} ncon={data_p.ncon} "
          f"qfrc_constraint[0]={data_p.qfrc_constraint[0]:.4f}")
