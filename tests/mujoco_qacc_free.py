"""Compare unconstrained qacc between MuJoCo and our engine."""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(
    ".pixi/envs/default/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/half_cheetah.xml"
)
data = mujoco.MjData(model)

# Settle
for _ in range(200):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)

print("Settled state:")
print(f"  qpos = {data.qpos[:9]}")
print(f"  qvel = {data.qvel[:9]}")

# Inject vx=1.0
data.qvel[0] = 1.0

# Compute forward dynamics without stepping
mujoco.mj_forward(model, data)

print(f"\nAfter mj_forward with vx=1.0:")
print(f"  qacc = {data.qacc[:9]}")
print(f"  qfrc_bias = {data.qfrc_bias[:9]}")
print(f"  qfrc_passive = {data.qfrc_passive[:9]}")
print(f"  qfrc_actuator = {data.qfrc_actuator[:9]}")
print(f"  qfrc_constraint = {data.qfrc_constraint[:9]}")
print(f"  ncon = {data.ncon}")

# Compute unconstrained qacc: M * qacc_free = f_passive - f_bias
# where f_bias = Coriolis + centrifugal + gravity
# In MuJoCo: qacc_free = M^-1 * (qfrc_passive + qfrc_actuator + qfrc_applied - qfrc_bias)
M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, M, data.qM)
M_inv = np.linalg.inv(M)

f_total_unconstrained = data.qfrc_passive + data.qfrc_actuator - data.qfrc_bias
qacc_free = M_inv @ f_total_unconstrained

print(f"\nM diagonal = {np.diag(M)[:9]}")
print(f"M_inv[0,:] = {M_inv[0,:9]}")
print(f"\nf_total_unconstrained = {f_total_unconstrained[:9]}")
print(f"qacc_free (computed) = {qacc_free[:9]}")
print(f"qacc_free[0] (rootx) = {qacc_free[0]:.6f}")

# Now with zero velocity for comparison
data.qvel[0] = 0.0
mujoco.mj_forward(model, data)
f_total_unconstrained_v0 = data.qfrc_passive + data.qfrc_actuator - data.qfrc_bias
qacc_free_v0 = M_inv @ f_total_unconstrained_v0
print(f"\nWith vx=0:")
print(f"  f_total_unconstrained = {f_total_unconstrained_v0[:9]}")
print(f"  qacc_free = {qacc_free_v0[:9]}")
print(f"  qacc_free[0] (rootx) = {qacc_free_v0[0]:.6f}")

# Difference shows velocity-dependent part
print(f"\nDifference (vx=1 - vx=0):")
print(f"  f_diff = {(f_total_unconstrained - f_total_unconstrained_v0)[:9]}")
print(f"  qacc_diff = {(qacc_free - qacc_free_v0)[:9]}")
