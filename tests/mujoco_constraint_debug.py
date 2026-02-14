"""Compare constraint solver outputs between MuJoCo and our engine.

Settles HalfCheetah, injects vx=1.0, then prints full constraint data
for one solver step.
"""
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

print("After settling:")
print(f"  qpos = {data.qpos[:9]}")
print(f"  qvel = {data.qvel[:9]}")
print(f"  ncon = {data.ncon}")

# Inject vx=1.0
data.qvel[0] = 1.0

# Run forward to get constraint data without stepping
mujoco.mj_forward(model, data)

print(f"\nAfter mj_forward with vx=1.0:")
print(f"  ncon = {data.ncon}")
print(f"  nefc = {data.nefc}")  # number of active constraints
print(f"  qacc = {data.qacc[:9]}")
print(f"  qfrc_constraint = {data.qfrc_constraint[:9]}")

# Print each constraint row
for i in range(data.nefc):
    efc_type = data.efc_type[i]  # constraint type
    type_names = {0: "EQUALITY", 1: "FRICTION_DOF", 2: "LIMIT", 3: "CONTACT_NORMAL",
                  4: "CONTACT_FRICTION_SLIDE", 5: "CONTACT_FRICTION_SPIN",
                  6: "CONTACT_FRICTION_ROLL"}
    name = type_names.get(efc_type, f"UNKNOWN({efc_type})")

    J = data.efc_J[i]  # Jacobian row
    force = data.efc_force[i]  # constraint force (lambda)
    aref = data.efc_aref[i]  # reference acceleration
    diagApprox = data.efc_diagApprox[i]  # Delassus diagonal approx
    R = data.efc_R[i]  # regularizer
    D = data.efc_D[i]  # 1/(diag + R) effective inverse
    b = data.efc_b[i]  # bias = -aref

    print(f"\n  Constraint {i}: type={efc_type}")
    print(f"    force(lambda) = {force:.6f}")
    print(f"    aref = {aref:.6f}")
    print(f"    b(bias) = {b:.6f}")
    print(f"    diagApprox = {diagApprox:.6f}")
    print(f"    R = {R:.6f}")
    print(f"    D = {D:.6f}")
    J_row = data.efc_J[i*model.nv:(i+1)*model.nv]
    print(f"    J[:9] = {J_row[:9]}")

# Print contact details
print(f"\n=== Contact details ===")
for i in range(data.ncon):
    c = data.contact[i]
    print(f"  Contact {i}:")
    print(f"    pos = [{c.pos[0]:.6f}, {c.pos[1]:.6f}, {c.pos[2]:.6f}]")
    print(f"    frame = {c.frame}")
    print(f"    dist = {c.dist:.6f}")
    print(f"    geom = [{c.geom[0]}, {c.geom[1]}]")
    print(f"    mu = {c.mu:.6f}")
    print(f"    dim = {c.dim}")

# Also print M_inv * J^T for first constraint to compare coupling
M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, M, data.qM)
M_inv = np.linalg.inv(M)
print(f"\n=== M_inv diagonal = {np.diag(M_inv)[:9]}")
print(f"=== M_inv[0,:] = {M_inv[0,:9]}")

# For each normal constraint, show M_inv * J_normal^T
efc_J = data.efc_J.reshape(data.nefc, model.nv)
for i in range(data.nefc):
    J_row = efc_J[i]
    MinvJT = M_inv @ J_row
    K_del = J_row @ MinvJT
    print(f"\n  Constraint {i}: K_delassus = {K_del:.6f}, M_inv*J^T[:9] = {MinvJT[:9]}")

# Tangential acceleration analysis
print(f"\n=== Tangential acceleration analysis ===")
f_total = data.qfrc_passive + data.qfrc_actuator - data.qfrc_bias
qacc_free = M_inv @ f_total
print(f"  qacc_free = {qacc_free[:9]}")

for i in range(data.nefc):
    J_row = efc_J[i]
    a_free = J_row @ qacc_free
    a_solved = J_row @ data.qacc
    print(f"  Constraint {i}: a_free={a_free:.6f}, a_solved={a_solved:.6f}, force={data.efc_force[i]:.6f}")
