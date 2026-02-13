"""
Side-by-side comparison of MuJoCo dynamics vs our engine.
Dumps: mass matrix, bias forces, unconstrained qacc, contacts, constraint forces.

Run: python tests/mujoco_dynamics_compare.py

Compare the output against our engine's values at the same state.
"""

import numpy as np
import mujoco

np.set_printoptions(precision=8, linewidth=200, suppress=True)

# Load HalfCheetah
model = mujoco.MjModel.from_xml_path("../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml")
data = mujoco.MjData(model)

print(f"=== MuJoCo HalfCheetah Dynamics ===")
print(f"nq={model.nq}, nv={model.nv}, nbody={model.nbody}")
print(f"dt={model.opt.timestep}")
print(f"gravity={model.opt.gravity}")
print()

# Joint names
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jnt_type = model.jnt_type[i]
    type_names = ['free', 'ball', 'slide', 'hinge']
    body_id = model.jnt_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    print(f"  Joint {i}: {name} (type={type_names[jnt_type]}, body={body_name})")

print()

# Body info
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    mass = model.body_mass[i]
    parent = model.body_parentid[i]
    print(f"  Body {i}: {name} (mass={mass:.4f}, parent={parent})")

print()

# Geom info
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    gtype = model.geom_type[i]
    body_id = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    type_names = {0: 'plane', 1: 'hfield', 2: 'sphere', 3: 'capsule', 4: 'ellipsoid', 5: 'cylinder', 6: 'box', 7: 'mesh'}
    size = model.geom_size[i]
    print(f"  Geom {i}: {name} (type={type_names.get(gtype, '?')}, body={body_name}, size={size})")

print()

# Solver parameters
print(f"solref_contact: {model.opt.cone}")
print(f"solver type: {model.opt.solver}")  # 0=PGS, 1=CG, 2=Newton
print(f"iterations: {model.opt.iterations}")
print(f"noslip_iterations: {model.opt.noslip_iterations}")
print()

def dump_state(label, qpos, qvel, ctrl=None):
    """Set state, compute forward dynamics, dump everything."""
    print(f"\n{'='*80}")
    print(f"=== {label} ===")
    print(f"{'='*80}")

    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    if ctrl is not None:
        data.ctrl[:] = ctrl

    # Forward pass (computes everything)
    mujoco.mj_forward(model, data)

    print(f"\nqpos = {data.qpos}")
    print(f"qvel = {data.qvel}")
    if ctrl is not None:
        print(f"ctrl = {data.ctrl}")

    # Mass matrix
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    print(f"\nMass matrix M ({model.nv}x{model.nv}):")
    print(M)

    # Mass matrix diagonal (for quick comparison)
    print(f"\nM diagonal: {np.diag(M)}")

    # Bias forces (gravity + Coriolis + centrifugal)
    print(f"\nqfrc_bias (gravity+Coriolis+centrifugal): {data.qfrc_bias}")

    # Passive forces (spring, damping)
    print(f"qfrc_passive: {data.qfrc_passive}")

    # Actuator forces
    print(f"qfrc_actuator: {data.qfrc_actuator}")

    # Applied forces (external)
    print(f"qfrc_applied: {data.qfrc_applied}")

    # Total unconstrained acceleration
    # MuJoCo: qacc0 = M^-1 * (qfrc_bias + qfrc_passive + qfrc_actuator + qfrc_applied)
    # Actually MuJoCo stores qacc (after constraints) but we can compute qacc0
    f_total = data.qfrc_bias + data.qfrc_passive + data.qfrc_actuator + data.qfrc_applied
    print(f"\nf_total (bias+passive+actuator+applied): {f_total}")

    M_inv = np.linalg.inv(M)
    qacc_unconstrained = M_inv @ f_total
    print(f"qacc_unconstrained (M^-1 * f_total): {qacc_unconstrained}")

    # Constrained acceleration (what MuJoCo actually computes)
    print(f"\nqacc (constrained): {data.qacc}")

    # Constraint forces
    print(f"qfrc_constraint: {data.qfrc_constraint}")

    # Contact info
    print(f"\nncon (number of contacts): {data.ncon}")
    for i in range(data.ncon):
        c = data.contact[i]
        print(f"  Contact {i}: pos={c.pos}, normal=geom1={c.geom1} geom2={c.geom2}, dist={c.dist:.6f}, dim={c.dim}")

    # Constraint info (efc = equality/friction/contact constraints)
    print(f"\nnefc (active constraints): {data.nefc}")
    if data.nefc > 0:
        print(f"efc_type: {data.efc_type[:data.nefc]}")
        print(f"efc_aref: {data.efc_aref[:data.nefc]}")
        print(f"efc_force: {data.efc_force[:data.nefc]}")
        print(f"efc_b: {data.efc_b[:data.nefc]}")
        print(f"efc_R: {data.efc_R[:data.nefc]}")
        print(f"efc_D: {data.efc_D[:data.nefc]}")
        print(f"efc_diagApprox: {data.efc_diagApprox[:data.nefc]}")

    # Body positions
    print(f"\nxpos (body positions):")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  {name}: {data.xpos[i]}")

    # After one step
    mujoco.mj_step(model, data)
    print(f"\nAfter one step (dt={model.opt.timestep}):")
    print(f"qpos = {data.qpos}")
    print(f"qvel = {data.qvel}")
    print(f"qacc = {data.qacc}")

    return M, f_total, qacc_unconstrained


# ===== TEST 1: Initial state (zero velocity, standing) =====
nq = model.nq
nv = model.nv

qpos0 = np.zeros(nq)
# MuJoCo's default qpos for HalfCheetah
mujoco.mj_resetData(model, data)
qpos0 = data.qpos.copy()
qvel0 = np.zeros(nv)

print(f"\nDefault qpos after reset: {qpos0}")
print(f"Default qvel after reset: {data.qvel}")

M0, f0, qacc0 = dump_state("TEST 1: Default state (zero velocity)", qpos0, qvel0)


# ===== TEST 2: State with moderate velocities =====
qpos1 = qpos0.copy()
qvel1 = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0, -3.0, 1.0, -0.5])[:nv]
dump_state("TEST 2: Moderate velocities", qpos1, qvel1)


# ===== TEST 3: State with action applied =====
ctrl = np.array([1.0, -1.0, 0.5, -0.5, 1.0, -1.0])[:model.nu]
dump_state("TEST 3: Default state with action", qpos0, qvel0, ctrl=ctrl)


# ===== TEST 4: State after a few random steps (with penetration) =====
mujoco.mj_resetData(model, data)
np.random.seed(42)
for i in range(20):
    data.ctrl[:] = np.random.uniform(-1, 1, model.nu)
    mujoco.mj_step(model, data)

qpos_after = data.qpos.copy()
qvel_after = data.qvel.copy()
print(f"\n\nAfter 20 random steps:")
print(f"qpos = {qpos_after}")
print(f"qvel = {qvel_after}")
dump_state("TEST 4: After 20 random steps", qpos_after, qvel_after, ctrl=np.zeros(model.nu))


# ===== TEST 5: High velocity state (stress test) =====
qpos5 = qpos0.copy()
qvel5 = np.ones(nv) * 5.0  # 5 rad/s everywhere
dump_state("TEST 5: High velocity (5 rad/s all)", qpos5, qvel5)


# ===== Summary: Joint dynamics params =====
print(f"\n\n{'='*80}")
print("=== Joint Parameters ===")
print(f"{'='*80}")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    stiffness = model.jnt_stiffness[i]

    # Get damping from dof
    dof_start = model.jnt_dofadr[i]
    n_dof = 1  # all are slide or hinge
    damping = model.dof_damping[dof_start]
    armature = model.dof_armature[dof_start]

    print(f"  Joint {i} ({name}): stiffness={stiffness:.2f}, damping={damping:.4f}, armature={armature:.4f}")

# Actuator info
print(f"\nActuators (nu={model.nu}):")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    gear = model.actuator_gear[i]
    ctrlrange = model.actuator_ctrlrange[i]
    print(f"  Actuator {i} ({name}): gear={gear[0]:.1f}, ctrlrange=[{ctrlrange[0]:.1f}, {ctrlrange[1]:.1f}]")

# Solref/solimp
print(f"\nDefault solref: {model.opt.o_solref}")
print(f"Default solimp: {model.opt.o_solimp}")
