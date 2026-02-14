"""Get MuJoCo's body masses and inertia for comparison."""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(
    ".pixi/envs/default/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/half_cheetah.xml"
)
data = mujoco.MjData(model)

print("=== Body masses ===")
total_mass = 0
for i in range(model.nbody):
    name = model.body(i).name
    mass = model.body_mass[i]
    total_mass += mass
    inertia = model.body_inertia[i]
    ipos = model.body_ipos[i]
    iquat = model.body_iquat[i]
    print(f"  body {i} ({name}): mass={mass:.6f} inertia={inertia} ipos={ipos} iquat={iquat}")
print(f"  Total mass: {total_mass:.6f}")

print(f"\n=== Mass matrix (M_hat = M + armature + dt*damping, at qpos=0) ===")
# Reset to default state
data2 = mujoco.MjData(model)
mujoco.mj_forward(model, data2)

M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, M, data2.qM)
print(f"M diagonal = {np.diag(M)}")

# M_hat = M + armature + dt*D
M_hat = M.copy()
for i in range(model.nv):
    M_hat[i,i] += model.dof_armature[i]
    M_hat[i,i] += 0.01 * model.dof_damping[i]

print(f"M_hat diagonal = {np.diag(M_hat)}")
print(f"armature = {model.dof_armature}")
print(f"damping = {model.dof_damping}")

print(f"\n=== Geom properties ===")
for i in range(model.ngeom):
    name = model.geom(i).name if model.geom(i).name else f"geom{i}"
    gtype = model.geom_type[i]
    gsize = model.geom_size[i]
    gmass = model.geom(i).mass if hasattr(model.geom(i), 'mass') else 'N/A'
    friction = model.geom_friction[i]
    print(f"  geom {i} ({name}): type={gtype} size={gsize} friction={friction}")
