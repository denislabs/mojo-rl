"""Compare MuJoCo pyramidal vs elliptic + dump detailed constraint info."""
import mujoco
import numpy as np

xml_path = "/Users/denislaboureyras/Documents/mojo-rl/Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"

# Test PYRAMIDAL with detailed constraint info at multiple contact steps
model = mujoco.MjModel.from_xml_path(xml_path)
model.opt.cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
data = mujoco.MjData(model)
print(f"=== PYRAMIDAL detailed ===")
print(f"dt={model.opt.timestep}, solver={model.opt.solver}")
print(f"solref={model.opt.o_solref}")
print(f"solimp={model.opt.o_solimp}")
print(f"K = 1/tc^2 = {1/(model.opt.o_solref[0]**2):.1f}")
print(f"B = 2*dr/tc = {2*model.opt.o_solref[1]/model.opt.o_solref[0]:.1f}")
print(f"friction = {model.geom_friction}")
print()

mujoco.mj_resetData(model, data)
data.qpos[:] = model.qpos0
data.qvel[:] = 0

for step in range(20):
    mujoco.mj_step(model, data)
    z = data.qpos[1]
    vz = data.qvel[1]
    vx = data.qvel[0]

    if data.nefc > 0:
        print(f"step {step:2d}  z={z:10.6f}  vz={vz:10.6f}  vx={vx:10.6f}  nefc={data.nefc} ncon={data.ncon}")
        for i in range(min(data.nefc, 8)):
            print(f"  efc[{i}]: aref={data.efc_aref[i]:12.4f}  "
                  f"force={data.efc_force[i]:10.4f}  "
                  f"R={data.efc_R[i]:10.4f}  "
                  f"diagApprox={data.efc_diagApprox[i]:10.4f}")
    else:
        print(f"step {step:2d}  z={z:10.6f}  vz={vz:10.6f}  vx={vx:10.6f}  no contact")

# Now test with solimp matching our values
print("\n" + "="*70)
print("=== PYRAMIDAL with our solimp [0.0, 0.8, 0.01] ===")
model2 = mujoco.MjModel.from_xml_path(xml_path)
model2.opt.cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
# Set custom solimp
model2.opt.o_solimp[0] = 0.0   # dmin
model2.opt.o_solimp[1] = 0.8   # dmax
model2.opt.o_solimp[2] = 0.01  # width
data2 = mujoco.MjData(model2)
mujoco.mj_resetData(model2, data2)
data2.qpos[:] = model2.qpos0
data2.qvel[:] = 0

for step in range(20):
    mujoco.mj_step(model2, data2)
    z = data2.qpos[1]
    vz = data2.qvel[1]
    vx = data2.qvel[0]

    if data2.nefc > 0:
        print(f"step {step:2d}  z={z:10.6f}  vz={vz:10.6f}  vx={vx:10.6f}  nefc={data2.nefc}")
        for i in range(min(data2.nefc, 4)):
            print(f"  efc[{i}]: aref={data2.efc_aref[i]:12.4f}  "
                  f"force={data2.efc_force[i]:10.4f}  "
                  f"R={data2.efc_R[i]:10.4f}")
    else:
        print(f"step {step:2d}  z={z:10.6f}  vz={vz:10.6f}  vx={vx:10.6f}  no contact")
    if z > 2.0:
        print("FLYING!")
        break
