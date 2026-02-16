"""Check MuJoCo's actual efc_aref and efc_b sign convention."""
import mujoco
import numpy as np

xml = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.01">
    <flag contact="enable"/>
  </option>
  <worldbody>
    <geom type="plane" size="10 10 0.1"/>
    <body pos="0 0 0.05">
      <joint type="free"/>
      <geom type="sphere" size="0.05" mass="1" friction="0.4"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

for i in range(5):
    mujoco.mj_step(model, data)
    if data.nefc > 0:
        print(f"\n=== Step {i}: {data.nefc} constraints ===")
        print(f"qpos z={data.qpos[2]:.6f}, qvel z={data.qvel[2]:.6f}")
        print(f"qacc = {data.qacc[:3]}")

        for c in range(min(data.nefc, 2)):
            # Full J row
            J_row = data.efc_J[c*model.nv:(c+1)*model.nv]
            vel = np.dot(J_row, data.qvel)

            print(f"\n  Constraint {c}:")
            print(f"    efc_aref  = {data.efc_aref[c]:.8f}")
            print(f"    efc_b     = {data.efc_b[c]:.8f}")
            print(f"    efc_pos   = {data.efc_pos[c]:.8f}")
            print(f"    efc_vel   = {data.efc_vel[c]:.8f}")
            print(f"    efc_force = {data.efc_force[c]:.8f}")
            print(f"    efc_J     = {J_row}")
            print(f"    J*qvel    = {vel:.8f} (should match efc_vel)")
            print(f"    efc_AR[c,c] = {data.efc_AR[c*data.nefc+c]:.8f}")

            # Check: efc_b should be efc_aref + J*qacc_free?
            # Or efc_b = residual = efc_aref + AR*force?
            AR_force = 0
            for j in range(data.nefc):
                AR_force += data.efc_AR[c*data.nefc+j] * data.efc_force[j]
            print(f"    AR*force  = {AR_force:.8f}")
            print(f"    aref + AR*force = {data.efc_aref[c] + AR_force:.8f}")

            # Check if efc_b = J*qacc0 + aref (before constraints)
            # qacc0 = M^{-1}*f_applied (before constraint forces)
            # Actually compute J * qacc
            J_qacc = np.dot(J_row, data.qacc)
            print(f"    J*qacc    = {J_qacc:.8f}")

        # Show solref
        tc = model.opt.o_solref[0]
        dr = model.opt.o_solref[1]
        K = 1.0 / (tc*tc)
        B = 2.0 * dr / tc
        print(f"\n  K={K:.1f}, B={B:.1f}")
        print(f"  solref={model.opt.o_solref}, solimp={model.opt.o_solimp}")
