"""Compare MuJoCo's impedance/regularizer at different penetration depths.

Computes what our solver would produce vs MuJoCo for the same contact.

Run with: python tests/mujoco_impedance_compare.py
"""
import numpy as np
import mujoco
import os

xml_path = os.path.expanduser("~/Documents/mojo-rl/Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f"dt = {model.opt.timestep}")
print(f"impratio = {model.opt.impratio}")
print(f"solref = {model.opt.o_solref}")
print(f"solimp = {model.opt.o_solimp}")
print(f"geom solimp = {model.geom_solimp[1]}")  # torso geom
print()

# MuJoCo's impedance function for solimp = [0.0, 0.8, 0.01, 0.5, 2]
solimp = model.geom_solimp[1]
dmin, dmax, width, midpoint, power = solimp
solref = model.geom_solref[1]
tc, dr = solref

print(f"solimp: dmin={dmin}, dmax={dmax}, width={width}, midpoint={midpoint}, power={power}")
print(f"solref: tc={tc}, dr={dr}")
print()

# MuJoCo impedance function (5-parameter version)
def mujoco_impedance(pen, dmin, dmax, width, midpoint, power):
    """MuJoCo's 5-parameter impedance function."""
    if pen <= 0:
        return max(dmin, 0.0001)
    x = pen / width
    if x >= 1:
        return dmax
    # 5-parameter sigmoid: d(x) = dmin + (dmax-dmin) * sigmoid(x, midpoint, power)
    # The exact formula from MuJoCo source:
    if x < midpoint:
        t = 0.5 * (x / midpoint) ** power
    else:
        t = 1.0 - 0.5 * ((1 - x) / (1 - midpoint)) ** power
    d = dmin + t * (dmax - dmin)
    return max(d, 0.0001)

# Our impedance function (3-parameter smoothstep)
def our_impedance(pen, dmin, dmax, width):
    """Our 3-parameter smoothstep impedance."""
    x = pen / width
    if x > 1:
        x = 1
    imp = dmin + (3*x*x - 2*x*x*x) * (dmax - dmin)
    return max(imp, 0.2)

# Compute K_spring and B_damp (acceleration-level)
K_spring = 1.0 / (dmax**2 * tc**2 * dr**2)
B_damp = 2.0 / (dmax * tc)
print(f"K_spring = {K_spring:.2f}")
print(f"B_damp = {B_damp:.2f}")
print()

K_contact = 0.17  # typical J*M_inv*J^T value from our output

print(f"{'pen_mm':>8} | {'d_mujoco':>10} | {'d_ours':>10} | {'R_mj':>10} | {'R_ours':>10} | {'AR_mj':>10} | {'AR_ours':>10} | {'aref':>10}")
print("-" * 100)

for pen_mm in [0.1, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 50, 100]:
    pen = pen_mm / 1000.0

    d_mj = mujoco_impedance(pen, dmin, dmax, width, midpoint, power)
    d_ours = our_impedance(pen, dmin, dmax, width)

    # R = (1/d - 1) * K for both
    R_mj = (1.0/d_mj - 1) * K_contact if d_mj > 0.0001 else 999
    R_ours = (1.0/d_ours - 1) * K_contact

    AR_mj = K_contact + R_mj
    AR_ours = K_contact + R_ours

    # aref = K_spring * d * pen (for v_n = 0)
    aref_mj = K_spring * d_mj * pen
    aref_ours = K_spring * d_ours * pen

    print(f"{pen_mm:8.1f} | {d_mj:10.6f} | {d_ours:10.6f} | {R_mj:10.4f} | {R_ours:10.4f} | {AR_mj:10.4f} | {AR_ours:10.4f} | {aref_ours:10.2f}")

print()
print("Key insight: smaller R = stiffer constraint, larger aref = stronger restoring force")
print()

# Now show what happens with velocity damping
v_n = -5.0  # entering ground at 5 m/s
print(f"\nWith v_n = {v_n} m/s (entering ground):")
print(f"{'pen_mm':>8} | {'aref_mj':>10} | {'aref_ours':>10} | {'force_mj':>10} | {'force_ours':>10} | {'a_eff_mj':>10} | {'a_eff_ours':>10}")
print("-" * 100)

for pen_mm in [0.1, 1, 5, 10, 20, 50]:
    pen = pen_mm / 1000.0

    d_mj = mujoco_impedance(pen, dmin, dmax, width, midpoint, power)
    d_ours = our_impedance(pen, dmin, dmax, width)

    R_mj = (1.0/d_mj - 1) * K_contact if d_mj > 0.0001 else 999
    R_ours = (1.0/d_ours - 1) * K_contact

    AR_mj = K_contact + R_mj
    AR_ours = K_contact + R_ours

    # aref = K_spring * d * pen + B_damp * abs(v_n)
    aref_mj = K_spring * d_mj * pen + B_damp * abs(v_n)
    aref_ours = K_spring * d_ours * pen + B_damp * abs(v_n)

    # force = aref / AR (single contact solution)
    force_mj = aref_mj / AR_mj if AR_mj > 0 else 0
    force_ours = aref_ours / AR_ours if AR_ours > 0 else 0

    # effective acceleration = K * force (= aref - R * force)
    a_eff_mj = K_contact * force_mj
    a_eff_ours = K_contact * force_ours

    print(f"{pen_mm:8.1f} | {aref_mj:10.2f} | {aref_ours:10.2f} | {force_mj:10.2f} | {force_ours:10.2f} | {a_eff_mj:10.2f} | {a_eff_ours:10.2f}")

print()
print("a_eff = effective contact acceleration in normal direction")
print("  = K * force = aref * K/AR = aref * d  (since AR = K/d)")
print("  So effective_accel = aref * d = d^2 * K_spring * pen + d * B * v_n")
print("  At shallow pen with d=0.2: effective = 0.04 * K_spring * pen + 0.2 * B * v_n")
print("  At shallow pen with d=0.0001 (MuJoCo): effective ≈ B * v_n * 0.0001")
print()
print("Wait — this means MuJoCo's effective acceleration is EVEN WEAKER than ours at shallow penetration!")
print("MuJoCo d=0.0001 at surface vs our d=0.2 floor. We're 2000x STRONGER initially.")
print("Yet MuJoCo has 75x less penetration. The issue must be elsewhere.")
