"""Diagnostic script: verify HalfCheetah XML parsing.

Tests parse_xml() and parse_xml_full() in isolation — does NOT import
HalfCheetahModel or any envs/ code, to avoid triggering full GPU kernel
compilation.

Run with:
    cd mojo-rl && pixi run mojo run test_xml_parse_diag.mojo
"""

from physics3d.parser import parse_xml, parse_xml_full


# Inline XML (copy of half_cheetah_xml.mojo) — avoids importing ModelDefFromXML
comptime _XML = """
<mujoco model="cheetah">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <size nstack="300000" nuser_geom="1"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 .7">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="-.5 0 0 .5 0 0" name="torso" size="0.046" type="capsule"/>
      <geom axisangle="0 1 0 .87" name="head" pos=".6 0 .1" size="0.046 .15" type="capsule"/>
      <!-- <site name='tip'  pos='.15 0 .11'/>-->
      <body name="bthigh" pos="-.5 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom axisangle="0 1 0 -3.8" name="bthigh" pos=".1 0 -.13" size="0.046 .145" type="capsule"/>
        <body name="bshin" pos=".16 0 -.25">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom axisangle="0 1 0 -2.03" name="bshin" pos="-.14 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .15" type="capsule"/>
          <body name="bfoot" pos="-.28 0 -.14">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom axisangle="0 1 0 -.27" name="bfoot" pos=".03 0 -.097" rgba="0.9 0.6 0.6 1" size="0.046 .094" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos=".5 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="fthigh" pos="-.07 0 -.12" size="0.046 .133" type="capsule"/>
        <body name="fshin" pos="-.14 0 -.24">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="fshin" pos=".065 0 -.09" rgba="0.9 0.6 0.6 1" size="0.046 .106" type="capsule"/>
          <body name="ffoot" pos=".13 0 -.18">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="ffoot" pos=".045 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .07" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco>
"""

comptime pm = parse_xml(_XML)
comptime fmd = parse_xml_full[
    pm.NBODY,
    pm.NJOINT,
    pm.NQ,
    pm.NV,
    pm.NGEOM,
    pm.NACT,
    pm.NTEX,
    pm.NMAT,
    pm.NLIGHT,
    pm.NCAM,
    pm.NSITE,
](_XML)


fn chk(label: String, got: Int, exp: Int, mut ok: Bool):
    if got == exp:
        print("  OK  ", label, "=", got)
    else:
        print("  FAIL", label, ": got", got, "expected", exp)
        ok = False


fn chkf(
    label: String, got: Float64, exp: Float64, mut ok: Bool, tol: Float64 = 1e-4
):
    var diff = got - exp
    if diff < 0:
        diff = -diff
    if diff <= tol:
        print("  OK  ", label, "=", got)
    else:
        print("  FAIL", label, ": got", got, "expected", exp, "diff", diff)
        ok = False


fn main() raises:
    print("=" * 60)
    print("HalfCheetah XML Parse Diagnostic")
    print("=" * 60)
    var ok = True

    # ── 1. Dimension counts ──────────────────────────────────────
    print()
    print("── 1. Dimension counts ──")
    chk("NBODY  ", pm.NBODY, 8, ok)
    chk("NJOINT ", pm.NJOINT, 10, ok)
    chk("NQ     ", pm.NQ, 10, ok)
    chk("NV     ", pm.NV, 10, ok)
    chk("NGEOM  ", pm.NGEOM, 9, ok)  # 1 floor plane + 8 capsules
    chk("NACT   ", pm.NACT, 6, ok)
    chk("NSITE  ", pm.NSITE, 0, ok)  # commented-out site
    chk("NCAM   ", pm.NCAM, 1, ok)
    chkf("TIMESTEP", pm.TIMESTEP, 0.01, ok)

    # ── 2. Body masses ───────────────────────────────────────────
    print()
    print("── 2. Body masses ──")
    var exp_masses: InlineArray[Float64, 8] = [
        0.0,  # body 0: worldbody
        6.250209,  # body 1: torso
        1.543515,  # body 2: bthigh
        1.587448,  # body 3: bshin
        1.095397,  # body 4: bfoot
        2.024566,  # body 5: fthigh
        1.383978,  # body 6: fshin
        1.115387,  # body 7: ffoot
    ]
    var bnames: InlineArray[String, 8] = [
        "worldbody",
        "torso",
        "bthigh",
        "bshin",
        "bfoot",
        "fthigh",
        "fshin",
        "ffoot",
    ]

    comptime for i in range(8):
        comptime bd = fmd.bodies[i]
        chkf(
            "body[" + String(i) + "] " + bnames[i] + " mass",
            bd.mass,
            exp_masses[i],
            ok,
            0.01,
        )

    # ── 3. Body positions ────────────────────────────────────────
    print()
    print("── 3. Body positions (pos_x, pos_z) ──")
    var exp_px: InlineArray[Float64, 8] = [
        0.0,
        0.0,
        -0.5,
        0.16,
        -0.28,
        0.5,
        -0.14,
        0.13,
    ]
    var exp_pz: InlineArray[Float64, 8] = [
        0.0,
        0.7,
        0.0,
        -0.25,
        -0.14,
        0.0,
        -0.24,
        -0.18,
    ]

    comptime for i in range(8):
        comptime bd = fmd.bodies[i]
        chkf(
            "body[" + String(i) + "] " + bnames[i] + " pos_x",
            bd.pos_x,
            exp_px[i],
            ok,
        )
        chkf(
            "body[" + String(i) + "] " + bnames[i] + " pos_z",
            bd.pos_z,
            exp_pz[i],
            ok,
        )

    # ── 4. Joint armature + ranges ───────────────────────────────
    print()
    print("── 4. Joint armature + ranges ──")
    var jnames: InlineArray[String, 9] = [
        "rootx",
        "rootz",
        "rooty",
        "bthigh",
        "bshin",
        "bfoot",
        "fthigh",
        "fshin",
        "ffoot",
    ]
    # Expected ranges for joints 3..8 (limited joints)
    var exp_rmin: InlineArray[Float64, 6] = [
        -0.52,
        -0.785,
        -0.4,
        -1.0,
        -1.2,
        -0.5,
    ]
    var exp_rmax: InlineArray[Float64, 6] = [1.05, 0.785, 0.785, 0.7, 0.87, 0.5]

    var joints = materialize[fmd.joints]()

    comptime for i in range(9):
        var jd = joints[i]
        var exp_arm = 0.0 if i < 3 else 0.1
        chkf(
            "joint[" + String(i) + "] " + jnames[i] + " armature",
            jd.armature,
            exp_arm,
            ok,
        )

        comptime if i >= 3 and i < 9:
            chkf(
                "joint[" + String(i) + "] " + jnames[i] + " range_min",
                jd.range_min,
                exp_rmin[i - 3],
                ok,
            )
            chkf(
                "joint[" + String(i) + "] " + jnames[i] + " range_max",
                jd.range_max,
                exp_rmax[i - 3],
                ok,
            )

    # ── 5. Actuator gears ────────────────────────────────────────
    print()
    print("── 5. Actuator gears ──")
    var exp_gears: InlineArray[Float64, 6] = [
        120.0,
        90.0,
        60.0,
        120.0,
        60.0,
        30.0,
    ]
    var anames: InlineArray[String, 6] = [
        "bthigh",
        "bshin",
        "bfoot",
        "fthigh",
        "fshin",
        "ffoot",
    ]

    var actuators = materialize[fmd.actuators]()

    comptime for i in range(6):
        var ad = actuators[i]
        chkf(
            "act[" + String(i) + "] " + anames[i] + " gear",
            ad.gear,
            exp_gears[i],
            ok,
            0.001,
        )

    # ── Summary ──────────────────────────────────────────────────
    print()
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see FAIL lines above")
    print("=" * 60)
