#!/usr/bin/env python3
"""Vendor robosuite's Panda + PandaGripper + RethinkMount as ONE flat MJCF.

    pixi run python tools/robots/vendor_panda_robosuite.py
    pixi run python tools/robots/vendor_panda_robosuite.py --check

L2 of docs/LIBERO_PORT_ASSESSMENT_2026_09_13.md. LIBERO's robot is robosuite's
`MountedPanda`: the arm XML with the gripper merged under `right_hand` and
the mount merged under the root body. robosuite does that merge in Python at
every env construction; this script does it ONCE, from the same three source
XMLs, and writes what MuJoCo would have seen — minus what LIBERO never uses.

## What is reproduced, quoted from robosuite

* `ManipulatorModel.add_gripper`: `merge(gripper, merge_body="right_hand")`
  — the gripper's root body `right_gripper` (pos 0 0 0, quat 0.707107 0 0
  -0.707107, as authored) becomes a child of `right_hand`.
* `RobotModel.add_mount`: the mount's root body pos is set to
  `base_offset - mount.top_offset = (0,0,0) - (0,0,-0.01) = (0,0,0.01)` and
  merged under the robot's root `base`. ⚠ Both roots are named `base`; the
  mount's is renamed `mount` here (robosuite would prefix it `mount0_`).
* Joint damping `(0.1 x5, 0.01 x2)` — already in the arm XML, and
  `MountedPanda.set_joint_attribute` writes the same values.
* Where the robot STANDS is NOT here: `set_base_xpos` moves the root body,
  and the family's `base_pos=` does that (root pos = xpos - bottom_offset,
  bottom_offset = (mount.bottom_offset - mount.top_offset) = (0,0,-0.912)).
* TWO VARIANTS: `panda_robosuite.xml` is `MountedPanda` (tabletop, kitchen,
  study problems); `panda_robosuite_nomount.xml` is `OnTheGroundPanda`
  (living room, floor), the same arm and gripper with no mount and
  bottom_offset 0.

## What is deliberately dropped

* The 50 per-material visual OBJ geoms of the arm (36 MB). The collision
  STLs (728 KB) render in their place until textured cameras (L5) need the
  visual set, which then becomes a pack. The gripper's small `*_vis.stl` and
  the mount's pedestal STL are kept.
* The gripper's `<sensor>` force/torque pair on `ft_frame`. LIBERO's policies
  and datasets never read them, and an unserved sensor kind is a load-time
  refusal in this engine (`PHYSICS3D_MUJOCO_312_AUDIT.md` AUD-23).
* robosuite's `robot0_` / `gripper0_` / `mount0_` name prefixes. The family
  composer prefixes the whole asset once (`robot_`); a second prefix would
  give `robot_robot0_joint1`.
* No keyframe: `<attach>` does not carry one, and the init pose is data
  (`init_table`). LIBERO's `init_qpos` is recorded in the header comment.

## Compiler / option

robosuite's `base.xml` sets `inertiagrouprange="0 0" autolimits="true"`,
`impratio="20" cone="elliptic" density="1.2" viscosity="0.00002"` and
`timestep=0.002` (`macros.SIMULATION_TIMESTEP`). They are written here so the
family composer can inherit them (`inherit_option=1`).
"""
import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

RS = "references/robosuite-master/robosuite/models/assets"
OUT_XML = "mojo_rl/envs/robots/assets/panda_robosuite.xml"
OUT_XML_NOMOUNT = "mojo_rl/envs/robots/assets/panda_robosuite_nomount.xml"
OUT_DIR = "mojo_rl/envs/robots/assets/panda_robosuite"

INIT_QPOS = "0 -0.161037389 0 -2.44459747 0 2.2267522 0.7853981633974483 0.020833 -0.020833"


def strip_visual_obj(arm):
    """Drop the OBJ visual geoms, their mesh decls and their materials."""
    asset = arm.find("asset")
    keep_mesh = set()
    for m in list(asset.findall("mesh")):
        if m.get("file", "").endswith(".obj"):
            asset.remove(m)
        else:
            keep_mesh.add(m.get("name"))
    for mat in list(asset.findall("material")):
        asset.remove(mat)
    for body in arm.iter("body"):
        for g in list(body.findall("geom")):
            if g.get("mesh") and g.get("mesh") not in keep_mesh:
                body.remove(g)
            elif g.get("material"):
                del g.attrib["material"]
    return keep_mesh


def build(mounted):
    """The flat MJCF text and the (src, dst) mesh copies, for one variant."""
    arm = ET.parse(os.path.join(RS, "robots/panda/robot.xml")).getroot()
    grip = ET.parse(os.path.join(RS, "grippers/panda_gripper.xml")).getroot()
    mount = ET.parse(os.path.join(RS, "bases/rethink_mount.xml")).getroot()

    strip_visual_obj(arm)

    root = ET.Element("mujoco", model="panda_robosuite" if mounted else "panda_robosuite_nomount")
    ET.SubElement(root, "compiler", angle="radian", inertiagrouprange="0 0", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", impratio="20", cone="elliptic",
                  density="1.2", viscosity="0.00002")

    # ── assets: every mesh, re-pathed into OUT_DIR ──────────────────────────
    asset = ET.SubElement(root, "asset")
    copies = []  # (src, dst)
    sources = [(arm, "robots/panda"), (grip, "grippers")]
    if mounted:
        sources.append((mount, "bases"))
    for src_root, sub in sources:
        for m in src_root.find("asset").findall("mesh"):
            f = m.get("file")
            src = os.path.join(RS, sub, f)
            dst_name = os.path.basename(f)
            copies.append((src, os.path.join(OUT_DIR, dst_name)))
            ET.SubElement(asset, "mesh", name=m.get("name"),
                          file=os.path.join(os.path.basename(OUT_DIR), dst_name))

    # ── worldbody: base > (mount, link0 > ... > right_hand > right_gripper) ─
    wb = ET.SubElement(root, "worldbody")
    base = arm.find("worldbody/body[@name='base']")
    wb.append(base)
    if mounted:
        mount_root = mount.find("worldbody/body[@name='base']")
        mount_root.set("name", "mount")
        mount_root.set("pos", "0 0 0.01")          # base_offset - top_offset
        base.insert(0, mount_root)
    right_hand = None
    for b in base.iter("body"):
        if b.get("name") == "right_hand":
            right_hand = b
    if right_hand is None:
        sys.exit("no right_hand body in the arm XML")
    right_hand.append(grip.find("worldbody/body[@name='right_gripper']"))

    # ── actuators: arm motors then gripper position servos ─────────────────
    act = ET.SubElement(root, "actuator")
    for src_root in (arm, grip):
        for el in src_root.find("actuator"):
            act.append(el)
    # ⚠ NO <sensor>: see the header.

    ET.indent(root, space="  ")
    who = "MountedPanda (RethinkMount)" if mounted else "OnTheGroundPanda (no mount)"
    text = (
        "<!-- GENERATED by tools/robots/vendor_panda_robosuite.py from\n"
        "     references/robosuite-master (MIT). Do not edit; re-run the tool.\n"
        f"     LIBERO {who}; init_qpos (7 arm + 2 finger): {INIT_QPOS} -->\n"
        + ET.tostring(root, encoding="unicode") + "\n"
    )
    return text, copies


def verify(text, copies, name):
    """MuJoCo must load it, with the dims robosuite's merge produces."""
    import mujoco
    tmpdir = "build/panda_vendor_check"
    os.makedirs(os.path.join(tmpdir, os.path.basename(OUT_DIR)), exist_ok=True)
    for src, dst in copies:
        shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(OUT_DIR), os.path.basename(dst)))
    with open(os.path.join(tmpdir, name), "w") as f:
        f.write(text)
    m = mujoco.MjModel.from_xml_path(os.path.join(tmpdir, name))
    assert m.nq == 9 and m.nv == 9 and m.nu == 9, (m.nq, m.nv, m.nu)
    assert m.njnt == 9 and m.nsensor == 0
    d = mujoco.MjData(m)
    d.qpos[:] = [float(x) for x in INIT_QPOS.split()]
    mujoco.mj_forward(m, d)
    site = m.site("grip_site").id
    print(f"{name}: MuJoCo {mujoco.__version__} nbody {m.nbody} njnt {m.njnt} nq {m.nq} "
          f"nv {m.nv} nu {m.nu} ngeom {m.ngeom} ncam {m.ncam} nsite {m.nsite}; "
          f"grip_site at init_qpos, base at origin: {d.site_xpos[site].round(4)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    outputs = [(OUT_XML, True), (OUT_XML_NOMOUNT, False)]
    all_copies = {}
    texts = {}
    for out, mounted in outputs:
        text, copies = build(mounted)
        verify(text, copies, os.path.basename(out))
        texts[out] = text
        for src, dst in copies:
            all_copies[dst] = src
    if a.check:
        for out, _ in outputs:
            old = open(out).read() if os.path.exists(out) else ""
            if old != texts[out]:
                sys.exit(f"STALE: {out} differs from what the tool generates")
        for dst in all_copies:
            if not os.path.exists(dst):
                sys.exit(f"MISSING: {dst}")
        print("up to date")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    for dst, src in all_copies.items():
        shutil.copyfile(src, dst)
    for out, _ in outputs:
        with open(out, "w") as f:
            f.write(texts[out])
    total = sum(os.path.getsize(d_) for d_ in all_copies)
    print(f"wrote {OUT_XML} and {OUT_XML_NOMOUNT} + {len(all_copies)} meshes "
          f"({total/1e6:.1f} MB) in {OUT_DIR}")


if __name__ == "__main__":
    main()
