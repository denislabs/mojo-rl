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
* `RobotModel.__init__` (1.4.0 `models/robots/robot_model.py:70-75`), the
  THREE joint attributes it writes at load with `force=False` (only where
  the XML is silent):

      self.set_joint_attribute(attrib="frictionloss", values=0.1 * np.ones(self.dof), force=False)
      self.set_joint_attribute(attrib="damping", values=0.1 * np.ones(self.dof), force=False)
      self.set_joint_attribute(attrib="armature", values=np.array([5.0 / (i + 1) for i in range(self.dof)]), force=False)

  Damping is already in the arm XML `(0.1 x5, 0.01 x2)`, so that line is a
  no-op; `frictionloss=0.1` and `armature=5/(i+1)` are NOT in the XML and
  are applied here. ⚠⚠ L2 SHIPPED WITHOUT THEM, and every L2 gate was
  blind to it: the MuJoCo oracle loaded OUR XML, so both sides lacked the
  same two attributes and agreed to 1e-16. L4's demo replay — the file's
  own merged model beside ours — printed armature 5.0 vs 0.0 on joint 1
  and a 0.29 rad joint-space gap over 80 steps. The recorded data has
  them; a model without them is not the benchmark's.
* Where the robot STANDS is NOT here: `set_base_xpos` moves the root body,
  and the family's `base_pos=` does that (root pos = xpos - bottom_offset,
  bottom_offset = (mount.bottom_offset - mount.top_offset) = (0,0,-0.912)).
* TWO VARIANTS: `panda_robosuite.xml` is `MountedPanda` (tabletop, kitchen,
  study problems); `panda_robosuite_nomount.xml` is `OnTheGroundPanda`
  (living room, floor), the same arm and gripper with no mount and
  bottom_offset 0.

## What is deliberately dropped

* Nothing of the visual set any more — L5 needed it. The 50 per-material
  visual OBJ geoms of the arm are vendored as legacy MuJoCo `.msh`
  (`obj_to_msh` below), which both MuJoCo and this engine's `msh_loader` read:
  **46 MB of ASCII OBJ becomes 12 MB of binary**, and the alternative — a
  second asset pack — is bytes nobody can fetch until they are uploaded.
  ⚠ WITHOUT THEM THE ROBOT CANNOT BE DRAWN AT ALL. robosuite renders with
  `render_collision_mesh=False`, so a picture of LIBERO contains the arm's
  group-1 visual meshes and none of its collision geometry; a vendored Panda
  carrying only the collision STLs renders an arm-shaped hole. That was
  invisible to every L2-L4 gate because all of them are physics gates, and it
  is 99.9% of the squared error the L5 camera gate opened with
  (`tools/tasks/libero_camera_gate.py`: 26.5 dB, and the robot crop is 0.1%
  of nothing else).
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
OUT_XML = "noeira/envs/robots/assets/panda_robosuite.xml"
OUT_XML_NOMOUNT = "noeira/envs/robots/assets/panda_robosuite_nomount.xml"
OUT_DIR = "noeira/envs/robots/assets/panda_robosuite"

INIT_QPOS = "0 -0.161037389 0 -2.44459747 0 2.2267522 0.7853981633974483 0.020833 -0.020833"


def obj_to_msh(src, dst):
    """One robosuite visual OBJ -> legacy MuJoCo `.msh`.

        int32   nvertex nnormal ntexcoord nface
        float32 vertex[3n] normal[3n] texcoord[2n]
        int32   face[3f]                       0-based vertex indices

    ⚠ THE FORMAT CANNOT INDEX POSITIONS, NORMALS AND UVs SEPARATELY — one
    index per corner addresses all three. OBJ can, so a general converter has
    to de-index into unique (v, vt, vn) triples. These files do not need it:
    every face corner in all 50 is spelt `i/i/i`, so the three arrays are
    already parallel. That is ASSERTED rather than assumed — a file that broke
    it would otherwise be written with its normals and UVs permuted, which
    renders as a plausibly-lit wrong surface.
    """
    import struct

    v, vn, vt, f = [], [], [], []
    with open(src, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vn "):
                vn.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                vt.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                corners = line.split()[1:]
                if len(corners) != 3:
                    sys.exit(f"{src}: a face with {len(corners)} corners; "
                             "this converter writes triangles only")
                tri = []
                for c in corners:
                    parts = c.split("/")
                    ids = [int(x) for x in parts if x != ""]
                    if len(set(ids)) != 1:
                        sys.exit(f"{src}: face corner {c!r} indexes position, "
                                 "texcoord and normal differently — see obj_to_msh")
                    tri.append(ids[0] - 1)
                f.append(tri)
    n = len(v)
    if len(vn) not in (0, n) or len(vt) not in (0, n):
        sys.exit(f"{src}: {n} vertices but {len(vn)} normals and {len(vt)} texcoords")
    out = bytearray()
    out += struct.pack("<4i", n, len(vn), len(vt), len(f))
    for a in v:
        out += struct.pack("<3f", *a)
    for a in vn:
        out += struct.pack("<3f", *a)
    for a in vt:
        out += struct.pack("<2f", *a)
    for t in f:
        out += struct.pack("<3i", *t)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(bytes(out))
    return len(v), len(f)


def repath_visual_obj(arm):
    """Point every `.obj` visual mesh at the `.msh` this tool writes.

    Returns `[(src_obj, dst_msh)]` for the conversion pass. The geoms and the
    twelve `<material>`s are left exactly as robosuite authored them.
    """
    asset = arm.find("asset")
    converts = []
    for m in list(asset.findall("mesh")):
        f = m.get("file", "")
        if not f.endswith(".obj"):
            continue
        src = os.path.join(RS, "robots/panda", f)
        dst = os.path.join(OUT_DIR, "visual", os.path.basename(f)[:-4] + ".msh")
        converts.append((src, dst))
        m.set("file", os.path.join("visual", os.path.basename(f)[:-4] + ".msh"))
    return converts


def apply_robot_model_defaults(arm):
    """`RobotModel.__init__`'s `set_joint_attribute(..., force=False)` trio on
    the seven arm joints, in joint order — see the header."""
    joints = [j for j in arm.find("worldbody").iter("joint")]
    if len(joints) != 7:
        sys.exit(f"expected 7 arm joints, found {len(joints)}")
    for i, j in enumerate(joints):
        if j.get("frictionloss") is None:
            j.set("frictionloss", "0.1")
        if j.get("damping") is None:
            j.set("damping", "0.1")
        if j.get("armature") is None:
            j.set("armature", repr(5.0 / (i + 1)))


def build(mounted):
    """The flat MJCF text and the (src, dst) mesh copies, for one variant."""
    arm = ET.parse(os.path.join(RS, "robots/panda/robot.xml")).getroot()
    grip = ET.parse(os.path.join(RS, "grippers/panda_gripper.xml")).getroot()
    mount = ET.parse(os.path.join(RS, "bases/rethink_mount.xml")).getroot()

    converts = repath_visual_obj(arm)
    apply_robot_model_defaults(arm)

    root = ET.Element("mujoco", model="panda_robosuite" if mounted else "panda_robosuite_nomount")
    ET.SubElement(root, "compiler", angle="radian", inertiagrouprange="0 0", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", impratio="20", cone="elliptic",
                  density="1.2", viscosity="0.00002")

    # ── assets: every mesh, re-pathed into OUT_DIR ──────────────────────────
    #
    # ⚠ THE ARM'S TWELVE `<material>`s COME TOO, and they are the robot's
    # colour: `Shell_001` is 0.25 grey and `Face636_001` is 0.90 white, which
    # is the dark-joint / white-shell Panda everyone recognises. A geom that
    # kept its `material=` and lost the material itself would fall back to the
    # 0.5 default and render a uniformly grey arm.
    asset = ET.SubElement(root, "asset")
    for mat in arm.find("asset").findall("material"):
        asset.append(mat)
    copies = []  # (src, dst) — a straight file copy
    sources = [(arm, "robots/panda"), (grip, "grippers")]
    if mounted:
        sources.append((mount, "bases"))
    for src_root, sub in sources:
        for m in src_root.find("asset").findall("mesh"):
            f = m.get("file")
            if f.startswith("visual/"):
                # written by `obj_to_msh`, not copied — see `repath_visual_obj`
                ET.SubElement(asset, "mesh", name=m.get("name"),
                              file=os.path.join(os.path.basename(OUT_DIR), f))
                continue
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
    return text, copies, converts


def verify(text, copies, converts, name):
    """MuJoCo must load it, with the dims robosuite's merge produces."""
    import mujoco
    tmpdir = "build/panda_vendor_check"
    os.makedirs(os.path.join(tmpdir, os.path.basename(OUT_DIR)), exist_ok=True)
    for src, dst in copies:
        shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(OUT_DIR), os.path.basename(dst)))
    for src, dst in converts:
        obj_to_msh(src, os.path.join(tmpdir, os.path.basename(OUT_DIR), "visual",
                                     os.path.basename(dst)))
    with open(os.path.join(tmpdir, name), "w") as f:
        f.write(text)
    m = mujoco.MjModel.from_xml_path(os.path.join(tmpdir, name))
    assert m.nq == 9 and m.nv == 9 and m.nu == 9, (m.nq, m.nv, m.nu)
    assert m.njnt == 9 and m.nsensor == 0
    # ⚠ THE VISUAL SET IS COUNTED, not assumed present. A `.msh` that failed to
    # convert would leave a mesh MuJoCo cannot find (a load error) or a geom
    # nothing draws (silent) — the second is the one this catches.
    nvis = sum(1 for i in range(m.ngeom) if m.geom_group[i] == 1)
    if nvis < 50:
        sys.exit(f"{name}: only {nvis} geoms in the visual group; the arm's 50 "
                 "per-material visual meshes are the point of the L5 vendoring")
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
    all_converts = {}
    texts = {}
    for out, mounted in outputs:
        text, copies, converts = build(mounted)
        verify(text, copies, converts, os.path.basename(out))
        texts[out] = text
        for src, dst in copies:
            all_copies[dst] = src
        for src, dst in converts:
            all_converts[dst] = src
    if a.check:
        for out, _ in outputs:
            old = open(out).read() if os.path.exists(out) else ""
            if old != texts[out]:
                sys.exit(f"STALE: {out} differs from what the tool generates")
        for dst in list(all_copies) + list(all_converts):
            if not os.path.exists(dst):
                sys.exit(f"MISSING: {dst}")
        print("up to date")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    for dst, src in all_copies.items():
        shutil.copyfile(src, dst)
    nv = nf = 0
    for dst, src in all_converts.items():
        a_, b_ = obj_to_msh(src, dst)
        nv += a_
        nf += b_
    for out, _ in outputs:
        with open(out, "w") as f:
            f.write(texts[out])
    total = sum(os.path.getsize(d_) for d_ in all_copies)
    vis = sum(os.path.getsize(d_) for d_ in all_converts)
    print(f"wrote {OUT_XML} and {OUT_XML_NOMOUNT} + {len(all_copies)} meshes "
          f"({total/1e6:.1f} MB) in {OUT_DIR}")
    print(f"  + {len(all_converts)} visual .msh ({vis/1e6:.1f} MB, {nv} verts, "
          f"{nf} tris) converted from robosuite's OBJ")


if __name__ == "__main__":
    main()
