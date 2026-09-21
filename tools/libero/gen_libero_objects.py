#!/usr/bin/env python3
"""Generate the attachable copy of every LIBERO object XML.

    pixi run python tools/tasks/gen_libero_objects.py          # write
    pixi run python tools/tasks/gen_libero_objects.py --check  # CI: stale?

L2 of docs/LIBERO_PORT_ASSESSMENT_2026_09_13.md. The pack's object XMLs are
byte-identical to upstream and upstream never loads them directly: robosuite's
`MujocoXML.merge_assets` appends each `<asset>` child ONLY IF no asset of the
same tag and name exists yet. `flat_stove.xml` declares `tex-stove_knob`
twice and relies on that rule; MuJoCo's `<attach>` applies no such rule and
refuses the composed scene ("repeated name 'tex-stove_knob' in texture").

This tool applies exactly that one rule — first declaration wins, by (tag,
name) — re-roots every `file=` at `../assets/`, and writes one XML per
category of `noeira/tasks/libero/categories.kv` into
`noeira/tasks/libero/objects/`. Each output is loaded by MuJoCo on its own
before it is written. The family composer attaches these, not the pack's.

⚠ WHAT ELSE robosuite DOES AND THIS DOES NOT: rename `object` to
`{name}_main` (our slot prefix is the identity), filter geoms by group
(LIBERO uses `obj_type="all"`, both groups kept), name unnamed geoms
`g{i}`, and append a `default_site` at the body origin. None of it changes
the physics; the first three are naming.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

TABLE = "noeira/tasks/libero/categories.kv"
PACK = "noeira/tasks/libero/assets"
OUT_DIR = "noeira/tasks/libero/objects"


def read_assets(path):
    """category -> pack-relative asset path, for kind=asset records."""
    out, cur, kind, asset = {}, None, None, None
    def flush():
        if cur and kind == "asset" and asset:
            out[cur] = asset
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = (x.strip() for x in line.split("=", 1))
        if k in ("category", "problem"):
            flush()
            cur, kind, asset = (v if k == "category" else None), None, None
        elif k == "kind":
            kind = v
        elif k == "asset":
            asset = v
    flush()
    return out


def generate(rel, free):
    """`free=True` injects robosuite's movable-object joint:
    `MujocoXMLObject(joints=[dict(type="free", damping="0.0005")])` becomes
    `<joint type="free" damping="0.0005" name="joint0"/>` as the FIRST child
    of the `object` body. A fixture (`joints=None`) gets none — that is the
    `.xml` file; the movable variant is `<category>_free.xml`, and a `.bddl`
    decides which one a slot is by listing the instance under `:fixtures`
    or `:objects`."""
    src = os.path.join(PACK, rel)
    root = ET.parse(src).getroot()
    # ⚠⚠ THE WRAPPER IS DROPPED AND THE AUTHORED POSE WITH IT. robosuite's
    # `MujocoXMLObject._get_object_subtree` takes `./body/body[@name='object']`
    # OUT of its unnamed wrapper and appends it to the worldbody as a ROOT
    # body (a free joint is only legal on a root). The wrapper's three sites
    # (`bottom_site`, `top_site`, `horizontal_radius_site`) are read for
    # placement and never enter the sim. And the object body's own
    # `pos="0 -0.1 0"` (the bowl's, for one) is overwritten in BOTH of
    # LIBERO's paths — `set_joint_qpos` for a movable, `model.body_pos` for
    # a fixture — so it is removed here; keeping it would offset every
    # fixture by its authored pose under our frame placement.
    wb = root.find("worldbody")
    wrapper = wb.find("./body")
    obj = wb.find("./body/body[@name='object']")
    if wrapper is None or obj is None:
        sys.exit(f"{rel}: no <body><body name='object'> wrapper — not a robosuite object")
    obj.attrib.pop("pos", None)
    obj.attrib.pop("quat", None)
    wb.remove(wrapper)
    wb.append(obj)
    if free:
        # robosuite appends the joint after the geoms; order is not semantic
        obj.append(ET.Element("joint", type="free", damping="0.0005", name="joint0"))
    obj_dir = os.path.dirname(rel)
    for el in root.iter():
        f = el.get("file")
        if f:
            el.set("file", os.path.join("..", "assets", os.path.normpath(os.path.join(obj_dir, f))))
    # ⚠ THE OBJECT'S OWN <compiler>/<option>/<size>/<visual>/<statistic> ARE
    # DROPPED, because robosuite's merge never imports them — it takes the
    # worldbody, the assets and the actuator/sensor/tendon/equality/contact
    # blocks, and the OBJECT is compiled under robosuite's base.xml settings.
    # `microwave.xml` carries `<compiler inertiagrouprange="4 4">`, under
    # which every one of its bodies has mass 0 and MuJoCo refuses the door;
    # under base.xml's `"0 0"` it is a 0.56 kg door. MuJoCo's <attach>
    # honours a child's compiler, our expander splices text under the
    # parent's — dropping the tag is what makes the oracle and the engine
    # see the same object, and it is what LIBERO actually ran.
    # ⚠⚠ EVERY GENERATED CHILD CARRIES robosuite's COMPILER LINE. MuJoCo's
    # <attach> compiles a child under the CHILD's own <compiler>, defaults
    # included — so a child without one gets `inertiagrouprange="0 5"` and the
    # arena table weighs 126.9 kg (visual box + legs) while LIBERO's single
    # merged document, compiled under base.xml's "0 0", weighs it at 60.0 kg.
    # Our expander splices text under the host's compiler and got 60.0; the
    # oracle got 126.9. Stating the line in every child makes MuJoCo, our
    # engine and robosuite agree, and `bcmp.py` on the composed scene is the
    # gate that found it (|d mass| 6.7e+01 at arena_table, 2026-09-13).
    dropped_top = []
    for child in list(root):
        if child.tag in ("compiler", "option", "size", "visual", "statistic"):
            root.remove(child)
            dropped_top.append(child.tag)
    # (inserted AFTER the drop above, which would otherwise remove it)
    root.insert(0, ET.Element("compiler", angle="radian", inertiagrouprange="0 0", autolimits="true"))
    asset = root.find("asset")
    dropped = []
    if asset is not None:
        seen = set()
        for child in list(asset):
            key = (child.tag, child.get("name"))
            if key in seen:
                asset.remove(child)
                dropped.append(key)
            else:
                seen.add(key)
    ET.indent(root, space="  ")
    text = (
        "<!-- GENERATED by tools/tasks/gen_libero_objects.py from\n"
        f"     {rel} in the libero asset pack. Do not edit; re-run the tool.\n"
        f"     {'movable: free joint injected (robosuite joints=[free, damping 0.0005])' if free else 'fixture: no joint (robosuite joints=None)'}.\n"
        f"     Duplicate asset declarations dropped (robosuite merge_assets rule): {len(dropped)};\n"
        f"     top-level tags dropped (robosuite merge imports none of them): {dropped_top} -->\n"
        + ET.tostring(root, encoding="unicode") + "\n"
    )
    return text, dropped + dropped_top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(PACK):
        sys.exit(f"no LIBERO pack at {PACK} — run `pixi run assets-pull libero`")
    import mujoco
    cats = read_assets(TABLE)
    if not cats:
        sys.exit("no asset categories in the table")
    os.makedirs(OUT_DIR, exist_ok=True)
    stale, n_dropped = 0, 0
    for cat, free in [(c, fr) for c in sorted(cats) for fr in (False, True)]:
        text, dropped = generate(cats[cat], free)
        n_dropped += len(dropped)
        out = os.path.join(OUT_DIR, cat + ("_free" if free else "") + ".xml")
        # ⚠ CHECKED THE WAY IT WILL BE USED: attached under robosuite's
        # compiler settings, not loaded bare. `inertiagrouprange="0 0"`
        # changes which geoms carry mass, and a bare load would judge a
        # different model from the one the family composes.
        tmp = out + ".check.xml"
        wrap = out + ".check_scene.xml"
        with open(tmp, "w") as f:
            f.write(text)
        with open(wrap, "w") as f:
            f.write(
                '<mujoco model="check"><compiler angle="radian" inertiagrouprange="0 0"'
                ' autolimits="true"/><asset><model name="obj" file="'
                + os.path.basename(tmp) + '"/></asset><worldbody>'
                '<attach model="obj" prefix="obj_"/></worldbody></mujoco>'
            )
        try:
            m = mujoco.MjModel.from_xml_path(wrap)
        finally:
            os.remove(tmp)
            os.remove(wrap)
        info = f"nbody {m.nbody} ngeom {m.ngeom} njnt {m.njnt} nmesh {m.nmesh} ntex {m.ntex}"
        if dropped:
            info += f" dropped {dropped}"
        old = open(out).read() if os.path.exists(out) else None
        if a.check:
            if old != text:
                print(f"  STALE      {out}")
                stale += 1
        elif old != text:
            with open(out, "w") as f:
                f.write(text)
            print(f"  wrote      {out}  ({info})")
        else:
            print(f"  unchanged  {out}  ({info})")
    print(f"{len(cats)} objects x 2 variants, {n_dropped} dropped declarations/tags")
    if a.check and stale:
        sys.exit(f"{stale} object XML(s) stale; run `pixi run python tools/tasks/gen_libero_objects.py`")


if __name__ == "__main__":
    main()
