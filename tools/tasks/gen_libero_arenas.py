#!/usr/bin/env python3
"""Generate the styled, re-posed LIBERO arena per problem class.

    pixi run python tools/tasks/gen_libero_arenas.py          # write
    pixi run python tools/tasks/gen_libero_arenas.py --check  # CI: stale?

L2 of docs/LIBERO_PORT_ASSESSMENT_2026_09_13.md. LIBERO never loads a scene
XML as shipped: its arena classes (`envs/arenas/*.py`) rewrite it at
construction — floor/wall textures from the style tables, the table body
re-posed from `table_full_size` / `table_offset` (TableArena and
KitchenTableArena only), and each problem class then moves `agentview` and
`frontview` in `_setup_camera`. This script performs exactly those edits,
from the numbers in `mojo_rl/tasks/libero/categories.kv`, and writes one
arena per problem into `mojo_rl/tasks/libero/scenes/`, which the family
composer attaches as a static slot at the origin.

Quoted from `TableArena.configure_location`:

    center_pos      = bottom_pos + (0, 0, -h/2) + table_offset
    table body pos  = center_pos
    table_collision size = half size, friction = table_friction
    table_visual size    = half size
    table_top site pos   = (0, 0, h/2)
    legs: x = sign(dx)*hx - dx (if hx > 2|dx|), same for y,
          z = (table_offset_z - h/2)/2, pos (x, y, -z), size (0.025, z)
    with dx in [0.1,-0.1,-0.1,0.1], dy in [0.1,0.1,-0.1,-0.1]

`bottom_pos` is the floor geom's pos, (0,0,0) in every LIBERO arena.

Two additions of our own, both flagged in the output:
* a `<site name="zone_plane">` at the problem's `zone_z` (L3) — the anchor of
  every TABLE box region, at the height LIBERO's `TargetZone` sites sit;
* a `<site name="workspace">` at `workspace_offset` — the anchor every
  imported region rect is measured from (LIBERO measures its rects from
  the same offset; the site just gives it a name);
* every `file=` is re-rooted at `../assets/` so the generated XML lives
  beside the pack rather than inside it.

⚠ ElementTree re-serialises the document, so comments and formatting are
not preserved. That is fine for a GENERATED file; it would not be for a
hand-authored one.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

TABLE = "mojo_rl/tasks/libero/categories.kv"
PACK = "mojo_rl/tasks/libero/assets"
OUT_DIR = "mojo_rl/tasks/libero/scenes"

# problems whose LIBERO arena class calls configure_location
REPOSED_WORKSPACES = {"table", "kitchen_table"}


def read_problems(path):
    probs, cur = [], None
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "problem":
            cur = {"name": v, "camera": []}
            probs.append(cur)
        elif k == "category":
            cur = None
        elif cur is not None:
            if k == "camera":
                cur["camera"].append(v)
            else:
                cur[k] = v
    return probs


def f3(s):
    return [float(x) for x in s.split(",")]


def fmt(v):
    return " ".join(repr(float(x)) if abs(float(x)) >= 1e-4 or x == 0 else repr(float(x)) for x in v)


def arena_path(problem_name):
    return os.path.join(OUT_DIR, problem_name.lower() + "_arena.xml")


def generate(prob):
    scene_rel = prob["scene"]
    src = os.path.join(PACK, scene_rel)
    tree = ET.parse(src)
    root = tree.getroot()
    scene_dir = os.path.dirname(scene_rel)          # "scenes"

    # ── file= re-rooting, then the style textures ────────────────────────
    for el in root.iter():
        f = el.get("file")
        if f:
            # relative to the scene's own dir inside the pack -> relative to OUT_DIR
            rel_in_pack = os.path.normpath(os.path.join(scene_dir, f))
            el.set("file", os.path.join("..", "assets", rel_in_pack))
    # ⚠⚠ EVERY GENERATED CHILD CARRIES robosuite's COMPILER LINE. MuJoCo's
    # <attach> compiles a child under the CHILD's own <compiler>, defaults
    # included — so a child without one gets `inertiagrouprange="0 5"` and the
    # arena table weighs 126.9 kg (visual box + legs) while LIBERO's single
    # merged document, compiled under base.xml's "0 0", weighs it at 60.0 kg.
    # Our expander splices text under the host's compiler and got 60.0; the
    # oracle got 126.9. Stating the line in every child makes MuJoCo, our
    # engine and robosuite agree, and `bcmp.py` on the composed scene is the
    # gate that found it (|d mass| 6.7e+01 at arena_table, 2026-09-13).
    root.insert(0, ET.Element("compiler", angle="radian", inertiagrouprange="0 0", autolimits="true"))
    asset = root.find("asset")
    texplane = asset.find("./texture[@name='texplane']")
    texwall = asset.find("./texture[@name='tex-wall']")
    if texplane is None or texwall is None:
        sys.exit(f"{scene_rel}: no texplane / tex-wall texture — not a LIBERO arena")
    texplane.set("file", os.path.join("..", "assets", prob["floor_texture"]))
    texwall.set("file", os.path.join("..", "assets", prob["wall_texture"]))

    wb = root.find("worldbody")
    off = f3(prob["workspace_offset"])

    # ── the table re-pose, where LIBERO does it ──────────────────────────
    if prob["workspace"] in REPOSED_WORKSPACES:
        full = f3(prob["table_size"])
        half = [x / 2 for x in full]
        fric = f3(prob["table_friction"])
        floor = wb.find("./geom[@name='floor']")
        bottom = f3(floor.get("pos").replace(" ", ",")) if floor is not None else [0, 0, 0]
        center = [bottom[0] + off[0], bottom[1] + off[1], bottom[2] - half[2] + off[2]]
        tb = wb.find("./body[@name='table']")
        if tb is None:
            sys.exit(f"{scene_rel}: no body 'table' to re-pose")
        tb.set("pos", fmt(center))
        col = tb.find("./geom[@name='table_collision']")
        vis = tb.find("./geom[@name='table_visual']")
        top = tb.find("./site[@name='table_top']")
        col.set("size", fmt(half))
        col.set("friction", fmt(fric))
        vis.set("size", fmt(half))
        top.set("pos", fmt([0, 0, half[2]]))
        dxs, dys = [0.1, -0.1, -0.1, 0.1], [0.1, 0.1, -0.1, -0.1]
        for i, (dx, dy) in enumerate(zip(dxs, dys), 1):
            leg = tb.find(f"./geom[@name='table_leg{i}_visual']")
            x = 0.0
            if half[0] > abs(dx * 2.0):
                x += (1 if dx > 0 else -1) * half[0] - dx
            y = 0.0
            if half[1] > abs(dy * 2.0):
                y += (1 if dy > 0 else -1) * half[1] - dy
            z = (off[2] - half[2]) / 2.0
            leg.set("pos", fmt([x, y, -z]))
            leg.set("size", fmt([0.025, z]))

    # ── cameras: the problem's _setup_camera overrides ───────────────────
    for spec in prob["camera"]:
        name, pos, quat = spec.split(":")
        cam = wb.find(f"./camera[@name='{name}']")
        if cam is None:
            cam = ET.SubElement(wb, "camera", mode="fixed", name=name)
        cam.set("pos", fmt(f3(pos)))
        cam.set("quat", fmt(f3(quat)))

    # ── the workspace anchor site ────────────────────────────────────────
    ET.SubElement(wb, "site", name="workspace", pos=fmt(off), size="0.001",
                  rgba="0 0 0 0")
    # ── the target-zone plane (L3) ───────────────────────────────────────
    # LIBERO gives every table region its own `<site type="box">` at the
    # zone's centroid, appended to the workspace body under a per-problem
    # z convention (quoted in categories.kv beside `zone_z=`). Our table
    # regions share ONE anchor and carry the centroid as their rect, so
    # the anchor must sit at the zone plane's WORLD height: `On(obj, zone)`
    # is `under`, a band measured from that site.
    ET.SubElement(wb, "site", name="zone_plane",
                  pos=fmt([off[0], off[1], float(prob["zone_z"])]),
                  size="0.001", rgba="0 0 0 0")

    ET.indent(root, space="  ")
    return (
        "<!-- GENERATED by tools/tasks/gen_libero_arenas.py from\n"
        f"     {scene_rel} in the libero asset pack + {TABLE}\n"
        f"     ({prob['name']}). Do not edit; re-run the tool. -->\n"
        + ET.tostring(root, encoding="unicode") + "\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(PACK):
        sys.exit(f"no LIBERO pack at {PACK} — run `pixi run assets-pull libero`")
    import mujoco
    probs = read_problems(TABLE)
    if not probs:
        sys.exit("no problem= records in the table")
    os.makedirs(OUT_DIR, exist_ok=True)
    stale = 0
    for prob in probs:
        text = generate(prob)
        out = arena_path(prob["name"])
        # ⚠ MuJoCo must load the result; write to a sibling temp name so the
        # relative asset paths resolve exactly as they will from OUT_DIR.
        tmp = out + ".check.xml"
        with open(tmp, "w") as f:
            f.write(text)
        try:
            m = mujoco.MjModel.from_xml_path(tmp)
        finally:
            os.remove(tmp)
        ws = m.site("workspace").id
        zp = m.site("zone_plane").id
        info = (f"nbody {m.nbody} ngeom {m.ngeom} ncam {m.ncam} nlight {m.nlight}"
                f" workspace {m.site_pos[ws].round(3).tolist()}"
                f" zone_plane z {float(m.site_pos[zp][2]):.3f}")
        old = open(out).read() if os.path.exists(out) else None
        if a.check:
            if old != text:
                print(f"  STALE     {out}")
                stale += 1
            else:
                print(f"  up to date {out}  ({info})")
        elif old != text:
            with open(out, "w") as f:
                f.write(text)
            print(f"  wrote     {out}  ({info})")
        else:
            print(f"  unchanged {out}  ({info})")
    if a.check and stale:
        sys.exit(f"{stale} arena(s) stale; run `pixi run python tools/tasks/gen_libero_arenas.py`")


if __name__ == "__main__":
    main()
