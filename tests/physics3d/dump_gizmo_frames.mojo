"""Dump where the gizmo would sit, and make one gizmo edit — for MuJoCo.

    pixi run mojo run -I . tests/physics3d/dump_gizmo_frames.mojo \\
        <model.xml> <out.xml>

⚠⚠ THIS ASSERTS NOTHING. It is the Mojo half of
`scripts/check_gizmo_vs_mujoco.py`: it prints the frames the studio would
draw the gizmo on, performs one move and one turn through the studio's OWN
path (`gizmo_edits` -> `apply_edit` -> `apply_edit_to_document`), and writes
the resulting document. MuJoCo is what says whether any of it is right.

⚠ THE OUTPUT FILE GOES BESIDE THE MODEL, not into /tmp. Asset paths in MJCF
are relative to the document, so a file written elsewhere cannot be loaded by
MuJoCo without moving every mesh with it.

The target element is chosen deterministically and PRINTED, so a run that
picked something degenerate is visible rather than being averaged away.
"""

from std.sys import argv

from mojo_rl.math3d import Vec3 as Vec3G, Quat as QuatG
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.flat_model import FlatModelDef
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.kinematics.mocap import reset_mocap_from_model
from mojo_rl.physics3d.studio.gizmo import (
    Frame, frame_to_cm, edit_frame, gizmo_edits, GIZMO_MOVE, GIZMO_TURN,
)
from mojo_rl.physics3d.studio.edit import (
    Edit, apply_edit, apply_edit_to_document, TARGET_GEOM, TARGET_BODY,
)

comptime DT = DType.float64
comptime Vec3 = Vec3G[DT]
comptime Quat = QuatG[DT]

comptime GEOM_PLANE: Int = 0
comptime GEOM_MESH: Int = 5

# ⚠ AN AWKWARD DELTA ON PURPOSE. Round numbers and axis-aligned rotations let
# a sign error or a transposed frame agree by accident; 0.037/-0.021/0.013 and
# a rotation about an oblique axis do not.
comptime DX: Float64 = 0.037
comptime DY: Float64 = -0.021
comptime DZ: Float64 = 0.013
comptime TURN_RAD: Float64 = 0.43


def _read(p: String) raises -> String:
    var f = open(p, "r")
    var s = f.read()
    f.close()
    return s^


def _dir_of(p: String) -> String:
    var cut = p.rfind("/")
    if cut <= 0:
        return String(".")
    return String(p[byte=0:cut])


def _row(tag: String, i: Int, name: String, f: Frame):
    print(tag, i, name if name.byte_length() > 0 else String("-"),
          f.pos.x, f.pos.y, f.pos.z,
          f.quat.w, f.quat.x, f.quat.y, f.quat.z)


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: dump_gizmo_frames <model.xml> <out.xml>")
    var path = String(args[1])
    var out_path = String(args[2])
    var base = _dir_of(path)

    var src = expand_mjcf(_read(path), base)
    var fmd = parse_xml_full(src, base)
    # ⚠ THE SAME RETRY-ON-RAISE BUDGET LOOP `Loaded._build` USES. The mesh
    # vertex count is only known INSIDE the builder, so a fixed guess makes
    # every mesh model unloadable here while loading fine in the studio —
    # which would leave the one model the outline bug was found on out of the
    # gate.
    var verts = 0
    var dims = dims_from_flat(fmd, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except be:
            if String(be).find("mesh vertex capacity") == -1:
                raise be
            tries += 1
            if tries > 24:
                raise be
            verts = 4096 if verts == 0 else verts * 2
            dims = dims_from_flat(fmd, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    # ⚠ THE SAME SEEDING THE STUDIO DOES. `forward_kinematics` skips mocap
    # bodies, so without it so_arm101's `target` is at the world origin here
    # and at (0.25, 0, 0.2) in MuJoCo — a 0.25 m "gizmo error" that is
    # nothing of the kind.
    var n_mocap = reset_mocap_from_model[DT, DynDims, 1](m, d)
    forward_kinematics["cpu", DT, DynDims, 1](d, m)

    var nbody = dims.get_nbody()
    var positions = List[Vec3]()
    var quats = List[Quat]()
    for b in range(nbody):
        positions.append(Vec3(
            Float64(d.xpos.data[b * 3 + 0]),
            Float64(d.xpos.data[b * 3 + 1]),
            Float64(d.xpos.data[b * 3 + 2]),
        ))
        quats.append(Quat(
            Float64(d.xquat.data[b * 4 + 3]),
            Float64(d.xquat.data[b * 4 + 0]),
            Float64(d.xquat.data[b * 4 + 1]),
            Float64(d.xquat.data[b * 4 + 2]),
        ))
    var body_parent = List[Int]()
    for bb in fmd.bodies:
        body_parent.append(bb.parent)

    print("MODEL", path)
    print("NGEOM", len(fmd.geoms))
    print("NBODY", nbody)
    print("NMOCAP", n_mocap)
    # ⚠⚠ THE POSE GOES OUT WITH THE FRAMES, because `qpos0` IS NOT AGREED.
    # This tree honours `<custom><numeric name="init_qpos">` as the reference
    # pose (Gymnasium's convention, deliberate — see `full_parser`); MuJoCo
    # derives `qpos0` from the body frames and ignores that section. ant's
    # torso is therefore at z=0.55 here and z=0.75 there. Comparing the two
    # sides at their OWN reference poses would measure that disagreement
    # instead of measuring the gizmo, so the reader is told which pose these
    # frames were computed at and sets it before asking.
    var qs = String("QPOS ") + String(dims.get_nq())
    for i in range(dims.get_nq()):
        qs += " " + String(Float64(d.qpos.data[i]))
    print(qs)

    # ── where the gizmo would sit, for every element ──────────────────────
    for g in range(len(fmd.geoms)):
        var nm = fmd.geom_names[g] if g < len(fmd.geom_names) else String("")
        _row(String("GEOM"), g, nm,
             edit_frame(fmd, positions, quats, body_parent, TARGET_GEOM, g))
        print("GTYPE", g, fmd.geoms[g].geom_type)
    for b in range(1, nbody):
        var bn = fmd.body_names[b] if b < len(fmd.body_names) else String("")
        _row(String("BODY"), b, bn,
             edit_frame(fmd, positions, quats, body_parent, TARGET_BODY, b))
        # ⚠⚠ THE FK POSE GOES OUT BESIDE THE EDIT FRAME, because THEY ARE NOT
        # THE SAME THING AND BOTH NEED CHECKING. `BODY` is the frame the
        # gizmo edits — the body's `pos=`/`quat=` composed onto its parent.
        # `FK` is where the body actually IS, joints included. On ant, whose
        # `init_qpos` parks every hinge at ±1 rad, they differ by 0.2 m and
        # 0.8 in quaternion distance — which is `frame_drift`, working, and
        # would read as a 0.2 m gizmo error to a checker that only knew about
        # one of them.
        _row(String("FK"), b, bn, Frame(positions[b], quats[b]))

    # ── the victim: a NAMED, non-mesh, non-plane geom on a real body ──────
    # ⚠ NAMED, because `apply_edit_to_document` can only locate an unnamed
    # geom by counting within its parent — which works, and would make a
    # failure here ambiguous between the locator and the frame algebra.
    # ⚠ NON-MESH, because MuJoCo BAKES a mesh's recentering into `geom_pos`,
    # so its `geom_xpos` is not the frame the document states and the
    # comparison would need that undone. The Python side skips mesh geoms in
    # the survey arm for the same reason, and SAYS how many it skipped.
    var vic = -1
    for g in range(len(fmd.geoms)):
        if fmd.geoms[g].geom_type == GEOM_MESH \
                or fmd.geoms[g].geom_type == GEOM_PLANE:
            continue
        if fmd.geoms[g].body_id <= 0:
            continue
        if g >= len(fmd.geom_names) or fmd.geom_names[g].byte_length() == 0:
            continue
        vic = g
        break
    if vic < 0:
        print("EDIT none")
        print("WROTE none")
        return

    var before = edit_frame(fmd, positions, quats, body_parent,
                            TARGET_GEOM, vic)
    # ⚠ THE AXIS IS OBLIQUE AND NORMALISED. A rotation about a coordinate
    # axis commutes with too much: three of the nine matrix entries stay put,
    # and a transposed or mis-ordered composition can still agree.
    var axis = Vec3(0.37, -0.55, 0.75).normalized()
    var want = Frame(
        before.pos + Vec3(DX, DY, DZ),
        (Quat.from_axis_angle(axis, TURN_RAD) * before.quat).normalized(),
    )

    print("EDIT", vic, fmd.geom_names[vic])
    print("WANT", want.pos.x, want.pos.y, want.pos.z,
          want.quat.w, want.quat.x, want.quat.y, want.quat.z)

    var moved = gizmo_edits(
        fmd, positions, quats, body_parent, TARGET_GEOM, vic, GIZMO_MOVE,
        frame_to_cm(Frame(want.pos, before.quat)), 1.0,
    )
    var doc = src
    for i in range(len(moved)):
        apply_edit(fmd, m, moved[i])
    for i in range(len(moved)):
        doc = apply_edit_to_document(fmd, m, doc, moved[i])

    # ⚠ THE TURN IS COMPUTED AGAINST THE **MOVED** FRAME. Reusing `before`
    # would ask for a rotation of a pose that no longer exists, and on a
    # rotated parent the two differ.
    var mid = edit_frame(fmd, positions, quats, body_parent, TARGET_GEOM, vic)
    var turned = gizmo_edits(
        fmd, positions, quats, body_parent, TARGET_GEOM, vic, GIZMO_TURN,
        frame_to_cm(Frame(mid.pos, want.quat)), 1.0,
    )
    for i in range(len(turned)):
        apply_edit(fmd, m, turned[i])
    for i in range(len(turned)):
        doc = apply_edit_to_document(fmd, m, doc, turned[i])

    print("NEDITS", len(moved), len(turned))
    var got = edit_frame(fmd, positions, quats, body_parent, TARGET_GEOM, vic)
    print("GOT", got.pos.x, got.pos.y, got.pos.z,
          got.quat.w, got.quat.x, got.quat.y, got.quat.z)

    var wf = open(out_path, "w")
    wf.write(doc)
    wf.close()
    print("WROTE", out_path)
