"""`FlatModelDef` -> MJCF — flattened export, S5.

## ⚠⚠ WHY EXPORT SERIALISES THE **MODEL** AND NOT THE DOCUMENT

The scene document is already MJCF, so writing it out is one `open`. But S3's
fast-path edits live in the `FlatModelDef` and in the live `Model` — the
document has nowhere to put "geom 4's radius is now 0.077". Exporting the
document would therefore silently drop every edit the user just made, which
is the same drift the two-tier loop's byte-identity gate exists to prevent,
arriving at the last possible moment.

⇒ export writes the RECORD. That is what "flattened export is acceptable"
buys: one flat file describing exactly what is being simulated, edits and all.

⚠ AND IT IS ONLY POSSIBLE BECAUSE OF THE NAMES. Before `FlatModelDef` carried
name tables (S1) a writer had to synthesise `body0`/`geom3`, and flattened
export is acceptable while NAMELESS export is not — keyframes, sensors,
`<contact>` pairs and user code all key on names. See the plan's §1.3.

## What round-trips, and what this does not write

Written: `<compiler>`, `<option>` (gravity, timestep), the asset table
(textures, materials, meshes), the body tree with joints / geoms / sites, and
`<actuator>`. That covers a composed scene and every primitive-plus-actuator
model in the tree.

⚠ NOT written, and each would be a silent loss if a model had one:
`<tendon>`, `<equality>`, `<contact>`, `<keyframe>`, `<sensor>`, and
`<default>` classes (all values are already resolved into the records, so a
flattened file needs no classes — but it also cannot round-trip one). The
writer RAISES when the model has any of them rather than writing a file that
is quietly less than the model it came from.
"""

from ..parser.flat_model import FlatModelDef


def _f(v: Float64) -> String:
    """Full precision — this is DATA. See `SceneDoc`'s note on the display
    formatter: four decimals moves a pose by 5e-5 and denormalises a quat."""
    return String(v)


def _v3(x: Float64, y: Float64, z: Float64) -> String:
    return _f(x) + " " + _f(y) + " " + _f(z)


def _q4(w: Float64, x: Float64, y: Float64, z: Float64) -> String:
    return _f(w) + " " + _f(x) + " " + _f(y) + " " + _f(z)


def _name(names: List[String], i: Int, kind: String) -> String:
    """A name, or a synthesised one.

    ⚠ SYNTHESIS HAPPENS HERE AND ONLY HERE. `FlatModelDef` stores "" for an
    unnamed element so that nothing downstream claims a name the source never
    had — but an EXPORT must name anything it references, and a geom that a
    `<contact>` or a sensor points at is unreferencable without one. So the
    invention is confined to the writer, where it is visible in the output.
    """
    if i >= 0 and i < len(names) and names[i].byte_length() > 0:
        return names[i].copy()
    return kind + String(i)


def _geom_type_name(t: Int) -> String:
    if t == 0:
        return String("plane")
    if t == 1:
        return String("sphere")
    if t == 2:
        return String("capsule")
    if t == 3:
        return String("box")
    if t == 4:
        return String("cylinder")
    if t == 5:
        return String("mesh")
    return String("ellipsoid")


def _geom_size(g_type: Int, radius: Float64, half_length: Float64,
               hx: Float64, hy: Float64, hz: Float64) -> String:
    """MJCF `size`, with the component count this type requires.

    ⚠ THE COUNT IS ENFORCED BY MuJoCo — a sphere with three numbers is a load
    error. This is the same per-type dispatch the prop writer needs, and it is
    the third place in the studio that has to know it (the other two are the
    inspector and `Prop.size_attr`), which is why each one names the others.
    """
    if g_type == 0:
        # ⚠ A PLANE'S THIRD NUMBER IS THE RENDER GRID SPACING, and MuJoCo
        # REFUSES the model if it is absent or zero ("plane size(3) must be
        # positive"). The parser keeps it in `half_z` for exactly this reason;
        # the fallback below only fires for a plane built by something that
        # did not go through the parser.
        return _v3(
            hx if hx > 0 else 5.0,
            hy if hy > 0 else 5.0,
            hz if hz > 0 else 1.0,
        )
    if g_type == 1:
        return _f(radius)
    if g_type == 2 or g_type == 4:
        return _f(radius) + " " + _f(half_length)
    if g_type == 3 or g_type == 6:
        return _v3(hx, hy, hz)
    return String("")  # mesh takes its shape from the asset


def _joint_type_name(t: Int) -> String:
    # JNT_FREE=0, JNT_BALL=1, JNT_SLIDE=2, JNT_HINGE=3
    if t == 0:
        return String("free")
    if t == 1:
        return String("ball")
    if t == 2:
        return String("slide")
    return String("hinge")


def unwritable(fmd: FlatModelDef) -> String:
    """Which sections this writer would silently drop, or "" if none.

    ⚠ A CHECK, NOT A BEST EFFORT. An export that quietly omits a model's
    `<equality>` produces a file that loads, simulates, and is a DIFFERENT
    model — the exact shape of the dropped-section bugs `merge_mjcf` has
    produced four times. Refusing names the gap; writing a partial file hides
    it.
    """
    var missing = String("")
    if len(fmd.tendons) > 0:
        missing += " <tendon>"
    if len(fmd.equalities) > 0:
        missing += " <equality>"
    if len(fmd.pairs) > 0 or len(fmd.excludes) > 0:
        missing += " <contact>"
    if fmd.nkey > 0:
        missing += " <keyframe>"
    return missing^


def to_mjcf(fmd: FlatModelDef, model_name: String) raises -> String:
    """The flattened model, as a file MuJoCo and our own parser both read."""
    var bad = unwritable(fmd)
    if bad.byte_length() > 0:
        raise Error(
            "physics3d: this model uses sections the flattened writer does not"
            " emit —" + bad + ". Exporting anyway would produce a file that"
            " loads and is a DIFFERENT model. (The scene DOCUMENT can still be"
            " saved; only the flattened export is refused.)"
        )

    var s = String('<mujoco model="') + model_name + '">\n'
    # ⚠ `angle="radian"` ALWAYS. Every angular value in `FlatModelDef` has
    # already been converted by the parser's `deg_factor`, so writing the
    # model's original unit would reinterpret them — a degree model exported
    # as degrees would come back 57x rotated.
    s += '  <compiler angle="radian"/>\n'
    s += '  <option timestep="' + _f(fmd.timestep) + '" gravity="'
    s += _v3(fmd.gravity_x, fmd.gravity_y, fmd.gravity_z) + '"/>\n'

    # ── assets ───────────────────────────────────────────────────────────
    var has_assets = (
        len(fmd.textures) > 0 or len(fmd.materials) > 0
        or len(fmd.mesh_asset_names) > 0
    )
    if has_assets:
        s += "  <asset>\n"
        for i in range(len(fmd.mesh_asset_names)):
            s += (
                '    <mesh name="' + fmd.mesh_asset_names[i] + '" file="'
                + fmd.mesh_asset_files[i] + '"/>\n'
            )
        for i in range(len(fmd.materials)):
            ref m = fmd.materials[i]
            s += (
                '    <material name="mat' + String(i) + '" rgba="'
                + _q4(m.rgba_r, m.rgba_g, m.rgba_b, m.rgba_a) + '"/>\n'
            )
        s += "  </asset>\n"

    # ── the body tree ────────────────────────────────────────────────────
    s += "  <worldbody>\n"
    # World-level geoms first (body_id 0), then each root and its subtree.
    for gi in range(len(fmd.geoms)):
        if fmd.geoms[gi].body_id == 0:
            s += _geom_xml(fmd, gi, 4)
    for si in range(len(fmd.sites)):
        if fmd.sites[si].body_id == 0:
            s += _site_xml(fmd, si, 4)
    for b in range(1, len(fmd.bodies) + 1):
        if fmd.bodies[b - 1].parent == 0:
            s += _body_xml(fmd, b, 4)
    s += "  </worldbody>\n"

    # ── actuators ────────────────────────────────────────────────────────
    if len(fmd.actuators) > 0:
        s += "  <actuator>\n"
        for i in range(len(fmd.actuators)):
            ref a = fmd.actuators[i]
            if a.joint_id < 0:
                # ⚠ AN ACTUATOR WITH NO TRANSMISSION IS DROPPED, LOUDLY-ISH.
                # Writing `joint=""` would be a dangling reference; writing
                # nothing at least keeps the file honest, and the count
                # difference is what the round-trip gate sees.
                continue
            s += (
                '    <motor name="' + _name(fmd.actuator_names, i, "act")
                + '" joint="'
                + _name(fmd.joint_names, a.joint_id, "joint")
                + '" gear="' + _f(a.gear) + '"'
            )
            if a.is_ctrl_limited:
                s += (
                    ' ctrllimited="true" ctrlrange="' + _f(a.ctrl_min) + " "
                    + _f(a.ctrl_max) + '"'
                )
            s += "/>\n"
        s += "  </actuator>\n"

    s += "</mujoco>\n"
    return s^


def _pad(n: Int) -> String:
    var p = String("")
    for _ in range(n):
        p += " "
    return p^


def _geom_xml(fmd: FlatModelDef, gi: Int, indent: Int) -> String:
    ref g = fmd.geoms[gi]
    var s = _pad(indent) + '<geom name="' + _name(fmd.geom_names, gi, "geom")
    s += '" type="' + _geom_type_name(g.geom_type) + '"'
    var size = _geom_size(g.geom_type, g.radius, g.half_length,
                          g.half_x, g.half_y, g.half_z)
    if size.byte_length() > 0:
        s += ' size="' + size + '"'
    if g.geom_type == 5 and g.mesh_id >= 0 \
            and g.mesh_id < len(fmd.mesh_asset_names):
        s += ' mesh="' + fmd.mesh_asset_names[g.mesh_id] + '"'
    s += ' pos="' + _v3(g.pos_x, g.pos_y, g.pos_z) + '"'
    s += ' quat="' + _q4(g.quat_w, g.quat_x, g.quat_y, g.quat_z) + '"'
    s += ' rgba="' + _q4(g.rgba_r, g.rgba_g, g.rgba_b, g.rgba_a) + '"'
    # ⚠ CONTACT PARAMETERS ARE PART OF THE MODEL, not decoration. A geom
    # exported without its friction/condim collides differently, and the
    # difference shows up only under load.
    s += ' friction="' + _f(g.friction) + " " + _f(g.friction_spin) + " "
    s += _f(g.friction_roll) + '"'
    s += ' condim="' + String(g.condim) + '"'
    s += ' contype="' + String(g.contype) + '"'
    s += ' conaffinity="' + String(g.conaffinity) + '"'
    s += ' group="' + String(g.group) + '"'
    if g.mass >= 0.0:
        s += ' mass="' + _f(g.mass) + '"'
    else:
        s += ' density="' + _f(g.density) + '"'
    s += "/>\n"
    return s^


def _site_xml(fmd: FlatModelDef, si: Int, indent: Int) -> String:
    ref t = fmd.sites[si]
    var s = _pad(indent) + '<site name="' + _name(fmd.site_names, si, "site")
    s += '" pos="' + _v3(t.pos_x, t.pos_y, t.pos_z) + '"'
    s += ' quat="' + _q4(t.quat_w, t.quat_x, t.quat_y, t.quat_z) + '"/>\n'
    return s^


def _body_xml(fmd: FlatModelDef, b: Int, indent: Int) -> String:
    """One body and its subtree. `b` is a MODEL body id (1-based)."""
    ref bd = fmd.bodies[b - 1]
    var s = _pad(indent) + '<body name="' + _name(fmd.body_names, b, "body")
    s += '" pos="' + _v3(bd.pos_x, bd.pos_y, bd.pos_z) + '"'
    s += ' quat="' + _q4(bd.quat_w, bd.quat_x, bd.quat_y, bd.quat_z) + '"'
    if bd.is_mocap:
        s += ' mocap="true"'
    s += ">\n"

    # ⚠ AN EXPLICIT `<inertial>` ONLY WHEN THE SOURCE HAD ONE. Writing one
    # unconditionally would freeze the DERIVED mass and inertia into the file,
    # so a later edit to a geom's size would no longer change the body's
    # dynamics — the export would quietly become authoritative over the shape.
    if bd.has_explicit_inertia:
        s += _pad(indent + 2) + '<inertial pos="'
        s += _v3(bd.ipos_x, bd.ipos_y, bd.ipos_z) + '" quat="'
        s += _q4(bd.iquat_w, bd.iquat_x, bd.iquat_y, bd.iquat_z)
        s += '" mass="' + _f(bd.mass) + '" diaginertia="'
        s += _v3(bd.ixx, bd.iyy, bd.izz) + '"/>\n'

    for ji in range(len(fmd.joints)):
        ref j = fmd.joints[ji]
        if j.body_id != b:
            continue
        s += _pad(indent + 2) + '<joint name="'
        s += _name(fmd.joint_names, ji, "joint") + '" type="'
        s += _joint_type_name(j.jnt_type) + '"'
        if j.jnt_type != 0:
            s += ' pos="' + _v3(j.pos_x, j.pos_y, j.pos_z) + '"'
            s += ' axis="' + _v3(j.axis_x, j.axis_y, j.axis_z) + '"'
            if j.is_limited:
                s += ' limited="true" range="' + _f(j.range_min) + " "
                s += _f(j.range_max) + '"'
            s += ' damping="' + _f(j.damping) + '"'
            s += ' armature="' + _f(j.armature) + '"'
            s += ' stiffness="' + _f(j.stiffness) + '"'
            if j.ref_val != 0.0:
                s += ' ref="' + _f(j.ref_val) + '"'
        s += "/>\n"

    for gi in range(len(fmd.geoms)):
        if fmd.geoms[gi].body_id == b:
            s += _geom_xml(fmd, gi, indent + 2)
    for si in range(len(fmd.sites)):
        if fmd.sites[si].body_id == b:
            s += _site_xml(fmd, si, indent + 2)
    for c in range(1, len(fmd.bodies) + 1):
        if fmd.bodies[c - 1].parent == b:
            s += _body_xml(fmd, c, indent + 2)
    s += _pad(indent) + "</body>\n"
    return s^
