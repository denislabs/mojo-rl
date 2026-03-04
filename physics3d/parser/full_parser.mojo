"""Full MJCF XML parser — populates a FlatModelDef from an embedded XML string.

Designed to run entirely at comptime:

    comptime pm  = parse_xml(xml)
    comptime fmd = parse_xml_full[pm.NBODY, pm.NJOINT, pm.NQ, pm.NV,
                                   pm.NGEOM, pm.NACT](xml)

    var model = Model[DType.float64, pm.NQ, pm.NV, pm.NBODY, pm.NJOINT, 10,
                      pm.NGEOM, ...]()
    fmd.setup_model[DType.float64, 10](model)

All string operations use String.find() and slice notation — proven comptime-safe.
No stdlib float parsing or trig imports are used; everything is computed
with arithmetic helpers defined in xml_parser.mojo.
"""

from std.collections import InlineArray
from .xml_parser import (
    _extract_section,
    _extract_opening_tag,
    _extract_attr,
    _trim,
    _parse_float,
    _parse_int_str,
    _parse_vec3,
    _parse_quat,
    _parse_axisangle_to_quat,
    _fromto_to_pos_quat,
    _find_joint_index_by_name,
    _sqrt_f64,
)
from .flat_model import (
    BodyData,
    JointData,
    GeomData,
    ActuatorData,
    TextureData,
    MaterialData,
    LightData,
    CameraData,
    SiteData,
    DefaultsData,
    FlatModelDef,
    _GEOM_PLANE,
    _GEOM_SPHERE,
    _GEOM_CAPSULE,
    _GEOM_BOX,
    _GEOM_CYLINDER,
    TEX_SKYBOX,
    TEX_2D,
    TEX_CUBE,
    TEX_BUILTIN_NONE,
    TEX_BUILTIN_GRADIENT,
    TEX_BUILTIN_CHECKER,
    TEX_BUILTIN_FLAT,
    TEX_MARK_NONE,
    TEX_MARK_EDGE,
    TEX_MARK_CROSS,
    TEX_MARK_RANDOM,
    LIGHT_MODE_FIXED,
    LIGHT_MODE_TRACK,
    LIGHT_MODE_TRACKCOM,
    LIGHT_MODE_TARGETBODY,
    LIGHT_MODE_TARGETBODYCOM,
    CAM_MODE_FIXED,
    CAM_MODE_TRACK,
    CAM_MODE_TRACKCOM,
    CAM_MODE_TARGETBODY,
    CAM_MODE_TARGETBODYCOM,
)
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE


# =============================================================================
# Internal: min of two ints treating -1 as +∞
# =============================================================================


fn _min_valid(a: Int, b: Int) -> Int:
    """Return the smaller of a and b, treating -1 as +infinity."""
    if a == -1:
        return b
    if b == -1:
        return a
    if a < b:
        return a
    return b


# =============================================================================
# Phase 1: Parse <option> — gravity + timestep
# =============================================================================


fn _parse_option(xml: String) -> Tuple[Float64, Float64, Float64, Float64]:
    """Extract (gravity_x, gravity_y, gravity_z, timestep) from <option .../>.

    Defaults: gravity=(0,0,-9.81), timestep=0.01.
    """
    var gx = Float64(0)
    var gy = Float64(0)
    var gz = Float64(-9.81)
    var ts = Float64(0.01)

    var pos = xml.find("<option")
    if pos == -1:
        return (gx, gy, gz, ts)

    var tag = _extract_opening_tag(xml, pos)

    var gravity_str = _extract_attr(tag, "gravity")
    if len(gravity_str) > 0:
        var gvec = _parse_vec3(gravity_str)
        gx = gvec[0]
        gy = gvec[1]
        gz = gvec[2]

    var ts_str = _extract_attr(tag, "timestep")
    if len(ts_str) > 0:
        ts = _parse_float(ts_str)

    return (gx, gy, gz, ts)


# =============================================================================
# Phase 2: Parse <default> block
# =============================================================================


fn _parse_defaults(xml: String) -> DefaultsData:
    """Extract default joint/geom/motor attrs from the <default> section."""
    var d = DefaultsData()
    var defaults_sec = _extract_section(xml, "default")
    if len(defaults_sec) == 0:
        return d

    # Find default <joint
    var jpos = defaults_sec.find("<joint")
    if jpos != -1:
        var jtag = _extract_opening_tag(defaults_sec, jpos)

        var arm_s = _extract_attr(jtag, "armature")
        if len(arm_s) > 0:
            d.joint_armature = _parse_float(arm_s)

        var damp_s = _extract_attr(jtag, "damping")
        if len(damp_s) > 0:
            d.joint_damping = _parse_float(damp_s)

        var stiff_s = _extract_attr(jtag, "stiffness")
        if len(stiff_s) > 0:
            d.joint_stiffness = _parse_float(stiff_s)

        var lim_s = _extract_attr(jtag, "limited")
        if lim_s == "true":
            d.joint_limited = True
        elif lim_s == "false":
            d.joint_limited = False

        var fl_s = _extract_attr(jtag, "frictionloss")
        if len(fl_s) > 0:
            d.joint_frictionloss = _parse_float(fl_s)

        var sr_s = _extract_attr(jtag, "springref")
        if len(sr_s) > 0:
            d.joint_springref = _parse_float(sr_s)

        var srl_s = _extract_attr(jtag, "solreflimit")
        if len(srl_s) > 0:
            var sv = _parse_vec3(srl_s)
            d.joint_solref_limit_0 = sv[0]
            d.joint_solref_limit_1 = sv[1]

        var sil_s = _extract_attr(jtag, "solimplimit")
        if len(sil_s) > 0:
            var parts = List[String]()
            from .xml_parser import _split_spaces

            _split_spaces(sil_s, parts)
            if len(parts) >= 1:
                d.joint_solimp_limit_0 = _parse_float(parts[0])
            if len(parts) >= 2:
                d.joint_solimp_limit_1 = _parse_float(parts[1])
            if len(parts) >= 3:
                d.joint_solimp_limit_2 = _parse_float(parts[2])
            if len(parts) >= 4:
                d.joint_solimp_limit_3 = _parse_float(parts[3])
            if len(parts) >= 5:
                d.joint_solimp_limit_4 = _parse_float(parts[4])

    # Find default <geom
    var gpos = defaults_sec.find("<geom")
    if gpos != -1:
        var gtag = _extract_opening_tag(defaults_sec, gpos)

        var dens_s = _extract_attr(gtag, "density")
        if len(dens_s) > 0:
            d.geom_density = _parse_float(dens_s)

        var fric_s = _extract_attr(gtag, "friction")
        if len(fric_s) > 0:
            var fvec = _parse_vec3(fric_s)
            d.geom_friction = fvec[0]
            d.geom_friction_spin = fvec[1]
            d.geom_friction_roll = fvec[2]

        var ct_s = _extract_attr(gtag, "contype")
        if len(ct_s) > 0:
            d.geom_contype = _parse_int_str(ct_s)

        var ca_s = _extract_attr(gtag, "conaffinity")
        if len(ca_s) > 0:
            d.geom_conaffinity = _parse_int_str(ca_s)

        var cd_s = _extract_attr(gtag, "condim")
        if len(cd_s) > 0:
            d.geom_condim = _parse_int_str(cd_s)

        var sr0_s = _extract_attr(gtag, "solref")
        if len(sr0_s) > 0:
            var sv = _parse_vec3(sr0_s)
            d.geom_solref_0 = sv[0]
            d.geom_solref_1 = sv[1]

        var si0_s = _extract_attr(gtag, "solimp")
        if len(si0_s) > 0:
            var parts = List[String]()
            from .xml_parser import _split_spaces

            _split_spaces(si0_s, parts)
            if len(parts) >= 1:
                d.geom_solimp_0 = _parse_float(parts[0])
            if len(parts) >= 2:
                d.geom_solimp_1 = _parse_float(parts[1])
            if len(parts) >= 3:
                d.geom_solimp_2 = _parse_float(parts[2])
            if len(parts) >= 4:
                d.geom_solimp_3 = _parse_float(parts[3])
            if len(parts) >= 5:
                d.geom_solimp_4 = _parse_float(parts[4])

        var mg_s = _extract_attr(gtag, "margin")
        if len(mg_s) > 0:
            d.geom_margin = _parse_float(mg_s)

        var rgba_s = _extract_attr(gtag, "rgba")
        if len(rgba_s) > 0:
            var cv = _parse_rgba4(rgba_s)
            d.geom_rgba_r = cv[0]
            d.geom_rgba_g = cv[1]
            d.geom_rgba_b = cv[2]
            d.geom_rgba_a = cv[3]

    # Find default <motor
    var mpos = defaults_sec.find("<motor")
    if mpos != -1:
        var mtag = _extract_opening_tag(defaults_sec, mpos)

        var cl_s = _extract_attr(mtag, "ctrllimited")
        if cl_s == "true":
            d.motor_ctrl_limited = True

        var cr_s = _extract_attr(mtag, "ctrlrange")
        if len(cr_s) > 0:
            var cvec = _parse_vec3(cr_s)
            d.motor_ctrl_min = cvec[0]
            d.motor_ctrl_max = cvec[1]

    return d


# =============================================================================
# Phase 3: Parse geom type string → Int constant
# =============================================================================


fn _geom_type_from_str(s: String) -> Int:
    """Convert geom type string to integer constant."""
    var t = _trim(s)
    if t == "plane":
        return _GEOM_PLANE
    elif t == "sphere":
        return _GEOM_SPHERE
    elif t == "capsule":
        return _GEOM_CAPSULE
    elif t == "box":
        return _GEOM_BOX
    elif t == "cylinder":
        return _GEOM_CYLINDER
    return _GEOM_SPHERE  # default


# =============================================================================
# Phase 4a: Parse <asset> section — textures and materials
# =============================================================================


fn _tex_type_from_str(s: String) -> Int:
    var t = _trim(s)
    if t == "skybox":
        return TEX_SKYBOX
    elif t == "cube":
        return TEX_CUBE
    return TEX_2D  # default


fn _tex_builtin_from_str(s: String) -> Int:
    var t = _trim(s)
    if t == "gradient":
        return TEX_BUILTIN_GRADIENT
    elif t == "checker":
        return TEX_BUILTIN_CHECKER
    elif t == "flat":
        return TEX_BUILTIN_FLAT
    return TEX_BUILTIN_NONE


fn _tex_mark_from_str(s: String) -> Int:
    var t = _trim(s)
    if t == "edge":
        return TEX_MARK_EDGE
    elif t == "cross":
        return TEX_MARK_CROSS
    elif t == "random":
        return TEX_MARK_RANDOM
    return TEX_MARK_NONE


fn _light_mode_from_str(s: String) -> Int:
    var t = _trim(s)
    if t == "track":
        return LIGHT_MODE_TRACK
    elif t == "trackcom":
        return LIGHT_MODE_TRACKCOM
    elif t == "targetbody":
        return LIGHT_MODE_TARGETBODY
    elif t == "targetbodycom":
        return LIGHT_MODE_TARGETBODYCOM
    return LIGHT_MODE_FIXED


fn _cam_mode_from_str(s: String) -> Int:
    var t = _trim(s)
    if t == "track":
        return CAM_MODE_TRACK
    elif t == "trackcom":
        return CAM_MODE_TRACKCOM
    elif t == "targetbody":
        return CAM_MODE_TARGETBODY
    elif t == "targetbodycom":
        return CAM_MODE_TARGETBODYCOM
    return CAM_MODE_FIXED


fn _find_texture_index_by_name(asset_sec: String, name: String) -> Int:
    """Return 0-based index of <texture name="name"> in asset_sec, or -1."""
    var search = 'name="' + name + '"'
    var scan_pos = 0
    var count = 0
    while True:
        var t = asset_sec.find("<texture", scan_pos)
        if t == -1:
            break
        var tag_end = asset_sec.find(">", t)
        if tag_end == -1:
            break
        var tag = String(asset_sec[t : tag_end + 1])
        if tag.find(search) != -1:
            return count
        count += 1
        scan_pos = tag_end + 1
    return -1


fn _parse_rgb3(s: String) -> Tuple[Float64, Float64, Float64]:
    """Parse "r g b" string into three Float64 values."""
    var parts = List[String]()
    from .xml_parser import _split_spaces

    _split_spaces(s, parts)
    var r = Float64(0)
    var g = Float64(0)
    var b = Float64(0)
    if len(parts) >= 1:
        r = _parse_float(parts[0])
    if len(parts) >= 2:
        g = _parse_float(parts[1])
    if len(parts) >= 3:
        b = _parse_float(parts[2])
    return (r, g, b)


fn _parse_rgba4(s: String) -> Tuple[Float64, Float64, Float64, Float64]:
    """Parse "r g b a" string into four Float64 values."""
    var parts = List[String]()
    from .xml_parser import _split_spaces

    _split_spaces(s, parts)
    var r = Float64(1)
    var g = Float64(1)
    var b = Float64(1)
    var a = Float64(1)
    if len(parts) >= 1:
        r = _parse_float(parts[0])
    if len(parts) >= 2:
        g = _parse_float(parts[1])
    if len(parts) >= 3:
        b = _parse_float(parts[2])
    if len(parts) >= 4:
        a = _parse_float(parts[3])
    return (r, g, b, a)


fn _xyaxes_to_quat(s: String) -> Tuple[Float64, Float64, Float64, Float64]:
    """Convert xyaxes="x1 x2 x3 y1 y2 y3" to quaternion (qx, qy, qz, qw).

    The six values define the X and Y axes of the frame in world coordinates.
    Z = normalize(cross(X, Y)).  Returns identity (0,0,0,1) on parse error.
    """
    var parts = List[String]()
    from .xml_parser import _split_spaces, _sqrt_f64

    _split_spaces(s, parts)
    if len(parts) < 6:
        return (Float64(0), Float64(0), Float64(0), Float64(1))

    var xx = _parse_float(parts[0])
    var xy = _parse_float(parts[1])
    var xz = _parse_float(parts[2])
    var yx = _parse_float(parts[3])
    var yy = _parse_float(parts[4])
    var yz = _parse_float(parts[5])

    # Normalize X axis
    var xn = _sqrt_f64(xx * xx + xy * xy + xz * xz)
    if xn > 0.0:
        xx /= xn
        xy /= xn
        xz /= xn

    # Z = cross(X, Y)
    var zx = xy * yz - xz * yy
    var zy = xz * yx - xx * yz
    var zz = xx * yy - xy * yx
    var zn = _sqrt_f64(zx * zx + zy * zy + zz * zz)
    if zn > 0.0:
        zx /= zn
        zy /= zn
        zz /= zn

    # Recompute Y = cross(Z, X) to ensure orthogonality
    yx = zy * xz - zz * xy
    yy = zz * xx - zx * xz
    yz = zx * xy - zy * xx

    # Rotation matrix (column-major: col0=X, col1=Y, col2=Z) → quaternion
    # Using standard matrix-to-quaternion (Shepperd method)
    var trace = xx + yy + zz
    var qx: Float64
    var qy: Float64
    var qz: Float64
    var qw: Float64

    if trace > 0.0:
        var s2 = _sqrt_f64(trace + 1.0) * 2.0  # s2 = 4*qw
        qw = 0.25 * s2
        qx = (zy - yz) / s2
        qy = (xz - zx) / s2
        qz = (yx - xy) / s2
    elif xx > yy and xx > zz:
        var s2 = _sqrt_f64(1.0 + xx - yy - zz) * 2.0  # s2 = 4*qx
        qw = (zy - yz) / s2
        qx = 0.25 * s2
        qy = (xy + yx) / s2
        qz = (xz + zx) / s2
    elif yy > zz:
        var s2 = _sqrt_f64(1.0 + yy - xx - zz) * 2.0  # s2 = 4*qy
        qw = (xz - zx) / s2
        qx = (xy + yx) / s2
        qy = 0.25 * s2
        qz = (yz + zy) / s2
    else:
        var s2 = _sqrt_f64(1.0 + zz - xx - yy) * 2.0  # s2 = 4*qz
        qw = (yx - xy) / s2
        qx = (xz + zx) / s2
        qy = (yz + zy) / s2
        qz = 0.25 * s2

    return (qx, qy, qz, qw)


fn _fill_assets[
    NBODY: Int,
    NJOINT: Int,
    NQ: Int,
    NV: Int,
    NGEOM: Int,
    NACT: Int,
    NTEX: Int,
    NMAT: Int,
    NLIGHT: Int,
    NCAM: Int,
    NSITE: Int,
](
    asset_sec: String,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE
    ],
):
    """Parse <asset> section: fill result.textures[] and result.materials[]."""

    # ---- Textures -----------------------------------------------------------
    var tex_pos = 0
    var tex_count = 0
    while tex_count < NTEX:
        var t = asset_sec.find("<texture", tex_pos)
        if t == -1:
            break
        var tag_end = asset_sec.find(">", t)
        if tag_end == -1:
            break
        var tag = String(asset_sec[t : tag_end + 1])

        var td = TextureData()

        var type_s = _extract_attr(tag, "type")
        td.tex_type = _tex_type_from_str(type_s)

        var builtin_s = _extract_attr(tag, "builtin")
        td.builtin = _tex_builtin_from_str(builtin_s)

        var mark_s = _extract_attr(tag, "mark")
        td.mark = _tex_mark_from_str(mark_s)

        var rgb1_s = _extract_attr(tag, "rgb1")
        if len(rgb1_s) > 0:
            var c = _parse_rgb3(rgb1_s)
            td.rgb1_r = c[0]
            td.rgb1_g = c[1]
            td.rgb1_b = c[2]

        var rgb2_s = _extract_attr(tag, "rgb2")
        if len(rgb2_s) > 0:
            var c = _parse_rgb3(rgb2_s)
            td.rgb2_r = c[0]
            td.rgb2_g = c[1]
            td.rgb2_b = c[2]

        var markrgb_s = _extract_attr(tag, "markrgb")
        if len(markrgb_s) > 0:
            var c = _parse_rgb3(markrgb_s)
            td.markrgb_r = c[0]
            td.markrgb_g = c[1]
            td.markrgb_b = c[2]

        var w_s = _extract_attr(tag, "width")
        if len(w_s) > 0:
            td.width = _parse_int_str(w_s)

        var h_s = _extract_attr(tag, "height")
        if len(h_s) > 0:
            td.height = _parse_int_str(h_s)

        var rand_s = _extract_attr(tag, "random")
        if len(rand_s) > 0:
            td.random = _parse_float(rand_s)

        result.textures[tex_count] = td
        tex_count += 1
        tex_pos = tag_end + 1

    # ---- Materials ----------------------------------------------------------
    var mat_pos = 0
    var mat_count = 0
    while mat_count < NMAT:
        var t = asset_sec.find("<material", mat_pos)
        if t == -1:
            break
        var tag_end = asset_sec.find(">", t)
        if tag_end == -1:
            break
        var tag = String(asset_sec[t : tag_end + 1])

        var md = MaterialData()

        # texture reference → index
        var tex_name = _extract_attr(tag, "texture")
        if len(tex_name) > 0:
            md.tex_id = _find_texture_index_by_name(asset_sec, tex_name)

        var rgba_s = _extract_attr(tag, "rgba")
        if len(rgba_s) > 0:
            var c = _parse_rgba4(rgba_s)
            md.rgba_r = c[0]
            md.rgba_g = c[1]
            md.rgba_b = c[2]
            md.rgba_a = c[3]

        var shin_s = _extract_attr(tag, "shininess")
        if len(shin_s) > 0:
            md.shininess = _parse_float(shin_s)

        var spec_s = _extract_attr(tag, "specular")
        if len(spec_s) > 0:
            md.specular = _parse_float(spec_s)

        var refl_s = _extract_attr(tag, "reflectance")
        if len(refl_s) > 0:
            md.reflectance = _parse_float(refl_s)

        var tr_s = _extract_attr(tag, "texrepeat")
        if len(tr_s) > 0:
            var tv = _parse_vec3(tr_s)
            md.texrepeat_u = tv[0]
            md.texrepeat_v = tv[1]

        var tu_s = _extract_attr(tag, "texuniform")
        if tu_s == "true":
            md.texuniform = True

        result.materials[mat_count] = md
        mat_count += 1
        mat_pos = tag_end + 1


# =============================================================================
# Phase 4b: Combined DFS scan — fills bodies, joints, geoms in one pass
# =============================================================================


fn _fill_model[
    NBODY: Int,
    NJOINT: Int,
    NQ: Int,
    NV: Int,
    NGEOM: Int,
    NACT: Int,
    NTEX: Int,
    NMAT: Int,
    NLIGHT: Int,
    NCAM: Int,
    NSITE: Int,
](
    worldbody: String,
    defaults: DefaultsData,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE
    ],
    deg_factor: Float64 = 1.0,
):
    """Single-pass DFS over worldbody XML to populate bodies, joints, geoms,
    lights, cameras, and sites.

    deg_factor: 1.0 for radian models, pi/180 for degree models.
    Applied to joint range values and axisangle rotation angles.

    Uses two-pointer scan: tracks `<body` and `</body>` to maintain depth/parent.
    Joints, geoms, lights, cameras, and sites encountered at each depth are
    assigned to the currently-open body.
    """
    # body_id_stack[depth] = body index at current depth
    # depth 0 = worldbody level (body_id=0)
    var body_id_stack = InlineArray[Int, NBODY + 1](fill=0)
    var depth = 0
    var body_count = 0  # bodies[0..NBODY-2] → model body indices 1..NBODY-1
    var joint_count = 0
    var geom_count = 0
    var light_count = 0
    var cam_count = 0
    var site_count = 0

    var scan_pos = 0
    var wlen = len(worldbody)

    while scan_pos < wlen:
        var next_body_open = worldbody.find("<body", scan_pos)
        var next_body_close = worldbody.find("</body>", scan_pos)
        var next_joint = worldbody.find("<joint", scan_pos)
        var next_geom = worldbody.find("<geom", scan_pos)
        var next_light = worldbody.find("<light", scan_pos)
        var next_cam = worldbody.find("<camera", scan_pos)
        var next_site = worldbody.find("<site", scan_pos)

        # Check for no more interesting tokens
        var all_invalid = (
            next_body_open == -1
            and next_body_close == -1
            and next_joint == -1
            and next_geom == -1
            and next_light == -1
            and next_cam == -1
            and next_site == -1
        )
        if all_invalid:
            break

        # Find the earliest token
        var earliest = _min_valid(
            _min_valid(
                _min_valid(next_body_open, next_body_close),
                _min_valid(next_joint, next_geom),
            ),
            _min_valid(
                _min_valid(next_light, next_cam),
                next_site,
            ),
        )

        if earliest == next_body_open:
            # Opening <body ...>
            var tag = _extract_opening_tag(worldbody, next_body_open)
            var parent_id = body_id_stack[depth]
            depth += 1
            var this_body_id = body_count + 1  # model body index (worldbody=0)
            body_id_stack[depth] = this_body_id

            if body_count < NBODY - 1:
                var b = BodyData()
                b.parent = parent_id

                # pos
                var pos_s = _extract_attr(tag, "pos")
                if len(pos_s) > 0:
                    var pv = _parse_vec3(pos_s)
                    b.pos_x = pv[0]
                    b.pos_y = pv[1]
                    b.pos_z = pv[2]

                # quat / axisangle / euler orientation
                var quat_s = _extract_attr(tag, "quat")
                if len(quat_s) > 0:
                    var qv = _parse_quat(quat_s)
                    b.quat_x = qv[0]
                    b.quat_y = qv[1]
                    b.quat_z = qv[2]
                    b.quat_w = qv[3]
                else:
                    var aa_s = _extract_attr(tag, "axisangle")
                    if len(aa_s) > 0:
                        var aq = _parse_axisangle_to_quat(aa_s, deg_factor)
                        b.quat_x = aq[0]
                        b.quat_y = aq[1]
                        b.quat_z = aq[2]
                        b.quat_w = aq[3]

                # inertial pos/quat (ipos, iquat)
                var ipos_s = _extract_attr(tag, "ipos")
                if len(ipos_s) > 0:
                    var iv = _parse_vec3(ipos_s)
                    b.ipos_x = iv[0]
                    b.ipos_y = iv[1]
                    b.ipos_z = iv[2]

                var iquat_s = _extract_attr(tag, "iquat")
                if len(iquat_s) > 0:
                    var iq = _parse_quat(iquat_s)
                    b.iquat_x = iq[0]
                    b.iquat_y = iq[1]
                    b.iquat_z = iq[2]
                    b.iquat_w = iq[3]

                # mass (may be absent — inertia computed from geoms)
                var mass_s = _extract_attr(tag, "mass")
                if len(mass_s) > 0:
                    b.mass = _parse_float(mass_s)

                # diaginertia
                var di_s = _extract_attr(tag, "diaginertia")
                if len(di_s) > 0:
                    var dv = _parse_vec3(di_s)
                    b.ixx = dv[0]
                    b.iyy = dv[1]
                    b.izz = dv[2]

                result.bodies[body_count] = b
            body_count += 1
            # Advance past the opening tag
            var tag_end = worldbody.find(">", next_body_open)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

        elif earliest == next_body_close:
            # Closing </body>
            if depth > 0:
                depth -= 1
            scan_pos = next_body_close + 7  # len("</body>") == 7

        elif earliest == next_joint:
            # <joint ...>
            var current_body = body_id_stack[depth]
            var tag = _extract_opening_tag(worldbody, next_joint)

            if joint_count < NJOINT:
                var jd = JointData()
                jd.body_id = current_body

                # type
                var type_s = _extract_attr(tag, "type")
                var t = _trim(type_s)
                if t == "hinge" or t == "":
                    jd.jnt_type = JNT_HINGE
                    jd.nq = 1
                    jd.nv = 1
                elif t == "slide":
                    jd.jnt_type = JNT_SLIDE
                    jd.nq = 1
                    jd.nv = 1
                elif t == "ball":
                    jd.jnt_type = JNT_BALL
                    jd.nq = 4
                    jd.nv = 3
                elif t == "free":
                    jd.jnt_type = JNT_FREE
                    jd.nq = 7
                    jd.nv = 6

                # pos
                var pos_s = _extract_attr(tag, "pos")
                if len(pos_s) > 0:
                    var pv = _parse_vec3(pos_s)
                    jd.pos_x = pv[0]
                    jd.pos_y = pv[1]
                    jd.pos_z = pv[2]

                # axis
                var axis_s = _extract_attr(tag, "axis")
                if len(axis_s) > 0:
                    var av = _parse_vec3(axis_s)
                    jd.axis_x = av[0]
                    jd.axis_y = av[1]
                    jd.axis_z = av[2]

                # range (convert deg→rad when deg_factor != 1.0)
                var range_s = _extract_attr(tag, "range")
                if len(range_s) > 0:
                    var rv = _parse_vec3(range_s)
                    jd.range_min = rv[0] * deg_factor
                    jd.range_max = rv[1] * deg_factor
                    jd.is_limited = True

                # limited (explicit override)
                var lim_s = _extract_attr(tag, "limited")
                if lim_s == "false":
                    jd.is_limited = False
                    jd.range_min = Float64(-1e10)
                    jd.range_max = Float64(1e10)
                elif lim_s == "true":
                    jd.is_limited = True

                # armature (explicit or default)
                var arm_s = _extract_attr(tag, "armature")
                if len(arm_s) > 0:
                    jd.armature = _parse_float(arm_s)
                else:
                    jd.armature = defaults.joint_armature

                # damping
                var damp_s = _extract_attr(tag, "damping")
                if len(damp_s) > 0:
                    jd.damping = _parse_float(damp_s)
                else:
                    jd.damping = defaults.joint_damping

                # stiffness
                var stiff_s = _extract_attr(tag, "stiffness")
                if len(stiff_s) > 0:
                    jd.stiffness = _parse_float(stiff_s)
                else:
                    jd.stiffness = defaults.joint_stiffness

                # springref
                var sr_s = _extract_attr(tag, "springref")
                if len(sr_s) > 0:
                    jd.springref = _parse_float(sr_s)
                else:
                    jd.springref = defaults.joint_springref

                # ref (MuJoCo joint reference position → qpos0)
                var ref_s = _extract_attr(tag, "ref")
                if len(ref_s) > 0:
                    jd.ref_val = _parse_float(ref_s)
                else:
                    jd.ref_val = 0.0

                # frictionloss
                var fl_s = _extract_attr(tag, "frictionloss")
                if len(fl_s) > 0:
                    jd.frictionloss = _parse_float(fl_s)
                else:
                    jd.frictionloss = defaults.joint_frictionloss

                # solreflimit (per-joint or default)
                var srl_s = _extract_attr(tag, "solreflimit")
                if len(srl_s) > 0:
                    var sv = _parse_vec3(srl_s)
                    jd.solref_limit_0 = sv[0]
                    jd.solref_limit_1 = sv[1]
                else:
                    jd.solref_limit_0 = defaults.joint_solref_limit_0
                    jd.solref_limit_1 = defaults.joint_solref_limit_1

                # solimplimit (per-joint or default)
                var sil_s = _extract_attr(tag, "solimplimit")
                if len(sil_s) > 0:
                    var parts2 = List[String]()
                    from .xml_parser import _split_spaces

                    _split_spaces(sil_s, parts2)
                    if len(parts2) >= 1:
                        jd.solimp_limit_0 = _parse_float(parts2[0])
                    if len(parts2) >= 2:
                        jd.solimp_limit_1 = _parse_float(parts2[1])
                    if len(parts2) >= 3:
                        jd.solimp_limit_2 = _parse_float(parts2[2])
                    if len(parts2) >= 4:
                        jd.solimp_limit_3 = _parse_float(parts2[3])
                    if len(parts2) >= 5:
                        jd.solimp_limit_4 = _parse_float(parts2[4])
                else:
                    jd.solimp_limit_0 = defaults.joint_solimp_limit_0
                    jd.solimp_limit_1 = defaults.joint_solimp_limit_1
                    jd.solimp_limit_2 = defaults.joint_solimp_limit_2
                    jd.solimp_limit_3 = defaults.joint_solimp_limit_3
                    jd.solimp_limit_4 = defaults.joint_solimp_limit_4

                result.joints[joint_count] = jd
            joint_count += 1
            var tag_end = worldbody.find(">", next_joint)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

        elif earliest == next_light:
            # <light ...>
            var current_body = body_id_stack[depth]
            var tag = _extract_opening_tag(worldbody, next_light)

            if light_count < NLIGHT:
                var ld = LightData()
                ld.body_id = current_body

                var pos_s = _extract_attr(tag, "pos")
                if len(pos_s) > 0:
                    var pv = _parse_vec3(pos_s)
                    ld.pos_x = pv[0]
                    ld.pos_y = pv[1]
                    ld.pos_z = pv[2]

                var dir_s = _extract_attr(tag, "dir")
                if len(dir_s) > 0:
                    var dv = _parse_vec3(dir_s)
                    ld.dir_x = dv[0]
                    ld.dir_y = dv[1]
                    ld.dir_z = dv[2]

                var diff_s = _extract_attr(tag, "diffuse")
                if len(diff_s) > 0:
                    var c = _parse_rgb3(diff_s)
                    ld.diffuse_r = c[0]
                    ld.diffuse_g = c[1]
                    ld.diffuse_b = c[2]

                var spec_s = _extract_attr(tag, "specular")
                if len(spec_s) > 0:
                    var c = _parse_rgb3(spec_s)
                    ld.specular_r = c[0]
                    ld.specular_g = c[1]
                    ld.specular_b = c[2]

                var amb_s = _extract_attr(tag, "ambient")
                if len(amb_s) > 0:
                    var c = _parse_rgb3(amb_s)
                    ld.ambient_r = c[0]
                    ld.ambient_g = c[1]
                    ld.ambient_b = c[2]

                var dir_flag_s = _extract_attr(tag, "directional")
                ld.directional = dir_flag_s == "true"

                var shadow_s = _extract_attr(tag, "castshadow")
                if shadow_s == "false":
                    ld.castshadow = False

                var cutoff_s = _extract_attr(tag, "cutoff")
                if len(cutoff_s) > 0:
                    ld.cutoff = _parse_float(cutoff_s)

                var exp_s = _extract_attr(tag, "exponent")
                if len(exp_s) > 0:
                    ld.exponent = _parse_float(exp_s)

                var mode_s = _extract_attr(tag, "mode")
                if len(mode_s) > 0:
                    ld.mode = _light_mode_from_str(mode_s)

                result.lights[light_count] = ld
            light_count += 1
            var tag_end = worldbody.find(">", next_light)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

        elif earliest == next_cam:
            # <camera ...>
            var current_body = body_id_stack[depth]
            var tag = _extract_opening_tag(worldbody, next_cam)

            if cam_count < NCAM:
                var cd = CameraData()
                cd.body_id = current_body

                var pos_s = _extract_attr(tag, "pos")
                if len(pos_s) > 0:
                    var pv = _parse_vec3(pos_s)
                    cd.pos_x = pv[0]
                    cd.pos_y = pv[1]
                    cd.pos_z = pv[2]

                # Orientation: quat > axisangle > xyaxes
                var quat_s = _extract_attr(tag, "quat")
                if len(quat_s) > 0:
                    var qv = _parse_quat(quat_s)
                    cd.quat_x = qv[0]
                    cd.quat_y = qv[1]
                    cd.quat_z = qv[2]
                    cd.quat_w = qv[3]
                else:
                    var aa_s = _extract_attr(tag, "axisangle")
                    if len(aa_s) > 0:
                        var aq = _parse_axisangle_to_quat(aa_s, deg_factor)
                        cd.quat_x = aq[0]
                        cd.quat_y = aq[1]
                        cd.quat_z = aq[2]
                        cd.quat_w = aq[3]
                    else:
                        var xy_s = _extract_attr(tag, "xyaxes")
                        if len(xy_s) > 0:
                            var xq = _xyaxes_to_quat(xy_s)
                            cd.quat_x = xq[0]
                            cd.quat_y = xq[1]
                            cd.quat_z = xq[2]
                            cd.quat_w = xq[3]

                var fovy_s = _extract_attr(tag, "fovy")
                if len(fovy_s) > 0:
                    cd.fovy = _parse_float(fovy_s)

                var ipd_s = _extract_attr(tag, "ipd")
                if len(ipd_s) > 0:
                    cd.ipd = _parse_float(ipd_s)

                var mode_s = _extract_attr(tag, "mode")
                if len(mode_s) > 0:
                    cd.mode = _cam_mode_from_str(mode_s)

                result.cameras[cam_count] = cd
            cam_count += 1
            var tag_end = worldbody.find(">", next_cam)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

        elif earliest == next_site:
            # <site ...>
            var current_body = body_id_stack[depth]
            var tag = _extract_opening_tag(worldbody, next_site)

            if site_count < NSITE:
                var sd = SiteData()
                sd.body_id = current_body

                var type_s = _extract_attr(tag, "type")
                sd.site_type = _geom_type_from_str(type_s)

                var pos_s = _extract_attr(tag, "pos")
                if len(pos_s) > 0:
                    var pv = _parse_vec3(pos_s)
                    sd.pos_x = pv[0]
                    sd.pos_y = pv[1]
                    sd.pos_z = pv[2]

                var quat_s = _extract_attr(tag, "quat")
                if len(quat_s) > 0:
                    var qv = _parse_quat(quat_s)
                    sd.quat_x = qv[0]
                    sd.quat_y = qv[1]
                    sd.quat_z = qv[2]
                    sd.quat_w = qv[3]
                else:
                    var aa_s = _extract_attr(tag, "axisangle")
                    if len(aa_s) > 0:
                        var aq = _parse_axisangle_to_quat(aa_s, deg_factor)
                        sd.quat_x = aq[0]
                        sd.quat_y = aq[1]
                        sd.quat_z = aq[2]
                        sd.quat_w = aq[3]

                var size_s = _extract_attr(tag, "size")
                if len(size_s) > 0:
                    var parts = List[String]()
                    from .xml_parser import _split_spaces

                    _split_spaces(size_s, parts)
                    if len(parts) >= 1:
                        sd.size_0 = _parse_float(parts[0])
                    if len(parts) >= 2:
                        sd.size_1 = _parse_float(parts[1])
                    if len(parts) >= 3:
                        sd.size_2 = _parse_float(parts[2])

                result.sites[site_count] = sd
            site_count += 1
            var tag_end = worldbody.find(">", next_site)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

        else:  # earliest == next_geom
            # <geom ...>
            var current_body = body_id_stack[depth]
            var tag = _extract_opening_tag(worldbody, next_geom)

            if geom_count < NGEOM:
                var gd = GeomData()
                gd.body_id = current_body

                # type
                var type_s = _extract_attr(tag, "type")
                gd.geom_type = _geom_type_from_str(type_s)

                # fromto — overrides pos and quat for capsule
                var fromto_s = _extract_attr(tag, "fromto")
                if len(fromto_s) > 0:
                    var ft = _fromto_to_pos_quat(fromto_s)
                    gd.pos_x = ft[0]
                    gd.pos_y = ft[1]
                    gd.pos_z = ft[2]
                    gd.quat_x = ft[3]
                    gd.quat_y = ft[4]
                    gd.quat_z = ft[5]
                    gd.quat_w = ft[6]
                    gd.half_length = ft[7]
                    # radius from size attr (parsed below)
                else:
                    # pos
                    var pos_s = _extract_attr(tag, "pos")
                    if len(pos_s) > 0:
                        var pv = _parse_vec3(pos_s)
                        gd.pos_x = pv[0]
                        gd.pos_y = pv[1]
                        gd.pos_z = pv[2]

                    # orientation: quat or axisangle
                    var quat_s = _extract_attr(tag, "quat")
                    if len(quat_s) > 0:
                        var qv = _parse_quat(quat_s)
                        gd.quat_x = qv[0]
                        gd.quat_y = qv[1]
                        gd.quat_z = qv[2]
                        gd.quat_w = qv[3]
                    else:
                        var aa_s = _extract_attr(tag, "axisangle")
                        if len(aa_s) > 0:
                            var aq = _parse_axisangle_to_quat(aa_s, deg_factor)
                            gd.quat_x = aq[0]
                            gd.quat_y = aq[1]
                            gd.quat_z = aq[2]
                            gd.quat_w = aq[3]
                        else:
                            gd.quat_w = Float64(1)

                # size — interpretation depends on geom_type
                var size_s = _extract_attr(tag, "size")
                if len(size_s) > 0:
                    var size_parts = List[String]()
                    from .xml_parser import _split_spaces

                    _split_spaces(size_s, size_parts)
                    var s0 = Float64(0)
                    var s1 = Float64(0)
                    var s2 = Float64(0)
                    if len(size_parts) >= 1:
                        s0 = _parse_float(size_parts[0])
                    if len(size_parts) >= 2:
                        s1 = _parse_float(size_parts[1])
                    if len(size_parts) >= 3:
                        s2 = _parse_float(size_parts[2])

                    if gd.geom_type == _GEOM_SPHERE:
                        gd.radius = s0
                        gd.half_x = s0
                        gd.half_y = s0
                        gd.half_z = s0
                    elif gd.geom_type == _GEOM_CAPSULE:
                        gd.radius = s0
                        if len(size_parts) >= 2:
                            gd.half_length = s1
                        # else half_length already set from fromto
                    elif gd.geom_type == _GEOM_BOX:
                        gd.half_x = s0
                        gd.half_y = s1
                        gd.half_z = s2
                        gd.radius = _sqrt_f64(s0 * s0 + s1 * s1 + s2 * s2)
                    elif gd.geom_type == _GEOM_CYLINDER:
                        gd.radius = s0
                        gd.half_length = s1
                    elif gd.geom_type == _GEOM_PLANE:
                        gd.half_x = s0
                        gd.half_y = s1
                        # s2 = grid spacing — not needed for collision
                    else:
                        gd.radius = s0

                # friction (explicit or default)
                var fric_s = _extract_attr(tag, "friction")
                if len(fric_s) > 0:
                    var fvec = _parse_vec3(fric_s)
                    gd.friction = fvec[0]
                    gd.friction_spin = fvec[1]
                    gd.friction_roll = fvec[2]
                else:
                    gd.friction = defaults.geom_friction
                    gd.friction_spin = defaults.geom_friction_spin
                    gd.friction_roll = defaults.geom_friction_roll

                # contype / conaffinity / condim
                var ct_s = _extract_attr(tag, "contype")
                gd.contype = (
                    _parse_int_str(ct_s) if len(ct_s)
                    > 0 else defaults.geom_contype
                )

                var ca_s = _extract_attr(tag, "conaffinity")
                gd.conaffinity = (
                    _parse_int_str(ca_s) if len(ca_s)
                    > 0 else defaults.geom_conaffinity
                )

                var cd_s = _extract_attr(tag, "condim")
                gd.condim = (
                    _parse_int_str(cd_s) if len(cd_s)
                    > 0 else defaults.geom_condim
                )

                # solref / solimp
                var sr_s = _extract_attr(tag, "solref")
                if len(sr_s) > 0:
                    var sv = _parse_vec3(sr_s)
                    gd.solref_0 = sv[0]
                    gd.solref_1 = sv[1]
                else:
                    gd.solref_0 = defaults.geom_solref_0
                    gd.solref_1 = defaults.geom_solref_1

                var si_s = _extract_attr(tag, "solimp")
                if len(si_s) > 0:
                    var sip = List[String]()
                    from .xml_parser import _split_spaces

                    _split_spaces(si_s, sip)
                    if len(sip) >= 1:
                        gd.solimp_0 = _parse_float(sip[0])
                    if len(sip) >= 2:
                        gd.solimp_1 = _parse_float(sip[1])
                    if len(sip) >= 3:
                        gd.solimp_2 = _parse_float(sip[2])
                    if len(sip) >= 4:
                        gd.solimp_3 = _parse_float(sip[3])
                    if len(sip) >= 5:
                        gd.solimp_4 = _parse_float(sip[4])
                else:
                    gd.solimp_0 = defaults.geom_solimp_0
                    gd.solimp_1 = defaults.geom_solimp_1
                    gd.solimp_2 = defaults.geom_solimp_2
                    gd.solimp_3 = defaults.geom_solimp_3
                    gd.solimp_4 = defaults.geom_solimp_4

                # margin
                var mg_s = _extract_attr(tag, "margin")
                gd.margin = (
                    _parse_float(mg_s) if len(mg_s)
                    > 0 else defaults.geom_margin
                )

                # density (per-geom overrides default; used when mass is absent)
                var dens_s = _extract_attr(tag, "density")
                gd.density = (
                    _parse_float(dens_s) if len(dens_s)
                    > 0 else defaults.geom_density
                )

                # mass: explicit if provided, else compute from density * volume
                var ms_s = _extract_attr(tag, "mass")
                if len(ms_s) > 0:
                    gd.mass = _parse_float(ms_s)
                else:
                    # Compute mass = density * volume based on geom type and size
                    var PI: Float64 = 3.14159265358979323846
                    var vol: Float64 = 0.0
                    if gd.geom_type == _GEOM_SPHERE:
                        vol = (
                            (Float64(4.0) / Float64(3.0))
                            * PI
                            * gd.radius
                            * gd.radius
                            * gd.radius
                        )
                    elif gd.geom_type == _GEOM_CAPSULE:
                        var cyl_vol = (
                            PI
                            * gd.radius
                            * gd.radius
                            * (Float64(2.0) * gd.half_length)
                        )
                        var sph_vol = (
                            (Float64(4.0) / Float64(3.0))
                            * PI
                            * gd.radius
                            * gd.radius
                            * gd.radius
                        )
                        vol = cyl_vol + sph_vol
                    elif gd.geom_type == _GEOM_BOX:
                        vol = Float64(8.0) * gd.half_x * gd.half_y * gd.half_z
                    elif gd.geom_type == _GEOM_CYLINDER:
                        vol = (
                            PI
                            * gd.radius
                            * gd.radius
                            * (Float64(2.0) * gd.half_length)
                        )
                    # PLANE has no volume → mass stays 0
                    if vol > Float64(0):
                        gd.mass = gd.density * vol
                    else:
                        gd.mass = Float64(-1)

                # rgba colour: per-geom > default > GeomData fallback (0.7 grey)
                var rgba_s = _extract_attr(tag, "rgba")
                if len(rgba_s) > 0:
                    var cv = _parse_rgba4(rgba_s)
                    gd.rgba_r = cv[0]
                    gd.rgba_g = cv[1]
                    gd.rgba_b = cv[2]
                    gd.rgba_a = cv[3]
                elif defaults.geom_rgba_r >= Float64(0):
                    gd.rgba_r = defaults.geom_rgba_r
                    gd.rgba_g = defaults.geom_rgba_g
                    gd.rgba_b = defaults.geom_rgba_b
                    gd.rgba_a = defaults.geom_rgba_a

                # material reference — stored as index into materials[]
                # (index resolved by caller if needed; stored as -1 when absent)
                # We store the raw name match here via a linear scan of asset_sec
                # NOTE: asset_sec is not available in _fill_model; material_id
                # is resolved in a post-pass inside parse_xml_full.

                result.geoms[geom_count] = gd
            geom_count += 1
            var tag_end = worldbody.find(">", next_geom)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen


# =============================================================================
# Phase 5: Parse <actuator> section
# =============================================================================


fn _fill_actuators[
    NBODY: Int,
    NJOINT: Int,
    NQ: Int,
    NV: Int,
    NGEOM: Int,
    NACT: Int,
    NTEX: Int,
    NMAT: Int,
    NLIGHT: Int,
    NCAM: Int,
    NSITE: Int,
](
    actuator_sec: String,
    worldbody: String,
    defaults: DefaultsData,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE
    ],
):
    """Parse <actuator> section and populate result.actuators[]."""
    var act_count = 0
    var scan_pos = 0
    var alen = len(actuator_sec)

    while scan_pos < alen and act_count < NACT:
        # Find next actuator tag: motor, position, velocity, general
        var nm = actuator_sec.find("<motor", scan_pos)
        var np_ = actuator_sec.find("<position", scan_pos)
        var nv_ = actuator_sec.find("<velocity", scan_pos)
        var ng = actuator_sec.find("<general", scan_pos)

        var earliest = _min_valid(_min_valid(nm, np_), _min_valid(nv_, ng))
        if earliest == -1:
            break

        var tag = _extract_opening_tag(actuator_sec, earliest)

        var ad = ActuatorData()

        # gear
        var gear_s = _extract_attr(tag, "gear")
        if len(gear_s) > 0:
            ad.gear = _parse_float(gear_s)

        # joint name → joint index
        var jname = _extract_attr(tag, "joint")
        if len(jname) > 0:
            ad.joint_id = _find_joint_index_by_name(worldbody, jname)

        # ctrlrange / ctrllimited
        var cr_s = _extract_attr(tag, "ctrlrange")
        if len(cr_s) > 0:
            var cv = _parse_vec3(cr_s)
            ad.ctrl_min = cv[0]
            ad.ctrl_max = cv[1]
            ad.is_ctrl_limited = True
        else:
            ad.ctrl_min = defaults.motor_ctrl_min
            ad.ctrl_max = defaults.motor_ctrl_max
            ad.is_ctrl_limited = defaults.motor_ctrl_limited

        var cl_s = _extract_attr(tag, "ctrllimited")
        if cl_s == "true":
            ad.is_ctrl_limited = True
        elif cl_s == "false":
            ad.is_ctrl_limited = False

        result.actuators[act_count] = ad
        act_count += 1

        var tag_end = actuator_sec.find(">", earliest)
        scan_pos = tag_end + 1 if tag_end != -1 else alen


# =============================================================================
# Phase 6: Resolve geom material references (post-pass)
# =============================================================================


fn _find_material_index_by_name(asset_sec: String, name: String) -> Int:
    """Return 0-based index of <material name="name"> in asset_sec, or -1."""
    var search = 'name="' + name + '"'
    var scan_pos = 0
    var count = 0
    while True:
        var t = asset_sec.find("<material", scan_pos)
        if t == -1:
            break
        var tag_end = asset_sec.find(">", t)
        if tag_end == -1:
            break
        var tag = String(asset_sec[t : tag_end + 1])
        if tag.find(search) != -1:
            return count
        count += 1
        scan_pos = tag_end + 1
    return -1


fn _resolve_geom_materials[
    NBODY: Int,
    NJOINT: Int,
    NQ: Int,
    NV: Int,
    NGEOM: Int,
    NACT: Int,
    NTEX: Int,
    NMAT: Int,
    NLIGHT: Int,
    NCAM: Int,
    NSITE: Int,
](
    worldbody: String,
    asset_sec: String,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE
    ],
):
    """Resolve material="name" on geoms → material index; copy material rgba."""
    var scan_pos = 0
    var geom_idx = 0
    var wlen = len(worldbody)

    while scan_pos < wlen and geom_idx < NGEOM:
        var t = worldbody.find("<geom", scan_pos)
        if t == -1:
            break
        var tag_end = worldbody.find(">", t)
        if tag_end == -1:
            break
        var tag = String(worldbody[t : tag_end + 1])
        var mat_name = _extract_attr(tag, "material")
        if len(mat_name) > 0:
            var mid = _find_material_index_by_name(asset_sec, mat_name)
            result.geoms[geom_idx].material_id = mid
            # Only inherit material rgba when the geom has no explicit rgba attr
            var has_explicit_rgba = len(_extract_attr(tag, "rgba")) > 0
            if not has_explicit_rgba and mid >= 0 and mid < NMAT:
                var md = result.materials[mid]
                result.geoms[geom_idx].rgba_r = md.rgba_r
                result.geoms[geom_idx].rgba_g = md.rgba_g
                result.geoms[geom_idx].rgba_b = md.rgba_b
                result.geoms[geom_idx].rgba_a = md.rgba_a
        geom_idx += 1
        scan_pos = tag_end + 1


# =============================================================================
# Main entry point
# =============================================================================


fn parse_xml_full[
    NBODY: Int,
    NJOINT: Int,
    NQ: Int,
    NV: Int,
    NGEOM: Int,
    NACT: Int,
    NTEX: Int = 0,
    NMAT: Int = 0,
    NLIGHT: Int = 0,
    NCAM: Int = 0,
    NSITE: Int = 0,
](xml: String) -> FlatModelDef[
    NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE
]:
    """Full MJCF parse: returns a populated FlatModelDef.

    Caller must obtain dimensions via parse_xml() first:

        comptime pm  = parse_xml(xml)
        comptime fmd = parse_xml_full[
            pm.NBODY, pm.NJOINT, pm.NQ, pm.NV, pm.NGEOM, pm.NACT,
            pm.NTEX, pm.NMAT, pm.NLIGHT, pm.NCAM, pm.NSITE,
        ](xml)

    The NTEX/NMAT/NLIGHT/NCAM/NSITE parameters default to 0 for backward
    compatibility — existing callers omitting them get no visual element arrays.
    All operations are comptime-safe (String.find + slice arithmetic only).
    """
    var result = FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE
    ]()

    # Extract top-level sections
    var worldbody = _extract_section(xml, "worldbody")
    var actuator_sec = _extract_section(xml, "actuator")
    var asset_sec = _extract_section(xml, "asset")

    # Global physics options
    var opt = _parse_option(xml)
    result.gravity_x = opt[0]
    result.gravity_y = opt[1]
    result.gravity_z = opt[2]
    result.timestep = opt[3]

    # Defaults (applied when specific attrs are absent)
    var defaults = _parse_defaults(xml)

    # Compiler angle units: detect degree mode and compute conversion factor
    var deg_factor = Float64(1.0)
    var compiler_t = xml.find("<compiler")
    if compiler_t != -1:
        var compiler_end = xml.find(">", compiler_t)
        if compiler_end != -1:
            var ctag = String(xml[compiler_t : compiler_end + 1])
            var angle_val = _extract_attr(ctag, "angle")
            if _trim(angle_val) == "degree":
                deg_factor = Float64(3.141592653589793) / Float64(180.0)

    # Assets: textures and materials
    _fill_assets[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE
    ](asset_sec, result)

    # Single DFS pass: bodies + joints + geoms + lights + cameras + sites
    _fill_model[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE
    ](worldbody, defaults, result, deg_factor)

    # Actuators
    _fill_actuators[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE
    ](actuator_sec, worldbody, defaults, result)

    # Post-pass: resolve geom material="name" references
    _resolve_geom_materials[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE
    ](worldbody, asset_sec, result)

    return result^
