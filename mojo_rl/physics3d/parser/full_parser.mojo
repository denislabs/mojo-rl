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
    _split_spaces,
    _extract_section,
    _extract_section_inner,
    _extract_opening_tag,
    _extract_attr,
    _trim,
    _parse_float,
    _parse_int_str,
    _parse_vec3,
    _parse_quat,
    _parse_axisangle_to_quat,
    _parse_euler_to_quat,
    _parse_zaxis_to_quat,
    _compiler_deg_factor,
    _fromto_to_pos_quat,
    _find_joint_index_by_name,
    _find_body_index_by_name,
    _find_site_index_by_name,
    _sqrt_f64,
)
from .flat_model import (
    BodyData,
    JointData,
    GeomData,
    ActuatorData,
    ACT_KIND_POSITION,
    ACT_KIND_VELOCITY,
    ACT_KIND_GENERAL,
    TextureData,
    MaterialData,
    LightData,
    CameraData,
    SiteData,
    DefaultsData,
    EqualityData,
    ExcludeData,
    TendonData,
    _TENDON_KIND_FIXED,
    _TENDON_KIND_SPATIAL,
    NamedDefaultsList,
    FlatModelDef,
    _EQ_CONNECT,
    _EQ_WELD,
    _GEOM_PLANE,
    _GEOM_SPHERE,
    _GEOM_CAPSULE,
    _GEOM_BOX,
    _GEOM_CYLINDER,
    _GEOM_MESH,
    _GEOM_ELLIPSOID,
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
from mojo_rl.physics3d.joint_types import (
    JNT_HINGE,
    JNT_SLIDE,
    JNT_BALL,
    JNT_FREE,
)


# =============================================================================
# Internal: min of two ints treating -1 as +∞
# =============================================================================


def _min_valid(a: Int, b: Int) -> Int:
    """Return the smaller of a and b, treating -1 as +infinity."""
    if a == -1:
        return b
    if b == -1:
        return a
    if a < b:
        return a
    return b


def _option_flag_disabled(xml: String, flag: String) -> Bool:
    """True when `<option><flag NAME="disable"/></option>` is present.

    MJCF puts `<flag>` INSIDE `<option>`, so this looks in the option section
    rather than the option opening tag. Only the flags the engine can honour
    are consulted — see `parse_xml_full` for how each is applied.
    """
    var opt_sec = _extract_section(xml, "option")
    if opt_sec.byte_length() == 0:
        return False
    var ft = opt_sec.find("<flag")
    while ft != -1:
        var tag = _extract_opening_tag(opt_sec, ft)
        if _trim(_extract_attr(tag, flag)) == "disable":
            return True
        ft = opt_sec.find("<flag", ft + 5)
    return False


# =============================================================================
# Phase 1: Parse <option> — gravity + timestep
# =============================================================================


def _parse_option(xml: String) -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
    """Extract (gravity_x, gravity_y, gravity_z, timestep, density, viscosity) from <option .../>.

    Defaults: gravity=(0,0,-9.81), timestep=0.002, density=0.0, viscosity=0.0.

    ⚠ The timestep default was 0.01 until 2026-07-31 — 5x MuJoCo's actual
    default (mjOption.timestep = 0.002) and, worse, 5x what the OTHER parser
    uses: `xml_parser.parse_xml` has always defaulted to 0.002, and that is
    what `ModelDefFromXML.TIMESTEP` and therefore `Phyics3dEnvConfig.
    get_timestep()` report. So a model with no `<option timestep>` STEPPED at
    0.01 while every consumer was told 0.002.

    Invisible until dm_control's ball_in_cup, which is the first ported model
    that omits `<option>` entirely — every other suite domain states its
    timestep, and the Gym-derived models do too. It showed up as a ball
    falling 0.054 m in the time MuJoCo fell 0.0022, i.e. exactly the (0.01 /
    0.002)^2 = 25x an integrator error of this shape produces. Two parsers,
    two defaults: see feedback_physics3d_two_parser_paths.
    """
    var gx = Float64(0)
    var gy = Float64(0)
    var gz = Float64(-9.81)
    var ts = Float64(0.002)
    var dens = Float64(0)
    var visc = Float64(0)

    var pos = xml.find("<option")
    if pos == -1:
        return (gx, gy, gz, ts, dens, visc)

    var tag = _extract_opening_tag(xml, pos)

    var gravity_str = _extract_attr(tag, "gravity")
    if gravity_str.byte_length() > 0:
        var gvec = _parse_vec3(gravity_str)
        gx = gvec[0]
        gy = gvec[1]
        gz = gvec[2]

    var ts_str = _extract_attr(tag, "timestep")
    if ts_str.byte_length() > 0:
        ts = _parse_float(ts_str)

    var dens_str = _extract_attr(tag, "density")
    if dens_str.byte_length() > 0:
        dens = _parse_float(dens_str)

    var visc_str = _extract_attr(tag, "viscosity")
    if visc_str.byte_length() > 0:
        visc = _parse_float(visc_str)

    return (gx, gy, gz, ts, dens, visc)


# =============================================================================
# Phase 2: Parse <default> block
# =============================================================================


def _parse_one_default_block(defaults_sec: String, parent: DefaultsData) -> DefaultsData:
    """Parse joint/geom/motor attrs from a default section, inheriting from parent."""
    var d = parent  # start with parent defaults

    # Find default <joint
    var jpos = defaults_sec.find("<joint")
    if jpos != -1:
        var jtag = _extract_opening_tag(defaults_sec, jpos)

        var arm_s = _extract_attr(jtag, "armature")
        if arm_s.byte_length() > 0:
            d.joint_armature = _parse_float(arm_s)

        var damp_s = _extract_attr(jtag, "damping")
        if damp_s.byte_length() > 0:
            d.joint_damping = _parse_float(damp_s)

        var stiff_s = _extract_attr(jtag, "stiffness")
        if stiff_s.byte_length() > 0:
            d.joint_stiffness = _parse_float(stiff_s)

        var lim_s = _extract_attr(jtag, "limited")
        if lim_s == "true":
            d.joint_limited = True
        elif lim_s == "false":
            d.joint_limited = False

        var fl_s = _extract_attr(jtag, "frictionloss")
        if fl_s.byte_length() > 0:
            d.joint_frictionloss = _parse_float(fl_s)

        var sr_s = _extract_attr(jtag, "springref")
        if sr_s.byte_length() > 0:
            d.joint_springref = _parse_float(sr_s)

        var srl_s = _extract_attr(jtag, "solreflimit")
        if srl_s.byte_length() > 0:
            var sv = _parse_vec3(srl_s)
            d.joint_solref_limit_0 = sv[0]
            d.joint_solref_limit_1 = sv[1]

        var sil_s = _extract_attr(jtag, "solimplimit")
        if sil_s.byte_length() > 0:
            var parts = List[String]()

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

        # Structural attrs — stored raw; only overwrite when this block
        # actually sets them, so a parent class's value survives.
        var jt_s = _extract_attr(jtag, "type")
        if jt_s.byte_length() > 0:
            d.joint_type_s = jt_s
        var jax_s = _extract_attr(jtag, "axis")
        if jax_s.byte_length() > 0:
            d.joint_axis_s = jax_s
        var jrng_s = _extract_attr(jtag, "range")
        if jrng_s.byte_length() > 0:
            d.joint_range_s = jrng_s
        var jp_s = _extract_attr(jtag, "pos")
        if jp_s.byte_length() > 0:
            d.joint_pos_s = jp_s

    # Find default <geom
    var gpos = defaults_sec.find("<geom")
    if gpos != -1:
        var gtag = _extract_opening_tag(defaults_sec, gpos)

        var dens_s = _extract_attr(gtag, "density")
        if dens_s.byte_length() > 0:
            d.geom_density = _parse_float(dens_s)

        var fric_s = _extract_attr(gtag, "friction")
        if fric_s.byte_length() > 0:
            var fvec = _parse_vec3(fric_s)
            d.geom_friction = fvec[0]
            d.geom_friction_spin = fvec[1]
            d.geom_friction_roll = fvec[2]

        var ct_s = _extract_attr(gtag, "contype")
        if ct_s.byte_length() > 0:
            d.geom_contype = _parse_int_str(ct_s)

        var ca_s = _extract_attr(gtag, "conaffinity")
        if ca_s.byte_length() > 0:
            d.geom_conaffinity = _parse_int_str(ca_s)

        var cd_s = _extract_attr(gtag, "condim")
        if cd_s.byte_length() > 0:
            d.geom_condim = _parse_int_str(cd_s)

        var sr0_s = _extract_attr(gtag, "solref")
        if sr0_s.byte_length() > 0:
            var sv = _parse_vec3(sr0_s)
            d.geom_solref_0 = sv[0]
            d.geom_solref_1 = sv[1]

        var si0_s = _extract_attr(gtag, "solimp")
        if si0_s.byte_length() > 0:
            var parts = List[String]()

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
        if mg_s.byte_length() > 0:
            d.geom_margin = _parse_float(mg_s)

        var rgba_s = _extract_attr(gtag, "rgba")
        if rgba_s.byte_length() > 0:
            var cv = _parse_rgba4(rgba_s)
            d.geom_rgba_r = cv[0]
            d.geom_rgba_g = cv[1]
            d.geom_rgba_b = cv[2]
            d.geom_rgba_a = cv[3]

        # Structural attrs — stored raw (see DefaultsData docstring).
        var gt_s = _extract_attr(gtag, "type")
        if gt_s.byte_length() > 0:
            d.geom_type_s = gt_s
        var gft_s = _extract_attr(gtag, "fromto")
        if gft_s.byte_length() > 0:
            d.geom_fromto_s = gft_s
        var gsz_s = _extract_attr(gtag, "size")
        if gsz_s.byte_length() > 0:
            d.geom_size_s = gsz_s
        var gm_s = _extract_attr(gtag, "mass")
        if gm_s.byte_length() > 0:
            d.geom_mass_s = gm_s
        var gmat_s = _extract_attr(gtag, "material")
        if gmat_s.byte_length() > 0:
            d.geom_material_s = gmat_s
        var gp_s = _extract_attr(gtag, "pos")
        if gp_s.byte_length() > 0:
            d.geom_pos_s = gp_s
        var gq_s = _extract_attr(gtag, "quat")
        if gq_s.byte_length() > 0:
            d.geom_quat_s = gq_s
        var gg_s = _extract_attr(gtag, "group")
        if gg_s.byte_length() > 0:
            d.geom_group_s = gg_s

    # Find default <site  (structural attrs only — the touch sensor's zone)
    var spos = defaults_sec.find("<site")
    if spos != -1:
        var stag = _extract_opening_tag(defaults_sec, spos)
        var st_s = _extract_attr(stag, "type")
        if st_s.byte_length() > 0:
            d.site_type_s = st_s
        var ss_s = _extract_attr(stag, "size")
        if ss_s.byte_length() > 0:
            d.site_size_s = ss_s

    # Find default <motor
    var mpos = defaults_sec.find("<motor")
    if mpos != -1:
        var mtag = _extract_opening_tag(defaults_sec, mpos)

        var cl_s = _extract_attr(mtag, "ctrllimited")
        if cl_s == "true":
            d.motor_ctrl_limited = True

        var cr_s = _extract_attr(mtag, "ctrlrange")
        if cr_s.byte_length() > 0:
            var cvec = _parse_vec3(cr_s)
            d.motor_ctrl_min = cvec[0]
            d.motor_ctrl_max = cvec[1]

        # `gear` was missing here (and in the comptime twin) until 2026-07-29,
        # so a default-class gear silently actuated at 1.0. dm_control's
        # point_mass declares `<motor gear=".1"/>` this way — a 10x error.
        var mg_s = _extract_attr(mtag, "gear")
        if mg_s.byte_length() > 0:
            d.motor_gear = _parse_float(mg_s)

    return d


def _strip_nested_defaults(sec: String) -> String:
    """Remove nested `<default class="...">...</default>` sub-blocks.

    `_extract_section(xml, "default")` hands back the inner text of the outer
    `<default>` element, which still contains the named class blocks. Feeding
    that straight to `_parse_one_default_block` makes the FIRST nested class's
    `<joint>`/`<geom>` masquerade as the top-level default — cartpole's cart
    geom inherited the pole class's `fromto`, putting the cart's CoM at
    z=0.5 instead of 0.

    Harmless while only tuning attributes were inherited; a real corruption
    once structural attributes are (2026-07-29). Nesting is tracked so that
    classes containing their own sub-classes are removed whole.
    """
    var out = String("")
    var i = 0
    var n = sec.byte_length()
    while i < n:
        var open_t = sec.find("<default", i)
        if open_t == -1:
            out += String(sec[byte=i:n])
            break
        out += String(sec[byte=i:open_t])
        # Walk forward to this block's matching </default>.
        var depth_ = 0
        var j = open_t
        while j < n:
            var next_open = sec.find("<default", j + 1)
            var next_close = sec.find("</default>", j + 1)
            if next_close == -1:
                j = n
                break
            if next_open != -1 and next_open < next_close:
                depth_ += 1
                j = next_open
                continue
            if depth_ == 0:
                j = next_close + 10  # len("</default>")
                break
            depth_ -= 1
            j = next_close
        i = j
    return out


def _parse_defaults(
    xml: String,
) -> Tuple[DefaultsData, NamedDefaultsList]:
    """Extract default joint/geom/motor attrs from the <default> section.

    Returns (top_level_defaults, named_defaults_list).
    Named defaults inherit from the top-level defaults and override specific attrs.
    """
    var defaults_sec = _extract_section(xml, "default")
    if defaults_sec.byte_length() == 0:
        return (DefaultsData(), NamedDefaultsList())

    # Parse top-level (unnamed) defaults. `_extract_section` KEEPS the outer
    # <default>...</default> tags, so take the inner text first (that helper
    # handles the nested same-name tags) and only then strip the class
    # sub-blocks — otherwise the strip would swallow the whole section.
    var top = _parse_one_default_block(
        _strip_nested_defaults(_extract_section_inner(xml, "default")),
        DefaultsData(),
    )

    # Parse named <default class="..."> sub-blocks, recursively, each one
    # inheriting from the block that ENCLOSES it.
    var named = NamedDefaultsList()
    _collect_named_defaults(
        _extract_section_inner(xml, "default"), top, named
    )
    return (top, named)


def _find_matching_default_close(sec: String, open_pos: Int) -> Int:
    """Index of the `</default>` matching the `<default` at `open_pos`.

    Returns -1 if unbalanced. Depth-tracked, because `<default>` blocks nest.
    """
    var n = sec.byte_length()
    var depth = 0
    var i = open_pos
    while i < n:
        var next_open = sec.find("<default", i + 1)
        var next_close = sec.find("</default>", i + 1)
        if next_close == -1:
            return -1
        if next_open != -1 and next_open < next_close:
            depth += 1
            i = next_open
            continue
        if depth == 0:
            return next_close
        depth -= 1
        i = next_close
    return -1


def _collect_named_defaults(
    inner: String,
    parent: DefaultsData,
    mut named: NamedDefaultsList,
):
    """Register every `<default class="...">` in `inner`, depth-first.

    `inner` is the INNER text of an enclosing `<default>` block and `parent`
    is that block's resolved defaults. Each direct child class inherits from
    `parent`, is registered, and is then recursed into so ITS children inherit
    from IT.

    This replaces a loop that took `defaults_sec.find("</default>")` — the
    FIRST close tag, not the matching one. With a flat `<default>` section
    that is the same thing, which is why it survived every domain up to
    humanoid. humanoid nests three deep:

        <default class="body">          <- worked (its <joint> is the first
          <joint armature=".01" .../>      one inside its span)
          <default class="big_joint">   <- NEVER REGISTERED: its opening tag
            <joint damping="5" .../>       sits inside the span the parent
            <default class="big_stiff_joint">   consumed, so the scan skipped
              <joint stiffness="20"/>          straight past it
            </default>
          </default>
        </default>

    so `named.find("big_joint")` returned an EMPTY DefaultsData and every
    joint naming a nested class silently got armature/damping/stiffness 0 —
    a humanoid with no hip or abdomen springs at all, which still simulates
    and still looks like a humanoid.

    Inheriting from `parent` rather than the top level is the second half of
    the fix: `big_stiff_joint` sets only `stiffness`, and must pick up
    `damping="5"` from `big_joint` and `armature=".01"` from `body`.

    Breadth-first over an explicit worklist rather than recursively: the
    natural recursive spelling is correct (`child_inner` shrinks every step)
    but Mojo flags the self-call with "will cause an infinite loop", and a
    warning on every model build is not worth the two saved lines.
    """
    var pending_text = List[String]()
    var pending_parent = List[DefaultsData]()
    pending_text.append(inner)
    pending_parent.append(parent)

    var q = 0
    while q < len(pending_text):
        var text = pending_text[q]
        var par = pending_parent[q]
        q += 1

        var n = text.byte_length()
        var scan = 0
        while scan < n:
            var dt = text.find("<default", scan)
            if dt == -1:
                break
            var close = _find_matching_default_close(text, dt)
            if close == -1:
                break
            var tag_end = text.find(">", dt)
            if tag_end == -1 or tag_end > close:
                break

            var tag = _extract_opening_tag(text, dt)
            var class_name = _extract_attr(tag, "class")
            var child_inner = String(text[byte = tag_end + 1 : close])

            var child_defaults = par
            if class_name.byte_length() > 0:
                # Own attributes only — strip the grandchildren first, or the
                # first grandchild's <joint>/<geom> masquerades as this
                # class's.
                child_defaults = _parse_one_default_block(
                    _strip_nested_defaults(child_inner), par
                )
                named.add(class_name, child_defaults)

            pending_text.append(child_inner)
            pending_parent.append(child_defaults)
            scan = close + 10  # len("</default>")


# =============================================================================
# Phase 3: Parse geom type string → Int constant
# =============================================================================


def _geom_type_from_str(s: String) -> Int:
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
    elif t == "mesh":
        return _GEOM_MESH
    elif t == "ellipsoid":
        return _GEOM_ELLIPSOID
    # ⚠ THE DEFAULT IS A SILENT SUBSTITUTION, not an error. `ellipsoid` used
    # to land here, which cost fish its whole mass distribution (bug 26).
    # Anything still falling through — `hfield`, `sdf` — is modelled as a
    # sphere of radius size[0] with no diagnostic.
    return _GEOM_SPHERE  # default


# =============================================================================
# Phase 4a: Parse <asset> section — textures and materials
# =============================================================================


def _tex_type_from_str(s: String) -> Int:
    var t = _trim(s)
    if t == "skybox":
        return TEX_SKYBOX
    elif t == "cube":
        return TEX_CUBE
    return TEX_2D  # default


def _tex_builtin_from_str(s: String) -> Int:
    var t = _trim(s)
    if t == "gradient":
        return TEX_BUILTIN_GRADIENT
    elif t == "checker":
        return TEX_BUILTIN_CHECKER
    elif t == "flat":
        return TEX_BUILTIN_FLAT
    return TEX_BUILTIN_NONE


def _tex_mark_from_str(s: String) -> Int:
    var t = _trim(s)
    if t == "edge":
        return TEX_MARK_EDGE
    elif t == "cross":
        return TEX_MARK_CROSS
    elif t == "random":
        return TEX_MARK_RANDOM
    return TEX_MARK_NONE


def _light_mode_from_str(s: String) -> Int:
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


def _cam_mode_from_str(s: String) -> Int:
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


def _find_texture_index_by_name(asset_sec: String, name: String) -> Int:
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
        var tag = String(asset_sec[byte = t : tag_end + 1])
        if tag.find(search) != -1:
            return count
        count += 1
        scan_pos = tag_end + 1
    return -1


def _parse_rgb3(s: String) -> Tuple[Float64, Float64, Float64]:
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


def _parse_rgba4(s: String) -> Tuple[Float64, Float64, Float64, Float64]:
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


def _orientation_to_quat(
    quat_s: String,
    axisangle_s: String,
    xyaxes_s: String,
    zaxis_s: String,
    euler_s: String,
    deg_factor: Float64,
    eulerseq: String,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Resolve MJCF's orientation attributes to a quaternion (qx, qy, qz, qw).

    Mirrors `ResolveOrientation` in MuJoCo's `user_objects.cc`: `quat` wins,
    otherwise at most one "alternative" attribute may be set. Returns identity
    when none is present.

    Callers pass already-resolved strings so each site can apply its own
    default-class fallback before calling in.
    """
    if quat_s.byte_length() > 0:
        return _parse_quat(quat_s)
    if axisangle_s.byte_length() > 0:
        return _parse_axisangle_to_quat(axisangle_s, deg_factor)
    if xyaxes_s.byte_length() > 0:
        return _xyaxes_to_quat(xyaxes_s)
    if zaxis_s.byte_length() > 0:
        return _parse_zaxis_to_quat(zaxis_s)
    if euler_s.byte_length() > 0:
        return _parse_euler_to_quat(euler_s, deg_factor, eulerseq)
    return (Float64(0), Float64(0), Float64(0), Float64(1))


def _xyaxes_to_quat(s: String) -> Tuple[Float64, Float64, Float64, Float64]:
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


def _fill_assets[
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
    NEQ: Int = 0,
    NEXCLUDE: Int = 0,
    NTENDON: Int = 0,
](
    asset_sec: String,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
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
        var tag = String(asset_sec[byte = t : tag_end + 1])

        var td = TextureData()

        var type_s = _extract_attr(tag, "type")
        td.tex_type = _tex_type_from_str(type_s)

        var builtin_s = _extract_attr(tag, "builtin")
        td.builtin = _tex_builtin_from_str(builtin_s)

        var mark_s = _extract_attr(tag, "mark")
        td.mark = _tex_mark_from_str(mark_s)

        var rgb1_s = _extract_attr(tag, "rgb1")
        if rgb1_s.byte_length() > 0:
            var c = _parse_rgb3(rgb1_s)
            td.rgb1_r = c[0]
            td.rgb1_g = c[1]
            td.rgb1_b = c[2]

        var rgb2_s = _extract_attr(tag, "rgb2")
        if rgb2_s.byte_length() > 0:
            var c = _parse_rgb3(rgb2_s)
            td.rgb2_r = c[0]
            td.rgb2_g = c[1]
            td.rgb2_b = c[2]

        var markrgb_s = _extract_attr(tag, "markrgb")
        if markrgb_s.byte_length() > 0:
            var c = _parse_rgb3(markrgb_s)
            td.markrgb_r = c[0]
            td.markrgb_g = c[1]
            td.markrgb_b = c[2]

        var w_s = _extract_attr(tag, "width")
        if w_s.byte_length() > 0:
            td.width = _parse_int_str(w_s)

        var h_s = _extract_attr(tag, "height")
        if h_s.byte_length() > 0:
            td.height = _parse_int_str(h_s)

        var rand_s = _extract_attr(tag, "random")
        if rand_s.byte_length() > 0:
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
        var tag = String(asset_sec[byte = t : tag_end + 1])

        var md = MaterialData()

        # texture reference → index
        var tex_name = _extract_attr(tag, "texture")
        if tex_name.byte_length() > 0:
            md.tex_id = _find_texture_index_by_name(asset_sec, tex_name)

        var rgba_s = _extract_attr(tag, "rgba")
        if rgba_s.byte_length() > 0:
            var c = _parse_rgba4(rgba_s)
            md.rgba_r = c[0]
            md.rgba_g = c[1]
            md.rgba_b = c[2]
            md.rgba_a = c[3]

        var shin_s = _extract_attr(tag, "shininess")
        if shin_s.byte_length() > 0:
            md.shininess = _parse_float(shin_s)

        var spec_s = _extract_attr(tag, "specular")
        if spec_s.byte_length() > 0:
            md.specular = _parse_float(spec_s)

        var refl_s = _extract_attr(tag, "reflectance")
        if refl_s.byte_length() > 0:
            md.reflectance = _parse_float(refl_s)

        var tr_s = _extract_attr(tag, "texrepeat")
        if tr_s.byte_length() > 0:
            var tv = _parse_vec3(tr_s)
            md.texrepeat_u = tv[0]
            md.texrepeat_v = tv[1]

        var tu_s = _extract_attr(tag, "texuniform")
        if tu_s == "true":
            md.texuniform = True

        result.materials[mat_count] = md
        mat_count += 1
        mat_pos = tag_end + 1

    # ---- Mesh assets ----------------------------------------------------------
    var mesh_pos = 0
    var mesh_count = 0
    while mesh_count < 16:
        var t = asset_sec.find("<mesh", mesh_pos)
        if t == -1:
            break
        var tag_end = asset_sec.find(">", t)
        if tag_end == -1:
            break
        # Skip if this is a self-closing tag for non-mesh elements
        var tag = String(asset_sec[byte = t : tag_end + 1])
        var mesh_name = _extract_attr(tag, "name")
        var mesh_file = _extract_attr(tag, "file")
        if mesh_name.byte_length() > 0 and mesh_file.byte_length() > 0:
            result.mesh_asset_names[mesh_count] = mesh_name
            result.mesh_asset_files[mesh_count] = mesh_file
            mesh_count += 1
        mesh_pos = tag_end + 1
    result.num_mesh_assets = mesh_count


# =============================================================================
# Phase 4b: Combined DFS scan — fills bodies, joints, geoms in one pass
# =============================================================================


def _fill_model[
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
    NEQ: Int = 0,
    NEXCLUDE: Int = 0,
    NTENDON: Int = 0,
](
    worldbody: String,
    defaults: DefaultsData,
    named_defaults: NamedDefaultsList,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ],
    deg_factor: Float64 = 1.0,
    eulerseq: String = "xyz",
):
    """Single-pass DFS over worldbody XML to populate bodies, joints, geoms,
    lights, cameras, and sites.

    deg_factor: 1.0 for radian models, pi/180 for degree models.
    Applied to joint range values and axisangle/euler rotation angles.
    eulerseq: `<compiler eulerseq="...">`, the axis order for `euler=`.

    Uses two-pointer scan: tracks `<body` and `</body>` to maintain depth/parent.
    Joints, geoms, lights, cameras, and sites encountered at each depth are
    assigned to the currently-open body.
    """
    # body_id_stack[depth] = body index at current depth
    # depth 0 = worldbody level (body_id=0)
    var body_id_stack = InlineArray[Int, NBODY + 1](fill=0)
    # childclass_stack[depth] = default class inherited by elements at this
    # depth. MJCF's `childclass` applies to every descendant of the body that
    # declares it, until a deeper body overrides it; an element's own
    # `class=` still wins. Empty string = no inherited class.
    var childclass_stack = InlineArray[String, NBODY + 1](fill=String(""))
    var depth = 0
    var body_count = 0  # bodies[0..NBODY-2] → model body indices 1..NBODY-1
    var joint_count = 0
    var geom_count = 0
    var light_count = 0
    var cam_count = 0
    var site_count = 0

    var scan_pos = 0
    var wlen = worldbody.byte_length()

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
            var inherited_class = childclass_stack[depth]
            depth += 1
            var this_body_id = body_count + 1  # model body index (worldbody=0)
            body_id_stack[depth] = this_body_id
            # `childclass` on this body replaces the inherited one for the
            # whole subtree; otherwise the parent's carries down.
            var cc_s = _extract_attr(tag, "childclass")
            childclass_stack[depth] = (
                cc_s if cc_s.byte_length() > 0 else inherited_class
            )

            if body_count < NBODY - 1:
                var b = BodyData()
                b.parent = parent_id

                # pos
                var pos_s = _extract_attr(tag, "pos")
                if pos_s.byte_length() > 0:
                    var pv = _parse_vec3(pos_s)
                    b.pos_x = pv[0]
                    b.pos_y = pv[1]
                    b.pos_z = pv[2]

                # orientation: quat > axisangle > xyaxes > zaxis > euler
                var bq = _orientation_to_quat(
                    _extract_attr(tag, "quat"),
                    _extract_attr(tag, "axisangle"),
                    _extract_attr(tag, "xyaxes"),
                    _extract_attr(tag, "zaxis"),
                    _extract_attr(tag, "euler"),
                    deg_factor,
                    eulerseq,
                )
                b.quat_x = bq[0]
                b.quat_y = bq[1]
                b.quat_z = bq[2]
                b.quat_w = bq[3]

                # inertial pos/quat (ipos, iquat)
                var ipos_s = _extract_attr(tag, "ipos")
                if ipos_s.byte_length() > 0:
                    var iv = _parse_vec3(ipos_s)
                    b.ipos_x = iv[0]
                    b.ipos_y = iv[1]
                    b.ipos_z = iv[2]

                var iquat_s = _extract_attr(tag, "iquat")
                if iquat_s.byte_length() > 0:
                    var iq = _parse_quat(iquat_s)
                    b.iquat_x = iq[0]
                    b.iquat_y = iq[1]
                    b.iquat_z = iq[2]
                    b.iquat_w = iq[3]

                # mass (may be absent — inertia computed from geoms)
                var mass_s = _extract_attr(tag, "mass")
                if mass_s.byte_length() > 0:
                    b.mass = _parse_float(mass_s)
                    b.has_explicit_inertia = True

                # diaginertia
                var di_s = _extract_attr(tag, "diaginertia")
                if di_s.byte_length() > 0:
                    var dv = _parse_vec3(di_s)
                    b.ixx = dv[0]
                    b.iyy = dv[1]
                    b.izz = dv[2]
                    b.has_explicit_inertia = True

                # mocap body flag
                var mocap_s = _extract_attr(tag, "mocap")
                if mocap_s == "true":
                    b.is_mocap = True

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

                # Effective defaults: the joint's own class="..." wins, else
                # the enclosing body's childclass, else the top-level block.
                # (Joints resolved NO class at all before 2026-07-29 — only
                # geoms did — so a class-defined joint silently became a
                # default hinge about the x axis.)
                var joint_class = _extract_attr(tag, "class")
                if joint_class.byte_length() == 0:
                    joint_class = childclass_stack[depth]
                var jdef = defaults
                if joint_class.byte_length() > 0:
                    jdef = named_defaults.find(joint_class)

                # type
                var type_s = _extract_attr(tag, "type")
                if type_s.byte_length() == 0:
                    type_s = jdef.joint_type_s
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
                if pos_s.byte_length() == 0:
                    pos_s = jdef.joint_pos_s
                if pos_s.byte_length() > 0:
                    var pv = _parse_vec3(pos_s)
                    jd.pos_x = pv[0]
                    jd.pos_y = pv[1]
                    jd.pos_z = pv[2]

                # axis (MuJoCo normalizes joint axes during compilation)
                var axis_s = _extract_attr(tag, "axis")
                if axis_s.byte_length() == 0:
                    axis_s = jdef.joint_axis_s
                if axis_s.byte_length() > 0:
                    var av = _parse_vec3(axis_s)
                    var ax = av[0]
                    var ay = av[1]
                    var az = av[2]
                    var ax_sq = ax*ax + ay*ay + az*az
                    # Normalize if not already unit length
                    var inv_len = Float64(1.0) / _sqrt_f64(ax_sq)
                    ax = ax * inv_len
                    ay = ay * inv_len
                    az = az * inv_len
                    jd.axis_x = ax
                    jd.axis_y = ay
                    jd.axis_z = az

                # range — deg→rad, but ONLY for angular joints. MuJoCo's
                # mjCJoint::Compile guards the conversion with
                # `type == mjJNT_HINGE || type == mjJNT_BALL`, because a SLIDE
                # range is in metres and must pass through untouched. Now that
                # degree is the default this matters: cartpole's
                # `<joint type="slide" range="-1.8 1.8">` would otherwise be
                # scaled to +-0.03 m and pin the cart at the origin.
                var range_s = _extract_attr(tag, "range")
                if range_s.byte_length() == 0:
                    range_s = jdef.joint_range_s
                if range_s.byte_length() > 0:
                    var angular = (
                        jd.jnt_type == JNT_HINGE or jd.jnt_type == JNT_BALL
                    )
                    var rf = deg_factor if angular else Float64(1.0)
                    var rv = _parse_vec3(range_s)
                    jd.range_min = rv[0] * rf
                    jd.range_max = rv[1] * rf
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
                if arm_s.byte_length() > 0:
                    jd.armature = _parse_float(arm_s)
                else:
                    jd.armature = jdef.joint_armature

                # damping
                var damp_s = _extract_attr(tag, "damping")
                if damp_s.byte_length() > 0:
                    jd.damping = _parse_float(damp_s)
                else:
                    jd.damping = jdef.joint_damping

                # stiffness
                var stiff_s = _extract_attr(tag, "stiffness")
                if stiff_s.byte_length() > 0:
                    jd.stiffness = _parse_float(stiff_s)
                else:
                    jd.stiffness = jdef.joint_stiffness

                # springref
                var sr_s = _extract_attr(tag, "springref")
                if sr_s.byte_length() > 0:
                    jd.springref = _parse_float(sr_s)
                else:
                    jd.springref = jdef.joint_springref

                # ref (MuJoCo joint reference position → qpos0). Same deg→rad
                # gate as `range` above — `ref` is an ANGLE for hinge/ball and
                # a LENGTH for slide. Without this, finger's `ref="-90"` became
                # -90 rad instead of -pi/2, which (per bug 18) silently skews
                # every constraint inverse weight since they are built at qpos0.
                var ref_s = _extract_attr(tag, "ref")
                if ref_s.byte_length() > 0:
                    var r_angular = (
                        jd.jnt_type == JNT_HINGE or jd.jnt_type == JNT_BALL
                    )
                    var rrf = deg_factor if r_angular else Float64(1.0)
                    jd.ref_val = _parse_float(ref_s) * rrf
                else:
                    jd.ref_val = 0.0

                # frictionloss
                var fl_s = _extract_attr(tag, "frictionloss")
                if fl_s.byte_length() > 0:
                    jd.frictionloss = _parse_float(fl_s)
                else:
                    jd.frictionloss = jdef.joint_frictionloss

                # `solreffriction` / `solimpfriction` set the dof-FRICTION
                # solver parameters, a DIFFERENT pair from the LIMIT ones
                # below — MuJoCo keeps them in dof_solref/dof_solimp, and a
                # model setting solimplimit leaves solimpfriction at the
                # default. `constraints/friction_dof.mojo` hardcodes MuJoCo's
                # defaults, exact for every model in the repo (none sets
                # these). Flag it here; `init_fields` raises, so the day one
                # does set them it is loud, not a silently wrong friction.
                jd.has_friction_solparams = (
                    _extract_attr(tag, "solreffriction").byte_length() > 0
                    or _extract_attr(tag, "solimpfriction").byte_length() > 0
                )

                # solreflimit (per-joint or default)
                var srl_s = _extract_attr(tag, "solreflimit")
                if srl_s.byte_length() > 0:
                    var sv = _parse_vec3(srl_s)
                    jd.solref_limit_0 = sv[0]
                    jd.solref_limit_1 = sv[1]
                else:
                    jd.solref_limit_0 = jdef.joint_solref_limit_0
                    jd.solref_limit_1 = jdef.joint_solref_limit_1

                # solimplimit (per-joint or default)
                var sil_s = _extract_attr(tag, "solimplimit")
                if sil_s.byte_length() > 0:
                    var parts2 = List[String]()

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
                    jd.solimp_limit_0 = jdef.joint_solimp_limit_0
                    jd.solimp_limit_1 = jdef.joint_solimp_limit_1
                    jd.solimp_limit_2 = jdef.joint_solimp_limit_2
                    jd.solimp_limit_3 = jdef.joint_solimp_limit_3
                    jd.solimp_limit_4 = jdef.joint_solimp_limit_4

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
                if pos_s.byte_length() > 0:
                    var pv = _parse_vec3(pos_s)
                    ld.pos_x = pv[0]
                    ld.pos_y = pv[1]
                    ld.pos_z = pv[2]

                var dir_s = _extract_attr(tag, "dir")
                if dir_s.byte_length() > 0:
                    var dv = _parse_vec3(dir_s)
                    ld.dir_x = dv[0]
                    ld.dir_y = dv[1]
                    ld.dir_z = dv[2]

                var diff_s = _extract_attr(tag, "diffuse")
                if diff_s.byte_length() > 0:
                    var c = _parse_rgb3(diff_s)
                    ld.diffuse_r = c[0]
                    ld.diffuse_g = c[1]
                    ld.diffuse_b = c[2]

                var spec_s = _extract_attr(tag, "specular")
                if spec_s.byte_length() > 0:
                    var c = _parse_rgb3(spec_s)
                    ld.specular_r = c[0]
                    ld.specular_g = c[1]
                    ld.specular_b = c[2]

                var amb_s = _extract_attr(tag, "ambient")
                if amb_s.byte_length() > 0:
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
                if cutoff_s.byte_length() > 0:
                    ld.cutoff = _parse_float(cutoff_s)

                var exp_s = _extract_attr(tag, "exponent")
                if exp_s.byte_length() > 0:
                    ld.exponent = _parse_float(exp_s)

                var mode_s = _extract_attr(tag, "mode")
                if mode_s.byte_length() > 0:
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
                if pos_s.byte_length() > 0:
                    var pv = _parse_vec3(pos_s)
                    cd.pos_x = pv[0]
                    cd.pos_y = pv[1]
                    cd.pos_z = pv[2]

                # Orientation: quat > axisangle > xyaxes
                var quat_s = _extract_attr(tag, "quat")
                if quat_s.byte_length() > 0:
                    var qv = _parse_quat(quat_s)
                    cd.quat_x = qv[0]
                    cd.quat_y = qv[1]
                    cd.quat_z = qv[2]
                    cd.quat_w = qv[3]
                else:
                    var aa_s = _extract_attr(tag, "axisangle")
                    if aa_s.byte_length() > 0:
                        var aq = _parse_axisangle_to_quat(aa_s, deg_factor)
                        cd.quat_x = aq[0]
                        cd.quat_y = aq[1]
                        cd.quat_z = aq[2]
                        cd.quat_w = aq[3]
                    else:
                        var xy_s = _extract_attr(tag, "xyaxes")
                        if xy_s.byte_length() > 0:
                            var xq = _xyaxes_to_quat(xy_s)
                            cd.quat_x = xq[0]
                            cd.quat_y = xq[1]
                            cd.quat_z = xq[2]
                            cd.quat_w = xq[3]

                var fovy_s = _extract_attr(tag, "fovy")
                if fovy_s.byte_length() > 0:
                    cd.fovy = _parse_float(fovy_s)

                var ipd_s = _extract_attr(tag, "ipd")
                if ipd_s.byte_length() > 0:
                    cd.ipd = _parse_float(ipd_s)

                var mode_s = _extract_attr(tag, "mode")
                if mode_s.byte_length() > 0:
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

                # Same class resolution as geoms: the site's own class="..."
                # wins, else the enclosing body's childclass, else top-level.
                var site_class = _extract_attr(tag, "class")
                if site_class.byte_length() == 0:
                    site_class = childclass_stack[depth]
                var site_defaults = defaults
                if site_class.byte_length() > 0:
                    site_defaults = named_defaults.find(site_class)

                var type_s = _extract_attr(tag, "type")
                if type_s.byte_length() == 0:
                    type_s = site_defaults.site_type_s
                sd.site_type = _geom_type_from_str(type_s)

                var pos_s = _extract_attr(tag, "pos")
                if pos_s.byte_length() > 0:
                    var pv = _parse_vec3(pos_s)
                    sd.pos_x = pv[0]
                    sd.pos_y = pv[1]
                    sd.pos_z = pv[2]

                var quat_s = _extract_attr(tag, "quat")
                if quat_s.byte_length() > 0:
                    var qv = _parse_quat(quat_s)
                    sd.quat_x = qv[0]
                    sd.quat_y = qv[1]
                    sd.quat_z = qv[2]
                    sd.quat_w = qv[3]
                else:
                    var aa_s = _extract_attr(tag, "axisangle")
                    if aa_s.byte_length() > 0:
                        var aq = _parse_axisangle_to_quat(aa_s, deg_factor)
                        sd.quat_x = aq[0]
                        sd.quat_y = aq[1]
                        sd.quat_z = aq[2]
                        sd.quat_w = aq[3]

                var size_s = _extract_attr(tag, "size")
                if size_s.byte_length() == 0:
                    size_s = site_defaults.site_size_s
                if size_s.byte_length() > 0:
                    var parts = List[String]()

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

                # Resolve effective defaults: the geom's own class="..." wins,
                # else the enclosing body's childclass, else top-level.
                var geom_class = _extract_attr(tag, "class")
                if geom_class.byte_length() == 0:
                    geom_class = childclass_stack[depth]
                var eff_defaults = defaults
                if geom_class.byte_length() > 0:
                    eff_defaults = named_defaults.find(geom_class)

                # type
                var type_s = _extract_attr(tag, "type")
                if type_s.byte_length() == 0:
                    type_s = eff_defaults.geom_type_s
                gd.geom_type = _geom_type_from_str(type_s)

                # mesh reference: mesh="name" → resolve to file path from asset section
                if gd.geom_type == _GEOM_MESH:
                    var mesh_attr = _extract_attr(tag, "mesh")
                    if mesh_attr.byte_length() > 0:
                        for mi in range(result.num_mesh_assets):
                            if result.mesh_asset_names[mi] == mesh_attr:
                                gd.mesh_id = mi
                                gd.mesh_filename = result.mesh_asset_files[mi]
                                break

                # fromto — overrides pos and quat for capsule
                var fromto_s = _extract_attr(tag, "fromto")
                if fromto_s.byte_length() == 0:
                    fromto_s = eff_defaults.geom_fromto_s
                if fromto_s.byte_length() > 0:
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
                    if pos_s.byte_length() == 0:
                        pos_s = eff_defaults.geom_pos_s
                    if pos_s.byte_length() > 0:
                        var pv = _parse_vec3(pos_s)
                        gd.pos_x = pv[0]
                        gd.pos_y = pv[1]
                        gd.pos_z = pv[2]

                    # orientation: quat > axisangle > xyaxes > zaxis > euler
                    var quat_s = _extract_attr(tag, "quat")
                    if quat_s.byte_length() == 0:
                        quat_s = eff_defaults.geom_quat_s
                    var gq = _orientation_to_quat(
                        quat_s,
                        _extract_attr(tag, "axisangle"),
                        _extract_attr(tag, "xyaxes"),
                        _extract_attr(tag, "zaxis"),
                        _extract_attr(tag, "euler"),
                        deg_factor,
                        eulerseq,
                    )
                    gd.quat_x = gq[0]
                    gd.quat_y = gq[1]
                    gd.quat_z = gq[2]
                    gd.quat_w = gq[3]

                # size — interpretation depends on geom_type
                var size_s = _extract_attr(tag, "size")
                if size_s.byte_length() == 0:
                    size_s = eff_defaults.geom_size_s
                if size_s.byte_length() > 0:
                    var size_parts = List[String]()

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
                        # Only use size[1] as half-length if no fromto
                        # (fromto already computed the correct value).
                        if len(size_parts) >= 2 and fromto_s.byte_length() == 0:
                            gd.half_length = s1
                    elif gd.geom_type == _GEOM_BOX:
                        gd.half_x = s0
                        gd.half_y = s1
                        gd.half_z = s2
                        gd.radius = _sqrt_f64(s0 * s0 + s1 * s1 + s2 * s2)
                    elif gd.geom_type == _GEOM_CYLINDER:
                        gd.radius = s0
                        if fromto_s.byte_length() == 0:
                            gd.half_length = s1
                    elif gd.geom_type == _GEOM_ELLIPSOID:
                        # `size` is the three SEMI-AXES, stored like a box's
                        # half-extents. `radius` keeps size[0] so the broad
                        # phase's bounding radius stays conservative.
                        gd.half_x = s0
                        gd.half_y = s1
                        gd.half_z = s2
                        gd.radius = s0
                    elif gd.geom_type == _GEOM_PLANE:
                        gd.half_x = s0
                        gd.half_y = s1
                        # s2 = grid spacing — not needed for collision
                    else:
                        gd.radius = s0

                # friction (explicit or default)
                var fric_s = _extract_attr(tag, "friction")
                if fric_s.byte_length() > 0:
                    var fvec = _parse_vec3(fric_s)
                    gd.friction = fvec[0]
                    gd.friction_spin = fvec[1]
                    gd.friction_roll = fvec[2]
                else:
                    gd.friction = eff_defaults.geom_friction
                    gd.friction_spin = eff_defaults.geom_friction_spin
                    gd.friction_roll = eff_defaults.geom_friction_roll

                # contype / conaffinity / condim
                var ct_s = _extract_attr(tag, "contype")
                gd.contype = (
                    _parse_int_str(ct_s) if ct_s.byte_length()
                    > 0 else eff_defaults.geom_contype
                )

                var ca_s = _extract_attr(tag, "conaffinity")
                gd.conaffinity = (
                    _parse_int_str(ca_s) if ca_s.byte_length()
                    > 0 else eff_defaults.geom_conaffinity
                )

                var cd_s = _extract_attr(tag, "condim")
                gd.condim = (
                    _parse_int_str(cd_s) if cd_s.byte_length()
                    > 0 else eff_defaults.geom_condim
                )

                # solref / solimp
                var sr_s = _extract_attr(tag, "solref")
                if sr_s.byte_length() > 0:
                    var sv = _parse_vec3(sr_s)
                    gd.solref_0 = sv[0]
                    gd.solref_1 = sv[1]
                else:
                    gd.solref_0 = eff_defaults.geom_solref_0
                    gd.solref_1 = eff_defaults.geom_solref_1

                var si_s = _extract_attr(tag, "solimp")
                if si_s.byte_length() > 0:
                    var sip = List[String]()

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
                    gd.solimp_0 = eff_defaults.geom_solimp_0
                    gd.solimp_1 = eff_defaults.geom_solimp_1
                    gd.solimp_2 = eff_defaults.geom_solimp_2
                    gd.solimp_3 = eff_defaults.geom_solimp_3
                    gd.solimp_4 = eff_defaults.geom_solimp_4

                # margin
                var mg_s = _extract_attr(tag, "margin")
                gd.margin = (
                    _parse_float(mg_s) if mg_s.byte_length()
                    > 0 else eff_defaults.geom_margin
                )

                # density (per-geom overrides default; used when mass is absent)
                var dens_s = _extract_attr(tag, "density")
                gd.density = (
                    _parse_float(dens_s) if dens_s.byte_length()
                    > 0 else eff_defaults.geom_density
                )

                # mass: explicit if provided, else compute from density * volume
                var ms_s = _extract_attr(tag, "mass")
                if ms_s.byte_length() == 0:
                    ms_s = eff_defaults.geom_mass_s
                if ms_s.byte_length() > 0:
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
                    elif gd.geom_type == _GEOM_ELLIPSOID:
                        vol = (
                            (Float64(4.0) / Float64(3.0))
                            * PI
                            * gd.half_x
                            * gd.half_y
                            * gd.half_z
                        )
                    # PLANE has no volume → mass stays 0
                    if vol > Float64(0):
                        gd.mass = gd.density * vol
                    else:
                        gd.mass = Float64(-1)

                # group (visual/collision grouping, 0-5)
                var grp_s = _extract_attr(tag, "group")
                if grp_s.byte_length() == 0:
                    grp_s = eff_defaults.geom_group_s
                if grp_s.byte_length() > 0:
                    gd.group = _parse_int_str(grp_s)

                # rgba colour: per-geom > default > GeomData fallback (0.7 grey)
                var rgba_s = _extract_attr(tag, "rgba")
                if rgba_s.byte_length() > 0:
                    var cv = _parse_rgba4(rgba_s)
                    gd.rgba_r = cv[0]
                    gd.rgba_g = cv[1]
                    gd.rgba_b = cv[2]
                    gd.rgba_a = cv[3]
                elif eff_defaults.geom_rgba_r >= Float64(0):
                    gd.rgba_r = eff_defaults.geom_rgba_r
                    gd.rgba_g = eff_defaults.geom_rgba_g
                    gd.rgba_b = eff_defaults.geom_rgba_b
                    gd.rgba_a = eff_defaults.geom_rgba_a

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


def _fill_actuators[
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
    NEQ: Int = 0,
    NEXCLUDE: Int = 0,
    NTENDON: Int = 0,
](
    actuator_sec: String,
    worldbody: String,
    defaults: DefaultsData,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ],
):
    """Parse <actuator> section and populate result.actuators[]."""
    var act_count = 0
    var scan_pos = 0
    var alen = actuator_sec.byte_length()

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

        # Record WHICH tag this was. Only <motor> (direct force = gear * ctrl)
        # is implemented; <position>/<velocity>/<general> are position/velocity
        # servos whose gainprm/biasprm this struct does not carry. Recording the
        # kind lets `init_fields` reject them instead of silently simulating a
        # servo as a torque motor. See docs/DM_CONTROL_PORT.md (gap G3).
        if earliest == np_:
            ad.kind = ACT_KIND_POSITION
        elif earliest == nv_:
            ad.kind = ACT_KIND_VELOCITY
        elif earliest == ng:
            ad.kind = ACT_KIND_GENERAL

        # gear (element attribute wins, else the <default><motor> class)
        var gear_s = _extract_attr(tag, "gear")
        if gear_s.byte_length() > 0:
            ad.gear = _parse_float(gear_s)
        else:
            ad.gear = defaults.motor_gear

        # joint name → joint index
        var jname = _extract_attr(tag, "joint")
        if jname.byte_length() > 0:
            ad.joint_id = _find_joint_index_by_name(worldbody, jname)

        # ctrlrange / ctrllimited
        var cr_s = _extract_attr(tag, "ctrlrange")
        if cr_s.byte_length() > 0:
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
# Phase 5b: Parse <equality> section — weld and connect constraints
# =============================================================================


def _fill_equality[
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
    NEQ: Int,
    NEXCLUDE: Int = 0,
    NTENDON: Int = 0,
](
    equality_sec: String,
    worldbody: String,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ],
):
    """Parse <equality> section: fill result.equalities[] with weld/connect data."""
    var eq_count = 0
    var scan_pos = 0
    var elen = equality_sec.byte_length()

    while scan_pos < elen and eq_count < NEQ:
        # Find next <weld or <connect tag
        var nw = equality_sec.find("<weld", scan_pos)
        var nc = equality_sec.find("<connect", scan_pos)

        var earliest = _min_valid(nw, nc)
        if earliest == -1:
            break

        var tag = _extract_opening_tag(equality_sec, earliest)
        var ed = EqualityData()

        # Determine type
        if earliest == nw:
            ed.eq_type = _EQ_WELD
        else:
            ed.eq_type = _EQ_CONNECT

        # body1 / body2 — resolve names to indices
        var b1_name = _extract_attr(tag, "body1")
        if b1_name.byte_length() > 0:
            ed.body_a = _find_body_index_by_name(worldbody, b1_name)

        var b2_name = _extract_attr(tag, "body2")
        if b2_name.byte_length() > 0:
            ed.body_b = _find_body_index_by_name(worldbody, b2_name)

        # anchor (connect) — point in body1 frame
        var anchor_s = _extract_attr(tag, "anchor")
        if anchor_s.byte_length() > 0:
            var av = _parse_vec3(anchor_s)
            ed.anchor_a_x = av[0]
            ed.anchor_a_y = av[1]
            ed.anchor_a_z = av[2]

        # relpose (weld) — relative position + quaternion (7 values: x y z qw qx qy qz)
        var relpose_s = _extract_attr(tag, "relpose")
        if relpose_s.byte_length() > 0:
            var parts = List[String]()
            _split_spaces(relpose_s, parts)
            if len(parts) >= 3:
                ed.anchor_a_x = _parse_float(parts[0])
                ed.anchor_a_y = _parse_float(parts[1])
                ed.anchor_a_z = _parse_float(parts[2])
            if len(parts) >= 7:
                # MuJoCo relpose quat is (w,x,y,z), convert to (x,y,z,w)
                ed.relpose_x = _parse_float(parts[4])
                ed.relpose_y = _parse_float(parts[5])
                ed.relpose_z = _parse_float(parts[6])
                ed.relpose_w = _parse_float(parts[3])

        # solref
        var sr_s = _extract_attr(tag, "solref")
        if sr_s.byte_length() > 0:
            var sv = _parse_vec3(sr_s)
            ed.solref_0 = sv[0]
            ed.solref_1 = sv[1]

        # solimp
        var si_s = _extract_attr(tag, "solimp")
        if si_s.byte_length() > 0:
            var parts = List[String]()
            _split_spaces(si_s, parts)
            if len(parts) >= 1:
                ed.solimp_0 = _parse_float(parts[0])
            if len(parts) >= 2:
                ed.solimp_1 = _parse_float(parts[1])
            if len(parts) >= 3:
                ed.solimp_2 = _parse_float(parts[2])
            if len(parts) >= 4:
                ed.solimp_3 = _parse_float(parts[3])
            if len(parts) >= 5:
                ed.solimp_4 = _parse_float(parts[4])

        result.equalities[eq_count] = ed
        eq_count += 1

        var tag_end = equality_sec.find(">", earliest)
        scan_pos = tag_end + 1 if tag_end != -1 else elen


# =============================================================================
# Phase 6: Resolve geom material references (post-pass)
# =============================================================================


def _find_material_index_by_name(asset_sec: String, name: String) -> Int:
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
        var tag = String(asset_sec[byte = t : tag_end + 1])
        if tag.find(search) != -1:
            return count
        count += 1
        scan_pos = tag_end + 1
    return -1


# =============================================================================
# Phase 5c: Parse <contact><exclude> section
# =============================================================================


def _tendon_index_by_name(tendon_sec: String, name: String) -> Int:
    """Index of the `<fixed>`/`<spatial>` tendon called `name`, or -1.

    Numbering is XML order, exactly as `_fill_tendons` assigns it — both walk
    the same two markers in the same order.
    """
    var count = 0
    var scan_pos = 0
    var tlen = tendon_sec.byte_length()
    while scan_pos < tlen:
        var nf = tendon_sec.find("<fixed", scan_pos)
        var ns = tendon_sec.find("<spatial", scan_pos)
        var earliest = _min_valid(nf, ns)
        if earliest == -1:
            return -1
        var tag_end = tendon_sec.find(">", earliest)
        if tag_end == -1:
            return -1
        var tag = String(tendon_sec[byte = earliest : tag_end + 1])
        if _trim(_extract_attr(tag, "name")) == name:
            return count
        count += 1
        scan_pos = tag_end + 1
    return -1


def _default_class_tag(xml: String, cls: String, tag_name: String) -> String:
    """First `<tag_name ...>` inside `<default class="cls">`, or "".

    Nested classes are stripped first, so a class containing sub-classes
    resolves to its OWN child rather than a grandchild's.
    """
    if cls.byte_length() == 0:
        return String("")
    var n = xml.byte_length()
    var scan = 0
    while scan < n:
        var t = xml.find("<default", scan)
        if t == -1:
            return String("")
        var te = xml.find(">", t)
        if te == -1:
            return String("")
        if _trim(_extract_attr(_extract_opening_tag(xml, t), "class")) != cls:
            scan = te + 1
            continue
        var close = _find_matching_default_close(xml, t)
        if close == -1:
            return String("")
        var inner = _strip_nested_defaults(String(xml[byte = te + 1 : close]))
        var it = inner.find("<" + tag_name)
        if it == -1:
            return String("")
        var ite = inner.find(">", it)
        if ite == -1:
            return String("")
        return String(inner[byte = it : ite + 1])
    return String("")


def _fill_tendon_equalities[
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
    NEQ: Int,
    NEXCLUDE: Int = 0,
    NTENDON: Int = 0,
](
    equality_sec: String,
    tendon_sec: String,
    xml: String,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ],
) raises:
    """`<equality><tendon tendon1="..."/>` -> `TendonData.is_equality` + solref.

    ⚠ This did not exist until 2026-07-31 and its absence was SILENT.
    `_fill_equality` scans only `<weld` and `<connect`, so `is_equality` stayed
    at its 0 default for every model, and `constraints/equality_tendon.mojo`
    — which gates on exactly that flag — never produced a row. The equality
    solref/solimp slots it reads were likewise written by nobody, so they were
    zero. `test_equality_tendon_fields` passes regardless because it builds the
    tendon record by hand rather than through the parser, so the constraint
    MATH was covered while the PARSING was not.

    dm_control's quadruped is the first model to need it: each leg's
    `coupling_*` tendon constrains .333*(pitch + knee + ankle) to zero, and
    without the constraint the three joints are independent — a different
    robot, converging to a different gait, with no error anywhere.

    MuJoCo's residual for a one-object tendon equality (engine_core_constraint
    .c:603) is `ten_length - tendon_length0 - eq_data[0]`, so `polycoef[0]`
    must be 0 for `length_ref` alone to describe the target; a non-default
    polycoef and the two-tendon (`tendon2`) polynomial coupling both RAISE
    rather than silently degrade to the simple case.
    """
    var scan_pos = 0
    var elen = equality_sec.byte_length()
    while scan_pos < elen:
        var t = equality_sec.find("<tendon", scan_pos)
        if t == -1:
            break
        var tag = _extract_opening_tag(equality_sec, t)
        var tag_end = equality_sec.find(">", t)
        scan_pos = tag_end + 1 if tag_end != -1 else elen

        var n2 = _trim(_extract_attr(tag, "tendon2"))
        if n2.byte_length() > 0:
            raise Error(
                "physics3d: <equality><tendon> with tendon2 couples two"
                " tendon lengths by a quartic polynomial; only the"
                " single-tendon form (length == polycoef[0]) is implemented."
            )
        var poly = _trim(_extract_attr(tag, "polycoef"))
        if poly.byte_length() > 0:
            var pp = List[String]()
            _split_spaces(poly, pp)
            var bad = False
            for i in range(len(pp)):
                var want = Float64(1.0) if i == 1 else Float64(0.0)
                if _parse_float(pp[i]) != want:
                    bad = True
            if bad:
                raise Error(
                    "physics3d: <equality><tendon polycoef=...> other than the"
                    " default '0 1 0 0 0' is not implemented; the residual"
                    " here is ten_length - length0 - polycoef[0]."
                )

        var n1 = _trim(_extract_attr(tag, "tendon1"))
        var idx = _tendon_index_by_name(tendon_sec, n1)
        if idx < 0 or idx >= NTENDON:
            raise Error(
                String(
                    "physics3d: <equality><tendon tendon1='",
                    n1,
                    "'/> names no tendon in <tendon>.",
                )
            )
        # READ-MODIFY-WRITE, matching `_fill_tendons`. Mutating
        # `result.tendons[idx].field` in place does NOT stick — the subscript
        # hands back a copy, so the writes are silently dropped and every
        # tendon still reads is_equality == 0.
        var td = result.tendons[idx]
        td.is_equality = 1

        # solref/solimp: element, then `class="..."`, then MuJoCo's defaults
        # (already in TendonData.__init__). quadruped keeps both in
        # `<default class="coupling"><equality .../></default>`.
        var cls = _trim(_extract_attr(tag, "class"))
        var cls_tag = _default_class_tag(xml, cls, "equality")
        var sr = _trim(_extract_attr(tag, "solref"))
        if sr.byte_length() == 0:
            sr = _trim(_extract_attr(cls_tag, "solref"))
        if sr.byte_length() > 0:
            var sp = List[String]()
            _split_spaces(sr, sp)
            if len(sp) >= 2:
                td.solref_eq_0 = _parse_float(sp[0])
                td.solref_eq_1 = _parse_float(sp[1])
        var si = _trim(_extract_attr(tag, "solimp"))
        if si.byte_length() == 0:
            si = _trim(_extract_attr(cls_tag, "solimp"))
        if si.byte_length() > 0:
            var ip = List[String]()
            _split_spaces(si, ip)
            if len(ip) >= 1:
                td.solimp_eq_0 = _parse_float(ip[0])
            if len(ip) >= 2:
                td.solimp_eq_1 = _parse_float(ip[1])
            if len(ip) >= 3:
                td.solimp_eq_2 = _parse_float(ip[2])
            if len(ip) >= 4:
                td.solimp_eq_3 = _parse_float(ip[3])
            if len(ip) >= 5:
                td.solimp_eq_4 = _parse_float(ip[4])
        result.tendons[idx] = td


def _fill_tendons[
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
    NEQ: Int,
    NEXCLUDE: Int,
    NTENDON: Int = 0,
](
    tendon_sec: String,
    worldbody: String,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ],
) raises:
    """Parse <tendon>: fill result.tendons[] with <fixed> and <spatial> data.

    Tendon order is XML order, matching how every other element is numbered
    here. `<fixed>` reads its <joint joint= coef=> children; `<spatial>` reads
    its <site site=> children. Both read `limited`/`range`/`margin` and the
    limit solref/solimp pair.

    RAISES rather than skips on a <spatial> containing <geom> (wrap surface)
    or <pulley>. Neither is implemented, and dropping them would silently
    shorten the tendon — the failure would surface only as a physics
    divergence, which is exactly how the `<tendon>`-dropped-by-merge_mjcf bug
    stayed hidden until 2026-07-30.
    """
    var count = 0
    var scan_pos = 0
    var tlen = tendon_sec.byte_length()

    while scan_pos < tlen and count < NTENDON:
        var nf = tendon_sec.find("<fixed", scan_pos)
        var ns = tendon_sec.find("<spatial", scan_pos)
        var earliest = _min_valid(nf, ns)
        if earliest == -1:
            break

        var is_spatial = earliest == ns
        var close_tag = String("</spatial>") if is_spatial else String(
            "</fixed>"
        )
        var open_tag = _extract_opening_tag(tendon_sec, earliest)

        # Body spans the element: either up to its closing tag, or (for a
        # self-closed <fixed/>) nothing at all.
        var body_start = tendon_sec.find(">", earliest) + 1
        var body_end = tendon_sec.find(close_tag, body_start)
        var inner = String("")
        if body_end != -1:
            inner = String(tendon_sec[byte=body_start:body_end])

        var td = TendonData()
        td.kind = _TENDON_KIND_SPATIAL if is_spatial else _TENDON_KIND_FIXED

        if is_spatial:
            if inner.find("<geom") != -1:
                raise Error(
                    "physics3d: <spatial> tendon with a wrap <geom> is not"
                    " supported (site-to-site routing only)"
                )
            if inner.find("<pulley") != -1:
                raise Error(
                    "physics3d: <spatial> tendon with a <pulley> is not"
                    " supported (site-to-site routing only)"
                )

            var spos = 0
            while td.num_sites < 4:
                var sp = inner.find("<site", spos)
                if sp == -1:
                    break
                var stag = _extract_opening_tag(inner, sp)
                var sname = _extract_attr(stag, "site")
                if sname.byte_length() > 0:
                    var sid = _find_site_index_by_name(worldbody, sname)
                    if sid < 0:
                        raise Error(
                            "physics3d: <spatial> tendon references unknown"
                            " site '" + sname + "'"
                        )
                    td.site_ids[td.num_sites] = sid
                    td.num_sites += 1
                spos = inner.find(">", sp) + 1
            if td.num_sites < 2:
                raise Error(
                    "physics3d: <spatial> tendon needs at least two <site>"
                    " waypoints"
                )
        else:
            var jpos = 0
            while td.num_joints < 4:
                var jp = inner.find("<joint", jpos)
                if jp == -1:
                    break
                var jtag = _extract_opening_tag(inner, jp)
                var jname = _extract_attr(jtag, "joint")
                if jname.byte_length() > 0:
                    var jid = _find_joint_index_by_name(worldbody, jname)
                    if jid < 0:
                        raise Error(
                            "physics3d: <fixed> tendon references unknown"
                            " joint '" + jname + "'"
                        )
                    td.joint_ids[td.num_joints] = jid
                    var coef_s = _extract_attr(jtag, "coef")
                    if coef_s.byte_length() > 0:
                        td.coefs[td.num_joints] = _parse_float(coef_s)
                    td.num_joints += 1
                jpos = inner.find(">", jp) + 1

        # limited / range / margin
        var limited_s = _extract_attr(open_tag, "limited")
        var range_s = _extract_attr(open_tag, "range")
        if range_s.byte_length() > 0:
            var parts = List[String]()
            _split_spaces(range_s, parts)
            if len(parts) >= 2:
                td.range_min = _parse_float(parts[0])
                td.range_max = _parse_float(parts[1])
        # MuJoCo's `limited="auto"` (the compiler default) enables the limit
        # whenever a range is present; an explicit "true"/"false" wins.
        if limited_s == "true":
            td.limited = 1
        elif limited_s == "false":
            td.limited = 0
        elif range_s.byte_length() > 0:
            td.limited = 1

        var margin_s = _extract_attr(open_tag, "margin")
        if margin_s.byte_length() > 0:
            td.margin = _parse_float(margin_s)

        var solref_s = _extract_attr(open_tag, "solreflimit")
        if solref_s.byte_length() > 0:
            var rp = List[String]()
            _split_spaces(solref_s, rp)
            if len(rp) >= 2:
                td.solref_lim_0 = _parse_float(rp[0])
                td.solref_lim_1 = _parse_float(rp[1])

        var solimp_s = _extract_attr(open_tag, "solimplimit")
        if solimp_s.byte_length() > 0:
            var ip = List[String]()
            _split_spaces(solimp_s, ip)
            if len(ip) >= 1:
                td.solimp_lim_0 = _parse_float(ip[0])
            if len(ip) >= 2:
                td.solimp_lim_1 = _parse_float(ip[1])
            if len(ip) >= 3:
                td.solimp_lim_2 = _parse_float(ip[2])
            if len(ip) >= 4:
                td.solimp_lim_3 = _parse_float(ip[3])
            if len(ip) >= 5:
                td.solimp_lim_4 = _parse_float(ip[4])

        result.tendons[count] = td
        count += 1

        if body_end != -1:
            scan_pos = body_end + close_tag.byte_length()
        else:
            scan_pos = tendon_sec.find(">", earliest) + 1


def _fill_excludes[
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
    NEQ: Int,
    NEXCLUDE: Int,
    NTENDON: Int = 0,
](
    contact_sec: String,
    worldbody: String,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ],
):
    """Parse <contact> section: fill result.excludes[] with body pair exclusions."""
    var ex_count = 0
    var scan_pos = 0
    var clen = contact_sec.byte_length()

    while scan_pos < clen and ex_count < NEXCLUDE:
        var ne = contact_sec.find("<exclude", scan_pos)
        if ne == -1:
            break
        var tag = _extract_opening_tag(contact_sec, ne)
        var body1_name = _trim(_extract_attr(tag, "body1"))
        var body2_name = _trim(_extract_attr(tag, "body2"))

        # Resolve body names to indices (1-based, 0=worldbody)
        var b1 = _find_body_index_by_name(worldbody, body1_name)
        var b2 = _find_body_index_by_name(worldbody, body2_name)

        if b1 >= 0 and b2 >= 0:
            # Store with canonical ordering (smaller first) for fast lookup
            if b1 <= b2:
                result.excludes[ex_count] = ExcludeData(b1, b2)
            else:
                result.excludes[ex_count] = ExcludeData(b2, b1)
            ex_count += 1

        var tag_end = contact_sec.find(">", ne)
        scan_pos = tag_end + 1 if tag_end != -1 else ne + 1


def _resolve_geom_materials[
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
    NEQ: Int = 0,
    NEXCLUDE: Int = 0,
    NTENDON: Int = 0,
](
    worldbody: String,
    asset_sec: String,
    mut result: FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ],
):
    """Resolve material="name" on geoms → material index; copy material rgba."""
    var scan_pos = 0
    var geom_idx = 0
    var wlen = worldbody.byte_length()

    while scan_pos < wlen and geom_idx < NGEOM:
        var t = worldbody.find("<geom", scan_pos)
        if t == -1:
            break
        var tag_end = worldbody.find(">", t)
        if tag_end == -1:
            break
        var tag = String(worldbody[byte = t : tag_end + 1])
        var mat_name = _extract_attr(tag, "material")
        if mat_name.byte_length() > 0:
            var mid = _find_material_index_by_name(asset_sec, mat_name)
            result.geoms[geom_idx].material_id = mid
            # Only inherit material rgba when the geom has no explicit rgba attr
            var has_explicit_rgba = _extract_attr(tag, "rgba").byte_length() > 0
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


def parse_xml_full[
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
    NEQ: Int = 0,
    NEXCLUDE: Int = 0,
    NTENDON: Int = 0,
](xml: String) raises -> FlatModelDef[
    NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE, NEQ,
    NEXCLUDE, NTENDON,
]:
    """Full MJCF parse: returns a populated FlatModelDef.

    Caller must obtain dimensions via parse_xml() first:

        comptime pm  = parse_xml(xml)
        comptime fmd = parse_xml_full[
            pm.NBODY, pm.NJOINT, pm.NQ, pm.NV, pm.NGEOM, pm.NACT,
            pm.NTEX, pm.NMAT, pm.NLIGHT, pm.NCAM, pm.NSITE,
            pm.NEQ, pm.NEXCLUDE, pm.NTENDON,
        ](xml)

    The NTEX/NMAT/NLIGHT/NCAM/NSITE parameters default to 0 for backward
    compatibility — existing callers omitting them get no visual element arrays.
    All operations are comptime-safe (String.find + slice arithmetic only).
    """
    var result = FlatModelDef[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ]()

    # Extract top-level sections
    var worldbody = _extract_section(xml, "worldbody")
    var actuator_sec = _extract_section(xml, "actuator")
    var asset_sec = _extract_section(xml, "asset")
    var equality_sec = _extract_section(xml, "equality")
    var contact_sec = _extract_section(xml, "contact")

    # Global physics options
    var opt = _parse_option(xml)
    result.gravity_x = opt[0]
    result.gravity_y = opt[1]
    result.gravity_z = opt[2]
    result.timestep = opt[3]
    result.opt_density = opt[4]
    result.opt_viscosity = opt[5]

    # <flag gravity="disable"/> — zero the gravity vector.
    if _option_flag_disabled(xml, "gravity"):
        result.gravity_x = Float64(0)
        result.gravity_y = Float64(0)
        result.gravity_z = Float64(0)

    # Defaults (applied when specific attrs are absent)
    var defaults_tuple = _parse_defaults(xml)
    var defaults = defaults_tuple[0]
    var named_defaults = defaults_tuple[1]

    # Compiler angle units (MuJoCo's MJCF default is degree) and euler order.
    var deg_factor = _compiler_deg_factor(xml)
    var eulerseq = String("xyz")
    var compiler_t = xml.find("<compiler")
    if compiler_t != -1:
        var compiler_end = xml.find(">", compiler_t)
        if compiler_end != -1:
            var ctag = String(xml[byte = compiler_t : compiler_end + 1])
            var seq_val = _trim(_extract_attr(ctag, "eulerseq"))
            if seq_val.byte_length() == 3:
                eulerseq = seq_val

    # Assets: textures and materials
    _fill_assets[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ](asset_sec, result)

    # Single DFS pass: bodies + joints + geoms + lights + cameras + sites
    _fill_model[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ](worldbody, defaults, named_defaults, result, deg_factor, eulerseq)

    # <flag contact="disable"/> — MuJoCo drops ALL contacts. We have no global
    # switch, but zeroing every geom's contype/conaffinity is exactly
    # equivalent: no pair can ever pass the collision mask.
    #
    # This is not cosmetic. Several suite models interpenetrate on purpose
    # because contacts are off — cartpole's cart box (size .2 .15 .1 at z=1)
    # straddles both rails (y = +-.07 at z=1), so with contacts live the cart
    # is launched on the first step.
    # <flag constraint="disable"/> — MuJoCo's mjDSBL_CONSTRAINT switches the
    # whole constraint solver off, so contacts, joint/tendon limits, friction
    # loss and equality constraints all stop generating rows. We reproduce the
    # two that our engine builds rows for: contacts (via the collision mask,
    # as above) and joint limits (via the unlimited sentinel).
    #
    # acrobot.xml relies on this — its lower arm sweeps a metre BELOW the
    # floor plane, so with contacts live the swing-up dynamics are wrong.
    var constraints_off = _option_flag_disabled(xml, "constraint")

    if _option_flag_disabled(xml, "contact") or constraints_off:
        for gi in range(NGEOM):
            result.geoms[gi].contype = 0
            result.geoms[gi].conaffinity = 0

    if constraints_off:
        for ji in range(NJOINT):
            result.joints[ji].range_min = Float64(-1e10)
            result.joints[ji].range_max = Float64(1e10)

    # Actuators
    _fill_actuators[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ](actuator_sec, worldbody, defaults, result)

    # Equality constraints
    comptime if NEQ > 0:
        _fill_equality[
            NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM,
            NSITE, NEQ, NEXCLUDE, NTENDON,
        ](equality_sec, worldbody, result)

    # Tendons
    comptime if NTENDON > 0:
        _fill_tendons[
            NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM,
            NSITE, NEQ, NEXCLUDE, NTENDON,
        ](_extract_section(xml, "tendon"), worldbody, result)
        # AFTER the tendons exist — this marks them by name. Note it is NOT
        # gated on NEQ: a tendon equality does not occupy an EqualityData
        # slot (it lives on the tendon record), so quadruped has neq==0 while
        # declaring four of them.
        _fill_tendon_equalities[
            NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM,
            NSITE, NEQ, NEXCLUDE, NTENDON,
        ](equality_sec, _extract_section(xml, "tendon"), xml, result)

    # Contact exclusion pairs
    comptime if NEXCLUDE > 0:
        _fill_excludes[
            NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM,
            NSITE, NEQ, NEXCLUDE, NTENDON,
        ](contact_sec, worldbody, result)

    # Post-pass: resolve geom material="name" references
    _resolve_geom_materials[
        NBODY, NJOINT, NQ, NV, NGEOM, NACT, NTEX, NMAT, NLIGHT, NCAM, NSITE,
        NEQ, NEXCLUDE, NTENDON,
    ](worldbody, asset_sec, result)

    return result^
