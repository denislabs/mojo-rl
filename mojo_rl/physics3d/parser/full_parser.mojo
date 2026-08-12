"""Full MJCF XML parser — populates a FlatModelDef from an embedded XML string.

Designed to run entirely at comptime:

    comptime pm  = parse_xml(xml)
    var fmd = parse_xml_full(xml)          # non-generic since 2026-08-05

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
    _strip_xml_comments,
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
    _find_geom_index_by_name,
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
    PairData,
    TendonData,
    _TENDON_KIND_FIXED,
    _TENDON_KIND_SPATIAL,
    NamedDefaultsList,
    FlatModelDef,
    _EQ_CONNECT,
    _EQ_WELD,
    _EQ_JOINT,
    _EQ_OBJ_BODY,
    _EQ_OBJ_SITE,
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
# How many joints/sites one tendon may wrap — shared with the packed field
# layout so the parser and the record cannot disagree.
from mojo_rl.physics3d.gpu.constants import TENDON_MAX_WRAPS
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

        # `springdamper="timeconst dampratio"` — MuJoCo DERIVES stiffness and
        # damping from these plus the body's own inertia, overwriting whatever
        # `stiffness`/`damping` said. dm_control's dog declares it exactly once
        # (in a class), which is why ~20 of its `jnt_stiffness` values appear
        # nowhere in the XML.
        var sd_s = _extract_attr(jtag, "springdamper")
        if sd_s.byte_length() > 0:
            var sdv = _parse_vec3(sd_s)
            d.joint_springdamper_0 = sdv[0]
            d.joint_springdamper_1 = sdv[1]

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

        # ⚠ A PARTIAL `friction` OVERWRITES ONLY THE COMPONENTS PRESENT.
        # `_parse_vec3` returns 0 for anything missing, so `friction="0.9"`
        # used to zero the TORSIONAL and ROLLING coefficients instead of
        # leaving MuJoCo's (0.005, 0.0001). Measured on dog: 86 of its 128
        # geoms are `(0.9, 0.005, 0.0001)` in MuJoCo and were `(0.9, 0, 0)`
        # here. The `solimp` block below already guards on `len(parts)`; this
        # one did not, which is the only reason the two behaved differently.
        var fric_s = _extract_attr(gtag, "friction")
        if fric_s.byte_length() > 0:
            var fparts = List[String]()
            _split_spaces(fric_s, fparts)
            if len(fparts) >= 1:
                d.geom_friction = _parse_float(fparts[0])
            if len(fparts) >= 2:
                d.geom_friction_spin = _parse_float(fparts[1])
            if len(fparts) >= 3:
                d.geom_friction_roll = _parse_float(fparts[2])

        var ct_s = _extract_attr(gtag, "contype")
        if ct_s.byte_length() > 0:
            d.geom_contype = _parse_int_str(ct_s)

        var ca_s = _extract_attr(gtag, "conaffinity")
        if ca_s.byte_length() > 0:
            d.geom_conaffinity = _parse_int_str(ca_s)

        var cd_s = _extract_attr(gtag, "condim")
        if cd_s.byte_length() > 0:
            d.geom_condim = _parse_int_str(cd_s)

        var pr_s = _extract_attr(gtag, "priority")
        if pr_s.byte_length() > 0:
            d.geom_priority = _parse_int_str(pr_s)

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
        var gmesh_s = _extract_attr(gtag, "mesh")
        if gmesh_s.byte_length() > 0:
            d.geom_mesh_s = gmesh_s
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
        # POSE. Sites take the same five orientation spellings as geoms and
        # bodies, and a default class may set any of them. Each is captured
        # separately so the child tag can override one without clearing the
        # others — which is what `class="fingertip"` does: it re-declares
        # `euler="0 0 0"` to cancel the `euler="0 15 0"` it would otherwise
        # inherit from `class="hand"`.
        var sp_s = _extract_attr(stag, "pos")
        if sp_s.byte_length() > 0:
            d.site_pos_s = sp_s
        var sq_s = _extract_attr(stag, "quat")
        if sq_s.byte_length() > 0:
            d.site_quat_s = sq_s
        var saa_s = _extract_attr(stag, "axisangle")
        if saa_s.byte_length() > 0:
            d.site_axisangle_s = saa_s
        var sxy_s = _extract_attr(stag, "xyaxes")
        if sxy_s.byte_length() > 0:
            d.site_xyaxes_s = sxy_s
        var sza_s = _extract_attr(stag, "zaxis")
        if sza_s.byte_length() > 0:
            d.site_zaxis_s = sza_s
        var seu_s = _extract_attr(stag, "euler")
        if seu_s.byte_length() > 0:
            d.site_euler_s = seu_s

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
        # ⚠ A self-closing `<default class="x"/>` encloses NOTHING. Walking for
        # its `</default>` swallows the NEXT class's whole block — see
        # `_is_self_closing_tag`. Drop just the tag and carry on.
        if _is_self_closing_tag(sec, open_t):
            var self_end = sec.find(">", open_t)
            i = self_end + 1 if self_end != -1 else n
            continue
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
) raises -> Tuple[DefaultsData, NamedDefaultsList]:
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


def _is_self_closing_tag(sec: String, open_pos: Int) -> Bool:
    """Does the tag starting at `open_pos` end in `/>` rather than `>`?

    ⚠⚠ THIS IS WHY JACO WOULD NOT LOAD. PyMJCF emits an EMPTY root class as
    `<default class="/"/>` — self-closing, no `</default>`. The depth tracker
    below counted it as an opening block, so the outer `<default>`'s matching
    close was never found (`_find_matching_default_close` returned -1) and NOT
    ONE named class was registered. Every geom that takes its `type` from a
    class then fell through to the sphere default: all 14 of Jaco's mesh geoms
    and all 6 of its cylinders became spheres of radius 0.5, and since the
    type was never MESH, `mesh=` was never resolved either — `mesh_id` stayed
    -1 across the board.

    It is invisible on every model ported before this one because none of them
    emits an empty `<default class="..."/>`; hand-written MJCF always puts
    something inside. It is also invisible in the COUNTS — ngeom, nbody and the
    body ids all came out right, so nothing upstream complained.
    """
    var end = sec.find(">", open_pos)
    if end <= open_pos:
        return False
    return sec[byte = end - 1 : end] == "/"


def _find_matching_default_close(sec: String, open_pos: Int) -> Int:
    """Index of the `</default>` matching the `<default` at `open_pos`.

    Returns -1 if unbalanced. Depth-tracked, because `<default>` blocks nest.

    ⚠ A SELF-CLOSING `<default .../>` opens nothing — see
    `_is_self_closing_tag` for what that cost.
    """
    var n = sec.byte_length()
    var depth = 0
    var i = open_pos
    while i < n:
        var next_open = sec.find("<default", i + 1)
        # Skip self-closing `<default .../>`: it has no `</default>` to pair
        # with, so counting it would leave the scan permanently one deep.
        while next_open != -1 and _is_self_closing_tag(sec, next_open):
            next_open = sec.find("<default", next_open + 1)
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
) raises:
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
            # ⚠ Self-closing `<default class="x"/>`: an EMPTY class, which is
            # legal MJCF and is what PyMJCF emits for its root (`class="/"`).
            # It owns no block, so registering it and advancing past the tag is
            # the whole job — asking `_find_matching_default_close` for its
            # close would hand back the NEXT class's, and that class would then
            # be registered under the wrong name and skipped.
            if _is_self_closing_tag(text, dt):
                var sc_end = text.find(">", dt)
                if sc_end == -1:
                    break
                var sc_tag = _extract_opening_tag(text, dt)
                var sc_name = _extract_attr(sc_tag, "class")
                if sc_name.byte_length() > 0:
                    named.add(sc_name, DefaultsData(copy=par))
                scan = sc_end + 1
                continue
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


def _fill_assets(

    asset_sec: String,
    mut result: FlatModelDef,
) raises:
    """Parse <asset> section: fill result.textures[] and result.materials[]."""

    # ---- Textures -----------------------------------------------------------
    var tex_pos = 0
    var tex_count = 0
    while True:
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

        result.textures.append(td)
        tex_count += 1
        tex_pos = tag_end + 1

    # ---- Materials ----------------------------------------------------------
    var mat_pos = 0
    var mat_count = 0
    while True:
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

        result.materials.append(md)
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
            result.mesh_asset_names.append(mesh_name)
            result.mesh_asset_files.append(mesh_file)
            mesh_count += 1
        mesh_pos = tag_end + 1
    result.num_mesh_assets = mesh_count


# =============================================================================
# Phase 4b: Combined DFS scan — fills bodies, joints, geoms in one pass
# =============================================================================


def _parse_one_joint(
    worldbody: String,
    next_joint: Int,
    current_body: Int,
    inherited_class: String,
    defaults: DefaultsData,
    named_defaults: NamedDefaultsList,
    deg_factor: Float64,
) raises -> JointData:
    """Parse ONE `<joint>` opening tag into a `JointData`.

    Lifted out of `_fill_model`'s dispatch (lever 3, 2026-08-11): that function
    was a single ~1050-line `while` loop nested 7 deep, and `parse_xml_full` is
    most of the per-binary compile floor.

    `current_body` is `body_id_stack[depth]` and `inherited_class` is
    `childclass_stack[depth]` at the call site -- the caller owns the DFS
    stacks, this only reads the resolved values. Appending to `result` and
    advancing the cursor stay with the caller too, so this is pure.
    """
    var tag = _extract_opening_tag(worldbody, next_joint)

    var jd = JointData()
    jd.body_id = current_body

    # Effective defaults: the joint's own class="..." wins, else
    # the enclosing body's childclass, else the top-level block.
    # (Joints resolved NO class at all before 2026-07-29 — only
    # geoms did — so a class-defined joint silently became a
    # default hinge about the x axis.)
    var joint_class = _extract_attr(tag, "class")
    if joint_class.byte_length() == 0:
        joint_class = inherited_class
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

    # springdamper — element wins over the resolved class. Both
    # values must be > 0 for MuJoCo to act on them, so 0/0 is the
    # "absent" encoding and needs no separate flag.
    var sd_s = _extract_attr(tag, "springdamper")
    if sd_s.byte_length() > 0:
        var sdv = _parse_vec3(sd_s)
        jd.springdamper_0 = sdv[0]
        jd.springdamper_1 = sdv[1]
    else:
        jd.springdamper_0 = jdef.joint_springdamper_0
        jd.springdamper_1 = jdef.joint_springdamper_1

    # springref — deg→rad, HINGE ONLY.
    #
    # ⚠ THIS CONVERSION WAS MISSING and `range` two blocks up plus
    # `ref` just below both had it, which is what made the gap
    # invisible on a read. dog's jaw spells `springref="-11.0"`
    # (degrees, `-0.191986` rad) with `stiffness="2.0"`, so the
    # mandible spring pulled towards -11 RADIANS — a rest position 56
    # revolutions away — and the resulting passive torque wrecked the
    # whole solve. Measured against MuJoCo's `qpos_spring`: max|d| was
    # 10.808 rad, which is exactly `|-11 - (-0.191986)|`.
    #
    # The guard is `mjJNT_HINGE` alone, NOT hinge-or-ball
    # (`user_objects.cc:3276`, byte-identical in 3.3.6, 3.6.0 and
    # main). `ref` below uses hinge-or-ball; that is inert rather than
    # wrong, because MuJoCo rejects a non-zero `ref` on a ball joint
    # outright, so the extra branch can only ever scale a zero.
    #
    # The class default is scaled here too rather than at the
    # `<default>` block, for the same reason `range` is kept as a
    # STRING until this point: the conversion depends on the JOINT's
    # type, which a default block does not know.
    var sr_s = _extract_attr(tag, "springref")
    var sr_raw = (
        _parse_float(sr_s) if sr_s.byte_length() > 0
        else jdef.joint_springref
    )
    var sr_f = deg_factor if jd.jnt_type == JNT_HINGE else Float64(1.0)
    jd.springref = sr_raw * sr_f

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

    return jd


def _parse_one_geom(
    worldbody: String,
    next_geom: Int,
    current_body: Int,
    inherited_class: String,
    defaults: DefaultsData,
    named_defaults: NamedDefaultsList,
    deg_factor: Float64,
    eulerseq: String,
    assets: FlatModelDef,
) raises -> GeomData:
    """Parse ONE `<geom>` opening tag into a `GeomData`.

    Companion to `_parse_one_joint`; see it for why these were lifted.

    `assets` is the partially-built `FlatModelDef`, borrowed READ-ONLY and
    used for one thing: resolving `mesh="name"` against the asset tables that
    `_fill_assets` already populated. It is deliberately not `mut` -- the
    caller does the `result.geoms.append`.
    """
    var tag = _extract_opening_tag(worldbody, next_geom)

    var gd = GeomData()
    gd.body_id = current_body

    # Resolve effective defaults: the geom's own class="..." wins,
    # else the enclosing body's childclass, else top-level.
    var geom_class = _extract_attr(tag, "class")
    if geom_class.byte_length() == 0:
        geom_class = inherited_class
    var eff_defaults = defaults
    if geom_class.byte_length() > 0:
        eff_defaults = named_defaults.find(geom_class)

    # type
    var type_s = _extract_attr(tag, "type")
    if type_s.byte_length() == 0:
        type_s = eff_defaults.geom_type_s
    gd.geom_type = _geom_type_from_str(type_s)

    # mesh reference: mesh="name" → resolve to file path from asset section.
    # ⚠ Element first, then the class — the same precedence every other
    # attribute here uses. Reading the element ONLY is what left Jaco's six
    # finger geoms with `mesh_id -1`: they are bare `<geom name="..."/>` tags
    # that take type, mass and mesh from a `childclass`.
    if gd.geom_type == _GEOM_MESH:
        var mesh_attr = _extract_attr(tag, "mesh")
        if mesh_attr.byte_length() == 0:
            mesh_attr = eff_defaults.geom_mesh_s
        if mesh_attr.byte_length() > 0:
            for mi in range(assets.num_mesh_assets):
                if assets.mesh_asset_names[mi] == mesh_attr:
                    gd.mesh_id = mi
                    gd.mesh_filename = assets.mesh_asset_files[mi]
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
    # ⚠ PARTIAL `friction` KEEPS THE INHERITED COMPONENTS — see the
    # identical guard in the `<default>` block above. MuJoCo starts a
    # geom's friction from its class (ultimately the global
    # `1 0.005 0.0001`) and overwrites only what the attribute spells,
    # so `friction="0.9"` is `(0.9, 0.005, 0.0001)` and NOT
    # `(0.9, 0, 0)`.
    #
    # Currently INERT on every gated pose — the torsional and rolling
    # coefficients are read only at condim >= 4, and dog's condim-6
    # teeth spell all three values — but it is a wrong number in
    # `geom_friction` (86 of dog's 128 geoms) and would bite the first
    # condim >= 4 contact against a partially-specified geom.
    var fric_s = _extract_attr(tag, "friction")
    gd.friction = eff_defaults.geom_friction
    gd.friction_spin = eff_defaults.geom_friction_spin
    gd.friction_roll = eff_defaults.geom_friction_roll
    if fric_s.byte_length() > 0:
        var fparts = List[String]()
        _split_spaces(fric_s, fparts)
        if len(fparts) >= 1:
            gd.friction = _parse_float(fparts[0])
        if len(fparts) >= 2:
            gd.friction_spin = _parse_float(fparts[1])
        if len(fparts) >= 3:
            gd.friction_roll = _parse_float(fparts[2])

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

    # `priority` — when two geoms differ, the higher one dictates
    # condim, solref, solimp AND friction wholesale, with no mixing
    # (`engine_collision_driver.c:1427-1438`). Default 0.
    # ⚠ THE CLASS FALLBACK IS LOAD-BEARING, and it was missing.
    # `condim` on the line above has always had one; `priority`
    # took the element attribute or 0, full stop. quadruped's ball
    # writes `priority="1"` inline so the gap never showed, and
    # dog's 42 teeth write only `class="tooth_primitive"` — so all
    # 42 came out priority 0 and silently lost the condim-6,
    # friction and solref override they exist to impose.
    var prio_s = _extract_attr(tag, "priority")
    gd.priority = (
        _parse_int_str(prio_s) if prio_s.byte_length()
        > 0 else eff_defaults.geom_priority
    )

    # ⚠ `solmix` IS NOT SUPPORTED, AND IS REJECTED RATHER THAN
    # IGNORED. At equal priority MuJoCo blends the two geoms'
    # solref/solimp with `mix = solmix1/(solmix1+solmix2)`; every
    # geom defaults to `solmix=1`, giving mix = 0.5 (a plain mean),
    # which is what the mixing code implements. A model that
    # declares a non-default solmix would silently get the mean
    # instead of its intended weighting — the same silent-default
    # shape as the dof friction solparams, which raise for the same
    # reason. No dm_control suite model sets it.
    var solmix_s = _extract_attr(tag, "solmix")
    if solmix_s.byte_length() > 0:
        var sm = _parse_float(solmix_s)
        if sm < 0.999999 or sm > 1.000001:
            raise Error(
                "physics3d: <geom solmix> is not supported (only"
                " the default 1.0). At equal priority it weights"
                " the solref/solimp blend; ignoring it would"
                " silently substitute a plain mean."
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

    return gd


def _fill_model(

    worldbody: String,
    defaults: DefaultsData,
    named_defaults: NamedDefaultsList,
    mut result: FlatModelDef,
    deg_factor: Float64 = 1.0,
    eulerseq: String = "xyz",
) raises:
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
    # ⚠ Sized by MJCF NESTING DEPTH, not body count. It was `NBODY + 1` only
    # because NBODY was in scope; the stack is indexed by `depth`, which is how
    # deeply `<body>` elements nest. 128 is far beyond any real model (dog, the
    # deepest here, nests ~12) and the guard below makes an overflow loud.
    comptime _MAX_BODY_DEPTH = 128
    var body_id_stack = InlineArray[Int, _MAX_BODY_DEPTH](fill=0)
    # childclass_stack[depth] = default class inherited by elements at this
    # depth. MJCF's `childclass` applies to every descendant of the body that
    # declares it, until a deeper body overrides it; an element's own
    # `class=` still wins. Empty string = no inherited class.
    var childclass_stack = InlineArray[String, _MAX_BODY_DEPTH](
        fill=String("")
    )
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
        var next_inertial = worldbody.find("<inertial", scan_pos)

        # Check for no more interesting tokens
        var all_invalid = (
            next_body_open == -1
            and next_body_close == -1
            and next_joint == -1
            and next_geom == -1
            and next_light == -1
            and next_cam == -1
            and next_site == -1
            and next_inertial == -1
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
                _min_valid(next_site, next_inertial),
            ),
        )

        if earliest == next_body_open:
            # Opening <body ...>
            var tag = _extract_opening_tag(worldbody, next_body_open)
            var parent_id = body_id_stack[depth]
            var inherited_class = childclass_stack[depth]
            depth += 1
            if depth >= _MAX_BODY_DEPTH:
                raise Error(
                    "physics3d: <body> nesting deeper than 128; raise"
                    " _MAX_BODY_DEPTH in _fill_model. Continuing would index"
                    " the depth stacks out of bounds."
                )
            var this_body_id = body_count + 1  # model body index (worldbody=0)
            body_id_stack[depth] = this_body_id
            # `childclass` on this body replaces the inherited one for the
            # whole subtree; otherwise the parent's carries down.
            var cc_s = _extract_attr(tag, "childclass")
            childclass_stack[depth] = (
                cc_s if cc_s.byte_length() > 0 else inherited_class
            )

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

            result.bodies.append(b)
            body_count += 1
            # Advance past the opening tag
            var tag_end = worldbody.find(">", next_body_open)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

        elif earliest == next_body_close:
            # Closing </body>
            if depth > 0:
                depth -= 1
            scan_pos = next_body_close + 7  # len("</body>") == 7

        elif earliest == next_inertial:
            # <inertial ...> — an explicit inertia for the ENCLOSING body.
            #
            # Both parsers read `mass`/`diaginertia` off the `<body>` tag only,
            # which MJCF also allows, and ignored this child element entirely
            # until manipulator. Its `pinch site` body is a massless marker
            # with no geom, so the whole of its inertia arrives here: without
            # it the body took the geomless default of 1.0 kg instead of 1e-6,
            # a 6x overstatement of the hand subtree's mass.
            #
            # MuJoCo (`mjCBody::Compile`) treats an explicit <inertial> as
            # AUTHORITATIVE — it replaces the geom-derived inertia rather than
            # adding to it — which is what `has_explicit_inertia` already
            # means downstream.
            var tag = _extract_opening_tag(worldbody, next_inertial)
            var cur_body = body_id_stack[depth]
            if cur_body >= 1 and cur_body - 1 < len(result.bodies):
                # READ-MODIFY-WRITE: `result.bodies[i].field = x` on an
                # InlineArray subscript mutates a COPY and silently drops.
                var b = result.bodies[cur_body - 1]

                var im_s = _extract_attr(tag, "mass")
                if im_s.byte_length() > 0:
                    b.mass = _parse_float(im_s)
                    b.has_explicit_inertia = True

                var idi_s = _extract_attr(tag, "diaginertia")
                if idi_s.byte_length() > 0:
                    var dv = _parse_vec3(idi_s)
                    b.ixx = dv[0]
                    b.iyy = dv[1]
                    b.izz = dv[2]
                    b.has_explicit_inertia = True

                # `fullinertia` is the 6-vector (ixx iyy izz ixy ixz iyz).
                # `BodyData` stores a DIAGONAL inertia plus `iquat`, so a
                # genuinely off-diagonal one would need eigendecomposition —
                # raise rather than silently dropping the off-diagonal terms,
                # which would read as a mild dynamics divergence.
                var ifi_s = _extract_attr(tag, "fullinertia")
                if ifi_s.byte_length() > 0:
                    raise Error(
                        "physics3d: <inertial fullinertia=...> needs an"
                        " eigendecomposition into diaginertia + iquat, which"
                        " BodyData cannot express; only diaginertia is"
                        " implemented."
                    )

                var ip_s = _extract_attr(tag, "pos")
                if ip_s.byte_length() > 0:
                    var iv = _parse_vec3(ip_s)
                    b.ipos_x = iv[0]
                    b.ipos_y = iv[1]
                    b.ipos_z = iv[2]

                var iq = _orientation_to_quat(
                    _extract_attr(tag, "quat"),
                    _extract_attr(tag, "axisangle"),
                    _extract_attr(tag, "xyaxes"),
                    _extract_attr(tag, "zaxis"),
                    _extract_attr(tag, "euler"),
                    deg_factor,
                    eulerseq,
                )
                b.iquat_x = iq[0]
                b.iquat_y = iq[1]
                b.iquat_z = iq[2]
                b.iquat_w = iq[3]

                result.bodies[cur_body - 1] = b
            var tag_end = worldbody.find(">", next_inertial)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

        elif earliest == next_joint:
            # <joint ...>
            var jd = _parse_one_joint(
                worldbody,
                next_joint,
                body_id_stack[depth],
                childclass_stack[depth],
                defaults,
                named_defaults,
                deg_factor,
            )
            result.joints.append(jd)
            joint_count += 1
            var tag_end = worldbody.find(">", next_joint)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

        elif earliest == next_light:
            # <light ...>
            var current_body = body_id_stack[depth]
            var tag = _extract_opening_tag(worldbody, next_light)

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

            result.lights.append(ld)
            light_count += 1
            var tag_end = worldbody.find(">", next_light)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

        elif earliest == next_cam:
            # <camera ...>
            var current_body = body_id_stack[depth]
            var tag = _extract_opening_tag(worldbody, next_cam)

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

            result.cameras.append(cd)
            cam_count += 1
            var tag_end = worldbody.find(">", next_cam)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

        elif earliest == next_site:
            # <site ...>
            var current_body = body_id_stack[depth]
            var tag = _extract_opening_tag(worldbody, next_site)

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

            # `fromto` is valid on a SITE, not just a geom
            # (user_objects.cc:3841, mjCSite::Compile — the same block as
            # mjCGeom's). It supersedes both pos and the orientation
            # attributes, so it is resolved first and they are skipped.
            # Until 2026-08-01 sites ignored it entirely and kept
            # pos (0,0,0), which put quadruped's twenty `rf_*`
            # rangefinder sites at the body origin — up to 0.4 m out.
            var site_fromto_s = _extract_attr(tag, "fromto")
            if site_fromto_s.byte_length() > 0:
                var sft = _fromto_to_pos_quat(site_fromto_s)
                sd.pos_x = sft[0]
                sd.pos_y = sft[1]
                sd.pos_z = sft[2]
                sd.quat_x = sft[3]
                sd.quat_y = sft[4]
                sd.quat_z = sft[5]
                sd.quat_w = sft[6]
            else:
                # pos and orientation both fall back to the default class,
                # which until manipulator they did not: a site declaring
                # only `name` and `group` inside `class="hand"` kept local
                # pos (0,0,0) and identity orientation, when the class
                # gives it `pos=".022 0 -.002" euler="0 15 0"`.
                var pos_s = _extract_attr(tag, "pos")
                if pos_s.byte_length() == 0:
                    pos_s = site_defaults.site_pos_s
                if pos_s.byte_length() > 0:
                    var pv = _parse_vec3(pos_s)
                    sd.pos_x = pv[0]
                    sd.pos_y = pv[1]
                    sd.pos_z = pv[2]

                # Same precedence as geoms and bodies:
                # quat > axisangle > xyaxes > zaxis > euler.
                var quat_s = _extract_attr(tag, "quat")
                if quat_s.byte_length() == 0:
                    quat_s = site_defaults.site_quat_s
                var aa_s = _extract_attr(tag, "axisangle")
                if aa_s.byte_length() == 0:
                    aa_s = site_defaults.site_axisangle_s
                var xy_s = _extract_attr(tag, "xyaxes")
                if xy_s.byte_length() == 0:
                    xy_s = site_defaults.site_xyaxes_s
                var za_s = _extract_attr(tag, "zaxis")
                if za_s.byte_length() == 0:
                    za_s = site_defaults.site_zaxis_s
                var eu_s = _extract_attr(tag, "euler")
                if eu_s.byte_length() == 0:
                    eu_s = site_defaults.site_euler_s
                var sq = _orientation_to_quat(
                    quat_s, aa_s, xy_s, za_s, eu_s, deg_factor, eulerseq
                )
                sd.quat_x = sq[0]
                sd.quat_y = sq[1]
                sd.quat_z = sq[2]
                sd.quat_w = sq[3]

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

            # `fromto` OVERRIDES the size read above: MuJoCo sets
            # size[1] to half the segment length, and for a box or an
            # ellipsoid then shifts it (size[2]=size[1], size[1]=size[0]).
            # Done after the size attr so it wins regardless of order.
            if site_fromto_s.byte_length() > 0:
                var half_len = _fromto_to_pos_quat(site_fromto_s)[7]
                if (
                    sd.site_type == _GEOM_ELLIPSOID
                    or sd.site_type == _GEOM_BOX
                ):
                    sd.size_2 = half_len
                    sd.size_1 = sd.size_0
                else:
                    sd.size_1 = half_len

            result.sites.append(sd)
            site_count += 1
            var tag_end = worldbody.find(">", next_site)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

        else:  # earliest == next_geom
            # <geom ...>
            var gd = _parse_one_geom(
                worldbody,
                next_geom,
                body_id_stack[depth],
                childclass_stack[depth],
                defaults,
                named_defaults,
                deg_factor,
                eulerseq,
                result,
            )
            result.geoms.append(gd)
            geom_count += 1
            var tag_end = worldbody.find(">", next_geom)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen

    # ── Re-order joints, geoms and sites into MuJoCo's element order ────────
    #
    # The walk above emits in XML TEXT order. MuJoCo emits GROUPED BY BODY:
    # all of body 0's elements, then body 1's, and so on, with declaration
    # order preserved inside each body. The two coincide only when every body
    # declares its own joints/geoms/sites BEFORE its nested `<body>` children
    # — which every ported model happened to do until dm_control's dog, whose
    # `skull` declares its 42 teeth AFTER its child bodies.
    #
    # ⚠ THIS WAS A REAL BUG, NOT A COSMETIC MISMATCH. `fields_build` assigns
    # `qpos_adr`/`dof_adr` as running counters over the JOINT ARRAY, so a
    # permuted array permutes the whole `qpos` layout. On dog that made
    # `joint_angles` — 73 of the 223 observation dims — a permutation of
    # dm_control's, and it also made every per-index model comparison
    # meaningless: `max|d(jnt_range)| = 1e10`, because our joint at that index
    # was an unlimited one where MuJoCo's had a real range. The armature /
    # stiffness / dof_invweight0 "mismatches" were all this one permutation.
    # Sites matter for the same reason one level up: sensors are addressed BY
    # SITE INDEX, so a permuted site array reads the wrong sensor.
    #
    # Body ids are already assigned in DFS order (the walk numbers them at
    # `<body>` open), which is MuJoCo's body order — so a STABLE sort by
    # `body_id` reproduces MuJoCo's ordering exactly. Stability is what
    # preserves declaration order within a body, and a counting sort over body
    # ids is stable by construction.
    _stable_group_by_body_joints(result.joints)
    _stable_group_by_body_geoms(result.geoms)
    _stable_group_by_body_sites(result.sites)


# =============================================================================
# Phase 4b: element ordering
# =============================================================================


def _stable_group_by_body_joints(mut items: List[JointData]):
    """Stable counting sort of `items` by `body_id`.

    Written out per element type rather than made generic because `JointData`,
    `GeomData` and `SiteData` share no trait carrying `body_id`, and adding one
    would touch every construction site in the parser.

    The body range is derived from the items themselves rather than taken as a
    dimension parameter — that is what keeps this (and the parser around it)
    non-generic. See `FlatModelDef`'s docstring for why that matters.
    """
    if len(items) <= 1:
        return
    var max_body = 0
    for i in range(len(items)):
        if items[i].body_id > max_body:
            max_body = items[i].body_id
    var out = List[JointData]()
    for b in range(max_body + 1):
        for i in range(len(items)):
            if items[i].body_id == b:
                out.append(items[i])
    # An item whose body_id fell outside [0, max_body] cannot exist by
    # construction, but a silent drop here would be the same class of bug this
    # function exists to fix — so the count is checked rather than assumed.
    if len(out) != len(items):
        return
    for i in range(len(items)):
        items[i] = out[i]


def _stable_group_by_body_geoms(mut items: List[GeomData]):
    """Stable counting sort of `items` by `body_id`."""
    if len(items) <= 1:
        return
    var max_body = 0
    for i in range(len(items)):
        if items[i].body_id > max_body:
            max_body = items[i].body_id
    var out = List[GeomData]()
    for b in range(max_body + 1):
        for i in range(len(items)):
            if items[i].body_id == b:
                out.append(items[i])
    if len(out) != len(items):
        return
    for i in range(len(items)):
        items[i] = out[i]


def _stable_group_by_body_sites(mut items: List[SiteData]):
    """Stable counting sort of `items` by `body_id`."""
    if len(items) <= 1:
        return
    var max_body = 0
    for i in range(len(items)):
        if items[i].body_id > max_body:
            max_body = items[i].body_id
    var out = List[SiteData]()
    for b in range(max_body + 1):
        for i in range(len(items)):
            if items[i].body_id == b:
                out.append(items[i])
    if len(out) != len(items):
        return
    for i in range(len(items)):
        items[i] = out[i]


def _fill_actuators(

    actuator_sec: String,
    worldbody: String,
    defaults: DefaultsData,
    mut result: FlatModelDef,
) raises:
    """Parse <actuator> section and populate result.actuators[]."""
    var act_count = 0
    var scan_pos = 0
    var alen = actuator_sec.byte_length()

    while scan_pos < alen:
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

        # Record WHICH tag this was. The gains themselves come from the OTHER
        # parser: `xml_parser`'s comptime `ComptimeActData` carries
        # `motor_kp`/`motor_kv` (MuJoCo's `gainprm[0]` / `-biasprm[2]`) and
        # `apply_actions` reads them from there. This struct carries the kind
        # alone, which is what `init_fields` needs to refuse a transmission
        # neither path models. See docs/DM_CONTROL_PORT.md (gap G3).
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

        result.actuators.append(ad)
        act_count += 1

        var tag_end = actuator_sec.find(">", earliest)
        scan_pos = tag_end + 1 if tag_end != -1 else alen


# =============================================================================
# Phase 5b: Parse <equality> section — weld and connect constraints
# =============================================================================


def _fill_equality_solparams(tag: String, mut ed: EqualityData) raises:
    """Read the attributes EVERY equality type shares: torquescale, solref,
    solimp.

    ⚠ EXTRACTED SO THERE IS EXACTLY ONE COPY. `_fill_equality`'s loop has
    three exits now (weld/connect body, connect site, joint), and the site
    branch originally `continue`d straight past this block — so every
    site-based equality silently took the MJCF defaults instead of its own
    solref/solimp. ToddlerBot's connects carry `solref="0.004 1"`, a far
    stiffer constraint than the 0.02/1 they were getting. A shared tail that
    any branch can skip is a defect waiting to be re-introduced; a call is not.
    """
    # torquescale (weld) — MuJoCo's eq_data[10], scaling BOTH the orientation
    # residual and the rotational Jacobian. Default 1.
    var ts_s = _trim(_extract_attr(tag, "torquescale"))
    if ts_s.byte_length() > 0:
        ed.torquescale = _parse_float(ts_s)

    var sr_s = _extract_attr(tag, "solref")
    if sr_s.byte_length() > 0:
        var sv = _parse_vec3(sr_s)
        ed.solref_0 = sv[0]
        ed.solref_1 = sv[1]

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


def _fill_equality(

    equality_sec: String,
    worldbody: String,
    mut result: FlatModelDef,
) raises:
    """Parse <equality> section: fill result.equalities[] with weld/connect data."""
    var eq_count = 0
    var scan_pos = 0
    var elen = equality_sec.byte_length()

    while scan_pos < elen:
        # Find next <weld, <connect or <joint tag.
        #
        # `<joint>` inside `<equality>` is `mjEQ_JOINT` — a coupling between
        # two scalar joints — and has nothing to do with `<worldbody>`'s
        # `<joint>`. `equality_sec` is the `<equality>` section only, so the
        # names cannot collide.
        var nw = equality_sec.find("<weld", scan_pos)
        var nc = equality_sec.find("<connect", scan_pos)
        var nj = equality_sec.find("<joint", scan_pos)

        var earliest = _min_valid(_min_valid(nw, nc), nj)
        if earliest == -1:
            break

        var tag = _extract_opening_tag(equality_sec, earliest)
        var ed = EqualityData()

        # Determine type
        if earliest == nw:
            ed.eq_type = _EQ_WELD
        elif earliest == nj:
            ed.eq_type = _EQ_JOINT
        else:
            ed.eq_type = _EQ_CONNECT

        # ── mjEQ_JOINT: q1 coupled to q2 by a quartic in (q2 - q2_ref) ───────
        #
        # A different shape from connect/weld — the objects are JOINTS, there
        # is no anchor, and the five `polycoef` values ride in the slots
        # connect/weld use for anchors. That is MuJoCo's own arrangement:
        # `eq_obj1id`/`eq_obj2id` and `eq_data[0:5]` are reused per `mjtEq`
        # (engine_core_constraint.c:556). Handled here and skipped by the
        # body/site machinery below.
        if ed.eq_type == _EQ_JOINT:
            ed.objtype = _EQ_OBJ_BODY  # unused for this type; keep it defined

            var j1_name = _extract_attr(tag, "joint1")
            if j1_name.byte_length() == 0:
                raise Error(
                    "physics3d: <equality><joint> requires joint1."
                )
            ed.body_a = _find_joint_index_by_name(worldbody, j1_name)
            if ed.body_a < 0:
                raise Error(
                    "physics3d: <equality><joint> references unknown"
                    " joint1='" + j1_name + "'."
                )

            # joint2 is OPTIONAL. Absent means "hold joint1 at its reference
            # plus polycoef[0]" — MuJoCo's `id[1] < 0` branch, which drops the
            # polynomial entirely. -1 is the marker, matching `eq_obj2id`.
            var j2_name = _extract_attr(tag, "joint2")
            if j2_name.byte_length() > 0:
                ed.body_b = _find_joint_index_by_name(worldbody, j2_name)
                if ed.body_b < 0:
                    raise Error(
                        "physics3d: <equality><joint> references unknown"
                        " joint2='" + j2_name + "'."
                    )
            else:
                ed.body_b = -1

            # polycoef — MJCF default "0 1 0 0 0", i.e. q1 tracks q2 one-to-one.
            # ⚠ The DEFAULT IS NOT ALL ZEROS. All-zero would pin q1 to its own
            # reference and ignore joint2 completely, which looks like a
            # working constraint and is a different one.
            ed.anchor_a_x = 0.0
            ed.anchor_a_y = 1.0
            ed.anchor_a_z = 0.0
            ed.anchor_b_x = 0.0
            ed.anchor_b_y = 0.0
            var pc_s = _extract_attr(tag, "polycoef")
            if pc_s.byte_length() > 0:
                var pc = List[String]()
                _split_spaces(pc_s, pc)
                if len(pc) >= 1:
                    ed.anchor_a_x = _parse_float(pc[0])
                if len(pc) >= 2:
                    ed.anchor_a_y = _parse_float(pc[1])
                else:
                    ed.anchor_a_y = 0.0
                if len(pc) >= 3:
                    ed.anchor_a_z = _parse_float(pc[2])
                if len(pc) >= 4:
                    ed.anchor_b_x = _parse_float(pc[3])
                if len(pc) >= 5:
                    ed.anchor_b_y = _parse_float(pc[4])

            _fill_equality_solparams(tag, ed)
            result.equalities.append(ed)
            eq_count += 1
            var j_end = equality_sec.find(">", earliest)
            scan_pos = j_end + 1 if j_end != -1 else elen
            continue

        # ── BODY vs SITE semantics (MuJoCo's `eq_objtype`) ───────────────────
        #
        # MJCF gives `connect` and `weld` two mutually exclusive spellings, and
        # `mjXReader::OneEquality` (xml_native_reader.cc:2118) validates the
        # choice rather than guessing:
        #
        #   connect: EITHER (body1 + anchor [+ body2])  OR  (site1 + site2)
        #   weld:    EITHER (body1 [+ body2, anchor, relpose])
        #                                                OR  (site1 + site2)
        #
        # Mixing the two is an error, and so is satisfying neither.
        #
        # ⚠ THE SITE FORM USED TO FALL THROUGH TO THE BODY FORM AND PRODUCE A
        # SELF-WELD. `body1`/`body2` are absent on a site-based equality, so
        # both indices stayed at their default 0 and the model got an equality
        # binding the WORLD TO ITSELF — three or six rows of pure zero. MuJoCo
        # rejects `obj1id == obj2id` outright ("element is repeated in equality
        # constraint"). Nothing caught it because the only weld in the tree
        # (sawyer's) is body-based.
        var b1_name = _extract_attr(tag, "body1")
        var b2_name = _extract_attr(tag, "body2")
        var s1_name = _extract_attr(tag, "site1")
        var s2_name = _extract_attr(tag, "site2")
        var anchor_s = _extract_attr(tag, "anchor")
        var relpose_s = _extract_attr(tag, "relpose")

        var has_s1 = s1_name.byte_length() > 0
        var has_s2 = s2_name.byte_length() > 0
        var has_b1 = b1_name.byte_length() > 0
        var has_b2 = b2_name.byte_length() > 0
        var has_anchor = anchor_s.byte_length() > 0
        var has_relpose = relpose_s.byte_length() > 0

        var maybe_site = has_s1 or has_s2
        # `relpose` counts as a body-side attribute on a weld only — connect
        # has no orientation half and so no relpose.
        var maybe_body = (
            has_b1
            or has_b2
            or has_anchor
            or (has_relpose and ed.eq_type == _EQ_WELD)
        )

        var kind = "weld" if ed.eq_type == _EQ_WELD else "connect"
        if maybe_site and maybe_body:
            raise Error(
                "physics3d: <equality><"
                + kind
                + "> mixes body and site semantics. Give EITHER body1 (+"
                " body2/anchor) OR site1 and site2, not both."
            )

        var site_semantic = has_s1 and has_s2
        # A connect needs its anchor to be body-semantic; a weld does not
        # (an absent anchor means the body origin).
        var body_semantic = has_b1 and (
            has_anchor if ed.eq_type == _EQ_CONNECT else True
        )
        if site_semantic == body_semantic:
            raise Error(
                "physics3d: <equality><"
                + kind
                + "> needs exactly one of: body1"
                + (" and anchor" if ed.eq_type == _EQ_CONNECT else "")
                + ", or site1 and site2."
            )

        if site_semantic:
            # ⚠ WELD + SITES IS STILL UNIMPLEMENTED, and raises for the same
            # reason `<connect>` used to: the three ORIENTATION rows compare
            # body quaternions, and a site carries its own `quat` on top of
            # its body's, so the residual needs `site_xmat` — which the
            # position reduction below does not give us. Doing it would mean
            # deriving the relpose from the SITE frames at qpos0. The
            # position half would work today; shipping half a weld is how the
            # spatial `<equality><tendon>` gap survived behind four comments
            # claiming it was handled.
            if ed.eq_type == _EQ_WELD:
                raise Error(
                    "physics3d: site-based <equality><weld site1= site2=> is"
                    " not implemented — the orientation rows need the SITE"
                    " frames (site_xmat), not the body frames. Use the"
                    " body form, or see the note at this raise."
                )

            # SITE SEMANTICS REDUCES EXACTLY TO THE BODY FORM. MuJoCo reads
            # `pos[j] = site_xpos[id[j]]` and takes the bodies from
            # `site_bodyid` (engine_core_constraint.c:459); FK defines
            # `site_xpos = xpos[body] + xmat[body] * site_pos`, which is the
            # same expression the body form builds from
            # `(body, anchor)`. So storing `(site_bodyid, site local pos)`
            # leaves the row builder and every solver path untouched.
            ed.objtype = _EQ_OBJ_SITE

            var s1 = _find_site_index_by_name(worldbody, s1_name)
            if s1 < 0 or s1 >= len(result.sites):
                raise Error(
                    "physics3d: <equality><"
                    + kind
                    + "> references unknown site1='"
                    + s1_name
                    + "'."
                )
            var s2 = _find_site_index_by_name(worldbody, s2_name)
            if s2 < 0 or s2 >= len(result.sites):
                raise Error(
                    "physics3d: <equality><"
                    + kind
                    + "> references unknown site2='"
                    + s2_name
                    + "'."
                )

            ed.body_a = result.sites[s1].body_id
            ed.anchor_a_x = result.sites[s1].pos_x
            ed.anchor_a_y = result.sites[s1].pos_y
            ed.anchor_a_z = result.sites[s1].pos_z
            ed.body_b = result.sites[s2].body_id
            ed.anchor_b_x = result.sites[s2].pos_x
            ed.anchor_b_y = result.sites[s2].pos_y
            ed.anchor_b_z = result.sites[s2].pos_z

        # ── body semantics ───────────────────────────────────────────────────
        #
        # ⚠ THE SITE BRANCH ABOVE FALLS THROUGH TO THE SHARED TAIL — it must
        # NOT `append` and `continue` here. It did at first, which jumped
        # clean over `solref` / `solimp` / `torquescale` below, so every
        # site-based equality silently took the MJCF defaults. ToddlerBot's
        # four connects all carry `solref="0.004 1"` and
        # `solimp="0.9999 0.9999 0.001 0.5 2"` — a far stiffer constraint than
        # the 0.02/1, 0.9/0.95 defaults they would have been given, and
        # nothing downstream could have told the difference.
        # `test_site_connect_leaves_eq_data_alone` pins solref/solimp against
        # the runtime for exactly this reason.
        if not site_semantic:
            ed.objtype = _EQ_OBJ_BODY

            # body1 / body2 — resolve names to indices
            if has_b1:
                ed.body_a = _find_body_index_by_name(worldbody, b1_name)

            if has_b2:
                ed.body_b = _find_body_index_by_name(worldbody, b2_name)

        # `anchor` — WHICH BODY IT ANCHORS DEPENDS ON THE TYPE.
        # `mj_equalityAnchors` (engine_core_constraint.c:561) is explicit:
        #
        #   CONNECT: pos1 = body1 * data[0:3],  pos2 = body2 * data[3:6]
        #   WELD:    pos1 = body1 * data[3:6],  pos2 = body2 * data[0:3]
        #            ("weld uses data+3*(1-j) for anchor")
        #
        # `data[0:3]` is the `anchor` attribute and `data[3:6]` is the relpose
        # POSITION, so on a weld the roles are SWAPPED relative to connect:
        # `anchor` rides on body2 and the relpose position rides on body1.
        #
        # ⚠ THIS USED TO PUT `anchor` ON BODY A FOR BOTH TYPES, and never set
        # `anchor_b` at all. Latent because no model in the tree gives a weld an
        # explicit `anchor` (sawyer's is `<weld body1="mocap" body2="hand"
        # solref="0.02 1"/>`), so both slots were 0 and the swap was invisible.
        if has_anchor:
            var av = _parse_vec3(anchor_s)
            if ed.eq_type == _EQ_WELD:
                ed.anchor_b_x = av[0]
                ed.anchor_b_y = av[1]
                ed.anchor_b_z = av[2]
            else:
                ed.anchor_a_x = av[0]
                ed.anchor_a_y = av[1]
                ed.anchor_a_z = av[2]

        # relpose (weld) — position + quaternion, 7 values "x y z qw qx qy qz".
        # The position half is body1's anchor (see above); the quaternion half
        # is the target relative orientation.
        #
        # ⚠ AN ABSENT `relpose`, OR ONE WHOSE QUATERNION IS ALL ZEROS, MEANS
        # "DERIVE IT FROM qpos0" — MJCF's default is literally `0 0 0 0 0 0 0`
        # and MuJoCo's compiler fills in the relative pose the two bodies
        # already have at the reference configuration. Verified against the
        # runtime: a body at z=0.3 welded to the world compiles to
        # `(0, 0, -0.3, 1, 0, 0, 0)`, and an EXPLICIT identity quaternion
        # (`relpose="0 0 0 1 0 0 0"`) is kept as identity. `relpose_w` is left
        # at 0 here so `compute_invweight0` can tell the two apart; it fills
        # the derived value in at qpos0, where the FK products already exist.
        # ⚠ `not site_semantic` guards a clobber. The position half of
        # `relpose` writes `anchor_a`, which on the site path already holds
        # site1's local offset. MJCF's schema gives `relpose` to `weld` only,
        # so a site-based equality carrying one is invalid input rather than a
        # real case — but "invalid input silently moves an anchor" is the kind
        # of thing that surfaces years later as a wrong model.
        if has_relpose and not site_semantic:
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

        _fill_equality_solparams(tag, ed)

        result.equalities.append(ed)
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


def _fill_tendon_equalities(

    equality_sec: String,
    tendon_sec: String,
    xml: String,
    mut result: FlatModelDef,
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
        if idx < 0 or idx >= len(result.tendons):
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


def _fill_tendons(

    tendon_sec: String,
    worldbody: String,
    mut result: FlatModelDef,
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

    while scan_pos < tlen:
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
            while True:
                var sp = inner.find("<site", spos)
                if sp == -1:
                    break
                var stag = _extract_opening_tag(inner, sp)
                var sname = _extract_attr(stag, "site")
                if sname.byte_length() > 0:
                    # ⚠ COUNT PAST THE CAP RATHER THAN STOPPING AT IT. Breaking
                    # out of the loop is what made this truncate in silence.
                    if td.num_sites >= TENDON_MAX_WRAPS:
                        td.wrap_overflow += 1
                    else:
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
            while True:
                var jp = inner.find("<joint", jpos)
                if jp == -1:
                    break
                var jtag = _extract_opening_tag(inner, jp)
                var jname = _extract_attr(jtag, "joint")
                if jname.byte_length() > 0:
                    # ⚠ COUNT PAST THE CAP RATHER THAN STOPPING AT IT — see the
                    # spatial branch above. dog's `caudal_extend` wraps 11.
                    if td.num_joints >= TENDON_MAX_WRAPS:
                        td.wrap_overflow += 1
                    else:
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

        # ⚠ RAISE, DO NOT TRUNCATE. Defect 17 was a bare `while n < 4` on the
        # comptime side: dog's `caudal_extend` wraps 11 joints, so seven were
        # dropped and the model ran with a third of its tail tendon. Nothing
        # said a word, and it took a driven rollout to notice. A model over the
        # cap now fails to BUILD, which is the only signal that cannot be
        # missed.
        if td.wrap_overflow > 0:
            raise Error(
                "physics3d: tendon '"
                + _trim(_extract_attr(open_tag, "name"))
                + "' declares "
                + String(
                    td.num_sites + td.wrap_overflow if is_spatial
                    else td.num_joints + td.wrap_overflow
                )
                + " wraps, over the TENDON_MAX_WRAPS cap of "
                + String(TENDON_MAX_WRAPS)
                + " — raise it in `gpu/constants.mojo` (and"
                + " MAX_COMPTIME_TENDON_WRAPS in `xml_parser.mojo`, which must"
                + " agree) rather than letting the tendon run truncated"
            )

        result.tendons.append(td)
        count += 1

        if body_end != -1:
            scan_pos = body_end + close_tag.byte_length()
        else:
            scan_pos = tendon_sec.find(">", earliest) + 1


def _fill_excludes(

    contact_sec: String,
    worldbody: String,
    mut result: FlatModelDef,
):
    """Parse <contact> section: fill result.excludes[] with body pair exclusions."""
    var ex_count = 0
    var scan_pos = 0
    var clen = contact_sec.byte_length()

    while scan_pos < clen:
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
                result.excludes.append(ExcludeData(b1, b2))
            else:
                result.excludes.append(ExcludeData(b2, b1))
            ex_count += 1

        var tag_end = contact_sec.find(">", ne)
        scan_pos = tag_end + 1 if tag_end != -1 else ne + 1


def _fill_pairs(

    contact_sec: String,
    worldbody: String,
    mut result: FlatModelDef,
) raises:
    """Parse `<contact><pair>`: fill result.pairs[] with predefined geom pairs.

    ⚠ EVERY PARAMETER DEFAULTS TO MuJoCo'S GLOBAL DEFAULT, NOT TO A VALUE MIXED
    FROM THE TWO GEOMS. `mjCPair::Compile` looks like it derives an omitted
    attribute from `geom1`/`geom2` (max condim, max friction, max margin,
    solmix-weighted solref/solimp) but `mjs_defaultPair` has already written
    concrete defaults into every field, so `mjuu_defined()` is true throughout
    and not one of those branches runs. Measured on the 3.10.0 runtime with two
    deliberately mismatched geoms: an attribute-less pair reports condim 3,
    friction 1.0 and solref 0.02 while the same two geoms colliding dynamically
    report condim 6, friction 1.5 and solref 0.0125. See `MODEL_PAIR_SIZE`.

    Everything this engine cannot represent is REJECTED here rather than
    silently dropped, following the `solmix` precedent:

      * `gap` — the three reference trees and the runtime disagree about what
        gap even does (`margin-gap` in 3.3.6/3.6.0/main, `includemargin ==
        margin` measured on 3.10.0, `margin + gap` in 3.11.0), and this engine
        models no gap at all.
      * anisotropic `friction` — `pair_friction` is a FIVE-vector filled
        positionally, so `friction=".7"` leaves `friction[1]` at its default 1.0
        and means an ELLIPTIC cone. Our contact record carries one sliding
        coefficient for both tangent directions and one rolling coefficient for
        both. Every `<pair friction=...>` in Menagerie is isotropic (`"1 1"`,
        `"2 2 0.01 0.0001 0.0001"`), so this rejects nothing that exists today.
    """
    var scan_pos = 0
    var clen = contact_sec.byte_length()

    while scan_pos < clen:
        var np = contact_sec.find("<pair", scan_pos)
        if np == -1:
            break
        var tag = _extract_opening_tag(contact_sec, np)

        var g1_name = _trim(_extract_attr(tag, "geom1"))
        var g2_name = _trim(_extract_attr(tag, "geom2"))
        if g1_name.byte_length() == 0 or g2_name.byte_length() == 0:
            raise Error(
                "physics3d: <contact><pair> requires both geom1 and geom2."
            )

        var g1 = _find_geom_index_by_name(worldbody, g1_name)
        if g1 < 0:
            raise Error(
                "physics3d: <contact><pair> references unknown geom1='"
                + g1_name
                + "'."
            )
        var g2 = _find_geom_index_by_name(worldbody, g2_name)
        if g2 < 0:
            raise Error(
                "physics3d: <contact><pair> references unknown geom2='"
                + g2_name
                + "'."
            )

        # MuJoCo's compiler SORTS the two geoms — declaring `geom1="b"
        # geom2="a"` still yields pair_geom1 < pair_geom2 (measured). The
        # duplicate-suppression test in the detection loops compares an
        # ordered (gi, gj) against this record, so the order has to be the
        # same one the loops iterate in.
        var pd = PairData(g1, g2) if g1 <= g2 else PairData(g2, g1)

        var gap_s = _trim(_extract_attr(tag, "gap"))
        if gap_s.byte_length() > 0 and _parse_float(gap_s) != 0.0:
            raise Error(
                "physics3d: <contact><pair gap=> is not supported (this"
                " engine models no contact gap, and MuJoCo 3.3.6/3.6.0,"
                " 3.10.0 and 3.11.0 disagree about its meaning). Remove the"
                " attribute or extend the contact record."
            )

        var condim_s = _trim(_extract_attr(tag, "condim"))
        if condim_s.byte_length() > 0:
            pd.condim = Int(_parse_float(condim_s))
            if (
                pd.condim != 1
                and pd.condim != 3
                and pd.condim != 4
                and pd.condim != 6
            ):
                raise Error(
                    "physics3d: invalid condim in <contact><pair> (must be"
                    " 1, 3, 4 or 6)."
                )

        var fr_s = _extract_attr(tag, "friction")
        if fr_s.byte_length() > 0:
            # Positional fill over MuJoCo's five-vector
            # [slide1, slide2, spin, roll1, roll2]; anything not given keeps
            # the default, which is what makes a lone value anisotropic.
            var f0 = 1.0
            var f1 = 1.0
            var f2 = 0.005
            var f3 = 0.0001
            var f4 = 0.0001
            var fv = List[String]()
            _split_spaces(fr_s, fv)
            if len(fv) >= 1:
                f0 = _parse_float(fv[0])
            if len(fv) >= 2:
                f1 = _parse_float(fv[1])
            if len(fv) >= 3:
                f2 = _parse_float(fv[2])
            if len(fv) >= 4:
                f3 = _parse_float(fv[3])
            if len(fv) >= 5:
                f4 = _parse_float(fv[4])
            if f0 != f1 or f3 != f4:
                raise Error(
                    "physics3d: anisotropic <contact><pair friction=> is not"
                    " supported — friction[0] must equal friction[1] and"
                    " friction[3] must equal friction[4]. Note MuJoCo fills"
                    " this five-vector POSITIONALLY, so a single value such"
                    " as friction='0.7' leaves friction[1] at its default"
                    " 1.0 and is anisotropic."
                )
            pd.friction = f0
            pd.friction_spin = f2
            pd.friction_roll = f3

        var sr_s = _extract_attr(tag, "solref")
        if sr_s.byte_length() > 0:
            var sv = List[String]()
            _split_spaces(sr_s, sv)
            if len(sv) >= 1:
                pd.solref_0 = _parse_float(sv[0])
            if len(sv) >= 2:
                pd.solref_1 = _parse_float(sv[1])

        var si_s = _extract_attr(tag, "solimp")
        if si_s.byte_length() > 0:
            var iv = List[String]()
            _split_spaces(si_s, iv)
            if len(iv) >= 1:
                pd.solimp_0 = _parse_float(iv[0])
            if len(iv) >= 2:
                pd.solimp_1 = _parse_float(iv[1])
            if len(iv) >= 3:
                pd.solimp_2 = _parse_float(iv[2])
            if len(iv) >= 4:
                pd.solimp_3 = _parse_float(iv[3])
            if len(iv) >= 5:
                pd.solimp_4 = _parse_float(iv[4])

        var mg_s = _trim(_extract_attr(tag, "margin"))
        if mg_s.byte_length() > 0:
            pd.margin = _parse_float(mg_s)

        result.pairs.append(pd)

        var tag_end = contact_sec.find(">", np)
        scan_pos = tag_end + 1 if tag_end != -1 else np + 1


def _resolve_geom_materials(

    worldbody: String,
    asset_sec: String,
    mut result: FlatModelDef,
):
    """Resolve material="name" on geoms → material index; copy material rgba."""
    var scan_pos = 0
    var geom_idx = 0
    var wlen = worldbody.byte_length()

    while scan_pos < wlen and geom_idx < len(result.geoms):
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
            if not has_explicit_rgba and mid >= 0 and mid < len(result.materials):
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


def parse_xml_full(
xml_in: String) raises -> FlatModelDef:
    """Full MJCF parse: returns a populated FlatModelDef.

    ⚠ NON-GENERIC since 2026-08-05. It used to take the fourteen dimensions as
    comptime parameters purely because `FlatModelDef` stored its output in
    `InlineArray`s sized by them — so every distinct model instantiated a fresh
    copy of this ~2900-line function. That was 94% of the build time
    (`docs/DM_CONTROL_PORT_PHASE2.md` §15): 1961 s at dm_control dog's
    dimensions, and a ~344 s floor even for a 2-geom model. Now it compiles
    ONCE for the whole program.

        var fmd = parse_xml_full(xml)
        # counts live in the Lists: len(fmd.bodies), len(fmd.joints), ...

    The caller still needs `parse_xml(xml)` for the COMPTIME dimensions that
    size `fields.Model` — those are unchanged. What went away is passing them
    back down into the parser, which never used them for anything but capacity.

    The NTEX/NMAT/NLIGHT/NCAM/NSITE parameters default to 0 for backward
    compatibility — existing callers omitting them get no visual element arrays.
    All operations are comptime-safe (String.find + slice arithmetic only).
    """
    var result = FlatModelDef()

    # ⚠ STRIP COMMENTS FIRST. Everything below is `find` + slice arithmetic
    # over the raw text, so a commented-out element is indistinguishable from a
    # live one — `<!-- <site name='tip' pos='.15 0 .11'/> -->` in Gymnasium's
    # `half_cheetah.xml` was parsed as a REAL site, giving nsite 1 where MuJoCo
    # reports 0.
    #
    # The comptime `xml_parser` has stripped comments all along
    # (`_strip_xml_comments`), which is the whole hazard of having two parsers:
    # they disagreed, the comptime one was right, and the runtime one silently
    # wrote a site nobody had declared. It stayed invisible because the old
    # capacity-bounded writes dropped the overflow without a word; the
    # dimension check in `ModelDefFromXML` now turns exactly this into a raise.
    var xml = _strip_xml_comments(xml_in)

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
            # boundmass / boundinertia — see `FlatModelDef.boundmass`.
            var bm_s = _trim(_extract_attr(ctag, "boundmass"))
            if bm_s.byte_length() > 0:
                result.boundmass = _parse_float(bm_s)
            var bi_s = _trim(_extract_attr(ctag, "boundinertia"))
            if bi_s.byte_length() > 0:
                result.boundinertia = _parse_float(bi_s)

    # Assets: textures and materials
    _fill_assets(asset_sec, result)

    # Single DFS pass: bodies + joints + geoms + lights + cameras + sites
    _fill_model(worldbody, defaults, named_defaults, result, deg_factor, eulerseq)

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
        for gi in range(len(result.geoms)):
            result.geoms[gi].contype = 0
            result.geoms[gi].conaffinity = 0

    if constraints_off:
        for ji in range(len(result.joints)):
            result.joints[ji].range_min = Float64(-1e10)
            result.joints[ji].range_max = Float64(1e10)

    # Actuators
    _fill_actuators(actuator_sec, worldbody, defaults, result)

    # Equality constraints
    _fill_equality(equality_sec, worldbody, result)
    # Tendons
    _fill_tendons(_extract_section(xml, "tendon"), worldbody, result)
    # AFTER the tendons exist — this marks them by name. Note it is NOT
    # gated on NEQ: a tendon equality does not occupy an EqualityData
    # slot (it lives on the tendon record), so quadruped has neq==0 while
    # declaring four of them.
    _fill_tendon_equalities(equality_sec, _extract_section(xml, "tendon"), xml, result)
    # Contact exclusion pairs
    _fill_excludes(contact_sec, worldbody, result)
    # Predefined contact pairs — resolved by GEOM name, so this must run
    # after the worldbody walk has grouped geoms by body.
    _fill_pairs(contact_sec, worldbody, result)
    # Post-pass: resolve geom material="name" references
    _resolve_geom_materials(worldbody, asset_sec, result)

    return result^
