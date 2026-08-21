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

# `mjuu_eig3`, for <inertial fullinertia>. `parse_xml_full` is a RUNTIME
# function (non-generic since 2026-08-05), so pulling in a `std.math`-using
# helper here does not violate the comptime constraint the docstring above
# describes — that constraint belongs to `xml_parser.parse_xml`.
from ..model.inertia_from_geom import eig3_symmetric, geom_volume
from .xml_parser import (
    _split_spaces,
    _strip_xml_comments,
    _normalize_freejoint,
    _extract_section,
    _extract_section_all,
    _file_stem,
    _extract_section_inner,
    _extract_opening_tag,
    _extract_attr,
    _trim,
    _parse_float,
    _nth_float,
    _find_tag,
    _parse_int_str,
    _parse_vec3,
    _parse_quat,
    _parse_axisangle_to_quat,
    _parse_euler_to_quat,
    _parse_zaxis_to_quat,
    _compiler_deg_factor,
    _last_compiler_attr,
    _fromto_to_pos_quat,
    _find_joint_index_by_name,
    _find_body_index_by_name,
    _find_site_index_by_name,
    _find_geom_index_by_name,
    names_in_element_order,
    body_names_in_order,
    _sqrt_f64,
)
from ..types import ConeType, SolverType, IntegratorType
from .flat_model import (
    BodyData,
    JointData,
    GeomData,
    ActuatorData,
    ACT_KIND_MOTOR,
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
from mojo_rl.physics3d.gpu.constants import (
    TENDON_MAX_WRAPS,
    TENDON_MAX_SPATIAL_WRAPS,
    WRAP_SITE,
    WRAP_SPHERE,
    WRAP_CYLINDER,
    MJ_CCD_TOLERANCE,
    MJ_CCD_ITERATIONS,
)
from mojo_rl.physics3d.gpu.constants import JOINT_RANGE_UNLIMITED
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


def _lower_ascii(s: String) -> String:
    """ASCII lowercase, for the `<option>` values MuJoCo writes capitalised."""
    var out = String("")
    for i in range(s.byte_length()):
        var c = Int(s.as_bytes()[i])
        if c >= ord("A") and c <= ord("Z"):
            out += chr(c + 32)
        else:
            out += chr(c)
    return out


def _parse_option(
    xml: String,
) -> Tuple[
    Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,
    Int, Float64, Int, Int, Int
]:
    """Extract (gravity_x, gravity_y, gravity_z, timestep, density, viscosity,
    noslip_tolerance, ccd_tolerance, ccd_iterations, impratio, cone, solver,
    integrator) from <option .../>.

    Defaults: gravity=(0,0,-9.81), timestep=0.002, density=0.0, viscosity=0.0,
    noslip_tolerance=1e-6, ccd_tolerance=1e-6, ccd_iterations=35,
    impratio=1.

    ⚠ `ccd_tolerance` / `ccd_iterations` ARE EXERCISED, unlike
    `noslip_tolerance` below. They set EPA's stopping rule, which decides which
    boundary face it settles on and therefore the contact NORMAL. We hardcoded
    1e-8 / 64 — TIGHTER than MuJoCo — until 2026-08-13, and tighter is not
    safer here: iterating past the reference walks away from its answer rather
    than toward it. Measured on Jaco `reach_site_features` pose 38, the
    cylinder-mesh normal sits 9.4e-3 from MuJoCo's at 1e-8 and 7.6e-3 at 1e-6.

    ⚠ MATCHING THE STOPPING RULE IS NECESSARY, NOT SUFFICIENT: our polytope
    expansion, face ordering and horizon construction all differ from
    `engine_collision_gjk.c`'s, so identical rules still stop on different
    faces. `test_epa_optimality_cylinder_mesh` gates the well-posed quantity
    instead.

    `noslip_tolerance` is not the solver's `tolerance`. It is the threshold
    `mj_solNoSlip` compares its scaled per-iteration improvement against, so it
    decides how many of the `noslip_iterations` sweeps actually run.
    dm_control's manipulation models set it to **0**, meaning "never stop
    early" — the loop then runs the full count unless improvement goes
    negative. We hardcoded MuJoCo's 1e-6 default until 2026-08-13.

    ⚠ PARSED FOR FIDELITY, NOT BECAUSE A DIVERGENCE WAS MEASURED, and saying so
    because an earlier draft of this comment claimed 4.2e-2 of qacc from a
    CONFOUNDED experiment (two rollouts settled separately, so 400 steps of
    divergence were attributed to the attribute). Re-measured with a shared
    settled state, MuJoCo against itself with only this attribute changed:
    **8.9e-10** worst over 20 steps on the elliptic slam fixture, 4.3e-10 on a
    plain sliding chain, **0.0** on `reach_site_features`, at 5, 20 and 50
    iterations alike. So no fixture available today can tell 0 from 1e-6.

    It is parsed anyway because the algorithm reads it, the model declares it,
    and substituting a different value is a divergence by construction — but a
    reader should know this line is unexercised rather than assume a gate
    covers it.

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
    var nstol = Float64(1e-6)
    var ccdtol = Float64(MJ_CCD_TOLERANCE)
    var ccditer = MJ_CCD_ITERATIONS
    var impratio = 1.0
    var cone = ConeType.PYRAMIDAL
    var solver = SolverType.NEWTON
    var integrator = IntegratorType.EULER

    var pos = xml.find("<option")
    if pos == -1:
        return (gx, gy, gz, ts, dens, visc, nstol, ccdtol, ccditer,
                impratio, cone, solver, integrator)

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

    # ⚠ `byte_length() > 0` is the presence test, NOT a truthiness test on the
    # value: `noslip_tolerance="0"` is the setting dm_control actually uses, so
    # a check that treated 0 as absent would restore the 1e-6 default on
    # exactly the models this exists for.
    var nstol_str = _extract_attr(tag, "noslip_tolerance")
    if nstol_str.byte_length() > 0:
        nstol = _parse_float(nstol_str)

    # `ccd_tolerance` / `ccd_iterations` — EPA's stopping rule. Unlike
    # `noslip_tolerance` a 0 is NOT a meaningful setting for either (it would
    # mean "iterate to the array cap on every pair" and "never iterate"), so
    # a non-positive value falls back to MuJoCo's default rather than being
    # copied verbatim.
    var ccdt_str = _extract_attr(tag, "ccd_tolerance")
    if ccdt_str.byte_length() > 0:
        var v = _parse_float(ccdt_str)
        if v > 0.0:
            ccdtol = v

    var ccdi_str = _extract_attr(tag, "ccd_iterations")
    if ccdi_str.byte_length() > 0:
        var vi = Int(_parse_float(ccdi_str))
        if vi > 0:
            ccditer = vi

    # ⚠⚠ `impratio` — READ BY FIVE SOLVERS AND WRITTEN BY NOBODY until now.
    # `fields_build` hardcoded `1.0` into `MODEL_META_IDX_IMPRATIO` while
    # `contact_solve`, `newton_solve` (x2), `cg_solve` and `island_pgs_solve`
    # all read that slot, so a model asking for anything else was silently
    # simulated at 1. Every model in the tree used 1, which is exactly why it
    # survived — `contact_solve.mojo`'s own note says "no gate here can move,
    # and none did".
    #
    # It is the ratio of frictional to normal constraint IMPEDANCE
    # (`engine_core_constraint.c:1886`): `R[1] = R[0]/impratio`, so
    # `mu = friction[0]*sqrt(R[1]/R[0]) = friction[0]/sqrt(impratio)`. With an
    # ELLIPTIC cone the normal force is coupled to the tangential through that
    # regularized `mu`, so getting it wrong moves the NORMAL force, not just
    # the friction.
    #
    # Measured, Boston Dynamics spot (`impratio="100"`), at its first
    # ground contact with the two engines in the same state and IDENTICAL
    # contact geometry (position, normal, dist, condim and friction all equal
    # to every printed digit):
    #     MuJoCo, impratio=100 : fn = 11.070, 10.256
    #     MuJoCo, impratio=1   : fn =  6.849,  6.342
    #     ours  (hardcoded 1)  : fn =  6.333,  5.837
    # i.e. 1.62x of a 1.75x normal-force deficit.
    #
    # ⚠ A NON-POSITIVE VALUE FALLS BACK TO 1 rather than being copied.
    # `R[1] = R[0]/impratio` divides by it, and `mu` takes its square root, so
    # 0 or a negative would be a division by zero or a NaN several call
    # frames away from the XML that caused it.
    var impr_str = _extract_attr(tag, "impratio")
    if impr_str.byte_length() > 0:
        var vr = _parse_float(impr_str)
        if vr > 0.0:
            impratio = vr

    # `cone` / `solver` / `integrator` — see the docstring. Unrecognised values
    # fall back to MuJoCo's default rather than guessing; MuJoCo's own compiler
    # rejects the file, which is not available to us here.
    var cone_s = _trim(_extract_attr(tag, "cone"))
    if cone_s == String("elliptic"):
        cone = ConeType.ELLIPTIC
    elif cone_s == String("pyramidal"):
        cone = ConeType.PYRAMIDAL

    # ⚠ CASE-INSENSITIVE: MuJoCo writes "PGS" / "CG" / "Newton" and "RK4" /
    # "implicitfast", and matches them without regard to case.
    var solver_s = _lower_ascii(_trim(_extract_attr(tag, "solver")))
    if solver_s == String("pgs"):
        solver = SolverType.PGS
    elif solver_s == String("cg"):
        solver = SolverType.CG
    elif solver_s == String("newton"):
        solver = SolverType.NEWTON

    var integ_s = _lower_ascii(_trim(_extract_attr(tag, "integrator")))
    if integ_s == String("rk4"):
        integrator = IntegratorType.RK4
    elif integ_s == String("implicitfast"):
        integrator = IntegratorType.IMPLICITFAST
    elif integ_s == String("implicit"):
        integrator = IntegratorType.IMPLICIT
    elif integ_s == String("euler"):
        integrator = IntegratorType.EULER

    return (gx, gy, gz, ts, dens, visc, nstol, ccdtol, ccditer,
            impratio, cone, solver, integrator)


# =============================================================================
# Phase 2: Parse <default> block
# =============================================================================


def _apply_actfrcrange(
    fr: String,
    fl_raw: String,
    mut limited: Bool,
    mut lo: Float64,
    mut hi: Float64,
):
    """`<joint actuatorfrcrange/actuatorfrclimited>` — MuJoCo's
    `jnt_actfrcrange`. The THIRD instance of the same "auto" rule, and it
    goes through a shared helper for the reason the other two now do.

    `actuatorfrclimited` defaults to "auto" = limited iff the range is
    DEFINED, `"0 0"` is the undefined marker, an explicit true/false wins.

    ⚠⚠ THIS IS NOT THE ACTUATOR'S `forcerange`. `mj_fwdActuation` clamps
    TWICE — `actuator_forcerange` on each actuator's own scalar force
    (`engine_forward.c:417`), and then

        clampVec(d->qfrc_actuator, m->jnt_actfrcrange, m->jnt_actfrclimited,
                 m->njnt, m->jnt_dofadr);                            // :477

    on the ACCUMULATED `qfrc_actuator`, per JOINT, at that joint's dof
    address. Having one is not having the other: on unitree_g1
    `actuator_forcelimited` is FALSE on all 29 actuators while
    `jnt_actfrclimited` is TRUE on 29 of 30 joints, so the joint clamp is the
    ONLY force limit that model has — and we implemented only the one it does
    not use.

    ⚠ THE CONSEQUENCE IS A SERVO WITH NO CEILING. g1's wrists declare
    `actuatorfrcrange="-5 5"` against a `kp=500` servo whose `ctrl` is a
    target ANGLE over a +-1.61 rad range, so a command at the end of the
    range asks for ~800 N.m and MuJoCo delivers 5. Driven by the studio's
    random policy from `qpos0`, ours moved the wrist 54x further than MuJoCo
    in a single step (-0.0686 rad against -0.0013).
    """
    if fr.byte_length() > 0:
        var parts = List[String]()
        _split_spaces(fr, parts)
        if len(parts) >= 2:
            lo = _parse_float(parts[0])
            hi = _parse_float(parts[1])
            limited = lo != 0.0 or hi != 0.0
    var fl = _trim(fl_raw)
    if fl == "true" or fl == "1":
        limited = True
    elif fl == "false" or fl == "0":
        limited = False


def _apply_ctrlrange(
    cr: String,
    cl_raw: String,
    mut limited: Bool,
    mut lo: Float64,
    mut hi: Float64,
):
    """MuJoCo's ctrlrange/ctrllimited resolution — the twin of
    `_apply_forcerange`, and shared by the SAME two callers for the same
    reason.

    `ctrllimited` defaults to "auto" = limited iff the range is DEFINED, and
    `"0 0"` is the undefined marker: an explicit `ctrlrange="0 0"` still
    reports ctrllimited 0 (`test_ctrllimited_vs_mujoco`'s `a5`). An explicit
    true/false overrides, in all four spellings MJCF admits.

    Writes nothing when the attribute is absent, so a class inherits its
    parent's value rather than being reset.

    ⚠⚠ THIS EXISTS BECAUSE THE RULE WAS WRITTEN INLINE TWICE AND ONLY ONE
    COPY GOT IT. The `<default>` block read `ctrlrange` into the class's
    min/max and left `ctrl_limited` alone, so a range stated in a class —
    which is where Menagerie states nearly all of them — produced a bound
    that NOTHING CLAMPED AGAINST. `apply_actions_fields` guards its clamp on
    `ACT_IDX_CTRL_LIMITED` (correctly: an unlimited actuator must not be
    squeezed into the fallback range), so the range was carried all the way
    to the force law and then ignored.

    ⚠ THE CONSEQUENCE IS A SERVO COMMANDED OUTSIDE THE RANGE THE FILE GAVE
    IT, which is the SAME failure `inheritrange` was implemented to fix and
    is invisible for the same reason: it only bites when `ctrl` is actually
    out of range. Measured on `google_barkour_vb`, whose knee class says
    `<general ctrlrange="0.1 2.34346"/>` while `ctrl` and `qpos0` are both 0
    — MuJoCo clamps to 0.1 and delivers `actuator_force` 5 N.m on each of the
    four knees; we delivered 0, on all twelve actuators, every step.

    ⚠ `autolimits` IS ASSUMED TRUE, matching MuJoCo's own default and the
    element path's long-standing behaviour. No model in the tree sets
    `autolimits="false"` (audited: zero files, Menagerie included).
    """
    if cr.byte_length() > 0:
        var parts = List[String]()
        _split_spaces(cr, parts)
        if len(parts) >= 2:
            lo = _parse_float(parts[0])
            hi = _parse_float(parts[1])
            limited = lo != 0.0 or hi != 0.0
    var cl = _trim(cl_raw)
    if cl == "true" or cl == "1":
        limited = True
    elif cl == "false" or cl == "0":
        limited = False


def _apply_forcerange(
    fr: String,
    fl_raw: String,
    mut limited: Bool,
    mut lo: Float64,
    mut hi: Float64,
):
    """MuJoCo's forcerange/forcelimited resolution, mirrored from
    `xml_parser.parse_xml_model_data` (the comptime twin) so the two agree.

    `forcelimited` defaults to "auto" = limited iff the range is DEFINED, and
    `"0 0"` is the undefined marker: an explicit `forcerange="0 0"` still
    reports forcelimited 0. An explicit true/false overrides. MuJoCo refuses
    `forcelimited="true"` with no range, so that combination cannot arrive.

    Writes nothing when the attribute is absent, so a class inherits its
    parent's value rather than being reset to zero.
    """
    if fr.byte_length() > 0:
        var parts = List[String]()
        _split_spaces(fr, parts)
        if len(parts) >= 2:
            lo = _parse_float(parts[0])
            hi = _parse_float(parts[1])
            limited = lo != 0.0 or hi != 0.0
    var fl = _trim(fl_raw)
    if fl == "true" or fl == "1":
        limited = True
    elif fl == "false" or fl == "0":
        limited = False


def _parse_one_default_block(defaults_sec: String, parent: DefaultsData) -> DefaultsData:
    """Parse joint/geom/motor attrs from a default section, inheriting from parent."""
    var d = parent  # start with parent defaults

    # ── `<default><mesh scale="x y z"/></default>` ────────────────────────
    # ⚠⚠ THE ONLY PLACE 38 MENAGERIE DECLARATIONS PUT IT. A model whose STLs
    # are in millimetres writes the scale ONCE here and leaves all 48 of its
    # `<mesh file=.../>` assets bare — robotis_op3 does exactly that. Reading
    # `scale` off the asset's own tag therefore finds nothing and the hulls
    # come out 1000x oversized, which is the failure this block exists for.
    var mshpos = defaults_sec.find("<mesh")
    if mshpos != -1:
        var mshtag = _extract_opening_tag(defaults_sec, mshpos)
        var msc_s = _extract_attr(mshtag, "scale")
        if msc_s.byte_length() > 0:
            d.mesh_scale_s = msc_s

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

        # `actuatorfrcrange` — see `_apply_actfrcrange`. Menagerie states it
        # in a class as often as on the element (aloha and berkeley_humanoid
        # in a class, g1 inline), so both paths must resolve it.
        _apply_actfrcrange(
            _extract_attr(jtag, "actuatorfrcrange"),
            _extract_attr(jtag, "actuatorfrclimited"),
            d.joint_actfrc_limited,
            d.joint_actfrc_min,
            d.joint_actfrc_max,
        )

        var fl_s = _extract_attr(jtag, "frictionloss")
        if fl_s.byte_length() > 0:
            d.joint_frictionloss = _parse_float(fl_s)

        var sr_s = _extract_attr(jtag, "springref")
        if sr_s.byte_length() > 0:
            d.joint_springref = _parse_float(sr_s)

        var srl_s = _extract_attr(jtag, "solreflimit")
        if srl_s.byte_length() > 0:
            var sv = _solref_into(
                srl_s, d.joint_solref_limit_0, d.joint_solref_limit_1
            )
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
            var sv = _solref_into(
                sr0_s, d.geom_solref_0, d.geom_solref_1
            )
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

    # Find the default actuator tag.
    #
    # ⚠ THIS SCANNED ONLY `<motor` UNTIL 2026-08-14, so a `<default>` block
    # declaring `<general>`, `<position>` or `<velocity>` was INVISIBLE here —
    # root block and named class alike. dm_control's dog and quadruped use
    # `<general>` exclusively (77 occurrences between them), so every one of
    # their actuator defaults was silently dropped and the fields below kept
    # their struct defaults. Measured by
    # `tests/physics3d/test_actuator_record_equivalence.mojo`: quadruped
    # ctrl_min 4/12 and ctrl_max 8/12 wrong, ctrl_limited wrong on 12/12 and
    # 38/38. `gear` looked correct only because 1.0 IS the struct default.
    #
    # ⚠ "ONE tag kind per block" WAS THE OLD CONTRACT, AND IT WAS WRONG. The
    # note here read "NO `<default>` block mixes actuator tags. If one ever
    # does, this silently takes `<motor>` and drops the rest" — a latent bug
    # filed against a future model. rainbow_robotics rby1 is that model, and
    # the loop below now walks them all. The suggested remedy ("carry a set
    # per ACT_KIND") turned out NOT to be MuJoCo's model either: it layers
    # every tag onto ONE record in document order. See
    # `DefaultsData.motor_gain`.
    # ⚠⚠ EVERY ACTUATOR TAG, IN DOCUMENT ORDER — NOT THE FIRST ONE FOUND.
    # This used to `find("<motor")`, then fall back to `<general>`, then
    # `<position>`, then `<velocity>`, and parse THAT ONE ALONE. The comment
    # here said "NO `<default>` block mixes actuator tags. If one ever does,
    # this silently takes `<motor>` and drops the rest" — and rainbow_robotics
    # rby1 does exactly that:
    #
    #     <motor    ctrllimited="true" ctrlrange="-100 100"/>
    #     <velocity ctrllimited="true"/>
    #     <position ctrllimited="true" kp="4000" kv="400"/>
    #
    # so `kp="4000" kv="400"` was dropped and all 24 position servos ran at
    # MuJoCo's base gain of 1. Measured: MuJoCo's `qfrc_actuator` at qpos0
    # saturates its joints at +-270/-120/-70/-40 N.m while ours delivered
    # ~0.1, worst |d| 654 N.m.
    #
    # ⚠ MuJoCo LAYERS THEM ONTO ONE RECORD rather than keeping a set per tag
    # kind — see `DefaultsData.motor_gain`. So the order matters and the
    # LAST tag to state a field wins, which is why this walks the section
    # instead of gathering the tags into slots.
    var _mscan = 0
    while True:
        var mpos = -1
        var _which = 0
        for _k in range(4):
            var needle = "<motor"
            if _k == 1:
                needle = "<general"
            elif _k == 2:
                needle = "<position"
            elif _k == 3:
                needle = "<velocity"
            var hit = defaults_sec.find(String(needle), _mscan)
            if hit != -1 and (mpos == -1 or hit < mpos):
                mpos = hit
                _which = _k
        if mpos == -1:
            break
        var mtag = _extract_opening_tag(defaults_sec, mpos)
        _mscan = mpos + 1

        # ctrlrange / ctrllimited (see `_apply_ctrlrange`). ⚠ THIS USED TO
        # READ THE RANGE AND LEAVE `motor_ctrl_limited` ALONE, so a class
        # `ctrlrange` gave a bound nothing clamped against — barkour's twelve
        # actuators, and every other model that states the range in a class.
        _apply_ctrlrange(
            _extract_attr(mtag, "ctrlrange"),
            _extract_attr(mtag, "ctrllimited"),
            d.motor_ctrl_limited,
            d.motor_ctrl_min,
            d.motor_ctrl_max,
        )

        # `gear` was missing here (and in the comptime twin) until 2026-07-29,
        # so a default-class gear silently actuated at 1.0. dm_control's
        # point_mass declares `<motor gear=".1"/>` this way — a 10x error.
        var mg_s = _extract_attr(mtag, "gear")
        if mg_s.byte_length() > 0:
            d.motor_gear = _parse_float(mg_s)

        # forcerange / forcelimited (phase 1a.1). `_apply_forcerange` holds the
        # shared "auto" rule so the block path and the element path cannot
        # drift — the two-parser split started as exactly that kind of
        # duplication.
        _apply_forcerange(
            _extract_attr(mtag, "forcerange"),
            _extract_attr(mtag, "forcelimited"),
            d.motor_force_limited,
            d.motor_force_min,
            d.motor_force_max,
        )

        # dyntype / dynprm, kept RAW. Absent leaves the parent's value so a
        # class inherits rather than resetting to "".
        var dt_s = _extract_attr(mtag, "dyntype")
        if dt_s.byte_length() > 0:
            d.motor_dyntype_s = dt_s
        var dp_s = _extract_attr(mtag, "dynprm")
        if dp_s.byte_length() > 0:
            d.motor_dynprm_s = dp_s

        # Gain attrs, raw. Absent leaves the parent's value so a class
        # inherits rather than resetting to "".
        var kp_s = _extract_attr(mtag, "kp")
        if kp_s.byte_length() > 0:
            d.motor_kp_s = kp_s
        var kv_s = _extract_attr(mtag, "kv")
        if kv_s.byte_length() > 0:
            d.motor_kv_s = kv_s
        var dr_s = _extract_attr(mtag, "dampratio")
        if dr_s.byte_length() > 0:
            d.motor_dampratio_s = dr_s
        var ir_s = _extract_attr(mtag, "inheritrange")
        if ir_s.byte_length() > 0:
            d.motor_inheritrange_s = ir_s
        var gt_s = _extract_attr(mtag, "gaintype")
        if gt_s.byte_length() > 0:
            d.motor_gaintype_s = gt_s
        var bt_s = _extract_attr(mtag, "biastype")
        if bt_s.byte_length() > 0:
            d.motor_biastype_s = bt_s
        var gp_s = _extract_attr(mtag, "gainprm")
        if gp_s.byte_length() > 0:
            d.motor_gainprm_s = gp_s
        var bp_s = _extract_attr(mtag, "biasprm")
        if bp_s.byte_length() > 0:
            d.motor_biasprm_s = bp_s

        # ── GAIN and DAMPING BIAS, layered per MuJoCo's own rules ─────────
        #
        # ⚠ EACH TAG WRITES WHAT ITS OWN KIND MEANS, and only when the
        # attribute is present — measured against the 3.10.0 runtime, one
        # row per rule:
        #
        #   <position kp kv>   gain = kp        bias2 = -kv
        #   <velocity kv>      gain = kv        bias2 = -kv   (BOTH)
        #   <general gainprm>  gain = gainprm[0]
        #   <general biasprm>  bias2 = biasprm[2]
        #   <motor ...>        neither — a motor has no gain or bias to give
        #
        # `<velocity>` writing the GAIN as well as the bias is what makes
        # rby1's wheels inherit `kv = 4000` from a `<position kp="4000">`
        # tag: the position tag sets gain 4000, and the wheel elements —
        # which state no `kv` — take `kv = gain`.
        if _which == 2:  # <position>
            if kp_s.byte_length() > 0:
                d.motor_gain = _parse_float(kp_s)
                d.motor_gain_set = True
            if kv_s.byte_length() > 0:
                d.motor_bias2 = -_parse_float(kv_s)
                d.motor_bias2_set = True
        elif _which == 3:  # <velocity>
            if kv_s.byte_length() > 0:
                var _vk = _parse_float(kv_s)
                d.motor_gain = _vk
                d.motor_gain_set = True
                d.motor_bias2 = -_vk
                d.motor_bias2_set = True
        elif _which == 1:  # <general>
            if gp_s.byte_length() > 0:
                d.motor_gain = _nth_float(gp_s, 0, 1.0)
                d.motor_gain_set = True
            if bp_s.byte_length() > 0:
                d.motor_bias2 = _nth_float(bp_s, 2, 0.0)
                d.motor_bias2_set = True

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
    """`<texture type=>`. ⚠ THE DEFAULT IS `cube`, NOT `2d`.

    MuJoCo's XMLreference.rst:1644 gives `[2d, cube, skybox], "cube"`. Both
    our parsers returned 2d for an absent `type`, so every texture that does
    not name one was mistyped — quadruped's `ball` is exactly that
    (`<texture name="ball" builtin="checker" mark="cross" .../>`), and MuJoCo
    reports `tex_type = 1` (cube) against our 2d.

    Found by the MuJoCo parity gate in phase 1a.5c, i.e. by the gate written
    to REPLACE the consistency gate that was about to be deleted with `_rcd`.
    The consistency gate could never have found it: both parsers had the same
    wrong default and agreed perfectly
    (`feedback_a_gate_that_shares_its_reference_implementation_is_blind`).
    """
    var t = _trim(s)
    if t == "skybox":
        return TEX_SKYBOX
    elif t == "2d":
        return TEX_2D
    return TEX_CUBE


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


def _solref_into(
    s: String, cur0: Float64, cur1: Float64
) -> Tuple[Float64, Float64]:
    """`solref`, overwriting ONLY the components the attribute supplies.

    ⚠⚠ A PARTIAL `solref` KEEPS ITS OTHER COMPONENT, and every site here used
    `_parse_vec3`, which returns 0 for anything the string does not contain.
    So `solref="0.01"` — one value, which MJCF allows — compiled to
    `(0.01, 0.0)`, and the second component is the DAMPRATIO.

    `contact_solve` builds the constraint stiffness as
    `K = 1/(dmax^2 * timeconst^2 * dampratio^2)`, so a zero dampratio is a
    DIVISION BY ZERO dressed up as a number. Measured on Menagerie's
    trossen_wxai, whose gripper pads declare `solref="0.01"`: the contact
    normal force came out 7.3e13 N and `qacc` 5.6e13 rad/s^2, against MuJoCo's
    102 N and 642 rad/s^2. Every solver — PGS, Newton, both cones — produced
    the same explosion, because none of them was the problem.

    ⚠ THE RULE IS MuJoCo'S, MEASURED ON THE 3.10.0 RUNTIME rather than
    assumed, on a bare sphere:

        (absent)            -> solref (0.02, 1)
        solref="0.01"       -> solref (0.01, 1)     <- component 1 KEPT
        solref="0.01 0.5"   -> solref (0.01, 0.5)
        solimp="0.8"        -> solimp (0.8, 0.95, 0.001, 0.5, 2)

    i.e. supplied components overwrite, omitted ones keep whatever they had —
    the default, or the value inherited from the `<default>` class. The
    `solimp` readers in this file already do exactly that, one component at a
    time; only `solref` took the whole vector.
    """
    var parts = List[String]()
    _split_spaces(_trim(s), parts)
    var a = cur0
    var b = cur1
    if len(parts) >= 1 and parts[0].byte_length() > 0:
        a = _parse_float(parts[0])
    if len(parts) >= 2:
        b = _parse_float(parts[1])
    return (a, b)


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
    #
    # ⚠⚠ THE VECTOR PART WAS NEGATED HERE — this returned the CONJUGATE, i.e.
    # the INVERSE rotation. The frame's axes are the COLUMNS of R, so
    # R[i][j] = (axis_j)_i and therefore R[2][1] = y_z (`yz`) while
    # R[1][2] = z_y (`zy`); the standard qx = (R[2][1] - R[1][2]) is
    # `yz - zy`, and every branch had the operands the other way round.
    # Consistently so — which is why it produced a plausible-looking unit
    # quaternion that simply rotated the wrong way.
    #
    # ⚠ THE COMPTIME TWIN WAS FIXED FOR EXACTLY THIS AND THE FIX NEVER
    # CROSSED. `xml_parser._rcd_xyaxes_to_quat` carries the same comment and
    # the corrected operand order; this copy kept the bug. Verified against
    # MuJoCo 3.10.0 on our own merged XML: for all four quadruped and all
    # five fish cameras, `mjModel.cam_quat` matches the comptime result and
    # is the conjugate of what this returned.
    #
    # ⚠ WHY NO PHYSICS GATE CAUGHT IT: `xyaxes` appears on CAMERAS ONLY —
    # every occurrence in `mojo_rl/envs/` and in every `.xml` asset is a
    # `<camera>`. Bodies, geoms and sites reach this through
    # `_orientation_to_quat` but none of them ever supplies the attribute, so
    # the wrong branch was never taken by anything the dm_control gates
    # measure. That makes this a RENDER-ONLY defect today and a silent
    # physics defect the first time a model orients a body this way.
    var trace = xx + yy + zz
    var qx: Float64
    var qy: Float64
    var qz: Float64
    var qw: Float64

    if trace > 0.0:
        var s2 = _sqrt_f64(trace + 1.0) * 2.0  # s2 = 4*qw
        qw = 0.25 * s2
        qx = (yz - zy) / s2
        qy = (zx - xz) / s2
        qz = (xy - yx) / s2
    elif xx > yy and xx > zz:
        var s2 = _sqrt_f64(1.0 + xx - yy - zz) * 2.0  # s2 = 4*qx
        qw = (yz - zy) / s2
        qx = 0.25 * s2
        qy = (xy + yx) / s2
        qz = (xz + zx) / s2
    elif yy > zz:
        var s2 = _sqrt_f64(1.0 + yy - xx - zz) * 2.0  # s2 = 4*qy
        qw = (zx - xz) / s2
        qx = (xy + yx) / s2
        qy = 0.25 * s2
        qz = (yz + zy) / s2
    else:
        var s2 = _sqrt_f64(1.0 + zz - xx - yy) * 2.0  # s2 = 4*qz
        qw = (xy - yx) / s2
        qx = (xz + zx) / s2
        qy = (yz + zy) / s2
        qz = 0.25 * s2

    return (qx, qy, qz, qw)


def _fill_assets(

    asset_sec: String,
    mut result: FlatModelDef,
    defaults: DefaultsData,
    named_defaults: NamedDefaultsList,
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

        # Asset identity. `_find_texture_index_by_name` re-scans `asset_sec`
        # for these on every material, so they were being read and thrown
        # away; the render path needs them kept (skybox lookup by name, PNG
        # load by file). Mirrors `parse_xml_render_data`'s `tex_names` /
        # `tex_files`, which trims neither.
        td.name = _extract_attr(tag, "name")
        td.file = _extract_attr(tag, "file")

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
    #
    # ⚠⚠ THIS LOOP USED TO STOP AT 16 ASSETS, SILENTLY. `<mesh>` number 17
    # onwards never entered `mesh_asset_names`, so the name lookup below found
    # nothing and left the geom at `mesh_id = -1`. A mesh geom with no mesh does
    # not fail loudly: `fields_build` skips the hull load, `rbound` keeps its
    # per-type fallback (`gd.radius`, i.e. MuJoCo's default size 0.5 for a
    # mesh) and the geom carries NO COLLISION GEOMETRY while still being 16-18x
    # too big for every broadphase test it takes part in.
    #
    # Measured on SO-ARM100, which declares 18 mesh assets: geoms 26 and 27 —
    # `Moving_Jaw_Collision_2` and `_3`, the moving jaw's actual contact
    # surfaces — had `mesh_id -1` and `rbound 0.5` against MuJoCo's 0.0279 and
    # 0.0309. The visible symptom was performance, not a missing contact: the
    # bounding-sphere reject let 11 pairs per step into GJK where MuJoCo
    # narrow-phases 2, because those two spheres swallow the whole arm.
    #
    # ⚠ THE CAP WAS NOT `MAX_GPU_MESHES` AND MUST NOT BE CONFUSED WITH IT. That
    # limit is on LOADED (collidable) meshes and is enforced in `fields_build`;
    # this list is the XML's asset table, most of which is usually visual-only.
    # SO-ARM100 loads 8 collidable meshes out of 18 declared, so it was nowhere
    # near the real limit when this silently truncated it.
    var mesh_pos = 0
    var mesh_count = 0
    while True:
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
        # ⚠⚠ `name` IS OPTIONAL, AND ITS DEFAULT IS THE FILE STEM. MuJoCo:
        # "If omitted, the mesh name equals the file name without the path and
        # extension" (XMLreference.rst, asset-mesh-name, verified against
        # 3.10.0 — the RUNTIME version, not an older tree).
        #
        # Requiring `name=` here skipped the asset entirely, so every
        # `mesh="head_visual"` on a geom resolved to `mesh_id = -1` — and a
        # mesh geom with no mesh IS INVISIBLE and has no collision geometry,
        # while raising nothing. Measured on Menagerie's ToddlerBot, which
        # writes the bare `<mesh file="head_visual.stl"/>` form for all 47 of
        # its assets: 45 mesh geoms drew nothing and the robot rendered as its
        # sites alone. The nameless form is common across Menagerie, so this
        # made most of that library silently unloadable.
        #
        # ⚠ THE STEM, NOT THE FILENAME: path and extension both go. A model
        # mixing `<mesh file="a/x.stl"/>` with `<mesh name="x" .../>` must
        # produce ONE name for both, or the geom reference picks whichever
        # happens to be first.
        if mesh_name.byte_length() == 0:
            mesh_name = _file_stem(mesh_file)
        if mesh_name.byte_length() > 0 and mesh_file.byte_length() > 0:
            result.mesh_asset_names.append(mesh_name)
            result.mesh_asset_files.append(mesh_file)
            # ── `scale`, own tag first, then the class chain ──────────────
            # ⚠ AN ASSET RESOLVES ITS CLASS LIKE ANY OTHER ELEMENT: an
            # explicit `class=` names one, otherwise the TOP-LEVEL default
            # applies. Only 1 of op3's 49 `<mesh>` tags carries `scale` — the
            # one inside `<default>` — so the fallback is the whole feature,
            # not a nicety.
            var sc_s = _extract_attr(tag, "scale")
            if sc_s.byte_length() == 0:
                var mcls = _extract_attr(tag, "class")
                if mcls.byte_length() > 0:
                    sc_s = named_defaults.find(mcls).mesh_scale_s
                if sc_s.byte_length() == 0:
                    sc_s = defaults.mesh_scale_s
            var sx = 1.0
            var sy = 1.0
            var sz = 1.0
            if sc_s.byte_length() > 0:
                var sv = _parse_rgb3(sc_s)
                # ⚠ A ZERO COMPONENT IS REJECTED, NOT COPIED. MuJoCo requires
                # three reals; a short or malformed value parses to 0 here,
                # and a 0 scale collapses the hull to a plane or a point —
                # far worse, and far harder to see, than ignoring the attr.
                if sv[0] != 0.0 and sv[1] != 0.0 and sv[2] != 0.0:
                    sx = sv[0]
                    sy = sv[1]
                    sz = sv[2]
            result.mesh_asset_scale.append(sx)
            result.mesh_asset_scale.append(sy)
            result.mesh_asset_scale.append(sz)
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
    # geoms did — so a class-defined joint silently fell back to the
    # default axis.)
    # ⚠ AND WHEN NEITHER SIDE SUPPLIES ONE, the `axis` block below is
    # skipped entirely and `JointData`'s default stands. That default was
    # WRONG (Y, where MuJoCo uses Z) until 2026-08-13 — see the note on
    # `JointData.__init__` in flat_model.mojo.
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

    # `actuatorfrcrange` — start from the class chain, then let the element
    # override, the same 3-way order the actuator's `forcerange` uses.
    #
    # ⚠ NO DEGREE CONVERSION. `range` above converts for angular joints
    # because it is an ANGLE; this is a TORQUE and `deg_factor` would scale
    # g1's +-5 N.m to +-0.087. MuJoCo's `mjCJoint::Compile` converts `range`
    # and `springref` and leaves `actfrcrange` alone for exactly this reason.
    jd.actfrc_min = jdef.joint_actfrc_min
    jd.actfrc_max = jdef.joint_actfrc_max
    jd.is_actfrc_limited = jdef.joint_actfrc_limited
    _apply_actfrcrange(
        _extract_attr(tag, "actuatorfrcrange"),
        _extract_attr(tag, "actuatorfrclimited"),
        jd.is_actfrc_limited,
        jd.actfrc_min,
        jd.actfrc_max,
    )

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
        var sv = _solref_into(
            srl_s, jdef.joint_solref_limit_0, jdef.joint_solref_limit_1
        )
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
    # ⚠⚠ RESOLVED FOR EVERY TYPE, NOT ONLY `mesh`. A geom that names a mesh
    # while its type is a PRIMITIVE is not an error and not a mesh: MuJoCo
    # FITS the primitive to that mesh (`mjCGeom::Compile` -> `mjCMesh::
    # FitGeom`, user_objects.cc:4038) and then CLEARS the mesh reference, so
    # the compiled geom is a sphere/capsule/box sized from the mesh's inertia
    # box. Gating this block on `_GEOM_MESH` left those geoms with no mesh at
    # all, and `fit_from_mesh` below is what tells `fields_build` to size them.
    #
    # ⚠ IT IS NOT RARE, AND IT IS NOT COSMETIC. rainbow_robotics rby1's
    # `<default class="in-model-collision">` sets contype/conaffinity and NO
    # type, so all 49 of its collidable arm and finger geoms are fitted
    # SPHERES. Without the fit they fell back to the default radius of 0.5 —
    # 33x too large for a finger — and the model self-collided everywhere:
    # 128 contacts at `qpos0` where MuJoCo has 0, including a "penetration" of
    # 0.56 m between two grippers 0.44 m apart (0.5 + 0.5 - 0.44, exactly).
    var mesh_attr = _extract_attr(tag, "mesh")
    if mesh_attr.byte_length() == 0:
        mesh_attr = eff_defaults.geom_mesh_s
    if mesh_attr.byte_length() > 0:
        for mi in range(assets.num_mesh_assets):
            if assets.mesh_asset_names[mi] == mesh_attr:
                gd.mesh_id = mi
                gd.mesh_filename = assets.mesh_asset_files[mi]
                if mi * 3 + 2 < len(assets.mesh_asset_scale):
                    gd.mesh_scale_x = assets.mesh_asset_scale[mi * 3 + 0]
                    gd.mesh_scale_y = assets.mesh_asset_scale[mi * 3 + 1]
                    gd.mesh_scale_z = assets.mesh_asset_scale[mi * 3 + 2]
                gd.fit_from_mesh = gd.geom_type != _GEOM_MESH
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
            # ⚠⚠ s2 IS THE RENDER GRID SPACING, AND IT IS KEPT — in `half_z`,
            # which a plane otherwise never uses (`build_render_fields` zeroes
            # it for this type explicitly). It is irrelevant to collision, and
            # it used to be dropped on exactly that reasoning.
            #
            # What made that wrong is EXPORT: MuJoCo REFUSES a plane whose
            # third size is absent or zero ("plane size(3) must be positive"),
            # so a writer working from a record that lost it has to invent a
            # value — and walker2d's floor came back as 0.1 where the source
            # says 40. Caught by loading our own export in MuJoCo, which is
            # the half our parser reading our writer structurally cannot see.
            gd.half_z = s2
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
        var sv = _solref_into(
            sr_s, eff_defaults.geom_solref_0, eff_defaults.geom_solref_1
        )
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
        gd.has_explicit_mass = True
    elif gd.fit_from_mesh:
        # ⚠⚠ A FITTED PRIMITIVE HAS NO SIZE YET, SO IT CANNOT HAVE A MASS YET.
        # `mjCMesh::FitGeom` runs in the COMPILER, after this parser has read
        # the tag: a `<geom mesh="base_link" class="collision"/>` whose class
        # says `type="capsule"` carries no `size` at all, so the fields below
        # still hold `GeomData`'s placeholder 0.5. Computing `density * volume`
        # from those gives the volume of a HALF-METRE capsule —
        # `pi*0.25*(4*0.5/3 + 2*0.5) = 1.309 m^3` — and arx_l5's base_link
        # weighed **1308.997 kg** against MuJoCo's 0.128420, its inertia 0.6506
        # against 6.4e-05. The fitted dimensions land in `fields_build`, and
        # `-1` is the sentinel that says "weigh me once you have them".
        #
        # ⚠ THE 49 FITTED SPHERES ON rby1 ESCAPED ONLY BY LUCK: every one of
        # their bodies declares an explicit `<inertial>`, so the geom-derived
        # pass skips them entirely and the 523 kg each would have contributed
        # never landed. A body without one is what exposes this.
        gd.mass = Float64(-1)
    else:
        # ⚠⚠ `geom_volume`, NOT A SECOND COPY OF THE FIVE FORMULAS. This block
        # used to spell them out again, and the two agreed to within 1 ULP —
        # which is exactly how a duplicate survives: close enough that nothing
        # notices, and different enough that a byte-identity gate fails.
        # `test_edit_reaches_the_document` found it as a 9.362922095815296 vs
        # ...295 mismatch between the live model and a re-parse of the same
        # numbers. Same formula, different association order.
        var vol = Float64(geom_volume[DType.float64](
            gd.geom_type, gd.radius, gd.half_length,
            gd.half_x, gd.half_y, gd.half_z,
        ))
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
    if rgba_s.byte_length() == 0 and eff_defaults.geom_rgba_r >= Float64(0):
        # A class-supplied colour IS the geom's own colour for the purpose of
        # the material fallback below — same rule as the comptime twin's
        # `geom_has_rgba`, which is set AFTER the class chain is applied.
        gd.rgba_r = eff_defaults.geom_rgba_r
        gd.rgba_g = eff_defaults.geom_rgba_g
        gd.rgba_b = eff_defaults.geom_rgba_b
        gd.rgba_a = eff_defaults.geom_rgba_a
        gd.has_own_rgba = True
    elif rgba_s.byte_length() > 0:
        var cv = _parse_rgba4(rgba_s)
        gd.rgba_r = cv[0]
        gd.rgba_g = cv[1]
        gd.rgba_b = cv[2]
        gd.rgba_a = cv[3]
        gd.has_own_rgba = True

    # ── material NAME, resolved through the class chain ───────────────────
    #
    # ⚠⚠ THIS RESOLUTION USED TO LIVE ENTIRELY IN `_resolve_geom_materials`,
    # WHICH READS THE GEOM'S OWN TAG AND NOTHING ELSE. dm_control declares the
    # colour once, in a default block — `<default class="body"><geom
    # material="self"/></default>` — and every geom that inherits it came out
    # with `material_id = -1` and `GeomData`'s 0.7 grey fallback.
    #
    # Measured over quadruped/fish/ball_in_cup/humanoid/manipulator/walker:
    # 72 of 88 geoms. That is the SAME defect shape as the 2026-08-03 geom
    # `type` bug and 1a.1's actuator-class bug — a parser reading an attribute
    # off the element's own tag and never consulting `<default>`.
    #
    # ⚠ AND THE RED CHANNEL HID IT. `<material name="self" rgba=".7 .5 .3 1"/>`
    # has r = 0.7, which is EXACTLY `GeomData`'s fallback grey, so red agreed
    # on 71 of the 72 wrong geoms and only green and blue ever disagreed. A
    # gate sampling one colour channel would have read clean.
    #
    # The index still resolves in the post-pass (`asset_sec` is not reachable
    # here); only the NAME moves, because the name is the part that needs the
    # class chain.
    var gmat_s = _extract_attr(tag, "material")
    if gmat_s.byte_length() == 0:
        gmat_s = eff_defaults.geom_material_s
    gd.material_name = gmat_s

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

            # ⚠⚠ A SELF-CLOSING `<body .../>` HAS NO `</body>` TO POP IT, and
            # without this the walk stays one level deeper for the rest of the
            # document: every LATER SIBLING becomes a child of this body's
            # parent chain. `hello_robot_stretch_3` has one
            # (`link_grasp_center`), and the scene's floor, table and two
            # free-jointed objects all ended up inside `base_link` — MuJoCo
            # parents them to the world.
            #
            # ⚠ AND NBODY STAYS CORRECT, which is why every count-based gate
            # passed. The bodies are all there; only the TREE is wrong, and a
            # wrong tree is a different robot. Found by `studio.validate`
            # reporting a free joint on a nested body and a plane in a moving
            # body — two rules MuJoCo enforces, on a model MuJoCo loads.
            # Three Menagerie models use one: stretch_3, apptronik_apollo,
            # franka_fr3_v2.
            # ⚠ VIA `_is_self_closing_tag`, the same helper the two `<default>`
            # walkers use. Both of those already handled this — the body walk
            # was the one that did not, and spelling the test a third way here
            # is how they drift.
            if _is_self_closing_tag(worldbody, next_body_open):
                if depth > 0:
                    depth -= 1

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
                elif (
                    im_s.byte_length() > 0
                    and _trim(_extract_attr(tag, "fullinertia")).byte_length()
                    == 0
                ):
                    # ⚠⚠ AN `<inertial>` REPLACES THE WHOLE INERTIA, INCLUDING
                    # THE PART IT DOES NOT MENTION. `<inertial pos="0 0 0"
                    # mass="0"/>` is a legal and common way to spell a massless
                    # frame body, and MuJoCo compiles it to `body_inertia
                    # [0, 0, 0]` — measured on 3.10.0. We left `BodyData`'s
                    # constructor default of **0.01** standing on all three
                    # axes, because the branch above only writes when
                    # `diaginertia` is present and the geom-derived path
                    # `continue`s past any body with an explicit inertial.
                    #
                    # ⚠ IT IS NOT A SMALL NUMBER WHERE IT LANDS. rby1's
                    # v1.3 arms carry two of these (`EE_GR_TF_L/R`, the
                    # gripper transforms) plus `NECK_0`, and 0.01 kg m^2 at
                    # the wrist is comparable to the whole forearm: its mass
                    # matrix was off by 3.0e-02 against MuJoCo and the arm
                    # joints diverged 1.9e-03 in one step. The v1.2 model has
                    # only `NECK_0` — one body, on a head nobody drives — and
                    # sat two orders lower at 2.1e-05, which is why the same
                    # defect looked like "1.3 is a different robot".
                    #
                    # ⚠ MASS ALONE IS THE TRIGGER. A `<inertial>` giving
                    # `fullinertia` has its own diagonalisation below and must
                    # not be zeroed on the way past.
                    b.ixx = 0.0
                    b.iyy = 0.0
                    b.izz = 0.0

                # `fullinertia` is the 6-vector (ixx iyy izz ixy ixz iyz).
                # It is APPLIED BELOW, after the orientation block, because
                # MuJoCo's compiler diagonalises it into `iquat` and that
                # write has to be the last one to land.
                var ifi_s = _trim(_extract_attr(tag, "fullinertia"))

                var ip_s = _extract_attr(tag, "pos")
                if ip_s.byte_length() > 0:
                    var iv = _parse_vec3(ip_s)
                    b.ipos_x = iv[0]
                    b.ipos_y = iv[1]
                    b.ipos_z = iv[2]

                var iquat_s = _trim(_extract_attr(tag, "quat"))
                var iaa_s = _trim(_extract_attr(tag, "axisangle"))
                var ixy_s = _trim(_extract_attr(tag, "xyaxes"))
                var iza_s = _trim(_extract_attr(tag, "zaxis"))
                var ieu_s = _trim(_extract_attr(tag, "euler"))

                var iq = _orientation_to_quat(
                    iquat_s,
                    iaa_s,
                    ixy_s,
                    iza_s,
                    ieu_s,
                    deg_factor,
                    eulerseq,
                )
                b.iquat_x = iq[0]
                b.iquat_y = iq[1]
                b.iquat_z = iq[2]
                b.iquat_w = iq[3]

                # ── <inertial fullinertia="ixx iyy izz ixy ixz iyz"> ────────
                #
                # MuJoCo's compiler diagonalises the 6-vector into
                # `diaginertia` + `iquat` (`mjCBody::Compile` ->
                # `mjuu_fullInertia` -> `mjuu_eig3`), which is exactly the
                # pair `BodyData` already stores. So this is a parser-side
                # DECOMPOSITION into existing fields, not a schema change —
                # and `eig3_symmetric` is already a transcription of
                # `mjuu_eig3`, landed for the mesh-inertia work.
                #
                # ⚠ IT MUST BE THE EIGENSOLVER, NOT AN EIGENSOLVER. MuJoCo's
                # Jacobi forms the half-angle as `sqrt(0.5 - 0.5c)`, which
                # cancels catastrophically as it converges: measured against
                # numpy on the 3.10.0 runtime, its eigenVALUES are good to
                # 1e-13 but its eigenVECTORS carry ~1e-7 of deterministic
                # noise. Any independently-correct solver therefore DISAGREES
                # with `body_iquat` at 1e-7 while looking perfectly valid.
                # `eig3_symmetric` reproduces it to 2e-16 on all twelve probe
                # cases, degenerate ones included.
                #
                # ⚠ MEASURED, not assumed — `fullinertia` is MUTUALLY
                # EXCLUSIVE with `diaginertia` and with EVERY inertial
                # orientation spelling, including a redundant `quat="1 0 0 0"`.
                # MuJoCo raises rather than picking a winner. So do we: with
                # both present there is no way to tell which the author meant,
                # and silently letting one override the other is exactly how a
                # wrong inertia FRAME hides behind a right inertia MAGNITUDE.
                if ifi_s.byte_length() > 0:
                    if idi_s.byte_length() > 0:
                        raise Error(
                            "physics3d: <inertial>: fullinertia and diagonal"
                            " inertia cannot both be specified"
                        )
                    if (
                        iquat_s.byte_length() > 0
                        or iaa_s.byte_length() > 0
                        or ixy_s.byte_length() > 0
                        or iza_s.byte_length() > 0
                        or ieu_s.byte_length() > 0
                    ):
                        raise Error(
                            "physics3d: <inertial>: fullinertia and inertial"
                            " orientation cannot both be specified"
                        )

                    var fi_parts = List[String]()
                    _split_spaces(ifi_s, fi_parts)
                    if len(fi_parts) != 6:
                        raise Error(
                            "physics3d: <inertial fullinertia=...> needs"
                            " exactly 6 values (ixx iyy izz ixy ixz iyz)"
                        )
                    var fi = InlineArray[Float64, 6](fill=Float64(0))
                    for fk in range(6):
                        fi[fk] = _parse_float(fi_parts[fk])

                    var ev = eig3_symmetric[DType.float64](fi)
                    # `mjuu_fullInertia` rejects a non-PSD tensor on the
                    # SMALLEST eigenvalue, and eig3 sorts DECREASING, so that
                    # is ev[2]. Without this a non-physical tensor would reach
                    # `body_inv_inertia = 1/eig` and produce a negative or
                    # infinite inverse inertia — a garbage rollout with no
                    # error anywhere.
                    if ev[2] < 1e-14:  # mjEPS
                        raise Error(
                            "physics3d: <inertial fullinertia=...>: inertia"
                            " must have positive eigenvalues"
                        )

                    b.ixx = ev[0]
                    b.iyy = ev[1]
                    b.izz = ev[2]
                    b.iquat_x = ev[3]
                    b.iquat_y = ev[4]
                    b.iquat_z = ev[5]
                    b.iquat_w = ev[6]
                    b.has_explicit_inertia = True

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

            # Orientation. ⚠⚠ THIS HANDLED ONLY quat / axisangle / xyaxes AND
            # DROPPED `zaxis` AND `euler` ON THE FLOOR — a camera declaring
            # either kept the identity quaternion and looked straight DOWN its
            # own -Z, i.e. top-down, whatever the model asked for. acrobot and
            # cartpole use `zaxis="0 -1 0"`; walker's side camera uses
            # `euler="60 0 0"`. The comptime twin was fixed for exactly this
            # and the fix never crossed to the runtime parser, which is the
            # one that survives phase 1a.5.
            #
            # Sharing `_orientation_to_quat` — the helper the site and body
            # paths already use — rather than re-spelling the precedence a
            # fourth time, so the next attribute is added once.
            var quat_s = _extract_attr(tag, "quat")
            var aa_s = _extract_attr(tag, "axisangle")
            var xy_s = _extract_attr(tag, "xyaxes")
            var za_s = _extract_attr(tag, "zaxis")
            var eu_s = _extract_attr(tag, "euler")
            if (
                quat_s.byte_length() > 0
                or aa_s.byte_length() > 0
                or xy_s.byte_length() > 0
                or za_s.byte_length() > 0
                or eu_s.byte_length() > 0
            ):
                var cq = _orientation_to_quat(
                    quat_s, aa_s, xy_s, za_s, eu_s, deg_factor, eulerseq
                )
                cd.quat_x = cq[0]
                cd.quat_y = cq[1]
                cd.quat_z = cq[2]
                cd.quat_w = cq[3]

            var fovy_s = _extract_attr(tag, "fovy")
            if fovy_s.byte_length() > 0:
                cd.fovy = _parse_float(fovy_s)

            var ipd_s = _extract_attr(tag, "ipd")
            if ipd_s.byte_length() > 0:
                cd.ipd = _parse_float(ipd_s)

            var mode_s = _extract_attr(tag, "mode")
            if mode_s.byte_length() > 0:
                cd.mode = _cam_mode_from_str(mode_s)

            # `target="body"` — only meaningful for targetbody(com), and
            # resolved here so the per-frame re-aim is pure arithmetic.
            # ⚠ `_trim` MATCHES THE COMPTIME TWIN (`xml_parser.mojo:4939`).
            # `_find_body_index_by_name` compares the name literally, so a
            # stray space returns -1 and the camera silently degrades to
            # "no target" instead of failing.
            var tgt_s = _trim(_extract_attr(tag, "target"))
            if tgt_s.byte_length() > 0:
                cd.target_body = _find_body_index_by_name(worldbody, tgt_s)

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

    # ⚠ THE NAME TABLES ARE BUILT HERE, AFTER THE GROUPING, BY THE SAME RULE.
    # `names_in_element_order` IS the walk `_index_by_name_grouped` now looks
    # up — one implementation of "which element is index i" for both the
    # resolver and the table, so a name and the record it labels cannot drift
    # apart. Building them before the sort, or by counting tags, is exactly
    # the bug the sort exists to fix (see the note above it).
    result.body_names = body_names_in_order(worldbody)
    result.joint_names = names_in_element_order(worldbody, "<joint")
    result.geom_names = names_in_element_order(worldbody, "<geom")
    result.site_names = names_in_element_order(worldbody, "<site")


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
    tendon_sec: String,
    defaults: DefaultsData,
    named_defaults: NamedDefaultsList,
    mut result: FlatModelDef,
) raises:
    """Parse <actuator> section and populate result.actuators[].

    ⚠ THIS RESOLVED NO `<default class=...>` UNTIL 2026-08-14. It read element
    attributes and fell back to the ROOT defaults, while joints (`:1239`),
    geoms (`:1486`) and sites (`:2302`) had resolved classes all along — the
    actuator path simply never got it, the same shape as
    `feedback_a_normalizer_on_one_entry_point_only`. dog resolves 24 distinct
    classes over 38 actuators whose tags carry only `name`/`class`/`tendon`,
    so essentially every dog and quadruped actuator value came from the root.
    Gated by `tests/physics3d/test_actuator_record_equivalence.mojo` against
    the comptime `_acd`, which had the resolution and is the side the engine
    reads."""
    var act_count = 0
    var scan_pos = 0
    var alen = actuator_sec.byte_length()

    # ── the actuator elements we do NOT model, said out loud ─────────────
    #
    # ⚠⚠ AN UNMODELLED ACTUATOR TAG SHORTENS `nu` AND NOTHING SAID SO. The
    # scan below looks for four spellings; MJCF has ten. Anything else in this
    # section is skipped, so the model still loads, still steps, and quietly
    # consumes a control vector of the wrong length — every index past the
    # first missing actuator lands on the wrong actuator. Measured: flybody
    # reports nu 78 to MuJoCo and 70 here (eight `<adhesion>`), shadow_dexee
    # 12 and 0 (twelve `<plugin plugin="mujoco.pid">`). Those are the only two
    # in Menagerie, and both were found by diffing counts rather than by
    # anything the parser volunteered.
    #
    # ⚠ A PRINT, NOT A RAISE. A missing actuator is a lost capability, not a
    # corrupt model — the same call `native_multicontact`'s caps make — and
    # refusing to load flybody over eight adhesion pads would be worse than
    # loading it with seventy working servos. The count is what matters: a
    # caller comparing `nact` against its policy's action size sees the gap.
    var _unmodelled: List[String] = [
        String("<intvelocity"), String("<damper"), String("<cylinder"),
        String("<muscle"), String("<adhesion"), String("<plugin"),
    ]
    for _u in range(len(_unmodelled)):
        var _n = 0
        var _at = 0
        while True:
            var _hit = actuator_sec.find(_unmodelled[_u], _at)
            if _hit == -1:
                break
            _n += 1
            _at = _hit + 1
        if _n > 0:
            result.unmodelled_actuators += _n
            print(
                "physics3d: <actuator> declares", _n, "`"
                + _unmodelled[_u] + ">` element(s), which this parser does not"
                " model — they are SKIPPED, so `nact` is short by that many"
                " and a control vector sized for MuJoCo's `nu` will be"
                " misaligned from the first one onwards.",
            )

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
        var is_position = earliest == np_
        var is_velocity = earliest == nv_
        var is_general = earliest == ng

        # Effective defaults: the actuator's own class="..." wins, else the
        # top-level block. Same precedence geoms/joints/sites already use.
        # `NamedDefaultsList` folds the parent chain in at COLLECTION time
        # (`_collect_named_defaults`), so `find` returns fully-resolved values.
        var act_class = _extract_attr(tag, "class")
        var eff = defaults
        if act_class.byte_length() > 0:
            eff = named_defaults.find(act_class)

        # Record WHICH tag this was. The gains themselves come from the OTHER
        # parser: `xml_parser`'s comptime `ComptimeActData` carries
        # `motor_kp`/`motor_kv` (MuJoCo's `gainprm[0]` / `-biasprm[2]`) and
        # `apply_actions` reads them from there. This struct carries the kind
        # alone, which is what `init_fields` needs to refuse a transmission
        # neither path models. See docs/DM_CONTROL_PORT.md (gap G3).
        # ⚠ SEMANTIC, NOT SYNTACTIC. `kind` selects the FORCE LAW in
        # `apply_actions` — POSITION subtracts the transmission length,
        # VELOCITY does not, MOTOR skips the servo path — so it must describe
        # the compiled actuator, not the tag that was typed. This mirrors
        # `xml_parser.mojo:4119` exactly, including `<general>` starting as
        # POSITION and being corrected below once `biastype` has been read.
        #
        # ⚠⚠ `ACT_KIND_GENERAL` IS NEVER PRODUCED. The comptime twin never
        # emits it either; the engine's real domain is the three laws. Mapping
        # `<general>` to GENERAL (which this did until 2026-08-14) made
        # quadruped's twelve leg servos read as plain torque motors — a
        # different robot — while dog happened to survive because its
        # bias-free `<general>` is a motor anyway. That accidental agreement on
        # the LARGER model is why the tag encoding looked fine.
        if is_velocity:
            ad.kind = ACT_KIND_VELOCITY
        elif is_position or is_general:
            ad.kind = ACT_KIND_POSITION
        else:
            ad.kind = ACT_KIND_MOTOR

        # gear (element attribute wins, else the <default><motor> class)
        var gear_s = _extract_attr(tag, "gear")
        if gear_s.byte_length() > 0:
            ad.gear = _parse_float(gear_s)
        else:
            ad.gear = eff.motor_gear

        # joint name → joint index
        var jname = _extract_attr(tag, "joint")
        if jname.byte_length() > 0:
            ad.joint_id = _find_joint_index_by_name(worldbody, jname)
            # ⚠⚠ THE WORST SILENT FAILURE IN THIS FILE, UNTIL NOW. An
            # unresolved `joint=` left `joint_id = -1`, and
            # `_fill_actuator_transmission` is `if joint … elif tendon …` with
            # NO ELSE — so `trn_n` stayed 0 and the actuator applied ZERO
            # FORCE. Nothing downstream could recover it either: -1 is a LEGAL
            # sentinel there ("no joint transmission"), so the record cannot
            # distinguish a typo from a tendon-driven actuator.
            #
            # The visible symptom is a limp robot, which reads as a control or
            # gain problem, not as a name. `<contact><pair>` and
            # `<equality>` joints already raised; this is the same class and
            # was the last of the three.
            if ad.joint_id < 0:
                raise Error(
                    "physics3d: actuator references unknown joint='"
                    + _trim(jname) + "'. That would be a ZERO-FORCE actuator"
                    " — `-1` is a legal 'no joint transmission' sentinel, so"
                    " nothing after the parser could tell it from a tendon."
                )
        else:
            # `tendon=` transmission. Resolved off the SECTION TEXT (which
            # exists now) rather than `result.tendons` (which does not yet) —
            # `_tendon_index_by_name` numbers in XML order exactly as
            # `_fill_tendons` will.
            var tname = _trim(_extract_attr(tag, "tendon"))
            if tname.byte_length() > 0:
                ad.tendon_id = _tendon_index_by_name(tendon_sec, tname)
            else:
                # ⚠ A TRANSMISSION WE DO NOT MODEL IS NOT THE SAME AS NONE.
                # MuJoCo also drives through sites, bodies and slider-cranks;
                # this engine does not, and the resolved record is
                # indistinguishable from an actuator with no target at all —
                # which MuJoCo REFUSES. Recording which case it is lets
                # `studio.validate` say the true thing about each.
                for a in ["site", "body", "cranksite", "slidersite",
                          "refsite"]:
                    if _trim(_extract_attr(tag, String(a))).byte_length() > 0:
                        ad.unsupported_transmission = True

        # ctrlrange / ctrllimited — the "auto" rule lives in
        # `_apply_ctrlrange`, which the `<default>` block calls too.
        #
        # ⚠⚠ IT WAS INLINE HERE, AND ONLY HERE, WHICH IS THE WHOLE BUG. The
        # `"0 0"` half of the rule was fixed on this path (that fix is
        # `test_ctrllimited_vs_mujoco`'s `a5`: an explicit `ctrlrange="0 0"`
        # still reports ctrllimited 0, and clamping to [0, 0] delivered ZERO
        # FORCE where MuJoCo delivers the full command) and the `<default>`
        # path never had the rule at all, so a range stated in a CLASS was
        # never limited. That is where Menagerie states nearly all of them.
        # `_apply_forcerange` was written as a shared helper from the start
        # and never drifted; this is now its twin.
        #
        # Start from the class-resolved defaults, then let the element
        # override — the same 3-way order (element -> class chain -> root)
        # `_apply_forcerange` uses below. `cr_s` stays in scope because
        # `inheritrange` is skipped when an explicit ctrlrange is present.
        var cr_s = _extract_attr(tag, "ctrlrange")
        ad.ctrl_min = eff.motor_ctrl_min
        ad.ctrl_max = eff.motor_ctrl_max
        ad.is_ctrl_limited = eff.motor_ctrl_limited
        _apply_ctrlrange(
            cr_s,
            _extract_attr(tag, "ctrllimited"),
            ad.is_ctrl_limited,
            ad.ctrl_min,
            ad.ctrl_max,
        )

        # forcerange / forcelimited: start from the class-resolved defaults,
        # then let the element override. Same 3-way order `_acd` uses
        # (element -> class chain -> root).
        ad.force_limited = eff.motor_force_limited
        ad.force_min = eff.motor_force_min
        ad.force_max = eff.motor_force_max
        _apply_forcerange(
            _extract_attr(tag, "forcerange"),
            _extract_attr(tag, "forcelimited"),
            ad.force_limited,
            ad.force_min,
            ad.force_max,
        )

        # ── gains, and for `<general>` the force-law classification ───────
        if is_position:
            # kp default 1, kv default 0 (the damping term is optional here).
            # ⚠⚠ THE FALLBACK IS THE MERGED GAIN/BIAS, NOT "the last kp/kv
            # string a default tag happened to write". MuJoCo layers every
            # actuator tag in the block onto ONE record, so a `<velocity kv>`
            # or `<general gainprm>` earlier in the block supplies this
            # element's gain just as a `<position kp>` would — see
            # `DefaultsData.motor_gain`. Measured:
            #   <default><position kp="100" kv="9"/></default> + <position/>
            #     -> gainprm[0] 100, biasprm [0, -100, -9]
            #   <default><velocity kv="7"/></default> + <position/>
            #     -> gainprm[0]   7, biasprm [0,   -7, -7]
            # The second is the row a kp-string fallback gets wrong.
            var pkp = _extract_attr(tag, "kp")
            if pkp.byte_length() > 0:
                ad.kp = _parse_float(pkp)
            elif eff.motor_gain_set:
                ad.kp = eff.motor_gain
            else:
                ad.kp = 1.0
            var pkv = _extract_attr(tag, "kv")
            if pkv.byte_length() > 0:
                ad.kv = _parse_float(pkv)
            elif eff.motor_bias2_set:
                ad.kv = -eff.motor_bias2
            else:
                ad.kv = 0.0

            # ── `dampratio` — a kv the MODEL cannot state yet ─────────────
            # MuJoCo allows it on `<position>` and `<intvelocity>` only, and
            # it is EXCLUSIVE with kv ("kv and dampratio cannot both be
            # defined", `user_api.cc:1213`). MuJoCo raises; a parser that must
            # keep loading cannot, so an explicit kv wins — the same
            # precedence `inheritrange` takes against an explicit ctrlrange,
            # and the one a saved file has, since MuJoCo always writes the
            # DERIVED kv back out.
            #
            # ⚠ IT CANNOT BE RESOLVED HERE. The value depends on the
            # reflected inertia at qpos0, which does not exist until the mass
            # matrix is built — see `apply_actuator_dampratio`.
            var pdr = _extract_attr(tag, "dampratio")
            if pdr.byte_length() == 0:
                pdr = eff.motor_dampratio_s
            if pdr.byte_length() > 0 and pkv.byte_length() == 0:
                ad.dampratio = _parse_float(pdr)

            # ── `inheritrange` (`user_objects.cc:7138`) ───────────────────
            # "Automatically set the actuator's ctrlrange to match the
            # transmission target's range", scaled about its midpoint:
            #
            #     mean   = 0.5*(hi + lo)
            #     radius = 0.5*(hi - lo) * inheritrange
            #     ctrlrange = [mean - radius, mean + radius]
            #
            # ⚠⚠ WITHOUT THIS THE SERVO TARGETS A POSE THE JOINT FORBIDS.
            # MuJoCo clamps `ctrl` to `ctrlrange`, so on spot — whose knees
            # have range [-2.793, -0.254] while `qpos0` puts them at 0 — a
            # commanded 0 is clamped to -0.254 and the actuator pulls the knee
            # INTO its limit, -127 N.m at reset on each of the four. We had no
            # ctrlrange at all (the [-1, 1] default, unlimited), so the servo
            # held the knee at 0, a configuration the joint limit constraint
            # is simultaneously pushing out of.
            #
            # ⚠ MuJoCo's guard is `gaintype == FIXED && biastype == AFFINE &&
            # gainprm[0] == -biasprm[1]` — "the actuator's semantics are the
            # transmission's". `<position>` satisfies it by construction
            # (`biasprm[1]` IS `-kp`), which is why this sits in the position
            # branch rather than being tested separately.
            #
            # ⚠ EXCLUSIVE WITH `ctrlrange`. MuJoCo raises when both are given;
            # we cannot raise from a parser that must keep loading, so an
            # explicit `ctrlrange` WINS and this is skipped — the same
            # precedence a saved XML would have, since MuJoCo always converts
            # `inheritrange` to an explicit `ctrlrange` on save.
            var ir_s = _extract_attr(tag, "inheritrange")
            if ir_s.byte_length() == 0:
                ir_s = eff.motor_inheritrange_s
            var inherit = _parse_float(ir_s) if ir_s.byte_length() > 0 else 0.0
            if inherit > 0.0 and cr_s.byte_length() == 0:
                # ⚠ HINGE/SLIDE ONLY, and the range must be DEFINED. MuJoCo
                # raises on a free/ball target or a target with no range; here
                # a model that would not compile in MuJoCo simply keeps the
                # default rather than acquiring a nonsense clamp.
                if ad.joint_id >= 0 and ad.joint_id < len(result.joints):
                    ref jd = result.joints[ad.joint_id]
                    var lo = jd.range_min
                    var hi = jd.range_max
                    var real_range = (
                        lo != hi
                        and lo > Float64(-JOINT_RANGE_UNLIMITED)
                        and hi < Float64(JOINT_RANGE_UNLIMITED)
                    )
                    if real_range and (jd.jnt_type == JNT_HINGE
                                       or jd.jnt_type == JNT_SLIDE):
                        var mean = 0.5 * (hi + lo)
                        var radius = 0.5 * (hi - lo) * inherit
                        ad.ctrl_min = mean - radius
                        ad.ctrl_max = mean + radius
                        # MuJoCo's `ctrllimited` is auto and a defined range
                        # makes it limited — measured, spot reports
                        # `actuator_ctrllimited` true on all twelve.
                        ad.is_ctrl_limited = True
        elif is_velocity:
            # ⚠ kv DEFAULTS TO 1 here, not 0 — it IS the actuator, and 0 would
            # be a dead motor. gainprm[0] and -biasprm[2] are both K.
            # ⚠ A `<velocity>` ELEMENT SETS `biasprm[2] = -gainprm[0]`, so
            # with no `kv` of its own it takes the merged GAIN — not the
            # merged bias. Measured: a block of `<motor/><position kp="4000"
            # kv="400"/>` gives a bare `<velocity>` gainprm[0] 4000 and
            # biasprm [0, 0, -4000], i.e. kv 4000 and NOT 400. rby1's two
            # wheels are exactly this.
            var vkv = _extract_attr(tag, "kv")
            var vk: Float64
            if vkv.byte_length() > 0:
                vk = _parse_float(vkv)
            elif eff.motor_gain_set:
                vk = eff.motor_gain
            else:
                vk = 1.0
            ad.kp = vk
            ad.kv = vk
        elif is_general:
            var gt = _trim(_extract_attr(tag, "gaintype"))
            if gt.byte_length() == 0:
                gt = _trim(eff.motor_gaintype_s)
            var bt = _trim(_extract_attr(tag, "biastype"))
            if bt.byte_length() == 0:
                bt = _trim(eff.motor_biastype_s)
            var gp = _extract_attr(tag, "gainprm")
            if gp.byte_length() == 0:
                gp = eff.motor_gainprm_s
            var bp = _extract_attr(tag, "biasprm")
            if bp.byte_length() == 0:
                bp = eff.motor_biasprm_s

            var gain = _nth_float(gp, 0, 1.0)  # MuJoCo gainprm default 1
            var b0 = _nth_float(bp, 0, 0.0)
            var b1 = _nth_float(bp, 1, 0.0)
            var b2 = _nth_float(bp, 2, 0.0)
            var no_bias = bt.byte_length() == 0 or bt == "none"

            # Shapes we do not model. First offender wins, as in the twin.
            if gt.byte_length() > 0 and gt != "fixed":
                if result.bad_actuator < 0:
                    result.bad_actuator = act_count
                    result.bad_actuator_code = 0
            elif not (no_bias or bt == "affine"):
                if result.bad_actuator < 0:
                    result.bad_actuator = act_count
                    result.bad_actuator_code = 1
            elif (not no_bias) and b0 != 0.0:
                if result.bad_actuator < 0:
                    result.bad_actuator = act_count
                    result.bad_actuator_code = 2
            elif (not no_bias) and b1 != -gain and b1 != 0.0:
                if result.bad_actuator < 0:
                    result.bad_actuator = act_count
                    result.bad_actuator_code = 3

            ad.kp = gain
            ad.kv = 0.0 if no_bias else -b2

            # Correct the provisional POSITION now that biastype is known.
            #   no bias   -> gained torque motor
            #   b1 == 0   -> velocity servo  (b1 == -gain stays POSITION)
            if no_bias:
                ad.kind = ACT_KIND_MOTOR
            elif b1 == 0.0 and gain != 0.0:
                # `gain != 0` only keeps the branches disjoint: at gain 0 both
                # laws collapse to `force = -kv*vel` and POSITION keeps it.
                ad.kind = ACT_KIND_VELOCITY

        # ── `biasprm[2]`'s SIGN IS A DISCRIMINATOR ────────────────────────
        # MuJoCo carries `kv` and `dampratio` in the SAME slot and tells them
        # apart by sign (`user_api.cc:1211`: "negative: regular damping,
        # positive: dampratio"). `mj_setConst` then converts the positive one
        # once the mass matrix exists (`engine_setconst.c:998-1035`):
        #
        #     if gainprm[0] != -biasprm[1]: skip     # not position-like
        #     if biasprm[2] <= 0:           skip     # a literal kv
        #     biasprm[2] = -dampratio * 2 * sqrt(gainprm[0] * mass)
        #
        # ⚠⚠ THE RULE IS ON THE COMPILED ACTUATOR, NOT ON THE TAG. It lives
        # in the ENGINE, downstream of every spelling, so `<position
        # dampratio="0.9">` and `<general biasprm="0 -6.95 0.9">` reach it as
        # the same record. Reading `dampratio` off the `<position>` tag alone
        # — which is all this did until now — leaves the `<general>` spelling
        # with `kv = -0.9`: not a weaker damper but an ANTI-damper, and the
        # implicit integrators subtract it from the mass matrix. Measured on
        # sharpa_wave, whose left hand spells it `<position dampratio>` and
        # whose right hand spells it `<general biasprm>`: the left stepped to
        # 4.3e-18 and the right to 6.2e-03, one file apart.
        #
        # `ad.kv` IS `-biasprm[2]`, so "biasprm[2] > 0" reads as "kv < 0", and
        # `kind == ACT_KIND_POSITION` is exactly the `gainprm[0] ==
        # -biasprm[1]` gate — that equality is what makes this parser call an
        # actuator a position servo in the first place.
        if ad.kv < 0.0 and ad.dampratio == 0.0:
            if ad.kind == ACT_KIND_POSITION:
                ad.dampratio = -ad.kv
                ad.kv = 0.0

        # dyntype/dynprm -> dyn_tau + act_adr, and the running `na`.
        #
        # ⚠ `<general>` ONLY, mirroring `xml_parser.mojo:4292`'s
        # `elif is_general:`. `earliest == ng` is that test.
        if earliest == ng:
            var dyntype = _trim(_extract_attr(tag, "dyntype"))
            if dyntype.byte_length() == 0:
                dyntype = _trim(eff.motor_dyntype_s)
            var dynprm = _trim(_extract_attr(tag, "dynprm"))
            if dynprm.byte_length() == 0:
                dynprm = _trim(eff.motor_dynprm_s)

            if dyntype.byte_length() == 0 or dyntype == "none":
                ad.dyn_tau = 0.0
                ad.act_adr = -1
            elif dyntype == "filter":
                # dynprm[0], defaulting to 1.0 — MuJoCo's mjDYN_FILTER tau.
                var parts = List[String]()
                _split_spaces(dynprm, parts)
                ad.dyn_tau = _parse_float(parts[0]) if len(parts) > 0 else 1.0
                ad.act_adr = result.na
                result.na += 1
            # Any other dyntype is an unsupported transmission. The comptime
            # twin records `bad_actuator_code = 4` and `init_fields` raises on
            # it; this path leaves dyn_tau 0 / act_adr -1 and lets that
            # existing guard stay the single place that refuses the model.

        result.actuators.append(ad)
        # ⚠ CAPTURED HERE, NOT BY A SECOND WALK. Actuators are the one family
        # that is NOT regrouped afterwards, so "the order they are appended"
        # IS their index order and a name taken at the append cannot drift
        # from the record it belongs to. The worldbody families need
        # `names_in_element_order` precisely because they ARE regrouped.
        result.actuator_names.append(_trim(_extract_attr(tag, "name")))
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
        var sv = _solref_into(sr_s, ed.solref_0, ed.solref_1)
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
            # ⚠ 0 IS THE WORLDBODY AND IS A LEGAL TARGET, so "not found"
            # cannot be tested with `>= 0` here the way it is for a joint.
            # `_find_body_index_by_name` returns 0 both for the worldbody and
            # for a name it never saw — hence the explicit name check, which
            # is why this one was silent for so long.
            if has_b1:
                ed.body_a = _find_body_index_by_name(worldbody, b1_name)
                if ed.body_a == 0 and _trim(b1_name) != "world":
                    raise Error(
                        "physics3d: <equality> references unknown body1='"
                        + _trim(b1_name) + "'. It would silently weld to the"
                        " WORLDBODY instead."
                    )

            if has_b2:
                ed.body_b = _find_body_index_by_name(worldbody, b2_name)
                if ed.body_b == 0 and _trim(b2_name) != "world":
                    raise Error(
                        "physics3d: <equality> references unknown body2='"
                        + _trim(b2_name) + "'. It would silently weld to the"
                        " WORLDBODY instead."
                    )

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


def _fill_qpos0(xml: String, mut result: FlatModelDef) raises:
    """Initial pose. Mirrors `xml_parser.parse_xml_model_data` (`:4504`,
    `:4520`, `:4554`) in that order:

      1. each joint's `ref` at its qpos address (`ref_val` is ALREADY
         deg-converted by `_fill_model`),
      2. a free joint's enclosing body `pos` into adr..adr+2 — taken here from
         `JointData.body_id` rather than by re-scanning the worldbody text for
         the nearest preceding `<body`, which is what the twin does,
      3. `<custom><numeric name="init_qpos">` OVERRIDING both, and setting
         `qpos0_nq` to its own length.

    When nothing overrides, `qpos0_nq` is the total qpos width and a free
    joint gets qw = 1 at adr+3 so FK does not start on a zero quaternion.
    """
    var q = 0
    for i in range(len(result.joints)):
        q += result.joints[i].nq
    result.qpos0 = List[Float64](length=q if q > 0 else 1, fill=0.0)

    var adr = 0
    for i in range(len(result.joints)):
        var jd = result.joints[i]
        if jd.jnt_type == JNT_FREE:
            if result.free_joint_qpos_adr == -1:
                result.free_joint_qpos_adr = adr
            # ⚠ `body_id` IS WORLD-INCLUSIVE (`this_body_id = body_count + 1`,
            # worldbody = 0) while `result.bodies` EXCLUDES the worldbody, so
            # the list index is `body_id - 1`. Indexing directly put
            # quadruped's free joint on `hip_front_left` (pos .2 .2 0) instead
            # of `torso` (pos 0 0 .57) — a plausible-looking pose, one body
            # off, and nothing but the differential gate would have said so.
            var b = jd.body_id - 1
            if b >= 0 and b < len(result.bodies) and adr + 2 < q:
                result.qpos0[adr + 0] = result.bodies[b].pos_x
                result.qpos0[adr + 1] = result.bodies[b].pos_y
                result.qpos0[adr + 2] = result.bodies[b].pos_z
            # ⚠ AND THE ORIENTATION, HERE, FOR **EVERY** FREE JOINT. The
            # identity used to be written once after this loop, from
            # `free_joint_qpos_adr` — which records only the FIRST free joint,
            # so a scene with two floating bodies left the second one's
            # quaternion at (0,0,0,0).
            #
            # ⚠⚠ AND IT IS THE BODY'S QUAT, NOT THE IDENTITY. A free joint's
            # `qpos0` is its body's pose in the parent frame — MuJoCo takes
            # BOTH `body_pos` and `body_quat`, and this took the position and
            # then hardcoded `w = 1`. Measured on anybotics_anymal_b, whose
            # base is `<body pos="0 0 0.58" quat="0 0 0 1">`: MuJoCo's
            # `qpos0[:7]` is `[0, 0, 0.58, 0, 0, 0, 1]` — a 180-degree yaw —
            # and ours was `[0, 0, 0.58, 1, 0, 0, 0]`. The robot reset facing
            # the other way, and since anymal ships NO keyframe that is the
            # pose everything starts from.
            #
            # ⚠ THE RECORD IS (x, y, z, w) AND `qpos` IS (w, x, y, z). The two
            # orders differ here and nowhere else in this loop, which is
            # exactly the kind of place this tree has lost a quaternion
            # before.
            if adr + 6 < q:
                var _bqw = 1.0
                var _bqx = 0.0
                var _bqy = 0.0
                var _bqz = 0.0
                if b >= 0 and b < len(result.bodies):
                    _bqw = result.bodies[b].quat_w
                    _bqx = result.bodies[b].quat_x
                    _bqy = result.bodies[b].quat_y
                    _bqz = result.bodies[b].quat_z
                # `mju_normalize4`: a norm below mjMINVAL becomes the
                # IDENTITY, not a scaled zero.
                var _n2 = (
                    _bqw * _bqw + _bqx * _bqx + _bqy * _bqy + _bqz * _bqz
                )
                if _n2 < 1e-30:
                    _bqw = 1.0
                    _bqx = 0.0
                    _bqy = 0.0
                    _bqz = 0.0
                else:
                    var _inv = 1.0 / _sqrt_f64(_n2)
                    _bqw *= _inv
                    _bqx *= _inv
                    _bqy *= _inv
                    _bqz *= _inv
                result.qpos0[adr + 3] = _bqw
                result.qpos0[adr + 4] = _bqx
                result.qpos0[adr + 5] = _bqy
                result.qpos0[adr + 6] = _bqz
            elif adr + 3 < q:
                result.qpos0[adr + 3] = 1.0
        elif jd.jnt_type == JNT_BALL:
            # ⚠⚠ A BALL JOINT'S qpos0 IS THE IDENTITY QUATERNION, AND ZERO IS
            # NOT A ROTATION. This branch did not exist: `qpos0` is
            # zero-filled and only the free joint's `w` was ever set, so every
            # `<joint type="ball">` reset to (0,0,0,0). Forward kinematics
            # multiplies by it, so the body AND EVERYTHING BELOW IT collapses
            # to a zero quaternion — measured on cassie, whose two achilles
            # rods are ball-jointed: `xquat` came out exactly (0,0,0,0), the
            # two closed-loop anchors derived from it were 0.91 m off, and the
            # robot settled at 1.128 against MuJoCo's 1.013.
            #
            # ⚠ `ref` DOES NOT APPLY. MuJoCo's `ref` is a hinge/slide scalar;
            # a ball joint's reference is the identity, which is why this is
            # its own branch and not `ref_val` in the `else`.
            if adr < q:
                result.qpos0[adr + 0] = 1.0
        elif adr < q:
            result.qpos0[adr] = jd.ref_val
        adr += jd.nq

    # <custom><numeric name="init_qpos" data="..."/> overrides everything.
    var custom_sec = _extract_section(xml, "custom")
    var found = False
    if custom_sec.byte_length() > 0:
        var num_pos = 0
        while True:
            var t = custom_sec.find("<numeric", num_pos)
            if t == -1:
                break
            var tag_end = custom_sec.find(">", t)
            if tag_end == -1:
                break
            var tag = String(custom_sec[byte = t : tag_end + 1])
            if _trim(_extract_attr(tag, "name")) == "init_qpos":
                var parts = List[String]()
                _split_spaces(_extract_attr(tag, "data"), parts)
                var count = len(parts)
                if count > 64:  # the twin's cap; kept so the two agree
                    count = 64
                for i in range(count):
                    if i < len(result.qpos0):
                        result.qpos0[i] = _parse_float(parts[i])
                result.qpos0_nq = count
                found = True
                break
            num_pos = t + 7

    if not found and q > 0:
        result.qpos0_nq = q
        # ⚠⚠ A `qpos0[qw] = 1.0` STAMP USED TO SIT HERE, and its own comment
        # said it was "NOW REDUNDANT with the loop above ... re-writing an
        # identical value is cheaper than proving the two can never disagree."
        # It stopped being identical the moment that loop started writing the
        # BODY's quaternion instead of the identity: on anybotics_anymal_b,
        # whose base is `quat="0 0 0 1"`, the loop wrote (0, 0, 0, 1) and this
        # line stamped `w = 1` back over it, leaving (1, 0, 0, 1) — norm
        # sqrt(2), which FK then normalised to a 90-degree yaw where MuJoCo
        # has 180. The residual was exactly 1/sqrt(2) = 0.7071, which is what
        # a half-corrected quaternion looks like.
        #
        # ⚠ A REDUNDANT WRITE IS A SECOND WRITER. It is only redundant while
        # the other one agrees, and nothing was checking that.
        pass


def _fill_keyframes(xml: String, mut result: FlatModelDef) raises:
    """`<keyframe><key time= qpos= qvel= ctrl=>`, mirroring
    `xml_parser.parse_xml_model_data` (`:4612`).

    ⚠ `act` / `mpos` / `mquat` are REFUSED (code 2), not dropped. A silently
    ignored `act` would be a wrong actuator state with nothing to notice it.
    """
    var kf_sec = _extract_section_all(xml, "keyframe")
    if kf_sec.byte_length() == 0:
        return

    var nq = 0
    for i in range(len(result.joints)):
        nq += result.joints[i].nq
    var nact = len(result.actuators)
    var stride_q = nq if nq > 0 else 1
    var stride_c = nact if nact > 0 else 1

    var kpos = 0
    while True:
        # ⚠ `_find_tag`, NOT `find`. `"<key"` is a PREFIX of `"<keyframe"`, so
        # a raw find matches the section's own opening tag and invents a
        # phantom key — measured on so_arm100: nkey 3 against the twin's 2.
        # `_find_tag` checks the delimiter after the marker.
        var t = _find_tag(kf_sec, "<key", kpos)
        if t == -1:
            break
        var tag_end = kf_sec.find(">", t)
        if tag_end == -1:
            break
        var ktag = String(kf_sec[byte = t : tag_end + 1])
        kpos = tag_end + 1

        if (
            _trim(_extract_attr(ktag, "act")).byte_length() > 0
            or _trim(_extract_attr(ktag, "mpos")).byte_length() > 0
            or _trim(_extract_attr(ktag, "mquat")).byte_length() > 0
        ):
            result.bad_keyframe_code = 2
            break

        result.key_time.append(0.0)
        result.key_nqpos.append(0)
        result.key_nqvel.append(0)
        result.key_nctrl.append(0)
        for _ in range(stride_q):
            result.key_qpos.append(0.0)
            result.key_qvel.append(0.0)
        for _ in range(stride_c):
            result.key_ctrl.append(0.0)
        var k = result.nkey
        result.nkey += 1

        var ktime = _trim(_extract_attr(ktag, "time"))
        if ktime.byte_length() > 0:
            result.key_time[k] = _parse_float(ktime)

        var kq = _trim(_extract_attr(ktag, "qpos"))
        if kq.byte_length() > 0:
            var pq = List[String]()
            _split_spaces(kq, pq)
            var n = len(pq)
            if n > stride_q:
                n = stride_q
            for i in range(n):
                result.key_qpos[k * stride_q + i] = _parse_float(pq[i])
            result.key_nqpos[k] = len(pq)

            # ── `mj_normalizeQuat` on the keyframe (`user_model.cc:5353`) ──
            #
            # ⚠⚠ A KEYFRAME MAY CARRY A DEGENERATE QUATERNION AND MuJoCo
            # REPAIRS IT. `mju_normalize4` sets a vector whose norm is below
            # `mjMINVAL` to the IDENTITY (1,0,0,0), and otherwise normalizes;
            # `mj_normalizeQuat` applies it to every BALL and FREE joint's
            # quaternion slots, and `mjCModel::Compile` runs it over every
            # key. We took the file's numbers verbatim.
            #
            # ⚠ IT IS NOT A HYPOTHETICAL. `pal_tiago/tiago_position.xml`
            # literally writes `qpos="0 0 -0.985 0 0 0 0 ..."` — a ZERO
            # quaternion on its free joint. MuJoCo compiles that to
            # `key_qpos[3] = 1`; we reset the robot with a zero quat, and FK
            # multiplies by it, so the base and everything under it collapsed.
            # Both `scene_position` and `scene_velocity` diverged by exactly
            # 1.000e+00 at step ONE while the same robot's `scene_motor` —
            # which ships no keyframe — was at 6.7e-16.
            #
            # ⚠ THE NON-DEGENERATE HALF MATTERS TOO: a keyframe quat that is
            # merely UNNORMALISED (a hand-written 0.7 0.7 0 0) is scaled by
            # MuJoCo and would otherwise stretch every child transform.
            #
            # ⚠ `mjuu_normvec`, which runs EARLIER in the compiler over the
            # same array, RETURNS ON A ZERO VECTOR WITHOUT TOUCHING IT — it is
            # `mju_normalize4` via `mj_normalizeQuat` that does the repair.
            # Reading only the first of the two would give the wrong rule.
            var _kadr = 0
            for ji in range(len(result.joints)):
                ref _jd = result.joints[ji]
                var _qa = -1
                if _jd.jnt_type == JNT_FREE:
                    _qa = _kadr + 3
                elif _jd.jnt_type == JNT_BALL:
                    _qa = _kadr
                if _qa >= 0 and _qa + 3 < stride_q:
                    var _o = k * stride_q + _qa
                    var _w = result.key_qpos[_o + 0]
                    var _x = result.key_qpos[_o + 1]
                    var _y = result.key_qpos[_o + 2]
                    var _z = result.key_qpos[_o + 3]
                    var _n2 = _w * _w + _x * _x + _y * _y + _z * _z
                    if _n2 < 1e-30:
                        result.key_qpos[_o + 0] = 1.0
                        result.key_qpos[_o + 1] = 0.0
                        result.key_qpos[_o + 2] = 0.0
                        result.key_qpos[_o + 3] = 0.0
                    # ⚠ THE TEST IS ON THE NORM, NOT ITS SQUARE. MuJoCo's is
                    # `mju_abs(norm - 1) > mjMINVAL`; testing `|n2 - 1|`
                    # instead is a ~2x tighter threshold near 1 and would make
                    # this "almost" MuJoCo's rule for no reason.
                    elif abs(_sqrt_f64(_n2) - 1.0) > 1e-15:
                        var _inv = 1.0 / _sqrt_f64(_n2)
                        result.key_qpos[_o + 0] = _w * _inv
                        result.key_qpos[_o + 1] = _x * _inv
                        result.key_qpos[_o + 2] = _y * _inv
                        result.key_qpos[_o + 3] = _z * _inv
                _kadr += _jd.nq

        var kv = _trim(_extract_attr(ktag, "qvel"))
        if kv.byte_length() > 0:
            var pv = List[String]()
            _split_spaces(kv, pv)
            var n2 = len(pv)
            if n2 > stride_q:
                n2 = stride_q
            for i in range(n2):
                result.key_qvel[k * stride_q + i] = _parse_float(pv[i])
            result.key_nqvel[k] = len(pv)

        var kc = _trim(_extract_attr(ktag, "ctrl"))
        if kc.byte_length() > 0:
            var pc = List[String]()
            _split_spaces(kc, pc)
            var n3 = len(pc)
            if n3 > stride_c:
                n3 = stride_c
            for i in range(n3):
                result.key_ctrl[k * stride_c + i] = _parse_float(pc[i])
            result.key_nctrl[k] = len(pc)


def _fill_actuator_transmission(mut result: FlatModelDef):
    """Fill `dof_adr` / `trn_*` once joints AND tendons both exist.

    Mirrors `xml_parser.parse_xml_model_data` (`:4381`) exactly:

      * `joint=`  -> one `(qadr, dadr, 1.0)` triple, `trn_n = 1`
      * `tendon=` -> the named tendon's whole wrap list, and `dof_adr` from
                     its FIRST wrap

    ⚠ SEPARATE PASS ON PURPOSE. `_fill_actuators` runs before `_fill_tendons`,
    so the tendon branch cannot be done in place. Reordering those two calls
    in a ~3900-line parser is a bigger change than adding a pass, and
    `_fill_tendon_equalities` already establishes the shape.

    Joint qpos/dof addresses are a cumulative sum over `result.joints` in XML
    order — the same order `_acd`'s joint table walks — so no extra parse is
    needed to recover them.
    """
    var nj = len(result.joints)
    var qadr = List[Int](capacity=nj)
    var dadr = List[Int](capacity=nj)
    var q = 0
    var d = 0
    for i in range(nj):
        qadr.append(q)
        dadr.append(d)
        q += result.joints[i].nq
        d += result.joints[i].nv

    var na_ = len(result.actuators)
    result.motor_trn_qadr = List[Int](length=na_ * TENDON_MAX_WRAPS, fill=-1)
    result.motor_trn_dadr = List[Int](length=na_ * TENDON_MAX_WRAPS, fill=-1)
    result.motor_trn_coef = List[Float64](
        length=na_ * TENDON_MAX_WRAPS, fill=0.0
    )

    for ai in range(na_):
        var a = result.actuators[ai]
        var base = ai * TENDON_MAX_WRAPS
        if a.joint_id >= 0 and a.joint_id < nj:
            a.dof_adr = dadr[a.joint_id]
            result.motor_trn_qadr[base] = qadr[a.joint_id]
            result.motor_trn_dadr[base] = dadr[a.joint_id]
            result.motor_trn_coef[base] = 1.0
            a.trn_n = 1
        elif a.tendon_id >= 0 and a.tendon_id < len(result.tendons):
            var td = result.tendons[a.tendon_id]
            var n = td.num_joints
            if n > TENDON_MAX_WRAPS:
                n = TENDON_MAX_WRAPS
            for k in range(n):
                var jid = td.joint_ids[k]
                if jid >= 0 and jid < nj:
                    result.motor_trn_qadr[base + k] = qadr[jid]
                    result.motor_trn_dadr[base + k] = dadr[jid]
                result.motor_trn_coef[base + k] = td.coefs[k]
            a.trn_n = n
            if n > 0:
                a.dof_adr = result.motor_trn_dadr[base]
        # ⚠ WRITE BACK. `result.actuators[ai].field = ...` does not stick —
        # the subscript yields a copy. Same trap `_fill_tendon_equalities`
        # documents at `:3264`.
        result.actuators[ai] = a
        # ⚠⚠ AND COUNT THE ONES THAT GOT NOTHING. The two branches above are
        # `if joint … elif tendon …` with NO ELSE, so an actuator driven
        # through a `site=`, `body=`, `slidersite=` or `cranksite=`
        # transmission keeps `trn_n = 0`: it occupies a slot in `nact`, eats
        # its control, and applies ZERO FORCE. An unresolved `joint=` already
        # raises above; this is the rest of that family, and it is a whole
        # robot class rather than a rounding error — skydio_x2 and
        # bitcraze_crazyflie_2 drive all four rotors through
        # `<motor site="thrust1" gear="0 0 1 0 0 -.0201"/>`, so neither
        # aircraft has any thrust here. MuJoCo answers skydio's first step with
        # `qfrc_actuator = [0, 0, 0.378896, 0.01744, -0.053045, -0.001947]`;
        # we answered six zeros.
        if a.trn_n == 0:
            result.zero_transmission_actuators += 1
    if result.zero_transmission_actuators > 0:
        print(
            "physics3d:", result.zero_transmission_actuators, "of",
            na_, "actuators resolved to NO transmission this engine models"
            " (a `site=`, `body=`, `slidersite=` or `cranksite=` motor). They"
            " keep their slot in `nact` and their control, and apply ZERO"
            " FORCE — a model driven only through those does not move at all.",
        )


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
            # ⚠ THE MIRROR OF THE OTHERS: this one required BOTH
            # components and dropped a one-value `solref` on the floor.
            var sv2 = _solref_into(sr, td.solref_eq_0, td.solref_eq_1)
            td.solref_eq_0 = sv2[0]
            td.solref_eq_1 = sv2[1]
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

    `<spatial>` reads its `<site site=>` AND `<geom geom= sidesite=>` children
    in document order — that sequence is MuJoCo's `wrap_type`/`wrap_objid`/
    `wrap_prm`, and it is the routing itself, not a set.

    RAISES rather than skips on `<pulley>`, on a wrap geom that is neither a
    sphere nor a cylinder, and on a sequence that does not alternate
    site-geom-site. Dropping any of those would silently SHORTEN the tendon,
    and the failure would surface only as a physics divergence — which is
    exactly how the `<tendon>`-dropped-by-merge_mjcf bug stayed hidden until
    2026-07-30.
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
            # `width` / `rgba` are RENDER-ONLY and default to MuJoCo's
            # (0.003, .5 .5 .5 1) — see `TendonData`. Read off the opening
            # tag; a `<default><tendon>` class is not resolved here because
            # the comptime twin did not either, and no model in the tree puts
            # them in one.
            var w_s = _trim(_extract_attr(open_tag, "width"))
            if w_s.byte_length() > 0:
                td.render_width = _parse_float(w_s)
            var rgba_s = _trim(_extract_attr(open_tag, "rgba"))
            if rgba_s.byte_length() > 0:
                var rp = List[String]()
                _split_spaces(rgba_s, rp)
                if len(rp) >= 3:
                    td.rgba_r = _parse_float(rp[0])
                    td.rgba_g = _parse_float(rp[1])
                    td.rgba_b = _parse_float(rp[2])
                if len(rp) >= 4:
                    td.rgba_a = _parse_float(rp[3])
            if inner.find("<pulley") != -1:
                raise Error(
                    "physics3d: <spatial> tendon with a <pulley> is not"
                    " supported"
                )

            # ⚠⚠ THE WAYPOINTS ARE WALKED IN DOCUMENT ORDER, site and geom
            # together. This used to scan `<site>` only and RAISE on a
            # `<geom>`, which is why iit_softfoot and ms_human_700 would not
            # open. Scanning the two kinds in separate passes would be worse
            # than raising: the sequence IS the routing, and
            # `site,geom,site,geom,site` collapsed to `site,site,site` is a
            # tendon that runs straight through every pulley it is meant to
            # hook around — shorter, plausible, and silent.
            var wpos = 0
            while True:
                var sp = _find_tag(inner, "<site", wpos)
                var gp = _find_tag(inner, "<geom", wpos)
                var at = _min_valid(sp, gp)
                if at == -1:
                    break
                var is_site = at == sp
                var wtag = _extract_opening_tag(inner, at)
                wpos = inner.find(">", at) + 1

                var wname = _extract_attr(
                    wtag, "site" if is_site else "geom"
                )
                if wname.byte_length() == 0:
                    continue

                # ⚠ COUNT PAST THE CAP RATHER THAN STOPPING AT IT. Breaking
                # out of the loop is what made this truncate in silence.
                if td.num_wraps >= TENDON_MAX_SPATIAL_WRAPS:
                    td.wrap_overflow += 1
                    continue

                if is_site:
                    var sid = _find_site_index_by_name(worldbody, wname)
                    if sid < 0:
                        raise Error(
                            "physics3d: <spatial> tendon references unknown"
                            " site '" + wname + "'"
                        )
                    td.wrap_objs[td.num_wraps] = sid
                    td.wrap_types[td.num_wraps] = WRAP_SITE
                    td.wrap_sides[td.num_wraps] = -1
                else:
                    var gid = _find_geom_index_by_name(worldbody, wname)
                    if gid < 0:
                        raise Error(
                            "physics3d: <spatial> tendon wraps unknown geom"
                            " '" + wname + "'"
                        )
                    # ⚠ MuJoCo WRAPS SPHERES AND CYLINDERS ONLY (`mju_wrap`
                    # calls `mjERROR` on anything else). Naming the geom and
                    # its type matters: the alternative is treating it as a
                    # site, which routes the tendon THROUGH the object.
                    var gt = result.geoms[gid].geom_type
                    if gt == _GEOM_SPHERE:
                        td.wrap_types[td.num_wraps] = WRAP_SPHERE
                    elif gt == _GEOM_CYLINDER:
                        td.wrap_types[td.num_wraps] = WRAP_CYLINDER
                    else:
                        raise Error(
                            "physics3d: <spatial> tendon wraps geom '"
                            + wname + "', which is neither a sphere nor a"
                            " cylinder (MuJoCo supports no other wrap shape)"
                        )
                    td.wrap_objs[td.num_wraps] = gid
                    var side_n = _trim(_extract_attr(wtag, "sidesite"))
                    if side_n.byte_length() > 0:
                        var side_id = _find_site_index_by_name(
                            worldbody, side_n
                        )
                        if side_id < 0:
                            raise Error(
                                "physics3d: <spatial> tendon wrap names"
                                " unknown sidesite '" + side_n + "'"
                            )
                        td.wrap_sides[td.num_wraps] = side_id
                    else:
                        td.wrap_sides[td.num_wraps] = -1
                td.num_wraps += 1

            if td.num_wraps < 2:
                raise Error(
                    "physics3d: <spatial> tendon needs at least two"
                    " waypoints"
                )
            # ⚠ MuJoCo's `mj_tendon` CONSUMES site-geom-site AS ONE STEP
            # (`engine_core_smooth.c:1022`), reading the site two entries
            # along. A sequence that opens or closes on a geom, or puts two
            # geoms in a row, walks off the end of its own routing — so it is
            # refused here rather than half-evaluated there.
            if (
                td.wrap_types[0] != WRAP_SITE
                or td.wrap_types[td.num_wraps - 1] != WRAP_SITE
            ):
                raise Error(
                    "physics3d: <spatial> tendon must start and end with a"
                    " <site>"
                )
            for wi in range(td.num_wraps - 1):
                if (
                    td.wrap_types[wi] != WRAP_SITE
                    and td.wrap_types[wi + 1] != WRAP_SITE
                ):
                    raise Error(
                        "physics3d: <spatial> tendon has two wrap geoms in a"
                        " row; MuJoCo routes site-geom-site"
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
                    td.num_wraps + td.wrap_overflow if is_spatial
                    else td.num_joints + td.wrap_overflow
                )
                + " wraps, over the TENDON_MAX_WRAPS cap of "
                + String(TENDON_MAX_WRAPS)
                + " — raise it in `gpu/constants.mojo` (and"
                + " MAX_COMPTIME_TENDON_WRAPS in `xml_parser.mojo`, which must"
                + " agree) rather than letting the tendon run truncated"
            )

        # ── springs ──────────────────────────────────────────────────────
        # `stiffness` is a plain attribute. `springlength` is one value (both
        # bounds) or two (the band); ABSENT means both bounds are `length0`,
        # the rest length implied by the joint refs — not zero.
        var st_s = _extract_attr(open_tag, "stiffness")
        if st_s.byte_length() > 0:
            td.stiffness = _parse_float(st_s)

        var length0 = Float64(0)
        for k in range(td.num_joints):
            var jid = td.joint_ids[k]
            if jid >= 0 and jid < len(result.joints):
                length0 += td.coefs[k] * result.joints[jid].ref_val

        var sl_s = _extract_attr(open_tag, "springlength")
        if sl_s.byte_length() > 0:
            var sparts = List[String]()
            _split_spaces(sl_s, sparts)
            if len(sparts) >= 2:
                td.spring_lo = _parse_float(sparts[0])
                td.spring_hi = _parse_float(sparts[1])
            elif len(sparts) == 1:
                td.spring_lo = _parse_float(sparts[0])
                td.spring_hi = _parse_float(sparts[0])
        else:
            td.spring_lo = length0
            td.spring_hi = length0

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
) raises:
    """Parse <contact> section: fill result.excludes[] with body pair exclusions.

    ⚠ `raises` SINCE 2026-08-19: an exclude naming a body that does not exist
    used to be skipped silently, and the consequence is a pair that COLLIDES
    where MuJoCo excludes it. See the check below."""
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

        # ⚠⚠ AN UNRESOLVED EXCLUDE USED TO BE SKIPPED WITHOUT A WORD, and the
        # consequence is a pair that COLLIDES where MuJoCo excludes it —
        # `nexclude == 0` against MuJoCo's real count has already read as a
        # solver divergence once in this tree. Same worldbody-is-0 problem as
        # `<equality>` above: the name has to be checked, not the index.
        if _trim(body1_name) != "world" and b1 == 0:
            raise Error(
                "physics3d: <contact><exclude> references unknown body1='"
                + _trim(body1_name) + "'. The pair would COLLIDE where MuJoCo"
                " excludes it, with no diagnostic."
            )
        if _trim(body2_name) != "world" and b2 == 0:
            raise Error(
                "physics3d: <contact><exclude> references unknown body2='"
                + _trim(body2_name) + "'. The pair would COLLIDE where MuJoCo"
                " excludes it, with no diagnostic."
            )
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


def _is_default_geom_rgba(g: GeomData) -> Bool:
    """Is this geom's colour still MuJoCo's `"0.5 0.5 0.5 1"` default?

    MuJoCo decides whether a material's colour applies by comparing against
    this exact value rather than by remembering whether the user wrote one
    (XMLreference.rst:2623). Kept as one named predicate so the constant lives
    beside the rule that reads it.
    """
    return (
        g.rgba_r == 0.5 and g.rgba_g == 0.5
        and g.rgba_b == 0.5 and g.rgba_a == 1.0
    )


def _resolve_geom_materials(

    asset_sec: String,
    mut result: FlatModelDef,
):
    """Resolve material="name" on geoms → material index; copy material rgba.

    ⚠ THE NAME COMES OFF THE RECORD, NOT OFF THE TAG. This pass used to
    re-scan `worldbody` for `<geom` and read `material=` from the tag it
    found — which skips the `<default>`/`childclass` chain and silently
    de-coloured 72 of 88 dm_control geoms. `_parse_one_geom` resolves the name
    with the class chain in scope and leaves it in `material_name`; all that
    is left here is name → index, which needs `asset_sec` and so cannot
    happen during the walk.

    Re-scanning also made this pass's geom ordering an independent
    reimplementation of the DFS that produced `result.geoms` — two walks that
    had to agree for the indices to line up, with nothing checking they did.
    """
    for gi in range(len(result.geoms)):
        var mat_name = result.geoms[gi].material_name
        if mat_name.byte_length() == 0:
            continue
        var mid = _find_material_index_by_name(asset_sec, mat_name)
        result.geoms[gi].material_id = mid
        # ⚠ MuJoCo'S RULE IS A VALUE TEST, NOT A "WAS IT SPECIFIED" FLAG:
        # the material's colour applies unless the geom's own rgba DIFFERS
        # FROM ITS INTERNAL DEFAULT (XMLreference.rst:2623). That is a
        # different rule from `has_own_rgba`, and the difference is
        # observable — a geom writing `rgba="0.5 0.5 0.5 1"` explicitly takes
        # the MATERIAL colour in MuJoCo and would have kept its own here.
        # `has_own_rgba` stays on the record because `_parse_one_geom` still
        # needs it to distinguish a class-supplied colour from an absent one
        # when the class itself sets the default value.
        if (
            _is_default_geom_rgba(result.geoms[gi])
            and mid >= 0
            and mid < len(result.materials)
        ):
            var md = result.materials[mid]
            result.geoms[gi].rgba_r = md.rgba_r
            result.geoms[gi].rgba_g = md.rgba_g
            result.geoms[gi].rgba_b = md.rgba_b
            result.geoms[gi].rgba_a = md.rgba_a


# =============================================================================
# Main entry point
# =============================================================================



def _fill_visual(xml: String, mut result: FlatModelDef) raises:
    """`<visual><map>/<quality>/<headlight>` — RENDER ONLY (phase 1a.5).

    Mirrors `xml_parser.parse_xml_render_data`'s block exactly, including that
    every attribute is OPTIONAL and an absent one leaves MuJoCo's default in
    place rather than zeroing it.

    ⚠ `vis_has_headlight` is a PRESENCE FLAG, not a colour test.
    `<headlight ambient="0 0 0"/>` is a real declaration whose value equals the
    unset default, so the renderer cannot tell "declared black" from "not
    declared" by looking at the colour.
    """
    var visual_sec = _extract_section(xml, "visual")
    if visual_sec.byte_length() == 0:
        return

    var map_pos = visual_sec.find("<map")
    if map_pos != -1:
        var map_tag = _extract_opening_tag(visual_sec, map_pos)
        var znear_s = _trim(_extract_attr(map_tag, "znear"))
        if znear_s.byte_length() > 0:
            result.vis_znear = _parse_float(znear_s)
        var fs_s = _trim(_extract_attr(map_tag, "fogstart"))
        if fs_s.byte_length() > 0:
            result.vis_fogstart = _parse_float(fs_s)
        var fe_s = _trim(_extract_attr(map_tag, "fogend"))
        if fe_s.byte_length() > 0:
            result.vis_fogend = _parse_float(fe_s)

    var qual_pos = visual_sec.find("<quality")
    if qual_pos != -1:
        var qual_tag = _extract_opening_tag(visual_sec, qual_pos)
        var ss_s = _trim(_extract_attr(qual_tag, "shadowsize"))
        if ss_s.byte_length() > 0:
            result.vis_shadowsize = Int(_parse_float(ss_s))

    var hl_pos = visual_sec.find("<headlight")
    if hl_pos != -1:
        var hl_tag = _extract_opening_tag(visual_sec, hl_pos)
        var amb_s = _trim(_extract_attr(hl_tag, "ambient"))
        if amb_s.byte_length() > 0:
            var ap = List[String]()
            _split_spaces(amb_s, ap)
            if len(ap) >= 3:
                result.vis_headlight_ambient_r = _parse_float(ap[0])
                result.vis_headlight_ambient_g = _parse_float(ap[1])
                result.vis_headlight_ambient_b = _parse_float(ap[2])
                result.vis_has_headlight = True


def parse_xml_full(
    xml_in: String, base_dir: String = ""
) raises -> FlatModelDef:
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
    #
    # ⚠⚠ AND NORMALIZE `<freejoint>` FOR THE SAME REASON, ONE LAYER UP. This
    # parser has never known the tag: `_normalize_freejoint` lived only in
    # `merge_mjcf`, so a single-file MJCF reached here with `<freejoint/>`
    # intact and every `find("<joint")` below missed it. The body then got no
    # dofs and `body_weldid` stayed 0, which makes
    # `pair_body_filtered`'s `weld_i == weld_j` clause discard every contact
    # pair it belongs to — a body that can neither move nor collide, reported
    # as nothing at all. Measured: free sphere overlapping a static box,
    # MuJoCo 1 contact, ours 0.
    #
    # THE TWO PARSERS DISAGREEING IS THE HAZARD, not either one being wrong:
    # `parse_xml` now normalizes too, and a model where only one of them did
    # would raise the dimension check rather than mis-simulate.
    var xml = _strip_xml_comments(_normalize_freejoint(xml_in))

    # Extract top-level sections
    # ⚠⚠ `_extract_section_all`, NOT `_extract_section`. MJCF lets these
    # sections repeat and MuJoCo merges them; taking the first silently
    # discarded the rest. A model with two `<worldbody>` blocks loaded as
    # whatever was in the first one — nbody 1 for a five-prop scene — and a
    # shorter model raises nothing.
    var worldbody = _extract_section_all(xml, "worldbody")
    var actuator_sec = _extract_section_all(xml, "actuator")
    var asset_sec = _extract_section_all(xml, "asset")
    var equality_sec = _extract_section_all(xml, "equality")
    var contact_sec = _extract_section_all(xml, "contact")

    # Global physics options
    var opt = _parse_option(xml)
    result.gravity_x = opt[0]
    result.gravity_y = opt[1]
    result.gravity_z = opt[2]
    result.timestep = opt[3]
    result.opt_density = opt[4]
    result.opt_viscosity = opt[5]
    result.noslip_tolerance = opt[6]
    result.ccd_tolerance = opt[7]
    result.ccd_iterations = opt[8]
    result.impratio = opt[9]
    result.cone = opt[10]
    result.solver = opt[11]
    result.integrator = opt[12]

    # <flag multiccd="disable" nativeccd="disable"/> — `mjDSBL_MULTICCD` and
    # `mjDSBL_NATIVECCD`.
    #
    # ⚠⚠ THESE WERE THE FLAGS THIS FUNCTION DID NOT READ, and the omission was
    # not visible as one: `_option_flag_disabled` has existed all along and was
    # simply never called for them, so every model asking for single-point
    # convex contacts got `multi_ccd`'s 4-point manifold instead. All 11 baked
    # dm_control manipulation models set `nativeccd`, 9 of them set `multiccd`,
    # and on `reassemble5` that is 437 contacts against MuJoCo's 111 and 3701
    # ms per control step against 13-49 ms.
    #
    # ⚠ ONLY `multiccd` HAS A CONSUMER — see `FlatModelDef.nativeccd_disabled`.
    result.multiccd_disabled = _option_flag_disabled(xml, "multiccd")
    result.nativeccd_disabled = _option_flag_disabled(xml, "nativeccd")

    # <flag gravity="disable"/> — zero the gravity vector.
    if _option_flag_disabled(xml, "gravity"):
        result.gravity_x = Float64(0)
        result.gravity_y = Float64(0)
        result.gravity_z = Float64(0)

    # Defaults (applied when specific attrs are absent)
    var defaults_tuple = _parse_defaults(xml)
    var defaults = defaults_tuple[0]
    var named_defaults = defaults_tuple[1]

    # The ROOT default's motor ctrlrange — `ModelDefLike.CTRL_MIN/CTRL_MAX`.
    # See `FlatModelDef.default_motor_ctrl_min` for why it is copied verbatim
    # rather than corrected.
    result.default_motor_ctrl_min = defaults.motor_ctrl_min
    result.default_motor_ctrl_max = defaults.motor_ctrl_max

    # Compiler angle units (MuJoCo's MJCF default is degree) and euler order.
    var deg_factor = _compiler_deg_factor(xml)
    var eulerseq = String("xyz")
    # `<compiler meshdir>` / `assetdir` — see `_apply_meshdir` below.
    var meshdir = String("")
    var assetdir = String("")
    var texturedir = String("")
    var compiler_t = xml.find("<compiler")
    if compiler_t != -1:
        var compiler_end = xml.find(">", compiler_t)
        if compiler_end != -1:
            var ctag = String(xml[byte = compiler_t : compiler_end + 1])
            var seq_val = _last_compiler_attr(xml, "eulerseq")
            if seq_val.byte_length() == 3:
                eulerseq = seq_val
            # boundmass / boundinertia — see `FlatModelDef.boundmass`.
            var bm_s = _last_compiler_attr(xml, "boundmass")
            if bm_s.byte_length() > 0:
                result.boundmass = _parse_float(bm_s)
            var bi_s = _last_compiler_attr(xml, "boundinertia")
            if bi_s.byte_length() > 0:
                result.boundinertia = _parse_float(bi_s)

            # inertiafromgeom / inertiagrouprange / settotalmass — the three
            # `<compiler>` build modes. See `FlatModelDef.inertiafromgeom`;
            # each keeps MuJoCo's default when the attribute is absent, and
            # ⚠ that default is AUTO for inertiafromgeom, not off.
            var ifg_s = _last_compiler_attr(xml, "inertiafromgeom")
            if ifg_s == "true":
                result.inertiafromgeom = 1
            elif ifg_s == "false":
                result.inertiafromgeom = 0
            elif ifg_s == "auto":
                result.inertiafromgeom = 2

            var igr_s = _last_compiler_attr(xml, "inertiagrouprange")
            if igr_s.byte_length() > 0:
                var igr_parts = List[String]()
                _split_spaces(igr_s, igr_parts)
                if len(igr_parts) >= 2:
                    result.inertiagrouprange_min = _parse_int_str(igr_parts[0])
                    result.inertiagrouprange_max = _parse_int_str(igr_parts[1])

            # ⚠ ABSENT is -1.0, not 0.0 — `settotalmass="0"` is a legal (if
            # odd) request and must not read as "not specified".
            var stm_s = _last_compiler_attr(xml, "settotalmass")
            if stm_s.byte_length() > 0:
                result.settotalmass = _parse_float(stm_s)

            # ⚠ THE THREE DIRECTORIES STAY ON THE FIRST TAG, DELIBERATELY.
            # Every attribute above moved to `_last_compiler_attr` because
            # MuJoCo lets a later `<compiler>` override an earlier one — but
            # these three are PATHS, and `expand_mjcf` has already rebased
            # every `file=` it spliced in against the directory of the file
            # that WROTE it. Re-resolving them against a different
            # `<compiler>`'s directory here is the exact double-application
            # that `50d99683` was written to remove. If this ever needs the
            # last-wins rule too, it needs it in the expander, not here.
            meshdir = _trim(_extract_attr(ctag, "meshdir"))
            assetdir = _trim(_extract_attr(ctag, "assetdir"))
            texturedir = _trim(_extract_attr(ctag, "texturedir"))

    # Assets: textures and materials
    _fill_assets(asset_sec, result, defaults, named_defaults)

    # `<compiler meshdir>` — UNPARSED until 2026-08-13, and silently.
    #
    # ⚠⚠ WHAT IT USED TO DO. `mesh_asset_files` held the `file=` attribute
    # verbatim, so `load_mesh_hull` was handed the bare stem ("Base.stl"), the
    # open failed, and `fields_build` printed `Warning: failed to load mesh:`
    # and CARRIED ON. The model built and stepped with every mesh geom
    # non-colliding — `feedback_a_change_can_invalidate_its_own_justification`
    # verbatim, "an error PRINTED that nobody read".
    #
    # ⚠ AND IT IS WORSE THAN "COLLISION OFF". Measured on a one-mesh fixture:
    # the failed geom keeps `GEOM_IDX_MESH_ID` pointing at its ASSET index
    # while no hull was loaded, and its `rbound` is left at the 0.5 fallback
    # rather than 0 — so the broadphase still accepts pairs for it and the
    # narrow phase indexes a table that has nothing in it. A zero rbound would
    # at least have filtered it out.
    #
    # ⚠ 96 XML FILES ACROSS 69 MENAGERIE MODELS declare `meshdir`/`assetdir`,
    # so this was a precondition for loading almost any of them unmodified.
    #
    # PRECEDENCE, MEASURED ON THE 3.10.0 RUNTIME rather than transcribed:
    #     meshdir="m"              -> m/       (works)
    #     assetdir="m"             -> m/       (assetdir is the fallback)
    #     assetdir="." meshdir="m" -> m/       (meshdir WINS)
    #     absolute meshdir         -> used as-is
    #
    # ⚠ MuJoCo resolves the directory relative to the MODEL FILE. When this
    # parser is handed a bare STRING there is no such directory and the base
    # is the PROCESS CWD — which is what every repo-root-relative ported model
    # relies on. Pass `base_dir` to get MuJoCo's rule instead; see the block
    # below `effective_dir`.
    #
    # ⚠ `texturedir` IS HANDLED NOW, and the note it replaces was true when
    # written: "textures are renderer-only and no ported model loads one from
    # disk". The studio changed that — it opens arbitrary Menagerie models,
    # and umi_gripper writes
    # `<compiler meshdir="assets" texturedir="assets"/>` with its ArUco decals
    # in `assets/`. Without this the loader looked beside the .xml and printed
    # "No such file or directory" for each one. Same fallback chain as meshes:
    # the specific dir, else `assetdir`.
    var effective_dir = meshdir if meshdir.byte_length() > 0 else assetdir
    if effective_dir.byte_length() > 0:
        var base = effective_dir
        if not base.endswith("/"):
            base = base + "/"
        for i in range(len(result.mesh_asset_files)):
            var f = result.mesh_asset_files[i]
            # An absolute path ignores meshdir, as MuJoCo does.
            if f.byte_length() > 0 and not f.startswith("/"):
                result.mesh_asset_files[i] = base + f

    var tex_dir = texturedir if texturedir.byte_length() > 0 else assetdir
    if tex_dir.byte_length() > 0:
        var tbase = tex_dir
        if not tbase.endswith("/"):
            tbase = tbase + "/"
        for i in range(len(result.textures)):
            var tf = result.textures[i].file
            if tf.byte_length() > 0 and not tf.startswith("/"):
                result.textures[i].file = tbase + tf

    # ── `base_dir`: WHAT A RELATIVE ASSET PATH IS RELATIVE TO ─────────────
    #
    # MuJoCo resolves `meshdir` and a bare `file=` against THE DIRECTORY OF
    # THE MODEL FILE. This parser is handed a STRING, so it historically had
    # no such directory and fell back to the process CWD — which is why every
    # ported model carries repo-root-relative paths and why
    # `mujoco.MjModel.from_xml_path()` CANNOT load our own assets (it looks
    # beside the .xml and finds nothing). Measured, not assumed:
    # `from_xml_path('mojo_rl/envs/robots/assets/so_arm100.xml')` raises
    # "Error opening file 'mojo_rl/envs/robots/assets/so_arm100/Base.stl'".
    #
    # `base_dir` supplies that directory. Callers that read the model FROM A
    # FILE pass its dirname and get MuJoCo's semantics; callers that still
    # hand over a bare string pass "" and get exactly today's CWD behaviour,
    # which is why this is inert until a caller opts in.
    #
    # ⚠ ABSOLUTE PATHS ESCAPE, at both levels — a `meshdir="/opt/..."` or a
    # `file="/opt/..."` is used as-is, as MuJoCo does.
    #
    # ⚠ TEXTURES ARE PREFIXED HERE TOO, and they get no `meshdir`. MuJoCo
    # would consult `texturedir` then `assetdir`; neither is handled (no
    # ported model uses one for a texture), so a texture resolves against the
    # model directory alone. Adding `texturedir` belongs with whatever first
    # needs it.
    if base_dir.byte_length() > 0:
        var bd = base_dir
        if not bd.endswith("/"):
            bd = bd + "/"
        for i in range(len(result.mesh_asset_files)):
            var f = result.mesh_asset_files[i]
            if f.byte_length() > 0 and not f.startswith("/"):
                result.mesh_asset_files[i] = bd + f
        for i in range(len(result.textures)):
            var tf = result.textures[i].file
            if tf.byte_length() > 0 and not tf.startswith("/"):
                result.textures[i].file = bd + tf

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
    _fill_actuators(
        actuator_sec,
        worldbody,
        _extract_section_all(xml, "tendon"),
        defaults,
        named_defaults,
        result,
    )

    # Equality constraints
    _fill_equality(equality_sec, worldbody, result)
    # Tendons
    # ⚠ `_extract_section_all`: MJCF lets `<tendon>` appear more than once and
    # MuJoCo merges the repeats. Every sibling call on line ~4495 was fixed for
    # this; this one was missed because tendons arrived through `merge_mjcf`,
    # which had already collapsed them.
    _fill_tendons(_extract_section_all(xml, "tendon"), worldbody, result)

    # Actuator transmission: needs BOTH joints and tendons, so it runs here
    # rather than inside `_fill_actuators`.
    _fill_actuator_transmission(result)

    # Initial pose and keyframes: both need joints (and keyframes need the
    # actuator count for the ctrl stride), so they run here.
    _fill_qpos0(xml, result)
    _fill_keyframes(xml, result)
    _fill_visual(xml, result)
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

    # ⚠⚠ AFTER `_fill_pairs`, NOT BEFORE. This block used to sit up beside the
    # geom walk, where `result.pairs` is still EMPTY — so adding the pair scan
    # there would have compiled, run, and changed nothing at all.
    # ⚠ THE CONDIM THE MODEL NEEDS, so a caller can compare it with the
    # `MAX_CONDIM` it built. `contact_solve` clamps a contact whose condim
    # exceeds the built bound SILENTLY, in both cone branches, so spot's
    # `condim="6"` feet were solved as condim 3 — torsional and rolling
    # friction dropped with no indication.
    #
    # ⚠⚠ `<contact><pair>` IS A SECOND SOURCE, AND IT WAS MISSING. A pair's
    # `condim` does not come from its geoms and is not bounded by them — it
    # REPLACES what the mask-based path would have computed. apptronik_apollo
    # is the case: every geom it owns is `condim="1"` (its root `<default>`
    # says so) and the soles reach condim 6 only through
    # `<pair condim="6" .../>`. So this returned the floor of 3 while MuJoCo
    # reported `contact.dim == 6` on all four foot contacts — and the caller
    # comparing "what I built" against "what the model needs" was told the
    # truth about a model it was not looking at.
    #
    # ⚠ THE COMPTIME TWIN NEVER HAD THIS GAP, for a reason worth keeping:
    # `_scan_max_condim` scans the WHOLE FILE for any `condim=` and does not
    # try to work out which element it belongs to. Deliberately coarse beats
    # precisely wrong here — over-estimating costs a few unused rows, and
    # under-estimating is silent.
    var mcd = 3
    for gi in range(len(result.geoms)):
        var gc = result.geoms[gi].condim
        if gc > mcd:
            mcd = gc
    for pi in range(len(result.pairs)):
        var pc = result.pairs[pi].condim
        if pc > mcd:
            mcd = pc
    result.max_condim = mcd
    # Post-pass: resolve geom material="name" references
    _resolve_geom_materials(asset_sec, result)

    return result^
