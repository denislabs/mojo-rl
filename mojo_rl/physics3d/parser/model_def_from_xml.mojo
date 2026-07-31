"""ModelDefFromXML — ModelDefLike implementation from an embedded MJCF XML string.

Enables zero-boilerplate physics environments from XML:

    comptime pm = parse_xml(my_xml)
    comptime XmlModel = ModelDefFromXML[
        my_xml,
        pm.NBODY, pm.NJOINT, pm.NQ, pm.NV, pm.NGEOM, pm.NACT,
        max_contacts=50,
    ]
    var env = Phyics3dEnv[XmlModel, MyConfig]()

CPU path:  parse_xml_full() → FlatModelDef.setup_model() + FK + invweight0.
GPU path:  CPU Model → HostBuffer → DeviceBuffer → _compute_invweight0_gpu().
GPU kernels: comptimefor loops over comptime scalar helpers.
Rendering: no-op stubs (XML models have no visual configuration yet).

Note: Mojo nightly requires struct parameters to be accessed as 'Self.param'
inside the struct body. All dimension parameters follow this convention.
"""

from std.collections import InlineArray

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from std.random.philox import Random as PhiloxRandom
from mojo_rl.render import Color, Renderer3D, Light, Camera3D
from mojo_rl.math3d import Vec3 as _Vec3G, Quat as _QuatG

from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.joint_types import (
    JNT_FREE,
    JNT_BALL,
    JNT_HINGE,
    JNT_SLIDE,
)
from mojo_rl.physics3d.gpu.constants import (
    TPB,
)
from mojo_rl.physics3d.fields import Model, Data, DynamicsScratch
from mojo_rl.physics3d.dynamics.invweight import (
    compute_invweight0,
)
from mojo_rl.physics3d.model.model_def import ModelDefLike
from .fields_build import build_model_fields_from_flat
from .flat_model import (
    ACT_KIND_MOTOR,
    ACT_KIND_POSITION,
    ACT_KIND_GENERAL,
    act_kind_name,
    _GEOM_ELLIPSOID,
)
from .full_parser import parse_xml_full
from .xml_parser import (
    MAX_COMPTIME_TENDONS,
    _xml_nth_motor_gear,
    _xml_nth_motor_dof_adr,
    _xml_nth_joint_qpos_adr,
    _xml_nth_joint_limited,
    _xml_nth_joint_range_min,
    _xml_nth_joint_range_max,
    _xml_compiler_inertiafromgeom,
    _xml_compiler_settotalmass,
    _xml_compiler_inertiagrouprange,
    ComptimeActData,
    parse_xml_model_data,
    ComptimeRenderData,
    parse_xml_render_data,
    _xml_default_motor_ctrlrange,
    _xml_fixed_tendon_njoints,
    _xml_fixed_tendon_joint_name,
    _xml_fixed_tendon_coef,
    _xml_find_joint_dof_adr,
    _xml_find_joint_index,
)

# Type aliases matching model_def.mojo module scope (required for trait conformance)
comptime _RVec3 = _Vec3G[DType.float64]
comptime _RQuat = _QuatG[DType.float64]


@fieldwise_init
struct ModelDefFromXML[
    xml: String,
    nbody: Int,
    njoint: Int,
    nq: Int,
    nv: Int,
    ngeom: Int,
    nact: Int,
    ntex: Int = 0,
    nmat: Int = 0,
    nlight: Int = 0,
    ncam: Int = 0,
    max_contacts: Int = 50,
    max_equality: Int = 0,
    cone_type: Int = ConeType.PYRAMIDAL,
    max_tendon: Int = 0,
    nsite: Int = 0,
    neq: Int = 0,
    nexclude: Int = 0,
    obs_qpos_skip: Int = 1,
    obs_dim_override: Int = -1,
    action_dim_override: Int = -1,
    timestep: Float64 = 0.01,
    allow_unsupported_actuators: Bool = False,
](ModelDefLike):
    """ModelDefLike implementation driven entirely from an embedded MJCF XML string.

    All physics dimensions must be provided; obtain them from `parse_xml()`:

        comptime pm = parse_xml(xml)
        comptime XmlModel = ModelDefFromXML[
            xml,
            pm.NBODY, pm.NJOINT, pm.NQ, pm.NV, pm.NGEOM, pm.NACT,
            pm.NTEX, pm.NMAT, pm.NLIGHT, pm.NCAM,
        ]

    Parameters:
        xml:           Embedded MJCF XML string (must be comptime-known).
        nbody:         Total body count including worldbody.
        njoint:        Total joint count.
        nq:            Total position DOF.
        nv:            Total velocity DOF.
        ngeom:         Total geometry count.
        nact:          Total actuator count.
        ntex:          Texture count from <asset> (default 0).
        nmat:          Material count from <asset> (default 0).
        nlight:        Light count in <worldbody> (default 0).
        ncam:          Camera count in <worldbody> (default 0).
        max_contacts:  Maximum contacts per step (default 50).
        max_equality:  Maximum equality constraints (default 0).
        cone_type:     Friction cone type (default ELLIPTIC).
        max_tendon:    Maximum fixed tendons (default 0).
        nsite:   Total site count (default 0).
        neq:           Number of equality constraints (default 0).
        nexclude:      Number of contact exclusion pairs (default 0).
        obs_qpos_skip: Leading qpos DOF to exclude from obs (default 1).
        obs_dim_override: Override OBS_DIM (default -1 = compute from nq-skip+nv).
            Use when custom_extract_obs_gpu produces different dimensionality than
            the default formula (e.g. InvertedDoublePendulum needs OBS_DIM=9 with
            sin/cos transforms despite nq-skip+nv=6).
        action_dim_override: Override ACTION_DIM (default -1 = use nact).
        timestep:      Simulation timestep (default 0.01).
        allow_unsupported_actuators: Build the model even when the XML declares
            <position>/<velocity>/<general> actuators. Those are servos the
            engine cannot simulate (no gainprm/biasprm — see gap G3 in
            docs/DM_CONTROL_PORT.md), so `init_fields` rejects them by default.
            Set True ONLY when the env's CONFIG bypasses `apply_actions`
            entirely (returns True from `custom_apply_actions_cpu`) and drives
            those DOFs itself — SawyerReach does exactly this for its two
            kp=400 gripper servos.
    """

    # === Dimensions required by ModelDefLike ===
    comptime NBODY: Int = Self.nbody
    comptime NJOINT: Int = Self.njoint
    comptime NQ: Int = Self.nq
    comptime NV: Int = Self.nv
    comptime NGEOM: Int = Self.ngeom
    comptime MAX_EQUALITY: Int = Self.max_equality
    comptime CONE_TYPE: Int = Self.cone_type
    comptime MAX_CONTACTS: Int = Self.max_contacts
    comptime MAX_TENDON: Int = Self.max_tendon
    comptime NSITE: Int = Self.nsite
    comptime NEXCLUDE: Int = Self.nexclude
    comptime OBS_DIM: Int = Self.obs_dim_override if Self.obs_dim_override > 0 else (
        Self.nq - Self.obs_qpos_skip + Self.nv
    )
    comptime ACTION_DIM: Int = Self.action_dim_override if Self.action_dim_override > 0 else Self.nact
    comptime TIMESTEP: Float64 = Self.timestep
    comptime _ctrlrange: Tuple[Float64, Float64] = _xml_default_motor_ctrlrange[Self.xml]()
    comptime CTRL_MIN: Float64 = Self._ctrlrange[0]
    comptime CTRL_MAX: Float64 = Self._ctrlrange[1]

    # Precomputed XML actuator/joint data — evaluated at struct level by the
    # regular Mojo interpreter (not the GPU kernel compiler), so String ops work.
    # GPU kernels access Self._acd.motor_gears[i] etc. with no String operations.
    comptime _acd: ComptimeActData = parse_xml_model_data(Self.xml)

    # Precomputed rendering data — evaluated once at struct level.
    # Replaces 11 separate parse_xml_full calls that crashed the comptime
    # interpreter for large (25+ body) models.
    comptime _rcd: ComptimeRenderData = parse_xml_render_data(Self.xml)

    # =========================================================================
    # CPU: state hooks (fields-native; G2). The legacy CPU model build
    # (setup_model_and_data + _reset_data_legacy) was deleted at G4 — the
    # model build is `init_fields` (spec-direct) below.
    # =========================================================================

    @staticmethod
    def reset_data[
        DTYPE: DType
    ](
        mut d: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.MAX_CONTACTS,
            Self.NSITE,
            1,
        ],
    ):
        """Reset qpos to initial pose, zero qvel/qacc/qfrc.

        If the XML has a <custom><numeric name="init_qpos"/> section, those
        values are applied directly.  Otherwise qpos is zeroed and the free
        joint quaternion (if any) is set to identity (qw=1) so that FK does
        not degenerate.
        """
        comptime if Self._acd.nq > 0:
            # Apply init_qpos from XML custom section.
            comptime for i in range(Self.NQ):
                comptime if i < Self._acd.nq:
                    comptime val = Self._acd.qpos0[i]
                    d.qpos.data[i] = Scalar[DTYPE](val)
                else:
                    d.qpos.data[i] = Scalar[DTYPE](0)
        else:
            # No init_qpos — zero everything, then fix free-joint quaternion.
            for i in range(Self.NQ):
                d.qpos.data[i] = Scalar[DTYPE](0)
            comptime if Self._acd.free_joint_qpos_adr >= 0:
                # qpos[adr+3] is qw for a free joint (MuJoCo convention:
                # [tx, ty, tz, qw, qx, qy, qz]).  Set qw=1 for identity.
                d.qpos.data[Self._acd.free_joint_qpos_adr + 3] = Scalar[
                    DTYPE
                ](1)
        for i in range(Self.NV):
            d.qvel.data[i] = Scalar[DTYPE](0)
            d.qacc.data[i] = Scalar[DTYPE](0)
            d.qfrc.data[i] = Scalar[DTYPE](0)

    @staticmethod
    def extract_obs[
        DTYPE: DType
    ](
        d: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.MAX_CONTACTS,
            Self.NSITE,
            1,
        ],
        mut obs: List[Scalar[DTYPE]],
    ):
        """Extract observation: qpos[obs_qpos_skip:] followed by qvel[:]."""
        for i in range(Self.NQ - Self.obs_qpos_skip):
            obs.append(d.qpos.data[Self.obs_qpos_skip + i])
        for i in range(Self.NV):
            obs.append(d.qvel.data[i])

    @staticmethod
    def enforce_limits[
        DTYPE: DType
    ](
        mut d: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.MAX_CONTACTS,
            Self.NSITE,
            1,
        ],
    ):
        """Clamp qpos to joint range limits (limited joints only)."""
        for j in range(Self.NJOINT):
            if Self._acd.joint_is_limited[j]:
                var qp_adr = Self._acd.joint_qpos_adr[j]
                var v = d.qpos.data[qp_adr]
                if v < Scalar[DTYPE](Self._acd.joint_range_min[j]):
                    d.qpos.data[qp_adr] = Scalar[DTYPE](
                        Self._acd.joint_range_min[j]
                    )
                elif v > Scalar[DTYPE](Self._acd.joint_range_max[j]):
                    d.qpos.data[qp_adr] = Scalar[DTYPE](
                        Self._acd.joint_range_max[j]
                    )

    @staticmethod
    def apply_actions[
        DTYPE: DType
    ](
        mut d: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.MAX_CONTACTS,
            Self.NSITE,
            1,
        ],
        actions: List[Float64],
    ):
        """Generalized forces from the model spec: actuators + tendon springs.

        MuJoCo recomputes `qfrc_actuator` inside every `mj_step`, and for a
        `<motor>` that is redundant — its force is `gear * ctrl`, constant
        across a control step. A `<position>` servo is not: its force reads
        `qpos`, which moves every substep. So `Phyics3dEnv.step` calls this
        ONCE PER SUBSTEP rather than once per control step. For a motor-only
        model that is bit-identical (the same constant is written each time);
        for a servo it is the difference between a spring and a constant push.

        Both actuator kinds go through the same
        `force -> moment^T force` shape, over the transmission triples the
        comptime parser resolved (`motor_trn_*`):

            MOTOR     force = ctrl
            POSITION  force = kp*(ctrl - length) - kv*velocity
            length    = gear * sum_k coef_k qpos[qadr_k]
            velocity  = gear * sum_k coef_k qvel[dadr_k]
            qfrc[dadr_k] += gear * coef_k * force

        A joint transmission is one triple with coef 1, so the motor path
        reduces to the previous `qfrc[dof] = gear * ctrl` exactly.

        Accumulates rather than assigns, because a tendon transmission and a
        tendon spring can land on the same DOF (fish's `fins_flap` actuator
        and `fins_sym` spring share both fin roll joints) — hence the zeroing
        pass first. `d.qfrc` has exactly two other writers: `reset_data`,
        which zeroes it, and a CONFIG's `custom_apply_actions_cpu`, which
        returns True and suppresses this method entirely.
        """
        for i in range(Self.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)

        for i in range(Self.nact):
            if i >= len(actions):
                break
            var n = Self._acd.motor_trn_n[i]
            if n == 0:
                continue
            # Clamp to per-actuator ctrlrange (per-element overrides default).
            var ctrl = actions[i]
            if ctrl > Self._acd.motor_ctrl_max[i]:
                ctrl = Self._acd.motor_ctrl_max[i]
            elif ctrl < Self._acd.motor_ctrl_min[i]:
                ctrl = Self._acd.motor_ctrl_min[i]

            var gear = Self._acd.motor_gears[i]
            var force = ctrl
            if Self._acd.motor_kind[i] == ACT_KIND_POSITION:
                var length = Float64(0)
                var vel = Float64(0)
                for k in range(n):
                    var qadr = Self._acd.motor_trn_qadr[i * 4 + k]
                    var dadr = Self._acd.motor_trn_dadr[i * 4 + k]
                    var coef = Self._acd.motor_trn_coef[i * 4 + k]
                    if qadr >= 0 and qadr < Self.NQ:
                        length += coef * Float64(d.qpos.data[qadr])
                    if dadr >= 0 and dadr < Self.NV:
                        vel += coef * Float64(d.qvel.data[dadr])
                length *= gear
                vel *= gear
                force = (
                    Self._acd.motor_kp[i] * (ctrl - length)
                    - Self._acd.motor_kv[i] * vel
                )

            for k in range(n):
                var dadr = Self._acd.motor_trn_dadr[i * 4 + k]
                if dadr < 0 or dadr >= Self.NV:
                    continue
                d.qfrc.data[dadr] += Scalar[DTYPE](
                    gear * Self._acd.motor_trn_coef[i * 4 + k] * force
                )

        # Fixed-tendon springs (`engine_passive.c`, tendon-level spring):
        # a DEADBAND on `tendon_lengthspring`, zero inside the band.
        for t in range(Self._acd.ntendon):
            var k_spring = Self._acd.tendon_stiffness[t]
            if k_spring == 0.0:
                continue
            var n = Self._acd.tendon_trn_n[t]
            if n == 0:
                continue
            var length = Float64(0)
            for k in range(n):
                var qadr = Self._acd.tendon_trn_qadr[t * 4 + k]
                if qadr >= 0 and qadr < Self.NQ:
                    length += (
                        Self._acd.tendon_trn_coef[t * 4 + k]
                        * Float64(d.qpos.data[qadr])
                    )
            var lo = Self._acd.tendon_spring_lo[t]
            var hi = Self._acd.tendon_spring_hi[t]
            var frc = Float64(0)
            if length > hi:
                frc = k_spring * (hi - length)
            elif length < lo:
                frc = k_spring * (lo - length)
            if frc == 0.0:
                continue
            for k in range(n):
                var dadr = Self._acd.tendon_trn_dadr[t * 4 + k]
                if dadr < 0 or dadr >= Self.NV:
                    continue
                d.qfrc.data[dadr] += Scalar[DTYPE](
                    Self._acd.tendon_trn_coef[t * 4 + k] * frc
                )

    # =========================================================================
    # Model build (spec-direct; G4)
    # =========================================================================

    # =========================================================================
    # GPU: _compute_invweight0_gpu (duplicated from ModelDef, dims from params)
    # =========================================================================

    @staticmethod
    def init_fields[
        DTYPE: DType, NMESHV: Int = 0
    ](
        ctx: DeviceContext,
        mut mf: Model[
            DTYPE,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.MAX_TENDON,
            Self.NSITE,
            Self.NEXCLUDE,
            NMESHV,
        ],
    ) raises:
        """Spec-direct fields model build (G4): parse the XML into a
        FlatModelDef and write the packed record tensors DIRECTLY
        (`fields_build.build_model_fields_from_flat`) — no CPU `Model`/`Data`
        staging, no `setup_model_and_data`, no `load_from_model`. invweight0
        is computed fields-natively (G1) from the reference pose given by the
        fields `reset_data`. The legacy trait-default (setup_model_and_data →
        load_from_model) was deleted at G4."""
        var fmd = parse_xml_full[
            Self.NBODY,
            Self.NJOINT,
            Self.NQ,
            Self.NV,
            Self.NGEOM,
            Self.nact,
            Self.ntex,
            Self.nmat,
            Self.nlight,
            Self.ncam,
            Self.NSITE,
            Self.neq,
            Self.nexclude,
            Self.MAX_TENDON,
        ](Self.xml)

        # Reject unplumbed dof-friction solver params LOUDLY, for the same
        # reason as the actuator guard below: the parser sees the attribute,
        # the model would build, and `friction_dof.mojo` would quietly use
        # MuJoCo's DEFAULT solref/solimp instead of the ones the XML asked for.
        for j in range(Self.njoint):
            if fmd.joints[j].has_friction_solparams:
                raise Error(
                    String(
                        "physics3d: joint index ",
                        j,
                        " sets solreffriction/solimpfriction, which are not"
                        " plumbed. constraints/friction_dof.mojo assumes"
                        " MuJoCo's defaults (solref 0.02 1, solimp"
                        " 0.9 0.95 0.001 0.5 2); give them their own model"
                        " meta slots beside the limit ones before using these"
                        " attributes.",
                    )
                )

        # Reject a <spatial> tendon that has nowhere to go. `max_tendon` sizes
        # the tendon record array, and a model that leaves it 0 while the XML
        # routes a string through sites would build fine and simulate with the
        # string simply absent — ball_in_cup would become a ball falling past
        # a cup. A <fixed> tendon is exempt: those also ride the comptime
        # transmission/spring path, which needs no records (fish).
        comptime if Self.MAX_TENDON == 0:
            comptime if Self.xml.find("<spatial") != -1:
                raise Error(
                    "physics3d: model declares a <spatial> tendon but"
                    " max_tendon=0, so it would be silently dropped. Set"
                    " max_tendon to the tendon count."
                )

        # Tendon LIMIT rows are built only on the PYRAMIDAL edge list. The
        # elliptic core keeps its scalar rows in (dof, sign) form to stay
        # under Metal's local-memory ceiling, so a dense tendon row there is
        # a separate change. No model needs it yet — but a model that did
        # would otherwise simulate with the limit simply absent.
        comptime if Self.cone_type == ConeType.ELLIPTIC:
            for t in range(Self.MAX_TENDON):
                if fmd.tendons[t].limited != 0:
                    raise Error(
                        "physics3d: tendon limits are implemented on the"
                        " PYRAMIDAL cone only; this model sets cone=elliptic"
                        " and has a limited tendon. See"
                        " constraints/tendon_limit.mojo."
                    )

        # A `<fixed>` tendon past the comptime cap is dropped SILENTLY by
        # `parse_xml_model_data`, taking every actuator transmitted through it
        # with it (`motor_trn_n == 0`, which `apply_actions` skips). Fail to
        # compile instead.
        comptime assert Self.MAX_TENDON <= MAX_COMPTIME_TENDONS, (
            "physics3d: this model has more <fixed> tendons than"
            " MAX_COMPTIME_TENDONS; raise it in xml_parser.mojo. Leaving it"
            " would silently drop the surplus tendons AND disable every"
            " actuator transmitted through them."
        )

        # A `<general>` whose gain/bias/dyn shape we do not implement. The
        # comptime parser cannot raise, so it records the offender and we turn
        # that into a compile error here. Codes are documented on the field.
        comptime assert Self._acd.bad_actuator < 0, (
            "physics3d: <general> actuator with an unsupported gain/bias/dyn"
            " shape. Supported: gaintype=fixed, biastype=affine, biasprm[0]"
            " == 0, biasprm[1] == -gainprm[0] (i.e. a position servo), and"
            " dyntype none|filter. See ComptimeActData.bad_actuator_code."
        )

        # Reject unimplemented actuator transmissions LOUDLY. Building the
        # model anyway would simulate a servo as a torque motor with no error
        # at all. `<motor>`, `<position>` and the `<general>` shape validated
        # just above are modelled (see `apply_actions`); `<velocity>` is
        # recognized by the parser purely so the tag count is right, and is
        # rejected here. See docs/DM_CONTROL_PORT.md (gap G3).
        comptime if not Self.allow_unsupported_actuators:
            for a in range(Self.nact):
                var kind = fmd.actuators[a].kind
                if (
                    kind != ACT_KIND_MOTOR
                    and kind != ACT_KIND_POSITION
                    and kind != ACT_KIND_GENERAL
                ):
                    raise Error(
                        String(
                            "physics3d: unimplemented actuator transmission ",
                            act_kind_name(kind),
                            " at actuator index ",
                            a,
                            ". <motor> (force = gear*ctrl), <position>",
                            " (force = kp*(ctrl - length) - kv*velocity) and",
                            " <general> restricted to that same affine shape",
                            " are supported; <velocity> needs its own",
                            " gainprm/biasprm handling. If this env's CONFIG",
                            " drives those DOFs itself",
                            " (custom_apply_actions_cpu -> True), pass",
                            " allow_unsupported_actuators=True.",
                        )
                    )
                # Same trap one level down: `site=`, `body=` and `cranksite=`
                # are valid MJCF transmissions that carry neither a `joint`
                # nor a `tendon` attribute, so nothing resolves and the
                # actuator would be built against a garbage index instead of
                # failing. `tendon=` IS resolved (fish's `fins_flap`), so the
                # check is on the comptime transmission list rather than on
                # `joint_id`, which a tendon transmission legitimately leaves
                # at its -1 sentinel.
                if Self._acd.motor_trn_n[a] == 0:
                    raise Error(
                        String(
                            "physics3d: actuator index ",
                            a,
                            " has no resolvable transmission. `joint=` and",
                            " `tendon=` (fixed tendons) are modelled;",
                            " site/body/cranksite are not. Rewrite it as a",
                            " joint actuator if the transmission is",
                            " equivalent, or pass",
                            " allow_unsupported_actuators=True if this env's",
                            " CONFIG drives the DOF itself.",
                        )
                    )

        # ELLIPSOID narrow phase, as of 2026-07-31, covers the PLANE pair only
        # (`ellipsoid_plane`, the closed-form support point that MuJoCo reaches
        # via mjc_PlaneConvex; verified to 1.1e-16 against MuJoCo over 3041
        # contacts). This check used to reject any collidable ellipsoid
        # outright, which is no longer right — but it is not yet right to
        # accept every pair either.
        #
        # ⚠ KNOWN LIMITATION, deliberately not a hard error. An ellipsoid
        # paired with a sphere/capsule/box/mesh matches NO branch of the
        # narrow-phase dispatch and therefore yields NO CONTACT — silently.
        # (Missing a contact is at least safer than the sphere-of-size[0]
        # substitution this check originally guarded against.) It is left
        # permitted because the static test that would catch it — "does a
        # collidable non-plane geom exist that MuJoCo would pair with this
        # ellipsoid" — is true for quadruped (torso vs the leg capsules, which
        # are not parent/child) while the contact is unreachable in practice:
        # measured over 60,000 MuJoCo steps of aggressive random control from
        # 40 random orientations, the torso ellipsoid touched ONLY the floor
        # (27,748 contacts, zero against any capsule). A model that does need
        # ellipsoid-vs-convex must add it; see mojo_rl/envs/ROADMAP.md.
        comptime if not Self.allow_unsupported_actuators:
            for g in range(Self.NGEOM):
                if fmd.geoms[g].geom_type != _GEOM_ELLIPSOID:
                    continue
                if fmd.geoms[g].contype == 0 and fmd.geoms[g].conaffinity == 0:
                    continue
                # Reachable and handled: plane pairs. Nothing to reject.
                break

        comptime ifg_mode = _xml_compiler_inertiafromgeom[Self.xml]()
        comptime igr = _xml_compiler_inertiagrouprange[Self.xml]()
        comptime stm = _xml_compiler_settotalmass[Self.xml]()
        build_model_fields_from_flat[
            DTYPE,
            Self.NBODY,
            Self.NJOINT,
            Self.NQ,
            Self.NV,
            Self.NGEOM,
            Self.nact,
            Self.ntex,
            Self.nmat,
            Self.nlight,
            Self.ncam,
            Self.NSITE,
            Self.neq,
            Self.nexclude,
            Self.MAX_TENDON,
            Self.MAX_EQUALITY,
            Self.MAX_TENDON,
            Self.NSITE,
            Self.NEXCLUDE,
            NMESHV,
            ifg_mode,
            igr[0],
            igr[1],
            stm,
        ](fmd, mf)

        # Reference pose + fields-native invweight0 (G1).
        var d_inv = Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.MAX_CONTACTS,
            Self.NSITE,
            1,
        ]()
        Self.reset_data[DTYPE](d_inv)
        var sc_inv = DynamicsScratch[DTYPE, Self.NV, Self.NBODY, 1]()
        compute_invweight0[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.MAX_TENDON,
            Self.NSITE,
            Self.NEXCLUDE,
            NMESHV,
        ](d_inv, mf, sc_inv)
        mf.upload_all(ctx)

    # =========================================================================
    # GPU: Joints / Actuators kernel delegates
    # =========================================================================

    @staticmethod
    def apply_actions_kernel_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        qfrc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, Self.NV), MutAnyOrigin
        ],
        actions: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
    ) raises:
        """GPU kernel: apply gear * clamp(action, ctrlrange) to qfrc for each
        actuator, straight on the per-field qfrc tensor (G5 — no state slab).
        """
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @parameter
        @always_inline
        def apply_kernel(
            qfrc: LayoutTensor[
                DTYPE, Layout.row_major(BATCH_SIZE, Self.NV), MutAnyOrigin
            ],
            actions: LayoutTensor[
                DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            comptime for act_i in range(Self.nact):
                comptime gear = Self._acd.motor_gears[act_i]
                comptime dof = Self._acd.motor_dof_adr[act_i]
                comptime c_min = Self._acd.motor_ctrl_min[act_i]
                comptime c_max = Self._acd.motor_ctrl_max[act_i]

                comptime if dof >= 0 and dof < Self.NV:
                    var ctrl = rebind[Scalar[DTYPE]](actions[env, act_i])
                    if ctrl > Scalar[DTYPE](c_max):
                        ctrl = Scalar[DTYPE](c_max)
                    elif ctrl < Scalar[DTYPE](c_min):
                        ctrl = Scalar[DTYPE](c_min)
                    qfrc[env, dof] = Scalar[DTYPE](gear) * ctrl

        ctx.enqueue_function[apply_kernel](
            qfrc,
            actions,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU inline: Per-env methods
    # =========================================================================

    @always_inline
    @staticmethod
    def reset_env_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, Self.NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, Self.NV), MutAnyOrigin
        ],
        qacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, Self.NV), MutAnyOrigin
        ],
        qfrc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, Self.NV), MutAnyOrigin
        ],
        env: Int,
        noise_scale: Scalar[DTYPE],
        seed: Int,
    ):
        """Reset a single env with small random noise around qpos0, qvel=0."""
        comptime TOTAL_VALS = Self.NQ + Self.NV
        comptime NUM_BATCHES = (TOTAL_VALS + 3) // 4

        var rng = PhiloxRandom(
            seed=UInt64(seed * 2654435761 + env * 12345), offset=0
        )
        var rand_vals = InlineArray[Scalar[DType.float32], NUM_BATCHES * 4](
            fill=Scalar[DType.float32](0)
        )
        for b in range(NUM_BATCHES):
            var batch = rng.step_uniform()
            rand_vals[b * 4 + 0] = batch[0]
            rand_vals[b * 4 + 1] = batch[1]
            rand_vals[b * 4 + 2] = batch[2]
            rand_vals[b * 4 + 3] = batch[3]

        comptime for i in range(Self.NQ):
            var noise = Scalar[DTYPE](rand_vals[i] * 2.0 - 1.0) * noise_scale
            comptime if Self._acd.nq > 0 and i < Self._acd.nq:
                comptime val = Self._acd.qpos0[i]
                qpos[env, i] = Scalar[DTYPE](val) + noise
            else:
                comptime if (
                    Self._acd.free_joint_qpos_adr >= 0
                    and i == Self._acd.free_joint_qpos_adr + 3
                ):
                    # Free-joint qw: start from identity (1.0) + small noise.
                    qpos[env, i] = Scalar[DTYPE](1) + noise
                else:
                    qpos[env, i] = noise

        comptime for i in range(Self.NV):
            var noise = (
                Scalar[DTYPE](rand_vals[Self.NQ + i] * 2.0 - 1.0) * noise_scale
            )
            qvel[env, i] = noise

        for i in range(Self.NV):
            qacc[env, i] = Scalar[DTYPE](0)
            qfrc[env, i] = Scalar[DTYPE](0)

    @always_inline
    @staticmethod
    def extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        OBS_DIM: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, Self.NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, Self.NV), MutAnyOrigin
        ],
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        """Extract obs = qpos[obs_qpos_skip:] + qvel[:] for a single env."""
        comptime for i in range(Self.NQ - Self.obs_qpos_skip):
            obs[env, i] = qpos[env, Self.obs_qpos_skip + i]

        comptime for i in range(Self.NV):
            obs[env, Self.NQ - Self.obs_qpos_skip + i] = qvel[env, i]

    # =========================================================================
    # Rendering — driven from parsed XML assets, lights, cameras, geoms
    # =========================================================================

    @staticmethod
    def setup_lights() raises -> List[Light]:
        """Return Light objects parsed from <light> elements in <worldbody>."""
        var lights = List[Light]()
        for i in range(Self.nlight):
            var mode = Int(1) if Self._rcd.light_directional[i] else Int(0)
            var amb = (Self._rcd.light_ambient_r[i] + Self._rcd.light_ambient_g[i] + Self._rcd.light_ambient_b[i]) / 3.0
            var spec_int = (Self._rcd.light_specular_r[i] + Self._rcd.light_specular_g[i] + Self._rcd.light_specular_b[i]) / 3.0
            lights.append(
                Light(
                    mode=mode,
                    dir_x=Self._rcd.light_dir_x[i],
                    dir_y=Self._rcd.light_dir_y[i],
                    dir_z=Self._rcd.light_dir_z[i],
                    color_r=Self._rcd.light_diffuse_r[i],
                    color_g=Self._rcd.light_diffuse_g[i],
                    color_b=Self._rcd.light_diffuse_b[i],
                    ambient=amb,
                    specular_intensity=spec_int,
                    specular_exponent=Self._rcd.light_exponent[i],
                    cast_shadow=Self._rcd.light_castshadow[i],
                )
            )
        return lights^

    @staticmethod
    def setup_cameras(width: Int, height: Int) raises -> List[Camera3D]:
        """Return Camera3D objects parsed from <camera> elements in <worldbody>."""
        var cameras = List[Camera3D]()
        for i in range(Self.ncam):
            var eye = _RVec3(Self._rcd.cam_pos_x[i], Self._rcd.cam_pos_y[i], Self._rcd.cam_pos_z[i])
            var target: _RVec3
            var cam_mode = Self._rcd.cam_mode[i]
            if cam_mode == 0 or cam_mode == 1 or cam_mode == 2:
                target = _RVec3(Self._rcd.cam_pos_x[i], Float64(0), Float64(0))
            else:
                var qx = Self._rcd.cam_quat_x[i]
                var qy = Self._rcd.cam_quat_y[i]
                var qz = Self._rcd.cam_quat_z[i]
                var qw = Self._rcd.cam_quat_w[i]
                var vx = Float64(0)
                var vy = Float64(0)
                var vz = Float64(-1)
                var tx = 2.0 * (qy * vz - qz * vy)
                var ty = 2.0 * (qz * vx - qx * vz)
                var tz = 2.0 * (qx * vy - qy * vx)
                var look_x = vx + qw * tx + qy * tz - qz * ty
                var look_y = vy + qw * ty + qz * tx - qx * tz
                var look_z = vz + qw * tz + qx * ty - qy * tx
                target = _RVec3(
                    Self._rcd.cam_pos_x[i] + look_x,
                    Self._rcd.cam_pos_y[i] + look_y,
                    Self._rcd.cam_pos_z[i] + look_z,
                )
            cameras.append(
                Camera3D(
                    eye=eye,
                    target=target,
                    up=_RVec3(0.0, 0.0, 1.0),
                    fov=Self._rcd.cam_fovy[i],
                    aspect=Float64(width) / Float64(height),
                    near=Float64(0.1),
                    far=Float64(100.0),
                    screen_width=width,
                    screen_height=height,
                )
            )
        return cameras^

    @staticmethod
    def setup_camera_modes() raises -> List[Int]:
        """Return camera modes (CAM_MODE_* constants) for each parsed camera."""
        var modes = List[Int]()
        for i in range(Self.ncam):
            var xml_mode = Self._rcd.cam_mode[i]
            if xml_mode == 0:
                modes.append(1)  # CAM_MODE_FIXED -> renderer CAM_FIXED=1
            else:
                modes.append(0)  # TRACK / TRACKCOM / TARGET* -> renderer CAM_TRACKCOM=0
        return modes^

    @staticmethod
    def get_skybox_colors() -> List[Float64]:
        """Return [top_r, top_g, top_b, bottom_r, bottom_g, bottom_b] from the
        first skybox/gradient texture, or an empty list if none exists."""
        # TEX_SKYBOX=1, TEX_BUILTIN_GRADIENT=1
        for i in range(Self.ntex):
            if Self._rcd.tex_type[i] == 1 or Self._rcd.tex_builtin[i] == 1:
                var result = List[Float64]()
                result.append(Self._rcd.tex_rgb1_r[i])
                result.append(Self._rcd.tex_rgb1_g[i])
                result.append(Self._rcd.tex_rgb1_b[i])
                result.append(Self._rcd.tex_rgb2_r[i])
                result.append(Self._rcd.tex_rgb2_g[i])
                result.append(Self._rcd.tex_rgb2_b[i])
                return result^
        return List[Float64]()

    @staticmethod
    def get_checker_colors() -> List[Float64]:
        """Return [r, g, b] of the checker texture's secondary (light square) colour,
        or an empty list if no checker texture is found."""
        # TEX_BUILTIN_CHECKER=2
        for i in range(Self.ntex):
            if Self._rcd.tex_builtin[i] == 2:
                var result = List[Float64]()
                result.append(Self._rcd.tex_rgb2_r[i])
                result.append(Self._rcd.tex_rgb2_g[i])
                result.append(Self._rcd.tex_rgb2_b[i])
                return result^
        return List[Float64]()

    @staticmethod
    def get_ground_rgba() -> List[Float64]:
        """Return [r, g, b] of the first plane geom's rgba color,
        or empty list if no plane geom exists."""
        for i in range(Self.NGEOM):
            if Self._rcd.geom_type[i] == 0:  # GEOM_PLANE
                var result = List[Float64]()
                result.append(Self._rcd.geom_rgba_r[i])
                result.append(Self._rcd.geom_rgba_g[i])
                result.append(Self._rcd.geom_rgba_b[i])
                return result^
        return List[Float64]()

    @staticmethod
    def get_visual_settings() -> List[Float64]:
        """Return [znear, fogstart, fogend, shadowsize, hl_r, hl_g, hl_b, has_headlight]."""
        var result = List[Float64]()
        result.append(Self._rcd.vis_znear)
        result.append(Self._rcd.vis_fogstart)
        result.append(Self._rcd.vis_fogend)
        result.append(Float64(Self._rcd.vis_shadowsize))
        result.append(Self._rcd.vis_headlight_ambient_r)
        result.append(Self._rcd.vis_headlight_ambient_g)
        result.append(Self._rcd.vis_headlight_ambient_b)
        result.append(Float64(1.0) if Self._rcd.vis_has_headlight else Float64(0.0))
        return result^

    @staticmethod
    def render_ground_geoms(
        mut renderer: Renderer3D,
        torso_x: Float64,
        follow: Bool,
        visual_radius_scale: Float64,
    ) raises:
        """Draw plane geoms (body_id=0) as ground grids; fallback if none."""
        # GEOM_PLANE=0
        var has_plane = False
        var max_body_radius = Float64(0.0)
        for j in range(Self.NGEOM):
            if Self._rcd.geom_body_id[j] > 0 and Self._rcd.geom_radius[j] > max_body_radius:
                max_body_radius = Self._rcd.geom_radius[j]
        for i in range(Self.NGEOM):
            if Self._rcd.geom_type[i] == 0:  # PLANE
                has_plane = True
                var ground_offset = Self._rcd.geom_pos_z[i] - max_body_radius * (visual_radius_scale - 1.0)
                var grid_cx = torso_x if follow else Float64(0.0)
                # Resolve material → texture for this plane geom
                var tex_name = String("")
                var tex_file = String("")
                var texrep_u = Float64(1.0)
                var texrep_v = Float64(1.0)
                var mid = Self._rcd.geom_material_id[i]
                if mid >= 0 and mid < Self.nmat:
                    var tex_id = Self._rcd.mat_tex_id[mid]
                    if tex_id >= 0 and tex_id < Self._rcd.ntex:
                        comptime for ti in range(Self._rcd.ntex):
                            if tex_id == ti:
                                comptime _tn: String = Self._rcd.tex_names[ti]
                                comptime _tf: String = Self._rcd.tex_files[ti]
                                tex_name = _tn
                                tex_file = _tf
                    texrep_u = Self._rcd.mat_texrepeat_u[mid]
                    texrep_v = Self._rcd.mat_texrepeat_v[mid]
                renderer.draw_ground_grid(
                    grid_cx, height=ground_offset,
                    texture_name=tex_name, texture_path=tex_file,
                    texrepeat_u=texrep_u, texrepeat_v=texrep_v,
                )
        if not has_plane:
            # No ground plane defined in XML — skip ground rendering.
            # Models like InvertedPendulum intentionally omit the ground.
            pass

    @staticmethod
    def render_body_geoms(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
        visual_radius_scale: Float64,
    ) raises:
        """Draw body-attached geoms using parsed geometry + colour."""
        # SPHERE=1, CAPSULE=2, BOX=3, CYLINDER=4, MESH=5
        for i in range(Self.NGEOM):
            var bid = Self._rcd.geom_body_id[i]
            if bid < 0:
                continue
            # Skip plane geoms (handled by render_ground_geoms)
            if Self._rcd.geom_type[i] == 0:
                continue
            if bid >= len(positions):
                continue
            # Skip geoms with alpha < 1 (collision-only / semi-transparent)
            if Self._rcd.geom_rgba_a[i] < 0.99:
                continue
            var body_pos = positions[bid]
            var body_quat = quaternions[bid]
            var gx = Self._rcd.geom_pos_x[i]
            var gy = Self._rcd.geom_pos_y[i]
            var gz = Self._rcd.geom_pos_z[i]
            var geom_pos: _RVec3
            if gx == 0.0 and gy == 0.0 and gz == 0.0:
                geom_pos = body_pos
            else:
                geom_pos = body_pos + body_quat.rotate_vec(_RVec3(gx, gy, gz))
            var gqx = Self._rcd.geom_quat_x[i]
            var gqy = Self._rcd.geom_quat_y[i]
            var gqz = Self._rcd.geom_quat_z[i]
            var gqw = Self._rcd.geom_quat_w[i]
            var geom_quat: _RQuat
            if gqx == 0.0 and gqy == 0.0 and gqz == 0.0 and gqw == 1.0:
                geom_quat = body_quat
            else:
                geom_quat = body_quat * _RQuat(gqw, gqx, gqy, gqz)
            var r = Float32(Self._rcd.geom_rgba_r[i])
            var g = Float32(Self._rcd.geom_rgba_g[i])
            var b = Float32(Self._rcd.geom_rgba_b[i])
            var a = Float32(Self._rcd.geom_rgba_a[i])
            var mid = Self._rcd.geom_material_id[i]
            if mid >= 0 and mid < Self.nmat:
                r = Float32(Self._rcd.mat_rgba_r[mid])
                g = Float32(Self._rcd.mat_rgba_g[mid])
                b = Float32(Self._rcd.mat_rgba_b[mid])
                a = Float32(Self._rcd.mat_rgba_a[mid])
            var geom_color = Color(UInt8(r * 255), UInt8(g * 255), UInt8(b * 255), UInt8(a * 255))
            var shininess = Float32(0.5)
            var specular = Float32(0.5)
            var reflectance = Float32(0.0)
            if mid >= 0 and mid < Self.nmat:
                shininess = Float32(Self._rcd.mat_shininess[mid])
                specular = Float32(Self._rcd.mat_specular[mid])
                reflectance = Float32(Self._rcd.mat_reflectance[mid])
            # Resolve material → texture chain for this geom
            var tex_name_str = String("")
            var tex_file_str = String("")
            if mid >= 0 and mid < Self.nmat:
                var tex_id = Self._rcd.mat_tex_id[mid]
                if tex_id >= 0 and tex_id < Self._rcd.ntex:
                    comptime for ti in range(Self._rcd.ntex):
                        if tex_id == ti:
                            comptime _tn: String = Self._rcd.tex_names[ti]
                            comptime _tf: String = Self._rcd.tex_files[ti]
                            tex_name_str = _tn
                            tex_file_str = _tf

            var gt = Self._rcd.geom_type[i]
            if gt == 2:  # CAPSULE
                renderer.draw_capsule(center=geom_pos, orientation=geom_quat,
                    radius=Self._rcd.geom_radius[i] * visual_radius_scale,
                    half_height=Self._rcd.geom_half_length[i], axis=2,
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance,
                    texture_name=tex_name_str, texture_path=tex_file_str)
            elif gt == 1:  # SPHERE
                renderer.draw_sphere(center=geom_pos,
                    radius=Self._rcd.geom_radius[i] * visual_radius_scale,
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance,
                    texture_name=tex_name_str, texture_path=tex_file_str)
            elif gt == 3:  # BOX
                renderer.draw_box(center=geom_pos, orientation=geom_quat,
                    half_extents=_RVec3(Self._rcd.geom_half_x[i], Self._rcd.geom_half_y[i], Self._rcd.geom_half_z[i]),
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance,
                    texture_name=tex_name_str, texture_path=tex_file_str)
            elif gt == 4:  # CYLINDER
                renderer.draw_cylinder(center=geom_pos, orientation=geom_quat,
                    radius=Self._rcd.geom_radius[i] * visual_radius_scale,
                    half_height=Self._rcd.geom_half_length[i], axis=2,
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance,
                    texture_name=tex_name_str, texture_path=tex_file_str)
            elif gt == 5:  # MESH
                var mid2 = Self._rcd.geom_mesh_id[i]
                # Draw mesh with optional texture
                comptime for mi in range(Self._rcd.nmesh):
                    if mid2 == mi:
                        comptime _mn: String = Self._rcd.mesh_names[mi]
                        comptime _mf: String = Self._rcd.mesh_files[mi]
                        renderer.draw_mesh(
                            name=_mn, file_path=_mf,
                            center=geom_pos, orientation=geom_quat,
                            color=geom_color, shininess=shininess,
                            specular=specular, reflectance=reflectance,
                            texture_name=tex_name_str,
                            texture_path=tex_file_str,
                        )

    @staticmethod
    def render_sites(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Draw all sites as small bright-green spheres (visual markers)."""
        for i in range(Self.NSITE):
            var sbid = Self._rcd.site_body_id[i]
            if sbid <= 0 or sbid >= len(positions):
                continue
            var body_pos = positions[sbid]
            var body_quat = quaternions[sbid]
            var sx = Self._rcd.site_pos_x[i]
            var sy = Self._rcd.site_pos_y[i]
            var sz = Self._rcd.site_pos_z[i]
            var site_world_pos: _RVec3
            if sx == 0.0 and sy == 0.0 and sz == 0.0:
                site_world_pos = body_pos
            else:
                site_world_pos = body_pos + body_quat.rotate_vec(_RVec3(sx, sy, sz))
            var radius = Self._rcd.site_size_0[i] if Self._rcd.site_size_0[i] > 0.0 else 0.005
            renderer.draw_sphere(
                center=site_world_pos,
                radius=radius,
                color=Color(0, 255, 0, 255),
                shininess=Float32(0.9),
                specular=Float32(0.9),
                reflectance=Float32(0.0),
            )
