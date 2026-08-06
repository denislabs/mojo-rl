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
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
)
from mojo_rl.physics3d.joint_types import JNT_FREE, JNT_BALL
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
    MAX_COMPTIME_TENDON_WRAPS,
    MAX_COMPTIME_RENDER_GEOMS,
    MAX_COMPTIME_RENDER_SITES,
    MAX_COMPTIME_ACTUATORS,
    MAX_COMPTIME_JOINTS,
    MAX_COMPTIME_MATERIALS,
    MAX_COMPTIME_NQ,
    MAX_COMPTIME_TEXTURES,
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


def _attr_between(src: String, lo: Int, hi: Int, attr: String) -> String:
    """`attr="value"` inside `src[lo:hi]`, or "".

    A local reader so `body_names` does not depend on the comptime parser's
    helpers — this one runs at RUNTIME and has no interpreter constraints to
    respect.
    """
    var needle = attr + '="'
    var nlen = needle.byte_length()
    var p = src.find(needle, lo)
    while p != -1 and p < hi:
        var ok = p == 0
        if not ok:
            var c = src.as_bytes()[p - 1]
            ok = c == 32 or c == 9 or c == 10 or c == 13
        if ok:
            var vs = p + nlen
            var ve = src.find('"', vs)
            if ve != -1 and ve <= hi:
                return String(src[byte=vs:ve])
        p = src.find(needle, p + 1)
    return String("")


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
    max_condim: Int = 3,
    noslip_iter: Int = 0,
    allow_missing_noslip: Bool = False,
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
        max_condim:    Largest condim over the model's geoms; pass
                       `parse_xml(xml).MAX_CONDIM`. Over-estimating only
                       costs unused rows, under-estimating is silent.
        noslip_iter:   `<option noslip_iterations>`; pass
                       `parse_xml(xml).NOSLIP_ITER`. 0 disables the pass,
                       which is MuJoCo's default and correct for every suite
                       model except dog.
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
    # Largest condim in the model — sizes the PYRAMIDAL edge list at
    # 2*(MAX_CONDIM-1) rows per contact. Pass `parse_xml(xml).MAX_CONDIM`;
    # leaving it at 3 on a model with condim 4/6 geoms builds the torsional
    # and rolling rows into a workspace the solver never reads.
    comptime MAX_CONDIM: Int = Self.max_condim
    # `<option noslip_iterations>`. Runs MuJoCo's `mj_solNoSlip` after the
    # primal solve: a friction-only Gauss-Seidel sweep with the normal forces
    # held fixed. 0 = off, which is MuJoCo's default. dm_control's dog is the
    # only in-scope model that sets it (to 4), and it is first-order there —
    # 2.9e-2 of qvel on the first contacting step.
    comptime NOSLIP_ITER: Int = Self.noslip_iter
    comptime MAX_CONTACTS: Int = Self.max_contacts
    comptime MAX_TENDON: Int = Self.max_tendon
    comptime NSITE: Int = Self.nsite
    # MuJoCo `m->na` — activation variables, NOT `nu`. Nonzero only for
    # actuators with a `dyntype`; 0 for every <motor>/<position>.
    comptime NA: Int = Self._acd.na
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
        mut act: List[Scalar[DTYPE]],
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

            # ACTIVATION (MuJoCo `d->act`). `force = gain .* [ctrl/act]`
            # (mj_fwdActuation): an actuator with a `dyntype` feeds its
            # activation to the gain where a plain one feeds `ctrl`. The
            # activation itself is a first-order lag of `ctrl`.
            #
            # `u` is what the gain multiplies. `act` is integrated AFTER the
            # force is computed, matching MuJoCo's order — `mj_fwdActuation`
            # reads the current `act`, and `mj_advance` advances it at the end
            # of the same step (`actearly` is off here). This function runs
            # ONCE PER SUBSTEP, which is the same cadence, so the two agree
            # step for step.
            var adr = Self._acd.motor_act_adr[i]
            var u = ctrl
            if adr >= 0 and adr < len(act):
                u = Float64(act[adr])

            # `motor_kp` is MuJoCo's `gainprm[0]`, whose default is 1 — so a
            # plain `<motor>`, which never writes it, is `force = ctrl`. A
            # bias-free `<general>` lands here too and its gain is real: dog's
            # actuators are `force = 0.02 * act`.
            var force = Self._acd.motor_kp[i] * u
            if Self._acd.motor_kind[i] == ACT_KIND_POSITION:
                var length = Float64(0)
                var vel = Float64(0)
                for k in range(n):
                    var qadr = Self._acd.motor_trn_qadr[i * MAX_COMPTIME_TENDON_WRAPS + k]
                    var dadr = Self._acd.motor_trn_dadr[i * MAX_COMPTIME_TENDON_WRAPS + k]
                    var coef = Self._acd.motor_trn_coef[i * MAX_COMPTIME_TENDON_WRAPS + k]
                    if qadr >= 0 and qadr < Self.NQ:
                        length += coef * Float64(d.qpos.data[qadr])
                    if dadr >= 0 and dadr < Self.NV:
                        vel += coef * Float64(d.qvel.data[dadr])
                length *= gear
                vel *= gear
                # `u`, not `ctrl` — for a dyntype actuator the servo setpoint
                # is the ACTIVATION, which lags the control. They coincide
                # only when the actuator has no activation (then u == ctrl).
                force = (
                    Self._acd.motor_kp[i] * (u - length)
                    - Self._acd.motor_kv[i] * vel
                )

            for k in range(n):
                var dadr = Self._acd.motor_trn_dadr[i * MAX_COMPTIME_TENDON_WRAPS + k]
                if dadr < 0 or dadr >= Self.NV:
                    continue
                d.qfrc.data[dadr] += Scalar[DTYPE](
                    gear * Self._acd.motor_trn_coef[i * MAX_COMPTIME_TENDON_WRAPS + k] * force
                )

            # mjDYN_FILTER, integrated by Euler exactly as `nextActivation`
            # does for a non-`filterexact` dyntype (engine_forward.c:341):
            #     act_dot = (ctrl - act) / tau ;  act += act_dot * timestep
            # `ctrl` here is already ctrlrange-clamped, matching MuJoCo, which
            # clamps `d->ctrl` before computing act_dot.
            if adr >= 0 and adr < len(act):
                var tau = Self._acd.motor_dyn_tau[i]
                if tau < 1e-10:
                    tau = 1e-10  # mjMINVAL guard, as MuJoCo applies
                act[adr] = Scalar[DTYPE](
                    u + (ctrl - u) / tau * Self.TIMESTEP
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
                var qadr = Self._acd.tendon_trn_qadr[t * MAX_COMPTIME_TENDON_WRAPS + k]
                if qadr >= 0 and qadr < Self.NQ:
                    length += (
                        Self._acd.tendon_trn_coef[t * MAX_COMPTIME_TENDON_WRAPS + k]
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
                var dadr = Self._acd.tendon_trn_dadr[t * MAX_COMPTIME_TENDON_WRAPS + k]
                if dadr < 0 or dadr >= Self.NV:
                    continue
                d.qfrc.data[dadr] += Scalar[DTYPE](
                    Self._acd.tendon_trn_coef[t * MAX_COMPTIME_TENDON_WRAPS + k] * frc
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
        var fmd = parse_xml_full(Self.xml)

        # ⚠ THE DIMENSION CHECK THAT REPLACED SILENT TRUNCATION.
        #
        # `FlatModelDef` used to be `InlineArray`-backed and sized by these
        # very parameters, so the parser wrote `if joint_count < NJOINT:` and
        # incremented the counter REGARDLESS — a model with more elements than
        # its declared dimension dropped the overflow without a word. Same
        # shape as `MAX_COMPTIME_TENDONS` and `MAX_NAMED_DEFAULTS`.
        #
        # With `List` storage the parser cannot truncate, so the failure mode
        # inverts: a disagreement between `parse_xml` (which supplied these
        # comptime dims) and `full_parser` (which produced the Lists) now shows
        # up as a LENGTH MISMATCH, and `fields_build` would index past the end.
        # Catch it here, where both numbers are in hand and the message can say
        # which element type disagreed.
        if len(fmd.bodies) != Self.NBODY - 1:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on BODIES — ",
                    "parse_xml declared nbody=", Self.NBODY, " (so ",
                    Self.NBODY - 1, " non-world bodies) but full_parser found ",
                    len(fmd.bodies),
                    ". The two MJCF paths disagree; fix the parser, do not",
                    " widen the dimension.",
                )
            )
        if len(fmd.joints) != Self.NJOINT:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on JOINTS — declared",
                    " njoint=", Self.NJOINT, ", full_parser found ",
                    len(fmd.joints), ".",
                )
            )
        if len(fmd.geoms) != Self.NGEOM:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on GEOMS — declared",
                    " ngeom=", Self.NGEOM, ", full_parser found ",
                    len(fmd.geoms), ".",
                )
            )
        if len(fmd.sites) != Self.NSITE:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on SITES — declared",
                    " nsite=", Self.NSITE, ", full_parser found ",
                    len(fmd.sites), ". Sensors are addressed BY SITE INDEX, so",
                    " a mismatch here reads the wrong sensor.",
                )
            )

        # ⚠ A TENDON THAT OUTGREW `MAX_COMPTIME_TENDON_WRAPS` MUST NOT BUILD.
        # The wrap loop used to stop at a bare 4 and write `tendon_trn_n = 4`,
        # so a longer tendon looked complete to every consumer. dm_control's
        # dog wraps 11 and 10 joints on its tail tendons, and drove a third of
        # them. Section 4.3 of the port plan is explicit that widening a cap is
        # the easy half and the assert is the load-bearing one — this is that
        # assert, and it is why the parse counts the overflow instead of
        # discarding it.
        # ⚠ SAME RULE FOR THE RENDER RECORD. `ComptimeRenderData`'s geom and
        # site arrays were bare literals (64 and 16) with a SILENT fill guard
        # (`if geom_count < 64:`), while every consumer loops `range(NGEOM)`.
        # dog has 128 geoms, so the viewer indexed 64 past the end — a
        # `debug_assert` in the viewer build, and an out-of-bounds READ in a
        # release one. quadruped (30 sites), humanoid (25) and manipulator (20)
        # were over the site cap identically.
        #
        # Checked against NGEOM/NSITE, which come from `parse_xml` and are the
        # TRUE counts, so this catches the overflow the render fill silently
        # dropped. No counter is needed on `ComptimeRenderData` itself, which
        # also avoids touching its constructors — adding fields there has
        # tripped the comptime interpreter before ("interpreting memcpy can't
        # get dst memory").
        if Self.NGEOM > MAX_COMPTIME_RENDER_GEOMS:
            raise Error(
                String(
                    "physics3d: model has ", Self.NGEOM, " geoms but",
                    " MAX_COMPTIME_RENDER_GEOMS=", MAX_COMPTIME_RENDER_GEOMS,
                    ". Raise it in xml_parser.mojo; truncating leaves the",
                    " renderer reading past the end of every geom array.",
                )
            )
        if Self.NSITE > MAX_COMPTIME_RENDER_SITES:
            raise Error(
                String(
                    "physics3d: model has ", Self.NSITE, " sites but",
                    " MAX_COMPTIME_RENDER_SITES=", MAX_COMPTIME_RENDER_SITES,
                    ". Raise it in xml_parser.mojo; truncating leaves the",
                    " renderer reading past the end of every site array.",
                )
            )

        comptime _acd_wrap = materialize[Self._acd]()
        if _acd_wrap.tendon_wrap_overflow > 0:
            raise Error(
                String(
                    "physics3d: a fixed tendon wraps more joints than",
                    " MAX_COMPTIME_TENDON_WRAPS=", MAX_COMPTIME_TENDON_WRAPS,
                    " (overflow ", _acd_wrap.tendon_wrap_overflow,
                    " on the worst tendon). Raise the cap in xml_parser.mojo;",
                    " it is NOT safe to truncate — the actuator would drive a",
                    " subset of the joints and every gate would still pass.",
                )
            )

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

        # ── The same trap, on three more comptime tables (2026-08-03) ────────
        #
        # `parse_xml_model_data` scans actuators and joints with
        # `while count < CAP`, while `ParsedModel` counts the tags
        # INDEPENDENTLY — so before these asserts a model past either cap built
        # and ran, with the right nact/njoint and a silently truncated `_acd`.
        # The two payloads differ and both are invisible at runtime:
        #   actuators — the env exposes the full action space and every
        #               actuator past the cap applies ZERO FORCE;
        #   joints    — the dof survives but its LIMIT ROW is never built, so
        #               the joint quietly loses its stops.
        # `qpos0` is worse still: it is indexed by the joint's own qpos address
        # rather than scanned, so `nq > MAX_COMPTIME_NQ` writes OUT OF BOUNDS.
        comptime assert Self.nact <= MAX_COMPTIME_ACTUATORS, (
            "physics3d: this model has more actuators than"
            " MAX_COMPTIME_ACTUATORS; raise it in xml_parser.mojo. Leaving it"
            " would expose the full action space while every actuator past the"
            " cap applies zero force."
        )
        comptime assert Self.njoint <= MAX_COMPTIME_JOINTS, (
            "physics3d: this model has more joints than MAX_COMPTIME_JOINTS;"
            " raise it in xml_parser.mojo. Leaving it would keep every dof past"
            " the cap while silently dropping its joint-limit row."
        )
        comptime assert Self.nq <= MAX_COMPTIME_NQ, (
            "physics3d: this model's nq exceeds MAX_COMPTIME_NQ; raise it in"
            " xml_parser.mojo. Unlike the caps above this one is not a"
            " truncating scan — `qpos0` is indexed by qpos address, so it"
            " writes out of bounds."
        )
        # And on the RENDER tables, where it had already gone off. `nmat` here
        # is `ParsedModel.NMAT`, an uncapped `_count_tag`; `_rcd`'s material
        # arrays hold MAX_COMPTIME_MATERIALS. `render_body_geoms` guards its
        # material lookup with `mid < Self.nmat`, so while those two disagreed
        # the guard was checking the WRONG BOUND and let an out-of-range index
        # straight through to the array — point_mass, fish and reacher aborted
        # at the first frame. Anything that survived did so by having every
        # material land under the cap.
        comptime assert Self.nmat <= MAX_COMPTIME_MATERIALS, (
            "physics3d: this model has more <material> records than"
            " MAX_COMPTIME_MATERIALS; raise it in xml_parser.mojo. Leaving it"
            " is not a cosmetic loss: `mid < nmat` then guards against a count"
            " larger than the array it indexes."
        )
        comptime assert Self.ntex <= MAX_COMPTIME_TEXTURES, (
            "physics3d: this model has more <texture> records than"
            " MAX_COMPTIME_TEXTURES; raise it in xml_parser.mojo. Textures past"
            " the cap are dropped, so materials referencing them fall back to"
            " flat colour — including the skybox."
        )

        # `<option noslip_iterations>` — `mj_solNoSlip` is implemented for the
        # PYRAMIDAL cone only (`solver/noslip.mojo`), and only the pyramidal
        # branch of the Newton solver calls it.
        #
        # ⚠ AN ELLIPTIC MODEL WITH noslip_iterations SET WOULD SKIP THE PASS
        # SILENTLY — the elliptic solve path has no call — so it is refused
        # here instead. That is the caller-side dispatch obligation
        # `noslip.mojo` documents, enforced at the one place that knows both
        # the cone type and the option.
        comptime assert (
            Self.noslip_iter == 0
            or Self.cone_type == ConeType.PYRAMIDAL
            or Self.allow_missing_noslip
        ), (
            "physics3d: <option noslip_iterations> is set on an ELLIPTIC-cone"
            " model, but mj_solNoSlip is implemented for the pyramidal cone"
            " only — the elliptic solve path would skip the pass without a"
            " word. Use cone=\"pyramidal\", or pass"
            " allow_missing_noslip=True to accept a rollout that will not"
            " match MuJoCo."
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
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.NGEOM,
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

        # ── AutoSpringDamper (mjCModel::AutoSpringDamper, user_model.cc:2369)
        #
        # ⚠ ORDER IS LOAD-BEARING: this READS `dof_invweight0`, so it must run
        # AFTER `compute_invweight0` and before the upload. MuJoCo does exactly
        # the same thing — `mj_setConst(m, d)` then `AutoSpringDamper(m)`
        # (user_model.cc:5242-5245).
        #
        # `<joint springdamper="timeconst dampratio">` asks MuJoCo to DERIVE
        # the spring from the body's own inertia rather than take a number:
        #
        #     inertia   = ndim / sum(dof_invweight0[adr .. adr+ndim])
        #     stiffness = inertia / (timeconst^2 * dampratio^2)
        #     damping   = 2 * inertia / timeconst
        #
        # and it OVERWRITES whatever `stiffness`/`damping` the XML or class
        # supplied. dm_control's dog declares `springdamper="0.001 50"` once,
        # in a default class, which is why ~20 of its `jnt_stiffness` values
        # (0.0400187, 0.0401469, ...) appear NOWHERE in the XML and why our
        # stiffness was wrong before this. It is not a cosmetic mismatch: the
        # same formula sets `dof_damping`, so getting it wrong is a passive-
        # force error, i.e. a dynamics defect.
        #
        # Both parameters must be strictly positive for MuJoCo to act, which
        # is why (0, 0) is a sufficient "absent" encoding.
        comptime _MJMINVAL = 1e-15
        for j in range(Self.NJOINT):
            # `mjCJoint::nv` — dofs per joint type. Free is 6 (not 7: qpos and
            # dof sizes differ for a free joint, and this formula wants DOFS).

            var jd = fmd.joints[j]
            var tc = jd.springdamper_0
            var dr = jd.springdamper_1
            if tc <= 0.0 or dr <= 0.0:
                continue
            var jo = j * MODEL_JOINT_SIZE
            var dof_adr = Int(mf.joints.data[jo + JOINT_IDX_DOF_ADR])
            var jt = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
            var ndim = 6 if jt == JNT_FREE else (3 if jt == JNT_BALL else 1)
            var acc = Float64(0)
            for i in range(ndim):
                acc += Float64(mf.dof_invweight0.data[dof_adr + i])
            if acc < _MJMINVAL:
                acc = _MJMINVAL
            var inertia = Float64(ndim) / acc
            var denom = tc * tc * dr * dr
            if denom < _MJMINVAL:
                denom = _MJMINVAL
            var tc_d = tc if tc > _MJMINVAL else _MJMINVAL
            mf.joints.data[jo + JOINT_IDX_STIFFNESS] = Scalar[DTYPE](
                inertia / denom
            )
            mf.joints.data[jo + JOINT_IDX_DAMPING] = Scalar[DTYPE](
                2.0 * inertia / tc_d
            )

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
    def _rcd_rotate_by_quat(
        qx: Float64, qy: Float64, qz: Float64, qw: Float64,
        vx: Float64, vy: Float64, vz: Float64,
    ) -> List[Float64]:
        """Rotate (vx,vy,vz) by the quaternion, returned as [x, y, z]."""
        var tx = 2.0 * (qy * vz - qz * vy)
        var ty = 2.0 * (qz * vx - qx * vz)
        var tz = 2.0 * (qx * vy - qy * vx)
        var out = List[Float64]()
        out.append(vx + qw * tx + qy * tz - qz * ty)
        out.append(vy + qw * ty + qz * tx - qx * tz)
        out.append(vz + qw * tz + qx * ty - qy * tx)
        return out^

    @staticmethod
    def setup_cameras(width: Int, height: Int) raises -> List[Camera3D]:
        """Return Camera3D objects parsed from <camera> elements in <worldbody>.

        ⚠ THE LOOK DIRECTION AND THE UP VECTOR BOTH COME FROM THE CAMERA'S OWN
        ORIENTATION, for every mode. MuJoCo's camera frame looks along its -Z
        with +Y up (`mjCCamera`), and `mode` governs only whether the POSITION
        follows a body — `track`/`trackcom` keep the declared orientation.

        Until 2026-08-03 this was inside out: the quaternion was used only for
        `targetbody`, and fixed/track/trackcom got the invented target
        `(cam_pos_x, 0, 0)` with `up` hardcoded to +Z. For a camera at
        (-3, 0, 1) — humanoid's default `back` camera — that aims straight DOWN
        at (-3, 0, 0), which makes the view direction PARALLEL to the up
        vector. `look_at` then has a zero cross product and the whole view
        matrix degenerates, so the model rendered as nothing at all. That is
        the "humanoid: I can't see anything" report, and point_mass's top-down
        camera hit the same wall.

        Cameras that already looked right are unaffected: cheetah's
        `xyaxes="1 0 0 0 0 1"` at (0,-3,0) resolves to the same +Y look with a
        +Z up it was getting by accident.
        """
        var cameras = List[Camera3D]()
        for i in range(Self.ncam):
            var eye = _RVec3(Self._rcd.cam_pos_x[i], Self._rcd.cam_pos_y[i], Self._rcd.cam_pos_z[i])
            var qx = Self._rcd.cam_quat_x[i]
            var qy = Self._rcd.cam_quat_y[i]
            var qz = Self._rcd.cam_quat_z[i]
            var qw = Self._rcd.cam_quat_w[i]
            var look = Self._rcd_rotate_by_quat(qx, qy, qz, qw, 0.0, 0.0, -1.0)
            var up_v = Self._rcd_rotate_by_quat(qx, qy, qz, qw, 0.0, 1.0, 0.0)
            var target = _RVec3(
                eye.x + look[0], eye.y + look[1], eye.z + look[2]
            )
            cameras.append(
                Camera3D(
                    eye=eye,
                    target=target,
                    up=_RVec3(up_v[0], up_v[1], up_v[2]),
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
        """MJCF `mode` -> the renderer's own encoding.

        Renderer: 0 = TRACKCOM (translate to follow the torso), 1 = FIXED,
        2 = TARGETBODY (re-aim at a body every frame). MJCF: 0 fixed, 1 track,
        2 trackcom, 3 targetbody, 4 targetbodycom.

        ⚠ targetbody USED TO COLLAPSE INTO TRACKCOM, which is a different
        behaviour entirely: trackcom moves the camera and keeps its
        orientation, targetbody holds the camera still and turns it. cartpole's
        `lookatcart` is the one model that asks for it.
        """
        var modes = List[Int]()
        for i in range(Self.ncam):
            var xml_mode = Self._rcd.cam_mode[i]
            if xml_mode == 0:
                modes.append(1)  # fixed
            elif xml_mode == 3 or xml_mode == 4:
                modes.append(2)  # targetbody / targetbodycom
            else:
                modes.append(0)  # track / trackcom
        return modes^

    @staticmethod
    def get_camera_target_bodies() -> List[Int]:
        """Body index each camera aims at, or -1. Parallel to `setup_cameras`."""
        var out = List[Int]()
        for i in range(Self.ncam):
            out.append(Self._rcd.cam_target_body[i])
        return out^

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
    def get_skybox_mark() -> List[Float64]:
        """Return [kind, r, g, b, density] for the skybox's `mark`, else empty.

        Only `mark="random"` (kind 3) means anything to the renderer: MuJoCo
        bakes random dots into the generated skybox texture, which over a dark
        gradient is a starfield. dm_control's `common/skybox.xml` asks for
        exactly that (`mark="random" markrgb="1 1 1"`), and a plain two-colour
        gradient has no way to show it — hence "the stars aren't showing".
        `edge` and `cross` mark 2D textures, not the sky, so they are ignored
        here rather than approximated.
        """
        for i in range(Self.ntex):
            if Self._rcd.tex_type[i] == 1 or Self._rcd.tex_builtin[i] == 1:
                var result = List[Float64]()
                result.append(Float64(Self._rcd.tex_mark[i]))
                result.append(Self._rcd.tex_markrgb_r[i])
                result.append(Self._rcd.tex_markrgb_g[i])
                result.append(Self._rcd.tex_markrgb_b[i])
                result.append(Self._rcd.tex_random[i])
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
    def render_spatial_tendons(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Draw `<spatial>` tendons as a chain of thin capsules.

        This is the only thing that makes ball_in_cup's string visible: the
        tendon is load-bearing in the physics (an inextensible 30 cm link
        between the ball's site and the cup's) but the renderer had no notion
        of tendons at all, so the ball appeared to fly free.

        Sites are stored in the body frame, so each endpoint is the body's
        world pose composed with the site's local offset — the same transform
        `render_body_geoms` applies to a geom.

        ⚠ STRAIGHT SEGMENTS ONLY. MuJoCo routes a spatial tendon around
        `<geom>` wrapping objects; those children are not recorded, so a
        wrapping tendon would be drawn as the chord it wraps around rather
        than the path it takes. ball_in_cup has none, which is why this is
        enough for it and would not be for, say, a pulley.
        """
        var base = 0
        for t in range(Self._rcd.nsten):
            var n = Self._rcd.sten_nsite[t]
            var radius = Self._rcd.sten_width[t]
            var col = Color(
                UInt8(Self._rcd.sten_rgba_r[t] * 255),
                UInt8(Self._rcd.sten_rgba_g[t] * 255),
                UInt8(Self._rcd.sten_rgba_b[t] * 255),
                255,
            )
            for k in range(n - 1):
                var ia = Self._rcd.sten_sites[base + k]
                var ib = Self._rcd.sten_sites[base + k + 1]
                if ia < 0 or ib < 0:
                    continue
                var ba = Self._rcd.site_body_id[ia]
                var bb = Self._rcd.site_body_id[ib]
                if ba >= len(positions) or bb >= len(positions):
                    continue
                var pa = positions[ba] + quaternions[ba].rotate_vec(
                    _RVec3(
                        Self._rcd.site_pos_x[ia],
                        Self._rcd.site_pos_y[ia],
                        Self._rcd.site_pos_z[ia],
                    )
                )
                var pb = positions[bb] + quaternions[bb].rotate_vec(
                    _RVec3(
                        Self._rcd.site_pos_x[ib],
                        Self._rcd.site_pos_y[ib],
                        Self._rcd.site_pos_z[ib],
                    )
                )
                var seg = pb - pa
                var length = seg.length()
                if length < 1e-9:
                    continue
                # `draw_capsule` takes a centre, an orientation and a half
                # height about its local +Z, so the segment becomes a rotation
                # of +Z onto its direction.
                var orient = _RQuat.from_two_vectors(
                    _RVec3(0.0, 0.0, 1.0), seg
                )
                renderer.draw_capsule(
                    center=_RVec3(
                        (pa.x + pb.x) * 0.5,
                        (pa.y + pb.y) * 0.5,
                        (pa.z + pb.z) * 0.5,
                    ),
                    orientation=orient,
                    radius=radius,
                    half_height=length * 0.5,
                    axis=2,
                    color=col,
                )
            base += n

    @staticmethod
    def render_ground_geoms(
        mut renderer: Renderer3D,
        torso_x: Float64,
        follow: Bool,
        visual_radius_scale: Float64,
    ) raises:
        """Draw plane geoms (body_id=0): the ground as a grid, walls as slabs.

        ⚠ NOT EVERY PLANE IS A FLOOR. This used to send every plane geom to
        `draw_ground_grid` at its own `pos_z`, dropping its x/y position and
        its orientation entirely — so manipulator's four planes (floor at z=0,
        two 45° walls at z=.283, and a `background` at z=.5) came out as four
        stacked horizontal grids, the topmost ABOVE the arm's base. Read as a
        picture that is the arm hanging under a ceiling, which is exactly how
        it was reported: "things are below the ground".

        A plane is treated as the ground when it is UNROTATED; anything tilted
        or vertical is drawn as a thin oriented box of its declared
        half-extents. The ground keeps the grid path because that is what
        carries the infinite extent, the texture repeat and the reflection
        pass — a slab cannot stand in for it.

        ⚠ ORIENTATION DECIDES, POSITION MUST NOT. This test also required the
        plane to sit at x=y=0, and that was a REGRESSION (introduced with the
        oriented-plane fix, `8f248290`): the running tracks of cheetah
        (`pos="98 0 0"`), hopper (`pos="48 0 0"`) and walker (`pos="248 0 0"`)
        are unrotated floors that are merely OFFSET, so they fell to the slab
        path and rendered as flat white rectangles — `draw_box` takes a single
        colour and cannot carry the `grid` material's texture. The position
        test bought nothing even for the case it was added for: every wall and
        backdrop in manipulator, stacker, point_mass and reacher is rotated, so
        `upright` alone already excludes all of them. Verified by enumerating
        every plane geom in the suite — exactly those three change class.

        ⚠ This is the RENDER half of a defect whose PHYSICS half is still open:
        the plane narrow phase also assumes every plane is a horizontal floor
        at its origin's z with normal (0,0,1). Fixing the picture does not make
        an inclined plane collide correctly.
        """
        # GEOM_PLANE=0
        var has_plane = False
        var max_body_radius = Float64(0.0)
        for j in range(Self.NGEOM):
            if Self._rcd.geom_body_id[j] > 0 and Self._rcd.geom_radius[j] > max_body_radius:
                max_body_radius = Self._rcd.geom_radius[j]
        for i in range(Self.NGEOM):
            if Self._rcd.geom_type[i] == 0:  # PLANE
                var pqx = Self._rcd.geom_quat_x[i]
                var pqy = Self._rcd.geom_quat_y[i]
                var pqz = Self._rcd.geom_quat_z[i]
                var pqw = Self._rcd.geom_quat_w[i]
                var upright = (
                    pqx == 0.0 and pqy == 0.0 and pqz == 0.0 and pqw == 1.0
                )
                var hx = Self._rcd.geom_half_x[i]
                var hy = Self._rcd.geom_half_y[i]
                if not upright and hx > 0.0 and hy > 0.0:
                    # A wall, a ramp or a backdrop. Half-extents come straight
                    # from MJCF `size="x y spacing"`; a zero there means the
                    # plane is INFINITE along that axis and there is no slab to
                    # draw, so it falls through to the grid path instead.
                    var wall_r = Float32(Self._rcd.geom_rgba_r[i])
                    var wall_g = Float32(Self._rcd.geom_rgba_g[i])
                    var wall_b = Float32(Self._rcd.geom_rgba_b[i])
                    var wall_a = Float32(Self._rcd.geom_rgba_a[i])
                    var wmid = Self._rcd.geom_material_id[i]
                    if wmid >= 0 and wmid < Self.nmat:
                        wall_r = Float32(Self._rcd.mat_rgba_r[wmid])
                        wall_g = Float32(Self._rcd.mat_rgba_g[wmid])
                        wall_b = Float32(Self._rcd.mat_rgba_b[wmid])
                        wall_a = Float32(Self._rcd.mat_rgba_a[wmid])
                    if wall_a >= 0.99:
                        renderer.draw_box(
                            center=_RVec3(
                                Self._rcd.geom_pos_x[i],
                                Self._rcd.geom_pos_y[i],
                                Self._rcd.geom_pos_z[i],
                            ),
                            orientation=_RQuat(pqw, pqx, pqy, pqz),
                            half_extents=_RVec3(hx, hy, 0.002),
                            color=Color(
                                UInt8(wall_r * 255), UInt8(wall_g * 255),
                                UInt8(wall_b * 255), UInt8(wall_a * 255),
                            ),
                        )
                    continue
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
            # ⚠ GROUP IS VISIBILITY, and skipping this check is what drew
            # dm_control's dog as a teal skeleton. MuJoCo's default
            # `mjvOption.geomgroup` is 1 for groups 0-2 and 0 for the rest
            # (`mjv_defaultOption`, engine_vis_init.c:320), and dog parks its
            # collision capsules in group 3 and its 162 bone meshes in group 5.
            # Drawing every group means drawing the collision proxy as if it
            # were the model.
            if Self._rcd.geom_group[i] >= 3:
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
            elif gt == 6:  # ELLIPSOID
                # ⚠ THIS BRANCH DID NOT EXIST until 2026-08-03, and there is no
                # trailing `else`, so every ellipsoid geom was silently skipped
                # — invisible in the renderer while colliding normally. It hit
                # quadruped (the torso), swimmer, fish and finger. Semi-axes
                # live in half_x/y/z, the same slots a box uses.
                renderer.draw_ellipsoid(center=geom_pos, orientation=geom_quat,
                    radii=_RVec3(
                        Self._rcd.geom_half_x[i] * visual_radius_scale,
                        Self._rcd.geom_half_y[i] * visual_radius_scale,
                        Self._rcd.geom_half_z[i] * visual_radius_scale,
                    ),
                    color=geom_color, shininess=shininess, specular=specular,
                    reflectance=reflectance)
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
    def has_skin() -> Bool:
        """Whether the model declares a `<skin>`.

        ⚠ A `find` ON THE XML, NOT A PARSED FLAG. Recording anything about the
        skin in the comptime render data does not compile (see the note above
        `MAX_COMPTIME_RENDER_GEOMS` in `xml_parser.mojo`) — but `find` never
        slices, so asking the question is safe even though storing the answer
        is not. Comptime-resolvable, so a model without a skin still compiles
        `render_skin` away to nothing.
        """
        return Self.xml.find("<skin") != -1

    @staticmethod
    def geom_group_at(i: Int) -> Int:
        """MuJoCo's geom `group` for geom `i` — visibility, not a tag.

        Exposed so a test can count what `render_body_geoms` will skip; see the
        group note there.
        """
        return Self._rcd.geom_group[i]

    @staticmethod
    def body_names() -> List[String]:
        """Body names by index (0 = worldbody), parsed from the model XML AT
        RUNTIME.

        ⚠ RUNTIME ON PURPOSE, and it is not a style choice. A `<skin>`'s bones
        name the bodies they follow and those names live inside the BINARY
        `.skn`, so the match cannot happen at compile time — which leaves the
        names needing to survive into runtime somehow. Writing them into the
        comptime render data does not compile, in every spelling tried (see
        `xml_parser.mojo`, above `MAX_COMPTIME_RENDER_GEOMS`). The XML is
        already a comptime parameter of this struct, so materializing it once
        and scanning it with ORDINARY runtime string code costs nothing extra
        and is subject to none of those restrictions.

        Document order, which is the order the parser assigns body ids: both
        walk `<body` opening tags left to right.

        ⚠ CALL IT ONCE. It materializes the whole model XML and rescans it;
        `render_skin` does so only on the frame that loads the skin.
        """
        var src = String(Self.xml)
        var out = List[String]()
        out.append(String(""))  # index 0 = worldbody, which has no <body> tag

        var wb = src.find("<worldbody")
        if wb == -1:
            return out^
        var stop = src.find("</worldbody>", wb)
        if stop == -1:
            stop = src.byte_length()

        var pos = src.find(">", wb)
        if pos == -1:
            return out^
        pos += 1
        while True:
            var bt = src.find("<body", pos)
            if bt == -1 or bt >= stop:
                break
            var bte = src.find(">", bt)
            if bte == -1:
                break
            out.append(_attr_between(src, bt, bte, "name"))
            pos = bte + 1
        return out^

    @staticmethod
    def render_skin(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Deform and draw the model's `<skin>`, if it has one.

        The skin is the ENVELOPE MuJoCo actually shows for a model like dog —
        the geoms it draws alongside are only those in groups 0-2. See
        `render_body_geoms` for the group rule.
        """
        comptime if Self.xml.find("<skin") == -1:
            return
        else:
            # ⚠ THE WHOLE ASSET CHAIN IS WALKED AT RUNTIME, from the XML this
            # struct already carries. `<skin file= material=>` ->
            # `<material texture=>` -> `<texture file=>` is three attribute
            # reads, and doing any of them in the comptime interpreter is a
            # compile failure the moment it hits. See `body_names`.
            var src = String(Self.xml)
            var st = src.find("<skin")
            if st == -1:
                return
            var se = src.find(">", st)
            if se == -1:
                return

            var skin_file = _attr_between(src, st, se, "file")
            if skin_file.byte_length() == 0:
                return
            var skin_mat = _attr_between(src, st, se, "material")

            var tex_name = String("")
            var tex_file = String("")
            if skin_mat.byte_length() > 0:
                var want_tex = String("")
                var mp = 0
                while True:
                    var mt = src.find("<material", mp)
                    if mt == -1:
                        break
                    var me = src.find(">", mt)
                    if me == -1:
                        break
                    if _attr_between(src, mt, me, "name") == skin_mat:
                        want_tex = _attr_between(src, mt, me, "texture")
                        break
                    mp = me + 1
                if want_tex.byte_length() > 0:
                    var tp = 0
                    while True:
                        var tt = src.find("<texture", tp)
                        if tt == -1:
                            break
                        var te = src.find(">", tt)
                        if te == -1:
                            break
                        if _attr_between(src, tt, te, "name") == want_tex:
                            tex_name = want_tex
                            tex_file = _attr_between(src, tt, te, "file")
                            break
                        tp = te + 1

            # Flatten the poses `skin_pose` wants: plain float arrays with
            # (w, x, y, z) quats, which is MuJoCo's order and `_RQuat`'s.
            var xpos = List[Float32]()
            var xquat = List[Float32]()
            for b in range(len(positions)):
                xpos.append(Float32(positions[b].x))
                xpos.append(Float32(positions[b].y))
                xpos.append(Float32(positions[b].z))
                xquat.append(Float32(quaternions[b].w))
                xquat.append(Float32(quaternions[b].x))
                xquat.append(Float32(quaternions[b].y))
                xquat.append(Float32(quaternions[b].z))

            renderer.draw_skin(
                name=skin_file,
                skn_path=skin_file,
                body_names=Self.body_names(),
                xpos=xpos,
                xquat=xquat,
                texture_name=tex_name,
                texture_path=tex_file,
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
