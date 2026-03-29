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

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.joint_types import (
    JNT_FREE,
    JNT_BALL,
    JNT_HINGE,
    JNT_SLIDE,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from mojo_rl.physics3d.dynamics.mass_matrix import (
    compute_body_invweight0,
    ldl_factor_gpu,
    compute_mass_matrix_full_gpu,
)
from mojo_rl.physics3d.dynamics.jacobian import (
    compute_cdof_gpu,
    compute_composite_inertia_gpu,
)
from mojo_rl.physics3d.gpu.constants import (
    TPB,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    model_size_with_invweight,
    state_size,
    integrator_workspace_size,
    model_metadata_offset,
    model_body_offset,
    model_body_invweight0_offset,
    model_dof_invweight0_offset,
    model_joint_offset,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_ARMATURE,
    ws_cdof_offset,
    ws_M_offset,
    ws_L_offset,
    ws_D_offset,
    BODY_IDX_PARENT,
    xipos_offset,
    xpos_offset,
    xquat_offset,
)
from mojo_rl.physics3d.gpu.buffer_utils import (
    copy_model_to_buffer,
    copy_geoms_to_buffer,
    copy_invweight0_to_buffer,
    copy_tendons_to_buffer,
    copy_mesh_hull_to_buffer,
)
from mojo_rl.physics3d.model.model_def import ModelDefLike
from .full_parser import parse_xml_full
from .xml_parser import (
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
from mojo_rl.physics3d.model.inertia_from_geom import compute_inertia_from_geoms

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
        obs_qpos_skip: Leading qpos DOF to exclude from obs (default 1).
        obs_dim_override: Override OBS_DIM (default -1 = compute from nq-skip+nv).
            Use when custom_extract_obs_gpu produces different dimensionality than
            the default formula (e.g. InvertedDoublePendulum needs OBS_DIM=9 with
            sin/cos transforms despite nq-skip+nv=6).
        timestep:      Simulation timestep (default 0.01).
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
    # CPU: Model setup
    # =========================================================================

    @staticmethod
    def setup_model_and_data[
        DTYPE: DType
    ](
        mut model: Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.CONE_TYPE,
            Self.MAX_TENDON,
            Self.NSITE,
        ],
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NSITE,
        ],
    ):
        """Parse XML, populate model struct, run FK and compute invweight0."""
        comptime assert (
            DTYPE.is_floating_point()
        ), "DTYPE must be a floating point type"
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
        ](Self.xml)
        fmd.setup_model[
            DTYPE,
            Self.MAX_CONTACTS,
            Self.MAX_EQUALITY,
            Self.CONE_TYPE,
            Self.MAX_TENDON,
            Self.NSITE,  # MODEL_NSITE in setup_model's renamed param
        ](model)
        comptime ifg_mode = _xml_compiler_inertiafromgeom[Self.xml]()
        comptime igr = _xml_compiler_inertiagrouprange[Self.xml]()
        comptime if ifg_mode == 1:
            compute_inertia_from_geoms[
                INERTIA_GROUP_MIN=igr[0], INERTIA_GROUP_MAX=igr[1],
            ](model)
        comptime if ifg_mode == 2:
            compute_inertia_from_geoms[
                INERTIA_GROUP_MIN=igr[0], INERTIA_GROUP_MAX=igr[1],
                AUTO_MODE=True,
            ](model)
        comptime if ifg_mode > 0:
            comptime settotalmass = _xml_compiler_settotalmass[Self.xml]()
            comptime if settotalmass > 0.0:
                var total_mass = Scalar[DTYPE](0)
                for i in range(1, Self.NBODY):
                    total_mass += model.body_mass[i]
                if total_mass > Scalar[DTYPE](0):
                    var scale = Scalar[DTYPE](settotalmass) / total_mass
                    for i in range(1, Self.NBODY):
                        model.body_mass[i] *= scale
                        model.body_inv_mass[i] = (
                            Scalar[DTYPE](1.0) / model.body_mass[i]
                        )
                        for k in range(3):
                            model.body_inertia[i * 3 + k] *= scale
                            model.body_inv_inertia[i * 3 + k] = (
                                Scalar[DTYPE](1.0)
                                / model.body_inertia[i * 3 + k]
                            )
        # Parse and add fixed tendons from XML <tendon><fixed> section.
        # NOTE: fixed tendons only define a kinematic quantity (length = Σ coef*qpos).
        # They produce NO forces unless referenced by <equality><tendon> or they have
        # limited="true"/stiffness/damping attributes. Since we only support
        # equality constraints on tendons, skip parsing when no equality block exists.
        # TODO: parse <equality><tendon> and only constrain those tendons.
        comptime if False and Self.MAX_TENDON > 0:
            comptime for t_idx in range(Self.MAX_TENDON):
                comptime nj = _xml_fixed_tendon_njoints[Self.xml, t_idx]()
                comptime if nj > 0:
                    var jidx = InlineArray[Int, 4](fill=-1)
                    var jcoefs = InlineArray[Scalar[DTYPE], 4](
                        fill=Scalar[DTYPE](0)
                    )
                    comptime for ji in range(nj):
                        comptime jn = _xml_fixed_tendon_joint_name[
                            Self.xml, t_idx, ji
                        ]()
                        comptime jnt_idx = _xml_find_joint_index(
                            Self.xml, jn
                        )
                        comptime cf = _xml_fixed_tendon_coef[
                            Self.xml, t_idx, ji
                        ]()
                        jidx[ji] = jnt_idx
                        jcoefs[ji] = Scalar[DTYPE](cf)
                    _ = model.add_tendon(nj, jidx, jcoefs)

        # Initialize data.qpos from qpos0 (joint ref values) before FK
        Self.reset_data(data)
        forward_kinematics(model, data)
        compute_body_invweight0(model, data)

    # =========================================================================
    # CPU: Joints / Actuators delegates
    # =========================================================================

    @staticmethod
    def reset_data[
        DTYPE: DType
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NSITE,
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
                    data.qpos[i] = Scalar[DTYPE](val)
                else:
                    data.qpos[i] = Scalar[DTYPE](0)
        else:
            # No init_qpos — zero everything, then fix free-joint quaternion.
            for i in range(Self.NQ):
                data.qpos[i] = Scalar[DTYPE](0)
            comptime if Self._acd.free_joint_qpos_adr >= 0:
                # qpos[adr+3] is qw for a free joint (MuJoCo convention:
                # [tx, ty, tz, qw, qx, qy, qz]).  Set qw=1 for identity.
                data.qpos[Self._acd.free_joint_qpos_adr + 3] = Scalar[DTYPE](1)
        for i in range(Self.NV):
            data.qvel[i] = Scalar[DTYPE](0)
            data.qacc[i] = Scalar[DTYPE](0)
            data.qfrc[i] = Scalar[DTYPE](0)

    @staticmethod
    def extract_obs[
        DTYPE: DType
    ](
        data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NSITE,
        ],
        mut obs: List[Scalar[DTYPE]],
    ):
        """Extract observation: qpos[obs_qpos_skip:] followed by qvel[:]."""
        for i in range(Self.NQ - Self.obs_qpos_skip):
            obs.append(data.qpos[Self.obs_qpos_skip + i])
        for i in range(Self.NV):
            obs.append(data.qvel[i])

    @staticmethod
    def enforce_limits[
        DTYPE: DType
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NSITE,
        ],
    ):
        """Clamp qpos to joint range limits (limited joints only)."""
        for j in range(Self.NJOINT):
            if Self._acd.joint_is_limited[j]:
                var qp_adr = Self._acd.joint_qpos_adr[j]
                var v = data.qpos[qp_adr]
                if v < Scalar[DTYPE](Self._acd.joint_range_min[j]):
                    data.qpos[qp_adr] = Scalar[DTYPE](Self._acd.joint_range_min[j])
                elif v > Scalar[DTYPE](Self._acd.joint_range_max[j]):
                    data.qpos[qp_adr] = Scalar[DTYPE](Self._acd.joint_range_max[j])

    @staticmethod
    def apply_actions[
        DTYPE: DType
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NSITE,
        ],
        actions: List[Float64],
    ):
        """Apply actuator forces to qfrc (gear * action for each motor)."""
        for i in range(Self.nact):
            if i >= len(actions):
                break
            var dof_adr = Self._acd.motor_dof_adr[i]
            if dof_adr < 0 or dof_adr >= Self.NV:
                continue
            data.qfrc[dof_adr] = Scalar[DTYPE](
                Self._acd.motor_gears[i] * actions[i]
            )

    # =========================================================================
    # GPU: Model init
    # =========================================================================

    @staticmethod
    def init_model_gpu[
        DTYPE: DType
    ](ctx: DeviceContext, mut model_buf: DeviceBuffer[DTYPE],) raises:
        """Serialize CPU model to GPU buffer, then compute invweight0 on GPU.

        Creates a Model + Data on CPU, runs setup_model_and_data,
        serializes to HostBuffer, copies to DeviceBuffer, then calls
        _compute_invweight0_gpu to compute accurate invweight0 on GPU.
        """
        comptime BUF_SIZE = model_size_with_invweight[
            Self.NBODY,
            Self.NJOINT,
            Self.NV,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.MAX_TENDON,
            Self.NSITE,
            Self.nexclude,
        ]()
        var host_buf = ctx.enqueue_create_host_buffer[DTYPE](BUF_SIZE)
        for i in range(BUF_SIZE):
            host_buf[i] = Scalar[DTYPE](0)

        # Create CPU model + data, populate from XML
        var model = Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.CONE_TYPE,
            Self.MAX_TENDON,
            Self.NSITE,
        ]()
        var data = Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NSITE,
        ]()
        Self.setup_model_and_data[DTYPE](model, data)

        # Serialize model to host buffer
        copy_model_to_buffer(model, host_buf)
        copy_geoms_to_buffer(model, host_buf)
        copy_tendons_to_buffer(model, host_buf)
        copy_invweight0_to_buffer(model, host_buf)
        copy_mesh_hull_to_buffer(model, host_buf)

        # Copy to GPU
        ctx.enqueue_copy(model_buf, host_buf)

        # Recompute invweight0 on GPU for accuracy
        Self._compute_invweight0_gpu[DTYPE](ctx, model_buf)

    # =========================================================================
    # GPU: _compute_invweight0_gpu (duplicated from ModelDef, dims from params)
    # =========================================================================

    @staticmethod
    def _compute_invweight0_gpu[
        DTYPE: DType,
    ](ctx: DeviceContext, mut model_buf: DeviceBuffer[DTYPE]) raises:
        """Compute invweight0 on GPU via a single-thread kernel.

        Identical algorithm to ModelDef._compute_invweight0_gpu,
        parameterized on struct dimensions via Self.NQ, Self.NV, etc.
        """
        comptime STATE_SIZE = state_size[Self.NQ, Self.NV, Self.NBODY, 1]()
        comptime MODEL_SIZE = model_size_with_invweight[
            Self.NBODY,
            Self.NJOINT,
            Self.NV,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.MAX_TENDON,
            Self.NSITE,
            Self.nexclude,
        ]()
        comptime WS_SIZE = integrator_workspace_size[Self.NV, Self.NBODY]()

        var state_buf = ctx.enqueue_create_buffer[DTYPE](STATE_SIZE)
        var ws_buf = ctx.enqueue_create_buffer[DTYPE](WS_SIZE)

        var state = LayoutTensor[
            DTYPE, Layout.row_major(1, STATE_SIZE), MutAnyOrigin
        ](state_buf)
        var model = LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf)
        var workspace = LayoutTensor[
            DTYPE, Layout.row_major(1, WS_SIZE), MutAnyOrigin
        ](ws_buf)

        @always_inline
        def invweight0_kernel(
            state: LayoutTensor[
                DTYPE, Layout.row_major(1, STATE_SIZE), MutAnyOrigin
            ],
            model: LayoutTensor[
                DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            workspace: LayoutTensor[
                DTYPE, Layout.row_major(1, WS_SIZE), MutAnyOrigin
            ],
        ):
            for i in range(STATE_SIZE):
                state[0, i] = Scalar[DTYPE](0)

            forward_kinematics_gpu[
                DTYPE,
                Self.NQ,
                Self.NV,
                Self.NBODY,
                Self.NJOINT,
                1,
                STATE_SIZE,
                MODEL_SIZE,
                1,
            ](0, state, model)

            compute_cdof_gpu[
                DTYPE,
                Self.NQ,
                Self.NV,
                Self.NBODY,
                Self.NJOINT,
                1,
                STATE_SIZE,
                MODEL_SIZE,
                1,
                WS_SIZE,
            ](0, state, model, workspace)

            compute_composite_inertia_gpu[
                DTYPE,
                Self.NQ,
                Self.NV,
                Self.NBODY,
                Self.NJOINT,
                1,
                STATE_SIZE,
                MODEL_SIZE,
                1,
                WS_SIZE,
            ](0, state, model, workspace)

            compute_mass_matrix_full_gpu[
                DTYPE,
                Self.NQ,
                Self.NV,
                Self.NBODY,
                Self.NJOINT,
                1,
                STATE_SIZE,
                MODEL_SIZE,
                1,
                WS_SIZE,
            ](0, state, model, workspace)

            # Add armature to M diagonal
            comptime M_idx = ws_M_offset[Self.NV, Self.NBODY]()
            var meta_off = model_metadata_offset[Self.NBODY, Self.NJOINT]()
            var num_joints = Int(
                rebind[Scalar[DTYPE]](
                    model[0, meta_off + MODEL_META_IDX_NJOINT]
                )
            )
            for j in range(num_joints):
                var joff = model_joint_offset[Self.NBODY](j)
                var jtype = Int(
                    rebind[Scalar[DTYPE]](model[0, joff + JOINT_IDX_TYPE])
                )
                var dof_adr = Int(
                    rebind[Scalar[DTYPE]](model[0, joff + JOINT_IDX_DOF_ADR])
                )
                var arm = rebind[Scalar[DTYPE]](
                    model[0, joff + JOINT_IDX_ARMATURE]
                )
                var ndof = 1
                if jtype == JNT_FREE:
                    ndof = 6
                elif jtype == JNT_BALL:
                    ndof = 3
                for d in range(ndof):
                    var idx = M_idx + (dof_adr + d) * Self.NV + (dof_adr + d)
                    workspace[0, idx] = (
                        rebind[Scalar[DTYPE]](workspace[0, idx]) + arm
                    )

            ldl_factor_gpu[DTYPE, Self.NV, Self.NBODY, 1, WS_SIZE](0, workspace)

            # Compute invweight0
            comptime cdof_idx = ws_cdof_offset()
            comptime L_idx = ws_L_offset[Self.NV, Self.NBODY]()
            comptime D_idx = ws_D_offset[Self.NV, Self.NBODY]()
            comptime scratch1 = D_idx + Self.NV
            comptime scratch2 = scratch1 + Self.NV
            comptime scratch3 = scratch2 + Self.NV
            var xi_off = xipos_offset[Self.NQ, Self.NV, Self.NBODY]()

            var bw_off = model_body_invweight0_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NGEOM,
                Self.MAX_EQUALITY,
                Self.MAX_TENDON,
                Self.NSITE,
            ]()
            var dw_off = model_dof_invweight0_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NGEOM,
                Self.MAX_EQUALITY,
                Self.MAX_TENDON,
                Self.NSITE,
            ]()

            model[0, bw_off + 0] = Scalar[DTYPE](0)
            model[0, bw_off + 1] = Scalar[DTYPE](0)

            for i in range(Self.NBODY):
                var ti_x = rebind[Scalar[DTYPE]](state[0, xi_off + i * 3 + 0])
                var ti_y = rebind[Scalar[DTYPE]](state[0, xi_off + i * 3 + 1])
                var ti_z = rebind[Scalar[DTYPE]](state[0, xi_off + i * 3 + 2])

                var A_diag_tran = Scalar[DTYPE](0)
                var A_diag_rot = Scalar[DTYPE](0)

                for k in range(6):
                    var dot_val = Scalar[DTYPE](0)

                    for d in range(Self.NV):
                        workspace[0, scratch1 + d] = Scalar[DTYPE](0)

                    for d in range(Self.NV):
                        var dof_body = 0
                        for jj in range(num_joints):
                            var jj_off = model_joint_offset[Self.NBODY](jj)
                            var jj_type = Int(
                                rebind[Scalar[DTYPE]](
                                    model[0, jj_off + JOINT_IDX_TYPE]
                                )
                            )
                            var jj_dof = Int(
                                rebind[Scalar[DTYPE]](
                                    model[0, jj_off + JOINT_IDX_DOF_ADR]
                                )
                            )
                            var jj_body = Int(
                                rebind[Scalar[DTYPE]](
                                    model[0, jj_off + JOINT_IDX_BODY_ID]
                                )
                            )
                            var jj_ndof = 1
                            if jj_type == JNT_FREE:
                                jj_ndof = 6
                            elif jj_type == JNT_BALL:
                                jj_ndof = 3
                            if d >= jj_dof and d < jj_dof + jj_ndof:
                                dof_body = jj_body
                                break

                        var affects = False
                        if dof_body == i:
                            affects = True
                        else:
                            var current = i
                            while current > 0:
                                var p_off = model_body_offset(current)
                                var parent = Int(
                                    rebind[Scalar[DTYPE]](
                                        model[0, p_off + BODY_IDX_PARENT]
                                    )
                                )
                                if parent == dof_body:
                                    affects = True
                                    break
                                current = parent

                        if not affects:
                            continue

                        var ang_x = rebind[Scalar[DTYPE]](
                            workspace[0, cdof_idx + d * 6 + 0]
                        )
                        var ang_y = rebind[Scalar[DTYPE]](
                            workspace[0, cdof_idx + d * 6 + 1]
                        )
                        var ang_z = rebind[Scalar[DTYPE]](
                            workspace[0, cdof_idx + d * 6 + 2]
                        )
                        var lin_x = rebind[Scalar[DTYPE]](
                            workspace[0, cdof_idx + d * 6 + 3]
                        )
                        var lin_y = rebind[Scalar[DTYPE]](
                            workspace[0, cdof_idx + d * 6 + 4]
                        )
                        var lin_z = rebind[Scalar[DTYPE]](
                            workspace[0, cdof_idx + d * 6 + 5]
                        )

                        var dx = ti_x - rebind[Scalar[DTYPE]](
                            state[0, xi_off + dof_body * 3 + 0]
                        )
                        var dy = ti_y - rebind[Scalar[DTYPE]](
                            state[0, xi_off + dof_body * 3 + 1]
                        )
                        var dz = ti_z - rebind[Scalar[DTYPE]](
                            state[0, xi_off + dof_body * 3 + 2]
                        )

                        var val: Scalar[DTYPE]
                        if k == 0:
                            val = lin_x + ang_y * dz - ang_z * dy
                        elif k == 1:
                            val = lin_y + ang_z * dx - ang_x * dz
                        elif k == 2:
                            val = lin_z + ang_x * dy - ang_y * dx
                        elif k == 3:
                            val = ang_x
                        elif k == 4:
                            val = ang_y
                        else:
                            val = ang_z
                        workspace[0, scratch1 + d] = val

                    # LDL forward substitution: y = L^{-1} * b
                    for ii in range(Self.NV):
                        var s = rebind[Scalar[DTYPE]](
                            workspace[0, scratch1 + ii]
                        )
                        for jj in range(ii):
                            s = s - rebind[Scalar[DTYPE]](
                                workspace[0, L_idx + ii * Self.NV + jj]
                            ) * rebind[Scalar[DTYPE]](
                                workspace[0, scratch2 + jj]
                            )
                        workspace[0, scratch2 + ii] = s

                    # Scale: z = D^{-1} * y
                    for ii in range(Self.NV):
                        var d_val = rebind[Scalar[DTYPE]](
                            workspace[0, D_idx + ii]
                        )
                        if d_val > Scalar[DTYPE](1e-14) or d_val < Scalar[
                            DTYPE
                        ](-1e-14):
                            workspace[0, scratch3 + ii] = (
                                rebind[Scalar[DTYPE]](
                                    workspace[0, scratch2 + ii]
                                )
                                / d_val
                            )
                        else:
                            workspace[0, scratch3 + ii] = Scalar[DTYPE](0)

                    # Back substitution: x = L^{-T} * z
                    for ii_rev in range(Self.NV):
                        var ii = Self.NV - 1 - ii_rev
                        var s = rebind[Scalar[DTYPE]](
                            workspace[0, scratch3 + ii]
                        )
                        for jj in range(ii + 1, Self.NV):
                            s = s - rebind[Scalar[DTYPE]](
                                workspace[0, L_idx + jj * Self.NV + ii]
                            ) * rebind[Scalar[DTYPE]](
                                workspace[0, scratch2 + jj]
                            )
                        workspace[0, scratch2 + ii] = s

                    # dot(J_row, x)
                    for d in range(Self.NV):
                        dot_val += rebind[Scalar[DTYPE]](
                            workspace[0, scratch1 + d]
                        ) * rebind[Scalar[DTYPE]](workspace[0, scratch2 + d])

                    if k < 3:
                        A_diag_tran += dot_val
                    else:
                        A_diag_rot += dot_val

                var tran = A_diag_tran / Scalar[DTYPE](3)
                var rot = A_diag_rot / Scalar[DTYPE](3)

                if tran < Scalar[DTYPE](1e-10) and rot > Scalar[DTYPE](1e-10):
                    tran = rot
                elif rot < Scalar[DTYPE](1e-10) and tran > Scalar[DTYPE](1e-10):
                    rot = tran

                model[0, bw_off + 2 * i] = tran
                model[0, bw_off + 2 * i + 1] = rot

            # Compute dof_invweight0: diagonal of M^{-1}
            for d in range(Self.NV):
                for ii in range(Self.NV):
                    workspace[0, scratch1 + ii] = Scalar[DTYPE](0)
                workspace[0, scratch1 + d] = Scalar[DTYPE](1)

                for ii in range(Self.NV):
                    var s = rebind[Scalar[DTYPE]](workspace[0, scratch1 + ii])
                    for jj in range(ii):
                        s = s - rebind[Scalar[DTYPE]](
                            workspace[0, L_idx + ii * Self.NV + jj]
                        ) * rebind[Scalar[DTYPE]](workspace[0, scratch2 + jj])
                    workspace[0, scratch2 + ii] = s

                for ii in range(Self.NV):
                    var d_val = rebind[Scalar[DTYPE]](workspace[0, D_idx + ii])
                    if d_val > Scalar[DTYPE](1e-14) or d_val < Scalar[DTYPE](
                        -1e-14
                    ):
                        workspace[0, scratch3 + ii] = (
                            rebind[Scalar[DTYPE]](workspace[0, scratch2 + ii])
                            / d_val
                        )
                    else:
                        workspace[0, scratch3 + ii] = Scalar[DTYPE](0)

                for ii_rev in range(Self.NV):
                    var ii = Self.NV - 1 - ii_rev
                    var s = rebind[Scalar[DTYPE]](workspace[0, scratch3 + ii])
                    for jj in range(ii + 1, Self.NV):
                        s = s - rebind[Scalar[DTYPE]](
                            workspace[0, L_idx + jj * Self.NV + ii]
                        ) * rebind[Scalar[DTYPE]](workspace[0, scratch2 + jj])
                    workspace[0, scratch2 + ii] = s

                model[0, dw_off + d] = rebind[Scalar[DTYPE]](
                    workspace[0, scratch2 + d]
                )

        ctx.enqueue_function[invweight0_kernel, invweight0_kernel](
            state,
            model,
            workspace,
            grid_dim=(1,),
            block_dim=(1,),
        )
        ctx.synchronize()

        # Keep buffers alive until kernel completes (prevent ASAP destruction)
        _ = state_buf^
        _ = ws_buf^

    # =========================================================================
    # GPU: Joints / Actuators kernel delegates
    # =========================================================================

    @staticmethod
    def apply_actions_kernel_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[DTYPE],
        actions_buf: DeviceBuffer[DTYPE],
    ) raises:
        """GPU kernel: apply gear * action to qfrc for each actuator.

        Uses comptime helpers to extract per-actuator gear and DOF address
        from the embedded XML at compile time.
        """
        var states = LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf)
        var actions = LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), ImmutAnyOrigin
        ](actions_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def apply_kernel(
            states: LayoutTensor[
                DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), ImmutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            comptime qfrc_base = qfrc_offset[Self.NQ, Self.NV]()

            comptime for act_i in range(Self.nact):
                comptime gear = Self._acd.motor_gears[act_i]
                comptime dof = Self._acd.motor_dof_adr[act_i]

                comptime if dof >= 0 and dof < Self.NV:
                    var ctrl = rebind[Scalar[DTYPE]](actions[env, act_i])
                    if ctrl > Scalar[DTYPE](Self.CTRL_MAX):
                        ctrl = Scalar[DTYPE](Self.CTRL_MAX)
                    elif ctrl < Scalar[DTYPE](Self.CTRL_MIN):
                        ctrl = Scalar[DTYPE](Self.CTRL_MIN)
                    states[env, qfrc_base + dof] = Scalar[DTYPE](gear) * ctrl

        ctx.enqueue_function[apply_kernel, apply_kernel](
            states,
            actions,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def enforce_limits_kernel_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[DTYPE]) raises:
        """GPU kernel: clamp qpos to joint limits for limited joints."""
        var states = LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def limits_kernel(
            states: LayoutTensor[
                DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            comptime qpos_base = qpos_offset[Self.NQ, Self.NV]()

            comptime for j in range(Self.njoint):
                comptime limited = Self._acd.joint_is_limited[j]

                comptime if limited:
                    comptime qp_adr = Self._acd.joint_qpos_adr[j]
                    comptime rmin = Self._acd.joint_range_min[j]
                    comptime rmax = Self._acd.joint_range_max[j]
                    var qpos = rebind[Scalar[DTYPE]](
                        states[env, qpos_base + qp_adr]
                    )
                    if qpos < Scalar[DTYPE](rmin):
                        states[env, qpos_base + qp_adr] = Scalar[DTYPE](rmin)
                    elif qpos > Scalar[DTYPE](rmax):
                        states[env, qpos_base + qp_adr] = Scalar[DTYPE](rmax)

        ctx.enqueue_function[limits_kernel, limits_kernel](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def extract_obs_kernel_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[DTYPE],
        mut obs_buf: DeviceBuffer[DTYPE],
    ) raises:
        """GPU kernel: extract qpos[obs_qpos_skip:] + qvel[:] as observation."""
        var states = LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), ImmutAnyOrigin
        ](states_buf)
        var obs = LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def obs_kernel(
            states: LayoutTensor[
                DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), ImmutAnyOrigin
            ],
            obs: LayoutTensor[
                DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            Self.extract_obs_gpu[DTYPE, BATCH_SIZE, STATE_SIZE, OBS_DIM](
                states, obs, env
            )

        ctx.enqueue_function[obs_kernel, obs_kernel](
            states,
            obs,
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
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        noise_scale: Scalar[DTYPE],
        seed: Int,
    ):
        """Reset a single env with small random noise around qpos=0, qvel=0."""
        comptime qpos_base = qpos_offset[Self.NQ, Self.NV]()
        comptime qvel_base = qvel_offset[Self.NQ, Self.NV]()
        comptime qacc_base = qacc_offset[Self.NQ, Self.NV]()
        comptime qfrc_base = qfrc_offset[Self.NQ, Self.NV]()
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
                states[env, qpos_base + i] = Scalar[DTYPE](val) + noise
            else:
                comptime if (
                    Self._acd.free_joint_qpos_adr >= 0
                    and i == Self._acd.free_joint_qpos_adr + 3
                ):
                    # Free-joint qw: start from identity (1.0) + small noise.
                    states[env, qpos_base + i] = Scalar[DTYPE](1) + noise
                else:
                    states[env, qpos_base + i] = noise

        comptime for i in range(Self.NV):
            var noise = (
                Scalar[DTYPE](rand_vals[Self.NQ + i] * 2.0 - 1.0) * noise_scale
            )
            states[env, qvel_base + i] = noise

        for i in range(Self.NV):
            states[env, qacc_base + i] = Scalar[DTYPE](0)
            states[env, qfrc_base + i] = Scalar[DTYPE](0)

    @always_inline
    @staticmethod
    def extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), ImmutAnyOrigin
        ],
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        """Extract obs = qpos[obs_qpos_skip:] + qvel[:] for a single env."""
        comptime qpos_base = qpos_offset[Self.NQ, Self.NV]()
        comptime qvel_base = qvel_offset[Self.NQ, Self.NV]()

        comptime for i in range(Self.NQ - Self.obs_qpos_skip):
            obs[env, i] = states[env, qpos_base + Self.obs_qpos_skip + i]

        comptime for i in range(Self.NV):
            obs[env, Self.NQ - Self.obs_qpos_skip + i] = states[
                env, qvel_base + i
            ]

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
                renderer.draw_ground_grid(grid_cx, height=ground_offset)
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
            var gt = Self._rcd.geom_type[i]
            if gt == 2:  # CAPSULE
                renderer.draw_capsule(center=geom_pos, orientation=geom_quat,
                    radius=Self._rcd.geom_radius[i] * visual_radius_scale,
                    half_height=Self._rcd.geom_half_length[i], axis=2,
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance)
            elif gt == 1:  # SPHERE
                renderer.draw_sphere(center=geom_pos,
                    radius=Self._rcd.geom_radius[i] * visual_radius_scale,
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance)
            elif gt == 3:  # BOX
                renderer.draw_box(center=geom_pos, orientation=geom_quat,
                    half_extents=_RVec3(Self._rcd.geom_half_x[i], Self._rcd.geom_half_y[i], Self._rcd.geom_half_z[i]),
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance)
            elif gt == 4:  # CYLINDER
                renderer.draw_cylinder(center=geom_pos, orientation=geom_quat,
                    radius=Self._rcd.geom_radius[i] * visual_radius_scale,
                    half_height=Self._rcd.geom_half_length[i], axis=2,
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance)
            elif gt == 5:  # MESH
                var mid2 = Self._rcd.geom_mesh_id[i]
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
