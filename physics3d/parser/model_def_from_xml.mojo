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

from collections import InlineArray

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from random.philox import Random as PhiloxRandom
from render import Color, Renderer3D, Light, Camera3D
from math3d import Vec3 as _Vec3G, Quat as _QuatG

from physics3d.types import Model, Data, ConeType
from physics3d.joint_types import JNT_FREE, JNT_BALL, JNT_HINGE, JNT_SLIDE
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from physics3d.dynamics.mass_matrix import (
    compute_body_invweight0,
    ldl_factor_gpu,
    compute_mass_matrix_full_gpu,
)
from physics3d.dynamics.jacobian import (
    compute_cdof_gpu,
    compute_composite_inertia_gpu,
)
from physics3d.gpu.constants import (
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
from physics3d.gpu.buffer_utils import (
    copy_model_to_buffer,
    copy_geoms_to_buffer,
    copy_invweight0_to_buffer,
    copy_tendons_to_buffer,
)
from physics3d.model.model_def import ModelDefLike
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
)
from physics3d.model.inertia_from_geom import compute_inertia_from_geoms

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
    cone_type: Int = ConeType.ELLIPTIC,
    max_tendon: Int = 0,
    nsite: Int = 0,
    obs_qpos_skip: Int = 1,
    obs_dim_override: Int = -1,
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
    comptime ACTION_DIM: Int = Self.nact
    comptime TIMESTEP: Float64 = Self.timestep

    # =========================================================================
    # CPU: Model setup
    # =========================================================================

    @staticmethod
    fn setup_model_and_data[
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
        ](Self.xml)
        fmd.setup_model[
            DTYPE,
            Self.MAX_CONTACTS,
            Self.MAX_EQUALITY,
            Self.CONE_TYPE,
            Self.MAX_TENDON,
            Self.NSITE,  # MODEL_NSITE in setup_model's renamed param
        ](model)
        comptime if _xml_compiler_inertiafromgeom[Self.xml]():
            compute_inertia_from_geoms(model)
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
        forward_kinematics(model, data)
        compute_body_invweight0(model, data)

    # =========================================================================
    # CPU: Joints / Actuators delegates
    # =========================================================================

    @staticmethod
    fn reset_data[
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
        """Zero out all qpos, qvel, qacc, qfrc."""
        for i in range(Self.NQ):
            data.qpos[i] = Scalar[DTYPE](0)
        for i in range(Self.NV):
            data.qvel[i] = Scalar[DTYPE](0)
            data.qacc[i] = Scalar[DTYPE](0)
            data.qfrc[i] = Scalar[DTYPE](0)

    @staticmethod
    fn extract_obs[
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
    fn enforce_limits[
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
        ](Self.xml)
        var qpos_adr = 0
        for j in range(Self.NJOINT):
            var jd = fmd.joints[j]
            if jd.is_limited:
                var v = data.qpos[qpos_adr]
                if v < Scalar[DTYPE](jd.range_min):
                    data.qpos[qpos_adr] = Scalar[DTYPE](jd.range_min)
                elif v > Scalar[DTYPE](jd.range_max):
                    data.qpos[qpos_adr] = Scalar[DTYPE](jd.range_max)
            qpos_adr += jd.nq

    @staticmethod
    fn apply_actions[
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
        ](Self.xml)
        for i in range(Self.nact):
            if i >= len(actions):
                break
            var ad = fmd.actuators[i]
            if ad.joint_id < 0:
                continue
            # Compute DOF address for this actuator's joint
            var dof_adr = 0
            for k in range(ad.joint_id):
                dof_adr += fmd.joints[k].nv
            if dof_adr < Self.NV:
                data.qfrc[dof_adr] = Scalar[DTYPE](ad.gear * actions[i])

    # =========================================================================
    # GPU: Model init
    # =========================================================================

    @staticmethod
    fn init_model_gpu[
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

        # Copy to GPU
        ctx.enqueue_copy(model_buf, host_buf.unsafe_ptr())

        # Recompute invweight0 on GPU for accuracy
        Self._compute_invweight0_gpu[DTYPE](ctx, model_buf)

    # =========================================================================
    # GPU: _compute_invweight0_gpu (duplicated from ModelDef, dims from params)
    # =========================================================================

    @staticmethod
    fn _compute_invweight0_gpu[
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
        ]()
        comptime WS_SIZE = integrator_workspace_size[Self.NV, Self.NBODY]()

        var state_buf = ctx.enqueue_create_buffer[DTYPE](STATE_SIZE)
        var ws_buf = ctx.enqueue_create_buffer[DTYPE](WS_SIZE)

        var state = LayoutTensor[
            DTYPE, Layout.row_major(1, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())
        var model = LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())
        var workspace = LayoutTensor[
            DTYPE, Layout.row_major(1, WS_SIZE), MutAnyOrigin
        ](ws_buf.unsafe_ptr())

        @always_inline
        fn invweight0_kernel(
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

    # =========================================================================
    # GPU: Joints / Actuators kernel delegates
    # =========================================================================

    @staticmethod
    fn apply_actions_kernel_gpu[
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
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn apply_kernel(
            states: LayoutTensor[
                DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            comptime qfrc_base = qfrc_offset[Self.NQ, Self.NV]()

            comptime for act_i in range(Self.nact):
                comptime gear = _xml_nth_motor_gear[Self.xml, act_i]()
                comptime dof = _xml_nth_motor_dof_adr[Self.xml, act_i]()

                comptime if dof >= 0 and dof < Self.NV:
                    var ctrl = rebind[Scalar[DTYPE]](actions[env, act_i])
                    if ctrl > Scalar[DTYPE](1.0):
                        ctrl = Scalar[DTYPE](1.0)
                    elif ctrl < Scalar[DTYPE](-1.0):
                        ctrl = Scalar[DTYPE](-1.0)
                    states[env, qfrc_base + dof] = Scalar[DTYPE](gear) * ctrl

        ctx.enqueue_function[apply_kernel, apply_kernel](
            states,
            actions,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn enforce_limits_kernel_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[DTYPE]) raises:
        """GPU kernel: clamp qpos to joint limits for limited joints."""
        var states = LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn limits_kernel(
            states: LayoutTensor[
                DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            comptime qpos_base = qpos_offset[Self.NQ, Self.NV]()

            comptime for j in range(Self.njoint):
                comptime limited = _xml_nth_joint_limited[Self.xml, j]()

                comptime if limited:
                    comptime qp_adr = _xml_nth_joint_qpos_adr[Self.xml, j]()
                    comptime rmin = _xml_nth_joint_range_min[Self.xml, j]()
                    comptime rmax = _xml_nth_joint_range_max[Self.xml, j]()
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
    fn extract_obs_kernel_gpu[
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
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var obs = LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn obs_kernel(
            states: LayoutTensor[
                DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
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
    fn reset_env_gpu[
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

        for i in range(Self.NQ):
            var noise = Scalar[DTYPE](rand_vals[i] * 2.0 - 1.0) * noise_scale
            states[env, qpos_base + i] = noise

        for i in range(Self.NV):
            var noise = (
                Scalar[DTYPE](rand_vals[Self.NQ + i] * 2.0 - 1.0) * noise_scale
            )
            states[env, qvel_base + i] = noise

        for i in range(Self.NV):
            states[env, qacc_base + i] = Scalar[DTYPE](0)
            states[env, qfrc_base + i] = Scalar[DTYPE](0)

    @always_inline
    @staticmethod
    fn extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
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
    fn setup_lights() raises -> List[Light]:
        """Return Light objects parsed from <light> elements in <worldbody>."""
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
        ](Self.xml)
        var lights = List[Light]()
        for i in range(Self.nlight):
            var ld = fmd.lights[i]
            # mode: 1 = directional, 0 = point/spot (render convention)
            var mode = Int(1) if ld.directional else Int(0)
            # ambient: scalar average of ambient RGB channels
            var amb = (ld.ambient_r + ld.ambient_g + ld.ambient_b) / 3.0
            # specular_intensity: average of specular RGB channels
            var spec_int = (ld.specular_r + ld.specular_g + ld.specular_b) / 3.0
            lights.append(
                Light(
                    mode=mode,
                    dir_x=ld.dir_x,
                    dir_y=ld.dir_y,
                    dir_z=ld.dir_z,
                    color_r=ld.diffuse_r,
                    color_g=ld.diffuse_g,
                    color_b=ld.diffuse_b,
                    ambient=amb,
                    specular_intensity=spec_int,
                    specular_exponent=ld.exponent,
                    cast_shadow=ld.castshadow,
                )
            )
        return lights^

    @staticmethod
    fn setup_cameras(width: Int, height: Int) raises -> List[Camera3D]:
        """Return Camera3D objects parsed from <camera> elements in <worldbody>.
        """
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
        ](Self.xml)
        var cameras = List[Camera3D]()
        for i in range(Self.ncam):
            var cd = fmd.cameras[i]
            var eye = _RVec3(cd.pos_x, cd.pos_y, cd.pos_z)
            var target: _RVec3
            if cd.mode == 0 or cd.mode == 1 or cd.mode == 2:
                # CAM_MODE_FIXED=0, CAM_MODE_TRACK=1, CAM_MODE_TRACKCOM=2:
                # Set target at world origin (x=pos_x, y=0, z=0) so that the
                # tracking offset preserved by the renderer is (0, pos_y, pos_z),
                # matching the TrackCamera convention.
                target = _RVec3(cd.pos_x, Float64(0), Float64(0))
            else:
                # For other modes derive look direction from quaternion
                var qx = cd.quat_x
                var qy = cd.quat_y
                var qz = cd.quat_z
                var qw = cd.quat_w
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
                    cd.pos_x + look_x,
                    cd.pos_y + look_y,
                    cd.pos_z + look_z,
                )
            cameras.append(
                Camera3D(
                    eye=eye,
                    target=target,
                    up=_RVec3(0.0, 0.0, 1.0),
                    fov=cd.fovy,
                    aspect=Float64(width) / Float64(height),
                    near=Float64(0.1),
                    far=Float64(100.0),
                    screen_width=width,
                    screen_height=height,
                )
            )
        return cameras^

    @staticmethod
    fn setup_camera_modes() raises -> List[Int]:
        """Return camera modes (CAM_MODE_* constants) for each parsed camera."""
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
        ](Self.xml)
        # Translate XML CAM_MODE_* (flat_model.mojo) to renderer CAM_* constants
        # (camera_spec.mojo): CAM_TRACKCOM=0 for tracking modes, CAM_FIXED=1 for fixed.
        # XML: CAM_MODE_FIXED=0, CAM_MODE_TRACK=1, CAM_MODE_TRACKCOM=2
        # Renderer: CAM_TRACKCOM=0, CAM_FIXED=1
        var modes = List[Int]()
        for i in range(Self.ncam):
            var xml_mode = fmd.cameras[i].mode
            if xml_mode == 0:
                modes.append(1)  # CAM_MODE_FIXED → renderer CAM_FIXED=1
            else:
                modes.append(
                    0
                )  # TRACK / TRACKCOM / TARGET* → renderer CAM_TRACKCOM=0
        return modes^

    @staticmethod
    fn get_skybox_colors() -> List[Float64]:
        """Return [top_r, top_g, top_b, bottom_r, bottom_g, bottom_b] from the
        first skybox/gradient texture, or an empty list if none exists."""
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
        ](Self.xml)
        from .flat_model import TEX_SKYBOX, TEX_BUILTIN_GRADIENT, TEX_2D

        for i in range(Self.ntex):
            var td = fmd.textures[i]
            if td.tex_type == TEX_SKYBOX or td.builtin == TEX_BUILTIN_GRADIENT:
                var result = List[Float64]()
                result.append(td.rgb1_r)
                result.append(td.rgb1_g)
                result.append(td.rgb1_b)
                result.append(td.rgb2_r)
                result.append(td.rgb2_g)
                result.append(td.rgb2_b)
                return result^
        return List[Float64]()

    @staticmethod
    fn get_checker_colors() -> List[Float64]:
        """Return [r, g, b] of the checker texture's secondary (light square) colour,
        or an empty list if no checker texture is found."""
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
        ](Self.xml)
        from .flat_model import TEX_BUILTIN_CHECKER

        for i in range(Self.ntex):
            var td = fmd.textures[i]
            if td.builtin == TEX_BUILTIN_CHECKER:
                var result = List[Float64]()
                result.append(td.rgb2_r)
                result.append(td.rgb2_g)
                result.append(td.rgb2_b)
                return result^
        return List[Float64]()

    @staticmethod
    fn render_ground_geoms(
        mut renderer: Renderer3D,
        torso_x: Float64,
        follow: Bool,
        visual_radius_scale: Float64,
    ) raises:
        """Draw plane geoms (body_id=0) as ground grids; fallback if none."""
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
        ](Self.xml)
        from .flat_model import _GEOM_PLANE

        var has_plane = False
        var max_body_radius = Float64(0.0)
        for j in range(Self.NGEOM):
            var gd = fmd.geoms[j]
            if gd.body_id > 0 and gd.radius > max_body_radius:
                max_body_radius = gd.radius
        for i in range(Self.NGEOM):
            var gd = fmd.geoms[i]
            if gd.geom_type == _GEOM_PLANE:
                has_plane = True
                var ground_offset = gd.pos_z - max_body_radius * (
                    visual_radius_scale - 1.0
                )
                var grid_cx = torso_x if follow else Float64(0.0)
                renderer.draw_ground_grid(grid_cx, height=ground_offset)
        if not has_plane:
            var ground_offset = -max_body_radius * (visual_radius_scale - 1.0)
            var grid_cx = torso_x if follow else Float64(0.0)
            renderer.draw_ground_grid(grid_cx, height=ground_offset)

    @staticmethod
    fn render_body_geoms(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
        visual_radius_scale: Float64,
    ) raises:
        """Draw body-attached geoms (body_id > 0) using parsed geometry + colour.
        """
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
        ](Self.xml)
        from .flat_model import (
            _GEOM_CAPSULE,
            _GEOM_SPHERE,
            _GEOM_BOX,
            _GEOM_CYLINDER,
        )

        for i in range(Self.NGEOM):
            var gd = fmd.geoms[i]
            if gd.body_id <= 0:
                continue  # skip worldbody / static geoms
            if gd.body_id >= len(positions):
                continue
            var body_pos = positions[gd.body_id]
            var body_quat = quaternions[gd.body_id]

            # World-space geom position: body_pos + body_quat.rotate(local_pos)
            var geom_pos: _RVec3
            if gd.pos_x == 0.0 and gd.pos_y == 0.0 and gd.pos_z == 0.0:
                geom_pos = body_pos
            else:
                var local = _RVec3(gd.pos_x, gd.pos_y, gd.pos_z)
                geom_pos = body_pos + body_quat.rotate_vec(local)

            # World-space geom orientation: body_quat * local_quat
            var geom_quat: _RQuat
            if (
                gd.quat_x == 0.0
                and gd.quat_y == 0.0
                and gd.quat_z == 0.0
                and gd.quat_w == 1.0
            ):
                geom_quat = body_quat
            else:
                var local_q = _RQuat(gd.quat_w, gd.quat_x, gd.quat_y, gd.quat_z)
                geom_quat = body_quat * local_q

            # Resolve colour: material rgba > geom rgba > grey default
            var r = Float32(gd.rgba_r)
            var g = Float32(gd.rgba_g)
            var b = Float32(gd.rgba_b)
            var a = Float32(gd.rgba_a)
            var mid = gd.material_id
            if mid >= 0 and mid < Self.nmat:
                var md = fmd.materials[mid]
                r = Float32(md.rgba_r)
                g = Float32(md.rgba_g)
                b = Float32(md.rgba_b)
                a = Float32(md.rgba_a)
            var geom_color = Color(
                UInt8(r * 255), UInt8(g * 255), UInt8(b * 255), UInt8(a * 255)
            )

            # Material shading properties (from material if referenced, else defaults)
            var shininess = Float32(0.5)
            var specular = Float32(0.5)
            var reflectance = Float32(0.0)
            if mid >= 0 and mid < Self.nmat:
                var md = fmd.materials[mid]
                shininess = Float32(md.shininess)
                specular = Float32(md.specular)
                reflectance = Float32(md.reflectance)

            if gd.geom_type == _GEOM_CAPSULE:
                renderer.draw_capsule(
                    center=geom_pos,
                    orientation=geom_quat,
                    radius=gd.radius * visual_radius_scale,
                    half_height=gd.half_length,
                    axis=2,
                    color=geom_color,
                    shininess=shininess,
                    specular=specular,
                    reflectance=reflectance,
                )
            elif gd.geom_type == _GEOM_SPHERE:
                renderer.draw_sphere(
                    center=geom_pos,
                    radius=gd.radius * visual_radius_scale,
                    color=geom_color,
                    shininess=shininess,
                    specular=specular,
                    reflectance=reflectance,
                )
            elif gd.geom_type == _GEOM_BOX:
                renderer.draw_box(
                    center=geom_pos,
                    orientation=geom_quat,
                    half_extents=_RVec3(gd.half_x, gd.half_y, gd.half_z),
                    color=geom_color,
                    shininess=shininess,
                    specular=specular,
                    reflectance=reflectance,
                )
            elif gd.geom_type == _GEOM_CYLINDER:
                renderer.draw_capsule(
                    center=geom_pos,
                    orientation=geom_quat,
                    radius=gd.radius * visual_radius_scale,
                    half_height=gd.half_length,
                    axis=2,
                    color=geom_color,
                    shininess=shininess,
                    specular=specular,
                    reflectance=reflectance,
                )

    @staticmethod
    fn render_sites(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Draw all sites as small bright-green spheres (visual markers).

        Parses site positions from the embedded XML and computes world-space
        position as body_pos + body_quat.rotate(local_pos).
        Uses radius=0.01m and bright green color to distinguish from geoms.
        """
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
        ](Self.xml)

        for i in range(Self.NSITE):
            var sd = fmd.sites[i]
            if sd.body_id <= 0 or sd.body_id >= len(positions):
                continue

            var body_pos = positions[sd.body_id]
            var body_quat = quaternions[sd.body_id]
            var site_world_pos: _RVec3

            if sd.pos_x == 0.0 and sd.pos_y == 0.0 and sd.pos_z == 0.0:
                site_world_pos = body_pos
            else:
                var local_pos = _RVec3(sd.pos_x, sd.pos_y, sd.pos_z)
                site_world_pos = body_pos + body_quat.rotate_vec(local_pos)

            # Use the site's size_0 as radius (XML default: 0.005), minimum 0.01
            var radius = sd.size_0 if sd.size_0 > 0.0 else 0.005
            renderer.draw_sphere(
                center=site_world_pos,
                radius=radius,
                color=Color(0, 255, 0, 255),
                shininess=Float32(0.9),
                specular=Float32(0.9),
                reflectance=Float32(0.0),
            )
