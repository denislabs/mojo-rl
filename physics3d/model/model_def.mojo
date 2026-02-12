"""ModelDef compositor for compile-time model definitions.

Composes Bodies and Joints into a ModelDef with auto-computed dimensions.
Uses Variadic.types + @parameter for to iterate at compile time, following
the same pattern as Sequential[*LAYERS: Model] in deep_rl/model/sequential.mojo.

Note: Bodies and Joints are standalone variadic containers. ModelDef takes
concrete Int parameters because Mojo cannot resolve variadic type packs
through multiple levels of nesting (accessing ModelDef.NQ would fail with
"unbound parameter" if ModelDef contained Bodies/Joints directly).

Usage:
    comptime HalfCheetahBodies = Bodies[Torso, BThigh, ...]
    comptime HalfCheetahJoints = Joints[RootX, RootZ, ...]
    comptime HalfCheetahModel = ModelDef[
        HalfCheetahBodies.N,
        HalfCheetahJoints.N,
        HalfCheetahJoints._sum_nq(),
        HalfCheetahJoints._sum_nv(),
    ]
"""

from collections import InlineArray
from std.builtin.variadics import Variadic
from random.philox import Random as PhiloxRandom

from .body_spec import BodySpec
from .joint_spec import JointSpec
from .geom_spec import GeomSpec
from ..types import Model, Data
from ..joint_types import JNT_HINGE, JNT_SLIDE
from ..constants import GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX
from ..collision.collision_primitives import (
    sphere_sphere,
    capsule_sphere,
    capsule_capsule,
    box_sphere,
    box_capsule,
    box_box,
)

# GPU imports
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from ..gpu.constants import (
    TPB,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    model_wgeom_offset,
    MODEL_WGEOM_SIZE,
    WGEOM_IDX_TYPE,
    WGEOM_IDX_POS_X,
    WGEOM_IDX_POS_Y,
    WGEOM_IDX_POS_Z,
    WGEOM_IDX_QUAT_X,
    WGEOM_IDX_QUAT_Y,
    WGEOM_IDX_QUAT_Z,
    WGEOM_IDX_QUAT_W,
    WGEOM_IDX_SIZE_X,
    WGEOM_IDX_SIZE_Y,
    WGEOM_IDX_SIZE_Z,
    WGEOM_IDX_RADIUS,
    WGEOM_IDX_FRICTION,
    WGEOM_IDX_CONTYPE,
    WGEOM_IDX_CONAFFINITY,
)


# =============================================================================
# WorldBody — variadic list of static worldbody geoms
# =============================================================================


@fieldwise_init
struct WorldBody[*G: GeomSpec]:
    """Compile-time list of static worldbody geom specifications.

    Mirrors MuJoCo <worldbody> element. Contains static geometry (ground plane,
    obstacles) that exists outside the kinematic tree.

    Phase 1: Only PlaneGeom is handled (writes ground_z, friction,
    ground_contype, ground_conaffinity to Model).
    """

    comptime geom_types = Variadic.types[T=GeomSpec, *Self.G]
    comptime N: Int = Variadic.size(Self.geom_types)

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ](mut model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]):
        """Populate model with worldbody geom properties.

        PlaneGeom writes to ground_z, friction, ground_contype, ground_conaffinity.
        Non-plane geoms are handled at compile time via WorldBody.detect_contacts.
        """

        @parameter
        for i in range(Self.N):
            comptime G = Self.geom_types[i]

            @parameter
            if G.GEOM_TYPE == GEOM_PLANE:
                model.ground_z = Scalar[DTYPE](G.POS_Z)
                model.friction = Scalar[DTYPE](G.FRICTION)
                model.ground_contype = G.CONTYPE
                model.ground_conaffinity = G.CONAFFINITY

    @staticmethod
    fn copy_geoms_to_buffer[
        DTYPE: DType,
        NBODY: Int,
        NJOINT: Int,
    ](buffer: HostBuffer[DTYPE]):
        """Copy worldbody geom data to GPU model buffer.

        Writes each non-plane geom's properties to the wgeom section of the
        model buffer. PlaneGeom data is already in the metadata section.

        Args:
            buffer: Host model buffer (must have room for NWGEOM geoms after joints).
        """

        @parameter
        for i in range(Self.N):
            comptime G = Self.geom_types[i]
            var off = model_wgeom_offset[NBODY, NJOINT](i)
            buffer[off + WGEOM_IDX_TYPE] = Scalar[DTYPE](G.GEOM_TYPE)
            buffer[off + WGEOM_IDX_POS_X] = Scalar[DTYPE](G.POS_X)
            buffer[off + WGEOM_IDX_POS_Y] = Scalar[DTYPE](G.POS_Y)
            buffer[off + WGEOM_IDX_POS_Z] = Scalar[DTYPE](G.POS_Z)
            buffer[off + WGEOM_IDX_QUAT_X] = Scalar[DTYPE](G.QUAT_X)
            buffer[off + WGEOM_IDX_QUAT_Y] = Scalar[DTYPE](G.QUAT_Y)
            buffer[off + WGEOM_IDX_QUAT_Z] = Scalar[DTYPE](G.QUAT_Z)
            buffer[off + WGEOM_IDX_QUAT_W] = Scalar[DTYPE](G.QUAT_W)
            buffer[off + WGEOM_IDX_SIZE_X] = Scalar[DTYPE](G.SIZE_X)
            buffer[off + WGEOM_IDX_SIZE_Y] = Scalar[DTYPE](G.SIZE_Y)
            buffer[off + WGEOM_IDX_SIZE_Z] = Scalar[DTYPE](G.SIZE_Z)
            buffer[off + WGEOM_IDX_RADIUS] = Scalar[DTYPE](G.RADIUS)
            buffer[off + WGEOM_IDX_FRICTION] = Scalar[DTYPE](G.FRICTION)
            buffer[off + WGEOM_IDX_CONTYPE] = Scalar[DTYPE](G.CONTYPE)
            buffer[off + WGEOM_IDX_CONAFFINITY] = Scalar[DTYPE](G.CONAFFINITY)

    @staticmethod
    fn detect_contacts[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ](
        model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    ):
        """Detect contacts between bodies and static worldbody geoms.

        Iterates worldbody geoms at compile time (zero runtime overhead for N=0).
        Skips PlaneGeom (handled by detect_ground_contacts).
        Dispatches body×wgeom collision based on geometry types.
        """

        @parameter
        for wg in range(Self.N):
            comptime G_item = Self.geom_types[wg]

            @parameter
            if G_item.GEOM_TYPE != GEOM_PLANE:
                # Static geom properties (all compile-time)
                comptime wg_px = Scalar[DTYPE](G_item.POS_X)
                comptime wg_py = Scalar[DTYPE](G_item.POS_Y)
                comptime wg_pz = Scalar[DTYPE](G_item.POS_Z)
                comptime wg_qx = Scalar[DTYPE](G_item.QUAT_X)
                comptime wg_qy = Scalar[DTYPE](G_item.QUAT_Y)
                comptime wg_qz = Scalar[DTYPE](G_item.QUAT_Z)
                comptime wg_qw = Scalar[DTYPE](G_item.QUAT_W)
                comptime wg_radius = Scalar[DTYPE](G_item.RADIUS)
                comptime wg_half_x = Scalar[DTYPE](G_item.SIZE_X)
                comptime wg_half_y = Scalar[DTYPE](G_item.SIZE_Y)
                comptime wg_half_z = Scalar[DTYPE](G_item.SIZE_Z)
                comptime wg_friction = Scalar[DTYPE](G_item.FRICTION)
                comptime wg_contype = G_item.CONTYPE
                comptime wg_conaffinity = G_item.CONAFFINITY

                for body in range(NBODY):
                    # Contype/conaffinity check
                    if (model.body_contype[body] & wg_conaffinity) == 0 and (
                        wg_contype & model.body_conaffinity[body]
                    ) == 0:
                        continue

                    if data.num_contacts >= MAX_CONTACTS:
                        return

                    var bpx = data.xpos[body * 3 + 0]
                    var bpy = data.xpos[body * 3 + 1]
                    var bpz = data.xpos[body * 3 + 2]
                    var bqx = data.xquat[body * 4 + 0]
                    var bqy = data.xquat[body * 4 + 1]
                    var bqz = data.xquat[body * 4 + 2]
                    var bqw = data.xquat[body * 4 + 3]
                    var b_radius = model.body_radius[body]
                    var b_half_length = model.body_half_length[body]
                    var b_half_x = model.body_half_x[body]
                    var b_half_y = model.body_half_y[body]
                    var b_half_z = model.body_half_z[body]
                    var gi = model.body_geom_type[body]

                    var dist: Scalar[DTYPE] = 1.0
                    var cx: Scalar[DTYPE] = 0
                    var cy: Scalar[DTYPE] = 0
                    var cz: Scalar[DTYPE] = 0
                    var nx: Scalar[DTYPE] = 0
                    var ny: Scalar[DTYPE] = 0
                    var nz: Scalar[DTYPE] = 1

                    # Dispatch: body_geom × wgeom_type
                    @parameter
                    if G_item.GEOM_TYPE == GEOM_SPHERE:
                        if gi == GEOM_SPHERE:
                            var result = sphere_sphere[DTYPE](
                                bpx, bpy, bpz, b_radius,
                                wg_px, wg_py, wg_pz, wg_radius,
                            )
                            dist = result[0]
                            cx = result[1]; cy = result[2]; cz = result[3]
                            nx = result[4]; ny = result[5]; nz = result[6]
                        elif gi == GEOM_CAPSULE:
                            var result = capsule_sphere[DTYPE](
                                bpx, bpy, bpz, bqx, bqy, bqz, bqw,
                                b_half_length, b_radius,
                                wg_px, wg_py, wg_pz, wg_radius,
                            )
                            dist = result[0]
                            cx = result[1]; cy = result[2]; cz = result[3]
                            nx = result[4]; ny = result[5]; nz = result[6]
                        elif gi == GEOM_BOX:
                            var result = box_sphere[DTYPE](
                                bpx, bpy, bpz, bqx, bqy, bqz, bqw,
                                b_half_x, b_half_y, b_half_z,
                                wg_px, wg_py, wg_pz, wg_radius,
                            )
                            dist = result[0]
                            cx = result[1]; cy = result[2]; cz = result[3]
                            nx = result[4]; ny = result[5]; nz = result[6]

                    @parameter
                    if G_item.GEOM_TYPE == GEOM_CAPSULE:
                        if gi == GEOM_SPHERE:
                            var result = capsule_sphere[DTYPE](
                                wg_px, wg_py, wg_pz, wg_qx, wg_qy, wg_qz, wg_qw,
                                wg_half_z, wg_radius,
                                bpx, bpy, bpz, b_radius,
                            )
                            dist = result[0]
                            cx = result[1]; cy = result[2]; cz = result[3]
                            nx = -result[4]; ny = -result[5]; nz = -result[6]
                        elif gi == GEOM_CAPSULE:
                            var result = capsule_capsule[DTYPE](
                                bpx, bpy, bpz, bqx, bqy, bqz, bqw,
                                b_half_length, b_radius,
                                wg_px, wg_py, wg_pz, wg_qx, wg_qy, wg_qz, wg_qw,
                                wg_half_z, wg_radius,
                            )
                            dist = result[0]
                            cx = result[1]; cy = result[2]; cz = result[3]
                            nx = result[4]; ny = result[5]; nz = result[6]
                        elif gi == GEOM_BOX:
                            var result = box_capsule[DTYPE](
                                bpx, bpy, bpz, bqx, bqy, bqz, bqw,
                                b_half_x, b_half_y, b_half_z,
                                wg_px, wg_py, wg_pz, wg_qx, wg_qy, wg_qz, wg_qw,
                                wg_half_z, wg_radius,
                            )
                            dist = result[0]
                            cx = result[1]; cy = result[2]; cz = result[3]
                            nx = result[4]; ny = result[5]; nz = result[6]

                    @parameter
                    if G_item.GEOM_TYPE == GEOM_BOX:
                        if gi == GEOM_SPHERE:
                            var result = box_sphere[DTYPE](
                                wg_px, wg_py, wg_pz, wg_qx, wg_qy, wg_qz, wg_qw,
                                wg_half_x, wg_half_y, wg_half_z,
                                bpx, bpy, bpz, b_radius,
                            )
                            dist = result[0]
                            cx = result[1]; cy = result[2]; cz = result[3]
                            nx = -result[4]; ny = -result[5]; nz = -result[6]
                        elif gi == GEOM_CAPSULE:
                            var result = box_capsule[DTYPE](
                                wg_px, wg_py, wg_pz, wg_qx, wg_qy, wg_qz, wg_qw,
                                wg_half_x, wg_half_y, wg_half_z,
                                bpx, bpy, bpz, bqx, bqy, bqz, bqw,
                                b_half_length, b_radius,
                            )
                            dist = result[0]
                            cx = result[1]; cy = result[2]; cz = result[3]
                            nx = -result[4]; ny = -result[5]; nz = -result[6]
                        elif gi == GEOM_BOX:
                            var result = box_box[DTYPE](
                                bpx, bpy, bpz, bqx, bqy, bqz, bqw,
                                b_half_x, b_half_y, b_half_z,
                                wg_px, wg_py, wg_pz, wg_qx, wg_qy, wg_qz, wg_qw,
                                wg_half_x, wg_half_y, wg_half_z,
                            )
                            dist = result[0]
                            cx = result[1]; cy = result[2]; cz = result[3]
                            nx = result[4]; ny = result[5]; nz = result[6]

                    if dist < Scalar[DTYPE](0) and data.num_contacts < MAX_CONTACTS:
                        var idx = data.num_contacts
                        data.contacts[idx].body_a = body
                        data.contacts[idx].body_b = -1
                        data.contacts[idx].pos_x = cx
                        data.contacts[idx].pos_y = cy
                        data.contacts[idx].pos_z = cz
                        data.contacts[idx].normal_x = nx
                        data.contacts[idx].normal_y = ny
                        data.contacts[idx].normal_z = nz
                        data.contacts[idx].dist = dist
                        data.contacts[idx].friction = wg_friction
                        data.num_contacts += 1


# =============================================================================
# EmptyWorldBody — zero-geom WorldBody for backward compatibility
# =============================================================================


@fieldwise_init
struct EmptyWorldBody:
    """Zero-geom WorldBody placeholder.

    Used as default when environments don't have static worldbody obstacles.
    detect_contacts is a no-op.
    """

    comptime N: Int = 0

    @staticmethod
    fn detect_contacts[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ](
        model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    ):
        """No-op: EmptyWorldBody has no geoms."""
        pass


# =============================================================================
# Bodies — variadic body list
# =============================================================================


@fieldwise_init
struct Bodies[*B: BodySpec]:
    """Compile-time list of body specifications.

    Provides N (body count) and type-level access to each body via body_types[i].
    """

    comptime body_types = Variadic.types[T=BodySpec, *Self.B]
    comptime N: Int = Variadic.size(Self.body_types)

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ](mut model: Model[DTYPE, NQ, NV, Self.N, NJOINT, MAX_CONTACTS]):
        """Populate model body properties from compile-time BodySpec list.

        Iterates over all body specs and sets mass, inertia, geometry, parent,
        local frame, and collision filtering on the model.
        """

        @parameter
        for i in range(Self.N):
            comptime B = Self.body_types[i]

            # Mass, inertia, radius
            model.set_body(
                i,
                mass=Scalar[DTYPE](B.MASS),
                inertia=(
                    Scalar[DTYPE](B.ixx()),
                    Scalar[DTYPE](B.iyy()),
                    Scalar[DTYPE](B.izz()),
                ),
                radius=Scalar[DTYPE](B.RADIUS),
            )

            # Kinematic tree
            model.set_body_parent(i, B.PARENT)

            # Geometry
            model.body_geom_type[i] = B.GEOM_TYPE
            model.body_half_length[i] = Scalar[DTYPE](B.HALF_LENGTH)
            model.body_half_x[i] = Scalar[DTYPE](B.HALF_X)
            model.body_half_y[i] = Scalar[DTYPE](B.HALF_Y)
            model.body_half_z[i] = Scalar[DTYPE](B.HALF_Z)

            # Local frame in parent
            model.set_body_local_frame(
                i,
                pos=(
                    Scalar[DTYPE](B.POS_X),
                    Scalar[DTYPE](B.POS_Y),
                    Scalar[DTYPE](B.POS_Z),
                ),
                quat=(
                    Scalar[DTYPE](B.QUAT_X),
                    Scalar[DTYPE](B.QUAT_Y),
                    Scalar[DTYPE](B.QUAT_Z),
                    Scalar[DTYPE](B.QUAT_W),
                ),
            )

            # Collision filtering
            model.body_contype[i] = B.CONTYPE
            model.body_conaffinity[i] = B.CONAFFINITY


# =============================================================================
# Joints — variadic joint list with sum helpers
# =============================================================================


@fieldwise_init
struct Joints[*J: JointSpec]:
    """Compile-time list of joint specifications.

    Provides N (joint count), sum helpers for total NQ/NV, and offset helpers
    for computing qpos/qvel addresses of each joint.
    """

    comptime joint_types = Variadic.types[T=JointSpec, *Self.J]
    comptime N: Int = Variadic.size(Self.joint_types)

    @staticmethod
    fn _sum_nq() -> Int:
        """Sum NQ across all joints (total qpos dimension)."""
        var total = 0

        @parameter
        for i in range(Self.N):
            total += Self.joint_types[i].NQ
        return total

    @staticmethod
    fn _sum_nv() -> Int:
        """Sum NV across all joints (total qvel dimension)."""
        var total = 0

        @parameter
        for i in range(Self.N):
            total += Self.joint_types[i].NV
        return total

    @staticmethod
    fn _qpos_offset[idx: Int]() -> Int:
        """Compute qpos address for joint idx (sum of NQ for joints 0..idx-1).
        """
        var total = 0

        @parameter
        for j in range(idx):
            total += Self.joint_types[j].NQ
        return total

    @staticmethod
    fn _qvel_offset[idx: Int]() -> Int:
        """Compute qvel/dof address for joint idx (sum of NV for joints 0..idx-1).
        """
        var total = 0

        @parameter
        for j in range(idx):
            total += Self.joint_types[j].NV
        return total

    @staticmethod
    fn reset_data[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](mut data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS]):
        """Reset qpos to initial joint positions (qpos0), zero qvel/qacc/qfrc.

        Sets each joint's qpos to its INIT_QPOS value and zeros all velocity,
        acceleration, and force arrays. Does NOT run forward kinematics.
        """

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]
            comptime offset = Self._qpos_offset[i]()
            data.qpos[offset] = Scalar[DTYPE](J.INIT_QPOS)
        for i in range(NV):
            data.qvel[i] = Scalar[DTYPE](0)
            data.qacc[i] = Scalar[DTYPE](0)
            data.qfrc[i] = Scalar[DTYPE](0)

    # =========================================================================
    # Dimension Helpers (observation / action)
    # =========================================================================

    @staticmethod
    fn _obs_qpos_dim() -> Int:
        """Count of qpos elements included in observation."""
        var total = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QPOS:
                total += J.NQ
        return total

    @staticmethod
    fn _obs_qvel_dim() -> Int:
        """Count of qvel elements included in observation."""
        var total = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QVEL:
                total += J.NV
        return total

    @staticmethod
    fn _obs_dim() -> Int:
        """Total observation dimension (included qpos + included qvel)."""
        return Self._obs_qpos_dim() + Self._obs_qvel_dim()

    @staticmethod
    fn _action_dim() -> Int:
        """Count of actuated DOFs (joints with IS_ACTUATED=True)."""
        var total = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.IS_ACTUATED:
                total += J.NV
        return total

    # =========================================================================
    # CPU Operations
    # =========================================================================

    @staticmethod
    fn extract_obs[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS],
        mut obs: List[Scalar[DTYPE]],
    ):
        """Extract observation from physics data into a list.

        Appends included qpos then included qvel to the obs list.
        """

        # Included qpos
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QPOS:
                comptime offset = Self._qpos_offset[i]()

                @parameter
                for k in range(J.NQ):
                    obs.append(data.qpos[offset + k])

        # Included qvel
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QVEL:
                comptime offset = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NV):
                    obs.append(data.qvel[offset + k])

    @staticmethod
    fn apply_actions[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](
        mut data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS],
        actions: List[Float64],
    ):
        """Apply normalized actions to actuated joints.

        Clamps each action to [-1, 1], scales by TAU_LIMIT, writes to qfrc.
        actions[k] corresponds to the k-th actuated joint in declaration order.
        """
        var act_idx = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.IS_ACTUATED:
                comptime offset = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NV):
                    var a = actions[act_idx] if act_idx < len(actions) else 0.0
                    # Clamp to [-1, 1]
                    if a > 1.0:
                        a = 1.0
                    elif a < -1.0:
                        a = -1.0
                    data.qfrc[offset + k] = Scalar[DTYPE](a * J.TAU_LIMIT)
                    act_idx += 1

    @staticmethod
    fn enforce_limits[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](mut data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS]):
        """Enforce joint position limits. Zeros velocity at limits."""

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.HAS_LIMITS:
                comptime qp_off = Self._qpos_offset[i]()
                comptime qv_off = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NQ):
                    var qpos = data.qpos[qp_off + k]
                    var qvel = data.qvel[qv_off + k]
                    if qpos < Scalar[DTYPE](J.RANGE_MIN):
                        data.qpos[qp_off + k] = Scalar[DTYPE](J.RANGE_MIN)
                        if qvel < Scalar[DTYPE](0):
                            data.qvel[qv_off + k] = Scalar[DTYPE](0)
                    elif qpos > Scalar[DTYPE](J.RANGE_MAX):
                        data.qpos[qp_off + k] = Scalar[DTYPE](J.RANGE_MAX)
                        if qvel > Scalar[DTYPE](0):
                            data.qvel[qv_off + k] = Scalar[DTYPE](0)

    # =========================================================================
    # GPU Operations — inline per-env (called from inside kernels)
    # =========================================================================

    @always_inline
    @staticmethod
    fn extract_obs_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        """Extract observation for a single env on GPU."""
        comptime NQ_VAL = Self._sum_nq()
        comptime NV_VAL = Self._sum_nv()
        comptime QPOS_OFF = qpos_offset[NQ_VAL, NV_VAL]()
        comptime QVEL_OFF = qvel_offset[NQ_VAL, NV_VAL]()

        var obs_idx = 0

        # Included qpos
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QPOS:
                comptime offset = Self._qpos_offset[i]()

                @parameter
                for k in range(J.NQ):
                    obs[env, obs_idx] = states[env, QPOS_OFF + offset + k]
                    obs_idx += 1

        # Included qvel
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QVEL:
                comptime offset = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NV):
                    obs[env, obs_idx] = states[env, QVEL_OFF + offset + k]
                    obs_idx += 1

    @always_inline
    @staticmethod
    fn apply_actions_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        """Apply actions for a single env on GPU."""
        comptime NQ_VAL = Self._sum_nq()
        comptime NV_VAL = Self._sum_nv()
        comptime QFRC_OFF = qfrc_offset[NQ_VAL, NV_VAL]()

        var act_idx = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.IS_ACTUATED:
                comptime offset = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NV):
                    var a = actions[env, act_idx]
                    if a > Scalar[GDTYPE](1.0):
                        a = Scalar[GDTYPE](1.0)
                    elif a < Scalar[GDTYPE](-1.0):
                        a = Scalar[GDTYPE](-1.0)
                    states[env, QFRC_OFF + offset + k] = a * Scalar[GDTYPE](
                        J.TAU_LIMIT
                    )
                    act_idx += 1

    @always_inline
    @staticmethod
    fn enforce_limits_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        """Enforce joint limits for a single env on GPU."""
        comptime NQ_VAL = Self._sum_nq()
        comptime NV_VAL = Self._sum_nv()
        comptime QPOS_OFF = qpos_offset[NQ_VAL, NV_VAL]()
        comptime QVEL_OFF = qvel_offset[NQ_VAL, NV_VAL]()

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.HAS_LIMITS:
                comptime qp_off = Self._qpos_offset[i]()
                comptime qv_off = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NQ):
                    var qpos = states[env, QPOS_OFF + qp_off + k]
                    if qpos < Scalar[GDTYPE](J.RANGE_MIN):
                        states[env, QPOS_OFF + qp_off + k] = Scalar[GDTYPE](
                            J.RANGE_MIN
                        )
                        var qvel = states[env, QVEL_OFF + qv_off + k]
                        if qvel < Scalar[GDTYPE](0):
                            states[env, QVEL_OFF + qv_off + k] = Scalar[GDTYPE](
                                0
                            )
                    elif qpos > Scalar[GDTYPE](J.RANGE_MAX):
                        states[env, QPOS_OFF + qp_off + k] = Scalar[GDTYPE](
                            J.RANGE_MAX
                        )
                        var qvel = states[env, QVEL_OFF + qv_off + k]
                        if qvel > Scalar[GDTYPE](0):
                            states[env, QVEL_OFF + qv_off + k] = Scalar[GDTYPE](
                                0
                            )

    @always_inline
    @staticmethod
    fn reset_env_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        noise_scale: Scalar[GDTYPE],
        seed: Int,
    ):
        """Reset a single env on GPU with random noise.

        Sets qpos = INIT_QPOS + noise, qvel = noise, qacc/qfrc = 0.
        """
        comptime NQ_VAL = Self._sum_nq()
        comptime NV_VAL = Self._sum_nv()
        comptime QPOS_OFF = qpos_offset[NQ_VAL, NV_VAL]()
        comptime QVEL_OFF = qvel_offset[NQ_VAL, NV_VAL]()
        comptime QACC_OFF = qacc_offset[NQ_VAL, NV_VAL]()
        comptime QFRC_OFF = qfrc_offset[NQ_VAL, NV_VAL]()

        # Create RNG with unique seed per environment
        var rng = PhiloxRandom(seed=seed * 2654435761 + env * 12345, offset=0)

        # Generate noise batches (4 values at a time from Philox)
        # We need NQ values for qpos + NV values for qvel
        # Generate enough batches to cover all values
        comptime TOTAL_VALS = NQ_VAL + NV_VAL
        comptime NUM_BATCHES = (TOTAL_VALS + 3) // 4

        var rand_vals = InlineArray[Scalar[DType.float32], NUM_BATCHES * 4](
            fill=Scalar[DType.float32](0)
        )
        for b in range(NUM_BATCHES):
            var batch = rng.step_uniform()
            rand_vals[b * 4 + 0] = batch[0]
            rand_vals[b * 4 + 1] = batch[1]
            rand_vals[b * 4 + 2] = batch[2]
            rand_vals[b * 4 + 3] = batch[3]

        # Reset qpos with noise
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]
            comptime offset = Self._qpos_offset[i]()

            @parameter
            for k in range(J.NQ):
                var noise = (
                    Scalar[GDTYPE](rand_vals[offset + k] * 2.0 - 1.0)
                    * noise_scale
                )
                states[env, QPOS_OFF + offset + k] = (
                    Scalar[GDTYPE](J.INIT_QPOS) + noise
                )

        # Reset qvel with noise
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]
            comptime offset = Self._qvel_offset[i]()

            @parameter
            for k in range(J.NV):
                var noise = (
                    Scalar[GDTYPE](rand_vals[NQ_VAL + offset + k] * 2.0 - 1.0)
                    * noise_scale
                )
                states[env, QVEL_OFF + offset + k] = noise

        # Reset qacc, qfrc to zero
        for i in range(NV_VAL):
            states[env, QACC_OFF + i] = Scalar[GDTYPE](0.0)
            states[env, QFRC_OFF + i] = Scalar[GDTYPE](0.0)

    # =========================================================================
    # GPU Operations — kernel launchers
    # =========================================================================

    @staticmethod
    fn extract_obs_kernel_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[GDTYPE],
        mut obs_buf: DeviceBuffer[GDTYPE],
    ) raises:
        """Launch kernel to extract observations for all envs."""
        var states = LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var obs = LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn kernel(
            states: LayoutTensor[
                GDTYPE,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            obs: LayoutTensor[
                GDTYPE,
                Layout.row_major(BATCH_SIZE, OBS_DIM),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            Self.extract_obs_gpu[GDTYPE, BATCH_SIZE, STATE_SIZE, OBS_DIM](
                states, obs, env
            )

        ctx.enqueue_function[kernel, kernel](
            states,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn apply_actions_kernel_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[GDTYPE],
        actions_buf: DeviceBuffer[GDTYPE],
    ) raises:
        """Launch kernel to apply actions for all envs."""
        var states = LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn kernel(
            states: LayoutTensor[
                GDTYPE,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                GDTYPE,
                Layout.row_major(BATCH_SIZE, ACTION_DIM),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            Self.apply_actions_gpu[GDTYPE, BATCH_SIZE, STATE_SIZE, ACTION_DIM](
                states, actions, env
            )

        ctx.enqueue_function[kernel, kernel](
            states,
            actions,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn enforce_limits_kernel_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[GDTYPE],) raises:
        """Launch kernel to enforce joint limits for all envs."""
        var states = LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn kernel(
            states: LayoutTensor[
                GDTYPE,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            Self.enforce_limits_gpu[GDTYPE, BATCH_SIZE, STATE_SIZE](states, env)

        ctx.enqueue_function[kernel, kernel](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Model Setup
    # =========================================================================

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](mut model: Model[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS]):
        """Populate model joints from compile-time JointSpec list.

        Iterates over all joint specs and calls add_hinge_joint or
        add_slide_joint with correct qpos/qvel offsets.
        """

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.JNT_TYPE == JNT_HINGE:
                _ = model.add_hinge_joint(
                    body_id=J.BODY_IDX,
                    pos=(
                        Scalar[DTYPE](J.POS_X),
                        Scalar[DTYPE](J.POS_Y),
                        Scalar[DTYPE](J.POS_Z),
                    ),
                    axis=(
                        Scalar[DTYPE](J.AXIS_X),
                        Scalar[DTYPE](J.AXIS_Y),
                        Scalar[DTYPE](J.AXIS_Z),
                    ),
                    tau_limit=Scalar[DTYPE](J.TAU_LIMIT),
                    range_min=Scalar[DTYPE](J.RANGE_MIN),
                    range_max=Scalar[DTYPE](J.RANGE_MAX),
                    armature=Scalar[DTYPE](J.ARMATURE),
                    damping=Scalar[DTYPE](J.DAMPING),
                    stiffness=Scalar[DTYPE](J.STIFFNESS),
                    springref=Scalar[DTYPE](J.SPRINGREF),
                    frictionloss=Scalar[DTYPE](J.FRICTIONLOSS),
                )
            elif J.JNT_TYPE == JNT_SLIDE:
                _ = model.add_slide_joint(
                    body_id=J.BODY_IDX,
                    pos=(
                        Scalar[DTYPE](J.POS_X),
                        Scalar[DTYPE](J.POS_Y),
                        Scalar[DTYPE](J.POS_Z),
                    ),
                    axis=(
                        Scalar[DTYPE](J.AXIS_X),
                        Scalar[DTYPE](J.AXIS_Y),
                        Scalar[DTYPE](J.AXIS_Z),
                    ),
                    force_limit=Scalar[DTYPE](J.TAU_LIMIT),
                    range_min=Scalar[DTYPE](J.RANGE_MIN),
                    range_max=Scalar[DTYPE](J.RANGE_MAX),
                    armature=Scalar[DTYPE](J.ARMATURE),
                    damping=Scalar[DTYPE](J.DAMPING),
                    stiffness=Scalar[DTYPE](J.STIFFNESS),
                    springref=Scalar[DTYPE](J.SPRINGREF),
                    frictionloss=Scalar[DTYPE](J.FRICTIONLOSS),
                )


# =============================================================================
# ModelDef — full model compositor (concrete Int parameters)
# =============================================================================


@fieldwise_init
struct ModelDef[nbody: Int, njoint: Int, nq: Int, nv: Int]:
    """Compile-time model definition with pre-computed dimensions.

    Takes concrete Int parameters rather than Bodies/Joints directly,
    because Mojo cannot resolve variadic type packs through nesting.

    Usage:
        comptime MyBodies = Bodies[...]
        comptime MyJoints = Joints[...]
        comptime MyModel = ModelDef[
            MyBodies.N, MyJoints.N,
            MyJoints._sum_nq(), MyJoints._sum_nv(),
        ]
    """

    comptime NBODY: Int = Self.nbody
    comptime NJOINT: Int = Self.njoint
    comptime NQ: Int = Self.nq
    comptime NV: Int = Self.nv
