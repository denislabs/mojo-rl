"""ModelDef compositor for compile-time model definitions.

Composes Bodies and Joints into a ModelDef with auto-computed dimensions.
Uses Variadic.types + comptimefor to iterate at compile time, following
the same pattern as Sequential[*LAYERS: Model] in nn/model/sequential.mojo.

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
from render import Color, Renderer3D, Light, Camera3D
from math3d import Vec3 as _Vec3G, Quat as _QuatG

from .body_spec import BodySpec
from .joint_spec import JointSpec
from .geom_spec import GeomSpec
from .equality_spec import EqualitySpec
from .camera_spec import CameraSpec
from .light_spec import LightSpec
from .texture_spec import TextureSpec
from .material_spec import MaterialSpec
from .actuator_spec import (
    ActuatorSpec,
    DYN_NONE,
    DYN_INTEGRATOR,
    DYN_FILTER,
    DYN_FILTEREXACT,
    GAIN_FIXED,
    GAIN_AFFINE,
    BIAS_NONE,
    BIAS_AFFINE,
)
from ..types import (
    Model,
    Data,
    ActuatorDef,
    EqualityConstraintDef,
    EQ_CONNECT,
    EQ_WELD,
    ConeType,
)
from ..joint_types import JNT_HINGE, JNT_SLIDE
from math import sqrt
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_PLANE,
    GEOM_CYLINDER,
)

# GPU imports
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from ..gpu.constants import (
    TPB,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    model_size_with_invweight,
    model_metadata_offset,
    model_body_offset,
    model_body_invweight0_offset,
    model_dof_invweight0_offset,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_META_IDX_SOLIMP_LIMIT_3,
    MODEL_META_IDX_SOLIMP_LIMIT_4,
    MODEL_META_IDX_IMPRATIO,
    MODEL_META_IDX_NEQUALITY,
    MODEL_META_IDX_NTENDON,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    BODY_IDX_MASS,
    BODY_IDX_INV_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_INV_IXX,
    BODY_IDX_INV_IYY,
    BODY_IDX_INV_IZZ,
    MODEL_BODY_SIZE,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_PARENT,
    BODY_IDX_IPOS_X,
    BODY_IDX_IPOS_Y,
    BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    state_size,
    model_size,
    xpos_offset,
    xquat_offset,
    xipos_offset,
    model_joint_offset,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_ARMATURE,
    integrator_workspace_size,
    ws_cdof_offset,
    ws_M_offset,
    ws_L_offset,
    ws_D_offset,
)
from ..gpu.buffer_utils import (
    copy_model_to_buffer,
    copy_geoms_to_buffer,
    copy_invweight0_to_buffer,
    copy_tendons_to_buffer,
)
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from ..dynamics.mass_matrix import (
    compute_body_invweight0,
    ldl_factor_gpu,
    compute_mass_matrix_full_gpu,
)
from ..dynamics.jacobian import compute_cdof_gpu, compute_composite_inertia_gpu
from memory import UnsafePointer
from .inertia_from_geom import (
    geom_volume,
    compute_inertia_from_geoms,
    compute_inertia_from_geoms_buffer,
)
from gpu.host import HostBuffer
from ..model.defaults_spec import ModelDefaults
from ..model.body_spec import BodiesLike, _EmptyBodies
from ..model.joint_spec import JointsLike, _EmptyJoints
from ..model.geom_spec import GeomsLike, _EmptyGeoms
from ..model.actuator_spec import ActuatorsLike, _EmptyActuators
from ..model.light_spec import LightsLike, _EmptyLights
from ..model.texture_spec import TexturesLike, _EmptyTextures
from ..model.material_spec import MaterialsLike, _EmptyMaterials
from ..model.camera_spec import CamerasLike, _EmptyCameras
from ..model.site_spec import SitesLike, _EmptySites


comptime _RVec3 = _Vec3G[DType.float64]
comptime _RQuat = _QuatG[DType.float64]


trait ModelDefLike:
    """Trait for model definition types."""

    # === Dimensions ===
    comptime NQ: Int
    comptime NV: Int
    comptime NBODY: Int
    comptime NJOINT: Int
    comptime NGEOM: Int
    comptime MAX_EQUALITY: Int
    comptime CONE_TYPE: Int
    comptime MAX_CONTACTS: Int
    comptime MAX_TENDON: Int
    comptime NSITE: Int
    comptime OBS_DIM: Int
    comptime ACTION_DIM: Int
    comptime TIMESTEP: Float64

    # === Components ===
    # comptime BODIES: BodiesLike
    # comptime JOINTS: JointsLike
    # comptime GEOMS: GeomsLike
    # comptime ACTUATORS: ActuatorsLike
    # comptime DEFAULTS: ModelDefaultsLike
    # comptime LIGHTS: LightsLike
    # comptime TEXTURES: TexturesLike
    # comptime MATERIALS: MaterialsLike
    # comptime CAMERAS: CamerasLike

    # === CPU: Model setup (calls Bodies/Joints/Geoms/Defaults internally) ===
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
        ...

    # === CPU: Joints/Actuators delegates ===
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
        ...

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
        ...

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
        ...

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
        ...

    # === CPU: Float getters (can't use Float64 as comptime in traits) ===

    # === GPU: Model init ===
    @staticmethod
    fn init_model_gpu[
        DTYPE: DType
    ](ctx: DeviceContext, mut model_buf: DeviceBuffer[DTYPE],) raises:
        ...

    # === GPU: Joints/Actuators kernel delegates ===
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
        ...

    @staticmethod
    fn enforce_limits_kernel_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[DTYPE]) raises:
        ...

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
        ...

    # === GPU inline: Per-env methods (called from inside GPU kernels) ===
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
        ...

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
        ...

    @staticmethod
    fn render_ground_geoms(
        mut renderer: Renderer3D,
        torso_x: Float64,
        follow: Bool,
        visual_radius_scale: Float64,
    ) raises:
        ...

    @staticmethod
    fn render_body_geoms(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
        visual_radius_scale: Float64,
    ) raises:
        ...

    @staticmethod
    fn setup_lights() raises -> List[Light]:
        ...

    @staticmethod
    fn setup_cameras(width: Int, height: Int) raises -> List[Camera3D]:
        ...

    @staticmethod
    fn setup_camera_modes() raises -> List[Int]:
        ...

    @staticmethod
    fn get_skybox_colors() -> List[Float64]:
        ...

    @staticmethod
    fn get_checker_colors() -> List[Float64]:
        ...

    @staticmethod
    fn render_sites(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        ...


@fieldwise_init
struct ModelDef[
    Bodies: BodiesLike = _EmptyBodies,
    Joints: JointsLike = _EmptyJoints,
    Geoms: GeomsLike = _EmptyGeoms,
    Actuators: ActuatorsLike = _EmptyActuators,
    Defaults: ModelDefaultsLike = ModelDefaults[],
    Lights: LightsLike = _EmptyLights,
    Textures: TexturesLike = _EmptyTextures,
    Materials: MaterialsLike = _EmptyMaterials,
    Cameras: CamerasLike = _EmptyCameras,
    Sites: SitesLike = _EmptySites,
    max_equality: Int = 0,
    max_contacts: Int = 0,
    cone_type: Int = ConeType.ELLIPTIC,
    max_tendon: Int = 0,
    # Embedded component types via trait bounds (optional, backward compatible)
](ModelDefLike):
    """Compile-time model definition with pre-computed dimensions.

    Optionally embeds component types (Bodies, Joints, Geoms, Actuators, Defaults) for the full robot
    definition — like a parsed MuJoCo XML. When component types are provided,
    convenience methods (setup_all, reset_data, etc.) delegate to them.

    Usage (full — with components):
        comptime MyModel = ModelDef[
            MyBodies, MyJoints, MyGeoms, MyActuators, MyDefaults,
        ]
    """

    comptime NBODY: Int = Self.Bodies.N + 1  # +1 for worldbody at index 0
    comptime NJOINT: Int = Self.Joints.N
    comptime NQ: Int = Self.Joints.NQ
    comptime NV: Int = Self.Joints.NV
    comptime NGEOM: Int = Self.Geoms.N
    comptime MAX_EQUALITY: Int = Self.max_equality
    comptime CONE_TYPE: Int = Self.cone_type
    comptime MAX_CONTACTS: Int = Self.max_contacts
    comptime MAX_TENDON: Int = Self.max_tendon
    comptime NSITE: Int = Self.Sites.N
    comptime TIMESTEP: Float64 = Self.Defaults.TIMESTEP

    # Derived from components (only meaningful when J is not _EmptyJoints)
    comptime OBS_DIM: Int = Self.Joints.OBS_DIM
    comptime ACTION_DIM: Int = Self.Joints.ACTION_DIM

    # comptime BODIES: BodiesLike = Self.Bodies
    # comptime JOINTS: JointsLike = Self.Joints
    # comptime GEOMS: GeomsLike = Self.Geoms
    # comptime ACTUATORS: ActuatorsLike = Self.Actuators
    # comptime DEFAULTS: ModelDefaultsLike = Self.Defaults
    # comptime LIGHTS: LightsLike = Self.Lights
    # comptime TEXTURES: TexturesLike = Self.Textures
    # comptime MATERIALS: MaterialsLike = Self.Materials
    # comptime CAMERAS: CamerasLike = Self.Cameras

    # =========================================================================
    # Model Creation Helpers
    # =========================================================================

    @staticmethod
    fn setup_solver_params[
        DTYPE: DType,
        MAX_CONTACTS: Int,
    ](
        mut model: Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            MAX_CONTACTS,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.CONE_TYPE,
            Self.MAX_TENDON,
            Self.NSITE,
        ],
    ):
        """Set all solver impedance params on a Model from ModelDefaults.

        Sets model-level solref/solimp for contacts and limits, plus impratio.
        Per-geom and per-joint overrides are set in Geoms/Joints.setup_model.
        """
        model.solref_contact[0] = Scalar[DTYPE](Self.Defaults.GEOM_SOLREF_0)
        model.solref_contact[1] = Scalar[DTYPE](Self.Defaults.GEOM_SOLREF_1)
        model.solimp_contact[0] = Scalar[DTYPE](Self.Defaults.GEOM_SOLIMP_0)
        model.solimp_contact[1] = Scalar[DTYPE](Self.Defaults.GEOM_SOLIMP_1)
        model.solimp_contact[2] = Scalar[DTYPE](Self.Defaults.GEOM_SOLIMP_2)
        model.solimp_contact[3] = Scalar[DTYPE](Self.Defaults.GEOM_SOLIMP_3)
        model.solimp_contact[4] = Scalar[DTYPE](Self.Defaults.GEOM_SOLIMP_4)
        model.solref_limit[0] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLREF_LIMIT_0
        )
        model.solref_limit[1] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLREF_LIMIT_1
        )
        model.solimp_limit[0] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLIMP_LIMIT_0
        )
        model.solimp_limit[1] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLIMP_LIMIT_1
        )
        model.solimp_limit[2] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLIMP_LIMIT_2
        )
        model.solimp_limit[3] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLIMP_LIMIT_3
        )
        model.solimp_limit[4] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLIMP_LIMIT_4
        )
        model.impratio = Scalar[DTYPE](Self.Defaults.IMPRATIO)
        model.gravity = SIMD[DTYPE, 4](
            Scalar[DTYPE](Self.Defaults.GRAVITY_X),
            Scalar[DTYPE](Self.Defaults.GRAVITY_Y),
            Scalar[DTYPE](Self.Defaults.GRAVITY_Z),
            Scalar[DTYPE](0),
        )
        model.timestep = Scalar[DTYPE](Self.Defaults.TIMESTEP)
        model.opt_density = Scalar[DTYPE](Self.Defaults.OPT_DENSITY)
        model.opt_viscosity = Scalar[DTYPE](Self.Defaults.OPT_VISCOSITY)

    @staticmethod
    fn finalize[
        DTYPE: DType,
        MAX_CONTACTS: Int,
    ](
        mut model: Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            MAX_CONTACTS,
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
            MAX_CONTACTS,
            Self.NSITE,
        ],
    ):
        """Run FK + compute_body_invweight0 in the correct order.

        Must be called after Bodies/Joints/Geoms.setup_model and after
        Joints.reset_data (or manual qpos initialization).

        Order: inertiafromgeom → settotalmass → FK → invweight0

        If Defaults.INERTIAFROMGEOM is True, computes body mass/inertia/ipos/iquat
        from child geoms (overwriting any values set by Bodies.setup_model).

        If Defaults.SETTOTALMASS > 0, rescales all body masses and inertias
        so the total mass equals the target (MuJoCo <compiler settotalmass>).
        """

        # MuJoCo <compiler inertiafromgeom> — compute body inertia from geoms
        comptime if Self.Defaults.INERTIAFROMGEOM and Self.NGEOM > 0:
            compute_inertia_from_geoms(model)

        # MuJoCo <compiler settotalmass> — rescale body masses/inertias
        comptime if Self.Defaults.SETTOTALMASS > 0.0:
            var total_mass = Scalar[DTYPE](0)
            for i in range(1, Self.NBODY):
                total_mass += model.body_mass[i]
            if total_mass > 0:
                var scale = (
                    Scalar[DTYPE](Self.Defaults.SETTOTALMASS) / total_mass
                )
                for i in range(1, Self.NBODY):
                    model.body_mass[i] *= scale
                    model.body_inv_mass[i] /= scale
                    for k in range(3):
                        model.body_inertia[i * 3 + k] *= scale
                        model.body_inv_inertia[i * 3 + k] /= scale

        forward_kinematics(model, data)
        compute_body_invweight0(model, data)

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
        Self.setup_solver_params(model)
        Self.Bodies.setup_model(model)
        Self.Joints.setup_model[Defaults = Self.Defaults](model)
        Self.Geoms.setup_model[Defaults = Self.Defaults](model)
        Self.Sites.setup_model(model)
        Self.Joints.reset_data(data)
        Self.finalize(model, data)

    # === GPU: Model init ===
    @staticmethod
    fn init_model_gpu[
        DTYPE: DType
    ](ctx: DeviceContext, mut model_buf: DeviceBuffer[DTYPE],) raises:
        """Initialize GPU model buffer by writing directly to HostBuffer.

        Bypasses creating a full Model struct on the stack (which causes
        stack overflow for large robots like Ant). Instead writes body, joint,
        geom, and metadata directly to the buffer using compile-time specs.
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
        # Zero-initialize
        for i in range(BUF_SIZE):
            host_buf[i] = Scalar[DTYPE](0)

        # Direct writes (no Model struct)
        Self.Bodies.write_to_buffer[DTYPE, Self.NBODY](host_buf)
        Self.Joints.write_to_buffer[
            DTYPE, Self.NBODY, Defaults = Self.Defaults
        ](host_buf)

        comptime if Self.NGEOM > 0:
            Self.Geoms.write_to_buffer[
                DTYPE, Self.NBODY, Self.NJOINT, Defaults = Self.Defaults
            ](host_buf)

        comptime if Self.NSITE > 0:
            Self.Sites.write_to_buffer[
                DTYPE,
                Self.NBODY,
                Self.NJOINT,
                Self.NGEOM,
                Self.MAX_EQUALITY,
                Self.MAX_TENDON,
            ](host_buf)
        Self._write_metadata_to_buffer[DTYPE](host_buf)

        # Derived computations on buffer
        comptime if Self.Defaults.INERTIAFROMGEOM and Self.NGEOM > 0:
            var geom_masses = Self.Geoms.compute_geom_masses[
                DTYPE, Defaults = Self.Defaults
            ]()
            compute_inertia_from_geoms_buffer[
                DTYPE, Self.NBODY, Self.NJOINT, Self.NGEOM
            ](host_buf, geom_masses)

        comptime if Self.Defaults.SETTOTALMASS > 0.0:
            Self._settotalmass_buffer[DTYPE](host_buf)

        # Copy to GPU (invweight0 slots are still zero)
        ctx.enqueue_copy(model_buf, host_buf.unsafe_ptr())

        # Compute invweight0 on GPU (avoids CPU stack overflow)
        Self._compute_invweight0_gpu[DTYPE](ctx, model_buf)

    @staticmethod
    fn _write_metadata_to_buffer[
        DTYPE: DType,
    ](buffer: HostBuffer[DTYPE]):
        """Write model metadata directly to GPU HostBuffer."""
        var off = model_metadata_offset[Self.NBODY, Self.NJOINT]()
        buffer[off + MODEL_META_IDX_NBODY] = Scalar[DTYPE](Self.NBODY)
        buffer[off + MODEL_META_IDX_NJOINT] = Scalar[DTYPE](Self.NJOINT)
        buffer[off + MODEL_META_IDX_GRAVITY_X] = Scalar[DTYPE](
            Self.Defaults.GRAVITY_X
        )
        buffer[off + MODEL_META_IDX_GRAVITY_Y] = Scalar[DTYPE](
            Self.Defaults.GRAVITY_Y
        )
        buffer[off + MODEL_META_IDX_GRAVITY_Z] = Scalar[DTYPE](
            Self.Defaults.GRAVITY_Z
        )
        buffer[off + MODEL_META_IDX_TIMESTEP] = Scalar[DTYPE](
            Self.Defaults.TIMESTEP
        )
        buffer[off + MODEL_META_IDX_SOLREF_CONTACT_0] = Scalar[DTYPE](
            Self.Defaults.GEOM_SOLREF_0
        )
        buffer[off + MODEL_META_IDX_SOLREF_CONTACT_1] = Scalar[DTYPE](
            Self.Defaults.GEOM_SOLREF_1
        )
        buffer[off + MODEL_META_IDX_SOLIMP_CONTACT_0] = Scalar[DTYPE](
            Self.Defaults.GEOM_SOLIMP_0
        )
        buffer[off + MODEL_META_IDX_SOLIMP_CONTACT_1] = Scalar[DTYPE](
            Self.Defaults.GEOM_SOLIMP_1
        )
        buffer[off + MODEL_META_IDX_SOLIMP_CONTACT_2] = Scalar[DTYPE](
            Self.Defaults.GEOM_SOLIMP_2
        )
        buffer[off + MODEL_META_IDX_SOLIMP_CONTACT_3] = Scalar[DTYPE](
            Self.Defaults.GEOM_SOLIMP_3
        )
        buffer[off + MODEL_META_IDX_SOLIMP_CONTACT_4] = Scalar[DTYPE](
            Self.Defaults.GEOM_SOLIMP_4
        )
        buffer[off + MODEL_META_IDX_SOLREF_LIMIT_0] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLREF_LIMIT_0
        )
        buffer[off + MODEL_META_IDX_SOLREF_LIMIT_1] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLREF_LIMIT_1
        )
        buffer[off + MODEL_META_IDX_SOLIMP_LIMIT_0] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLIMP_LIMIT_0
        )
        buffer[off + MODEL_META_IDX_SOLIMP_LIMIT_1] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLIMP_LIMIT_1
        )
        buffer[off + MODEL_META_IDX_SOLIMP_LIMIT_2] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLIMP_LIMIT_2
        )
        buffer[off + MODEL_META_IDX_SOLIMP_LIMIT_3] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLIMP_LIMIT_3
        )
        buffer[off + MODEL_META_IDX_SOLIMP_LIMIT_4] = Scalar[DTYPE](
            Self.Defaults.JOINT_SOLIMP_LIMIT_4
        )
        buffer[off + MODEL_META_IDX_IMPRATIO] = Scalar[DTYPE](
            Self.Defaults.IMPRATIO
        )
        buffer[off + MODEL_META_IDX_NEQUALITY] = Scalar[DTYPE](
            Self.MAX_EQUALITY
        )
        buffer[off + MODEL_META_IDX_NTENDON] = Scalar[DTYPE](Self.MAX_TENDON)
        buffer[off + MODEL_META_IDX_DENSITY] = Scalar[DTYPE](
            Self.Defaults.OPT_DENSITY
        )
        buffer[off + MODEL_META_IDX_VISCOSITY] = Scalar[DTYPE](
            Self.Defaults.OPT_VISCOSITY
        )

    @staticmethod
    fn _settotalmass_buffer[
        DTYPE: DType,
    ](buffer: HostBuffer[DTYPE]):
        """Rescale body masses/inertias so total matches target (buffer version).
        """
        var total_mass = Scalar[DTYPE](0)
        for i in range(1, Self.NBODY):
            var off = model_body_offset(i)
            total_mass += buffer[off + BODY_IDX_MASS]
        if total_mass > Scalar[DTYPE](0):
            var scale = Scalar[DTYPE](Self.Defaults.SETTOTALMASS) / total_mass
            for i in range(1, Self.NBODY):
                var off = model_body_offset(i)
                buffer[off + BODY_IDX_MASS] *= scale
                buffer[off + BODY_IDX_INV_MASS] /= scale
                buffer[off + BODY_IDX_IXX] *= scale
                buffer[off + BODY_IDX_IYY] *= scale
                buffer[off + BODY_IDX_IZZ] *= scale
                buffer[off + BODY_IDX_INV_IXX] /= scale
                buffer[off + BODY_IDX_INV_IYY] /= scale
                buffer[off + BODY_IDX_INV_IZZ] /= scale

    @staticmethod
    fn _compute_invweight0_gpu[
        DTYPE: DType,
    ](ctx: DeviceContext, mut model_buf: DeviceBuffer[DTYPE]) raises:
        """Compute invweight0 on GPU via a single-thread kernel.

        Avoids creating Model/Data on the CPU stack entirely. Uses existing GPU
        functions (FK, cdof, crb, mass matrix, LDL) and writes invweight0
        directly to the model buffer.
        """
        from ..joint_types import JNT_FREE, JNT_BALL

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

        # Allocate temporary state + workspace on GPU
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

        # Kernel: init state to zero, run FK + mass matrix + LDL, compute invweight0
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
            # Zero-init state (qpos=0 is correct for slide/hinge joints)
            for i in range(STATE_SIZE):
                state[0, i] = Scalar[DTYPE](0)

            # FK
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

            # cdof
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

            # Composite rigid body inertia
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

            # Mass matrix
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

            # LDL factor
            ldl_factor_gpu[DTYPE, Self.NV, Self.NBODY, 1, WS_SIZE](0, workspace)

            # === Compute invweight0 ===
            # Build dof_to_body mapping
            comptime cdof_idx = ws_cdof_offset()
            comptime L_idx = ws_L_offset[Self.NV, Self.NBODY]()
            comptime D_idx = ws_D_offset[Self.NV, Self.NBODY]()
            comptime scratch1 = D_idx + Self.NV  # after D (reuses fnet slot)
            comptime scratch2 = scratch1 + Self.NV  # reuses qacc slot
            comptime scratch3 = scratch2 + Self.NV  # reuses qvel_pred slot
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

            # World body: zero weights
            model[0, bw_off + 0] = Scalar[DTYPE](0)
            model[0, bw_off + 1] = Scalar[DTYPE](0)

            # For each body, compute invweight0
            for i in range(Self.NBODY):
                var ti_x = rebind[Scalar[DTYPE]](state[0, xi_off + i * 3 + 0])
                var ti_y = rebind[Scalar[DTYPE]](state[0, xi_off + i * 3 + 1])
                var ti_z = rebind[Scalar[DTYPE]](state[0, xi_off + i * 3 + 2])

                # Build dof_body for "affects" check
                # For each DOF k, check if it affects body i
                var A_diag_tran = Scalar[DTYPE](0)
                var A_diag_rot = Scalar[DTYPE](0)

                for k in range(6):
                    # Build J_row for this spatial component
                    # Then solve M*x = J_row and compute A[k,k] = dot(J_row, x)

                    # Step 1: Build J_row by zeroing a scratch area
                    # Use dw_off + NV as scratch for J_row (temporarily)
                    # Actually, we need a real temp. Use a small loop approach.
                    # Solve M*x = J_row:
                    # Forward sub: y[i] = b[i] - sum(L[i,j]*y[j])
                    # Scale: z[i] = y[i]/D[i]
                    # Back sub: x[i] = z[i] - sum(L[j,i]*x[j])

                    var dot_val = Scalar[DTYPE](0)

                    # For each DOF d, compute J_row[d] and accumulate
                    # We do the LDL solve inline, one row at a time
                    # This avoids allocating NV-sized arrays in GPU registers

                    # Build J_row in scratch1
                    for d in range(Self.NV):
                        workspace[0, scratch1 + d] = Scalar[DTYPE](0)

                    for d in range(Self.NV):
                        # Find which body owns this DOF
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

                        # Check if DOF d affects body i
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

                    # LDL solve: M*x = J_row
                    # Forward substitution: y = L^{-1} * b
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
                # e_d unit vector
                for ii in range(Self.NV):
                    workspace[0, scratch1 + ii] = Scalar[DTYPE](0)
                workspace[0, scratch1 + d] = Scalar[DTYPE](1)

                # LDL solve
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

    # === CPU: Joints/Actuators delegates ===
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
        Self.Joints.reset_data(data)

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
        Self.Joints.extract_obs(data, obs)

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
        Self.Joints.enforce_limits(data)

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
        Self.Actuators.apply_actions(data, actions)

    # === GPU: Joints/Actuators kernel delegates ===
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
        Self.Actuators.apply_actions_kernel_gpu[
            DTYPE, BATCH_SIZE, STATE_SIZE, ACTION_DIM, Self.NQ, Self.NV
        ](ctx, states_buf, actions_buf)

    @staticmethod
    fn enforce_limits_kernel_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[DTYPE]) raises:
        Self.Joints.enforce_limits_kernel_gpu[DTYPE, BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
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
        Self.Joints.extract_obs_kernel_gpu[
            DTYPE, BATCH_SIZE, STATE_SIZE, OBS_DIM
        ](ctx, states_buf, obs_buf)

    # === GPU inline: Per-env delegates ===
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
        Self.Joints.reset_env_gpu[DTYPE, BATCH_SIZE, STATE_SIZE](
            states, env, noise_scale, seed
        )

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
        Self.Joints.extract_obs_gpu[DTYPE, BATCH_SIZE, STATE_SIZE, OBS_DIM](
            states, obs, env
        )

    @staticmethod
    fn create_gpu_model_buffer[
        DTYPE: DType,
        MAX_CONTACTS: Int,
    ](
        ctx: DeviceContext,
        model: Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            MAX_CONTACTS,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.CONE_TYPE,
            Self.MAX_TENDON,
            Self.NSITE,
        ],
    ) raises -> HostBuffer[DTYPE]:
        """Create a GPU host buffer from a fully-configured model.

        Allocates buffer with model_size_with_invweight, copies model data,
        geoms, and invweight0 arrays. Returns host buffer ready for
        ctx.enqueue_copy to a DeviceBuffer.
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
        copy_model_to_buffer(model, host_buf)
        copy_geoms_to_buffer(model, host_buf)
        copy_tendons_to_buffer(model, host_buf)
        copy_invweight0_to_buffer(model, host_buf)
        return host_buf^

    @staticmethod
    fn render_ground_geoms(
        mut renderer: Renderer3D,
        torso_x: Float64,
        follow: Bool,
        visual_radius_scale: Float64,
    ) raises:
        Self.Geoms.render_ground_geoms(
            renderer, torso_x, follow, visual_radius_scale
        )

    @staticmethod
    fn render_body_geoms(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
        visual_radius_scale: Float64,
    ) raises:
        Self.Geoms.render_body_geoms(
            renderer, positions, quaternions, visual_radius_scale
        )

    @staticmethod
    fn setup_lights() raises -> List[Light]:
        return Self.Lights.setup_lights()

    @staticmethod
    fn setup_cameras(width: Int, height: Int) raises -> List[Camera3D]:
        return Self.Cameras.setup_cameras(width, height)

    @staticmethod
    fn setup_camera_modes() raises -> List[Int]:
        return Self.Cameras.setup_camera_modes()

    @staticmethod
    fn get_skybox_colors() -> List[Float64]:
        return Self.Textures.get_skybox_colors()

    @staticmethod
    fn get_checker_colors() -> List[Float64]:
        return Self.Textures.get_checker_colors()

    @staticmethod
    fn render_sites(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        Self.Sites.render_sites(renderer, positions, quaternions)
