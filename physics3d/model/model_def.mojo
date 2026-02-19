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
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_META_IDX_IMPRATIO,
    MODEL_META_IDX_NEQUALITY,
    MODEL_META_IDX_NTENDON,
    BODY_IDX_MASS,
    BODY_IDX_INV_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_INV_IXX,
    BODY_IDX_INV_IYY,
    BODY_IDX_INV_IZZ,
    MODEL_BODY_SIZE,
)
from ..gpu.buffer_utils import (
    copy_model_to_buffer,
    copy_geoms_to_buffer,
    copy_invweight0_to_buffer,
    copy_tendons_to_buffer,
)
from ..kinematics.forward_kinematics import forward_kinematics
from ..dynamics.mass_matrix import compute_body_invweight0
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
    comptime OBS_DIM: Int
    comptime ACTION_DIM: Int

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
        DTYPE: DType where DTYPE.is_floating_point()
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
        ],
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
    ):
        ...

    # === CPU: Joints/Actuators delegates ===
    @staticmethod
    fn reset_data[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
    ):
        ...

    @staticmethod
    fn extract_obs[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
        mut obs: List[Scalar[DTYPE]],
    ):
        ...

    @staticmethod
    fn enforce_limits[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
    ):
        ...

    @staticmethod
    fn apply_actions[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
        actions: List[Float64],
    ):
        ...

    # === CPU: Float getters (can't use Float64 as comptime in traits) ===

    # === GPU: Model init ===
    @staticmethod
    fn init_model_gpu[
        DTYPE: DType where DTYPE.is_floating_point()
    ](ctx: DeviceContext, mut model_buf: DeviceBuffer[DTYPE],) raises:
        ...

    # === GPU: Joints/Actuators kernel delegates ===
    @staticmethod
    fn apply_actions_kernel_gpu[
        DTYPE: DType where DTYPE.is_floating_point(),
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
        DTYPE: DType where DTYPE.is_floating_point(),
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[DTYPE]) raises:
        ...

    @staticmethod
    fn extract_obs_kernel_gpu[
        DTYPE: DType where DTYPE.is_floating_point(),
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
        model.impratio = Scalar[DTYPE](Self.Defaults.IMPRATIO)
        model.gravity = SIMD[DTYPE, 4](
            Scalar[DTYPE](Self.Defaults.GRAVITY_X),
            Scalar[DTYPE](Self.Defaults.GRAVITY_Y),
            Scalar[DTYPE](Self.Defaults.GRAVITY_Z),
            Scalar[DTYPE](0),
        )
        model.timestep = Scalar[DTYPE](Self.Defaults.TIMESTEP)

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
        ],
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            MAX_CONTACTS,
        ],
    ) where DTYPE.is_floating_point():
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
        @parameter
        if Self.Defaults.INERTIAFROMGEOM and Self.NGEOM > 0:
            compute_inertia_from_geoms(model)

        # MuJoCo <compiler settotalmass> — rescale body masses/inertias
        @parameter
        if Self.Defaults.SETTOTALMASS > 0.0:
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
        DTYPE: DType where DTYPE.is_floating_point()
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
        ],
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
    ):
        Self.setup_solver_params(model)
        Self.Bodies.setup_model(model)
        Self.Joints.setup_model[Defaults = Self.Defaults](model)
        Self.Geoms.setup_model[Defaults = Self.Defaults](model)
        Self.Joints.reset_data(data)
        Self.finalize(model, data)

    # === GPU: Model init ===
    @staticmethod
    fn init_model_gpu[
        DTYPE: DType where DTYPE.is_floating_point()
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
        ]()
        var host_buf = ctx.enqueue_create_host_buffer[DTYPE](BUF_SIZE)
        # Zero-initialize
        for i in range(BUF_SIZE):
            host_buf[i] = Scalar[DTYPE](0)

        # Direct writes (no Model struct)
        Self.Bodies.write_to_buffer[DTYPE, Self.NBODY](host_buf)
        Self.Joints.write_to_buffer[DTYPE, Self.NBODY, Defaults = Self.Defaults](
            host_buf
        )
        @parameter
        if Self.NGEOM > 0:
            Self.Geoms.write_to_buffer[
                DTYPE, Self.NBODY, Self.NJOINT, Defaults = Self.Defaults
            ](host_buf)
        Self._write_metadata_to_buffer[DTYPE](host_buf)

        # Derived computations on buffer
        @parameter
        if Self.Defaults.INERTIAFROMGEOM and Self.NGEOM > 0:
            var geom_masses = Self.Geoms.compute_geom_masses[
                DTYPE, Defaults = Self.Defaults
            ]()
            compute_inertia_from_geoms_buffer[
                DTYPE, Self.NBODY, Self.NJOINT, Self.NGEOM
            ](host_buf, geom_masses)

        @parameter
        if Self.Defaults.SETTOTALMASS > 0.0:
            Self._settotalmass_buffer[DTYPE](host_buf)

        # invweight0 via small Model (reuses existing FK + invweight0 code)
        Self._compute_invweight0_from_buffer[DTYPE](host_buf)

        # Copy to GPU
        ctx.enqueue_copy(model_buf, host_buf.unsafe_ptr())

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
        buffer[off + MODEL_META_IDX_IMPRATIO] = Scalar[DTYPE](
            Self.Defaults.IMPRATIO
        )
        buffer[off + MODEL_META_IDX_NEQUALITY] = Scalar[DTYPE](
            Self.MAX_EQUALITY
        )
        buffer[off + MODEL_META_IDX_NTENDON] = Scalar[DTYPE](Self.MAX_TENDON)

    @staticmethod
    fn _settotalmass_buffer[
        DTYPE: DType,
    ](buffer: HostBuffer[DTYPE]):
        """Rescale body masses/inertias so total matches target (buffer version)."""
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
    fn _compute_invweight0_from_buffer[
        DTYPE: DType where DTYPE.is_floating_point(),
    ](buffer: HostBuffer[DTYPE]):
        """Compute body_invweight0/dof_invweight0 via a small Model on the stack.

        Creates a minimal Model (NGEOM=0, MAX_CONTACTS=1) which is small enough
        for the stack. Reads body+joint data from the buffer, runs FK + invweight0,
        then writes results back to the buffer.
        """
        # Small model with minimal geom/contact footprint
        comptime SmallModel = Model[
            DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            1,  # MAX_CONTACTS=1 (minimal)
            0,  # NGEOM=0
            0,  # MAX_EQUALITY=0
            Self.CONE_TYPE,
            0,  # MAX_TENDON=0
        ]
        comptime SmallData = Data[
            DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT, 1,
        ]

        var model = SmallModel()
        var data = SmallData()

        # Copy body data from buffer → small model
        for b in range(Self.NBODY):
            var off = model_body_offset(b)
            model.body_mass[b] = buffer[off + BODY_IDX_MASS]
            model.body_inv_mass[b] = buffer[off + BODY_IDX_INV_MASS]
            model.body_inertia[b * 3 + 0] = buffer[off + BODY_IDX_IXX]
            model.body_inertia[b * 3 + 1] = buffer[off + BODY_IDX_IYY]
            model.body_inertia[b * 3 + 2] = buffer[off + BODY_IDX_IZZ]
            model.body_inv_inertia[b * 3 + 0] = buffer[off + BODY_IDX_INV_IXX]
            model.body_inv_inertia[b * 3 + 1] = buffer[off + BODY_IDX_INV_IYY]
            model.body_inv_inertia[b * 3 + 2] = buffer[off + BODY_IDX_INV_IZZ]
            model.body_pos[b * 3 + 0] = buffer[
                off + 8
            ]  # BODY_IDX_POS_X=8
            model.body_pos[b * 3 + 1] = buffer[off + 9]
            model.body_pos[b * 3 + 2] = buffer[off + 10]
            model.body_quat[b * 4 + 0] = buffer[off + 11]  # BODY_IDX_QUAT_X
            model.body_quat[b * 4 + 1] = buffer[off + 12]
            model.body_quat[b * 4 + 2] = buffer[off + 13]
            model.body_quat[b * 4 + 3] = buffer[off + 14]
            model.body_parent[b] = Int(buffer[off + 15])  # BODY_IDX_PARENT
            model.body_ipos[b * 3 + 0] = buffer[off + 16]  # BODY_IDX_IPOS_X
            model.body_ipos[b * 3 + 1] = buffer[off + 17]
            model.body_ipos[b * 3 + 2] = buffer[off + 18]
            model.body_iquat[b * 4 + 0] = buffer[off + 19]  # BODY_IDX_IQUAT_X
            model.body_iquat[b * 4 + 1] = buffer[off + 20]
            model.body_iquat[b * 4 + 2] = buffer[off + 21]
            model.body_iquat[b * 4 + 3] = buffer[off + 22]

        # Populate joints via Joints.setup_model on the small model
        Self.Joints.setup_model[
            DTYPE=DTYPE, NQ=Self.NQ, NV=Self.NV, NBODY=Self.NBODY,
            MAX_CONTACTS=1, NGEOM=0, MAX_EQUALITY=0,
            CONE_TYPE=Self.CONE_TYPE, MAX_TENDON=0,
            Defaults=Self.Defaults,
        ](model)

        # Initialize qpos from JointSpec defaults
        Self.Joints.reset_data(data)

        # Set gravity and timestep
        model.gravity = SIMD[DTYPE, 4](
            Scalar[DTYPE](Self.Defaults.GRAVITY_X),
            Scalar[DTYPE](Self.Defaults.GRAVITY_Y),
            Scalar[DTYPE](Self.Defaults.GRAVITY_Z),
            Scalar[DTYPE](0),
        )
        model.timestep = Scalar[DTYPE](Self.Defaults.TIMESTEP)

        # Run FK + invweight0
        forward_kinematics(model, data)
        compute_body_invweight0(model, data)

        # Write invweight0 results back to buffer
        var bw_off = model_body_invweight0_offset[
            Self.NBODY, Self.NJOINT, Self.NGEOM, Self.MAX_EQUALITY,
            Self.MAX_TENDON,
        ]()
        for i in range(Self.NBODY * 2):
            buffer[bw_off + i] = model.body_invweight0[i]
        var dw_off = model_dof_invweight0_offset[
            Self.NBODY, Self.NJOINT, Self.NGEOM, Self.MAX_EQUALITY,
            Self.MAX_TENDON,
        ]()
        for i in range(Self.NV):
            buffer[dw_off + i] = model.dof_invweight0[i]

    # === CPU: Joints/Actuators delegates ===
    @staticmethod
    fn reset_data[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
    ):
        Self.Joints.reset_data(data)

    @staticmethod
    fn extract_obs[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
        mut obs: List[Scalar[DTYPE]],
    ):
        Self.Joints.extract_obs(data, obs)

    @staticmethod
    fn enforce_limits[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
    ):
        Self.Joints.enforce_limits(data)

    @staticmethod
    fn apply_actions[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
        actions: List[Float64],
    ):
        Self.Actuators.apply_actions(data, actions)

    # === GPU: Joints/Actuators kernel delegates ===
    @staticmethod
    fn apply_actions_kernel_gpu[
        DTYPE: DType where DTYPE.is_floating_point(),
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
        DTYPE: DType where DTYPE.is_floating_point(),
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[DTYPE]) raises:
        Self.Joints.enforce_limits_kernel_gpu[DTYPE, BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
        )

    @staticmethod
    fn extract_obs_kernel_gpu[
        DTYPE: DType where DTYPE.is_floating_point(),
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
