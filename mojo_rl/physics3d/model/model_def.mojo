"""`ModelDefLike` — the trait every physics3d model definition implements.

Declares the compile-time dimensions, the CPU state hooks (fields-native:
`reset_data` / `extract_obs` / `enforce_limits` / `apply_actions` over
`DataFields`), the spec-direct fields model build (`init_fields`), the GPU
kernel delegates, and the render hooks. Sole implementer: `ModelDefFromXML`
(the spec-based `ModelDef` compositor was deleted at the G2 fields sunset;
the legacy CPU `Model`/`Data` build at G4).
"""

from mojo_rl.render import Renderer3D, Light, Camera3D
from mojo_rl.math3d import Vec3 as _Vec3G, Quat as _QuatG

from ..fields import ModelFields, DataFields

# GPU imports
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor


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
    comptime NEXCLUDE: Int
    comptime OBS_DIM: Int
    comptime ACTION_DIM: Int
    comptime TIMESTEP: Float64
    comptime CTRL_MIN: Float64
    comptime CTRL_MAX: Float64

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

    # === CPU: state hooks (fields-native; G2) ===
    @staticmethod
    def reset_data[
        DTYPE: DType
    ](
        mut d: DataFields[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.MAX_CONTACTS,
            Self.NSITE,
            1,
        ],
    ):
        ...

    @staticmethod
    def extract_obs[
        DTYPE: DType
    ](
        d: DataFields[
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
        ...

    @staticmethod
    def enforce_limits[
        DTYPE: DType
    ](
        mut d: DataFields[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.MAX_CONTACTS,
            Self.NSITE,
            1,
        ],
    ):
        ...

    @staticmethod
    def apply_actions[
        DTYPE: DType
    ](
        mut d: DataFields[
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
        ...

    # === CPU: Float getters (can't use Float64 as comptime in traits) ===

    # === Fields-native model build (spec-direct; G4) ===
    @staticmethod
    def init_fields[
        DTYPE: DType, NMESHV: Int = 0
    ](
        ctx: DeviceContext,
        mut mf: ModelFields[
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
        """Build the ModelFields record tensors + fields-native invweight0
        and upload. Implemented spec-direct by `ModelDefFromXML`
        (parse_xml_full -> fields_build.build_model_fields_from_flat)."""
        ...

    # === GPU: Joints/Actuators kernel delegates ===
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
        ...

    @staticmethod
    def enforce_limits_kernel_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[DTYPE]) raises:
        ...

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
        ...

    # === GPU inline: Per-env methods (called from inside GPU kernels) ===
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
        ...

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
        ...

    @staticmethod
    def render_ground_geoms(
        mut renderer: Renderer3D,
        torso_x: Float64,
        follow: Bool,
        visual_radius_scale: Float64,
    ) raises:
        ...

    @staticmethod
    def render_body_geoms(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
        visual_radius_scale: Float64,
    ) raises:
        ...

    @staticmethod
    def setup_lights() raises -> List[Light]:
        ...

    @staticmethod
    def setup_cameras(width: Int, height: Int) raises -> List[Camera3D]:
        ...

    @staticmethod
    def setup_camera_modes() raises -> List[Int]:
        ...

    @staticmethod
    def get_skybox_colors() -> List[Float64]:
        ...

    @staticmethod
    def get_checker_colors() -> List[Float64]:
        ...

    @staticmethod
    def get_ground_rgba() -> List[Float64]:
        ...

    @staticmethod
    def get_visual_settings() -> List[Float64]:
        """Return visual settings: [znear, fogstart, fogend, shadowsize,
        headlight_r, headlight_g, headlight_b, has_headlight].
        Empty list = use defaults."""
        ...

    @staticmethod
    def render_sites(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        ...


# The spec-based `ModelDef` compositor struct (Bodies/Joints/Geoms/Actuators
# cascade) was deleted at the G2 fields sunset: it had ZERO instantiation
# sites — every env model is a `ModelDefFromXML`, the trait's only
# implementer. The `ModelDefLike` trait above is the surviving surface.
