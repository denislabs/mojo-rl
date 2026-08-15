"""`ModelDefLike` — the trait every physics3d model definition implements.

Declares the compile-time dimensions, the CPU state hooks (fields-native:
`reset_data` / `extract_obs` / `enforce_limits` / `apply_actions` over
`Data`), the spec-direct fields model build (`init_fields`), the GPU
kernel delegates, and the render hooks. Sole implementer: `ModelDefFromXML`
(the spec-based `ModelDef` compositor was deleted at the G2 fields sunset;
the legacy CPU `Model`/`Data` build at G4).
"""

from mojo_rl.render import Renderer3D, Light, Camera3D
from mojo_rl.math3d import Vec3 as _Vec3G, Quat as _QuatG

from ..fields import Model, Data, SpecFields
from ..gpu.constants import MODEL_ACTUATOR_SIZE, MODEL_ACT_TENDON_SIZE

# GPU imports
from max.gpu.host import DeviceContext, DeviceBuffer
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
    # MuJoCo `m->na`: ACTIVATION variables, not `nu`.
    comptime NA: Int
    comptime NA_F: Int
    comptime NEXCLUDE: Int
    # `<contact><pair>` count — sizes the predefined-pair table.
    #
    # ⚠ ON THE TRAIT FOR THE SAME REASON `MAX_CONDIM` IS (see below). The env
    # forwards these dims into the integrator, and a dimension the trait does
    # not expose is one the env can quietly forget: the model would still
    # build, the pair table would be empty, and the declared collisions would
    # simply not happen. Declaring it here makes omitting it a compile error.
    comptime NPAIR: Int
    # Largest `condim` in the model — sizes the PYRAMIDAL edge list at
    # `2*(MAX_CONDIM-1)` rows per contact.
    #
    # ⚠ ON THE TRAIT ON PURPOSE, not just on `ModelDefFromXML`. `Phyics3dEnv`
    # reads it through this trait to forward into the integrator, and until
    # 2026-08-03 it did not forward it at all — so every env silently ran the
    # pyramidal builder at condim 3 regardless of its model, and quadruped
    # `fetch` and dog both lost their condim-6 friction rows. Declaring it here
    # is what makes omitting it a compile error rather than a silent downgrade.
    comptime MAX_CONDIM: Int
    # `<option noslip_iterations>`; 0 disables `mj_solNoSlip`. Same reasoning —
    # the env has to forward it, so the trait has to expose it.
    comptime NOSLIP_ITER: Int
    comptime OBS_DIM: Int
    comptime ACTION_DIM: Int
    comptime TIMESTEP: Float64
    # ⚠⚠ THESE ARE A SUMMARY, NOT THE CLAMP, AND THEY LIE ON SOME MODELS.
    # A single pair for the whole model, read by
    # `_xml_default_motor_ctrlrange` from a ROOT `<default><motor ctrlrange>`
    # — only that, and only a `<motor>` tag. A model that sets its ranges per
    # actuator or per default CLASS falls back to (-1, 1) here while
    # `apply_actions` correctly clamps each actuator to its own range.
    #
    # MEASURED against dm_control's `action_spec`:
    #   reach_site_features  advertised (-1, 1); real +/-0.6283 x3,
    #                        +/-0.8378 x3, +/-5.0 x3  -> every one wrong
    #   quadruped walk       advertised (-1, 1); real lo in [-1, -0.8],
    #                        hi in [0.8, 1.1]        -> already live
    #   dog / humanoid / walker / cheetah / finger    -> uniform +/-1, fine
    #
    # Use `ctrl_min_at` / `ctrl_max_at` for anything that matters. These two
    # stay because `BoxContinuousActionEnv.action_low/high` are scalars by
    # contract, and changing THAT changes the action scaling of every shipped
    # env — a behaviour change that needs its own before/after, not a
    # side-effect of a bug fix.
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
        ...

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
        ...

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
        ...

    # === Actuation records (phase 1a.2) ===
    #
    # `NACT` is MuJoCo's `m->nu`; `NACT_F`/`NTEN_F` are the same numbers
    # floored at 1, which is the STORAGE capacity. Both are needed and they
    # are not interchangeable: `build_spec_fields` checks the real count, and
    # a zero-extent tensor aborts at bind.
    comptime NACT: Int
    comptime NACT_F: Int
    comptime NTEN_F: Int
    comptime NKEY: Int

    @staticmethod
    def init_spec_fields[
        DTYPE: DType
    ](
        ctx: DeviceContext,
        mut sf: SpecFields[
            DTYPE, Self.NACT, Self.NTEN_F, Self.NQ, Self.NV, Self.NKEY
        ],
    ) raises:
        """Build + upload the actuation record tensors (`SpecFields`), the
        runtime replacement for the comptime `_acd` actuator arrays."""
        ...

    @staticmethod
    def apply_actions[
        DTYPE: DType
    ](
        sf: SpecFields[
            DTYPE, Self.NACT, Self.NTEN_F, Self.NQ, Self.NV, Self.NKEY
        ],
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
        ...

    @staticmethod
    def ctrl_min_at(i: Int) -> Float64:
        """Lower `ctrlrange` bound of actuator `i` — MuJoCo's
        `actuator_ctrlrange[i][0]`, and what `apply_actions` actually clamps
        against.

        The per-actuator answer that `CTRL_MIN` cannot give. Resolved through
        the element attribute, then `class=`, then the root default, so a
        model that keeps its ranges in a default class (quadruped does, and so
        does the Jaco manipulation task) reports them correctly here.
        """
        ...

    @staticmethod
    def ctrl_max_at(i: Int) -> Float64:
        """Upper `ctrlrange` bound of actuator `i`. See `ctrl_min_at`."""
        ...

    # === CPU: Float getters (can't use Float64 as comptime in traits) ===

    # === Fields-native model build (spec-direct; G4) ===
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
            Self.NPAIR,
        ],
    ) raises:
        """Build the Model record tensors + fields-native invweight0
        and upload. Implemented spec-direct by `ModelDefFromXML`
        (parse_xml_full -> fields_build.build_model_fields_from_flat)."""
        ...

    # === GPU: kernel delegates (per-field tensors; G5) ===
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
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, Self.NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, Self.NV), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, Self.NA_F), MutAnyOrigin
        ],
        acts: LayoutTensor[
            DTYPE,
            Layout.row_major(Self.NACT_F * MODEL_ACTUATOR_SIZE),
            MutAnyOrigin,
        ],
        act_tendons: LayoutTensor[
            DTYPE,
            Layout.row_major(Self.NTEN_F * MODEL_ACT_TENDON_SIZE),
            MutAnyOrigin,
        ],
    ) raises:
        """⚠ `acts`/`act_tendons` added 2026-08-15 with phase 1a.3. Every
        actuator value used to be a comptime literal baked into a fully
        unrolled loop; they are now loads from the SAME `SpecFields` records
        the CPU `apply_actions` reads, so the two targets cannot drift.

        ⚠ `qpos`/`qvel`/`act` added 2026-08-07 with the blocker-G fix. The GPU
        actuator path used to apply `gear * ctrl` to ONE dof; it now mirrors
        `apply_actions` term for term, which means walking the transmission
        triples and — for position servos and fixed-tendon springs — reading
        the joint state. See `ModelDefFromXML.apply_actions_kernel_gpu`."""
        ...

    # === GPU inline: Per-env methods (called from inside GPU kernels) ===
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
        ...

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
    def render_spatial_tendons(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Draw `<spatial>` tendons. Defaults to nothing, since most models
        have none and a model definition predating this should still build."""
        pass

    @staticmethod
    def render_skin(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Deform and draw a MuJoCo `<skin>`. Defaults to nothing.

        A DEFAULT for the same reason the tendons have one: exactly one ported
        model (dog) declares a skin, and a model definition written before this
        existed must still conform. `ModelDefFromXML` overrides it and compiles
        the body away when its XML has no `<skin>`, so the cost to everyone
        else is zero.
        """
        pass

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
    def get_camera_target_bodies() -> List[Int]:
        """Body each camera aims at (mode="targetbody"), or -1 for none.

        Defaults to empty: a model definition with no cameras, or one written
        before targetbody was honoured, simply gets no re-aiming."""
        return List[Int]()

    @staticmethod
    def get_skybox_colors() -> List[Float64]:
        ...

    @staticmethod
    def get_skybox_mark() -> List[Float64]:
        """[kind, r, g, b, density] for the skybox `mark`, or empty for none.

        Defaults to empty so a model definition that predates the starfield
        keeps compiling — the renderer treats an empty list, and any kind other
        than 3 (`random`), as "no stars".
        """
        return List[Float64]()

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
