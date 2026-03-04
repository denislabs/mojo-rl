"""SiteSpec trait and concrete site types for compile-time model definitions.

Sites are body-attached reference points with zero mass/inertia.
They participate in FK (get world position) but not dynamics. Used for:
  - Observation reference points (e.g. tip of pendulum)
  - Sensor attachment points
  - Reward computation

Usage:
    comptime TipSite = Site[body_idx=3, pos_z=0.6, name="tip"]
    comptime MySites = Sites[TipSite]

Example (InvertedDoublePendulum tip):
    comptime Tip = Site[body_idx=3, pos_z=0.6, name="tip"]
    comptime IDP_Sites = Sites[Tip]
"""

from std.builtin.variadics import Variadic
from std.gpu.host import HostBuffer
from render import Color, Renderer3D
from math3d import Vec3 as _Vec3G, Quat as _QuatG

comptime _RVec3 = _Vec3G[DType.float64]
comptime _RQuat = _QuatG[DType.float64]

from ..gpu.constants import (
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
    model_site_offset,
)
from ..types import Model, ConeType


# =============================================================================
# SiteSpec Trait
# =============================================================================


trait SiteSpec:
    """Compile-time site specification for physics3d model definitions.

    Sites are body-attached reference points with zero mass/inertia.
    They participate in FK to yield a world-space position (site_xpos).

    Fields:
        BODY_IDX: Index of the body the site is attached to (>=1).
        POS_X/Y/Z: Site position in body-local frame.
        NAME: Optional human-readable name.
    """

    comptime BODY_IDX: Int  # Body index (>=1, 0=worldbody)
    comptime POS_X: Float64  # Local position in body frame
    comptime POS_Y: Float64
    comptime POS_Z: Float64
    comptime NAME: String


# =============================================================================
# Site — concrete SiteSpec implementation
# =============================================================================


@fieldwise_init
struct Site[
    body_idx: Int,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    name: String = "site",
](SiteSpec):
    """Concrete site specification.

    Parameters:
        body_idx: Body index the site is attached to.
        pos_x: X-coordinate of site in body-local frame.
        pos_y: Y-coordinate of site in body-local frame.
        pos_z: Z-coordinate of site in body-local frame.
        name: Human-readable name for debugging/rendering.
    """

    comptime BODY_IDX: Int = Self.body_idx
    comptime POS_X: Float64 = Self.pos_x
    comptime POS_Y: Float64 = Self.pos_y
    comptime POS_Z: Float64 = Self.pos_z
    comptime NAME: String = Self.name


# =============================================================================
# SitesLike Trait
# =============================================================================


trait SitesLike:
    """Trait for compile-time site container types."""

    comptime N: Int  # Number of sites

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
        MAX_TENDON: Int,
        NSITE: Int,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ]
    ):
        """Populate model.site_body and model.site_pos from compile-time specs.
        """
        ...

    @staticmethod
    fn write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
        NJOINT: Int,
        NGEOM: Int = 0,
        NEQUALITY: Int = 0,
        NTENDON: Int = 0,
    ](buffer: HostBuffer[DTYPE]):
        """Write site data directly to GPU HostBuffer."""
        ...

    @staticmethod
    fn render_sites(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Draw all sites as small bright-green spheres (visual markers)."""
        ...


# =============================================================================
# _EmptySites — stub for environments without sites
# =============================================================================


@fieldwise_init
struct _EmptySites(SitesLike):
    """Empty site container (no sites). Used as default."""

    comptime N: Int = 0

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
        MAX_TENDON: Int,
        NSITE: Int,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ]
    ):
        pass

    @staticmethod
    fn write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
        NJOINT: Int,
        NGEOM: Int = 0,
        NEQUALITY: Int = 0,
        NTENDON: Int = 0,
    ](buffer: HostBuffer[DTYPE]):
        pass

    @staticmethod
    fn render_sites(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        pass


# =============================================================================
# Sites — variadic site list
# =============================================================================


@fieldwise_init
struct Sites[*S: SiteSpec](SitesLike):
    """Compile-time list of site specifications.

    Provides N (site count) and type-level access to each site.
    """

    comptime site_types = Variadic.types[T=SiteSpec, *Self.S]
    comptime N: Int = Variadic.size(Self.site_types)

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
        MAX_TENDON: Int,
        NSITE: Int,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ]
    ):
        """Populate model site arrays from compile-time SiteSpec list."""

        comptime for i in range(Self.N):
            comptime SS = Self.site_types[i]
            model.site_body[i] = SS.BODY_IDX
            model.site_pos[i * 3 + 0] = Scalar[DTYPE](SS.POS_X)
            model.site_pos[i * 3 + 1] = Scalar[DTYPE](SS.POS_Y)
            model.site_pos[i * 3 + 2] = Scalar[DTYPE](SS.POS_Z)

    @staticmethod
    fn write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
        NJOINT: Int,
        NGEOM: Int = 0,
        NEQUALITY: Int = 0,
        NTENDON: Int = 0,
    ](buffer: HostBuffer[DTYPE]):
        """Write site data directly to GPU HostBuffer.

        Sites are stored after tendons in the model buffer.
        Layout per site: [body_idx, pos_x, pos_y, pos_z]
        """

        comptime for i in range(Self.N):
            comptime SS = Self.site_types[i]
            comptime base = model_site_offset[
                NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON
            ](i)
            buffer[base + SITE_IDX_BODY] = Scalar[DTYPE](SS.BODY_IDX)
            buffer[base + SITE_IDX_POS_X] = Scalar[DTYPE](SS.POS_X)
            buffer[base + SITE_IDX_POS_Y] = Scalar[DTYPE](SS.POS_Y)
            buffer[base + SITE_IDX_POS_Z] = Scalar[DTYPE](SS.POS_Z)

    @staticmethod
    fn render_sites(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Draw all sites as small bright-green spheres (visual markers).

        Site world position = body_pos + body_quat.rotate(site_local_pos).
        Uses radius=0.01m and bright green color to distinguish from geoms.
        """

        comptime for i in range(Self.N):
            comptime SS = Self.site_types[i]
            var body_pos = positions[SS.BODY_IDX]
            var body_quat = quaternions[SS.BODY_IDX]
            var site_world_pos: _RVec3

            comptime if SS.POS_X == 0.0 and SS.POS_Y == 0.0 and SS.POS_Z == 0.0:
                site_world_pos = body_pos
            else:
                var local_pos = _RVec3(SS.POS_X, SS.POS_Y, SS.POS_Z)
                site_world_pos = body_pos + body_quat.rotate_vec(local_pos)

            renderer.draw_sphere(
                center=site_world_pos,
                radius=0.01,
                color=Color(0, 255, 0, 255),
                shininess=Float32(0.9),
                specular=Float32(0.9),
                reflectance=Float32(0.0),
            )
