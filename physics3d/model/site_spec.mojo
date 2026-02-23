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
from gpu.host import HostBuffer

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

    comptime BODY_IDX: Int   # Body index (>=1, 0=worldbody)
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
        pos_x: x-coordinate of site in body-local frame.
        pos_y: y-coordinate of site in body-local frame.
        pos_z: z-coordinate of site in body-local frame.
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
        """Populate model.site_body and model.site_pos from compile-time specs."""
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

        @parameter
        for i in range(Self.N):
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

        @parameter
        for i in range(Self.N):
            comptime SS = Self.site_types[i]
            comptime base = model_site_offset[
                NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON
            ](i)
            buffer[base + SITE_IDX_BODY] = Scalar[DTYPE](SS.BODY_IDX)
            buffer[base + SITE_IDX_POS_X] = Scalar[DTYPE](SS.POS_X)
            buffer[base + SITE_IDX_POS_Y] = Scalar[DTYPE](SS.POS_Y)
            buffer[base + SITE_IDX_POS_Z] = Scalar[DTYPE](SS.POS_Z)
