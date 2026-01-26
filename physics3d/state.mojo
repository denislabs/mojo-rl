"""PhysicsState3D - Helper for accessing 3D body state in flat buffers.

Provides convenient methods for reading and writing 3D body state
from flat tensor buffers used in GPU kernels.
"""

from math import sqrt
from layout import LayoutTensor, Layout

from .constants import (
    dtype,
    BODY_STATE_SIZE_3D,
    IDX_PX,
    IDX_PY,
    IDX_PZ,
    IDX_QW,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
    IDX_VX,
    IDX_VY,
    IDX_VZ,
    IDX_WX,
    IDX_WY,
    IDX_WZ,
    IDX_FX,
    IDX_FY,
    IDX_FZ,
    IDX_TX,
    IDX_TY,
    IDX_TZ,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_IXX,
    IDX_IYY,
    IDX_IZZ,
    IDX_SHAPE_3D,
    IDX_BODY_TYPE,
    BODY_DYNAMIC,
    BODY_STATIC,
)

from math3d import Vec3, Quat


struct PhysicsState3D[
    BATCH: Int,
    NUM_BODIES: Int,
    STATE_SIZE: Int,
    BODIES_OFFSET: Int,
]:
    """Helper struct for accessing 3D body state from flat buffers.

    Parameters:
        BATCH: Number of parallel environments.
        NUM_BODIES: Number of bodies per environment.
        STATE_SIZE: Total state size per environment.
        BODIES_OFFSET: Offset to body data within state.

    Memory layout:
        state[env, BODIES_OFFSET + body * BODY_STATE_SIZE_3D + field]
    """

    # =========================================================================
    # Position Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn get_position(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Vec3:
        """Get body position as Vec3."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        return Vec3(
            Float64(state[env, off + IDX_PX]),
            Float64(state[env, off + IDX_PY]),
            Float64(state[env, off + IDX_PZ]),
        )

    @staticmethod
    @always_inline
    fn set_position(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
        pos: Vec3,
    ):
        """Set body position from Vec3."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        state[env, off + IDX_PX] = Scalar[dtype](pos.x)
        state[env, off + IDX_PY] = Scalar[dtype](pos.y)
        state[env, off + IDX_PZ] = Scalar[dtype](pos.z)

    # =========================================================================
    # Orientation Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn get_orientation(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Quat:
        """Get body orientation as Quat."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        return Quat(
            Float64(state[env, off + IDX_QW]),
            Float64(state[env, off + IDX_QX]),
            Float64(state[env, off + IDX_QY]),
            Float64(state[env, off + IDX_QZ]),
        )

    @staticmethod
    @always_inline
    fn set_orientation(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
        quat: Quat,
    ):
        """Set body orientation from Quat (will be normalized)."""
        var q = quat.normalized()
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        state[env, off + IDX_QW] = Scalar[dtype](q.w)
        state[env, off + IDX_QX] = Scalar[dtype](q.x)
        state[env, off + IDX_QY] = Scalar[dtype](q.y)
        state[env, off + IDX_QZ] = Scalar[dtype](q.z)

    @staticmethod
    @always_inline
    fn normalize_orientation(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ):
        """Normalize the body's orientation quaternion in place."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        var w = state[env, off + IDX_QW]
        var x = state[env, off + IDX_QX]
        var y = state[env, off + IDX_QY]
        var z = state[env, off + IDX_QZ]

        var len_sq = w * w + x * x + y * y + z * z
        if len_sq > Scalar[dtype](1e-10):
            var inv_len = Scalar[dtype](1.0) / sqrt(len_sq)
            state[env, off + IDX_QW] = w * inv_len
            state[env, off + IDX_QX] = x * inv_len
            state[env, off + IDX_QY] = y * inv_len
            state[env, off + IDX_QZ] = z * inv_len

    # =========================================================================
    # Velocity Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn get_linear_velocity(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Vec3:
        """Get body linear velocity as Vec3."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        return Vec3(
            Float64(state[env, off + IDX_VX]),
            Float64(state[env, off + IDX_VY]),
            Float64(state[env, off + IDX_VZ]),
        )

    @staticmethod
    @always_inline
    fn set_linear_velocity(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
        vel: Vec3,
    ):
        """Set body linear velocity from Vec3."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        state[env, off + IDX_VX] = Scalar[dtype](vel.x)
        state[env, off + IDX_VY] = Scalar[dtype](vel.y)
        state[env, off + IDX_VZ] = Scalar[dtype](vel.z)

    @staticmethod
    @always_inline
    fn get_angular_velocity(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Vec3:
        """Get body angular velocity as Vec3."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        return Vec3(
            Float64(state[env, off + IDX_WX]),
            Float64(state[env, off + IDX_WY]),
            Float64(state[env, off + IDX_WZ]),
        )

    @staticmethod
    @always_inline
    fn set_angular_velocity(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
        omega: Vec3,
    ):
        """Set body angular velocity from Vec3."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        state[env, off + IDX_WX] = Scalar[dtype](omega.x)
        state[env, off + IDX_WY] = Scalar[dtype](omega.y)
        state[env, off + IDX_WZ] = Scalar[dtype](omega.z)

    # =========================================================================
    # Force/Torque Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn apply_force(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
        force: Vec3,
    ):
        """Add force to body force accumulator."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        state[env, off + IDX_FX] = state[env, off + IDX_FX] + Scalar[dtype](force.x)
        state[env, off + IDX_FY] = state[env, off + IDX_FY] + Scalar[dtype](force.y)
        state[env, off + IDX_FZ] = state[env, off + IDX_FZ] + Scalar[dtype](force.z)

    @staticmethod
    @always_inline
    fn apply_torque(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
        torque: Vec3,
    ):
        """Add torque to body torque accumulator."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        state[env, off + IDX_TX] = state[env, off + IDX_TX] + Scalar[dtype](torque.x)
        state[env, off + IDX_TY] = state[env, off + IDX_TY] + Scalar[dtype](torque.y)
        state[env, off + IDX_TZ] = state[env, off + IDX_TZ] + Scalar[dtype](torque.z)

    @staticmethod
    @always_inline
    fn apply_force_at_point(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
        force: Vec3,
        point: Vec3,  # World-space point where force is applied
    ):
        """Apply force at a world-space point (generates torque)."""
        var pos = Self.get_position(state, env, body)
        var r = point - pos  # Vector from CoM to application point

        # Apply force
        Self.apply_force(state, env, body, force)

        # Apply torque: τ = r × F
        Self.apply_torque(state, env, body, r.cross(force))

    @staticmethod
    @always_inline
    fn clear_forces(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ):
        """Clear force and torque accumulators."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        state[env, off + IDX_FX] = Scalar[dtype](0.0)
        state[env, off + IDX_FY] = Scalar[dtype](0.0)
        state[env, off + IDX_FZ] = Scalar[dtype](0.0)
        state[env, off + IDX_TX] = Scalar[dtype](0.0)
        state[env, off + IDX_TY] = Scalar[dtype](0.0)
        state[env, off + IDX_TZ] = Scalar[dtype](0.0)

    # =========================================================================
    # Mass Properties Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn get_mass(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Scalar[dtype]:
        """Get body mass."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        return state[env, off + IDX_MASS]

    @staticmethod
    @always_inline
    fn get_inv_mass(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Scalar[dtype]:
        """Get inverse mass (0 for static bodies)."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        return state[env, off + IDX_INV_MASS]

    @staticmethod
    @always_inline
    fn get_inertia_diagonal(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Vec3:
        """Get diagonal inertia tensor elements (local frame)."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        return Vec3(
            Float64(state[env, off + IDX_IXX]),
            Float64(state[env, off + IDX_IYY]),
            Float64(state[env, off + IDX_IZZ]),
        )

    @staticmethod
    @always_inline
    fn get_inv_inertia_diagonal(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Vec3:
        """Get inverse diagonal inertia tensor elements (local frame).

        Note: For a full 3D solver, you'd need world-space inverse inertia,
        which requires rotation by the current orientation.
        """
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        var ixx = state[env, off + IDX_IXX]
        var iyy = state[env, off + IDX_IYY]
        var izz = state[env, off + IDX_IZZ]

        # Compute inverse, handling zero case (static/infinite inertia)
        var inv_ixx = Scalar[dtype](0.0)
        var inv_iyy = Scalar[dtype](0.0)
        var inv_izz = Scalar[dtype](0.0)

        if ixx > Scalar[dtype](1e-10):
            inv_ixx = Scalar[dtype](1.0) / ixx
        if iyy > Scalar[dtype](1e-10):
            inv_iyy = Scalar[dtype](1.0) / iyy
        if izz > Scalar[dtype](1e-10):
            inv_izz = Scalar[dtype](1.0) / izz

        return Vec3(Float64(inv_ixx), Float64(inv_iyy), Float64(inv_izz))

    @staticmethod
    @always_inline
    fn set_mass_properties(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
        mass: Float64,
        inertia: Vec3,
    ):
        """Set mass and inertia. Pass mass=0 for static bodies."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D

        state[env, off + IDX_MASS] = Scalar[dtype](mass)
        if mass > 1e-10:
            state[env, off + IDX_INV_MASS] = Scalar[dtype](1.0 / mass)
        else:
            state[env, off + IDX_INV_MASS] = Scalar[dtype](0.0)

        state[env, off + IDX_IXX] = Scalar[dtype](inertia.x)
        state[env, off + IDX_IYY] = Scalar[dtype](inertia.y)
        state[env, off + IDX_IZZ] = Scalar[dtype](inertia.z)

    # =========================================================================
    # Body Type Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn get_body_type(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Int:
        """Get body type (BODY_DYNAMIC, BODY_KINEMATIC, BODY_STATIC)."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        return Int(state[env, off + IDX_BODY_TYPE])

    @staticmethod
    @always_inline
    fn is_dynamic(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Bool:
        """Check if body is dynamic."""
        return Self.get_body_type(state, env, body) == BODY_DYNAMIC

    @staticmethod
    @always_inline
    fn is_static(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Bool:
        """Check if body is static."""
        return Self.get_body_type(state, env, body) == BODY_STATIC

    # =========================================================================
    # Shape Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn get_shape_index(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
    ) -> Int:
        """Get body's shape index."""
        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        return Int(state[env, off + IDX_SHAPE_3D])

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    fn init_body(
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        body: Int,
        position: Vec3,
        orientation: Quat,
        mass: Float64,
        inertia: Vec3,
        body_type: Int = BODY_DYNAMIC,
        shape_idx: Int = 0,
    ):
        """Initialize a body with given properties."""
        Self.set_position(state, env, body, position)
        Self.set_orientation(state, env, body, orientation)
        Self.set_linear_velocity(state, env, body, Vec3.zero())
        Self.set_angular_velocity(state, env, body, Vec3.zero())
        Self.clear_forces(state, env, body)
        Self.set_mass_properties(state, env, body, mass, inertia)

        var off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
        state[env, off + IDX_SHAPE_3D] = Scalar[dtype](shape_idx)
        state[env, off + IDX_BODY_TYPE] = Scalar[dtype](body_type)
