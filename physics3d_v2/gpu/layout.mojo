"""Physics3D v2 GPU Layout - Compile-time layout computation.

Provides a struct for computing offsets and accessing state buffer fields.
"""

from .constants import (
    STATE_SIZE,
    QPOS_OFFSET,
    QPOS_SIZE,
    QVEL_OFFSET,
    QVEL_SIZE,
    QACC_OFFSET,
    QACC_SIZE,
    QFRC_OFFSET,
    QFRC_SIZE,
    XPOS_OFFSET,
    XPOS_SIZE,
    CONTACT_OFFSET,
    CONTACT_SIZE,
)


struct Physics3DV2Layout:
    """Compile-time layout for single-body physics buffers.

    The state buffer has shape [BATCH, STATE_SIZE] where STATE_SIZE=36.
    This struct provides utility methods for computing field offsets.

    Usage in GPU kernel:
        var x = state[env, Physics3DV2Layout.x_offset()]
        state[env, Physics3DV2Layout.vz_offset()] = new_vz
    """

    # Section sizes (exposed for external use)
    comptime QPOS_SIZE: Int = QPOS_SIZE
    comptime QVEL_SIZE: Int = QVEL_SIZE
    comptime QACC_SIZE: Int = QACC_SIZE
    comptime QFRC_SIZE: Int = QFRC_SIZE
    comptime XPOS_SIZE: Int = XPOS_SIZE
    comptime CONTACT_SIZE: Int = CONTACT_SIZE
    comptime STATE_SIZE: Int = STATE_SIZE

    # =========================================================================
    # qpos accessors (position + quaternion)
    # =========================================================================

    @always_inline
    @staticmethod
    fn x_offset() -> Int:
        return QPOS_OFFSET + 0

    @always_inline
    @staticmethod
    fn y_offset() -> Int:
        return QPOS_OFFSET + 1

    @always_inline
    @staticmethod
    fn z_offset() -> Int:
        return QPOS_OFFSET + 2

    @always_inline
    @staticmethod
    fn qx_offset() -> Int:
        return QPOS_OFFSET + 3

    @always_inline
    @staticmethod
    fn qy_offset() -> Int:
        return QPOS_OFFSET + 4

    @always_inline
    @staticmethod
    fn qz_offset() -> Int:
        return QPOS_OFFSET + 5

    @always_inline
    @staticmethod
    fn qw_offset() -> Int:
        return QPOS_OFFSET + 6

    # =========================================================================
    # qvel accessors (linear + angular velocity)
    # =========================================================================

    @always_inline
    @staticmethod
    fn vx_offset() -> Int:
        return QVEL_OFFSET + 0

    @always_inline
    @staticmethod
    fn vy_offset() -> Int:
        return QVEL_OFFSET + 1

    @always_inline
    @staticmethod
    fn vz_offset() -> Int:
        return QVEL_OFFSET + 2

    @always_inline
    @staticmethod
    fn wx_offset() -> Int:
        return QVEL_OFFSET + 3

    @always_inline
    @staticmethod
    fn wy_offset() -> Int:
        return QVEL_OFFSET + 4

    @always_inline
    @staticmethod
    fn wz_offset() -> Int:
        return QVEL_OFFSET + 5

    # =========================================================================
    # qacc accessors (linear + angular acceleration)
    # =========================================================================

    @always_inline
    @staticmethod
    fn ax_offset() -> Int:
        return QACC_OFFSET + 0

    @always_inline
    @staticmethod
    fn ay_offset() -> Int:
        return QACC_OFFSET + 1

    @always_inline
    @staticmethod
    fn az_offset() -> Int:
        return QACC_OFFSET + 2

    @always_inline
    @staticmethod
    fn alpha_x_offset() -> Int:
        return QACC_OFFSET + 3

    @always_inline
    @staticmethod
    fn alpha_y_offset() -> Int:
        return QACC_OFFSET + 4

    @always_inline
    @staticmethod
    fn alpha_z_offset() -> Int:
        return QACC_OFFSET + 5

    # =========================================================================
    # qfrc accessors (applied force + torque)
    # =========================================================================

    @always_inline
    @staticmethod
    fn fx_offset() -> Int:
        return QFRC_OFFSET + 0

    @always_inline
    @staticmethod
    fn fy_offset() -> Int:
        return QFRC_OFFSET + 1

    @always_inline
    @staticmethod
    fn fz_offset() -> Int:
        return QFRC_OFFSET + 2

    @always_inline
    @staticmethod
    fn tau_x_offset() -> Int:
        return QFRC_OFFSET + 3

    @always_inline
    @staticmethod
    fn tau_y_offset() -> Int:
        return QFRC_OFFSET + 4

    @always_inline
    @staticmethod
    fn tau_z_offset() -> Int:
        return QFRC_OFFSET + 5

    # =========================================================================
    # xpos accessors (world-frame position)
    # =========================================================================

    @always_inline
    @staticmethod
    fn xpos_x_offset() -> Int:
        return XPOS_OFFSET + 0

    @always_inline
    @staticmethod
    fn xpos_y_offset() -> Int:
        return XPOS_OFFSET + 1

    @always_inline
    @staticmethod
    fn xpos_z_offset() -> Int:
        return XPOS_OFFSET + 2

    # =========================================================================
    # contact accessors
    # =========================================================================

    @always_inline
    @staticmethod
    fn contact_active_offset() -> Int:
        return CONTACT_OFFSET + 0

    @always_inline
    @staticmethod
    fn contact_depth_offset() -> Int:
        return CONTACT_OFFSET + 1

    @always_inline
    @staticmethod
    fn contact_nx_offset() -> Int:
        return CONTACT_OFFSET + 2

    @always_inline
    @staticmethod
    fn contact_ny_offset() -> Int:
        return CONTACT_OFFSET + 3

    @always_inline
    @staticmethod
    fn contact_nz_offset() -> Int:
        return CONTACT_OFFSET + 4

    @always_inline
    @staticmethod
    fn contact_px_offset() -> Int:
        return CONTACT_OFFSET + 5

    @always_inline
    @staticmethod
    fn contact_py_offset() -> Int:
        return CONTACT_OFFSET + 6

    @always_inline
    @staticmethod
    fn contact_pz_offset() -> Int:
        return CONTACT_OFFSET + 7
