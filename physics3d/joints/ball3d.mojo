"""3D Ball (Spherical) Joint.

A ball joint constrains two bodies to keep their anchor points coincident
while allowing rotation on all axes. It has 3 degrees of freedom.

Used for hip joints in humanoid-like robots.
"""

from math import sqrt
from layout import LayoutTensor, Layout

from ..constants import (
    dtype,
    BODY_STATE_SIZE_3D,
    JOINT_DATA_SIZE_3D,
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
    IDX_INV_MASS,
    IDX_IXX,
    IDX_IYY,
    IDX_IZZ,
    JOINT3D_TYPE,
    JOINT3D_BODY_A,
    JOINT3D_BODY_B,
    JOINT3D_ANCHOR_AX,
    JOINT3D_ANCHOR_AY,
    JOINT3D_ANCHOR_AZ,
    JOINT3D_ANCHOR_BX,
    JOINT3D_ANCHOR_BY,
    JOINT3D_ANCHOR_BZ,
    JOINT3D_IMPULSE_X,
    JOINT3D_IMPULSE_Y,
    JOINT3D_IMPULSE_Z,
    JOINT_BALL,
)

from math3d import Vec3, Quat


struct Ball3D:
    """3D Ball (Spherical) Joint Constraint Solver.

    Constrains two bodies to keep their anchor points coincident (3 linear constraints).
    Allows free rotation on all axes (3 rotational DOF).
    """

    # =========================================================================
    # Joint Initialization
    # =========================================================================

    @staticmethod
    fn init_joint[
        BATCH: Int,
        STATE_SIZE: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        joint_idx: Int,
        body_a: Int,
        body_b: Int,
        anchor_a: Vec3,  # Local anchor on body A
        anchor_b: Vec3,  # Local anchor on body B
    ):
        """Initialize a ball joint between two bodies."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        # Joint type
        state[env, joint_off + JOINT3D_TYPE] = Scalar[dtype](JOINT_BALL)

        # Body indices
        state[env, joint_off + JOINT3D_BODY_A] = Scalar[dtype](body_a)
        state[env, joint_off + JOINT3D_BODY_B] = Scalar[dtype](body_b)

        # Local anchors
        state[env, joint_off + JOINT3D_ANCHOR_AX] = Scalar[dtype](anchor_a.x)
        state[env, joint_off + JOINT3D_ANCHOR_AY] = Scalar[dtype](anchor_a.y)
        state[env, joint_off + JOINT3D_ANCHOR_AZ] = Scalar[dtype](anchor_a.z)
        state[env, joint_off + JOINT3D_ANCHOR_BX] = Scalar[dtype](anchor_b.x)
        state[env, joint_off + JOINT3D_ANCHOR_BY] = Scalar[dtype](anchor_b.y)
        state[env, joint_off + JOINT3D_ANCHOR_BZ] = Scalar[dtype](anchor_b.z)

        # Clear accumulated impulses
        state[env, joint_off + JOINT3D_IMPULSE_X] = Scalar[dtype](0.0)
        state[env, joint_off + JOINT3D_IMPULSE_Y] = Scalar[dtype](0.0)
        state[env, joint_off + JOINT3D_IMPULSE_Z] = Scalar[dtype](0.0)

    # =========================================================================
    # Velocity Constraint Solving
    # =========================================================================

    @staticmethod
    fn solve_velocity[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        joint_idx: Int,
        dt: Scalar[dtype],
    ):
        """Solve velocity constraints for ball joint.

        Only constrains anchor point velocities (3 linear constraints).
        """
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var joint_type = Int(state[env, joint_off + JOINT3D_TYPE])
        if joint_type != JOINT_BALL:
            return

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get mass properties
        var inv_ma = Float64(state[env, body_a_off + IDX_INV_MASS])
        var inv_mb = Float64(state[env, body_b_off + IDX_INV_MASS])

        # Get orientations
        var qa = Quat(
            Float64(state[env, body_a_off + IDX_QW]),
            Float64(state[env, body_a_off + IDX_QX]),
            Float64(state[env, body_a_off + IDX_QY]),
            Float64(state[env, body_a_off + IDX_QZ]),
        )
        var qb = Quat(
            Float64(state[env, body_b_off + IDX_QW]),
            Float64(state[env, body_b_off + IDX_QX]),
            Float64(state[env, body_b_off + IDX_QY]),
            Float64(state[env, body_b_off + IDX_QZ]),
        )

        # Get velocities
        var va = Vec3(
            Float64(state[env, body_a_off + IDX_VX]),
            Float64(state[env, body_a_off + IDX_VY]),
            Float64(state[env, body_a_off + IDX_VZ]),
        )
        var vb = Vec3(
            Float64(state[env, body_b_off + IDX_VX]),
            Float64(state[env, body_b_off + IDX_VY]),
            Float64(state[env, body_b_off + IDX_VZ]),
        )
        var wa = Vec3(
            Float64(state[env, body_a_off + IDX_WX]),
            Float64(state[env, body_a_off + IDX_WY]),
            Float64(state[env, body_a_off + IDX_WZ]),
        )
        var wb = Vec3(
            Float64(state[env, body_b_off + IDX_WX]),
            Float64(state[env, body_b_off + IDX_WY]),
            Float64(state[env, body_b_off + IDX_WZ]),
        )

        # Local anchors
        var anchor_a_local = Vec3(
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AX]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AY]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AZ]),
        )
        var anchor_b_local = Vec3(
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BX]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BY]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BZ]),
        )

        # Transform anchors to world frame
        var ra = qa.rotate_vec(anchor_a_local)
        var rb = qb.rotate_vec(anchor_b_local)

        # Velocity at anchor points
        var va_anchor = va + wa.cross(ra)
        var vb_anchor = vb + wb.cross(rb)

        # Relative velocity
        var cdot = vb_anchor - va_anchor

        # Effective mass (simplified scalar)
        var inv_ia_avg = (
            Float64(1.0 / (state[env, body_a_off + IDX_IXX] + 1e-10))
            + Float64(1.0 / (state[env, body_a_off + IDX_IYY] + 1e-10))
            + Float64(1.0 / (state[env, body_a_off + IDX_IZZ] + 1e-10))
        ) / 3.0

        var inv_ib_avg = (
            Float64(1.0 / (state[env, body_b_off + IDX_IXX] + 1e-10))
            + Float64(1.0 / (state[env, body_b_off + IDX_IYY] + 1e-10))
            + Float64(1.0 / (state[env, body_b_off + IDX_IZZ] + 1e-10))
        ) / 3.0

        var eff_mass = inv_ma + inv_mb + ra.length_squared() * inv_ia_avg + rb.length_squared() * inv_ib_avg

        if eff_mass < 1e-10:
            eff_mass = 1e-10

        # Compute and apply impulse
        var inv_eff_mass = 1.0 / eff_mass
        var impulse = cdot * (-inv_eff_mass)

        # Apply linear impulse
        var new_va = va - impulse * inv_ma
        var new_vb = vb + impulse * inv_mb

        # Apply angular impulse
        var delta_wa = ra.cross(impulse) * (-inv_ia_avg)
        var delta_wb = rb.cross(impulse) * inv_ib_avg

        var new_wa = wa + delta_wa
        var new_wb = wb + delta_wb

        # Write back
        state[env, body_a_off + IDX_VX] = Scalar[dtype](new_va.x)
        state[env, body_a_off + IDX_VY] = Scalar[dtype](new_va.y)
        state[env, body_a_off + IDX_VZ] = Scalar[dtype](new_va.z)
        state[env, body_a_off + IDX_WX] = Scalar[dtype](new_wa.x)
        state[env, body_a_off + IDX_WY] = Scalar[dtype](new_wa.y)
        state[env, body_a_off + IDX_WZ] = Scalar[dtype](new_wa.z)

        state[env, body_b_off + IDX_VX] = Scalar[dtype](new_vb.x)
        state[env, body_b_off + IDX_VY] = Scalar[dtype](new_vb.y)
        state[env, body_b_off + IDX_VZ] = Scalar[dtype](new_vb.z)
        state[env, body_b_off + IDX_WX] = Scalar[dtype](new_wb.x)
        state[env, body_b_off + IDX_WY] = Scalar[dtype](new_wb.y)
        state[env, body_b_off + IDX_WZ] = Scalar[dtype](new_wb.z)

    # =========================================================================
    # Position Constraint Solving
    # =========================================================================

    @staticmethod
    fn solve_position[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        joint_idx: Int,
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """Solve position constraints for ball joint."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var joint_type = Int(state[env, joint_off + JOINT3D_TYPE])
        if joint_type != JOINT_BALL:
            return

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get positions and orientations
        var pa = Vec3(
            Float64(state[env, body_a_off + IDX_PX]),
            Float64(state[env, body_a_off + IDX_PY]),
            Float64(state[env, body_a_off + IDX_PZ]),
        )
        var pb = Vec3(
            Float64(state[env, body_b_off + IDX_PX]),
            Float64(state[env, body_b_off + IDX_PY]),
            Float64(state[env, body_b_off + IDX_PZ]),
        )

        var qa = Quat(
            Float64(state[env, body_a_off + IDX_QW]),
            Float64(state[env, body_a_off + IDX_QX]),
            Float64(state[env, body_a_off + IDX_QY]),
            Float64(state[env, body_a_off + IDX_QZ]),
        )
        var qb = Quat(
            Float64(state[env, body_b_off + IDX_QW]),
            Float64(state[env, body_b_off + IDX_QX]),
            Float64(state[env, body_b_off + IDX_QY]),
            Float64(state[env, body_b_off + IDX_QZ]),
        )

        # Local anchors
        var anchor_a_local = Vec3(
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AX]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AY]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AZ]),
        )
        var anchor_b_local = Vec3(
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BX]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BY]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BZ]),
        )

        # World anchors
        var anchor_a_world = pa + qa.rotate_vec(anchor_a_local)
        var anchor_b_world = pb + qb.rotate_vec(anchor_b_local)

        # Position error
        var error = anchor_b_world - anchor_a_world
        var error_mag = error.length()

        if error_mag < Float64(slop):
            return

        # Get mass properties
        var inv_ma = Float64(state[env, body_a_off + IDX_INV_MASS])
        var inv_mb = Float64(state[env, body_b_off + IDX_INV_MASS])

        var total_inv_mass = inv_ma + inv_mb
        if total_inv_mass < 1e-10:
            return

        # Position correction
        var correction = error * Float64(baumgarte)
        var delta_a = correction * (inv_ma / total_inv_mass)
        var delta_b = correction * (-inv_mb / total_inv_mass)

        state[env, body_a_off + IDX_PX] = Scalar[dtype](pa.x - delta_a.x)
        state[env, body_a_off + IDX_PY] = Scalar[dtype](pa.y - delta_a.y)
        state[env, body_a_off + IDX_PZ] = Scalar[dtype](pa.z - delta_a.z)

        state[env, body_b_off + IDX_PX] = Scalar[dtype](pb.x + delta_b.x)
        state[env, body_b_off + IDX_PY] = Scalar[dtype](pb.y + delta_b.y)
        state[env, body_b_off + IDX_PZ] = Scalar[dtype](pb.z + delta_b.z)
