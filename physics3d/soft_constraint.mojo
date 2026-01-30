"""Soft Constraint Compliance (MuJoCo solref/solimp style).

This module implements MuJoCo-style soft constraints that replace hard
Baumgarte stabilization with a proper spring-damper constraint model.

Key equations:
    k = 1 / (timeconst² * dampratio²)  # Constraint stiffness
    b = 2 / timeconst                   # Constraint damping
    compliance = 1 / (k*dt² + b*dt)     # Constraint softness
    softened_eff_mass = eff_mass / (1 + compliance * eff_mass)

This approach prevents numerical instability when high torques would otherwise
cause joint separation, by allowing controlled "give" in the constraints.

GPU support: All functions use only scalar operations, no Vec3/Quat struct
instantiation, following the physics2d/3d pattern.
"""

from math import sqrt
from layout import LayoutTensor, Layout

from .constants import (
    dtype,
    JOINT_DATA_SIZE_3D,
    JOINT3D_TIMECONST,
    JOINT3D_DAMPRATIO,
)


struct SoftConstraint:
    """Soft constraint compliance computation.

    Based on MuJoCo's solref/solimp model for constraint softening.
    Default parameters provide critical damping (dampratio=1.0).
    """

    # Default soft constraint parameters
    comptime DEFAULT_TIMECONST: Float64 = 0.02   # Time constant (s)
    comptime DEFAULT_DAMPRATIO: Float64 = 1.0    # Damping ratio (critical damping)

    @always_inline
    @staticmethod
    fn compute_compliance(
        timeconst: Scalar[dtype],
        dampratio: Scalar[dtype],
        dt: Scalar[dtype],
    ) -> Scalar[dtype]:
        """Compute constraint compliance from soft constraint parameters.

        Args:
            timeconst: Time constant (s), controls response speed
            dampratio: Damping ratio (1.0 = critical damping)
            dt: Timestep (s)

        Returns:
            Compliance value to soften constraint response
        """
        var eps = Scalar[dtype](1e-10)
        var one = Scalar[dtype](1.0)

        # Protect against zero values
        var tc = timeconst
        if tc < eps:
            tc = eps
        var dr = dampratio
        if dr < eps:
            dr = eps

        # k = 1 / (timeconst² * dampratio²)
        var k = one / (tc * tc * dr * dr)

        # b = 2 / timeconst
        var b = Scalar[dtype](2.0) / tc

        # compliance = 1 / (k*dt² + b*dt)
        var denom = k * dt * dt + b * dt
        if denom < eps:
            denom = eps

        return one / denom

    @always_inline
    @staticmethod
    fn compute_compliance_default(
        dt: Scalar[dtype],
    ) -> Scalar[dtype]:
        """Compute compliance using default parameters."""
        var timeconst = Scalar[dtype](Self.DEFAULT_TIMECONST)
        var dampratio = Scalar[dtype](Self.DEFAULT_DAMPRATIO)
        return Self.compute_compliance(timeconst, dampratio, dt)

    @always_inline
    @staticmethod
    fn soften_effective_mass(
        eff_mass: Scalar[dtype],
        compliance: Scalar[dtype],
    ) -> Scalar[dtype]:
        """Apply compliance to soften effective mass.

        Args:
            eff_mass: Original effective mass from constraint
            compliance: Compliance value from compute_compliance()

        Returns:
            Softened effective mass
        """
        var one = Scalar[dtype](1.0)
        var eps = Scalar[dtype](1e-10)

        # softened = eff_mass / (1 + compliance * eff_mass)
        var denom = one + compliance * eff_mass
        if denom < eps:
            denom = eps

        return eff_mass / denom

    @always_inline
    @staticmethod
    fn compute_softened_impulse(
        velocity_error: Scalar[dtype],
        eff_mass: Scalar[dtype],
        compliance: Scalar[dtype],
        position_error: Scalar[dtype],
        dt: Scalar[dtype],
        baumgarte: Scalar[dtype],
    ) -> Scalar[dtype]:
        """Compute impulse with soft constraint response.

        Combines velocity constraint with soft position correction.

        Args:
            velocity_error: Relative velocity along constraint (to be zeroed)
            eff_mass: Effective mass for constraint
            compliance: Compliance value from compute_compliance()
            position_error: Position error (drift)
            dt: Timestep (s)
            baumgarte: Baumgarte stabilization factor (0-1)

        Returns:
            Impulse to apply (softer than hard constraint)
        """
        var one = Scalar[dtype](1.0)
        var eps = Scalar[dtype](1e-10)

        # Soften effective mass
        var soft_eff_mass = Self.soften_effective_mass(eff_mass, compliance)

        # Velocity constraint with position bias (Baumgarte stabilization)
        var bias = baumgarte * position_error / dt
        var target_vel = -velocity_error - bias

        # Compute impulse with softened effective mass
        return soft_eff_mass * target_vel

    @always_inline
    @staticmethod
    fn get_joint_compliance_gpu[
        BATCH: Int,
        STATE_SIZE: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
        dt: Scalar[dtype],
    ) -> Scalar[dtype]:
        """Get compliance for a specific joint from state (GPU-compatible).

        Reads soft constraint parameters from joint data and computes compliance.
        Falls back to defaults if not set (zero values).
        """
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var timeconst = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_TIMECONST])
        var dampratio = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_DAMPRATIO])

        var eps = Scalar[dtype](1e-10)

        # Use defaults if not set
        if timeconst < eps:
            timeconst = Scalar[dtype](Self.DEFAULT_TIMECONST)
        if dampratio < eps:
            dampratio = Scalar[dtype](Self.DEFAULT_DAMPRATIO)

        return Self.compute_compliance(timeconst, dampratio, dt)


struct AxisProjectedMass:
    """Compute axis-projected effective mass for angular constraints.

    For hinge joints, the effective mass depends on how the inertia tensors
    project onto the joint axis. This provides more accurate constraint
    solving than using average scalar inertia.

    I_axis^-1 = axis^T * I_world^-1 * axis
    eff_mass = 1 / (I_a_axis^-1 + I_b_axis^-1 + 1/armature)
    """

    @always_inline
    @staticmethod
    fn compute_axis_inv_inertia(
        # World-space inverse inertia tensor (symmetric, 6 unique values)
        inv_i00: Scalar[dtype], inv_i01: Scalar[dtype], inv_i02: Scalar[dtype],
        inv_i11: Scalar[dtype], inv_i12: Scalar[dtype], inv_i22: Scalar[dtype],
        # World-space axis (normalized)
        axis_x: Scalar[dtype], axis_y: Scalar[dtype], axis_z: Scalar[dtype],
    ) -> Scalar[dtype]:
        """Compute inverse inertia projected onto an axis.

        I_axis^-1 = axis^T * I_world^-1 * axis

        For symmetric I^-1:
        result = ax*ax*I00 + 2*ax*ay*I01 + 2*ax*az*I02 + ay*ay*I11 + 2*ay*az*I12 + az*az*I22
        """
        var two = Scalar[dtype](2.0)
        return (
            axis_x * axis_x * inv_i00
            + two * axis_x * axis_y * inv_i01
            + two * axis_x * axis_z * inv_i02
            + axis_y * axis_y * inv_i11
            + two * axis_y * axis_z * inv_i12
            + axis_z * axis_z * inv_i22
        )

    @always_inline
    @staticmethod
    fn compute_angular_effective_mass(
        inv_i_a_axis: Scalar[dtype],  # Body A's inv inertia projected onto axis
        inv_i_b_axis: Scalar[dtype],  # Body B's inv inertia projected onto axis
        armature: Scalar[dtype],       # Rotor inertia (kg·m²)
    ) -> Scalar[dtype]:
        """Compute effective mass for angular constraint around axis.

        eff_mass = 1 / (I_a^-1 + I_b^-1 + 1/armature)

        Armature adds rotational inertia, making the system less responsive
        to torques and more stable.
        """
        var eps = Scalar[dtype](1e-10)
        var one = Scalar[dtype](1.0)

        # Handle armature (rotor inertia)
        var inv_armature = Scalar[dtype](0.0)
        if armature > eps:
            inv_armature = one / armature

        var total_inv_inertia = inv_i_a_axis + inv_i_b_axis + inv_armature
        if total_inv_inertia < eps:
            total_inv_inertia = eps

        return one / total_inv_inertia

    @always_inline
    @staticmethod
    fn compute_linear_effective_mass(
        inv_ma: Scalar[dtype],  # Body A's inverse mass
        inv_mb: Scalar[dtype],  # Body B's inverse mass
        # Lever arms (world-space vectors from body CoM to constraint point)
        ra_x: Scalar[dtype], ra_y: Scalar[dtype], ra_z: Scalar[dtype],
        rb_x: Scalar[dtype], rb_y: Scalar[dtype], rb_z: Scalar[dtype],
        # World-space inverse inertia tensors
        inv_ia_00: Scalar[dtype], inv_ia_01: Scalar[dtype], inv_ia_02: Scalar[dtype],
        inv_ia_11: Scalar[dtype], inv_ia_12: Scalar[dtype], inv_ia_22: Scalar[dtype],
        inv_ib_00: Scalar[dtype], inv_ib_01: Scalar[dtype], inv_ib_02: Scalar[dtype],
        inv_ib_11: Scalar[dtype], inv_ib_12: Scalar[dtype], inv_ib_22: Scalar[dtype],
        # Direction of constraint (normalized)
        dir_x: Scalar[dtype], dir_y: Scalar[dtype], dir_z: Scalar[dtype],
    ) -> Scalar[dtype]:
        """Compute effective mass for linear constraint along a direction.

        For a point constraint, effective mass includes both linear and angular
        contributions:

        K = m_a^-1 + m_b^-1 + [r_a×]^T * I_a^-1 * [r_a×] + [r_b×]^T * I_b^-1 * [r_b×]

        Then project K onto the constraint direction:
        eff_mass^-1 = dir^T * K * dir

        For simplicity, we compute a scalar approximation along the direction.
        """
        var eps = Scalar[dtype](1e-10)
        var one = Scalar[dtype](1.0)

        # Linear contribution
        var total_inv_mass = inv_ma + inv_mb

        # Angular contribution for body A: (r × dir)^T * I^-1 * (r × dir)
        # r × dir
        var ra_cross_x = ra_y * dir_z - ra_z * dir_y
        var ra_cross_y = ra_z * dir_x - ra_x * dir_z
        var ra_cross_z = ra_x * dir_y - ra_y * dir_x

        # Apply I_a^-1 to ra_cross
        var ia_ra_x = inv_ia_00 * ra_cross_x + inv_ia_01 * ra_cross_y + inv_ia_02 * ra_cross_z
        var ia_ra_y = inv_ia_01 * ra_cross_x + inv_ia_11 * ra_cross_y + inv_ia_12 * ra_cross_z
        var ia_ra_z = inv_ia_02 * ra_cross_x + inv_ia_12 * ra_cross_y + inv_ia_22 * ra_cross_z

        # Dot with ra_cross
        var ang_contrib_a = ra_cross_x * ia_ra_x + ra_cross_y * ia_ra_y + ra_cross_z * ia_ra_z

        # Angular contribution for body B
        var rb_cross_x = rb_y * dir_z - rb_z * dir_y
        var rb_cross_y = rb_z * dir_x - rb_x * dir_z
        var rb_cross_z = rb_x * dir_y - rb_y * dir_x

        var ib_rb_x = inv_ib_00 * rb_cross_x + inv_ib_01 * rb_cross_y + inv_ib_02 * rb_cross_z
        var ib_rb_y = inv_ib_01 * rb_cross_x + inv_ib_11 * rb_cross_y + inv_ib_12 * rb_cross_z
        var ib_rb_z = inv_ib_02 * rb_cross_x + inv_ib_12 * rb_cross_y + inv_ib_22 * rb_cross_z

        var ang_contrib_b = rb_cross_x * ib_rb_x + rb_cross_y * ib_rb_y + rb_cross_z * ib_rb_z

        # Total inverse effective mass
        var total_inv_eff_mass = total_inv_mass + ang_contrib_a + ang_contrib_b
        if total_inv_eff_mass < eps:
            total_inv_eff_mass = eps

        return one / total_inv_eff_mass
