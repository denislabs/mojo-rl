"""Revolute Joint constraint solver.

A revolute joint constrains two bodies to rotate around a common anchor point.
It can optionally have:
- Angle limits (lower/upper bounds on relative rotation)
- Motor (applies torque to reach target angular velocity)
- Spring (soft constraint with stiffness and damping)

The constraint ensures the anchor points on both bodies remain coincident.
"""

from math import cos, sin, sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    JOINT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    JOINT_TYPE,
    JOINT_BODY_A,
    JOINT_BODY_B,
    JOINT_ANCHOR_AX,
    JOINT_ANCHOR_AY,
    JOINT_ANCHOR_BX,
    JOINT_ANCHOR_BY,
    JOINT_REF_ANGLE,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    JOINT_MAX_MOTOR_TORQUE,
    JOINT_MOTOR_SPEED,
    JOINT_STIFFNESS,
    JOINT_DAMPING,
    JOINT_FLAGS,
    JOINT_IMPULSE,
    JOINT_MOTOR_IMPULSE,
    JOINT_REVOLUTE,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_MOTOR_ENABLED,
    JOINT_FLAG_SPRING_ENABLED,
)


struct RevoluteJointSolver:
    """Solver for revolute joint constraints.

    Implements sequential impulse solving for:
    1. Point-to-point constraint (anchor coincidence)
    2. Angle limit constraint (optional)
    3. Motor constraint (optional)
    4. Spring force (optional)
    """

    # =========================================================================
    # CPU Implementation
    # =========================================================================

    @staticmethod
    fn solve_velocity[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
    ](
        mut bodies: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE), MutAnyOrigin
        ],
        mut joints: LayoutTensor[
            dtype, Layout.row_major(BATCH, MAX_JOINTS, JOINT_DATA_SIZE), MutAnyOrigin
        ],
        joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dt: Scalar[dtype],
    ):
        """Solve velocity constraints for all joints."""
        for env in range(BATCH):
            var n_joints = Int(joint_counts[env])

            for j in range(n_joints):
                if j >= MAX_JOINTS:
                    break

                var joint_type = Int(joints[env, j, JOINT_TYPE])
                if joint_type != JOINT_REVOLUTE:
                    continue

                # Get body indices
                var body_a = Int(joints[env, j, JOINT_BODY_A])
                var body_b = Int(joints[env, j, JOINT_BODY_B])

                # Get body states
                var xa = bodies[env, body_a, IDX_X]
                var ya = bodies[env, body_a, IDX_Y]
                var angle_a = bodies[env, body_a, IDX_ANGLE]
                var vxa = bodies[env, body_a, IDX_VX]
                var vya = bodies[env, body_a, IDX_VY]
                var wa = bodies[env, body_a, IDX_OMEGA]
                var inv_ma = bodies[env, body_a, IDX_INV_MASS]
                var inv_ia = bodies[env, body_a, IDX_INV_INERTIA]

                var xb = bodies[env, body_b, IDX_X]
                var yb = bodies[env, body_b, IDX_Y]
                var angle_b = bodies[env, body_b, IDX_ANGLE]
                var vxb = bodies[env, body_b, IDX_VX]
                var vyb = bodies[env, body_b, IDX_VY]
                var wb = bodies[env, body_b, IDX_OMEGA]
                var inv_mb = bodies[env, body_b, IDX_INV_MASS]
                var inv_ib = bodies[env, body_b, IDX_INV_INERTIA]

                # Local anchors
                var local_ax = joints[env, j, JOINT_ANCHOR_AX]
                var local_ay = joints[env, j, JOINT_ANCHOR_AY]
                var local_bx = joints[env, j, JOINT_ANCHOR_BX]
                var local_by = joints[env, j, JOINT_ANCHOR_BY]

                # Transform anchors to world space
                var cos_a = cos(angle_a)
                var sin_a = sin(angle_a)
                var cos_b = cos(angle_b)
                var sin_b = sin(angle_b)

                # r_a = anchor in world relative to body A center
                var rax = local_ax * cos_a - local_ay * sin_a
                var ray = local_ax * sin_a + local_ay * cos_a
                # r_b = anchor in world relative to body B center
                var rbx = local_bx * cos_b - local_by * sin_b
                var rby = local_bx * sin_b + local_by * cos_b

                # Relative velocity at anchor point
                # v_a_anchor = v_a + w_a × r_a
                # v_b_anchor = v_b + w_b × r_b
                var va_anchor_x = vxa - wa * ray
                var va_anchor_y = vya + wa * rax
                var vb_anchor_x = vxb - wb * rby
                var vb_anchor_y = vyb + wb * rbx

                # Constraint velocity (should be zero for rigid joint)
                var cdot_x = vb_anchor_x - va_anchor_x
                var cdot_y = vb_anchor_y - va_anchor_y

                # Compute effective mass for point-to-point constraint
                # K = [inv_ma + inv_mb + inv_ia*ray^2 + inv_ib*rby^2, -inv_ia*rax*ray - inv_ib*rbx*rby]
                #     [-inv_ia*rax*ray - inv_ib*rbx*rby, inv_ma + inv_mb + inv_ia*rax^2 + inv_ib*rbx^2]
                var k11 = inv_ma + inv_mb + inv_ia * ray * ray + inv_ib * rby * rby
                var k12 = -inv_ia * rax * ray - inv_ib * rbx * rby
                var k22 = inv_ma + inv_mb + inv_ia * rax * rax + inv_ib * rbx * rbx

                # Invert 2x2 matrix
                var det = k11 * k22 - k12 * k12
                if det < Scalar[dtype](1e-10):
                    det = Scalar[dtype](1e-10)

                var inv_det = Scalar[dtype](1.0) / det
                var inv_k11 = k22 * inv_det
                var inv_k12 = -k12 * inv_det
                var inv_k22 = k11 * inv_det

                # Compute impulse: lambda = -K^-1 * Cdot
                var impulse_x = -(inv_k11 * cdot_x + inv_k12 * cdot_y)
                var impulse_y = -(inv_k12 * cdot_x + inv_k22 * cdot_y)

                # Apply impulse
                bodies[env, body_a, IDX_VX] = vxa - inv_ma * impulse_x
                bodies[env, body_a, IDX_VY] = vya - inv_ma * impulse_y
                bodies[env, body_a, IDX_OMEGA] = wa - inv_ia * (rax * impulse_y - ray * impulse_x)

                bodies[env, body_b, IDX_VX] = vxb + inv_mb * impulse_x
                bodies[env, body_b, IDX_VY] = vyb + inv_mb * impulse_y
                bodies[env, body_b, IDX_OMEGA] = wb + inv_ib * (rbx * impulse_y - rby * impulse_x)

                # Handle spring (soft constraint)
                var flags = Int(joints[env, j, JOINT_FLAGS])
                if flags & JOINT_FLAG_SPRING_ENABLED:
                    var stiffness = joints[env, j, JOINT_STIFFNESS]
                    var damping = joints[env, j, JOINT_DAMPING]
                    var ref_angle = joints[env, j, JOINT_REF_ANGLE]

                    # Current angle difference
                    var current_angle = angle_b - angle_a
                    var angle_error = current_angle - ref_angle

                    # Relative angular velocity
                    var rel_omega = wb - wa

                    # Spring torque: tau = -k * angle_error - c * rel_omega
                    var spring_torque = -stiffness * angle_error - damping * rel_omega

                    # Apply as impulse: impulse = torque * dt
                    var angular_impulse = spring_torque * dt

                    # Effective inertia
                    var eff_inertia = inv_ia + inv_ib
                    if eff_inertia > Scalar[dtype](1e-10):
                        var omega_change = angular_impulse * eff_inertia
                        bodies[env, body_a, IDX_OMEGA] = bodies[env, body_a, IDX_OMEGA] - inv_ia * angular_impulse / eff_inertia
                        bodies[env, body_b, IDX_OMEGA] = bodies[env, body_b, IDX_OMEGA] + inv_ib * angular_impulse / eff_inertia

                # Handle angle limits
                if flags & JOINT_FLAG_LIMIT_ENABLED:
                    var lower_limit = rebind[Scalar[dtype]](joints[env, j, JOINT_LOWER_LIMIT])
                    var upper_limit = rebind[Scalar[dtype]](joints[env, j, JOINT_UPPER_LIMIT])
                    var lim_ref_angle = rebind[Scalar[dtype]](joints[env, j, JOINT_REF_ANGLE])

                    # Current relative angle
                    var current_wa = rebind[Scalar[dtype]](bodies[env, body_a, IDX_OMEGA])
                    var current_wb = rebind[Scalar[dtype]](bodies[env, body_b, IDX_OMEGA])
                    var current_angle_a = rebind[Scalar[dtype]](bodies[env, body_a, IDX_ANGLE])
                    var current_angle_b = rebind[Scalar[dtype]](bodies[env, body_b, IDX_ANGLE])
                    var relative_angle = current_angle_b - current_angle_a - lim_ref_angle

                    # Relative angular velocity
                    var rel_omega = current_wb - current_wa

                    # Effective inertia for angular constraint
                    var lim_eff_inertia = rebind[Scalar[dtype]](inv_ia) + rebind[Scalar[dtype]](inv_ib)
                    if lim_eff_inertia > Scalar[dtype](1e-10):
                        var limit_impulse = Scalar[dtype](0.0)

                        # Check lower limit
                        if relative_angle <= lower_limit:
                            # At lower limit, prevent further clockwise rotation (negative rel_omega)
                            if rel_omega < Scalar[dtype](0.0):
                                limit_impulse = -rel_omega / lim_eff_inertia

                        # Check upper limit
                        elif relative_angle >= upper_limit:
                            # At upper limit, prevent further counter-clockwise rotation (positive rel_omega)
                            if rel_omega > Scalar[dtype](0.0):
                                limit_impulse = -rel_omega / lim_eff_inertia

                        # Apply limit impulse
                        if limit_impulse != Scalar[dtype](0.0):
                            bodies[env, body_a, IDX_OMEGA] = current_wa - rebind[Scalar[dtype]](inv_ia) * limit_impulse
                            bodies[env, body_b, IDX_OMEGA] = current_wb + rebind[Scalar[dtype]](inv_ib) * limit_impulse

                # Handle motor
                if flags & JOINT_FLAG_MOTOR_ENABLED:
                    var motor_speed = joints[env, j, JOINT_MOTOR_SPEED]
                    var max_torque = joints[env, j, JOINT_MAX_MOTOR_TORQUE]

                    # Current relative angular velocity
                    var current_wa = bodies[env, body_a, IDX_OMEGA]
                    var current_wb = bodies[env, body_b, IDX_OMEGA]
                    var rel_omega = current_wb - current_wa

                    # Motor wants to achieve target speed
                    var speed_error = motor_speed - rel_omega

                    # Effective inertia for motor
                    var eff_inertia = inv_ia + inv_ib
                    if eff_inertia > Scalar[dtype](1e-10):
                        var motor_impulse = speed_error / eff_inertia

                        # Clamp to max torque
                        var max_impulse = max_torque * dt
                        if motor_impulse > max_impulse:
                            motor_impulse = max_impulse
                        if motor_impulse < -max_impulse:
                            motor_impulse = -max_impulse

                        bodies[env, body_a, IDX_OMEGA] = current_wa - inv_ia * motor_impulse
                        bodies[env, body_b, IDX_OMEGA] = current_wb + inv_ib * motor_impulse

    @staticmethod
    fn solve_position[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
    ](
        mut bodies: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE), MutAnyOrigin
        ],
        joints: LayoutTensor[
            dtype, Layout.row_major(BATCH, MAX_JOINTS, JOINT_DATA_SIZE), MutAnyOrigin
        ],
        joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """Solve position constraints for all joints (position correction)."""
        for env in range(BATCH):
            var n_joints = Int(joint_counts[env])

            for j in range(n_joints):
                if j >= MAX_JOINTS:
                    break

                var joint_type = Int(joints[env, j, JOINT_TYPE])
                if joint_type != JOINT_REVOLUTE:
                    continue

                # Get body indices
                var body_a = Int(joints[env, j, JOINT_BODY_A])
                var body_b = Int(joints[env, j, JOINT_BODY_B])

                # Get body positions
                var xa = bodies[env, body_a, IDX_X]
                var ya = bodies[env, body_a, IDX_Y]
                var angle_a = bodies[env, body_a, IDX_ANGLE]
                var inv_ma = bodies[env, body_a, IDX_INV_MASS]
                var inv_ia = bodies[env, body_a, IDX_INV_INERTIA]

                var xb = bodies[env, body_b, IDX_X]
                var yb = bodies[env, body_b, IDX_Y]
                var angle_b = bodies[env, body_b, IDX_ANGLE]
                var inv_mb = bodies[env, body_b, IDX_INV_MASS]
                var inv_ib = bodies[env, body_b, IDX_INV_INERTIA]

                # Local anchors
                var local_ax = joints[env, j, JOINT_ANCHOR_AX]
                var local_ay = joints[env, j, JOINT_ANCHOR_AY]
                var local_bx = joints[env, j, JOINT_ANCHOR_BX]
                var local_by = joints[env, j, JOINT_ANCHOR_BY]

                # Transform anchors to world space
                var cos_a = cos(angle_a)
                var sin_a = sin(angle_a)
                var cos_b = cos(angle_b)
                var sin_b = sin(angle_b)

                var rax = local_ax * cos_a - local_ay * sin_a
                var ray = local_ax * sin_a + local_ay * cos_a
                var rbx = local_bx * cos_b - local_by * sin_b
                var rby = local_bx * sin_b + local_by * cos_b

                # World anchor positions
                var anchor_ax = xa + rax
                var anchor_ay = ya + ray
                var anchor_bx = xb + rbx
                var anchor_by = yb + rby

                # Position error (separation)
                var cx = anchor_bx - anchor_ax
                var cy = anchor_by - anchor_ay

                var error = sqrt(cx * cx + cy * cy)
                if error < slop:
                    continue

                # Compute effective mass (same as velocity solve)
                var k11 = inv_ma + inv_mb + inv_ia * ray * ray + inv_ib * rby * rby
                var k12 = -inv_ia * rax * ray - inv_ib * rbx * rby
                var k22 = inv_ma + inv_mb + inv_ia * rax * rax + inv_ib * rbx * rbx

                var det = k11 * k22 - k12 * k12
                if det < Scalar[dtype](1e-10):
                    det = Scalar[dtype](1e-10)

                var inv_det = Scalar[dtype](1.0) / det
                var inv_k11 = k22 * inv_det
                var inv_k12 = -k12 * inv_det
                var inv_k22 = k11 * inv_det

                # Position correction impulse with Baumgarte stabilization
                var correction_x = -baumgarte * (inv_k11 * cx + inv_k12 * cy)
                var correction_y = -baumgarte * (inv_k12 * cx + inv_k22 * cy)

                # Apply position correction
                bodies[env, body_a, IDX_X] = xa - inv_ma * correction_x
                bodies[env, body_a, IDX_Y] = ya - inv_ma * correction_y
                bodies[env, body_a, IDX_ANGLE] = angle_a - inv_ia * (rax * correction_y - ray * correction_x)

                bodies[env, body_b, IDX_X] = xb + inv_mb * correction_x
                bodies[env, body_b, IDX_Y] = yb + inv_mb * correction_y
                bodies[env, body_b, IDX_ANGLE] = angle_b + inv_ib * (rbx * correction_y - rby * correction_x)

                # Handle angle limit position correction
                var flags = Int(joints[env, j, JOINT_FLAGS])
                if flags & JOINT_FLAG_LIMIT_ENABLED:
                    var lower_limit = rebind[Scalar[dtype]](joints[env, j, JOINT_LOWER_LIMIT])
                    var upper_limit = rebind[Scalar[dtype]](joints[env, j, JOINT_UPPER_LIMIT])
                    var pos_ref_angle = rebind[Scalar[dtype]](joints[env, j, JOINT_REF_ANGLE])

                    # Current relative angle (after point constraint correction)
                    var cur_angle_a = rebind[Scalar[dtype]](bodies[env, body_a, IDX_ANGLE])
                    var cur_angle_b = rebind[Scalar[dtype]](bodies[env, body_b, IDX_ANGLE])
                    var relative_angle = cur_angle_b - cur_angle_a - pos_ref_angle

                    # Effective inertia
                    var pos_eff_inertia = rebind[Scalar[dtype]](inv_ia) + rebind[Scalar[dtype]](inv_ib)
                    if pos_eff_inertia > Scalar[dtype](1e-10):
                        var angle_correction = Scalar[dtype](0.0)

                        # Check lower limit
                        if relative_angle < lower_limit:
                            angle_correction = baumgarte * (lower_limit - relative_angle)

                        # Check upper limit
                        elif relative_angle > upper_limit:
                            angle_correction = baumgarte * (upper_limit - relative_angle)

                        # Apply angle correction
                        if angle_correction != Scalar[dtype](0.0):
                            bodies[env, body_a, IDX_ANGLE] = cur_angle_a - rebind[Scalar[dtype]](inv_ia) * angle_correction / pos_eff_inertia
                            bodies[env, body_b, IDX_ANGLE] = cur_angle_b + rebind[Scalar[dtype]](inv_ib) * angle_correction / pos_eff_inertia

    # =========================================================================
    # GPU Implementation
    # =========================================================================

    @staticmethod
    fn solve_velocity_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
    ](
        ctx: DeviceContext,
        mut bodies_buf: DeviceBuffer[dtype],
        mut joints_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Solve velocity constraints on GPU."""
        var bodies = LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE), MutAnyOrigin
        ](bodies_buf.unsafe_ptr())
        var joints = LayoutTensor[
            dtype, Layout.row_major(BATCH, MAX_JOINTS, JOINT_DATA_SIZE), MutAnyOrigin
        ](joints_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            bodies: LayoutTensor[
                dtype, Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE), MutAnyOrigin
            ],
            joints: LayoutTensor[
                dtype, Layout.row_major(BATCH, MAX_JOINTS, JOINT_DATA_SIZE), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            dt: Scalar[dtype],
        ):
            RevoluteJointSolver._solve_velocity_kernel[BATCH, NUM_BODIES, MAX_JOINTS](
                bodies, joints, joint_counts, dt
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            bodies, joints, joint_counts, dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    fn _solve_velocity_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
    ](
        bodies: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE), MutAnyOrigin
        ],
        joints: LayoutTensor[
            dtype, Layout.row_major(BATCH, MAX_JOINTS, JOINT_DATA_SIZE), MutAnyOrigin
        ],
        joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dt: Scalar[dtype],
    ):
        """GPU kernel for velocity constraint solving."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var n_joints = Int(joint_counts[env])

        for j in range(MAX_JOINTS):
            if j >= n_joints:
                break

            var joint_type = Int(joints[env, j, JOINT_TYPE])
            if joint_type != JOINT_REVOLUTE:
                continue

            var body_a = Int(joints[env, j, JOINT_BODY_A])
            var body_b = Int(joints[env, j, JOINT_BODY_B])

            var xa = bodies[env, body_a, IDX_X]
            var ya = bodies[env, body_a, IDX_Y]
            var angle_a = bodies[env, body_a, IDX_ANGLE]
            var vxa = bodies[env, body_a, IDX_VX]
            var vya = bodies[env, body_a, IDX_VY]
            var wa = bodies[env, body_a, IDX_OMEGA]
            var inv_ma = bodies[env, body_a, IDX_INV_MASS]
            var inv_ia = bodies[env, body_a, IDX_INV_INERTIA]

            var xb = bodies[env, body_b, IDX_X]
            var yb = bodies[env, body_b, IDX_Y]
            var angle_b = bodies[env, body_b, IDX_ANGLE]
            var vxb = bodies[env, body_b, IDX_VX]
            var vyb = bodies[env, body_b, IDX_VY]
            var wb = bodies[env, body_b, IDX_OMEGA]
            var inv_mb = bodies[env, body_b, IDX_INV_MASS]
            var inv_ib = bodies[env, body_b, IDX_INV_INERTIA]

            var local_ax = joints[env, j, JOINT_ANCHOR_AX]
            var local_ay = joints[env, j, JOINT_ANCHOR_AY]
            var local_bx = joints[env, j, JOINT_ANCHOR_BX]
            var local_by = joints[env, j, JOINT_ANCHOR_BY]

            var cos_a = cos(angle_a)
            var sin_a = sin(angle_a)
            var cos_b = cos(angle_b)
            var sin_b = sin(angle_b)

            var rax = local_ax * cos_a - local_ay * sin_a
            var ray = local_ax * sin_a + local_ay * cos_a
            var rbx = local_bx * cos_b - local_by * sin_b
            var rby = local_bx * sin_b + local_by * cos_b

            var va_anchor_x = vxa - wa * ray
            var va_anchor_y = vya + wa * rax
            var vb_anchor_x = vxb - wb * rby
            var vb_anchor_y = vyb + wb * rbx

            var cdot_x = vb_anchor_x - va_anchor_x
            var cdot_y = vb_anchor_y - va_anchor_y

            var k11 = inv_ma + inv_mb + inv_ia * ray * ray + inv_ib * rby * rby
            var k12 = -inv_ia * rax * ray - inv_ib * rbx * rby
            var k22 = inv_ma + inv_mb + inv_ia * rax * rax + inv_ib * rbx * rbx

            var det = k11 * k22 - k12 * k12
            if det < Scalar[dtype](1e-10):
                det = Scalar[dtype](1e-10)

            var inv_det = Scalar[dtype](1.0) / det
            var inv_k11 = k22 * inv_det
            var inv_k12 = -k12 * inv_det
            var inv_k22 = k11 * inv_det

            var impulse_x = -(inv_k11 * cdot_x + inv_k12 * cdot_y)
            var impulse_y = -(inv_k12 * cdot_x + inv_k22 * cdot_y)

            bodies[env, body_a, IDX_VX] = vxa - inv_ma * impulse_x
            bodies[env, body_a, IDX_VY] = vya - inv_ma * impulse_y
            bodies[env, body_a, IDX_OMEGA] = wa - inv_ia * (rax * impulse_y - ray * impulse_x)

            bodies[env, body_b, IDX_VX] = vxb + inv_mb * impulse_x
            bodies[env, body_b, IDX_VY] = vyb + inv_mb * impulse_y
            bodies[env, body_b, IDX_OMEGA] = wb + inv_ib * (rbx * impulse_y - rby * impulse_x)

            # Spring handling
            var flags = Int(joints[env, j, JOINT_FLAGS])
            if flags & JOINT_FLAG_SPRING_ENABLED:
                var stiffness = joints[env, j, JOINT_STIFFNESS]
                var damping = joints[env, j, JOINT_DAMPING]
                var ref_angle = joints[env, j, JOINT_REF_ANGLE]

                var current_angle = angle_b - angle_a
                var angle_error = current_angle - ref_angle
                var rel_omega = bodies[env, body_b, IDX_OMEGA] - bodies[env, body_a, IDX_OMEGA]

                var spring_torque = -stiffness * angle_error - damping * rel_omega
                var angular_impulse = spring_torque * dt

                var eff_inertia = inv_ia + inv_ib
                if eff_inertia > Scalar[dtype](1e-10):
                    bodies[env, body_a, IDX_OMEGA] = bodies[env, body_a, IDX_OMEGA] - inv_ia * angular_impulse / eff_inertia
                    bodies[env, body_b, IDX_OMEGA] = bodies[env, body_b, IDX_OMEGA] + inv_ib * angular_impulse / eff_inertia

            # Handle angle limits (GPU)
            if flags & JOINT_FLAG_LIMIT_ENABLED:
                var lower_limit = rebind[Scalar[dtype]](joints[env, j, JOINT_LOWER_LIMIT])
                var upper_limit = rebind[Scalar[dtype]](joints[env, j, JOINT_UPPER_LIMIT])
                var lim_ref_angle = rebind[Scalar[dtype]](joints[env, j, JOINT_REF_ANGLE])

                # Current relative angle
                var current_wa = rebind[Scalar[dtype]](bodies[env, body_a, IDX_OMEGA])
                var current_wb = rebind[Scalar[dtype]](bodies[env, body_b, IDX_OMEGA])
                var current_angle_a = rebind[Scalar[dtype]](bodies[env, body_a, IDX_ANGLE])
                var current_angle_b = rebind[Scalar[dtype]](bodies[env, body_b, IDX_ANGLE])
                var relative_angle = current_angle_b - current_angle_a - lim_ref_angle

                # Relative angular velocity
                var rel_omega = current_wb - current_wa

                # Effective inertia for angular constraint
                var lim_eff_inertia = rebind[Scalar[dtype]](inv_ia) + rebind[Scalar[dtype]](inv_ib)
                if lim_eff_inertia > Scalar[dtype](1e-10):
                    var limit_impulse = Scalar[dtype](0.0)

                    # Check lower limit
                    if relative_angle <= lower_limit:
                        if rel_omega < Scalar[dtype](0.0):
                            limit_impulse = -rel_omega / lim_eff_inertia

                    # Check upper limit
                    elif relative_angle >= upper_limit:
                        if rel_omega > Scalar[dtype](0.0):
                            limit_impulse = -rel_omega / lim_eff_inertia

                    # Apply limit impulse
                    if limit_impulse != Scalar[dtype](0.0):
                        bodies[env, body_a, IDX_OMEGA] = current_wa - rebind[Scalar[dtype]](inv_ia) * limit_impulse
                        bodies[env, body_b, IDX_OMEGA] = current_wb + rebind[Scalar[dtype]](inv_ib) * limit_impulse

    @staticmethod
    fn solve_position_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
    ](
        ctx: DeviceContext,
        mut bodies_buf: DeviceBuffer[dtype],
        joints_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ) raises:
        """Solve position constraints on GPU."""
        var bodies = LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE), MutAnyOrigin
        ](bodies_buf.unsafe_ptr())
        var joints = LayoutTensor[
            dtype, Layout.row_major(BATCH, MAX_JOINTS, JOINT_DATA_SIZE), MutAnyOrigin
        ](joints_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            bodies: LayoutTensor[
                dtype, Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE), MutAnyOrigin
            ],
            joints: LayoutTensor[
                dtype, Layout.row_major(BATCH, MAX_JOINTS, JOINT_DATA_SIZE), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            baumgarte: Scalar[dtype],
            slop: Scalar[dtype],
        ):
            RevoluteJointSolver._solve_position_kernel[BATCH, NUM_BODIES, MAX_JOINTS](
                bodies, joints, joint_counts, baumgarte, slop
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            bodies, joints, joint_counts, baumgarte, slop,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    fn _solve_position_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
    ](
        bodies: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE), MutAnyOrigin
        ],
        joints: LayoutTensor[
            dtype, Layout.row_major(BATCH, MAX_JOINTS, JOINT_DATA_SIZE), MutAnyOrigin
        ],
        joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """GPU kernel for position constraint solving."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var n_joints = Int(joint_counts[env])

        for j in range(MAX_JOINTS):
            if j >= n_joints:
                break

            var joint_type = Int(joints[env, j, JOINT_TYPE])
            if joint_type != JOINT_REVOLUTE:
                continue

            var body_a = Int(joints[env, j, JOINT_BODY_A])
            var body_b = Int(joints[env, j, JOINT_BODY_B])

            var xa = bodies[env, body_a, IDX_X]
            var ya = bodies[env, body_a, IDX_Y]
            var angle_a = bodies[env, body_a, IDX_ANGLE]
            var inv_ma = bodies[env, body_a, IDX_INV_MASS]
            var inv_ia = bodies[env, body_a, IDX_INV_INERTIA]

            var xb = bodies[env, body_b, IDX_X]
            var yb = bodies[env, body_b, IDX_Y]
            var angle_b = bodies[env, body_b, IDX_ANGLE]
            var inv_mb = bodies[env, body_b, IDX_INV_MASS]
            var inv_ib = bodies[env, body_b, IDX_INV_INERTIA]

            var local_ax = joints[env, j, JOINT_ANCHOR_AX]
            var local_ay = joints[env, j, JOINT_ANCHOR_AY]
            var local_bx = joints[env, j, JOINT_ANCHOR_BX]
            var local_by = joints[env, j, JOINT_ANCHOR_BY]

            var cos_a = cos(angle_a)
            var sin_a = sin(angle_a)
            var cos_b = cos(angle_b)
            var sin_b = sin(angle_b)

            var rax = local_ax * cos_a - local_ay * sin_a
            var ray = local_ax * sin_a + local_ay * cos_a
            var rbx = local_bx * cos_b - local_by * sin_b
            var rby = local_bx * sin_b + local_by * cos_b

            var anchor_ax = xa + rax
            var anchor_ay = ya + ray
            var anchor_bx = xb + rbx
            var anchor_by = yb + rby

            var cx = anchor_bx - anchor_ax
            var cy = anchor_by - anchor_ay

            var error = sqrt(cx * cx + cy * cy)
            if error < slop:
                continue

            var k11 = inv_ma + inv_mb + inv_ia * ray * ray + inv_ib * rby * rby
            var k12 = -inv_ia * rax * ray - inv_ib * rbx * rby
            var k22 = inv_ma + inv_mb + inv_ia * rax * rax + inv_ib * rbx * rbx

            var det = k11 * k22 - k12 * k12
            if det < Scalar[dtype](1e-10):
                det = Scalar[dtype](1e-10)

            var inv_det = Scalar[dtype](1.0) / det
            var inv_k11 = k22 * inv_det
            var inv_k12 = -k12 * inv_det
            var inv_k22 = k11 * inv_det

            var correction_x = -baumgarte * (inv_k11 * cx + inv_k12 * cy)
            var correction_y = -baumgarte * (inv_k12 * cx + inv_k22 * cy)

            bodies[env, body_a, IDX_X] = xa - inv_ma * correction_x
            bodies[env, body_a, IDX_Y] = ya - inv_ma * correction_y
            bodies[env, body_a, IDX_ANGLE] = angle_a - inv_ia * (rax * correction_y - ray * correction_x)

            bodies[env, body_b, IDX_X] = xb + inv_mb * correction_x
            bodies[env, body_b, IDX_Y] = yb + inv_mb * correction_y
            bodies[env, body_b, IDX_ANGLE] = angle_b + inv_ib * (rbx * correction_y - rby * correction_x)

            # Handle angle limit position correction (GPU)
            var flags = Int(joints[env, j, JOINT_FLAGS])
            if flags & JOINT_FLAG_LIMIT_ENABLED:
                var lower_limit = rebind[Scalar[dtype]](joints[env, j, JOINT_LOWER_LIMIT])
                var upper_limit = rebind[Scalar[dtype]](joints[env, j, JOINT_UPPER_LIMIT])
                var pos_ref_angle = rebind[Scalar[dtype]](joints[env, j, JOINT_REF_ANGLE])

                # Current relative angle (after point constraint correction)
                var cur_angle_a = rebind[Scalar[dtype]](bodies[env, body_a, IDX_ANGLE])
                var cur_angle_b = rebind[Scalar[dtype]](bodies[env, body_b, IDX_ANGLE])
                var relative_angle = cur_angle_b - cur_angle_a - pos_ref_angle

                # Effective inertia
                var pos_eff_inertia = rebind[Scalar[dtype]](inv_ia) + rebind[Scalar[dtype]](inv_ib)
                if pos_eff_inertia > Scalar[dtype](1e-10):
                    var angle_correction = Scalar[dtype](0.0)

                    # Check lower limit
                    if relative_angle < lower_limit:
                        angle_correction = baumgarte * (lower_limit - relative_angle)

                    # Check upper limit
                    elif relative_angle > upper_limit:
                        angle_correction = baumgarte * (upper_limit - relative_angle)

                    # Apply angle correction
                    if angle_correction != Scalar[dtype](0.0):
                        bodies[env, body_a, IDX_ANGLE] = cur_angle_a - rebind[Scalar[dtype]](inv_ia) * angle_correction / pos_eff_inertia
                        bodies[env, body_b, IDX_ANGLE] = cur_angle_b + rebind[Scalar[dtype]](inv_ib) * angle_correction / pos_eff_inertia

    # =========================================================================
    # Strided GPU Kernels for Flat State Layout
    # =========================================================================
    #
    # These methods work with flat [BATCH, STATE_SIZE] layout for bodies.
    # Joints are kept in standard layout (stored in state at JOINTS_OFFSET).
    # Memory layout: state[env * ENV_STRIDE + OFFSET + ...]
    # =========================================================================

    @always_inline
    @staticmethod
    fn _solve_velocity_kernel_strided[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        ENV_STRIDE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH * ENV_STRIDE),
            MutAnyOrigin,
        ],
        joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dt: Scalar[dtype],
    ):
        """GPU kernel for velocity constraint solving with strided body layout."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var env_base = env * ENV_STRIDE
        var bodies_base = env_base + BODIES_OFFSET
        var joints_base = env_base + JOINTS_OFFSET
        var n_joints = Int(joint_counts[env])

        for j in range(MAX_JOINTS):
            if j >= n_joints:
                break

            var joint_base = joints_base + j * JOINT_DATA_SIZE
            var joint_type = Int(state[joint_base + JOINT_TYPE])
            if joint_type != JOINT_REVOLUTE:
                continue

            var body_a = Int(state[joint_base + JOINT_BODY_A])
            var body_b = Int(state[joint_base + JOINT_BODY_B])

            var body_a_base = bodies_base + body_a * BODY_STATE_SIZE
            var body_b_base = bodies_base + body_b * BODY_STATE_SIZE

            var xa = state[body_a_base + IDX_X]
            var ya = state[body_a_base + IDX_Y]
            var angle_a = state[body_a_base + IDX_ANGLE]
            var vxa = state[body_a_base + IDX_VX]
            var vya = state[body_a_base + IDX_VY]
            var wa = state[body_a_base + IDX_OMEGA]
            var inv_ma = state[body_a_base + IDX_INV_MASS]
            var inv_ia = state[body_a_base + IDX_INV_INERTIA]

            var xb = state[body_b_base + IDX_X]
            var yb = state[body_b_base + IDX_Y]
            var angle_b = state[body_b_base + IDX_ANGLE]
            var vxb = state[body_b_base + IDX_VX]
            var vyb = state[body_b_base + IDX_VY]
            var wb = state[body_b_base + IDX_OMEGA]
            var inv_mb = state[body_b_base + IDX_INV_MASS]
            var inv_ib = state[body_b_base + IDX_INV_INERTIA]

            var local_ax = state[joint_base + JOINT_ANCHOR_AX]
            var local_ay = state[joint_base + JOINT_ANCHOR_AY]
            var local_bx = state[joint_base + JOINT_ANCHOR_BX]
            var local_by = state[joint_base + JOINT_ANCHOR_BY]

            var cos_a = cos(angle_a)
            var sin_a = sin(angle_a)
            var cos_b = cos(angle_b)
            var sin_b = sin(angle_b)

            var rax = local_ax * cos_a - local_ay * sin_a
            var ray = local_ax * sin_a + local_ay * cos_a
            var rbx = local_bx * cos_b - local_by * sin_b
            var rby = local_bx * sin_b + local_by * cos_b

            var va_anchor_x = vxa - wa * ray
            var va_anchor_y = vya + wa * rax
            var vb_anchor_x = vxb - wb * rby
            var vb_anchor_y = vyb + wb * rbx

            var cdot_x = vb_anchor_x - va_anchor_x
            var cdot_y = vb_anchor_y - va_anchor_y

            var k11 = inv_ma + inv_mb + inv_ia * ray * ray + inv_ib * rby * rby
            var k12 = -inv_ia * rax * ray - inv_ib * rbx * rby
            var k22 = inv_ma + inv_mb + inv_ia * rax * rax + inv_ib * rbx * rbx

            var det = k11 * k22 - k12 * k12
            if det < Scalar[dtype](1e-10):
                det = Scalar[dtype](1e-10)

            var inv_det = Scalar[dtype](1.0) / det
            var inv_k11 = k22 * inv_det
            var inv_k12 = -k12 * inv_det
            var inv_k22 = k11 * inv_det

            var impulse_x = -(inv_k11 * cdot_x + inv_k12 * cdot_y)
            var impulse_y = -(inv_k12 * cdot_x + inv_k22 * cdot_y)

            state[body_a_base + IDX_VX] = vxa - inv_ma * impulse_x
            state[body_a_base + IDX_VY] = vya - inv_ma * impulse_y
            state[body_a_base + IDX_OMEGA] = wa - inv_ia * (rax * impulse_y - ray * impulse_x)

            state[body_b_base + IDX_VX] = vxb + inv_mb * impulse_x
            state[body_b_base + IDX_VY] = vyb + inv_mb * impulse_y
            state[body_b_base + IDX_OMEGA] = wb + inv_ib * (rbx * impulse_y - rby * impulse_x)

            # Spring handling
            var flags = Int(state[joint_base + JOINT_FLAGS])
            if flags & JOINT_FLAG_SPRING_ENABLED:
                var stiffness = state[joint_base + JOINT_STIFFNESS]
                var damping = state[joint_base + JOINT_DAMPING]
                var ref_angle = state[joint_base + JOINT_REF_ANGLE]

                var current_angle = angle_b - angle_a
                var angle_error = current_angle - ref_angle
                var rel_omega = state[body_b_base + IDX_OMEGA] - state[body_a_base + IDX_OMEGA]

                var spring_torque = -stiffness * angle_error - damping * rel_omega
                var angular_impulse = spring_torque * dt

                var eff_inertia = inv_ia + inv_ib
                if eff_inertia > Scalar[dtype](1e-10):
                    state[body_a_base + IDX_OMEGA] = state[body_a_base + IDX_OMEGA] - inv_ia * angular_impulse / eff_inertia
                    state[body_b_base + IDX_OMEGA] = state[body_b_base + IDX_OMEGA] + inv_ib * angular_impulse / eff_inertia

            # Handle angle limits
            if flags & JOINT_FLAG_LIMIT_ENABLED:
                var lower_limit = rebind[Scalar[dtype]](state[joint_base + JOINT_LOWER_LIMIT])
                var upper_limit = rebind[Scalar[dtype]](state[joint_base + JOINT_UPPER_LIMIT])
                var lim_ref_angle = rebind[Scalar[dtype]](state[joint_base + JOINT_REF_ANGLE])

                var current_wa = rebind[Scalar[dtype]](state[body_a_base + IDX_OMEGA])
                var current_wb = rebind[Scalar[dtype]](state[body_b_base + IDX_OMEGA])
                var current_angle_a = rebind[Scalar[dtype]](state[body_a_base + IDX_ANGLE])
                var current_angle_b = rebind[Scalar[dtype]](state[body_b_base + IDX_ANGLE])
                var relative_angle = current_angle_b - current_angle_a - lim_ref_angle

                var rel_omega = current_wb - current_wa
                var lim_eff_inertia = rebind[Scalar[dtype]](inv_ia) + rebind[Scalar[dtype]](inv_ib)

                if lim_eff_inertia > Scalar[dtype](1e-10):
                    var limit_impulse = Scalar[dtype](0.0)

                    if relative_angle <= lower_limit:
                        if rel_omega < Scalar[dtype](0.0):
                            limit_impulse = -rel_omega / lim_eff_inertia
                    elif relative_angle >= upper_limit:
                        if rel_omega > Scalar[dtype](0.0):
                            limit_impulse = -rel_omega / lim_eff_inertia

                    if limit_impulse != Scalar[dtype](0.0):
                        state[body_a_base + IDX_OMEGA] = current_wa - rebind[Scalar[dtype]](inv_ia) * limit_impulse
                        state[body_b_base + IDX_OMEGA] = current_wb + rebind[Scalar[dtype]](inv_ib) * limit_impulse

    @always_inline
    @staticmethod
    fn _solve_position_kernel_strided[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        ENV_STRIDE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH * ENV_STRIDE),
            MutAnyOrigin,
        ],
        joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """GPU kernel for position constraint solving with strided body layout."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var env_base = env * ENV_STRIDE
        var bodies_base = env_base + BODIES_OFFSET
        var joints_base = env_base + JOINTS_OFFSET
        var n_joints = Int(joint_counts[env])

        for j in range(MAX_JOINTS):
            if j >= n_joints:
                break

            var joint_base = joints_base + j * JOINT_DATA_SIZE
            var joint_type = Int(state[joint_base + JOINT_TYPE])
            if joint_type != JOINT_REVOLUTE:
                continue

            var body_a = Int(state[joint_base + JOINT_BODY_A])
            var body_b = Int(state[joint_base + JOINT_BODY_B])

            var body_a_base = bodies_base + body_a * BODY_STATE_SIZE
            var body_b_base = bodies_base + body_b * BODY_STATE_SIZE

            var xa = state[body_a_base + IDX_X]
            var ya = state[body_a_base + IDX_Y]
            var angle_a = state[body_a_base + IDX_ANGLE]
            var inv_ma = state[body_a_base + IDX_INV_MASS]
            var inv_ia = state[body_a_base + IDX_INV_INERTIA]

            var xb = state[body_b_base + IDX_X]
            var yb = state[body_b_base + IDX_Y]
            var angle_b = state[body_b_base + IDX_ANGLE]
            var inv_mb = state[body_b_base + IDX_INV_MASS]
            var inv_ib = state[body_b_base + IDX_INV_INERTIA]

            var local_ax = state[joint_base + JOINT_ANCHOR_AX]
            var local_ay = state[joint_base + JOINT_ANCHOR_AY]
            var local_bx = state[joint_base + JOINT_ANCHOR_BX]
            var local_by = state[joint_base + JOINT_ANCHOR_BY]

            var cos_a = cos(angle_a)
            var sin_a = sin(angle_a)
            var cos_b = cos(angle_b)
            var sin_b = sin(angle_b)

            var rax = local_ax * cos_a - local_ay * sin_a
            var ray = local_ax * sin_a + local_ay * cos_a
            var rbx = local_bx * cos_b - local_by * sin_b
            var rby = local_bx * sin_b + local_by * cos_b

            var anchor_ax = xa + rax
            var anchor_ay = ya + ray
            var anchor_bx = xb + rbx
            var anchor_by = yb + rby

            var cx = anchor_bx - anchor_ax
            var cy = anchor_by - anchor_ay

            var error = sqrt(cx * cx + cy * cy)
            if error < slop:
                continue

            var k11 = inv_ma + inv_mb + inv_ia * ray * ray + inv_ib * rby * rby
            var k12 = -inv_ia * rax * ray - inv_ib * rbx * rby
            var k22 = inv_ma + inv_mb + inv_ia * rax * rax + inv_ib * rbx * rbx

            var det = k11 * k22 - k12 * k12
            if det < Scalar[dtype](1e-10):
                det = Scalar[dtype](1e-10)

            var inv_det = Scalar[dtype](1.0) / det
            var inv_k11 = k22 * inv_det
            var inv_k12 = -k12 * inv_det
            var inv_k22 = k11 * inv_det

            var correction_x = -baumgarte * (inv_k11 * cx + inv_k12 * cy)
            var correction_y = -baumgarte * (inv_k12 * cx + inv_k22 * cy)

            state[body_a_base + IDX_X] = xa - inv_ma * correction_x
            state[body_a_base + IDX_Y] = ya - inv_ma * correction_y
            state[body_a_base + IDX_ANGLE] = angle_a - inv_ia * (rax * correction_y - ray * correction_x)

            state[body_b_base + IDX_X] = xb + inv_mb * correction_x
            state[body_b_base + IDX_Y] = yb + inv_mb * correction_y
            state[body_b_base + IDX_ANGLE] = angle_b + inv_ib * (rbx * correction_y - rby * correction_x)

            # Handle angle limit position correction
            var flags = Int(state[joint_base + JOINT_FLAGS])
            if flags & JOINT_FLAG_LIMIT_ENABLED:
                var lower_limit = rebind[Scalar[dtype]](state[joint_base + JOINT_LOWER_LIMIT])
                var upper_limit = rebind[Scalar[dtype]](state[joint_base + JOINT_UPPER_LIMIT])
                var pos_ref_angle = rebind[Scalar[dtype]](state[joint_base + JOINT_REF_ANGLE])

                var cur_angle_a = rebind[Scalar[dtype]](state[body_a_base + IDX_ANGLE])
                var cur_angle_b = rebind[Scalar[dtype]](state[body_b_base + IDX_ANGLE])
                var relative_angle = cur_angle_b - cur_angle_a - pos_ref_angle

                var pos_eff_inertia = rebind[Scalar[dtype]](inv_ia) + rebind[Scalar[dtype]](inv_ib)
                if pos_eff_inertia > Scalar[dtype](1e-10):
                    var angle_correction = Scalar[dtype](0.0)

                    if relative_angle < lower_limit:
                        angle_correction = baumgarte * (lower_limit - relative_angle)
                    elif relative_angle > upper_limit:
                        angle_correction = baumgarte * (upper_limit - relative_angle)

                    if angle_correction != Scalar[dtype](0.0):
                        state[body_a_base + IDX_ANGLE] = cur_angle_a - rebind[Scalar[dtype]](inv_ia) * angle_correction / pos_eff_inertia
                        state[body_b_base + IDX_ANGLE] = cur_angle_b + rebind[Scalar[dtype]](inv_ib) * angle_correction / pos_eff_inertia

    @staticmethod
    fn solve_velocity_gpu_strided[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        ENV_STRIDE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Solve velocity constraints on GPU with strided layout."""
        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH * ENV_STRIDE), MutAnyOrigin
        ](state_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH * ENV_STRIDE), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            dt: Scalar[dtype],
        ):
            RevoluteJointSolver._solve_velocity_kernel_strided[
                BATCH, NUM_BODIES, MAX_JOINTS, ENV_STRIDE, BODIES_OFFSET, JOINTS_OFFSET
            ](state, joint_counts, dt)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state, joint_counts, dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn solve_position_gpu_strided[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        ENV_STRIDE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ) raises:
        """Solve position constraints on GPU with strided layout."""
        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH * ENV_STRIDE), MutAnyOrigin
        ](state_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH * ENV_STRIDE), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            baumgarte: Scalar[dtype],
            slop: Scalar[dtype],
        ):
            RevoluteJointSolver._solve_position_kernel_strided[
                BATCH, NUM_BODIES, MAX_JOINTS, ENV_STRIDE, BODIES_OFFSET, JOINTS_OFFSET
            ](state, joint_counts, baumgarte, slop)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state, joint_counts, baumgarte, slop,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
