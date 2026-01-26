"""Semi-Implicit Euler Integrator for 3D Physics.

Integrates velocities and positions for 3D rigid bodies, including
quaternion integration for rotations.
"""

from math import sqrt
from math3d import Vec3, Quat
from ..constants import (
    dtype,
    BODY_STATE_SIZE_3D,
    IDX_PX, IDX_PY, IDX_PZ,
    IDX_QW, IDX_QX, IDX_QY, IDX_QZ,
    IDX_VX, IDX_VY, IDX_VZ,
    IDX_WX, IDX_WY, IDX_WZ,
    IDX_FX, IDX_FY, IDX_FZ,
    IDX_TX, IDX_TY, IDX_TZ,
    IDX_MASS, IDX_INV_MASS,
    IDX_IXX, IDX_IYY, IDX_IZZ,
    IDX_BODY_TYPE,
    BODY_DYNAMIC,
    DEFAULT_GRAVITY_Z_3D,
)


fn integrate_quaternion(q: Quat, omega: Vec3, dt: Float64) -> Quat:
    """Integrate quaternion with angular velocity.

    Uses the standard quaternion integration formula:
    q(t+dt) = q(t) + 0.5 * dt * omega * q(t)

    where omega * q is the quaternion product of (0, omega) and q.

    Args:
        q: Current orientation quaternion.
        omega: Angular velocity vector (world space).
        dt: Time step.

    Returns:
        New orientation quaternion (normalized).
    """
    # Convert angular velocity to quaternion: (0, omega_x, omega_y, omega_z)
    var omega_quat = Quat(0.0, omega.x, omega.y, omega.z)

    # q_dot = 0.5 * omega_quat * q
    var q_dot = omega_quat * q
    q_dot.w *= 0.5
    q_dot.x *= 0.5
    q_dot.y *= 0.5
    q_dot.z *= 0.5

    # Integrate: q_new = q + dt * q_dot
    var q_new = Quat(
        q.w + dt * q_dot.w,
        q.x + dt * q_dot.x,
        q.y + dt * q_dot.y,
        q.z + dt * q_dot.z,
    )

    # Normalize to maintain unit quaternion
    return q_new.normalized()


fn integrate_velocities_3d(
    mut state: List[Float64],
    body_idx: Int,
    gravity: Vec3,
    dt: Float64,
):
    """Integrate velocities with forces and gravity.

    Updates linear and angular velocities based on accumulated forces/torques.

    Args:
        state: Physics state array (modified in place).
        body_idx: Index of body in state array.
        gravity: Gravity acceleration vector.
        dt: Time step.
    """
    var base = body_idx * BODY_STATE_SIZE_3D

    # Check if body is dynamic
    var body_type = Int(state[base + IDX_BODY_TYPE])
    if body_type != BODY_DYNAMIC:
        return

    # Get inverse mass and inertia
    var inv_mass = state[base + IDX_INV_MASS]
    if inv_mass <= 0.0:
        return

    var inv_ixx = 1.0 / state[base + IDX_IXX] if state[base + IDX_IXX] > 0.0 else 0.0
    var inv_iyy = 1.0 / state[base + IDX_IYY] if state[base + IDX_IYY] > 0.0 else 0.0
    var inv_izz = 1.0 / state[base + IDX_IZZ] if state[base + IDX_IZZ] > 0.0 else 0.0

    # Get current forces and torques
    var fx = state[base + IDX_FX]
    var fy = state[base + IDX_FY]
    var fz = state[base + IDX_FZ]
    var tx = state[base + IDX_TX]
    var ty = state[base + IDX_TY]
    var tz = state[base + IDX_TZ]

    # Add gravity force (F = m * g)
    var mass = state[base + IDX_MASS]
    fx += mass * gravity.x
    fy += mass * gravity.y
    fz += mass * gravity.z

    # Integrate linear velocity: v += dt * (F / m)
    state[base + IDX_VX] += dt * fx * inv_mass
    state[base + IDX_VY] += dt * fy * inv_mass
    state[base + IDX_VZ] += dt * fz * inv_mass

    # Integrate angular velocity: w += dt * (I^-1 * tau)
    # Note: Using diagonal inertia approximation
    state[base + IDX_WX] += dt * tx * inv_ixx
    state[base + IDX_WY] += dt * ty * inv_iyy
    state[base + IDX_WZ] += dt * tz * inv_izz

    # Clear force/torque accumulators
    state[base + IDX_FX] = 0.0
    state[base + IDX_FY] = 0.0
    state[base + IDX_FZ] = 0.0
    state[base + IDX_TX] = 0.0
    state[base + IDX_TY] = 0.0
    state[base + IDX_TZ] = 0.0


fn integrate_positions_3d(
    mut state: List[Float64],
    body_idx: Int,
    dt: Float64,
):
    """Integrate positions with velocities.

    Updates position and orientation based on linear and angular velocities.

    Args:
        state: Physics state array (modified in place).
        body_idx: Index of body in state array.
        dt: Time step.
    """
    var base = body_idx * BODY_STATE_SIZE_3D

    # Check if body is dynamic
    var body_type = Int(state[base + IDX_BODY_TYPE])
    if body_type != BODY_DYNAMIC:
        return

    # Integrate linear position: x += dt * v
    state[base + IDX_PX] += dt * state[base + IDX_VX]
    state[base + IDX_PY] += dt * state[base + IDX_VY]
    state[base + IDX_PZ] += dt * state[base + IDX_VZ]

    # Integrate orientation using quaternion
    var q = Quat(
        state[base + IDX_QW],
        state[base + IDX_QX],
        state[base + IDX_QY],
        state[base + IDX_QZ],
    )
    var omega = Vec3(
        state[base + IDX_WX],
        state[base + IDX_WY],
        state[base + IDX_WZ],
    )

    var q_new = integrate_quaternion(q, omega, dt)

    # Store normalized quaternion
    state[base + IDX_QW] = q_new.w
    state[base + IDX_QX] = q_new.x
    state[base + IDX_QY] = q_new.y
    state[base + IDX_QZ] = q_new.z


struct SemiImplicitEuler3D:
    """Semi-implicit Euler integrator for 3D rigid body dynamics.

    This integrator:
    1. Updates velocities with forces/gravity
    2. Updates positions with new velocities (semi-implicit)

    The semi-implicit scheme is more stable than explicit Euler
    while being simple and fast.
    """

    var gravity: Vec3
    var dt: Float64
    var damping_linear: Float64
    var damping_angular: Float64

    fn __init__(
        out self,
        gravity: Vec3 = Vec3(0.0, 0.0, DEFAULT_GRAVITY_Z_3D),
        dt: Float64 = 1.0 / 60.0,
        damping_linear: Float64 = 0.01,
        damping_angular: Float64 = 0.01,
    ):
        """Initialize the integrator.

        Args:
            gravity: Gravity acceleration vector.
            dt: Time step.
            damping_linear: Linear velocity damping (0-1).
            damping_angular: Angular velocity damping (0-1).
        """
        self.gravity = gravity
        self.dt = dt
        self.damping_linear = damping_linear
        self.damping_angular = damping_angular

    fn step(self, mut state: List[Float64], num_bodies: Int):
        """Perform one integration step for all bodies.

        Args:
            state: Physics state array (modified in place).
            num_bodies: Number of bodies in state.
        """
        for i in range(num_bodies):
            self._step_body(state, i)

    fn _step_body(self, mut state: List[Float64], body_idx: Int):
        """Integrate a single body.

        Args:
            state: Physics state array.
            body_idx: Index of body.
        """
        var base = body_idx * BODY_STATE_SIZE_3D

        # Check if body is dynamic
        var body_type = Int(state[base + IDX_BODY_TYPE])
        if body_type != BODY_DYNAMIC:
            return

        # Step 1: Integrate velocities
        integrate_velocities_3d(state, body_idx, self.gravity, self.dt)

        # Apply damping
        var damp_lin = 1.0 - self.damping_linear
        var damp_ang = 1.0 - self.damping_angular

        state[base + IDX_VX] *= damp_lin
        state[base + IDX_VY] *= damp_lin
        state[base + IDX_VZ] *= damp_lin
        state[base + IDX_WX] *= damp_ang
        state[base + IDX_WY] *= damp_ang
        state[base + IDX_WZ] *= damp_ang

        # Step 2: Integrate positions
        integrate_positions_3d(state, body_idx, self.dt)

    fn apply_impulse(
        self,
        mut state: List[Float64],
        body_idx: Int,
        impulse: Vec3,
        point: Vec3,
    ):
        """Apply an impulse to a body at a world-space point.

        Args:
            state: Physics state array.
            body_idx: Index of body.
            impulse: Impulse vector (world space).
            point: Application point (world space).
        """
        var base = body_idx * BODY_STATE_SIZE_3D

        # Check if body is dynamic
        var body_type = Int(state[base + IDX_BODY_TYPE])
        if body_type != BODY_DYNAMIC:
            return

        var inv_mass = state[base + IDX_INV_MASS]
        if inv_mass <= 0.0:
            return

        # Get body position
        var pos = Vec3(
            state[base + IDX_PX],
            state[base + IDX_PY],
            state[base + IDX_PZ],
        )

        # Linear velocity change: dv = impulse / m
        state[base + IDX_VX] += impulse.x * inv_mass
        state[base + IDX_VY] += impulse.y * inv_mass
        state[base + IDX_VZ] += impulse.z * inv_mass

        # Angular velocity change: dw = I^-1 * (r x impulse)
        var r = point - pos
        var torque_impulse = r.cross(impulse)

        var inv_ixx = 1.0 / state[base + IDX_IXX] if state[base + IDX_IXX] > 0.0 else 0.0
        var inv_iyy = 1.0 / state[base + IDX_IYY] if state[base + IDX_IYY] > 0.0 else 0.0
        var inv_izz = 1.0 / state[base + IDX_IZZ] if state[base + IDX_IZZ] > 0.0 else 0.0

        state[base + IDX_WX] += torque_impulse.x * inv_ixx
        state[base + IDX_WY] += torque_impulse.y * inv_iyy
        state[base + IDX_WZ] += torque_impulse.z * inv_izz

    fn get_velocity_at_point(
        self,
        state: List[Float64],
        body_idx: Int,
        point: Vec3,
    ) -> Vec3:
        """Get velocity of body at a world-space point.

        Args:
            state: Physics state array.
            body_idx: Index of body.
            point: Query point (world space).

        Returns:
            Velocity at point (includes angular contribution).
        """
        var base = body_idx * BODY_STATE_SIZE_3D

        var pos = Vec3(
            state[base + IDX_PX],
            state[base + IDX_PY],
            state[base + IDX_PZ],
        )
        var vel = Vec3(
            state[base + IDX_VX],
            state[base + IDX_VY],
            state[base + IDX_VZ],
        )
        var omega = Vec3(
            state[base + IDX_WX],
            state[base + IDX_WY],
            state[base + IDX_WZ],
        )

        # v_point = v_cm + omega x r
        var r = point - pos
        return vel + omega.cross(r)
