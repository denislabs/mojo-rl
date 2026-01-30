"""Physics3D v2 integrator - Semi-implicit Euler integration.

Semi-implicit Euler (also called symplectic Euler):
1. Update velocities using current accelerations
2. Update positions using NEW velocities

This is more stable than explicit Euler for oscillatory systems.
"""

from .types import Model, Data
from math import sqrt


fn integrate[DTYPE: DType](model: Model[DTYPE], mut data: Data[DTYPE]):
    """Semi-implicit Euler: update vel then pos.

    Step 1: v(t+dt) = v(t) + dt * a(t)
    Step 2: x(t+dt) = x(t) + dt * v(t+dt)  <- uses NEW velocity
    Step 3: Quaternion integration with normalization
    """
    var dt = model.timestep

    # 1. Update velocities (using current accelerations)
    data.qvel[0] += dt * data.qacc[0]
    data.qvel[1] += dt * data.qacc[1]
    data.qvel[2] += dt * data.qacc[2]
    data.qvel[3] += dt * data.qacc[3]
    data.qvel[4] += dt * data.qacc[4]
    data.qvel[5] += dt * data.qacc[5]

    # 2. Update positions (using NEW velocities - semi-implicit)
    data.qpos[0] += dt * data.qvel[0]
    data.qpos[1] += dt * data.qvel[1]
    data.qpos[2] += dt * data.qvel[2]

    # 3. Quaternion integration: q' = q + 0.5*dt*ω⊗q
    # Using Hamilton convention where q_dot = 0.5 * [ω, 0] ⊗ q
    var half_dt = Scalar[DTYPE](0.5) * dt
    var wx = data.qvel[3]
    var wy = data.qvel[4]
    var wz = data.qvel[5]
    var qx = data.qpos[3]
    var qy = data.qpos[4]
    var qz = data.qpos[5]
    var qw = data.qpos[6]

    # Quaternion derivative using Hamilton product
    # d/dt[q] = 0.5 * [wx, wy, wz, 0] ⊗ [qx, qy, qz, qw]
    # Result: [qx', qy', qz', qw']
    data.qpos[3] += half_dt * (wx * qw + wy * qz - wz * qy)
    data.qpos[4] += half_dt * (-wx * qz + wy * qw + wz * qx)
    data.qpos[5] += half_dt * (wx * qy - wy * qx + wz * qw)
    data.qpos[6] += half_dt * (-wx * qx - wy * qy - wz * qz)

    # 4. Normalize quaternion to prevent drift
    _normalize_quat(data)


fn _normalize_quat[DTYPE: DType](mut data: Data[DTYPE]):
    """Normalize quaternion stored at qpos[3:7]."""
    var qx = data.qpos[3]
    var qy = data.qpos[4]
    var qz = data.qpos[5]
    var qw = data.qpos[6]
    var norm_sq = qx * qx + qy * qy + qz * qz + qw * qw

    if norm_sq > Scalar[DTYPE](1e-10):
        var inv_norm = Scalar[DTYPE](1.0) / sqrt(norm_sq)
        data.qpos[3] *= inv_norm
        data.qpos[4] *= inv_norm
        data.qpos[5] *= inv_norm
        data.qpos[6] *= inv_norm
