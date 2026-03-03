"""Fluid dynamics forces - inertia-box model (MuJoCo engine_passive.c).

Implements mj_inertiaBoxFluidModel from engine_passive.c lines 701-757:

For each body, approximate shape as an equivalent box derived from the diagonal
inertia tensor. Compute viscous (Stokes) and pressure (quadratic) drag forces in
the body local frame, rotate back to world frame, then apply via Jacobian transpose.

  box_x = sqrt(6 * (Iyy + Izz - Ixx) / mass)   (box half-extent along body x)
  box_y = sqrt(6 * (Ixx + Izz - Iyy) / mass)
  box_z = sqrt(6 * (Ixx + Iyy - Izz) / mass)

Viscous drag (Stokes, linear in velocity):
  F_lin = -3*π*diam*μ * v_local      where diam = (bx+by+bz)/3
  T_ang = -π*diam³*μ * ω_local

Pressure drag (quadratic in velocity):
  F_x = -0.5*ρ*(by*bz)*|vx|*vx  (cross-section = by*bz)
  T_x = -ρ*bx*(by⁴+bz⁴)*|ωx|*ωx / 64

Enabled when model.opt_density > 0 OR model.opt_viscosity > 0.
Used by: Swimmer environment (density=4000, viscosity=0.1).

Reference: MuJoCo 3.3.6 src/engine/engine_passive.c, mj_inertiaBoxFluidModel.
"""

from math import sqrt, abs

from ..types import Model, Data, _max_one, ConeType
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import quat_rotate


fn compute_fluid_forces[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
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
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
    cdof: List[Scalar[DTYPE]],
    mut f_net: List[Scalar[DTYPE]],
):
    """Apply inertia-box fluid forces to f_net (MuJoCo-matching).

    Computes per-body viscous and pressure drag forces in the body's local frame
    using equivalent box dimensions from the diagonal inertia tensor. Forces are
    applied to f_net via the Jacobian transpose (walking the kinematic tree).

    Matches MuJoCo's mj_inertiaBoxFluidModel behavior.

    Args:
        model: Static model configuration (must have opt_density or opt_viscosity > 0).
        data: Mutable simulation state (xquat, xvel, xangvel, xipos must be current).
        cdof: Spatial motion subspace matrix (NV x 6), [ang; lin] per DOF.
        f_net: Generalized force vector to accumulate fluid forces into.
    """
    var rho = model.opt_density
    var mu = model.opt_viscosity

    # Early-out: no fluid forces when both density and viscosity are zero
    if rho <= Scalar[DTYPE](0) and mu <= Scalar[DTYPE](0):
        return

    comptime PI: Scalar[DTYPE] = 3.14159265358979323846

    # Process each non-world body
    for b in range(1, NBODY):
        var mass = model.body_mass[b]
        if mass <= Scalar[DTYPE](1e-10):
            continue

        # --- 1. Equivalent box dimensions from diagonal inertia (MuJoCo formula) ---
        # For a uniform box: Ixx = m*(by²+bz²)/12 => bx² = 6*(Iyy+Izz-Ixx)/m
        var Ixx = model.body_inertia[b * 3 + 0]
        var Iyy = model.body_inertia[b * 3 + 1]
        var Izz = model.body_inertia[b * 3 + 2]

        var bx2 = Scalar[DTYPE](6) * (Iyy + Izz - Ixx) / mass
        var by2 = Scalar[DTYPE](6) * (Ixx + Izz - Iyy) / mass
        var bz2 = Scalar[DTYPE](6) * (Ixx + Iyy - Izz) / mass

        # Guard against negative values (non-physical or diagonal inertia)
        var bx = sqrt(max(bx2, Scalar[DTYPE](0)))
        var by = sqrt(max(by2, Scalar[DTYPE](0)))
        var bz = sqrt(max(bz2, Scalar[DTYPE](0)))

        # --- 2. Body world velocity (at body frame origin, approximating CoM) ---
        var vx_w = data.xvel[b * 3 + 0]
        var vy_w = data.xvel[b * 3 + 1]
        var vz_w = data.xvel[b * 3 + 2]
        var wx_w = data.xangvel[b * 3 + 0]
        var wy_w = data.xangvel[b * 3 + 1]
        var wz_w = data.xangvel[b * 3 + 2]

        # --- 3. Rotate velocity to body local frame ---
        # xquat = [qx,qy,qz,qw] is body-to-world rotation.
        # Inverse (world-to-body) uses conjugate: [-qx,-qy,-qz,qw].
        var qx = data.xquat[b * 4 + 0]
        var qy = data.xquat[b * 4 + 1]
        var qz = data.xquat[b * 4 + 2]
        var qw = data.xquat[b * 4 + 3]

        var vloc = quat_rotate[DTYPE](-qx, -qy, -qz, qw, vx_w, vy_w, vz_w)
        var wloc = quat_rotate[DTYPE](-qx, -qy, -qz, qw, wx_w, wy_w, wz_w)
        var vx = vloc[0]
        var vy = vloc[1]
        var vz = vloc[2]
        var wx = wloc[0]
        var wy = wloc[1]
        var wz = wloc[2]

        # --- 4. Equivalent sphere diameter for Stokes drag ---
        var diam = (bx + by + bz) / Scalar[DTYPE](3)

        # --- 5. Accumulate local-frame forces and torques ---
        var lfx = Scalar[DTYPE](0)
        var lfy = Scalar[DTYPE](0)
        var lfz = Scalar[DTYPE](0)
        var ltx = Scalar[DTYPE](0)
        var lty = Scalar[DTYPE](0)
        var ltz = Scalar[DTYPE](0)

        # Viscous (Stokes) drag - linear in velocity (engine_passive.c lines 725-734)
        if mu > Scalar[DTYPE](0):
            var visc_lin = Scalar[DTYPE](3) * PI * diam * mu
            lfx = lfx - visc_lin * vx
            lfy = lfy - visc_lin * vy
            lfz = lfz - visc_lin * vz

            var d3 = diam * diam * diam
            var visc_ang = PI * d3 * mu
            ltx = ltx - visc_ang * wx
            lty = lty - visc_ang * wy
            ltz = ltz - visc_ang * wz

        # Pressure (quadratic) drag (engine_passive.c lines 737-750)
        if rho > Scalar[DTYPE](0):
            # Linear drag: F_i = -0.5*rho*A_i*|v_i|*v_i
            var half_rho = Scalar[DTYPE](0.5) * rho
            lfx = lfx - half_rho * by * bz * abs(vx) * vx
            lfy = lfy - half_rho * bx * bz * abs(vy) * vy
            lfz = lfz - half_rho * bx * by * abs(vz) * vz

            # Angular drag: T_x = -rho*bx*(by⁴+bz⁴)*|wx|*wx / 64
            var bx4 = bx * bx * bx * bx
            var by4 = by * by * by * by
            var bz4 = bz * bz * bz * bz
            ltx = ltx - rho * bx * (by4 + bz4) * abs(wx) * wx / Scalar[DTYPE](
                64
            )
            lty = lty - rho * by * (bx4 + bz4) * abs(wy) * wy / Scalar[DTYPE](
                64
            )
            ltz = ltz - rho * bz * (bx4 + by4) * abs(wz) * wz / Scalar[DTYPE](
                64
            )

        # --- 6. Rotate forces/torques back to world frame ---
        var fw = quat_rotate[DTYPE](qx, qy, qz, qw, lfx, lfy, lfz)
        var tw = quat_rotate[DTYPE](qx, qy, qz, qw, ltx, lty, ltz)
        var fx_w = fw[0]
        var fy_w = fw[1]
        var fz_w = fw[2]
        var tx_w = tw[0]
        var ty_w = tw[1]
        var tz_w = tw[2]

        # --- 7. Apply wrench at CoM (xipos) via Jacobian transpose ---
        # Principle of virtual work: qfrc[d] += J[d]^T * F_spatial
        # For force f at point p and torque tau:
        #   qfrc[d] += dot(cdof[d,3:6], f) + dot(cdof[d,0:3], tau + p×f)
        # where cdof[:,0:3] = angular component, cdof[:,3:6] = linear component.
        var px = data.xipos[b * 3 + 0]
        var py = data.xipos[b * 3 + 1]
        var pz = data.xipos[b * 3 + 2]

        # Transport torque to world origin: tau_origin = tau + p × f
        var tau_ox = tx_w + py * fz_w - pz * fy_w
        var tau_oy = ty_w + pz * fx_w - px * fz_w
        var tau_oz = tz_w + px * fy_w - py * fx_w

        # Walk kinematic tree from body b to root (worldbody = 0)
        var body = b
        while body > 0:
            # Find joints belonging to this ancestor body and accumulate
            for j in range(model.num_joints):
                if model.joints[j].body_id != body:
                    continue
                var joint = model.joints[j]
                var dof_adr = joint.dof_adr
                var ndof = 1
                if joint.jnt_type == JNT_FREE:
                    ndof = 6
                elif joint.jnt_type == JNT_BALL:
                    ndof = 3

                for d in range(ndof):
                    var di = dof_adr + d
                    # cdof layout per DOF: [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
                    var ca0 = cdof[di * 6 + 0]
                    var ca1 = cdof[di * 6 + 1]
                    var ca2 = cdof[di * 6 + 2]
                    var cl0 = cdof[di * 6 + 3]
                    var cl1 = cdof[di * 6 + 4]
                    var cl2 = cdof[di * 6 + 5]
                    f_net[di] = (
                        f_net[di]
                        + cl0 * fx_w
                        + cl1 * fy_w
                        + cl2 * fz_w
                        + ca0 * tau_ox
                        + ca1 * tau_oy
                        + ca2 * tau_oz
                    )

            body = model.body_parent[body]
