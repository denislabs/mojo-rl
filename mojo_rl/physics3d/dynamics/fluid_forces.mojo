"""Passive fluid forces over per-field tensors (Stage-A, single-source).

Per-field port of `compute_fluid_forces` (dynamics/fluid_forces.mojo) — the
MuJoCo inertia-box fluid model (mj_inertiaBoxFluidModel, engine_passive.c):
per non-world body, approximate the shape as an equivalent box from the
diagonal inertia, compute viscous (Stokes) + pressure (quadratic) drag in the
body local frame, rotate back to world, transport the wrench to
subtree_com[rootid], and accumulate into `scratch.fnet` via Jᵀ (walking the
kinematic tree). Arithmetic verbatim from the legacy CPU routine.

Enabled only when meta density > 0 OR viscosity > 0 (early-out otherwise);
in this codebase that is the Swimmer environment (density=4000, viscosity=0.1).
Every other env has both zero, so this is a no-op there.

Inserted in the integrator passive seam right after fnet assembly
(qfrc - bias - damping - stiffness - frictionloss) and before the LDL solve,
matching the legacy euler/rk4/implicit integrators (`compute_fluid_forces`
call site, into f_net). The fields path already has all inputs current at the
seam: d.xvel/xangvel/xquat/xipos/subtree_com, m.bodies/joints/meta,
scratch.cdof, scratch.fnet.

Structural transformation (same as the other fields ports): serialized per
env. Operands (10): xvel, xangvel, xquat, xipos, subtree_com (data) + bodies,
joints, meta (model) + cdof, fnet (scratch)."""

from std.math import sqrt, abs
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..kinematics.quat_math import quat_rotate
from ..joint_types import JNT_FREE, JNT_BALL
from ..fields import Data, Model, DynamicsScratch, Dims, DimsLike, AsStatic
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)

comptime FLUID_TPB: Int = 64


def _fluid_forces_env[
    DTYPE: DType,
    D: DimsLike,
    L_XVEL: Layout,
    L_XQUAT: Layout,
    L_BODIES: Layout,
    L_JOINTS: Layout,
    L_MMETA: Layout,
    L_CDOF: Layout,
    L_FNET: Layout,
](
    env: Int,
    dims: D,
    xvel: LayoutTensor[DTYPE, L_XVEL, MutAnyOrigin],
    xangvel: LayoutTensor[
        DTYPE, L_XVEL, MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, L_XQUAT, MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, L_XVEL, MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, L_XVEL, MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, L_BODIES, MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, L_JOINTS, MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, L_MMETA, MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, L_CDOF, MutAnyOrigin],
    fnet: LayoutTensor[DTYPE, L_FNET, MutAnyOrigin],
):
    """Inertia-box fluid drag for one env (verbatim from compute_fluid_forces,
    serialized per env)."""
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    var rho = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_DENSITY])
    var mu = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_VISCOSITY])

    # Early-out: no fluid forces when both density and viscosity are zero
    if rho <= Scalar[DTYPE](0) and mu <= Scalar[DTYPE](0):
        return

    comptime PI: Scalar[DTYPE] = 3.14159265358979323846

    for b in range(1, nbody):
        var mass = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
        if mass <= Scalar[DTYPE](1e-10):
            continue

        # --- 1. Equivalent box dimensions from diagonal inertia ---
        var Ixx = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IXX])
        var Iyy = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IYY])
        var Izz = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IZZ])

        var bx2 = Scalar[DTYPE](6) * (Iyy + Izz - Ixx) / mass
        var by2 = Scalar[DTYPE](6) * (Ixx + Izz - Iyy) / mass
        var bz2 = Scalar[DTYPE](6) * (Ixx + Iyy - Izz) / mass

        var bx = sqrt(max(bx2, Scalar[DTYPE](0)))
        var by = sqrt(max(by2, Scalar[DTYPE](0)))
        var bz = sqrt(max(bz2, Scalar[DTYPE](0)))

        # --- 2. Body world velocity (at body frame origin) ---
        var vx_w = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 0])
        var vy_w = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 1])
        var vz_w = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 2])
        var wx_w = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 0])
        var wy_w = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 1])
        var wz_w = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 2])

        # --- 3. Rotate velocity to body local frame (conjugate quat) ---
        var qx = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 0])
        var qy = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 1])
        var qz = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 2])
        var qw = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 3])

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

        # Viscous (Stokes) drag — linear in velocity
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

        # Pressure (quadratic) drag
        if rho > Scalar[DTYPE](0):
            var half_rho = Scalar[DTYPE](0.5) * rho
            lfx = lfx - half_rho * by * bz * abs(vx) * vx
            lfy = lfy - half_rho * bx * bz * abs(vy) * vy
            lfz = lfz - half_rho * bx * by * abs(vz) * vz

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

        # --- 7. Apply wrench at xipos via Jacobian transpose ---
        var px = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 0])
        var py = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 1])
        var pz = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 2])

        # subtree_com reference point for this body's root (fields path always
        # has subtree_com computed — the legacy has_stcom branch)
        var root = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_ROOTID]))
        var ref_x = rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 0])
        var ref_y = rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 1])
        var ref_z = rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 2])

        # Transport torque from xipos to subtree_com reference
        var dx = px - ref_x
        var dy = py - ref_y
        var dz = pz - ref_z
        var tau_ox = tx_w + dy * fz_w - dz * fy_w
        var tau_oy = ty_w + dz * fx_w - dx * fz_w
        var tau_oz = tz_w + dx * fy_w - dy * fx_w

        # Walk the kinematic tree from body b to root, accumulating via cdof
        var body = b
        while body > 0:
            for j in range(njoint):
                if Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])) != (
                    body
                ):
                    continue
                var dof_adr = Int(
                    rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
                )
                var jtype = Int(
                    rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE])
                )
                var ndof = 1
                if jtype == JNT_FREE:
                    ndof = 6
                elif jtype == JNT_BALL:
                    ndof = 3

                for d in range(ndof):
                    var di = dof_adr + d
                    # cdof layout per DOF: [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
                    var ca0 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 0])
                    var ca1 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 1])
                    var ca2 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 2])
                    var cl0 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 3])
                    var cl1 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 4])
                    var cl2 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 5])
                    fnet[env, di] = (
                        rebind[Scalar[DTYPE]](fnet[env, di])
                        + cl0 * fx_w
                        + cl1 * fy_w
                        + cl2 * fz_w
                        + ca0 * tau_ox
                        + ca1 * tau_oy
                        + ca2 * tau_oz
                    )

            body = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_PARENT]))


def _fluid_forces_fields_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    xvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xangvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    fnet: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _fluid_forces_env[DTYPE](
        env, Dims[nv=NV, nbody=NBODY, njoint=NJOINT](), xvel, xangvel, xquat, xipos, subtree_com, bodies, joints, mmeta,
        cdof, fnet,
    )


def compute_fluid_forces[

    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Accumulate inertia-box fluid drag into `scratch.fnet`, both targets, one
    body. No-op when meta density and viscosity are both zero (early-out inside
    the env helper). Call in the passive seam after fnet assembly, before the
    LDL solve."""
    comptime L_NV = Layout.row_major(BATCH, D.NV)
    comptime L_B3 = Layout.row_major(BATCH, D.NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, D.NBODY * 4)
    comptime L_BODY = Layout.row_major(D.NBODY, MODEL_BODY_SIZE)
    comptime L_JOINT = Layout.row_major(D.NJOINT, MODEL_JOINT_SIZE)
    comptime L_META = Layout.row_major(MODEL_META_SIZE)
    comptime L_CDOF = Layout.row_major(BATCH, D.NV * 6)

    comptime if target == "cpu":
        var xvel_v = d.xvel.lt["cpu", L_B3]()
        var xangvel_v = d.xangvel.lt["cpu", L_B3]()
        var xquat_v = d.xquat.lt["cpu", L_B4]()
        var xipos_v = d.xipos.lt["cpu", L_B3]()
        var stcom_v = d.subtree_com.lt["cpu", L_B3]()
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var joints_v = m.joints.lt["cpu", L_JOINT]()
        var meta_v = m.meta.lt["cpu", L_META]()
        var cdof_v = scratch.cdof.lt["cpu", L_CDOF]()
        var fnet_v = scratch.fnet.lt["cpu", L_NV]()
        for e in range(BATCH):
            _fluid_forces_env[DTYPE](
                e, AsStatic[D](), xvel_v, xangvel_v, xquat_v, xipos_v, stcom_v, bodies_v,
                joints_v, meta_v, cdof_v, fnet_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + FLUID_TPB - 1) // FLUID_TPB
        c.enqueue_function[
            _fluid_forces_fields_kernel[DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH]
        ](
            d.xvel.lt["gpu", L_B3](),
            d.xangvel.lt["gpu", L_B3](),
            d.xquat.lt["gpu", L_B4](),
            d.xipos.lt["gpu", L_B3](),
            d.subtree_com.lt["gpu", L_B3](),
            m.bodies.lt["gpu", L_BODY](),
            m.joints.lt["gpu", L_JOINT](),
            m.meta.lt["gpu", L_META](),
            scratch.cdof.lt["gpu", L_CDOF](),
            scratch.fnet.lt["gpu", L_NV](),
            grid_dim=(BLOCKS,),
            block_dim=(FLUID_TPB,),
        )
