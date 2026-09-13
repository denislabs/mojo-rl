"""OSC_POSE — robosuite 1.4.0's operational-space controller, on the CPU. L4.

    var osc = OscPose(dof, qadr, site, site_body, tmin, tmax, OscPoseConfig())
    osc.update(d, m, scratch)          # every substep, after the state is current
    osc.set_goal(action6)              # on a POLICY step only (25 substeps apart)
    var tau = osc.run()                # 7 torques, clipped to the actuators' ctrlrange
    var g = PandaGripperRamp(...)
    var ctrl = g.step(action[6])       # every substep — the ramp is per SIM step

## WHAT THIS IS, QUOTED

`robosuite-1.4.0/robosuite/controllers/osc.py` + `base_controller.py` +
`utils/control_utils.py`, the version LIBERO pins (`requirements.txt`); the
1.5 tree beside LIBERO moved the goal into a base frame and is NOT what
recorded the demos. The law, per `run_controller`:

    F  = kp (goal_pos - ee_pos) - kd ee_vel_pos
    T  = kp ori_error(goal_R, ee_R) - kd ee_vel_ori
    Λp = pinv(Jp M⁻¹ Jpᵀ)      Λo = pinv(Jr M⁻¹ Jrᵀ)      (uncouple_pos_ori)
    Λ  = pinv(J M⁻¹ Jᵀ)        J̄ = M⁻¹ Jᵀ Λ                N = I - J̄ J
    τ  = Jᵀ [Λp F; Λo T] + qfrc_bias + Nᵀ M (10 (q₀ - q) - 2√10 q̇)
    τ  = clip(τ, ctrlrange)

with `kp = 150`, `kd = 2√kp` (damping_ratio 1), M and J the 7-dof arm
BLOCKS of the full matrices (`mass_matrix[qvel_index][:, qvel_index]`, then
`np.linalg.inv` of THAT block — not the full inverse restricted), and
`ee_vel = J @ qvel` (`binding_utils.get_site_xvelp/xvelr` are exactly that,
not `mj_objectVelocity`). `q₀` is the joint configuration at controller
construction, which `Robot.reset` does AFTER writing `init_qpos` (+ noise),
so it is the reset pose.

`set_goal(action)`: `scale_action` maps `clip(a, -1, 1)` to
`±[0.05 m, 0.5 rad]`; `goal_pos = ee_pos + δp`; `goal_R = R(δθ) @ ee_R`
— BUT ONLY WHEN SOME δθ COMPONENT IS NOT EXACTLY ZERO (`math.isclose(elem,
0.0)` has no absolute tolerance, so 1e-12 counts as a rotation and 0.0
does not), else the previous goal orientation is kept. The goal is set
once per policy step and held for the 25 substeps; `update()` re-reads
the state on every substep.

`orientation_error(desired, current)` is `0.5 Σ_i current[:, i] ×
desired[:, i]`, columns.

## ⚠ WHAT IS NOT VERBATIM, AND WHY IT DOES NOT MATTER AT THIS PRECISION

* `pinv` is `np.linalg.pinv` (SVD, rcond 1e-15). At any configuration
  where the 3x3 / 6x6 operational-space inertia is full rank — every
  configuration a manipulation demo visits — pinv IS inv, and this uses a
  partial-pivot Gauss-Jordan inverse that RAISES on a pivot below
  `SINGULAR_PIVOT` rather than returning a pseudo-inverse silently.
* `T.quat2mat` casts the delta-rotation quaternion to float32 before
  forming the matrix; the goal rotation therefore carries ~1e-7 relative
  error in robosuite. This forms it in float64 (Rodrigues).
* The demo replay (`tools/tasks/libero_demo_replay.py`) is where those
  two claims are measured, against MuJoCo running the same transcription
  on the same model and against the recorded trajectory.

## THE GRIPPER

`PandaGripper.format_action` (1.4.0): `current += [-1, +1] * 0.01 *
sign(a)`, clipped to ±1, CALLED EVERY SIM STEP (`Robot.control` runs per
substep; only `set_goal` is gated on `policy_step`) — so a policy step of
25 substeps moves it by 0.25: neutral→closed takes 4 policy steps, fully
open→fully closed 8. Then
`ctrl = bias + weight * current` with `bias/weight = ctrlrange
midpoint/half-width`: finger 1 `(0, 0.04)`, finger 2 `(-0.04, 0)`.
`speed` was quoted as 0.2 per policy step in the assessment; it is 0.01
per SUBSTEP in the pinned version, which is 0.25 per policy step.

⚠ THE MASS MATRIX MUST INCLUDE THE ARMATURE. `mj_fullM` returns `qM`,
which has `dof_armature` on its diagonal; our CRBA leaves it out and the
integrator adds it in a separate pass. `update()` adds it the same way.
The demo replay's substep trace is what showed a 4.2 N m torque gap at an
identical state before this was done.
"""

from std.math import sqrt, sin, cos

from ..fields import (
    Data, Model, DynamicsScratch, DynDims, DYN1, DYN2, rl1, rl2,
)
from ..fields.scratch import Scratch, cap
from ..kinematics.forward_kinematics import forward_kinematics
from ..kinematics.site_frame import site_world_quat_list
from ..gpu.constants import (
    MODEL_BODY_SIZE, MODEL_JOINT_SIZE, MODEL_META_SIZE, MODEL_SITE_SIZE,
)
from .subtree_com import compute_subtree_com
from .cdof import compute_cdof
from .jac_point import jac_site
from .mass_matrix import compute_mass_matrix
from .rne import compute_bias_forces_rne
from ..integrator.euler import _armature_env


comptime DT = DType.float64
comptime ARM_DOF: Int = 7
comptime SINGULAR_PIVOT: Float64 = 1e-12


struct OscPoseConfig(Copyable, ImplicitlyCopyable, Movable):
    """`controllers/config/osc_pose.json`, the file LIBERO loads."""

    var kp: Float64
    var damping_ratio: Float64
    var output_max_pos: Float64
    var output_max_ori: Float64
    var nullspace_kp: Float64

    def __init__(out self):
        self.kp = 150.0
        self.damping_ratio = 1.0
        self.output_max_pos = 0.05
        self.output_max_ori = 0.5
        self.nullspace_kp = 10.0


# ── small dense algebra, row-major Lists ───────────────────────────────────


def mat_inverse(a: List[Float64], n: Int) raises -> List[Float64]:
    """Gauss-Jordan with partial pivoting. RAISES on a pivot below
    `SINGULAR_PIVOT` — see the header on `pinv`."""
    var w = List[Float64]()
    for i in range(n):
        for j in range(n):
            w.append(a[i * n + j])
        for j in range(n):
            w.append(1.0 if i == j else 0.0)
    var cols = 2 * n
    for c in range(n):
        var piv = c
        var best = w[c * cols + c]
        if best < 0.0:
            best = -best
        for r in range(c + 1, n):
            var v = w[r * cols + c]
            if v < 0.0:
                v = -v
            if v > best:
                best = v
                piv = r
        if best < SINGULAR_PIVOT:
            raise Error(
                "osc: singular matrix (pivot " + String(best) + " at column "
                + String(c) + " of " + String(n) + "). robosuite's pinv would"
                " return a pseudo-inverse here; this controller refuses rather"
                " than guess — the arm is at a singular configuration."
            )
        if piv != c:
            for j in range(cols):
                var t = w[c * cols + j]
                w[c * cols + j] = w[piv * cols + j]
                w[piv * cols + j] = t
        var p = w[c * cols + c]
        for j in range(cols):
            w[c * cols + j] /= p
        for r in range(n):
            if r == c:
                continue
            var f = w[r * cols + c]
            if f == 0.0:
                continue
            for j in range(cols):
                w[r * cols + j] -= f * w[c * cols + j]
    var out = List[Float64]()
    for i in range(n):
        for j in range(n):
            out.append(w[i * cols + n + j])
    return out^


def mat_mul(
    a: List[Float64], b: List[Float64], n: Int, k: Int, m: Int
) -> List[Float64]:
    """(n x k) @ (k x m)."""
    var out = List[Float64]()
    for i in range(n):
        for j in range(m):
            var s = 0.0
            for t in range(k):
                s += a[i * k + t] * b[t * m + j]
            out.append(s)
    return out^


def mat_t(a: List[Float64], n: Int, m: Int) -> List[Float64]:
    """(n x m) -> (m x n)."""
    var out = List[Float64]()
    for j in range(m):
        for i in range(n):
            out.append(a[i * m + j])
    return out^


def quat_to_mat(qx: Float64, qy: Float64, qz: Float64, qw: Float64) -> List[Float64]:
    """Row-major 3x3 of a unit (x, y, z, w) quaternion — `site_xmat`."""
    var out = List[Float64]()
    out.append(1.0 - 2.0 * (qy * qy + qz * qz))
    out.append(2.0 * (qx * qy - qz * qw))
    out.append(2.0 * (qx * qz + qy * qw))
    out.append(2.0 * (qx * qy + qz * qw))
    out.append(1.0 - 2.0 * (qx * qx + qz * qz))
    out.append(2.0 * (qy * qz - qx * qw))
    out.append(2.0 * (qx * qz - qy * qw))
    out.append(2.0 * (qy * qz + qx * qw))
    out.append(1.0 - 2.0 * (qx * qx + qy * qy))
    return out^


def axisangle_to_mat(vx: Float64, vy: Float64, vz: Float64) -> List[Float64]:
    """`T.quat2mat(T.axisangle2quat(v))` in float64 (Rodrigues)."""
    var angle = sqrt(vx * vx + vy * vy + vz * vz)
    var out = List[Float64]()
    if angle == 0.0:
        for i in range(3):
            for j in range(3):
                out.append(1.0 if i == j else 0.0)
        return out^
    var kx = vx / angle
    var ky = vy / angle
    var kz = vz / angle
    var s = sin(angle)
    var c = 1.0 - cos(angle)
    # R = I + s K + c K^2, K the cross-product matrix of k
    out.append(1.0 + c * (-ky * ky - kz * kz))
    out.append(-s * kz + c * kx * ky)
    out.append(s * ky + c * kx * kz)
    out.append(s * kz + c * kx * ky)
    out.append(1.0 + c * (-kx * kx - kz * kz))
    out.append(-s * kx + c * ky * kz)
    out.append(-s * ky + c * kx * kz)
    out.append(s * kx + c * ky * kz)
    out.append(1.0 + c * (-kx * kx - ky * ky))
    return out^


def orientation_error(desired: List[Float64], current: List[Float64]) -> List[Float64]:
    """`control_utils.orientation_error`: 0.5 * sum_i current[:, i] x
    desired[:, i], over the three COLUMNS."""
    var ex = 0.0
    var ey = 0.0
    var ez = 0.0
    for i in range(3):
        var cx = current[0 * 3 + i]
        var cy = current[1 * 3 + i]
        var cz = current[2 * 3 + i]
        var dx = desired[0 * 3 + i]
        var dy = desired[1 * 3 + i]
        var dz = desired[2 * 3 + i]
        ex += cy * dz - cz * dy
        ey += cz * dx - cx * dz
        ez += cx * dy - cy * dx
    var out = List[Float64]()
    out.append(0.5 * ex)
    out.append(0.5 * ey)
    out.append(0.5 * ez)
    return out^


# ── the controller ─────────────────────────────────────────────────────────


struct OscPose(Movable & Deinitable):
    var dof: List[Int]
    """Velocity-space indices of the seven arm joints, in order."""
    var qadr: List[Int]
    """Their qpos indices."""
    var site: Int
    var site_body: Int
    var torque_min: List[Float64]
    var torque_max: List[Float64]
    var kp: Float64
    var kd: Float64
    var out_pos: Float64
    var out_ori: Float64
    var ns_kp: Float64
    var ns_kv: Float64

    var initial_joint: List[Float64]
    var goal_pos: List[Float64]
    var goal_mat: List[Float64]

    # the state `update` reads, kept between update() and run()
    var ee_pos: List[Float64]
    var ee_mat: List[Float64]
    var ee_vel: List[Float64]        # 6: linear then angular
    var J: List[Float64]             # 6 x 7, row-major
    var M: List[Float64]             # 7 x 7
    var bias: List[Float64]          # 7
    var q: List[Float64]
    var qd: List[Float64]
    var torques: List[Float64]       # the last run()'s output, unclipped

    def __init__(
        out self,
        var dof: List[Int], var qadr: List[Int], site: Int, site_body: Int,
        var torque_min: List[Float64], var torque_max: List[Float64],
        cfg: OscPoseConfig,
    ) raises:
        if len(dof) != ARM_DOF or len(qadr) != ARM_DOF:
            raise Error("osc: OSC_POSE drives a 7-dof arm")
        self.dof = dof^
        self.qadr = qadr^
        self.site = site
        self.site_body = site_body
        self.torque_min = torque_min^
        self.torque_max = torque_max^
        self.kp = cfg.kp
        self.kd = 2.0 * sqrt(cfg.kp) * cfg.damping_ratio
        self.out_pos = cfg.output_max_pos
        self.out_ori = cfg.output_max_ori
        self.ns_kp = cfg.nullspace_kp
        self.ns_kv = 2.0 * sqrt(cfg.nullspace_kp)
        self.initial_joint = List[Float64]()
        self.goal_pos = List[Float64]()
        self.goal_mat = List[Float64]()
        self.ee_pos = List[Float64]()
        self.ee_mat = List[Float64]()
        self.ee_vel = List[Float64]()
        self.J = List[Float64]()
        self.M = List[Float64]()
        self.bias = List[Float64]()
        self.q = List[Float64]()
        self.qd = List[Float64]()
        self.torques = List[Float64]()

    def __init__(out self, *, deinit move: Self):
        self.dof = move.dof^
        self.qadr = move.qadr^
        self.site = move.site
        self.site_body = move.site_body
        self.torque_min = move.torque_min^
        self.torque_max = move.torque_max^
        self.kp = move.kp
        self.kd = move.kd
        self.out_pos = move.out_pos
        self.out_ori = move.out_ori
        self.ns_kp = move.ns_kp
        self.ns_kv = move.ns_kv
        self.initial_joint = move.initial_joint^
        self.goal_pos = move.goal_pos^
        self.goal_mat = move.goal_mat^
        self.ee_pos = move.ee_pos^
        self.ee_mat = move.ee_mat^
        self.ee_vel = move.ee_vel^
        self.J = move.J^
        self.M = move.M^
        self.bias = move.bias^
        self.q = move.q^
        self.qd = move.qd^
        self.torques = move.torques^

    def update(
        mut self,
        mut d: Data[DT, DynDims, 1],
        mut m: Model[DT, DynDims],
        mut scratch: DynamicsScratch[DT, DynDims, 1],
    ) raises:
        """`BaseController.update`: `sim.forward()` then read the eef pose,
        its velocity (J @ qvel), the arm's qpos/qvel, the 6 x 7 site
        Jacobian block, the 7 x 7 mass-matrix block and `qfrc_bias`."""
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        compute_subtree_com["cpu"](d, m)
        compute_cdof["cpu"](d, m, scratch)
        compute_mass_matrix["cpu", DT, DynDims, 1, False, True](d, m, scratch)
        compute_bias_forces_rne["cpu", DT, DynDims, 1](d, m, scratch)

        var dm = d.dims
        var nv = dm.get_nv()
        # ⚠⚠ CRBA LEAVES THE ARMATURE OUT; `mj_fullM` HAS IT IN. `scratch.M`
        # is the rigid-body inertia only — the integrator adds
        # `dof_armature` to the diagonal as a separate pass (euler.mojo
        # "6b") before factoring. robosuite's mass matrix is `mj_fullM(qM)`,
        # which is the SUM. Without this line joint 1's M was 2.29 vs 7.29
        # (armature 5.0) and the first torque was off by 4.2 N m at an
        # identical state — the demo replay's substep trace found it.
        var rl_JNT0 = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_M = rl2(1, nv * nv)
        var joints_a = m.joints.lt_dyn["cpu", DYN2](rl_JNT0)
        var M_v = scratch.M.lt_dyn["cpu", DYN2](rl_M)
        _armature_env[DT](0, dm, joints_a, M_v)
        var rl_NB3 = rl2(1, dm.get_nbody() * 3)
        var rl_JNT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_BOD = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_MET = rl1(MODEL_META_SIZE)
        var rl_CDOF = rl2(1, nv * 6)
        var rl_SITE = rl2(dm.get_nsite(), MODEL_SITE_SIZE)
        var rl_SX = rl2(1, dm.get_nsite() * 3)
        var joints_v = m.joints.lt_dyn["cpu", DYN2](rl_JNT)
        var bodies_v = m.bodies.lt_dyn["cpu", DYN2](rl_BOD)
        var sites_v = m.sites.lt_dyn["cpu", DYN2](rl_SITE)
        var mmeta_v = m.meta.lt_dyn["cpu", DYN1](rl_MET)
        var subtree_v = d.subtree_com.lt_dyn["cpu", DYN2](rl_NB3)
        var cdof_v = scratch.cdof.lt_dyn["cpu", DYN2](rl_CDOF)
        var sxpos_v = d.site_xpos.lt_dyn["cpu", DYN2](rl_SX)
        var jp = Scratch[Scalar[DT], 3 * cap[DynDims.NV]()](
            3 * nv, fill=Scalar[DT](0)
        )
        var jr = Scratch[Scalar[DT], 3 * cap[DynDims.NV]()](
            3 * nv, fill=Scalar[DT](0)
        )
        jac_site[DT, cap[DynDims.NV]()](
            0, subtree_v, joints_v, bodies_v, mmeta_v, cdof_v,
            sites_v, sxpos_v, self.site, jp, jr, nv,
        )

        # eef pose
        self.ee_pos = List[Float64]()
        for k in range(3):
            self.ee_pos.append(Float64(d.site_xpos.data[self.site * 3 + k]))
        var sq = site_world_quat_list[DT](
            m.sites.data, d.xquat.data, self.site_body, self.site
        )
        self.ee_mat = quat_to_mat(
            Float64(sq[0]), Float64(sq[1]), Float64(sq[2]), Float64(sq[3])
        )
        # the arm's J block, q, qd
        self.J = List[Float64]()
        for r in range(3):
            for a in range(ARM_DOF):
                self.J.append(Float64(jp[r * nv + self.dof[a]]))
        for r in range(3):
            for a in range(ARM_DOF):
                self.J.append(Float64(jr[r * nv + self.dof[a]]))
        self.q = List[Float64]()
        self.qd = List[Float64]()
        for a in range(ARM_DOF):
            self.q.append(Float64(d.qpos.data[self.qadr[a]]))
            self.qd.append(Float64(d.qvel.data[self.dof[a]]))
        # ee velocity = J_full @ qvel_full — over every dof, as the binding
        # does; the site's Jacobian is zero on dofs that do not move it
        self.ee_vel = List[Float64]()
        for r in range(3):
            var s = 0.0
            for i in range(nv):
                s += Float64(jp[r * nv + i]) * Float64(d.qvel.data[i])
            self.ee_vel.append(s)
        for r in range(3):
            var s = 0.0
            for i in range(nv):
                s += Float64(jr[r * nv + i]) * Float64(d.qvel.data[i])
            self.ee_vel.append(s)
        # M block and bias
        self.M = List[Float64]()
        for a in range(ARM_DOF):
            for b in range(ARM_DOF):
                self.M.append(Float64(scratch.M.data[self.dof[a] * nv + self.dof[b]]))
        self.bias = List[Float64]()
        for a in range(ARM_DOF):
            self.bias.append(Float64(scratch.bias.data[self.dof[a]]))

    def reset(mut self) raises:
        """`Robot.reset` -> `_load_controller`: the initial joints are the
        CURRENT ones (call `update` first) and the goal is the current eef
        pose (`OperationalSpaceController.__init__`'s last lines)."""
        if len(self.q) != ARM_DOF:
            raise Error("osc: reset() needs update() first")
        self.initial_joint = self.q.copy()
        self.goal_pos = self.ee_pos.copy()
        self.goal_mat = self.ee_mat.copy()

    def set_goal(mut self, action: List[Float64]) raises:
        """`set_goal(action)` on a policy step, after `update()`."""
        if len(action) < 6:
            raise Error("osc: set_goal needs six numbers")
        if len(self.ee_pos) != 3:
            raise Error("osc: set_goal needs update() first")
        var scaled = List[Float64]()
        for i in range(6):
            var a = action[i]
            if a > 1.0:
                a = 1.0
            if a < -1.0:
                a = -1.0
            scaled.append(a * (self.out_pos if i < 3 else self.out_ori))
        # ⚠ `bools`: only an EXACTLY non-zero rotation delta moves goal_ori
        var any_rot = scaled[3] != 0.0 or scaled[4] != 0.0 or scaled[5] != 0.0
        if any_rot:
            var dr = axisangle_to_mat(scaled[3], scaled[4], scaled[5])
            self.goal_mat = mat_mul(dr, self.ee_mat, 3, 3, 3)
        self.goal_pos = List[Float64]()
        for i in range(3):
            self.goal_pos.append(self.ee_pos[i] + scaled[i])

    def run(mut self) raises -> List[Float64]:
        """`run_controller` + `clip_torques`, from the last `update()`."""
        if len(self.goal_pos) != 3 or len(self.initial_joint) != ARM_DOF:
            raise Error("osc: run() needs reset() (and update()) first")
        var n = ARM_DOF
        # desired force / torque
        var force = List[Float64]()
        for i in range(3):
            force.append(
                (self.goal_pos[i] - self.ee_pos[i]) * self.kp
                + (-self.ee_vel[i]) * self.kd
            )
        var oerr = orientation_error(self.goal_mat, self.ee_mat)
        var torque = List[Float64]()
        for i in range(3):
            torque.append(oerr[i] * self.kp + (-self.ee_vel[3 + i]) * self.kd)
        # operational-space matrices
        var minv = mat_inverse(self.M, n)
        var jt = mat_t(self.J, 6, n)                       # 7 x 6
        var jp = List[Float64]()
        var jr = List[Float64]()
        for i in range(3 * n):
            jp.append(self.J[i])
            jr.append(self.J[3 * n + i])
        var jpt = mat_t(jp, 3, n)
        var jrt = mat_t(jr, 3, n)
        var lf_inv = mat_mul(mat_mul(self.J, minv, 6, n, n), jt, 6, n, 6)
        var lp_inv = mat_mul(mat_mul(jp, minv, 3, n, n), jpt, 3, n, 3)
        var lo_inv = mat_mul(mat_mul(jr, minv, 3, n, n), jrt, 3, n, 3)
        var lf = mat_inverse(lf_inv, 6)
        var lp = mat_inverse(lp_inv, 3)
        var lo = mat_inverse(lo_inv, 3)
        var jbar = mat_mul(mat_mul(minv, jt, n, n, 6), lf, n, 6, 6)   # 7 x 6
        var jbar_j = mat_mul(jbar, self.J, n, 6, n)                  # 7 x 7
        var nullspace = List[Float64]()
        for i in range(n):
            for j in range(n):
                nullspace.append((1.0 if i == j else 0.0) - jbar_j[i * n + j])
        # decoupled wrench
        var wrench = List[Float64]()
        var df = mat_mul(lp, force, 3, 3, 1)
        var dt_ = mat_mul(lo, torque, 3, 3, 1)
        for i in range(3):
            wrench.append(df[i])
        for i in range(3):
            wrench.append(dt_[i])
        var tau = mat_mul(jt, wrench, n, 6, 1)
        for i in range(n):
            tau[i] += self.bias[i]
        # nullspace torques: N^T M (kp (q0 - q) - kv qd)
        var pose = List[Float64]()
        for i in range(n):
            pose.append(
                self.ns_kp * (self.initial_joint[i] - self.q[i]) - self.ns_kv * self.qd[i]
            )
        var mp = mat_mul(self.M, pose, n, n, 1)
        var nt = mat_t(nullspace, n, n)
        var ns_tau = mat_mul(nt, mp, n, n, 1)
        for i in range(n):
            tau[i] += ns_tau[i]
        self.torques = tau.copy()
        var out = List[Float64]()
        for i in range(n):
            var t = tau[i]
            if t < self.torque_min[i]:
                t = self.torque_min[i]
            if t > self.torque_max[i]:
                t = self.torque_max[i]
            out.append(t)
        return out^


struct PandaGripperRamp(Copyable, Movable):
    """`PandaGripper.format_action` + `Manipulator.grip_action` (1.4.0)."""

    var current0: Float64
    var current1: Float64
    var speed: Float64
    var bias0: Float64
    var weight0: Float64
    var bias1: Float64
    var weight1: Float64

    def __init__(
        out self,
        ctrl0_min: Float64, ctrl0_max: Float64,
        ctrl1_min: Float64, ctrl1_max: Float64,
    ):
        self.current0 = 0.0
        self.current1 = 0.0
        self.speed = 0.01
        self.bias0 = 0.5 * (ctrl0_max + ctrl0_min)
        self.weight0 = 0.5 * (ctrl0_max - ctrl0_min)
        self.bias1 = 0.5 * (ctrl1_max + ctrl1_min)
        self.weight1 = 0.5 * (ctrl1_max - ctrl1_min)

    def step(mut self, action: Float64) -> List[Float64]:
        """One SIM step of the ramp; returns the two finger ctrls."""
        var sgn = 0.0
        if action > 0.0:
            sgn = 1.0
        elif action < 0.0:
            sgn = -1.0
        self.current0 = self.current0 - self.speed * sgn
        self.current1 = self.current1 + self.speed * sgn
        if self.current0 > 1.0:
            self.current0 = 1.0
        if self.current0 < -1.0:
            self.current0 = -1.0
        if self.current1 > 1.0:
            self.current1 = 1.0
        if self.current1 < -1.0:
            self.current1 = -1.0
        var out = List[Float64]()
        out.append(self.bias0 + self.weight0 * self.current0)
        out.append(self.bias1 + self.weight1 * self.current1)
        return out^
