"""BFM-Zero's `max_local_self` — the 463-D privileged observation, LIVE, from
the simulator's bodies. `docs/BFM_ZERO_G1_REPRODUCTION.md` §12, G3.0.

The reference builds it in `legged_robot_motions.py:205-364`: the simulator's
30 rigid bodies plus one virtual `head_link` (torso + 0.35 m along the
torso's z), then `compute_humanoid_observations_max` (`:596-645`):

    root_height          1     pelvis z
    local_body_pos      90     (body − root) rotated into the HEADING frame,
                               bodies 1..30 (the root's own zero is dropped)
    local_body_rot     186     heading⁻¹ · body quaternion as tangent-normal:
                               rotate (1,0,0) and (0,0,1), 31 bodies
    local_body_vel      93     linear velocity in the heading frame, 31 bodies
    local_body_ang_vel  93     angular velocity in the heading frame, 31 bodies

The heading frame is the rotation by −heading about z, heading = atan2 of
the root's rotated x axis (`calc_heading_quat_inv`). Quaternions are
(x, y, z, w) on both sides — our `xquat` layout and the reference's.

⚠ TWO REFERENCE QUIRKS, REPRODUCED ON PURPOSE. The expert rows the learner
compares against (`lafan_g1_50hz.h5`, the reference's own dump) carry the
motion library's finite-difference velocities; the LIVE rows carry what
the simulator says, through the reference's code — and that code is what
BFM-Zero trained under. (1) The head's linear velocity is
`v_torso + ω_torso × (0, 0, 0.35)` with the offset NOT rotated into the
world frame (`legged_robot_motions.py:233`: `torch.cross(ω, offset_in_parent)`).
(2) The body linear velocity is the LINK-FRAME ORIGIN's in Isaac
(`_rigid_body_vel`), where our engine's `xvel` is the COM point's
(`_vel_body` propagates through `xipos`): `v_origin = v_com + ω × (xpos −
xipos)`. Both are listed in §12 as G4 ablation candidates; neither is
"fixed" here, because fidelity to what the released model saw is the
point of G3.

⚠ THE HOOKS THAT CALL THIS NEED FRESH FK AND BODY VELOCITIES. `xpos`,
`xquat`, `xipos`, `xvel`, `xangvel` are written inside the integrator and
describe the state BEFORE the last substep unless the config sets
`SYNC_FK_AFTER_STEP` (`unitree_g1_config.mojo` does, as of G3.0).

Gate: `tests/robots/test_unitree_g1_privileged_obs.mojo` — the CPU hook
against a numpy transcription of the reference's function fed OUR raw
body states, itself checked against the reference's torch function
(`tools/g1/privileged_obs_oracle.py --selfcheck`). GPU vs CPU on the 5090
through `test_unitree_g1_gpu_vs_cpu.mojo`'s obs column.
"""

from std.math import atan2, sin, cos

from mojo_rl.physics3d.kinematics.quat_math import quat_mul, quat_rotate


comptime G1_PRIV_DIM: Int = 463
comptime G1_N_SKELETON: Int = 30           # the reference's `body_names`
comptime G1_N_PRIV_BODIES: Int = 31        # + head_link
comptime G1_HEAD_OFFSET_Z: Float64 = 0.35
comptime G1_PRIV_OFF_HEIGHT: Int = 0
comptime G1_PRIV_OFF_POS: Int = 1          # 30 bodies x 3 (root dropped)
comptime G1_PRIV_OFF_ROT: Int = 91         # 31 bodies x 6
comptime G1_PRIV_OFF_VEL: Int = 277        # 31 bodies x 3
comptime G1_PRIV_OFF_ANGVEL: Int = 370     # 31 bodies x 3
comptime G1_PRIV_BODY_FEATS: Int = 15      # lpos 3 | tan 3 | norm 3 | lvel 3 | langvel 3


@always_inline
def g1_skeleton_body(i: Int) -> Int:
    """Skeleton body `i` (0..29, the reference's `body_names` order) -> our
    model's body id. Same DFS tree with ten massless foot frames and two
    rubber hands interleaved; pinned by name in `test_unitree_g1_vs_mujoco`
    and used unchanged by the G1 store gate."""
    if i < 7:
        return 1 + i                 # pelvis, left leg (1..7)
    if i < 13:
        return 12 + (i - 7)          # right leg (12..17)
    if i < 16:
        return 22 + (i - 13)         # waist yaw, waist roll, torso (22..24)
    if i < 23:
        return 25 + (i - 16)         # left arm (25..31)
    return 33 + (i - 23)             # right arm (33..39)


@always_inline
def g1_atan2f(y: Float32, x: Float32) -> Float32:
    """`atan2` in float32 with no libm call — Cephes `atan2f`, so it lowers
    on every target.

    ⚠ THE STDLIB'S `atan2` IS A LIBM SYMBOL. On NVIDIA it reaches ptxas as an
    unresolved `atan2f` (the 5090 GPU-vs-CPU gate at obs 527 died there,
    2026-09-10); `sin`/`cos`/`sqrt` have device lowerings, `atan2` does
    not. Cephes' single-precision routine: reduce to |t| ≤ tan(π/8) by the
    two identities, a degree-9 odd polynomial, the quadrant from the signs.
    Worst error against libm over the circle 3e-7 rad (gated in
    `test_unitree_g1_privileged_obs`), which is float32's own resolution
    of an angle near π. `(0, 0)` returns 0, as `torch.atan2` does.
    """
    comptime PIO2: Float32 = 1.5707963267948966
    comptime PIO4: Float32 = 0.7853981633974483
    comptime PI: Float32 = 3.141592653589793
    if x == Float32(0):
        if y == Float32(0):
            return Float32(0)
        return PIO2 if y > Float32(0) else -PIO2
    if y == Float32(0):
        return Float32(0) if x > Float32(0) else PI
    var t = y / x
    var neg = t < Float32(0)
    if neg:
        t = -t
    var base: Float32
    if t > Float32(2.414213562373095):
        base = PIO2
        t = Float32(-1) / t
    elif t > Float32(0.4142135623730950):
        base = PIO4
        t = (t - Float32(1)) / (t + Float32(1))
    else:
        base = Float32(0)
    var z = t * t
    var a = base + (
        ((Float32(8.05374449538e-2) * z - Float32(1.38776856032e-1)) * z
         + Float32(1.99777106478e-1)) * z - Float32(3.33329491539e-1)
    ) * z * t + t
    if neg:
        a = -a
    if x < Float32(0):
        a = a + (PI if y >= Float32(0) else -PI)
    return a


@always_inline
def g1_heading_inv[
    DTYPE: DType
](
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE]
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """`calc_heading_quat_inv`: rotate (1,0,0) by q, heading = atan2(y, x),
    the rotation by −heading about z, (x, y, z, w)."""
    var d = quat_rotate[DTYPE](
        qx, qy, qz, qw, Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0)
    )
    # The transcendental at a concrete float width chosen at compile time:
    # `atan2` wants a proof that DTYPE is floating point, which a generic
    # parameter cannot give, and Metal kernels cannot carry a double at all
    # (`air.sin.f64` fails IR verification) — so float64 lanes (CPU only)
    # compute in float64 through libm, and every other dtype in float32
    # through `g1_atan2f`, which has no libm call to leave unresolved on a
    # device (`sin`/`cos` lower on every target; `atan2` does not).
    comptime if DTYPE == DType.float64:
        var heading = atan2(Float64(d[1]), Float64(d[0]))
        var half = -0.5 * heading
        return (
            Scalar[DTYPE](0), Scalar[DTYPE](0),
            Scalar[DTYPE](sin(half)), Scalar[DTYPE](cos(half)),
        )
    else:
        var heading32 = g1_atan2f(Float32(d[1]), Float32(d[0]))
        var half32 = Float32(-0.5) * heading32
        return (
            Scalar[DTYPE](0), Scalar[DTYPE](0),
            Scalar[DTYPE](sin(half32)), Scalar[DTYPE](cos(half32)),
        )


@always_inline
def g1_origin_velocity[
    DTYPE: DType
](
    vx: Scalar[DTYPE], vy: Scalar[DTYPE], vz: Scalar[DTYPE],
    wx: Scalar[DTYPE], wy: Scalar[DTYPE], wz: Scalar[DTYPE],
    px: Scalar[DTYPE], py: Scalar[DTYPE], pz: Scalar[DTYPE],
    cx: Scalar[DTYPE], cy: Scalar[DTYPE], cz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Link-frame origin velocity from the COM point's: `v + ω × (p − c)`."""
    var rx = px - cx
    var ry = py - cy
    var rz = pz - cz
    return (
        vx + (wy * rz - wz * ry),
        vy + (wz * rx - wx * rz),
        vz + (wx * ry - wy * rx),
    )


@always_inline
def g1_priv_body[
    DTYPE: DType
](
    hx: Scalar[DTYPE], hy: Scalar[DTYPE], hz: Scalar[DTYPE], hw: Scalar[DTYPE],
    rootx: Scalar[DTYPE], rooty: Scalar[DTYPE], rootz: Scalar[DTYPE],
    px: Scalar[DTYPE], py: Scalar[DTYPE], pz: Scalar[DTYPE],
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    vx: Scalar[DTYPE], vy: Scalar[DTYPE], vz: Scalar[DTYPE],
    wx: Scalar[DTYPE], wy: Scalar[DTYPE], wz: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], G1_PRIV_BODY_FEATS]:
    """One body's fifteen features under the heading inverse `h`:
    `[local pos 3 | tangent 3 | normal 3 | local vel 3 | local ang vel 3]`.
    The local position of the root itself is (0, 0, 0) and is dropped by
    the caller."""
    var out = Array[Scalar[DTYPE], G1_PRIV_BODY_FEATS](fill=Scalar[DTYPE](0))
    var lp = quat_rotate[DTYPE](hx, hy, hz, hw, px - rootx, py - rooty, pz - rootz)
    out[0] = lp[0]
    out[1] = lp[1]
    out[2] = lp[2]
    var lr = quat_mul[DTYPE](hx, hy, hz, hw, qx, qy, qz, qw)
    var tan = quat_rotate[DTYPE](
        lr[0], lr[1], lr[2], lr[3], Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0)
    )
    var nrm = quat_rotate[DTYPE](
        lr[0], lr[1], lr[2], lr[3], Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1)
    )
    out[3] = tan[0]
    out[4] = tan[1]
    out[5] = tan[2]
    out[6] = nrm[0]
    out[7] = nrm[1]
    out[8] = nrm[2]
    var lv = quat_rotate[DTYPE](hx, hy, hz, hw, vx, vy, vz)
    out[9] = lv[0]
    out[10] = lv[1]
    out[11] = lv[2]
    var lw = quat_rotate[DTYPE](hx, hy, hz, hw, wx, wy, wz)
    out[12] = lw[0]
    out[13] = lw[1]
    out[14] = lw[2]
    return out^


@always_inline
def g1_priv_scatter[
    DTYPE: DType
](
    s: Int,
    feats: Array[Scalar[DTYPE], G1_PRIV_BODY_FEATS],
    mut priv: Array[Scalar[DTYPE], G1_PRIV_DIM],
):
    """Place body `s`'s features (0..30, head last) into the 463-slot block
    layout: positions for s >= 1, then rotations, velocities, angular
    velocities for every body."""
    if s >= 1:
        for c in range(3):
            priv[G1_PRIV_OFF_POS + (s - 1) * 3 + c] = feats[c]
    for c in range(6):
        priv[G1_PRIV_OFF_ROT + s * 6 + c] = feats[3 + c]
    for c in range(3):
        priv[G1_PRIV_OFF_VEL + s * 3 + c] = feats[9 + c]
    for c in range(3):
        priv[G1_PRIV_OFF_ANGVEL + s * 3 + c] = feats[12 + c]


@always_inline
def g1_head_pose_vel[
    DTYPE: DType
](
    tpx: Scalar[DTYPE], tpy: Scalar[DTYPE], tpz: Scalar[DTYPE],
    tqx: Scalar[DTYPE], tqy: Scalar[DTYPE], tqz: Scalar[DTYPE], tqw: Scalar[DTYPE],
    tvx: Scalar[DTYPE], tvy: Scalar[DTYPE], tvz: Scalar[DTYPE],
    twx: Scalar[DTYPE], twy: Scalar[DTYPE], twz: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 6]:
    """The virtual head from the torso's ORIGIN pose and velocities:
    position `torso + R_torso (0, 0, 0.35)`; linear velocity `v_torso +
    ω_torso × (0, 0, 0.35)` with the offset UNROTATED (the reference's
    `:233`). Returns `[pos 3 | vel 3]`; the head's quaternion and angular
    velocity are the torso's."""
    var off = Scalar[DTYPE](G1_HEAD_OFFSET_Z)
    var r = quat_rotate[DTYPE](tqx, tqy, tqz, tqw, Scalar[DTYPE](0), Scalar[DTYPE](0), off)
    var out = Array[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
    out[0] = tpx + r[0]
    out[1] = tpy + r[1]
    out[2] = tpz + r[2]
    # ω × (0, 0, off) = (ω_y·off, −ω_x·off, 0)
    out[3] = tvx + twy * off
    out[4] = tvy - twx * off
    out[5] = tvz
    return out^
