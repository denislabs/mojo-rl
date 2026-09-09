"""The LAFAN1 store against our G1 forward kinematics — the G1 rung's gate.

    pixi run mojo run -I . tests/robots/test_unitree_g1_lafan_store.mojo
    G1_LAFAN_STORE=/path/to/store.h5 pixi run mojo run -I . tests/robots/test_unitree_g1_lafan_store.mojo

The store (`examples/g1/lafan_import.mojo`, default `lafan_g1_50hz.h5` in the
repo root) carries what BFM-Zero's OWN motion library produced for every
clip at 50 Hz: `qpos` in MuJoCo's layout, the 31 body poses its torch FK
computed, and the two observations its `compute_humanoid_observations_max`
built from them. This gate feeds each sampled `qpos` row into OUR engine,
runs our forward kinematics, and asks three questions:

  1. FK — do our 30 named bodies land where the reference's FK put them,
     and does the virtual `head_link` (torso + 0.35 m along the torso's z)
     land where its extended FK put it? Positions and quaternions.
  2. PRIVILEGED, the pose half — from OUR body poses, does the heading-frame
     construction the reference applies (root height, local body positions
     with the root removed, tangent-normal local rotations) reproduce
     `privileged[0:277]`? The velocity half (277:463) is the reference's
     Gaussian-filtered finite differences and is data, not FK; it is carried
     in the store and not recomputed here.
  3. STATE — `state[0:29]` is `qpos[7:] - default` and `state[58:61]` is the
     projected gravity from `qpos[3:7]`, both recomputable from the row.

⚠ WHAT THIS PROVES AND DOES NOT. The store's features are the reference's,
produced by the reference's code (`tools/g1/lafan_reference_dump.py`), so
this gate is not "our transcription against our transcription". It proves
that (a) our engine's FK on the sim-to-sim model agrees with the
reference's torch FK on its 30-body skeleton, i.e. the two models ARE the
same kinematic tree with the same offsets, and (b) the heading-frame
observation we will build live in the env (G3/G4's privileged obs) is the
reference's, frame by frame. It does not prove anything about velocities,
which the live env will produce from the simulator and the expert rows
carry from filtered differences — a difference the reference itself has.

⚠⚠ FOUR ROWS IN FIVE ARE NOT THE FK OF THEIR OWN `qpos`, BY THE REFERENCE'S
CONSTRUCTION. The clips are 30 fps and the store is 50 Hz. `get_motion_state`
blends the two neighbouring source frames LINEARLY IN JOINT SPACE for
`dof_pos` and, separately, linearly in CARTESIAN space for the body
positions and by slerp for the body rotations. `FK(blend(q))` is not
`blend(FK(q))` — second order in the motion across the 33 ms gap — so on a
blended row our FK of the row's `qpos` sits millimetres from the row's
`body_pos` (measured over the 352 890 blended rows of the 40 clips: median
0.56 mm, p99 6 mm, max 10 cm — a wrist thrown at 5 m/s in the fight clips —
and up to 0.12 in a quaternion component, 0.27 in the heading-frame obs). Every fifth row (0.1 s = 3 source frames) lands on a
source frame with blend 0, and there the two are the same computation. The
gate therefore splits the rows: the EXACT rows (WITHIN-CLIP index % 5 == 0;
clips are not multiples of five rows long, so the global row index is the
wrong thing to test — 30 of the 40 clips start at a non-zero residue) are
gated, the BLENDED rows are measured and printed as the interpolation
artefact.
That artefact is a property of the reference's data path — its own tracking
targets carry it — and it is recorded, not hidden.

⚠⚠ EVEN THE EXACT ROWS CARRY A ROTATION ARTEFACT OF THE REFERENCE'S SLERP.
`torch_utils.slerp` returns the UNNORMALISED MIDPOINT `0.5*q0 + 0.5*q1`
whenever `sin(half-angle) < 1e-3` between the two source frames — for ANY
blend weight, blend 0 included. A body that turns less than ~2e-3 rad in a
33 ms gap (the root of a lying figure, a planted foot, a still arm) gets a
quaternion up to 1e-3 rad off its own frame's, and the row's `qpos[3:7]`
is that root quaternion while `body_pos` is the frame's exact position.
Our FK of the row therefore rotates the whole tree by that root error:
0.7 m of leg times 8.5e-4 rad was the 6.3e-4 m at an ankle roll link that
this gate first flagged. Just above the threshold the same function
computes `sin(theta)/sqrt(1 - c^2)` in float32 with `c` within 1e-6 of 1,
and the ratio loses digits: the quaternion comes back with a norm off by
up to 4e-5, which the reference's unit-norm-assuming rotation formulas
(`quat_rotate_inverse`, `quat_to_tan_norm`) turn into ~1e-4 in the obs.
Verified in the reference's own Python: the store row equals
`slerp(q0, q1, 0)` bit for bit and differs from `q0` by 4e-4 at exactly the
flagged rows, with 12-26 of 31 bodies in the midpoint branch there; torch
FK and MuJoCo FK agree to 5e-7 at those frames. So the artefact is the
reference's data path, not our kinematics, and its expert rows carry it.

HOW THE GATE SEPARATES THE TWO. On exact rows the FK residual is bimodal:
the float32 floor (~1e-7) on the rows where no body sat in the slerp
branch, and up to ~1e-3 on the rows where the root did. A wrong body
offset or axis would move EVERY row, so the gate asks (a) that at most
`MAX_SLERP_FRACTION` of the exact rows sit above `FLOOR_TOL` — the FK is
right — and (b) that no row exceeds the bound the slerp mechanism allows
(`SLERP_POS_TOL`: 1e-3 rad times the ~1 m reach of a limb, `SLERP_QUAT_TOL`,
`SLERP_PRIV_TOL`: heading error plus body error plus the norm artefact).
The state check uses the reference's own unit-norm-assuming formula for
the projected gravity, so it is tight (`STATE_TOL`) whatever the norm.

⚠ RUN FROM THE REPO ROOT, like every gate that loads `unitree_g1.xml`.
"""

from std.math import abs, sqrt, atan2, sin, cos
from std.os import getenv
from std.testing import assert_true, TestSuite

from mojo_rl.data.store import TrajectoryStore
from mojo_rl.envs.robots import UnitreeG1
from mojo_rl.envs.robots.unitree_g1_xml import UnitreeG1Model, TORSO_BODY_IDX
from mojo_rl.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos
from mojo_rl.physics3d.kinematics.quat_math import quat_mul, quat_rotate


comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV
comptime N_BODIES = 31          # 30 skeleton bodies + head_link
comptime PRIV_DIM = 463
comptime PRIV_POSE_DIM = 1 + 90 + 186   # root height, local pos, local rot
comptime STATE_DIM = 64
comptime STRIDE = 1             # every row: the whole loop is ~6 us per row, 3 s for 470 k rows
comptime FLOOR_TOL = 1e-5          # a row's FK residual at the float32 floor
comptime MAX_SLERP_FRACTION = 0.10  # exact rows allowed above FLOOR_TOL (slerp branch)
comptime SLERP_POS_TOL = 1.5e-3     # 1e-3 rad root midpoint error x ~1 m reach, + floor
comptime SLERP_QUAT_TOL = 1.5e-3    # midpoint vs endpoint: <= half-angle 1e-3, + norm 4e-5
comptime SLERP_PRIV_TOL = 3e-3      # heading 1e-3 + body 1e-3 + non-unit norm ~2e-4
comptime STATE_TOL = 1e-5           # dof and gravity recomputed the reference's way
comptime HEAD_OFFSET_Z = 0.35


def _store_path() -> String:
    """`G1_LAFAN_STORE`, else the repo-root default. An env var rather than
    argv because `TestSuite` owns the argument list (`--only` / `--skip`)."""
    var p = getenv("G1_LAFAN_STORE")
    if p.byte_length() > 0:
        return p
    return String("lafan_g1_50hz.h5")


def _skeleton_to_ours(i: Int) -> Int:
    """Skeleton body i (0..29, the reference's `body_names` order) -> our
    model's body id. Our model carries the same DFS tree with ten massless
    foot frames and two rubber hands interleaved; pinned by name in
    `test_unitree_g1_vs_mujoco`."""
    if i < 7:
        return 1 + i                 # pelvis, left leg (1..7)
    if i < 13:
        return 12 + (i - 7)          # right leg (12..17)
    if i < 16:
        return 22 + (i - 13)         # waist yaw, waist roll, torso (22..24)
    if i < 23:
        return 25 + (i - 16)         # left arm (25..31)
    return 33 + (i - 23)             # right arm (33..39)


def _c3(t: Tuple[Float64, Float64, Float64], c: Int) -> Float64:
    """Component `c` of a 3-tuple — tuples take literal indices only."""
    if c == 0:
        return t[0]
    if c == 1:
        return t[1]
    return t[2]


def _quat_from_z_angle(a: Float64) -> Tuple[Float64, Float64, Float64, Float64]:
    """`quat_from_angle_axis(a, z)`, (x, y, z, w)."""
    return (0.0, 0.0, sin(0.5 * a), cos(0.5 * a))


def _heading_inv(qx: Float64, qy: Float64, qz: Float64, qw: Float64) -> Tuple[Float64, Float64, Float64, Float64]:
    """`calc_heading_quat_inv`: rotate (1,0,0) by q, heading = atan2(y, x),
    return the rotation by -heading about z."""
    var d = quat_rotate[DType.float64](qx, qy, qz, qw, 1.0, 0.0, 0.0)
    var heading = atan2(d[1], d[0])
    return _quat_from_z_angle(-heading)


def test_store_layout() raises:
    var store = TrajectoryStore(_store_path())
    var names = store.column_names()
    var want = List[String]()
    want.append(String("qpos"))
    want.append(String("qvel"))
    want.append(String("state"))
    want.append(String("privileged"))
    want.append(String("body_pos"))
    want.append(String("body_quat"))
    want.append(String("motion_id"))
    for w in want:
        var found = False
        for n in names:
            if n == w:
                found = True
        assert_true(found, "store lacks column " + w)
    assert_true(store.column(String("qpos")).row_dim() == NQ, "qpos row_dim")
    assert_true(store.column(String("qvel")).row_dim() == NV, "qvel row_dim")
    assert_true(store.column(String("state")).row_dim() == STATE_DIM, "state row_dim")
    assert_true(store.column(String("privileged")).row_dim() == PRIV_DIM, "privileged row_dim")
    assert_true(store.column(String("body_pos")).row_dim() == N_BODIES * 3, "body_pos row_dim")
    assert_true(store.column(String("body_quat")).row_dim() == N_BODIES * 4, "body_quat row_dim")
    assert_true(store.n_episodes() >= 1 and store.n_rows() > 1000, "an empty store gates nothing")
    print("  store:", store.n_rows(), "rows,", store.n_episodes(), "clips, columns", len(names))


def test_fk_and_privileged_pose_match_reference() raises:
    var store = TrajectoryStore(_store_path())
    var n = store.n_rows()
    var qpos = store.load_column[DType.float32](String("qpos"))
    var state = store.load_column[DType.float32](String("state"))
    var priv = store.load_column[DType.float32](String("privileged"))
    var bpos = store.load_column[DType.float32](String("body_pos"))
    var bquat = store.load_column[DType.float32](String("body_quat"))

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()

    # [0] = exact rows (blend 0, gated), [1] = blended rows (reported)
    var worst_pos = List[Float64](length=2, fill=0.0)
    var worst_quat = List[Float64](length=2, fill=0.0)
    var worst_priv = List[Float64](length=2, fill=0.0)
    var worst_state = List[Float64](length=2, fill=0.0)
    var n_checked = List[Int](length=2, fill=0)
    var n_above_floor = 0   # exact rows whose FK residual is above the float32 floor
    var worst_row = -1
    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)
    var our_pos = List[Float64](length=N_BODIES * 3, fill=0.0)
    var our_quat = List[Float64](length=N_BODIES * 4, fill=0.0)

    var n_ep = store.n_episodes()
    var ep = 0
    var ep_start = store.episodes.start_of(0)
    var ep_next = store.episodes.start_of(1) if n_ep > 1 else n
    var r = 0
    while r < n:
        while r >= ep_next:
            ep += 1
            ep_start = ep_next
            ep_next = store.episodes.start_of(ep + 1) if ep + 1 < n_ep else n
        for i in range(NQ):
            qp[i] = Float64(qpos[r * NQ + i])
        env.set_state(qp, qv)
        var bucket = 0 if (r - ep_start) % 5 == 0 else 1
        n_checked[bucket] += 1

        # ── 1. FK: our 30 named bodies + the virtual head ────────────────
        for s in range(30):
            var b = _skeleton_to_ours(s)
            for c in range(3):
                our_pos[s * 3 + c] = Float64(env.d.xpos.data[b * 3 + c])
            for c in range(4):
                our_quat[s * 4 + c] = Float64(env.d.xquat.data[b * 4 + c])
        # head_link: torso frame + (0, 0, 0.35) in the torso's frame, torso's quat
        var tqx = Float64(env.d.xquat.data[TORSO_BODY_IDX * 4 + 0])
        var tqy = Float64(env.d.xquat.data[TORSO_BODY_IDX * 4 + 1])
        var tqz = Float64(env.d.xquat.data[TORSO_BODY_IDX * 4 + 2])
        var tqw = Float64(env.d.xquat.data[TORSO_BODY_IDX * 4 + 3])
        var off = quat_rotate[DType.float64](tqx, tqy, tqz, tqw, 0.0, 0.0, Float64(HEAD_OFFSET_Z))
        for c in range(3):
            our_pos[30 * 3 + c] = Float64(env.d.xpos.data[TORSO_BODY_IDX * 3 + c]) + _c3(off, c)
        our_quat[30 * 4 + 0] = tqx
        our_quat[30 * 4 + 1] = tqy
        our_quat[30 * 4 + 2] = tqz
        our_quat[30 * 4 + 3] = tqw
        var row_pos = 0.0
        for k in range(N_BODIES * 3):
            var e = abs(our_pos[k] - Float64(bpos[r * N_BODIES * 3 + k]))
            if e > row_pos:
                row_pos = e
        if row_pos > worst_pos[bucket]:
            worst_pos[bucket] = row_pos
            if bucket == 0:
                worst_row = r
        if bucket == 0 and row_pos > FLOOR_TOL:
            n_above_floor += 1
        for s in range(N_BODIES):
            # q and -q are the same rotation; compare up to sign.
            var dp = 0.0
            var dm = 0.0
            for c in range(4):
                var ours = our_quat[s * 4 + c]
                var rv = Float64(bquat[r * N_BODIES * 4 + s * 4 + c])
                dp = max(dp, abs(ours - rv))
                dm = max(dm, abs(ours + rv))
            var e = min(dp, dm)
            if e > worst_quat[bucket]:
                worst_quat[bucket] = e

        # ── 2. privileged pose half, from OUR poses ───────────────────────
        var rq = _heading_inv(our_quat[0], our_quat[1], our_quat[2], our_quat[3])
        var e_h = abs(our_pos[2] - Float64(priv[r * PRIV_DIM + 0]))
        if e_h > worst_priv[bucket]:
            worst_priv[bucket] = e_h
        for s in range(1, N_BODIES):
            var lp = quat_rotate[DType.float64](
                rq[0], rq[1], rq[2], rq[3],
                our_pos[s * 3 + 0] - our_pos[0], our_pos[s * 3 + 1] - our_pos[1],
                our_pos[s * 3 + 2] - our_pos[2],
            )
            for c in range(3):
                var e = abs(_c3(lp, c) - Float64(priv[r * PRIV_DIM + 1 + (s - 1) * 3 + c]))
                if e > worst_priv[bucket]:
                    worst_priv[bucket] = e
        for s in range(N_BODIES):
            var lr = quat_mul[DType.float64](
                rq[0], rq[1], rq[2], rq[3],
                our_quat[s * 4 + 0], our_quat[s * 4 + 1], our_quat[s * 4 + 2], our_quat[s * 4 + 3],
            )
            var tan = quat_rotate[DType.float64](lr[0], lr[1], lr[2], lr[3], 1.0, 0.0, 0.0)
            var nrm = quat_rotate[DType.float64](lr[0], lr[1], lr[2], lr[3], 0.0, 0.0, 1.0)
            var base = 1 + 90 + s * 6
            for c in range(3):
                var e1 = abs(_c3(tan, c) - Float64(priv[r * PRIV_DIM + base + c]))
                var e2 = abs(_c3(nrm, c) - Float64(priv[r * PRIV_DIM + base + 3 + c]))
                if e1 > worst_priv[bucket]:
                    worst_priv[bucket] = e1
                if e2 > worst_priv[bucket]:
                    worst_priv[bucket] = e2

        # ── 3. state: dof part and projected gravity from the row ─────────
        for i in range(G1_N_DOF):
            var e = abs(qp[7 + i] - g1_default_pos(i) - Float64(state[r * STATE_DIM + i]))
            if e > worst_state[bucket]:
                worst_state[bucket] = e
        # The reference's `quat_rotate_inverse(q, (0,0,-1))`: `a - b + c` with
        # a = v (2w^2 - 1), b = 2w (q_v x v), c = 2 q_v (q_v . v). Exact for
        # unit quaternions; NOT for the non-unit ones its slerp emits, and
        # the store carries what it emitted, so the same formula is the
        # only tight comparison.
        var w = qp[3]
        var x = qp[4]
        var y = qp[5]
        var z = qp[6]
        var g = (2.0 * w * y - 2.0 * x * z, -2.0 * w * x - 2.0 * y * z, -(2.0 * w * w - 1.0) - 2.0 * z * z)
        for c in range(3):
            var e = abs(_c3(g, c) - Float64(state[r * STATE_DIM + 58 + c]))
            if e > worst_state[bucket]:
                worst_state[bucket] = e
        r += STRIDE

    var frac = Float64(n_above_floor) / Float64(max(n_checked[0], 1))
    print("  EXACT rows (blend 0, gated):", n_checked[0], "of", n, " worst row", worst_row)
    print("    rows above the float32 floor", FLOOR_TOL, ":", n_above_floor, "(fraction", frac, "- the slerp midpoint branch)")
    print("    FK    |d body pos|", worst_pos[0], "  |d body quat|", worst_quat[0])
    print("    PRIV  |d| root height / local pos / local rot:", worst_priv[0])
    print("    STATE |d| dof part / projected gravity:", worst_state[0])
    print("  BLENDED rows (the reference's joint-space vs Cartesian interpolation, reported):", n_checked[1])
    print("    FK    |d body pos|", worst_pos[1], "  |d body quat|", worst_quat[1])
    print("    PRIV  |d|", worst_priv[1], "   STATE |d|", worst_state[1])
    assert_true(n_checked[0] >= 100, "too few exact rows to gate anything")
    assert_true(frac <= MAX_SLERP_FRACTION, "too many exact rows off the float32 floor - a kinematic disagreement, not the slerp branch: fraction " + String(frac))
    assert_true(worst_pos[0] < SLERP_POS_TOL, "our FK disagrees with the reference's on body positions by " + String(worst_pos[0]))
    assert_true(worst_quat[0] < SLERP_QUAT_TOL, "our FK disagrees with the reference's on body rotations by " + String(worst_quat[0]))
    assert_true(worst_priv[0] < SLERP_PRIV_TOL, "the heading-frame privileged obs differs from the reference's by " + String(worst_priv[0]))
    assert_true(worst_state[0] < STATE_TOL, "state's dof / gravity part differs from the row by " + String(worst_state[0]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
