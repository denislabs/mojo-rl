"""The native LAFAN conversion against the reference's own dump — G1b's gate.

    pixi run mojo run -I . tests/robots/test_lafan_import_vs_oracle.mojo
    G1_LAFAN_ORACLE=/path/to/oracle.h5 LAFAN_PKL=/path/to/lafan_29dof.pkl ...

`mojo_rl/data/lafan.mojo` converts a clip from the pickle; the oracle store
(`lafan_g1_50hz_oracle.h5`, written by the reference's motion library
through `tools/g1/lafan_reference_dump.py` + `lafan_to_store.py`) holds
what the reference computed for the same clip. Two clips are converted —
fallAndGetUp1_subject4 (8410 rows, lie-still stretches that exercise the
slerp midpoint branch) and walk1_subject1 (13065 rows) — and every column
is compared row for row, each band the size of a mechanism the reference's
own code was shown to have (docs §13):

    positions      1e-5   our FK is float64 MuJoCo cast to float32, theirs a
                          float32 matrix chain: 2e-6 apart, then blended
    quaternions    1e-3   the same 1e-7 noise decides the reference's slerp
                          branch (midpoint below sin(θ/2) < 1e-3, slerp above)
                          for the ~0.5 % of body-frames sitting at the
                          threshold; the two branches differ by |t − ½|·2·1e-3
    linear vel     1e-4   finite differences of 2e-6 positions × 30 fps,
                          smoothed by the filter
    joint vel      1e-4   forward differences of the pickle's own angles
    angular vel    1.5e-2 `_compute_angular_velocity` amplifies quaternion
                          noise of 1e-7 into 1.2e-2 on ~10 % of elements
                          (measured on the reference's function; the arccos
                          of a near-unit w)
    state / priv   1.5e-2 the observation functions on the above; the worst
                          element is always an angular-velocity one

A wrong formula (blend, slerp branch, quaternion product order, gradient
edge, filter radius, a fused multiply-add the reference does not do) is
1e-2 or more on a POSITION or a whole column, and shows in one named row.
Every band also carries the FRACTION of elements above the float32 floor,
so a residual that is right in size but wrong in extent still fails. Sign
flips of `body_quat` are the sign rule's ties and are counted, not banned;
the row count of each clip must match exactly (the resampling's `ceil`).

⚠ NEEDS the oracle store and the pickle (both gitignored). Run from the
repo root. The oracle is NOT the store the importer writes — keep the
Python dump under its own name, or this gate compares ours with ours.
"""

from std.math import abs
from std.os import getenv
from std.os.path import exists
from std.testing import assert_true, TestSuite

from mojo_rl.data.store import TrajectoryStore
from mojo_rl.data.lafan import (
    load_lafan_clips, convert_clip, LAFAN_NQ, LAFAN_NV, LAFAN_STATE_DIM,
    LAFAN_PRIV_DIM, LAFAN_N_BODIES,
)
from mojo_rl.envs.robots import UnitreeG1


comptime TOL_POS = 1e-5
comptime TOL_QUAT = 1e-3
comptime TOL_LINVEL = 1e-4
comptime TOL_DOFVEL = 1e-4
comptime TOL_ANGVEL = 1.5e-2
comptime TOL_OBS = 1.5e-2
comptime FLOOR_POS = 1e-5       # the float32 floor of a position / quaternion
comptime FLOOR_VEL = 1e-4       # of a velocity
comptime FRAC_QUAT = 0.03       # body-frames at the slerp threshold (~0.5 %)
comptime FRAC_ANGVEL = 0.15     # elements the reference's arccos amplifies (~10 %)
comptime FRAC_OBS = 0.05
comptime FRAC_FLIPS = 0.001


def _oracle_path() -> String:
    var p = getenv("G1_LAFAN_ORACLE")
    if p.byte_length() > 0:
        return p
    return String("lafan_g1_50hz_oracle.h5")


def _pkl_path() -> String:
    var p = getenv("LAFAN_PKL")
    if p.byte_length() > 0:
        return p
    return String("references/BFM-Zero-main/humanoidverse/data/lafan_29dof.pkl")


struct Band(Copyable, Movable):
    """One column range compared: the worst |d|, where, and how many
    elements sit above the floor — the attribution a bare maximum cannot
    give."""
    var worst: Float64
    var row: Int
    var col: Int
    var above: Int
    var n: Int

    def __init__(out self):
        self.worst = 0.0
        self.row = 0
        self.col = 0
        self.above = 0
        self.n = 0

    def frac(self) -> Float64:
        return Float64(self.above) / Float64(max(self.n, 1))


def _profile(
    label: String, ours: List[Float32], theirs: List[Scalar[DType.float32]],
    off_theirs: Int, n_rows: Int, width: Int, c0: Int, c1: Int, floor: Float64,
) -> Band:
    """Columns `c0..c1` of every row."""
    var b = Band()
    for r in range(n_rows):
        for c in range(c0, c1):
            var i = r * width + c
            var e = abs(Float64(ours[i]) - Float64(theirs[off_theirs + i]))
            if e > floor:
                b.above += 1
            if e > b.worst:
                b.worst = e
                b.row = r
                b.col = c
            b.n += 1
    print(
        "    " + label + ": worst", b.worst, "at row", b.row, "col", b.col,
        "  >", floor, ":", b.above, "of", b.n,
    )
    return b^


def _profile_quat(
    label: String, ours: List[Float32], theirs: List[Scalar[DType.float32]],
    off_theirs: Int, n_rows: Int, width: Int, c0: Int, n_quats_per_row: Int,
    floor: Float64, mut flips: Int,
) -> Band:
    """Per quaternion (four columns from `c0`, `n_quats_per_row` of them):
    the smaller of |q − q'| and |q + q'|; a row-body that only matches with
    the sign flipped counts as a flip."""
    var b = Band()
    for r in range(n_rows):
        for k in range(n_quats_per_row):
            var dp = 0.0
            var dm = 0.0
            for c in range(4):
                var i = r * width + c0 + k * 4 + c
                var a = Float64(ours[i])
                var t = Float64(theirs[off_theirs + i])
                dp = max(dp, abs(a - t))
                dm = max(dm, abs(a + t))
            if dm < dp:
                flips += 1
            var e = min(dp, dm)
            if e > floor:
                b.above += 1
            if e > b.worst:
                b.worst = e
                b.row = r
                b.col = c0 + k * 4
            b.n += 1
    print(
        "    " + label + " (up to sign): worst", b.worst, "at row", b.row, "col", b.col,
        "  >", floor, ":", b.above, "of", b.n, " flips", flips,
    )
    return b^


def _check(mut fails: Int, b: Band, tol: Float64, frac: Float64, what: String):
    if b.worst > tol:
        print("    FAIL " + what + ": worst", b.worst, "> band", tol)
        fails += 1
    if b.frac() > frac:
        print("    FAIL " + what + ": fraction above the floor", b.frac(), ">", frac)
        fails += 1


def test_two_clips_match_the_oracle_store() raises:
    if not exists(_pkl_path()) or not exists(_oracle_path()):
        print("  SKIP: pickle or oracle store not present")
        return
    var store = TrajectoryStore(_oracle_path())
    var qpos = store.load_column[DType.float32](String("qpos"))
    var qvel = store.load_column[DType.float32](String("qvel"))
    var state = store.load_column[DType.float32](String("state"))
    var priv = store.load_column[DType.float32](String("privileged"))
    var bpos = store.load_column[DType.float32](String("body_pos"))
    var bquat = store.load_column[DType.float32](String("body_quat"))
    var bvel = store.load_column[DType.float32](String("body_vel"))
    var bang = store.load_column[DType.float32](String("body_ang_vel"))

    var clips = load_lafan_clips(_pkl_path())
    assert_true(len(clips) == store.n_episodes(), "clip count vs the store's episodes")
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()

    var ids = List[Int]()
    ids.append(0)
    ids.append(25)
    var fails = 0
    comptime NB3 = LAFAN_N_BODIES * 3
    comptime NB4 = LAFAN_N_BODIES * 4
    for t in range(len(ids)):
        var ci = ids[t]
        var rows = convert_clip(clips[ci], env, ci)
        var off = store.episodes.start_of(ci)
        var n = store.episodes.length_of(ci)
        print("  clip", ci, clips[ci].name, ": ours", rows.n_rows, "rows, oracle", n)
        assert_true(rows.n_rows == n, "row count differs for clip " + String(ci))

        # positions
        _check(fails, _profile(String("qpos root pos"), rows.qpos, qpos, off * LAFAN_NQ, n, LAFAN_NQ, 0, 3, FLOOR_POS), TOL_POS, 0.0, String("root position"))
        _check(fails, _profile(String("qpos dof"), rows.qpos, qpos, off * LAFAN_NQ, n, LAFAN_NQ, 7, LAFAN_NQ, FLOOR_POS), TOL_POS, 0.0, String("joint angles"))
        _check(fails, _profile(String("body_pos"), rows.body_pos, bpos, off * NB3, n, NB3, 0, NB3, FLOOR_POS), TOL_POS, 0.0, String("body positions"))
        _check(fails, _profile(String("state dof-default"), rows.state, state, off * LAFAN_STATE_DIM, n, LAFAN_STATE_DIM, 0, 29, FLOOR_POS), TOL_POS, 0.0, String("state joint angles"))
        # quaternions
        var flips_root = 0
        _check(fails, _profile_quat(String("qpos root quat"), rows.qpos, qpos, off * LAFAN_NQ, n, LAFAN_NQ, 3, 1, FLOOR_POS, flips_root), TOL_QUAT, FRAC_QUAT, String("root quaternion"))
        var flips = 0
        var bq = _profile_quat(String("body_quat"), rows.body_quat, bquat, off * NB4, n, NB4, 0, LAFAN_N_BODIES, FLOOR_POS, flips)
        _check(fails, bq, TOL_QUAT, FRAC_QUAT, String("body quaternions"))
        if Float64(flips + flips_root) > FRAC_FLIPS * Float64(bq.n + n):
            print("    FAIL sign flips", flips + flips_root, "of", bq.n + n)
            fails += 1
        _check(fails, _profile(String("state gravity"), rows.state, state, off * LAFAN_STATE_DIM, n, LAFAN_STATE_DIM, 58, 61, FLOOR_POS), TOL_QUAT, FRAC_QUAT, String("state gravity"))
        # velocities
        _check(fails, _profile(String("qvel root lin"), rows.qvel, qvel, off * LAFAN_NV, n, LAFAN_NV, 0, 3, FLOOR_VEL), TOL_LINVEL, 0.0, String("root linear velocity"))
        _check(fails, _profile(String("qvel dof"), rows.qvel, qvel, off * LAFAN_NV, n, LAFAN_NV, 6, LAFAN_NV, FLOOR_VEL), TOL_DOFVEL, 0.0, String("joint velocities"))
        _check(fails, _profile(String("body_vel"), rows.body_vel, bvel, off * NB3, n, NB3, 0, NB3, FLOOR_VEL), TOL_LINVEL, 0.0, String("body velocities"))
        _check(fails, _profile(String("state dof_vel"), rows.state, state, off * LAFAN_STATE_DIM, n, LAFAN_STATE_DIM, 29, 58, FLOOR_VEL), TOL_DOFVEL, 0.0, String("state joint velocities"))
        _check(fails, _profile(String("qvel root ang"), rows.qvel, qvel, off * LAFAN_NV, n, LAFAN_NV, 3, 6, FLOOR_VEL), TOL_ANGVEL, FRAC_ANGVEL, String("root angular velocity"))
        _check(fails, _profile(String("body_ang_vel"), rows.body_ang_vel, bang, off * NB3, n, NB3, 0, NB3, FLOOR_VEL), TOL_ANGVEL, FRAC_ANGVEL, String("body angular velocities"))
        _check(fails, _profile(String("state ang vel"), rows.state, state, off * LAFAN_STATE_DIM, n, LAFAN_STATE_DIM, 61, 64, FLOOR_VEL), TOL_ANGVEL, FRAC_ANGVEL, String("state angular velocity"))
        # the privileged observation as a whole
        _check(fails, _profile(String("privileged"), rows.privileged, priv, off * LAFAN_PRIV_DIM, n, LAFAN_PRIV_DIM, 0, LAFAN_PRIV_DIM, FLOOR_VEL), TOL_OBS, FRAC_OBS, String("privileged"))
    assert_true(fails == 0, "native conversion differs from the oracle store: " + String(fails) + " checks failed")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
