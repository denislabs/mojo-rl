"""The reference-state-init table and the lie-down transform against numpy — G3.1's host gate.

    pixi run mojo run -I . tests/robots/test_g1_rsi_table.mojo
    G1_LAFAN_STORE=/path/to/store.h5 pixi run mojo run -I . tests/robots/test_g1_rsi_table.mojo

`unitree_g1_rsi.mojo` turns the store into `qpos | qvel` rows with the root
angular velocity rotated into the body frame (MuJoCo's free-joint
convention, and what the reference's own MuJoCo backend does before it
writes `qvel[3:6]`), and applies BFM-Zero's lie-down transform (z ← 0.5,
root rotated ±90° about x on the left). This gate checks both on the
host against the G2 protocol module's quaternion helpers — the same
`quat_rotate_inverse` that reproduced the released actor's tracking row —
and the reference's `quat_mul`, on rows sampled across the store:

  * table rows: `qpos` verbatim, `qvel[0:3]`, `qvel[6:]` verbatim,
    `qvel[3:6]` = `quat_rotate_inverse(root_quat, ω_world)`, 1e-6 (float32 rows);
  * lie-down: the transformed row's quaternion equals
    `quat_mul(from_angle_axis(sign·(−π/2), x), q)` for both signs, 1e-12,
    and its norm is preserved;
  * the per-motion offsets and lengths equal the store's episode index.

The device kernel (`rsi_inject_kernel`) is the same arithmetic as the host
functions gated here plus a two-level uniform draw; it is exercised on the
5090 by the driver's reset diagnostics (row ids read back, lie-down
fraction, obs of an injected lane against `set_state` on the CPU env).

⚠ RUN FROM THE REPO ROOT (`tools/g1` on the Python path).
"""

from std.math import abs, sqrt
from std.os import getenv
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.data.store import TrajectoryStore
from mojo_rl.envs.robots.unitree_g1_rsi import (
    G1RsiTable, G1_RSI_NQ, G1_RSI_NV, G1_RSI_ROW, G1_LIE_DOWN_Z, lie_down_row,
)


comptime STRIDE = 2999
comptime TOL_ROW = 1e-5   # float32 rows, values up to tens of rad/s
comptime TOL_QUAT = 1e-12


def _store_path() -> String:
    var p = getenv("G1_LAFAN_STORE")
    if p.byte_length() > 0:
        return p
    return String("lafan_g1_50hz.h5")


def _py4(builtins: PythonObject, a: Float64, b: Float64, c: Float64, d: Float64) raises -> PythonObject:
    var out = builtins.list()
    _ = out.append(a)
    _ = out.append(b)
    _ = out.append(c)
    _ = out.append(d)
    return out


def _py3(builtins: PythonObject, a: Float64, b: Float64, c: Float64) raises -> PythonObject:
    var out = builtins.list()
    _ = out.append(a)
    _ = out.append(b)
    _ = out.append(c)
    return out


def test_table_rows_and_lie_down_match_numpy() raises:
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools/g1")
    var builtins = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    var proto = Python.import_module("bfm_zero_tracking_oracle")
    var pobs = Python.import_module("privileged_obs_oracle")

    var store = TrajectoryStore(_store_path())
    var n = store.n_rows()
    var qpos = store.load_column[DType.float32](String("qpos"))
    var qvel = store.load_column[DType.float32](String("qvel"))
    var table = G1RsiTable.from_store(store)
    assert_true(table.n_rows == n, "table row count")
    assert_true(table.n_ep == store.n_episodes(), "table episode count")
    for e in range(table.n_ep):
        assert_true(Int(table.ep_offset.data[e]) == store.episodes.start_of(e), "ep offset")
        assert_true(Int(table.ep_len.data[e]) == store.episodes.length_of(e), "ep length")

    var worst_row = 0.0
    var worst_lie = 0.0
    var worst_norm = 0.0
    var n_checked = 0
    var r = 0
    while r < n:
        var row = table.row(r)
        # verbatim parts
        for i in range(G1_RSI_NQ):
            var e = abs(row[i] - Float64(qpos[r * G1_RSI_NQ + i]))
            if e > worst_row:
                worst_row = e
        for i in range(G1_RSI_NV):
            if i >= 3 and i < 6:
                continue
            var e = abs(row[G1_RSI_NQ + i] - Float64(qvel[r * G1_RSI_NV + i]))
            if e > worst_row:
                worst_row = e
        # body-frame angular velocity vs the protocol module's helper
        var q_xyzw = proto.wxyz_to_xyzw(_py4(
            builtins, Float64(qpos[r * G1_RSI_NQ + 3]), Float64(qpos[r * G1_RSI_NQ + 4]),
            Float64(qpos[r * G1_RSI_NQ + 5]), Float64(qpos[r * G1_RSI_NQ + 6]),
        ))
        var wb = proto.quat_rotate_inverse_xyzw(q_xyzw, _py3(
            builtins, Float64(qvel[r * G1_RSI_NV + 3]), Float64(qvel[r * G1_RSI_NV + 4]),
            Float64(qvel[r * G1_RSI_NV + 5]),
        ))
        for c in range(3):
            var e = abs(row[G1_RSI_NQ + 3 + c] - Float64(py=wb[c]))
            if e > worst_row:
                worst_row = e
        # lie-down, both signs, vs quat_mul(from_angle_axis(sign*(-pi/2), x), q)
        for s in range(2):
            var sign = 1.0 if s == 0 else -1.0
            var ld = List[Float64](capacity=G1_RSI_ROW)
            for i in range(G1_RSI_ROW):
                ld.append(row[i])
            lie_down_row(ld, sign)
            assert_true(ld[2] == G1_LIE_DOWN_Z, "lie-down z")
            var half = sign * (-3.141592653589793 / 2.0) / 2.0
            var rot = _py4(builtins, Float64(py=np.sin(half)), 0.0, 0.0, Float64(py=np.cos(half)))  # xyzw
            var want = pobs.quat_mul_xyzw(rot, q_xyzw)  # xyzw
            var e0 = abs(ld[3] - Float64(py=want[3]))
            var e1 = abs(ld[4] - Float64(py=want[0]))
            var e2 = abs(ld[5] - Float64(py=want[1]))
            var e3 = abs(ld[6] - Float64(py=want[2]))
            for e in [e0, e1, e2, e3]:
                if e > worst_lie:
                    worst_lie = e
            var nrm_in = sqrt(row[3] * row[3] + row[4] * row[4] + row[5] * row[5] + row[6] * row[6])
            var nrm_out = sqrt(ld[3] * ld[3] + ld[4] * ld[4] + ld[5] * ld[5] + ld[6] * ld[6])
            var en = abs(nrm_in - nrm_out)
            if en > worst_norm:
                worst_norm = en
        n_checked += 1
        r += STRIDE

    print("  rows checked:", n_checked, " table vs store (body-frame omega included): worst |d|", worst_row)
    print("  lie-down quaternion vs quat_mul(rot_x(±90°), q): worst |d|", worst_lie, " norm drift", worst_norm)
    assert_true(n_checked >= 100, "too few rows")
    assert_true(worst_row < TOL_ROW, "table row differs from the store / helper by " + String(worst_row))
    assert_true(worst_lie < TOL_QUAT, "lie-down quaternion differs by " + String(worst_lie))
    assert_true(worst_norm < TOL_QUAT, "lie-down changed the quaternion norm by " + String(worst_norm))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
