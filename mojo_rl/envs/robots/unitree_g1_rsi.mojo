"""Reference-state initialisation for the G1 from the LAFAN1 store — G3.1.

BFM-Zero resets every lane to a random frame of a random clip
(`legged_robot_motions.py::_reset_root_states` / `_reset_dofs`): root pose
and velocities plus joint angles and velocities from the motion library at
a time drawn uniformly within the clip, the clip drawn per motion weight
(uniform here; the tracking-EMD prioritisation reweights later), and with
probability `lie_down_init_prob` 0.3 the LIE-DOWN transform — root z set
to 0.5 and the root rotated ±90° about x (`quat_mul(rot_x(±π/2), q)`,
the sign a coin per reset batch), so the policy also learns to stand up.
No initial noise (`noise_to_initial_level 0`).

Two halves, one file:

  HOST — `G1RsiTable`: every store row as a `qpos | qvel` device row
         (71 floats), with the root angular velocity rotated from the
         store's WORLD frame into the BODY frame MuJoCo's free joint
         wants (their MuJoCo backend does the same,
         `set_actor_root_state_tensor`); the store's episode index as the
         per-motion sampling table. `lie_down_row` is the transform on
         one host row, the device kernel's twin for the gate.
  DEVICE — `rsi_inject_kernel[LANES, NQ, NV]`: one thread per lane:
         draw a motion uniformly then a row uniformly inside it (two
         uniforms the driver supplies), copy the row into the lane's
         `qpos`/`qvel`, apply the lie-down transform when the lane's
         third uniform is below `lie_prob`, with `sign` shared by the batch.

The driver calls `reset_batch` first (the engine's own bookkeeping:
warm-start zero, `act` zero, step count zero), then this kernel, then
`_run_fields_fk`, `_run_fields_vel`, `_extract_obs_only` so the lane's
observation describes the injected state — the sequence `set_state`
runs on the CPU env.

Gate: `tests/robots/test_g1_rsi_table.mojo` — the host transform against
a numpy oracle on store rows, and the table's rows against the store.
"""

from std.math import sqrt
from std.gpu import global_idx

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.data.store import TrajectoryStore
from mojo_rl.physics3d.kinematics.quat_math import quat_mul

from .unitree_g1_xml import UnitreeG1Model


comptime G1_RSI_NQ: Int = UnitreeG1Model.NQ          # 36
comptime G1_RSI_NV: Int = UnitreeG1Model.NV          # 35
comptime G1_RSI_ROW: Int = G1_RSI_NQ + G1_RSI_NV     # 71
comptime G1_LIE_DOWN_Z: Float64 = 0.5
comptime G1_LIE_DOWN_PROB: Float64 = 0.3


@always_inline
def world_ang_vel_to_body[
    DTYPE: DType
](
    qw: Scalar[DTYPE], qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE],
    wx: Scalar[DTYPE], wy: Scalar[DTYPE], wz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """The reference's `quat_rotate_inverse(q, ω_world)` — `a − b + c` with
    a = v (2w² − 1), b = 2w (q_v × v), c = 2 q_v (q_v · v) — as its MuJoCo
    backend applies it before writing `qvel[3:6]`. Written out rather than
    the engine's `q⁻¹ v q`: the two agree only for a unit quaternion, and
    the store's root quaternions are the reference's slerp output, unit to
    ~4e-5 on some rows (doc §10), which moved a body-frame ω by 1.7e-4.
    Quaternion given WXYZ (MuJoCo's qpos order)."""
    var two = Scalar[DTYPE](2)
    var s = two * qw * qw - Scalar[DTYPE](1)
    var dot = qx * wx + qy * wy + qz * wz
    # q_v × v
    var cx = qy * wz - qz * wy
    var cy = qz * wx - qx * wz
    var cz = qx * wy - qy * wx
    return (
        wx * s - two * qw * cx + two * qx * dot,
        wy * s - two * qw * cy + two * qy * dot,
        wz * s - two * qw * cz + two * qz * dot,
    )


@always_inline
def lie_down_quat[
    DTYPE: DType
](
    qw: Scalar[DTYPE], qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE],
    sign: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """`quat_mul(quat_from_angle_axis(sign·(−π/2), x), q)` — the reference's
    lie-down rotation, applied on the LEFT. In and out WXYZ."""
    # half-angle = sign · (−π/4): sin = −sign · √2/2, cos = √2/2 — constants,
    # so the generic DTYPE needs no floating-point proof and no transcendental
    var rx = Scalar[DTYPE](-0.7071067811865476) * sign
    var rw = Scalar[DTYPE](0.7071067811865476)
    # quat_mul takes and returns (x, y, z, w)
    var r = quat_mul[DTYPE](rx, Scalar[DTYPE](0), Scalar[DTYPE](0), rw, qx, qy, qz, qw)
    return (r[3], r[0], r[1], r[2])


def lie_down_row(mut row: List[Float64], sign: Float64):
    """The transform on one host `qpos | qvel` row: z ← 0.5, root quaternion
    rotated. The velocities are the reference's (unchanged)."""
    row[2] = G1_LIE_DOWN_Z
    var q = lie_down_quat[DType.float64](row[3], row[4], row[5], row[6], sign)
    row[3] = q[0]
    row[4] = q[1]
    row[5] = q[2]
    row[6] = q[3]


struct G1RsiTable(Movable, Deinitable):
    """Every store row as a device `qpos | qvel` row, plus the per-motion
    offsets and lengths for the two-level draw."""
    var rows: Tensor            # n_rows * G1_RSI_ROW, device
    var ep_offset: Tensor       # n_ep, as float (device draws index it)
    var ep_len: Tensor          # n_ep
    var n_rows: Int
    var n_ep: Int

    def __init__(out self):
        self.rows = Tensor()
        self.ep_offset = Tensor()
        self.ep_len = Tensor()
        self.n_rows = 0
        self.n_ep = 0

    def __init__(out self, *, deinit move: Self):
        self.rows = move.rows^
        self.ep_offset = move.ep_offset^
        self.ep_len = move.ep_len^
        self.n_rows = move.n_rows
        self.n_ep = move.n_ep

    @staticmethod
    def from_store(mut store: TrajectoryStore) raises -> Self:
        """Host build: `qpos` verbatim, `qvel` with the root angular velocity
        rotated into the body frame. Upload is the caller's (`upload`)."""
        var t = Self()
        var n = store.n_rows()
        var qpos = store.load_column[DType.float32](String("qpos"))
        var qvel = store.load_column[DType.float32](String("qvel"))
        t.rows.ensure(n * G1_RSI_ROW)
        for r in range(n):
            var base = r * G1_RSI_ROW
            for i in range(G1_RSI_NQ):
                t.rows.data[base + i] = Scalar[DT](qpos[r * G1_RSI_NQ + i])
            for i in range(G1_RSI_NV):
                t.rows.data[base + G1_RSI_NQ + i] = Scalar[DT](qvel[r * G1_RSI_NV + i])
            var w = world_ang_vel_to_body[DType.float64](
                Float64(qpos[r * G1_RSI_NQ + 3]), Float64(qpos[r * G1_RSI_NQ + 4]),
                Float64(qpos[r * G1_RSI_NQ + 5]), Float64(qpos[r * G1_RSI_NQ + 6]),
                Float64(qvel[r * G1_RSI_NV + 3]), Float64(qvel[r * G1_RSI_NV + 4]),
                Float64(qvel[r * G1_RSI_NV + 5]),
            )
            t.rows.data[base + G1_RSI_NQ + 3] = Scalar[DT](w[0])
            t.rows.data[base + G1_RSI_NQ + 4] = Scalar[DT](w[1])
            t.rows.data[base + G1_RSI_NQ + 5] = Scalar[DT](w[2])
        t.n_rows = n
        t.n_ep = store.n_episodes()
        t.ep_offset.ensure(t.n_ep)
        t.ep_len.ensure(t.n_ep)
        for e in range(t.n_ep):
            t.ep_offset.data[e] = Scalar[DT](store.episodes.start_of(e))
            t.ep_len.data[e] = Scalar[DT](store.episodes.length_of(e))
        return t^

    def row(self, r: Int) -> List[Float64]:
        var out = List[Float64](capacity=G1_RSI_ROW)
        for i in range(G1_RSI_ROW):
            out.append(Float64(self.rows.data[r * G1_RSI_ROW + i]))
        return out^


def rsi_inject_kernel[LANES: Int, NQ: Int, NV: Int](
    rows: Pointer[Scalar[DT], MutAnyOrigin],        # n_rows * (NQ + NV)
    ep_offset: Pointer[Scalar[DT], MutAnyOrigin],   # n_ep
    ep_len: Pointer[Scalar[DT], MutAnyOrigin],      # n_ep
    n_ep: Int32,
    u: Pointer[Scalar[DT], MutAnyOrigin],           # LANES * 3 uniforms in [0, 1)
    lie_prob: Scalar[DT],
    lie_sign: Scalar[DT],
    qpos: Pointer[Scalar[DT], MutAnyOrigin],        # LANES * NQ
    qvel: Pointer[Scalar[DT], MutAnyOrigin],        # LANES * NV
    row_out: Pointer[Scalar[DT], MutAnyOrigin],     # LANES, the row each lane got (diagnostics)
):
    """One thread per lane: motion ← u0 uniform over motions, row ← u1
    uniform inside it, the lie-down transform when u2 < lie_prob."""
    var l = Int(global_idx.x)
    if l >= LANES:
        return
    comptime W = NQ + NV
    var e = Int(u[unsafe_offset=l * 3 + 0] * Scalar[DT](Int(n_ep)))
    if e >= Int(n_ep):
        e = Int(n_ep) - 1
    var length = Int(ep_len[unsafe_offset=e])
    var k = Int(u[unsafe_offset=l * 3 + 1] * Scalar[DT](length))
    if k >= length:
        k = length - 1
    var r = Int(ep_offset[unsafe_offset=e]) + k
    row_out[unsafe_offset=l] = Scalar[DT](r)
    for i in range(NQ):
        qpos[unsafe_offset=l * NQ + i] = rows[unsafe_offset=r * W + i]
    for i in range(NV):
        qvel[unsafe_offset=l * NV + i] = rows[unsafe_offset=r * W + NQ + i]
    if u[unsafe_offset=l * 3 + 2] < lie_prob:
        qpos[unsafe_offset=l * NQ + 2] = Scalar[DT](G1_LIE_DOWN_Z)
        var q = lie_down_quat[DT](
            qpos[unsafe_offset=l * NQ + 3], qpos[unsafe_offset=l * NQ + 4],
            qpos[unsafe_offset=l * NQ + 5], qpos[unsafe_offset=l * NQ + 6], lie_sign,
        )
        qpos[unsafe_offset=l * NQ + 3] = q[0]
        qpos[unsafe_offset=l * NQ + 4] = q[1]
        qpos[unsafe_offset=l * NQ + 5] = q[2]
        qpos[unsafe_offset=l * NQ + 6] = q[3]
