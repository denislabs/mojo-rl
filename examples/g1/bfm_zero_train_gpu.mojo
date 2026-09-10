"""BFM-Zero on the Unitree G1 — the privileged arm, trained the reference's
way: FB-CPR online, 1024 lanes, reference-state init, expert-tracking
rollouts, 16 updates per batched step. G3.3 of
`docs/BFM_ZERO_G1_REPRODUCTION.md` §12.

    pixi run -e nvidia mojo run -I . examples/g1/bfm_zero_train_gpu.mojo --smoke
    pixi run -e nvidia mojo run -I . examples/g1/bfm_zero_train_gpu.mojo --steps 192000000 --tag g3_priv

⚠ NVIDIA ONLY, AND NOT YET COMPILED ANYWHERE. The G1 batched env does not
compile for Metal (§12.2), so this file was written on the laptop and is
first built on the 5090. Everything it composes is gated on its own:
the env (G0), the store (G1), the released actor in the env (G2), the
527-D observation (G3.0), the reset table (G3.1 host half), the towers,
the normaliser and the agent at these dims (G3.2). What only the box can
check — the GPU obs hook, the injection kernel, the tracking-z pipeline,
the capture of 16 updates — has a diagnostic here (`--smoke`).

THE LOOP (`train.py::train_online`, `fb/agent.py::maybe_update_rollout_context`,
`legged_robot_motions.py` resets), per batched step `s` of `N_ENVS` env steps:

    every T_EPISODE steps    reset every lane: `reset_batch` (the engine's
                             own bookkeeping) then `rsi_inject_kernel`: a
                             motion uniform, a row uniform inside it, the
                             lie-down transform with probability 0.3 and
                             a sign shared by the batch; then FK, body
                             velocities, observation. No termination;
                             `truncated` is not a terminal, so `done` is 0.
    every TRACK_LEN steps    draw N_TRACK lanes (with replacement) and
                             N_TRACK expert windows of TRACK_LEN + 1 rows;
                             z_track[n, t] = project(mean_{j<8} B(row t+1+j))
                             with the CURRENT B on NORMALISED rows
                             (`_sample_tracking_z`); those lanes are PINNED
                             to z_track[n, s mod TRACK_LEN] every step
                             (`FBOnlineAgent.enable_z_pin`), the others keep
                             the agent's own rule: hold 100 steps, redraw
                             from the z-buffer (`zbuf_frac 1.0`: the
                             reference draws from the buffer only, once it
                             is non-empty).
    every step               obs → prev_obs; action (uniform for the first
                             SEED_STEPS env steps, else π_z + N(0, 0.05²));
                             `step_batch`; record (prev_obs, a, obs, z, 0).
    after SEED_STEPS         16 updates per batched step, one captured
                             CUDA graph replayed (`maybe_capture_replay`),
                             each update = `FBCPROnlineAgent._train_kernels`
                             with the running normaliser inside.

Config = the released `config.json` at Fig. 13's 60 M size: h 1024, L 3
for F / Q_D / actor, B 256, D 1024 × 3, z 256, batch 1024, γ 0.98, τ_FB
0.01, τ_Q 0.005, lr 3e-4 (B and D 1e-5), ortho 100, reg 0.05, GP 10, actor
std 0.05, relabel 80 %, mixture 0.2 goal / 0.6 expert / 0.2 sphere,
z-buffer 8192, seq 8. No aux critic (G4), no noise, no DR, no history.

Ring: CAP 2 M rows × (527 + 29 + 527 + 256 + 1) floats = 10.7 GB; expert
table 441 k × 527 = 0.93 GB; tracking rows 512 × 250 × 527 = 0.27 GB.

Checkpoints `runs/<id>/checkpoints/step_<batched step>.ckpt` (FB file + `.cpr`
+ `.norm` sidecars), scored
by the G3.4 eval, never one alone.
"""

from std.math import sqrt
from std.random import seed, random_float64
from std.sys import argv
from std.time import perf_counter_ns
from std.gpu import global_idx

from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.cuda import CUDAGraph, maybe_capture_replay
from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.call import call_forward
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.core.run import RunContext
from mojo_rl.data.store import TrajectoryStore
from mojo_rl.data.resident import IDX_DT
from mojo_rl.deep_agents.fb import FBCPROnlineAgent
from mojo_rl.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMActorTower, BFMBNet, BFMDNet,
)
from mojo_rl.deep_agents.fb.kernels import (
    gather_rows_kernel, project_sphere_kernel, ensure_t, _blocks,
)
from mojo_rl.deep_agents.fb.kernels import uniform01_kernel
from mojo_rl.envs.robots import UnitreeG1Batched
from mojo_rl.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM, UNITREE_G1_PRIV_DIM,
)
from mojo_rl.envs.robots.unitree_g1_rsi import (
    G1RsiTable, rsi_inject_kernel, G1_RSI_NQ, G1_RSI_NV, G1_LIE_DOWN_PROB,
    lie_down_selected,
)


# ── the recipe ────────────────────────────────────────────────────────────
comptime N_ENVS: Int = 1024
comptime OBS: Int = UNITREE_G1_OBS_DIM        # 527
comptime ACT: Int = UnitreeG1Model.ACTION_DIM  # 29
comptime D: Int = 256
comptime H: Int = 1024
comptime L: Int = 3
comptime HB: Int = 256
comptime HD: Int = 1024
comptime BATCH: Int = 1024
comptime CAP: Int = 2_000_000
comptime SEQ: Int = 8
comptime ZBUF: Int = 8192
comptime T_EPISODE: Int = 500
comptime Z_HOLD: Int = 100
comptime TRACK_LEN: Int = 250
comptime N_TRACK: Int = N_ENVS // 2
comptime SEED_STEPS: Int = 10_240
comptime UPDATES_PER_STEP: Int = 16
comptime EXPL_STD: Float64 = 0.05
comptime B_CHUNK: Int = 4096                  # rows per B forward while encoding tracking z

comptime EnvT = UnitreeG1Batched[N_ENVS]
comptime FNet = BFMFTower[OBS, ACT, D, H, L, D]
comptime BNet = BFMBNet[OBS, D, HB]
comptime ANet = BFMActorTower[OBS, D, H, L, ACT]
comptime DNet = BFMDNet[OBS, D, HD]
comptime QNet = BFMFTower[OBS, ACT, D, H, L, 1]
comptime Agent = FBCPROnlineAgent[
    FNet, BNet, ANet, DNet, QNet, OBS, ACT, D, BATCH, CAP, N_ENVS, SEQ, ZBUF
]
comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV


# ── kernels of this driver ────────────────────────────────────────────────
def zero_kernel[N: Int](p: Pointer[Scalar[DT], MutAnyOrigin]):
    var i = Int(global_idx.x)
    if i < N:
        p[unsafe_offset=i] = Scalar[DT](0)


def draw_index_kernel[N: Int](
    u: Pointer[Scalar[DT], MutAnyOrigin],       # N uniforms
    table: Pointer[Scalar[DT], MutAnyOrigin],   # a table of candidates (as float)
    n_table: Int32,
    picks: Pointer[Scalar[DT], MutAnyOrigin],   # N picks (as float)
):
    """`picks[i] = table[floor(u[i] · n_table)]`."""
    var i = Int(global_idx.x)
    if i >= N:
        return
    var k = Int(u[unsafe_offset=i] * Scalar[DT](Int(n_table)))
    if k >= Int(n_table):
        k = Int(n_table) - 1
    picks[unsafe_offset=i] = table[unsafe_offset=k]


def track_row_index_kernel[N: Int, T: Int](
    starts: Pointer[Scalar[DT], MutAnyOrigin],  # N window starts (as float)
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],         # N * T row ids: start + 1 + t
):
    var i = Int(global_idx.x)
    if i >= N * T:
        return
    var n = i // T
    var t = i % T
    idx[unsafe_offset=i] = Scalar[IDX_DT](Int(starts[unsafe_offset=n]) + 1 + t)


def chunk_index_kernel[ROWS: Int](
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],   # the full row-id table
    off: Int32,
    dst: Pointer[Scalar[IDX_DT], MutAnyOrigin],   # ROWS ids from `off`
):
    var i = Int(global_idx.x)
    if i < ROWS:
        dst[unsafe_offset=i] = idx[unsafe_offset=Int(off) + i]


def copy_rows_kernel[W: Int, ROWS: Int](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    dst_off_rows: Int32,
):
    var i = Int(global_idx.x)
    if i < ROWS * W:
        dst[unsafe_offset=Int(dst_off_rows) * W + i] = src[unsafe_offset=i]


def track_mean_project_kernel[Dz: Int, N: Int, T: Int, S: Int](
    b: Pointer[Scalar[DT], MutAnyOrigin],      # N * T * Dz: B(row t+1) per window step
    z: Pointer[Scalar[DT], MutAnyOrigin],      # N * T * Dz: the tracking z
    radius: Scalar[DT],
):
    """`z[n, t] = project(mean_{j < min(S, T − t)} b[n, t + j])` — the
    reference's `_sample_tracking_z` mean over the next `seq_length` rows,
    then onto the radius-sqrt(d) sphere. One thread per (n, t)."""
    var i = Int(global_idx.x)
    if i >= N * T:
        return
    var n = i // T
    var t = i % T
    var m = S
    if T - t < S:
        m = T - t
    var acc = Array[Scalar[DT], Dz](fill=Scalar[DT](0))
    for j in range(m):
        var base = ((n * T) + t + j) * Dz
        for k in range(Dz):
            acc[k] += b[unsafe_offset=base + k]
    var nrm2 = Scalar[DT](0)
    for k in range(Dz):
        acc[k] = acc[k] / Scalar[DT](m)
        nrm2 += acc[k] * acc[k]
    var scale = radius / sqrt(nrm2 + Scalar[DT](1e-12))
    var out_base = (n * T + t) * Dz
    for k in range(Dz):
        z[unsafe_offset=out_base + k] = acc[k] * scale


def pin_step_kernel[Dz: Int, N: Int, T: Int](
    lanes: Pointer[Scalar[DT], MutAnyOrigin],   # N lane ids (as float)
    z_track: Pointer[Scalar[DT], MutAnyOrigin], # N * T * Dz
    t: Int32,
    z_pin: Pointer[Scalar[DT], MutAnyOrigin],   # LANES * Dz
    mask: Pointer[Scalar[DT], MutAnyOrigin],    # LANES
):
    """`z_pin[lane_n] = z_track[n, t]`, `mask[lane_n] = 1`. A lane drawn
    twice takes whichever writer lands last — either is a valid pin."""
    var i = Int(global_idx.x)
    if i >= N * Dz:
        return
    var n = i // Dz
    var k = i % Dz
    var lane = Int(lanes[unsafe_offset=n])
    z_pin[unsafe_offset=lane * Dz + k] = z_track[unsafe_offset=(n * T + Int(t)) * Dz + k]
    if k == 0:
        mask[unsafe_offset=lane] = Scalar[DT](1)


# ── helpers ───────────────────────────────────────────────────────────────
def _flag(name: String, default: String) -> String:
    var args = argv()
    for i in range(len(args)):
        if String(args[i]) == name and i + 1 < len(args):
            return String(args[i + 1])
    return default


def _has(name: String) -> Bool:
    var args = argv()
    for i in range(len(args)):
        if String(args[i]) == name:
            return True
    return False


def _valid_starts(mut store: TrajectoryStore, window: Int) raises -> List[Int]:
    """Window starts whose rows `start .. start + window` stay inside one
    episode (`start + window` is the last NEXT row the window touches)."""
    var out = List[Int]()
    for e in range(store.n_episodes()):
        var off = store.episodes.start_of(e)
        var n = store.episodes.length_of(e)
        var last = off + n - 1 - window
        var s = off
        while s <= last:
            out.append(s)
            s += 1
    return out^


def _upload_ints(ctx: DeviceContext, xs: List[Int]) raises -> DeviceBuffer[IDX_DT]:
    var h = ctx.enqueue_create_host_buffer[IDX_DT](len(xs))
    for i in range(len(xs)):
        h[i] = Scalar[IDX_DT](xs[i])
    var d = ctx.enqueue_create_buffer[IDX_DT](len(xs))
    ctx.enqueue_copy(d, h)
    ctx.synchronize()
    return d^


def _upload_floats(ctx: DeviceContext, xs: List[Int]) raises -> Tensor:
    var t = Tensor()
    ensure_t["gpu"](t, len(xs), Optional(ctx))
    for i in range(len(xs)):
        t.data[i] = Scalar[DT](xs[i])
    t.upload(ctx)
    return t^


def main() raises:
    var smoke = _has("--smoke")
    var no_graph = _has("--no-graph")
    var total_env_steps = atol(_flag(String("--steps"), String(192_000_000)))
    if smoke:
        total_env_steps = N_ENVS * 40
    var ups = atol(_flag(String("--ups"), String(UPDATES_PER_STEP)))
    var tag = _flag(String("--tag"), String("g3_priv"))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var ckpt_every = atol(_flag(String("--ckpt-every"), String(2000)))   # batched steps
    var print_every = atol(_flag(String("--print-every"), String(100)))
    var seed_v = atol(_flag(String("--seed"), String(20260909)))
    var lie_prob = atof(_flag(String("--lie-prob"), String(G1_LIE_DOWN_PROB)))
    var track_on = not _has("--no-track")
    seed(seed_v)

    # ⚠⚠ `agent.save_state(tag + "." + String(s))` WAS THE SECOND HAND-ROLLED
    # TAG MECHANISM IN THIS TREE, beside the one `examples/fb/fb_train_gpu.mojo`
    # carried. Two copies of a rule is the shape `_a_rule_written_inline_twice_drifts`
    # names as the most frequent defect here, so both are replaced by
    # `RunContext` in one change — removing only one would have made the
    # survivor the third copy.
    #
    # ⚠ CHECKPOINTS MOVE. They used to land as `g3_priv.2000` in the working
    # directory; they now land in `runs/<id>/checkpoints/step_2000.ckpt`, with
    # the `.cpr` and `.norm` sidecars beside them. `bfm_zero_eval_tracking`
    # already takes `--ckpt <path>` and needs no change; the paths this run
    # writes are printed below so the next command is copy-pasteable.
    var run = RunContext(
        project=String("g1"),
        driver=String("examples/g1/bfm_zero_train_gpu.mojo"),
        slug=String("bfm-zero") + ("-" + tag if tag.byte_length() > 0 else ""),
        env=String("builtin:unitree_g1"),
        dataset=store_path,
        seed=seed_v,
    )
    run.set_tag(tag)
    print("run:", run.dir)

    var ctx = DeviceContext()
    print("BFM-Zero G1 privileged arm: lanes", N_ENVS, " obs", OBS, " act", ACT, " d", D, " h", H, " L", L)

    # ── the store: expert rows, window starts, the reset table ────────
    var store = TrajectoryStore(store_path)
    var n_rows = store.n_rows()
    var st = store.load_column[DType.float32](String("state"))
    var pv = store.load_column[DType.float32](String("privileged"))
    var eobs = Tensor()
    ensure_t["gpu"](eobs, n_rows * OBS, Optional(ctx))
    for r in range(n_rows):
        for i in range(UNITREE_G1_STATE_DIM):
            eobs.data[r * OBS + i] = Scalar[DT](st[r * UNITREE_G1_STATE_DIM + i])
        for i in range(UNITREE_G1_PRIV_DIM):
            eobs.data[r * OBS + UNITREE_G1_STATE_DIM + i] = Scalar[DT](pv[r * UNITREE_G1_PRIV_DIM + i])
    eobs.upload(ctx)
    var starts8 = _valid_starts(store, SEQ)
    var starts250 = _valid_starts(store, TRACK_LEN + 1)
    var starts8_dev = _upload_ints(ctx, starts8)
    var starts250_t = _upload_floats(ctx, starts250)
    var rsi = G1RsiTable.from_store(store)
    ensure_t["gpu"](rsi.rows, rsi.n_rows * (G1_RSI_NQ + G1_RSI_NV), Optional(ctx))
    ensure_t["gpu"](rsi.ep_offset, rsi.n_ep, Optional(ctx))
    ensure_t["gpu"](rsi.ep_len, rsi.n_ep, Optional(ctx))
    rsi.rows.upload(ctx)
    rsi.ep_offset.upload(ctx)
    rsi.ep_len.upload(ctx)
    print("  store:", n_rows, "rows,", store.n_episodes(), "clips;", len(starts8), "expert windows,", len(starts250), "tracking windows")

    # ── env and agent ─────────────────────────────────────────────────
    var env = EnvT(ctx)
    var agent = Agent.make(
        ctx, lr=3e-4, lr_b=1e-5, lr_d=1e-5, lr_q=3e-4, gamma=0.98, tau=0.01,
        tau_q=0.005, ortho_weight=100.0, max_grad_norm=0.0, bc_weight=0.0,
        act_l2_weight=0.0, reg_coeff=0.05, gp_coef=10.0,
        learning_starts=SEED_STEPS, action_scale=1.0, expl_std=EXPL_STD,
        z_hold=Z_HOLD, zbuf_frac=1.0, keep_frac=0.2, p_goal=0.2, p_expert=0.6,
        seed=UInt64(seed_v), normalize_obs=True,
    )
    agent.attach_expert_windows(eobs^, starts8_dev^, len(starts8))
    if track_on:
        agent.base.enable_z_pin()

    # ── rollout buffers ───────────────────────────────────────────────
    var prev_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var reward0 = ctx.enqueue_create_buffer[DT](N_ENVS)
    var done0 = ctx.enqueue_create_buffer[DT](N_ENVS)
    var ao = ctx.enqueue_create_buffer[DT](N_ENVS * 2 * ACT)
    var alp = ctx.enqueue_create_buffer[DT](N_ENVS * (ACT + 1))
    ctx.enqueue_function[zero_kernel[N_ENVS]](mptr(reward0.unsafe_ptr()), grid_dim=_blocks(N_ENVS), block_dim=TPB)
    ctx.enqueue_function[zero_kernel[N_ENVS]](mptr(done0.unsafe_ptr()), grid_dim=_blocks(N_ENVS), block_dim=TPB)
    var u_rsi = ctx.enqueue_create_buffer[DT](N_ENVS * 3)
    var row_got = ctx.enqueue_create_buffer[DT](N_ENVS)
    var h_row_got = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    # The lie-down diagnostic reads the SAME uniforms the inject kernel drew,
    # and decides with the same `lie_down_selected` — no second copy of the rule.
    var h_u_rsi = ctx.enqueue_create_host_buffer[DT](N_ENVS * 3)
    var rng_seed = UInt64(seed_v) + 99
    var rng_off = UInt64(0)

    # tracking-z pipeline buffers
    var u_track = ctx.enqueue_create_buffer[DT](N_TRACK * 2)
    var track_starts = ctx.enqueue_create_buffer[DT](N_TRACK)
    var track_lanes = ctx.enqueue_create_buffer[DT](N_TRACK)
    var lane_table = Tensor()
    ensure_t["gpu"](lane_table, N_ENVS, Optional(ctx))
    for l in range(N_ENVS):
        lane_table.data[l] = Scalar[DT](l)
    lane_table.upload(ctx)
    # the B forward runs in whole chunks, so the row-id table and the B
    # output are PADDED to N_CHUNKS * B_CHUNK: the tail rows past
    # N_TRACK * TRACK_LEN gather row 0 and are never read by the mean
    comptime N_CHUNKS = (N_TRACK * TRACK_LEN + B_CHUNK - 1) // B_CHUNK
    comptime N_PAD = N_CHUNKS * B_CHUNK
    var track_idx = ctx.enqueue_create_buffer[IDX_DT](N_PAD)
    ctx.enqueue_memset(track_idx, 0)
    var chunk_idx = ctx.enqueue_create_buffer[IDX_DT](B_CHUNK)
    var chunk_in = Tensor()
    var chunk_out = Tensor()
    ensure_t["gpu"](chunk_in, B_CHUNK * OBS, Optional(ctx))
    ensure_t["gpu"](chunk_out, B_CHUNK * D, Optional(ctx))
    var track_b = ctx.enqueue_create_buffer[DT](N_PAD * D)
    var track_z = ctx.enqueue_create_buffer[DT](N_TRACK * TRACK_LEN * D)

    def _rsi_reset(lie_sign: Float64) capturing raises:
        """`reset_batch`, then every lane injected from the store, then the
        FK / velocity / observation refresh — what `set_state` does on CPU."""
        env.reset_batch[N_ENVS](Optional(ctx), rng_seed + rng_off)
        comptime NU = N_ENVS * 3
        ctx.enqueue_function[uniform01_kernel[NU]](
            mptr(u_rsi.unsafe_ptr()), rng_seed + 1, rng_off,
            grid_dim=_blocks(NU), block_dim=TPB,
        )
        rng_off += UInt64(2 * NU)
        ctx.enqueue_function[rsi_inject_kernel[N_ENVS, NQ, NV]](
            mptr(rsi.rows.dev.value().unsafe_ptr()),
            mptr(rsi.ep_offset.dev.value().unsafe_ptr()),
            mptr(rsi.ep_len.dev.value().unsafe_ptr()),
            Int32(rsi.n_ep),
            mptr(u_rsi.unsafe_ptr()),
            Scalar[DT](lie_prob), Scalar[DT](lie_sign),
            mptr(env.d.qpos.dev.value().unsafe_ptr()),
            mptr(env.d.qvel.dev.value().unsafe_ptr()),
            mptr(row_got.unsafe_ptr()),
            grid_dim=_blocks(N_ENVS), block_dim=TPB,
        )
        env._run_fields_fk(ctx)
        env._run_fields_vel(ctx)
        env._extract_obs_only(ctx)

    def _draw_tracking() capturing raises:
        """N_TRACK lanes and windows; B on the windows' next rows with the
        current normaliser and B; the mean-of-8 projected tracking z."""
        comptime NU2 = N_TRACK * 2
        ctx.enqueue_function[uniform01_kernel[NU2]](
            mptr(u_track.unsafe_ptr()), rng_seed + 2, rng_off,
            grid_dim=_blocks(NU2), block_dim=TPB,
        )
        rng_off += UInt64(2 * NU2)
        ctx.enqueue_function[draw_index_kernel[N_TRACK]](
            mptr(u_track.unsafe_ptr()), mptr(starts250_t.dev.value().unsafe_ptr()),
            Int32(len(starts250)), mptr(track_starts.unsafe_ptr()),
            grid_dim=_blocks(N_TRACK), block_dim=TPB,
        )
        ctx.enqueue_function[draw_index_kernel[N_TRACK]](
            mptr(u_track.unsafe_ptr()).unsafe_offset(N_TRACK),
            mptr(lane_table.dev.value().unsafe_ptr()),
            Int32(N_ENVS), mptr(track_lanes.unsafe_ptr()),
            grid_dim=_blocks(N_TRACK), block_dim=TPB,
        )
        ctx.enqueue_function[track_row_index_kernel[N_TRACK, TRACK_LEN]](
            mptr(track_starts.unsafe_ptr()), mptr(track_idx.unsafe_ptr()),
            grid_dim=_blocks(N_TRACK * TRACK_LEN), block_dim=TPB,
        )
        for c in range(N_CHUNKS):
            ctx.enqueue_function[chunk_index_kernel[B_CHUNK]](
                mptr(track_idx.unsafe_ptr()), Int32(c * B_CHUNK), mptr(chunk_idx.unsafe_ptr()),
                grid_dim=_blocks(B_CHUNK), block_dim=TPB,
            )
            ctx.enqueue_function[gather_rows_kernel[OBS, B_CHUNK]](
                mptr(agent.exp_obs.dev.value().unsafe_ptr()), mptr(chunk_idx.unsafe_ptr()),
                mptr(chunk_in.dev.value().unsafe_ptr()),
                grid_dim=_blocks(B_CHUNK * OBS), block_dim=TPB,
            )
            agent.base.obs_ema.apply[B_CHUNK](chunk_in)
            call_forward["gpu", B_CHUNK](
                agent.base.t.bnet.online, TensorRefs[1, MutAnyOrigin](chunk_in),
                chunk_out, Optional(ctx),
            )
            ctx.enqueue_function[copy_rows_kernel[D, B_CHUNK]](
                mptr(chunk_out.dev.value().unsafe_ptr()), mptr(track_b.unsafe_ptr()),
                Int32(c * B_CHUNK),
                grid_dim=_blocks(B_CHUNK * D), block_dim=TPB,
            )
        ctx.enqueue_function[track_mean_project_kernel[D, N_TRACK, TRACK_LEN, SEQ]](
            mptr(track_b.unsafe_ptr()), mptr(track_z.unsafe_ptr()),
            Scalar[DT](sqrt(Float64(D))),
            grid_dim=_blocks(N_TRACK * TRACK_LEN), block_dim=TPB,
        )
        ctx.enqueue_function[zero_kernel[N_ENVS]](
            mptr(agent.base.z_pin_mask.dev.value().unsafe_ptr()),
            grid_dim=_blocks(N_ENVS), block_dim=TPB,
        )

    def _pin_step(t: Int) capturing raises:
        ctx.enqueue_function[pin_step_kernel[D, N_TRACK, TRACK_LEN]](
            mptr(track_lanes.unsafe_ptr()), mptr(track_z.unsafe_ptr()), Int32(t),
            mptr(agent.base.z_pin.dev.value().unsafe_ptr()),
            mptr(agent.base.z_pin_mask.dev.value().unsafe_ptr()),
            grid_dim=_blocks(N_TRACK * D), block_dim=TPB,
        )

    def _captured_updates() capturing raises -> None:
        for _ in range(ups):
            agent.train_device_kernels()

    var train_graph: Optional[CUDAGraph] = None
    var n_batched = total_env_steps // N_ENVS
    var t0 = perf_counter_ns()
    var lie_total = 0   # lanes given the lie-down transform, over diagnostic resets
    var lie_resets = 0  # diagnostic resets counted, the denominator for the above
    print("  batched steps", n_batched, " seed steps", SEED_STEPS, " updates/step", ups, " tracking", track_on)
    for s in range(n_batched):
        var env_steps = s * N_ENVS
        if s % T_EPISODE == 0:
            var sign = 1.0 if random_float64() < 0.5 else -1.0
            _rsi_reset(sign)
            if smoke or s % (T_EPISODE * 20) == 0:
                # diagnostics: which rows the lanes got, and how many lie down
                ctx.enqueue_copy(h_row_got, row_got)
                ctx.enqueue_copy(h_u_rsi, u_rsi)
                ctx.synchronize()
                var lo = 1e30
                var hi = -1e30
                var lie_n = 0
                for l in range(N_ENVS):
                    var v = Float64(h_row_got[l])
                    if v < lo:
                        lo = v
                    if v > hi:
                        hi = v
                    if lie_down_selected(h_u_rsi[l * 3 + 2], Scalar[DT](lie_prob)):
                        lie_n += 1
                lie_total += lie_n
                lie_resets += 1
                print(
                    "  [reset @", s, "] rows in [", lo, ",", hi, "] of", n_rows,
                    " lie-down", Float64(lie_n) / Float64(N_ENVS),
                    "(want", lie_prob, ", cumulative",
                    Float64(lie_total) / Float64(lie_resets * N_ENVS), ")",
                )
        if track_on and s % TRACK_LEN == 0:
            _draw_tracking()
        if track_on:
            _pin_step(s % TRACK_LEN)

        ctx.enqueue_copy(prev_obs, env._obs)
        agent.select_action_batched[N_ENVS](
            LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin](env._obs),
            LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin](env._action),
            LayoutTensor[DT, Layout.row_major(N_ENVS, 2 * ACT), MutAnyOrigin](ao),
            LayoutTensor[DT, Layout.row_major(N_ENVS, ACT + 1), MutAnyOrigin](alp),
            env_steps,
        )
        env.step_batch[N_ENVS](Optional(ctx), UInt64(seed_v) + UInt64(s))
        agent.record_batch_gpu[N_ENVS](ctx, prev_obs, env._action, reward0, env._obs, done0)

        if env_steps >= SEED_STEPS:
            if no_graph:
                # `--no-graph`: the 16 updates as plain launches, so
                # `MODULAR_DEBUG=device-sync-mode` can name a faulting kernel
                # (a graph replay reports its fault only at the next check)
                _captured_updates()
            else:
                maybe_capture_replay[_captured_updates](train_graph, ctx)
            for _ in range(ups):
                agent.note_train_update()

        if s % print_every == 0 or (smoke and s % 10 == 0):
            ctx.synchronize()
            var el = Float64(perf_counter_ns() - t0) * 1e-9
            var measure = 0.0
            var ortho = 0.0
            var actor = 0.0
            var f_norm = 0.0
            var b_norm = 0.0
            if env_steps >= SEED_STEPS:
                agent.base.peek_losses(measure, ortho, actor, f_norm, b_norm)
            print(
                "  step", s, " env", env_steps, " ", Float64(env_steps) / (el + 1e-9), "env st/s",
                " ring", agent.base.size, " updates", agent.total_train_steps(),
                " measure", measure, " ortho", ortho, " actor", actor, " |F|", f_norm, " |B|", b_norm,
            )
        if ckpt_every > 0 and s > 0 and s % ckpt_every == 0:
            var p = run.checkpoint_path(String("step_") + String(s))
            agent.save_state(p)
            print("  checkpoint", p)
    ctx.synchronize()
    # ⚠ STOP THE CLOCK BEFORE THE CHECKPOINT. `el` used to be taken AFTER
    # `save_state`, so the headline rate measured a file write: on the 40-step
    # smoke that one write was ~17 s of a 49 s run and dragged 1030 env st/s
    # down to 830 (docs/BFM_ZERO_G1_REPRODUCTION.md §12.9).
    var el = Float64(perf_counter_ns() - t0) * 1e-9
    var final_ckpt = run.checkpoint_path(String("step_") + String(n_batched))
    agent.save_state(final_ckpt)
    print("done:", n_batched, "batched steps,", agent.total_train_steps(), "updates in", el, "s;", Float64(n_batched * N_ENVS) / el, "env st/s")
    print("final checkpoint:", final_ckpt)
    print("run record      :", run.kv_path())
    run.close()
