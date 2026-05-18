"""Phase 3b: ``GenericGPUMCTS.search_gpu_alphazero`` variant test.

Drives the AlphaZero MCTS pipeline (env.step expansion, no dynamics
network) end-to-end on a tiny stub env + stub prediction net. The stub
env is a deterministic "counter" with state size 2:

* ``state = [counter, depth]``. Action ``a ∈ [0, ACT)`` increments the
  counter by ``a + 1`` and the depth by 1. Reward = ``0`` always; done
  = ``True`` when depth ≥ ``DEPTH_LIMIT``; legal mask = all-ones.
  Obs = state itself.

This is enough to exercise the orchestrator's plumbing:

* Encoder is not used (AlphaZero pipeline) — only ``PRED`` + ``ENV``.
* Tree's ``game_states`` carry the env state through expansion.
* ``expand_backup_masked`` infers terminal from ``|step_rewards| > 0.5``
  for ``NEGATE_BACKUP=True`` (SelfPlay), so we keep rewards at 0 and
  rely on the kernel's non-terminal path with ``VALUE_SQUASH=True``.

Asserts:
* Visit counts at the root sum to ``NUM_SIMULATIONS`` per env.
* All policies sum to 1.0 (legal mask is all-ones so every action is
  reachable).
* ``actions_out`` lands in ``[0, ACT)`` for every env.
* ``state.game_states[node 0]`` matches the initial ``root_states``
  fed in — verifies ``gpu_mcts_copy_root_state_kernel`` ran.
* ``root_value_out`` is finite.

Usage:
    pixi run -e apple mojo run -I . tests/planners/tree_search/test_mcts_gpu_alphazero.mojo
"""

from std.math import abs as math_abs
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true, assert_equal
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, Sequential
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import Network, GPUNetworkState

from mojo_rl.planners.tree_search import (
    GenericGPUMCTS,
    PredictionGPU,
    EnvStepGPU,
    MuZeroPUCT,
    NoNoise,
    SelfPlay,
)


# ─── Tiny config ──────────────────────────────────────────────────────────


comptime OBS: Int = 2
comptime ACT: Int = 3
comptime BINS: Int = 1
comptime PRED_OUT: Int = ACT + BINS
comptime STATE_SIZE: Int = 2
comptime N_ENVS: Int = 2
comptime MAX_NODES: Int = 32
comptime BATCH_SIMS: Int = 4
comptime NUM_SIMS: Int = 16
comptime DEPTH_LIMIT: Int = 100  # never terminates within NUM_SIMS

# AlphaZero pred net: obs → policy_logits + scalar_value.
comptime PredModel = Sequential[Linear[OBS, PRED_OUT]]
comptime OptType = Adam[LR=1e-3]


# ─── Stub PredictionGPU (obs → policy + value) ────────────────────────────


@fieldwise_init
struct StubPredGPU(Movable, ImplicitlyDestructible, PredictionGPU):
    comptime LATENT_DIM: Int = OBS  # AlphaZero: pred input is obs
    comptime ACTION_DIM: Int = ACT
    comptime PRED_OUT_DIM: Int = PRED_OUT

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var workspace: DeviceBuffer[dtype]

    def predict_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        hidden: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut pred_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.PRED_OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        var p_t = LayoutTensor[
            dtype, Layout.row_major(PredModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var s_t = LayoutTensor[
            dtype, Layout.row_major(PredModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[PredModel, OptType].forward_gpu[B](
            ctx, hidden, pred_out, p_t, s_t, self.workspace
        )


# ─── Counter env kernel ───────────────────────────────────────────────────


def counter_env_step_kernel[
    B: Int,
    STATE: Int,
    OBSD: Int,
    A: Int,
    dtype: DType,
](
    states: LayoutTensor[dtype, Layout.row_major(B * STATE), MutAnyOrigin],
    actions: LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin],
    rewards_out: LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin],
    dones_out: LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin],
    terminated_out: LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin],
    obs_out: LayoutTensor[dtype, Layout.row_major(B * OBSD), MutAnyOrigin],
    legal_masks_out: LayoutTensor[
        dtype, Layout.row_major(B * A), MutAnyOrigin
    ],
    depth_limit: Scalar[dtype],
) where dtype.is_floating_point():
    """Deterministic counter env: state=[counter, depth], action a
    adds (a+1) to counter, depth += 1. Reward = 0, legal mask all ones,
    done = depth >= limit. Obs = state.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= B:
        return

    var off = b * STATE
    var counter = rebind[Scalar[dtype]](states[off])
    var depth = rebind[Scalar[dtype]](states[off + 1])
    var a = Int(rebind[Scalar[dtype]](actions[b]))

    counter += Scalar[dtype](a + 1)
    depth += Scalar[dtype](1.0)
    states[off] = counter
    states[off + 1] = depth

    rewards_out[b] = Scalar[dtype](0.0)
    var done = depth >= depth_limit
    dones_out[b] = Scalar[dtype](1.0) if done else Scalar[dtype](0.0)
    terminated_out[b] = dones_out[b]

    var obs_off = b * OBSD
    obs_out[obs_off] = counter
    obs_out[obs_off + 1] = depth

    var lm_off = b * A
    for j in range(A):
        legal_masks_out[lm_off + j] = Scalar[dtype](1.0)


# ─── Stub EnvStepGPU adapter ──────────────────────────────────────────────


@fieldwise_init
struct StubEnvGPU(Movable, ImplicitlyDestructible, EnvStepGPU):
    comptime STATE_SIZE: Int = 2
    comptime OBS_DIM: Int = OBS
    comptime ACTION_DIM: Int = ACT

    def step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        rewards_out: DeviceBuffer[dtype],
        dones_out: DeviceBuffer[dtype],
        terminated_out: DeviceBuffer[dtype],
        obs_out: DeviceBuffer[dtype],
        legal_masks_out: DeviceBuffer[dtype],
        rng_seed: UInt64,
    ) raises:
        var st_t = LayoutTensor[
            dtype, Layout.row_major(B * Self.STATE_SIZE), MutAnyOrigin
        ](states.unsafe_ptr())
        var ac_t = LayoutTensor[
            dtype, Layout.row_major(B), MutAnyOrigin
        ](actions.unsafe_ptr())
        var rw_t = LayoutTensor[
            dtype, Layout.row_major(B), MutAnyOrigin
        ](rewards_out.unsafe_ptr())
        var dn_t = LayoutTensor[
            dtype, Layout.row_major(B), MutAnyOrigin
        ](dones_out.unsafe_ptr())
        var tm_t = LayoutTensor[
            dtype, Layout.row_major(B), MutAnyOrigin
        ](terminated_out.unsafe_ptr())
        var ob_t = LayoutTensor[
            dtype, Layout.row_major(B * Self.OBS_DIM), MutAnyOrigin
        ](obs_out.unsafe_ptr())
        var lm_t = LayoutTensor[
            dtype, Layout.row_major(B * Self.ACTION_DIM), MutAnyOrigin
        ](legal_masks_out.unsafe_ptr())

        _ = rng_seed  # deterministic env, ignore
        comptime TPB = 256
        comptime BLK = (B + TPB - 1) // TPB
        comptime run = counter_env_step_kernel[
            B, Self.STATE_SIZE, Self.OBS_DIM, Self.ACTION_DIM, dtype
        ]
        ctx.enqueue_function[run](
            st_t, ac_t, rw_t, dn_t, tm_t, ob_t, lm_t,
            Scalar[dtype](DEPTH_LIMIT),
            grid_dim=(BLK,),
            block_dim=(TPB,),
        )


# ─── Helpers ──────────────────────────────────────────────────────────────


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-6) -> Bool:
    return math_abs(a - b) <= tol


# ─── Test ─────────────────────────────────────────────────────────────────


def test_alphazero_smoke() raises:
    var ctx = DeviceContext()

    comptime BATCHED: Int = N_ENVS * BATCH_SIMS
    var ws_per_sample = PredModel.WORKSPACE_SIZE_PER_SAMPLE
    if ws_per_sample <= 0:
        ws_per_sample = 1
    var workspace = ctx.enqueue_create_buffer[dtype](BATCHED * ws_per_sample)

    var pred_state = GPUNetworkState[PredModel, OptType](ctx)
    var pred = StubPredGPU(
        params=pred_state.params_buf.unsafe_ptr(),
        model_state=pred_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )
    var env = StubEnvGPU()

    # AlphaZero orchestrator: LATENT = OBS (no separate hidden state),
    # STATE_SIZE = 2, BINS = 1.
    var planner = GenericGPUMCTS[
        N_ENVS, ACT, OBS, BINS, MAX_NODES, NUM_SIMS, BATCH_SIMS,
        MuZeroPUCT[],
        NoNoise,
        SelfPlay,
        STATE_SIZE,
    ](ctx, gamma=0.997, v_min=-5.0, v_max=5.0)

    # Root obs = state itself (counter=0, depth=0 for env 0; counter=10, depth=0 for env 1).
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var obs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    obs_host[0] = Scalar[dtype](0.0)
    obs_host[1] = Scalar[dtype](0.0)
    obs_host[2] = Scalar[dtype](10.0)
    obs_host[3] = Scalar[dtype](0.0)
    ctx.enqueue_copy(obs_buf, obs_host)
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
    ](obs_buf.unsafe_ptr())

    # Root states identical to obs.
    var root_states = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var rs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * STATE_SIZE)
    rs_host[0] = Scalar[dtype](0.0)
    rs_host[1] = Scalar[dtype](0.0)
    rs_host[2] = Scalar[dtype](10.0)
    rs_host[3] = Scalar[dtype](0.0)
    ctx.enqueue_copy(root_states, rs_host)

    # All-ones legal mask.
    var lm_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * ACT)
    var lm_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    for i in range(N_ENVS * ACT):
        lm_host[i] = Scalar[dtype](1.0)
    ctx.enqueue_copy(lm_buf, lm_host)
    var lm_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](lm_buf.unsafe_ptr())

    planner.search_gpu_alphazero[StubPredGPU, StubEnvGPU](
        ctx, pred, env, obs_t, root_states, lm_t, rng_seed=UInt64(7),
    )
    ctx.synchronize()

    # ── Read back outputs ────────────────────────────────────────────
    var actions_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var policies_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    var rv_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var vc_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * MAX_NODES * ACT
    )
    var gs_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * MAX_NODES * STATE_SIZE
    )
    ctx.enqueue_copy(actions_host, planner.actions_out)
    ctx.enqueue_copy(policies_host, planner.policies_out)
    ctx.enqueue_copy(rv_host, planner.root_value_out)
    ctx.enqueue_copy(vc_host, planner.state.visit_count)
    ctx.enqueue_copy(gs_host, planner.state.game_states)
    ctx.synchronize()

    # ── 1. Root state copy: game_states[node 0] == initial state ─────
    var root0_counter = Float64(gs_host[0 * MAX_NODES * STATE_SIZE + 0])
    var root0_depth = Float64(gs_host[0 * MAX_NODES * STATE_SIZE + 1])
    assert_true(
        _approx(root0_counter, 0.0),
        "env 0 root counter = " + String(root0_counter) + " — expected 0",
    )
    assert_true(
        _approx(root0_depth, 0.0),
        "env 0 root depth = " + String(root0_depth) + " — expected 0",
    )
    var root1_counter = Float64(gs_host[1 * MAX_NODES * STATE_SIZE + 0])
    assert_true(
        _approx(root1_counter, 10.0),
        "env 1 root counter = " + String(root1_counter) + " — expected 10",
    )

    # ── 2. Visit counts at root sum to NUM_SIMULATIONS ───────────────
    for e in range(N_ENVS):
        var s: Int = 0
        for a in range(ACT):
            s += Int(Float64(vc_host[e * MAX_NODES * ACT + a]))
        assert_equal(
            s, NUM_SIMS, "env " + String(e) + " visit sum != NUM_SIMS"
        )

    # ── 3. Policy sums to 1, actions in [0, ACT), root_value finite ──
    for e in range(N_ENVS):
        var pol_sum = Float64(0.0)
        for a in range(ACT):
            pol_sum += Float64(policies_host[e * ACT + a])
        assert_true(
            _approx(pol_sum, 1.0, tol=1e-5),
            "env " + String(e) + " policy sum=" + String(pol_sum),
        )

        var picked = Int(Float64(actions_host[e]))
        assert_true(
            picked >= 0 and picked < ACT,
            "env " + String(e) + " action=" + String(picked) + " out of range",
        )

        var rv = Float64(rv_host[e])
        assert_true(
            rv > -1e10 and rv < 1e10,
            "env " + String(e) + " root_value=" + String(rv) + " not finite",
        )


def main() raises:
    print("=== Phase 3b: GenericGPUMCTS.search_gpu_alphazero ===")
    test_alphazero_smoke()
    print(
        "  PASS env-step pipeline drives end-to-end: root state copied,"
        " visit counts sum to NUM_SIMS, policies sum to 1, actions in"
        " [0, ACT), root_value finite"
    )
    print("OK")
