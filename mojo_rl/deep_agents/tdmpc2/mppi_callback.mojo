"""TDMPC2 → RolloutCallbackGPU adapter for the new MPPI planner.

Closes over the world-model networks + bin tensor + per-Q target
parameter pointers, and implements the three
``RolloutCallbackGPU`` methods by dispatching to the existing
tdmpc2 kernel set.

Ownership:
  - The agent (``TDMPC2GPUState``) owns the network parameter
    DeviceBuffers and the bin tensor; this struct holds raw
    pointers / shapes into them, NOT copies.
  - The struct *does* own per-call scratch (za, rew/q logits,
    pi_out, terminal action) + 4 network workspace buffers, sized
    to ``MAX_BATCH`` at construction. These were previously
    allocated by ``BatchedMPPIGPUBuffers``; moving them onto the
    callback was a Phase-2 cleanup goal — the agent no longer
    knows what MPPI's internal scratch shape is.

Per-MPPI-iter Q-pair sampling: ``terminal_value_gpu`` accepts a
``seed`` from the planner, draws two distinct Q indices on host
via Philox (same recipe as the original
``plan_gpu_batched``), then dispatches the two
``QModel.forward_gpu_no_cache`` calls. Decoded into ``v_out`` via
``tdmpc2_decode_scaled_kernel`` (scale=0.5) +
``tdmpc2_decode_add_scaled_kernel`` (scale=0.5).
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import tanh
from std.memory import UnsafePointer
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.training import Network
from mojo_rl.planners.trajectory import (
    RolloutCallbackCPU,
    RolloutCallbackGPU,
)

from .kernels import (
    tdmpc2_build_za_kernel,
    tdmpc2_apply_tanh_build_za_deterministic_kernel,
    tdmpc2_decode_scaled_kernel,
    tdmpc2_decode_add_scaled_kernel,
)
from .world_model import WorldModel, decode_value_batch_scalar


# =============================================================================
# Small helper kernel
# =============================================================================


@always_inline
def tdmpc2_extract_action_mean_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    ACT_DIM: Int,
    POL_OUT: Int,
](
    pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, POL_OUT), MutAnyOrigin
    ],
    action_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACT_DIM), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Copy the first ``ACT_DIM`` columns of ``pi_out`` (which holds
    ``[mean, log_std]`` concatenated, POL_OUT = 2 * ACT_DIM) into
    ``action_out``. Used by ``policy_action_gpu`` to expose just the
    policy mean to the planner — without exposing the log_std half,
    which MPPI doesn't use.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return
    for j in range(ACT_DIM):
        action_out[i, j] = pi_out[i, j]


# =============================================================================
# TDMPC2RolloutCallback
# =============================================================================


struct TDMPC2RolloutCallback[
    DynModel: Model,
    DynOpt: Optimizer,
    RewModel: Model,
    RewOpt: Optimizer,
    PolModel: Model,
    PolOpt: Optimizer,
    QModel: Model,
    QOpt: Optimizer,
    LATENT_DIM_PARAM: Int,
    ACTION_DIM_PARAM: Int,
    NUM_BINS: Int,
    NUM_Q: Int,
    MAX_BATCH: Int,
](Movable, ImplicitlyDestructible, RolloutCallbackGPU):
    """Implements ``RolloutCallbackGPU`` against TDMPC2's
    world-model networks.

    Comptime params:
      * ``DynModel`` / ``DynOpt``: dynamics network type.
        Input dim ``LATENT + ACTION``, output dim ``LATENT``.
      * ``RewModel`` / ``RewOpt``: reward network (categorical).
        Input ``LATENT + ACTION``, output ``NUM_BINS``.
      * ``PolModel`` / ``PolOpt``: policy actor.
        Input ``LATENT``, output ``2 * ACTION`` (mean, log_std).
      * ``QModel`` / ``QOpt``: per-Q-target categorical Q.
        Input ``LATENT + ACTION``, output ``NUM_BINS``.
      * ``LATENT_DIM_PARAM`` / ``ACTION_DIM_PARAM``: latent / action
        dims (renamed from ``LATENT_DIM`` / ``ACTION_DIM`` because
        the trait already requires the unprefixed names as comptime
        constants — see body).
      * ``NUM_BINS``, ``NUM_Q``: categorical reward/Q discretization
        + target ensemble size.
      * ``MAX_BATCH``: largest batch size the callback's internal
        scratch must handle. For MPPI production this is
        ``N_ENVS * (NUM_SAMPLES + NUM_PI_TRAJS)``.
    """

    # Trait conformance — reify the trait comptime constants from
    # the struct's renamed template params.
    comptime LATENT_DIM: Int = Self.LATENT_DIM_PARAM
    comptime ACTION_DIM: Int = Self.ACTION_DIM_PARAM

    comptime ZA_DIM: Int = Self.LATENT_DIM + Self.ACTION_DIM
    comptime POL_OUT: Int = Self.PolModel.OUT_DIM
    comptime Q_PS: Int = Self.QModel.PARAM_SIZE

    # ── Network parameter pointers (agent owns the buffers) ──────
    var dyn_params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var rew_params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var pol_params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var qt_param_ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], Self.NUM_Q
    ]
    """Per-Q-target parameter pointers — chosen pair per
    ``terminal_value_gpu`` call."""
    var bins_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    # ── Internal scratch ─────────────────────────────────────────
    var za_buf: DeviceBuffer[dtype]
    """(MAX_BATCH, ZA_DIM) — concat of (z, a) consumed by dyn/rew/Q."""
    var rew_logits_buf: DeviceBuffer[dtype]
    """(MAX_BATCH, NUM_BINS) — reward-network categorical logits."""
    var q_logits_buf: DeviceBuffer[dtype]
    """(MAX_BATCH, NUM_BINS) — per-Q categorical logits (reused for
    both qa and qb in terminal_value_gpu)."""
    var pi_out_buf: DeviceBuffer[dtype]
    """(MAX_BATCH, POL_OUT) — full policy output (mean+log_std)."""
    var act_step_internal: DeviceBuffer[dtype]
    """(MAX_BATCH, ACTION_DIM) — terminal value's tanh-squashed
    policy action (separate from the planner's ``act_step_buf``)."""

    # ── Network workspaces ────────────────────────────────────────
    var dyn_ws_buf: DeviceBuffer[dtype]
    var rew_ws_buf: DeviceBuffer[dtype]
    var pol_ws_buf: DeviceBuffer[dtype]
    var q_ws_buf: DeviceBuffer[dtype]

    def __init__(
        out self,
        ctx: DeviceContext,
        dyn_params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        rew_params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        pol_params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        qt_param_ptrs: InlineArray[
            UnsafePointer[Scalar[dtype], MutAnyOrigin], Self.NUM_Q
        ],
        bins_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        # Self.NUM_Q reified above for the trait-required InlineArray
        # size match.
        self.dyn_params_ptr = dyn_params_ptr
        self.rew_params_ptr = rew_params_ptr
        self.pol_params_ptr = pol_params_ptr
        self.qt_param_ptrs = qt_param_ptrs
        self.bins_ptr = bins_ptr

        var bt_za = Self.MAX_BATCH * Self.ZA_DIM
        var bt_act = Self.MAX_BATCH * Self.ACTION_DIM
        var bt_bins = Self.MAX_BATCH * Self.NUM_BINS
        var bt_pol = Self.MAX_BATCH * Self.POL_OUT

        self.za_buf = ctx.enqueue_create_buffer[dtype](bt_za)
        self.rew_logits_buf = ctx.enqueue_create_buffer[dtype](bt_bins)
        self.q_logits_buf = ctx.enqueue_create_buffer[dtype](bt_bins)
        self.pi_out_buf = ctx.enqueue_create_buffer[dtype](bt_pol)
        self.act_step_internal = ctx.enqueue_create_buffer[dtype](bt_act)

        comptime DYN_W = Network[
            Self.DynModel, Self.DynOpt
        ].WORKSPACE_SIZE_PER_SAMPLE
        comptime REW_W = Network[
            Self.RewModel, Self.RewOpt
        ].WORKSPACE_SIZE_PER_SAMPLE
        comptime POL_W = Network[
            Self.PolModel, Self.PolOpt
        ].WORKSPACE_SIZE_PER_SAMPLE
        comptime Q_W = Network[
            Self.QModel, Self.QOpt
        ].WORKSPACE_SIZE_PER_SAMPLE

        self.dyn_ws_buf = ctx.enqueue_create_buffer[dtype](
            Self.MAX_BATCH * DYN_W
        )
        self.rew_ws_buf = ctx.enqueue_create_buffer[dtype](
            Self.MAX_BATCH * REW_W
        )
        self.pol_ws_buf = ctx.enqueue_create_buffer[dtype](
            Self.MAX_BATCH * POL_W
        )
        self.q_ws_buf = ctx.enqueue_create_buffer[dtype](
            Self.MAX_BATCH * Q_W
        )

    # ────────────────────────────────────────────────────────────
    # RolloutCallbackGPU trait methods
    # ────────────────────────────────────────────────────────────

    def policy_action_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        action_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.ACTION_DIM), MutAnyOrigin
        ],
    ) raises:
        """Run PolModel forward at z → write the mean (first
        ACT_DIM cols of policy output) into ``action_out``.
        """
        var pol_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.PolModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.pol_params_ptr)
        var pol_state = LayoutTensor[
            dtype,
            Layout.row_major(Self.PolModel.STATE_SIZE),
            MutAnyOrigin,
        ](
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                z.ptr
            )
        )
        var pi_out_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.POL_OUT), MutAnyOrigin
        ](self.pi_out_buf.unsafe_ptr())
        var z_pol_in = LayoutTensor[
            dtype, Layout.row_major(B, Self.PolModel.IN_DIM), MutAnyOrigin
        ](z.ptr)

        Self.PolModel.forward_gpu_no_cache[B](
            ctx,
            pi_out_view,
            z_pol_in,
            pol_params,
            pol_state,
            self.pol_ws_buf,
        )

        comptime extract_kernel = tdmpc2_extract_action_mean_kernel[
            dtype, B, Self.ACTION_DIM, Self.POL_OUT
        ]
        comptime BLOCKS = (B + TPB - 1) // TPB
        ctx.enqueue_function[extract_kernel](
            pi_out_view,
            action_out,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    def rollout_step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        a: LayoutTensor[
            dtype, Layout.row_major(B, Self.ACTION_DIM), MutAnyOrigin
        ],
        z_next_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        r_out: LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin],
    ) raises:
        """One world-model step: build_za → rew_forward → dyn_forward
        → decode reward.

        Order matters here: ``rew_forward`` reads ``za_buf`` BEFORE
        ``dyn_forward`` overwrites ``z_next``. Both share the same
        device queue so subsequent enqueues see each previous one's
        writes.
        """
        # 1. Build za = [z, a]
        var za_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.ZA_DIM), MutAnyOrigin
        ](self.za_buf.unsafe_ptr())
        comptime build_za_kernel = tdmpc2_build_za_kernel[
            dtype, B, Self.LATENT_DIM, Self.ACTION_DIM
        ]
        comptime BLOCKS = (B + TPB - 1) // TPB
        ctx.enqueue_function[build_za_kernel](
            z,
            a,
            za_view,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

        # 2. Reward forward
        var rew_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.RewModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.rew_params_ptr)
        var rew_state = LayoutTensor[
            dtype,
            Layout.row_major(Self.RewModel.STATE_SIZE),
            MutAnyOrigin,
        ](
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                z.ptr
            )
        )
        var rew_in = LayoutTensor[
            dtype, Layout.row_major(B, Self.RewModel.IN_DIM), MutAnyOrigin
        ](self.za_buf.unsafe_ptr())
        var rew_out_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.RewModel.OUT_DIM), MutAnyOrigin
        ](self.rew_logits_buf.unsafe_ptr())
        Self.RewModel.forward_gpu_no_cache[B](
            ctx,
            rew_out_view,
            rew_in,
            rew_params,
            rew_state,
            self.rew_ws_buf,
        )

        # 3. Dynamics forward → z_next
        var dyn_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.DynModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.dyn_params_ptr)
        var dyn_state = LayoutTensor[
            dtype,
            Layout.row_major(Self.DynModel.STATE_SIZE),
            MutAnyOrigin,
        ](
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                z.ptr
            )
        )
        var dyn_in = LayoutTensor[
            dtype, Layout.row_major(B, Self.DynModel.IN_DIM), MutAnyOrigin
        ](self.za_buf.unsafe_ptr())
        var dyn_out = LayoutTensor[
            dtype, Layout.row_major(B, Self.DynModel.OUT_DIM), MutAnyOrigin
        ](z_next_out.ptr)
        Self.DynModel.forward_gpu_no_cache[B](
            ctx,
            dyn_out,
            dyn_in,
            dyn_params,
            dyn_state,
            self.dyn_ws_buf,
        )

        # 4. Decode reward logits → scalar reward
        var rew_logits = LayoutTensor[
            dtype, Layout.row_major(B, Self.NUM_BINS), MutAnyOrigin
        ](self.rew_logits_buf.unsafe_ptr())
        var bins = LayoutTensor[
            dtype, Layout.row_major(Self.NUM_BINS), MutAnyOrigin
        ](self.bins_ptr)
        comptime decode_kernel = tdmpc2_decode_scaled_kernel[
            dtype, B, Self.NUM_BINS
        ]
        ctx.enqueue_function[decode_kernel](
            rew_logits,
            bins,
            r_out,
            Scalar[dtype](1.0),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    def terminal_value_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        v_out: LayoutTensor[dtype, Layout.row_major(B), MutAnyOrigin],
        seed: UInt32,
    ) raises:
        """Bootstrap V at end-of-horizon: π(z) → 2 random Q-targets
        → 0.5 * (decode_qa + decode_qb) into v_out.

        Picks the Q-pair on host (Philox over ``seed``), then
        dispatches the two QModel forwards + decode kernels on
        device.
        """
        # ── Q-pair host sample ───────────────────────────────────
        var q_pair_rng = PhiloxRandom(
            seed=UInt64(seed) + UInt64(0xA1B2C3D4),
            offset=0,
        )
        var q_pair_uniform = q_pair_rng.step_uniform()
        var qa = (
            Int(Float64(q_pair_uniform[0]) * Float64(Self.NUM_Q))
            % Self.NUM_Q
        )
        var qb = (
            qa
            + 1
            + Int(Float64(q_pair_uniform[1]) * Float64(Self.NUM_Q - 1))
            % (Self.NUM_Q - 1)
        ) % Self.NUM_Q

        # ── Policy forward ───────────────────────────────────────
        var pol_params = LayoutTensor[
            dtype,
            Layout.row_major(Self.PolModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.pol_params_ptr)
        var pol_state = LayoutTensor[
            dtype,
            Layout.row_major(Self.PolModel.STATE_SIZE),
            MutAnyOrigin,
        ](
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                z.ptr
            )
        )
        var pi_out = LayoutTensor[
            dtype, Layout.row_major(B, Self.POL_OUT), MutAnyOrigin
        ](self.pi_out_buf.unsafe_ptr())
        var z_pol_in = LayoutTensor[
            dtype, Layout.row_major(B, Self.PolModel.IN_DIM), MutAnyOrigin
        ](z.ptr)
        Self.PolModel.forward_gpu_no_cache[B](
            ctx,
            pi_out,
            z_pol_in,
            pol_params,
            pol_state,
            self.pol_ws_buf,
        )

        # ── Apply tanh + build za (terminal action) ──────────────
        var act_step_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.ACTION_DIM), MutAnyOrigin
        ](self.act_step_internal.unsafe_ptr())
        var za_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.ZA_DIM), MutAnyOrigin
        ](self.za_buf.unsafe_ptr())
        comptime tanh_build_za = (
            tdmpc2_apply_tanh_build_za_deterministic_kernel[
                dtype,
                B,
                Self.ACTION_DIM,
                Self.LATENT_DIM,
                Self.POL_OUT,
            ]
        )
        comptime BLOCKS = (B + TPB - 1) // TPB
        ctx.enqueue_function[tanh_build_za](
            pi_out,
            act_step_view,
            z,
            za_view,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

        # ── Q[qa] forward → decode_scaled (overwrite v_out) ─────
        var qta_params = LayoutTensor[
            dtype, Layout.row_major(Self.Q_PS), MutAnyOrigin
        ](self.qt_param_ptrs[qa])
        var q_state = LayoutTensor[
            dtype,
            Layout.row_major(Self.QModel.STATE_SIZE),
            MutAnyOrigin,
        ](
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                z.ptr
            )
        )
        var q_in = LayoutTensor[
            dtype, Layout.row_major(B, Self.QModel.IN_DIM), MutAnyOrigin
        ](self.za_buf.unsafe_ptr())
        var q_out = LayoutTensor[
            dtype, Layout.row_major(B, Self.QModel.OUT_DIM), MutAnyOrigin
        ](self.q_logits_buf.unsafe_ptr())
        Self.QModel.forward_gpu_no_cache[B](
            ctx,
            q_out,
            q_in,
            qta_params,
            q_state,
            self.q_ws_buf,
        )

        var bins = LayoutTensor[
            dtype, Layout.row_major(Self.NUM_BINS), MutAnyOrigin
        ](self.bins_ptr)
        var q_logits = LayoutTensor[
            dtype, Layout.row_major(B, Self.NUM_BINS), MutAnyOrigin
        ](self.q_logits_buf.unsafe_ptr())
        comptime decode_scaled = tdmpc2_decode_scaled_kernel[
            dtype, B, Self.NUM_BINS
        ]
        var half = Scalar[dtype](0.5)
        ctx.enqueue_function[decode_scaled](
            q_logits,
            bins,
            v_out,
            half,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

        # ── Q[qb] forward → decode_add_scaled (accumulate to v_out) ─
        var qtb_params = LayoutTensor[
            dtype, Layout.row_major(Self.Q_PS), MutAnyOrigin
        ](self.qt_param_ptrs[qb])
        Self.QModel.forward_gpu_no_cache[B](
            ctx,
            q_out,
            q_in,
            qtb_params,
            q_state,
            self.q_ws_buf,
        )
        comptime decode_add_scaled = tdmpc2_decode_add_scaled_kernel[
            dtype, B, Self.NUM_BINS
        ]
        ctx.enqueue_function[decode_add_scaled](
            q_logits,
            bins,
            v_out,
            half,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )


# =============================================================================
# TDMPC2CPURolloutCallback — CPU adapter for ``MPPICPU``
# =============================================================================


struct TDMPC2CPURolloutCallback[
    OBS_DIM_PARAM: Int,
    ACTION_DIM_PARAM: Int,
    LATENT_DIM_PARAM: Int = 512,
    MLP_DIM: Int = 512,
    ENC_DIM: Int = 256,
    NUM_BINS: Int = 101,
    NUM_Q: Int = 5,
    SIMPLEX_DIM: Int = 8,
    V_MIN: Float64 = -10.0,
    V_MAX: Float64 = 10.0,
    ENC_LR: Float64 = 9e-5,
    WM_LR: Float64 = 3e-4,
    PI_LR: Float64 = 3e-4,
](Movable, ImplicitlyDestructible, RolloutCallbackCPU):
    """``RolloutCallbackCPU`` against TDMPC2's ``WorldModel``.

    Used by the agent's eval-only ``select_action`` path (B = 1).
    The three trait methods build temporary ``(1, dim)``
    ``LayoutTensor`` views over ``InlineArray`` scratch on each call
    and dispatch to the matching ``WorldModel.*_forward[1]`` method.
    Per-call allocation overhead is microseconds — negligible vs the
    forward passes themselves.

    The terminal-value method **matches the reference TD-MPC2 recipe**
    (`tdmpc2.py:137`, ``Q(..., return_type='avg')``): sample two
    distinct random Q-targets via ``random_float64()`` and average
    their decoded values. The legacy CPU ``plan()`` used
    ``q_min_forward`` (min over all 5 Q-targets) which is **over-
    pessimistic** vs reference — fixing that bug here is one of the
    motivations for migrating off ``plan()``.
    """

    comptime LATENT_DIM: Int = Self.LATENT_DIM_PARAM
    comptime ACTION_DIM: Int = Self.ACTION_DIM_PARAM
    comptime ZA_DIM: Int = Self.LATENT_DIM + Self.ACTION_DIM

    comptime WM = WorldModel[
        Self.OBS_DIM_PARAM,
        Self.ACTION_DIM_PARAM,
        Self.LATENT_DIM_PARAM,
        Self.MLP_DIM,
        Self.ENC_DIM,
        Self.NUM_BINS,
        Self.NUM_Q,
        Self.SIMPLEX_DIM,
        Self.V_MIN,
        Self.V_MAX,
        Self.ENC_LR,
        Self.WM_LR,
        Self.PI_LR,
    ]

    var wm: UnsafePointer[Self.WM, MutAnyOrigin]
    """Borrowed pointer to the agent's WorldModel — caller guarantees
    lifetime exceeds the callback's. Raw pointer rather than reference
    because the callback is constructed locally inside
    ``select_action`` and the WorldModel lives on the agent struct."""

    def __init__(
        out self,
        wm: UnsafePointer[Self.WM, MutAnyOrigin],
    ) raises:
        self.wm = wm

    def policy_action_cpu(
        mut self,
        z: List[Float64],
        mut action_out: List[Float64],
    ) raises:
        # WorldModel.policy_forward[1] writes mean + log_std as
        # separate (1, ACTION_DIM) outputs — we only need mean.
        var z_arr = InlineArray[Scalar[dtype], Self.LATENT_DIM](fill=0)
        for i in range(Self.LATENT_DIM):
            z_arr[i] = Scalar[dtype](z[i])
        var mean_arr = InlineArray[Scalar[dtype], Self.ACTION_DIM](
            uninitialized=True
        )
        var log_std_arr = InlineArray[
            Scalar[dtype], Self.ACTION_DIM
        ](uninitialized=True)
        var z_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.LATENT_DIM), MutAnyOrigin
        ](z_arr.unsafe_ptr())
        var mean_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTION_DIM), MutAnyOrigin
        ](mean_arr.unsafe_ptr())
        var log_std_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTION_DIM), MutAnyOrigin
        ](log_std_arr.unsafe_ptr())
        self.wm[].policy_forward[1](z_t, mean_t, log_std_t)
        for i in range(Self.ACTION_DIM):
            action_out[i] = Float64(mean_arr[i])

    def rollout_step_cpu(
        mut self,
        z: List[Float64],
        a: List[Float64],
        mut z_next_out: List[Float64],
    ) raises -> Float64:
        # 1. Build za = [z, a] in InlineArray scratch.
        var za = InlineArray[Scalar[dtype], Self.ZA_DIM](fill=0)
        for i in range(Self.LATENT_DIM):
            za[i] = Scalar[dtype](z[i])
        for i in range(Self.ACTION_DIM):
            za[Self.LATENT_DIM + i] = Scalar[dtype](a[i])
        var za_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ZA_DIM), MutAnyOrigin
        ](za.unsafe_ptr())

        # 2. Dynamics: z' = Dyn(za)
        var z_next_arr = InlineArray[
            Scalar[dtype], Self.LATENT_DIM
        ](uninitialized=True)
        var z_next_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.LATENT_DIM), MutAnyOrigin
        ](z_next_arr.unsafe_ptr())
        self.wm[].dynamics_forward[1](za_t, z_next_t)
        for i in range(Self.LATENT_DIM):
            z_next_out[i] = Float64(z_next_arr[i])

        # 3. Reward logits + categorical → scalar decode.
        var rew_logits = InlineArray[
            Scalar[dtype], Self.NUM_BINS
        ](uninitialized=True)
        var rew_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.NUM_BINS), MutAnyOrigin
        ](rew_logits.unsafe_ptr())
        self.wm[].reward_forward[1](za_t, rew_t)
        var rew_logits_f32 = InlineArray[Float32, Self.NUM_BINS](
            uninitialized=True
        )
        for i in range(Self.NUM_BINS):
            rew_logits_f32[i] = Float32(rew_logits[i])
        return Float64(
            decode_value_batch_scalar[Self.NUM_BINS](
                rew_logits_f32, self.wm[].bins
            )
        )

    def terminal_value_cpu(
        mut self,
        z: List[Float64],
    ) raises -> Float64:
        """Q-bootstrap at end-of-horizon — matches the reference
        TDMPC2 recipe: π(z) → tanh(mean) → 2 random target-Q heads →
        decode each → 0.5 * (q_a + q_b).

        Uses two distinct Q indices sampled via ``random_float64()``
        (same global RNG stream the planner uses for sampling noise —
        deterministic w.r.t. ``_set_seed``). Fixes the OLD CPU
        ``plan()`` bug of min-over-all-Q (over-pessimistic vs
        reference's avg-of-2).
        """
        var z_arr = InlineArray[Scalar[dtype], Self.LATENT_DIM](fill=0)
        for i in range(Self.LATENT_DIM):
            z_arr[i] = Scalar[dtype](z[i])
        var z_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.LATENT_DIM), MutAnyOrigin
        ](z_arr.unsafe_ptr())

        # π(z) → mean (deterministic — tanh applied below for the
        # actor's bounded action).
        var mean_arr = InlineArray[Scalar[dtype], Self.ACTION_DIM](
            uninitialized=True
        )
        var log_std_arr = InlineArray[
            Scalar[dtype], Self.ACTION_DIM
        ](uninitialized=True)
        var mean_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTION_DIM), MutAnyOrigin
        ](mean_arr.unsafe_ptr())
        var log_std_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTION_DIM), MutAnyOrigin
        ](log_std_arr.unsafe_ptr())
        self.wm[].policy_forward[1](z_t, mean_t, log_std_t)

        # Build za_terminal = [z, tanh(mean)] for the Q-net input.
        var za_term = InlineArray[Scalar[dtype], Self.ZA_DIM](fill=0)
        for i in range(Self.LATENT_DIM):
            za_term[i] = Scalar[dtype](z[i])
        for i in range(Self.ACTION_DIM):
            var raw = Float64(mean_arr[i])
            # Bounded actor uses tanh; eval-mode action is the
            # deterministic squashed mean.
            var act = (
                1.0
                if raw > 20.0
                else (-1.0 if raw < -20.0 else _tanh_f64(raw))
            )
            za_term[Self.LATENT_DIM + i] = Scalar[dtype](act)

        # Sample 2 distinct Q indices, run all 5 target-Q forwards
        # in one call (q_forward writes a flat NUM_Q*1*NUM_BINS
        # tensor with use_target=True), then decode + average the
        # picked pair. Slightly wasteful (5 forwards instead of 2)
        # but ``q_forward_single_no_cache`` doesn't take a
        # ``use_target`` flag, and this is the eval-only B=1 path
        # so the cost is negligible.
        var qa = Int(random_float64() * Float64(Self.NUM_Q)) % Self.NUM_Q
        var qb_offset = (
            Int(random_float64() * Float64(Self.NUM_Q - 1))
            % (Self.NUM_Q - 1)
        )
        var qb = (qa + 1 + qb_offset) % Self.NUM_Q

        var za_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ZA_DIM), MutAnyOrigin
        ](za_term.unsafe_ptr())

        var q_size = Self.NUM_Q * Self.NUM_BINS
        var all_q_logits = List[Scalar[dtype]](
            length=q_size, fill=Scalar[dtype](0)
        )
        var all_q_t = LayoutTensor[
            dtype, Layout.row_major(Self.NUM_Q * 1 * Self.NUM_BINS), MutAnyOrigin
        ](all_q_logits.unsafe_ptr())
        self.wm[].q_forward[1](za_t, all_q_t, True)

        var v_a = self._decode_one_q(all_q_logits, qa)
        var v_b = self._decode_one_q(all_q_logits, qb)
        return 0.5 * (v_a + v_b)

    def _decode_one_q(
        mut self,
        all_q_logits: List[Scalar[dtype]],
        q_idx: Int,
    ) raises -> Float64:
        """Decode one slice of the (NUM_Q * NUM_BINS) flat output of
        ``q_forward`` (B=1) into a scalar via softmax-expected-bin +
        symexp.
        """
        var base = q_idx * Self.NUM_BINS
        var logits_f32 = InlineArray[Float32, Self.NUM_BINS](
            uninitialized=True
        )
        for i in range(Self.NUM_BINS):
            logits_f32[i] = Float32(all_q_logits[base + i])
        return Float64(
            decode_value_batch_scalar[Self.NUM_BINS](
                logits_f32, self.wm[].bins
            )
        )


@always_inline
def _tanh_f64(x: Float64) -> Float64:
    """Float64 tanh using Float32 underneath — sufficient precision
    for action clamping (we don't need exact reproducibility of
    GPU's `tdmpc2_apply_tanh_build_za_deterministic_kernel` since
    this is the eval-only CPU path).
    """
    return Float64(tanh(Float32(x)))