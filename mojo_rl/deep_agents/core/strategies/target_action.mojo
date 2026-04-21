"""Target action computation strategies for off-policy agents.

Each strategy computes next-state actions for TD target computation.
Methods are generic over ActorModel/ActorOpt (compile-time type params
on the method, like BATCH on nn/Model.forward).

All LayoutTensor dimensions are derived from ActorModel (IN_DIM, OUT_DIM,
PARAM_SIZE) to match Network[ActorModel, ActorOpt].forward expectations.

Implementations:
  - DeterministicTarget: actor_target(next_obs) (DDPG)
  - SmoothedTarget: actor_target(next_obs) + clipped noise (TD3)
  - ReparamTarget: current_actor(next_obs) -> rsample + log_probs (SAC)
"""

from layout import Layout, LayoutTensor
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.training import Network
from mojo_rl.nn.gpu.random import gaussian_noise
from mojo_rl.nn.model.stochastic_actor import rsample
from mojo_rl.deep_agents.core.kernels import (
    add_gaussian_noise_kernel,
    sac_rsample_with_cache_kernel,
)


trait TargetAction:
    """Trait for target action strategies. Methods are duck-typed."""

    comptime NEEDS_LOG_PROBS: Bool

    @staticmethod
    def ws_size[BATCH: Int, ACTIONS: Int, ACTOR_OUT: Int]() -> Int:
        ...

    @staticmethod
    def compute_cpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
    ](
        next_obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        mut out_actions: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        out_log_probs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        ws: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        action_scale: Float64,
    ):
        ...

    @staticmethod
    def compute_gpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
    ](
        ctx: DeviceContext,
        next_obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        mut out_actions: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        mut out_log_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        actor_ws: DeviceBuffer[dtype],
        strat_ws: DeviceBuffer[dtype],
        rng_counter: DeviceBuffer[DType.uint32],
        action_scale: Scalar[dtype],
    ) raises:
        ...


# =============================================================================
# DeterministicTarget — DDPG: forward target actor
# =============================================================================


struct DeterministicTarget(TargetAction):
    """Compute next actions by forwarding the target actor network.

    Used by DDPG: next_actions = actor_target(next_obs).
    No noise, no log_probs. Simplest target action computation.
    """

    comptime NEEDS_LOG_PROBS: Bool = False

    @staticmethod
    def ws_size[BATCH: Int, ACTIONS: Int, ACTOR_OUT: Int]() -> Int:
        """No extra workspace needed."""
        return 0

    @staticmethod
    def compute_cpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
    ](
        next_obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        mut out_actions: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        out_log_probs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        ws: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        action_scale: Float64,  # unused by DDPG (actor outputs tanh; caller scales if needed)
    ):
        """Forward target actor on next_obs -> next_actions."""
        # Rebind out_actions to ActorModel.OUT_DIM (== ACTIONS for DDPG)
        var out_act = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](out_actions.ptr)
        Network[ActorModel, ActorOpt].forward[BATCH](
            next_obs, out_act, actor_params
        )

    @staticmethod
    def compute_gpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
    ](
        ctx: DeviceContext,
        next_obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        mut out_actions: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        mut out_log_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        actor_ws: DeviceBuffer[dtype],
        strat_ws: DeviceBuffer[dtype],
        rng_counter: DeviceBuffer[DType.uint32],
        action_scale: Scalar[dtype],  # unused by DDPG
    ) raises:
        """GPU forward target actor on next_obs -> next_actions."""
        var out_act = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](out_actions.ptr)
        Network[ActorModel, ActorOpt].forward_gpu[BATCH](
            ctx, next_obs, out_act, actor_params, actor_ws
        )


# =============================================================================
# SmoothedTarget — TD3: forward target actor + clipped noise
# =============================================================================


struct SmoothedTarget[
    target_noise_std: Float64 = 0.2,
    target_noise_clip: Float64 = 0.5,
](TargetAction):
    """Compute next actions with target policy smoothing.

    Used by TD3: next_actions = actor_target(next_obs) + clip(noise),
    where noise ~ N(0, target_noise_std), clipped to [-clip, clip],
    and final actions clipped to [-1, 1].

    target_noise_std and target_noise_clip are compile-time parameters
    (fixed hyperparams, like Adam's BETA1).
    """

    comptime NEEDS_LOG_PROBS: Bool = False

    @staticmethod
    def ws_size[BATCH: Int, ACTIONS: Int, ACTOR_OUT: Int]() -> Int:
        """No extra workspace needed (noise applied in-place)."""
        return 0

    @staticmethod
    def compute_cpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
    ](
        next_obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        mut out_actions: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        out_log_probs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        ws: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        action_scale: Float64,  # unused by TD3 (noise added in unit-range)
    ):
        """Forward target actor, then add clipped Gaussian noise."""

        # Rebind out_actions to ActorModel.OUT_DIM (== ACTIONS for TD3)
        var out_act = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](out_actions.ptr)
        Network[ActorModel, ActorOpt].forward[BATCH](
            next_obs, out_act, actor_params
        )

        # Target policy smoothing: add clipped noise, clip actions to [-1, 1]
        for b in range(BATCH):
            for i in range(ACTIONS):
                var idx = b * ACTIONS + i
                var noise = gaussian_noise() * Self.target_noise_std
                if noise > Self.target_noise_clip:
                    noise = Self.target_noise_clip
                elif noise < -Self.target_noise_clip:
                    noise = -Self.target_noise_clip
                var noisy_a = Float64(out_actions.ptr[idx]) + noise
                if noisy_a > 1.0:
                    noisy_a = 1.0
                elif noisy_a < -1.0:
                    noisy_a = -1.0
                out_actions.ptr[idx] = Scalar[dtype](noisy_a)

    @staticmethod
    def compute_gpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
    ](
        ctx: DeviceContext,
        next_obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        mut out_actions: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        mut out_log_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        actor_ws: DeviceBuffer[dtype],
        strat_ws: DeviceBuffer[dtype],
        rng_counter: DeviceBuffer[DType.uint32],
        action_scale: Scalar[dtype],  # unused by TD3
    ) raises:
        """GPU forward target actor, then add clipped Gaussian noise.

        strat_ws must hold at least BATCH * ACTIONS elements for clean actions.
        """
        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        # Forward actor into strat_ws (clean actions before noise)
        var clean_act = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](strat_ws.unsafe_ptr())
        Network[ActorModel, ActorOpt].forward_gpu[BATCH](
            ctx, next_obs, clean_act, actor_params, actor_ws
        )

        # Rebind clean_act as [BATCH, ACTIONS] for the noise kernel
        var clean_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](strat_ws.unsafe_ptr())

        var noise_std_s = Scalar[dtype](Self.target_noise_std)
        var noise_clip_s = Scalar[dtype](Self.target_noise_clip)
        var act_min_s = Scalar[dtype](-1.0)
        var act_max_s = Scalar[dtype](1.0)
        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](rng_counter.unsafe_ptr())

        @always_inline
        def noise_wrapper(
            noisy: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            clean: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            ns: Scalar[dtype],
            nc: Scalar[dtype],
            amin: Scalar[dtype],
            amax: Scalar[dtype],
            rng: LayoutTensor[
                DType.uint32, Layout.row_major(1), MutAnyOrigin
            ],
        ):
            add_gaussian_noise_kernel[dtype, BATCH, ACTIONS](
                noisy, clean, ns, nc, amin, amax, rng
            )

        ctx.enqueue_function[noise_wrapper, noise_wrapper](
            out_actions,
            clean_act_t,
            noise_std_s,
            noise_clip_s,
            act_min_s,
            act_max_s,
            rng_t,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )


# =============================================================================
# ReparamTarget — SAC: current actor -> rsample + log_probs
# =============================================================================


struct ReparamTarget(TargetAction):
    """Compute next actions via reparameterized sampling from current actor.

    Used by SAC: forward current actor (not target!) -> extract mean+log_std
    -> rsample with tanh squashing -> actions + log_probs.

    No target actor is used — SAC computes targets from the online policy.
    The log_probs are needed by EntropicTwinQTarget for the entropy term.

    Extra workspace: BATCH * ActorModel.OUT_DIM for the raw actor output.
    ACTIONS (action_dim) must be passed separately because
    ActorModel.OUT_DIM = 2 * ACTIONS for SAC's Parallel[mean, log_std].
    """

    comptime NEEDS_LOG_PROBS: Bool = True

    @staticmethod
    def ws_size[BATCH: Int, ACTIONS: Int, ACTOR_OUT: Int]() -> Int:
        """Extra workspace for raw actor output + eps_cache for rsample kernel."""
        return BATCH * ACTOR_OUT + BATCH * ACTIONS

    @staticmethod
    def compute_cpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
    ](
        next_obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        mut out_actions: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        out_log_probs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        ws: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        action_scale: Float64,
    ):
        """Forward current actor -> rsample -> actions + log_probs.

        Output actions are scaled by action_scale so the critic sees the same
        distribution as the replay buffer (which stores scaled actions).

        Workspace layout: [BATCH * ActorModel.OUT_DIM] raw actor output.
        """
        comptime ACTOR_OUT = ActorModel.OUT_DIM

        # Forward actor -> raw output [BATCH, ACTOR_OUT] (mean || log_std)
        var raw_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](ws)
        Network[ActorModel, ActorOpt].forward[BATCH](
            next_obs, raw_out, actor_params
        )

        # Extract mean and log_std from Parallel output
        var mean = InlineArray[Scalar[dtype], BATCH * ACTIONS](
            uninitialized=True
        )
        var log_std = InlineArray[Scalar[dtype], BATCH * ACTIONS](
            uninitialized=True
        )
        for b in range(BATCH):
            for a in range(ACTIONS):
                mean[b * ACTIONS + a] = ws[b * ACTOR_OUT + a]
                log_std[b * ACTIONS + a] = ws[b * ACTOR_OUT + ACTIONS + a]

        # Generate noise for reparameterization
        var noise = InlineArray[Scalar[dtype], BATCH * ACTIONS](
            uninitialized=True
        )
        for i in range(BATCH * ACTIONS):
            noise[i] = Scalar[dtype](gaussian_noise())

        # Reparameterized sample: z = mean + exp(log_std) * noise, action = tanh(z)
        var out_lp = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](out_log_probs)
        rsample[BATCH, ACTIONS](
            LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin](
                mean.unsafe_ptr()
            ),
            LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin](
                log_std.unsafe_ptr()
            ),
            LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin](
                noise.unsafe_ptr()
            ),
            out_actions,
            out_lp,
        )

        # Scale actions (rsample writes unscaled tanh(z) in [-1, 1]).
        # Buffer stores SCALED actions, so target critic must see scaled too.
        if action_scale != 1.0:
            var scale_s = Scalar[dtype](action_scale)
            for b in range(BATCH):
                for a in range(ACTIONS):
                    out_actions[b, a] = out_actions[b, a] * scale_s

        # Guard NaN in log_probs
        for b in range(BATCH):
            var lp = Float64(out_log_probs[b])
            if lp != lp or lp > 100.0 or lp < -100.0:
                out_log_probs[b] = Scalar[dtype](-1.0)

    @staticmethod
    def compute_gpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
    ](
        ctx: DeviceContext,
        next_obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        mut out_actions: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ],
        mut out_log_probs: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        actor_ws: DeviceBuffer[dtype],
        strat_ws: DeviceBuffer[dtype],
        rng_counter: DeviceBuffer[DType.uint32],
        action_scale: Scalar[dtype],
    ) raises:
        """GPU forward current actor -> rsample -> actions + log_probs.

        Output actions are scaled by action_scale so the critic sees the same
        distribution as the replay buffer (which stores scaled actions).

        strat_ws layout: [BATCH * ACTOR_OUT] raw actor output.
        The rsample kernel reads raw_out and writes actions + log_probs.
        """
        comptime BLOCKS = (BATCH + TPB - 1) // TPB
        comptime ACTOR_OUT = ActorModel.OUT_DIM

        # Forward actor -> raw_out [BATCH, ACTOR_OUT] in strat_ws
        var raw_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](strat_ws.unsafe_ptr())
        Network[ActorModel, ActorOpt].forward_gpu[BATCH](
            ctx, next_obs, raw_out, actor_params, actor_ws
        )

        # SAC rsample: raw_out -> actions, log_probs (no eps_cache needed
        # for target actions, but kernel writes it — use tail of strat_ws)
        var eps_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](strat_ws.unsafe_ptr() + BATCH * ACTOR_OUT)

        var log_std_min_s = Scalar[dtype](-5.0)
        var log_std_max_s = Scalar[dtype](2.0)
        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](rng_counter.unsafe_ptr())

        @always_inline
        def rsample_wrapper(
            acts: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            lp: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            eps: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            ao: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, ACTIONS + ACTIONS),
                MutAnyOrigin,
            ],
            lsmin: Scalar[dtype],
            lsmax: Scalar[dtype],
            ascale: Scalar[dtype],
            rng: LayoutTensor[
                DType.uint32, Layout.row_major(1), MutAnyOrigin
            ],
        ):
            sac_rsample_with_cache_kernel[dtype, BATCH, ACTIONS](
                acts, lp, eps, ao, lsmin, lsmax, ascale, rng
            )

        ctx.enqueue_function[rsample_wrapper, rsample_wrapper](
            out_actions,
            out_log_probs,
            eps_cache_t,
            raw_out,
            log_std_min_s,
            log_std_max_s,
            action_scale,
            rng_t,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
