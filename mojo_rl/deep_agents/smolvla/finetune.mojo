# +--------------------------------------------------------------------------+ #
# | SmolVLA — what gets trained, and the optimizer walk over exactly that set
# +--------------------------------------------------------------------------+ #
"""The trainable set, written down ONCE.

`train_expert_only = True` makes the trainable parameters five components, and
`train_state_proj` decides whether there is a sixth:

    the 16 expert layers + the expert's final norm
    action_in       [ADIM -> EW]
    time_mlp_in     [2*EW -> EW]
    time_mlp_out    [EW   -> EW]
    action_out      [EW   -> ADIM]
    state_proj      [32   -> 960]        only when TRAIN_STATE_PROJ

Everything else — the SigLIP tower, the sixteen VLM layers, the connector, the
token embedding, `state_proj` — is frozen, and under this regime is also
upstream of nothing that is trained, so no gradient is even formed for it.

## Why these two functions exist rather than five call sites

A training loop has to do three things to each component: zero its gradient,
accumulate into it, apply the update. Miss one component from the ZERO list
and its gradient accumulates across every step of the run — a growing,
plausible number that eventually dominates. Miss one from the UPDATE list and
it silently never trains at all. Neither raises, neither shows up in a loss
curve, and both would be found months later by a model that is worse than it
should be for reasons nobody can point at.

So the set is enumerated exactly twice, here, adjacent, in the same order.
`test_finetune_overfit.mojo` then asserts that after a run EVERY parameter
group has actually moved, which is the check that catches an omission from
either list.

⚠ **`TRAIN_STATE_PROJ` is a COMPTIME parameter on both walks, not a runtime
flag and not a sixth call site.** The shipped config trains `state_proj`, and
it is the one trainable parameter that sits UPSTREAM of the frozen tower — its
gradient arrives only after a full backward through all sixteen VLM layers
(`SmolVLAPrefill.backward`). Turning it on therefore changes what a training
step must compute, not just which parameters it updates, so the flag being
visible in the type is worth more than the convenience of a bool.

⚠ **The per-parameter walk, not Adam's grouped arena.** `Adam.adopt` packs
every parameter into one slab so the GPU update is a single kernel instead of
one per parameter — worth ~10% of all launches in the ACT profile. Adopting
requires the trainable set to be ONE `ParamWalkable`, and it is five separate
objects that the inference path also owns. That refactor belongs with the
Jetson work, where the launch overhead is actually being paid; a correctness
milestone does not need it. The cost is named here so it is a decision and
not an oversight.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.param import ParamVersionBump
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.primitives.linear import Linear

from .expert import SmolVLAExpert
from .grad_ops import suffix_tail


def state_proj_backward[
    target: StaticString, B: Int, P: Int, W: Int, SDIM: Int
](
    mut state_proj: Linear[SDIM, W],
    mut state: Tensor,
    mut grad_x: Tensor,
    mut g_tok: Tensor,
    mut grad_state: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`state_proj`'s weight gradient, from the prefix embeddings' gradient.

    The state is the LAST prefix token — `smolvla_ar` lays out image, then
    language, then state — and it is written UNSCALED, unlike the image and
    language segments which carry a sqrt(W) factor. So its slice of `grad_x`
    is dL/d(state_proj's output) directly, with nothing to undo.

    ⚠ The slice is taken with `suffix_tail`, the same routine that separates
    the expert's K/V from the VLM's in the denoise backward. Same batch-major
    rule, one implementation: `[B, P*W]` with the last `W` of each row.

    `grad_state` is written and discarded — the robot's pose is data — but
    `Linear.vjp` needs a destination, and forming it is what accumulates
    `state_proj.weight.grd`.
    """
    suffix_tail[target, B, P * W, (P - 1) * W, W](grad_x, g_tok, ctx)
    state_proj.vjp[target, B](
        TensorRefs[1](state), g_tok, TensorRefs[1](grad_state), ctx
    )


def zero_trainable_grads[
    target: StaticString,
    LAYERS: Int, EW: Int, EFF: Int, W: Int, KVW: Int, ADIM: Int,
    SDIM: Int = 32, VW: Int = 960, TRAIN_STATE_PROJ: Bool = False,
](
    mut expert: SmolVLAExpert[LAYERS, EW, EFF, W, KVW, 2],
    mut action_in: Linear[ADIM, EW],
    mut time_mlp_in: Linear[2 * EW, EW],
    mut time_mlp_out: Linear[EW, EW],
    mut action_out: Linear[EW, ADIM],
    mut state_proj: Linear[SDIM, VW],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Zero every trainable gradient. Call ONCE per step, before the forward.

    ⚠ `Linear.vjp` ACCUMULATES (`grad_w += ...`), which is the `nn` convention
    and is what makes gradient accumulation across micro-batches possible. It
    also means a forgotten zero is a silent running sum.
    """
    expert.zero_grad[target](ctx)
    action_in.zero_grad[target](ctx)
    time_mlp_in.zero_grad[target](ctx)
    time_mlp_out.zero_grad[target](ctx)
    action_out.zero_grad[target](ctx)
    comptime if TRAIN_STATE_PROJ:
        state_proj.zero_grad[target](ctx)


def adam_step_trainables[
    target: StaticString,
    LAYERS: Int, EW: Int, EFF: Int, W: Int, KVW: Int, ADIM: Int,
    SDIM: Int = 32, VW: Int = 960, TRAIN_STATE_PROJ: Bool = False,
](
    mut opt: Adam,
    mut expert: SmolVLAExpert[LAYERS, EW, EFF, W, KVW, 2],
    mut action_in: Linear[ADIM, EW],
    mut time_mlp_in: Linear[2 * EW, EW],
    mut time_mlp_out: Linear[EW, EW],
    mut action_out: Linear[EW, ADIM],
    mut state_proj: Linear[SDIM, VW],
    ctx: Optional[DeviceContext] = None,
) raises:
    """One Adam update over the same five components `zero_trainable_grads`
    covers.

    ⚠ `begin_step()` is called ONCE and then five walks follow. `Adam.step`
    bundles the two, which is right for a single model and wrong here: calling
    it five times would advance the step counter five times per optimizer
    step, so the bias corrections would run ahead of the moments and the early
    steps would take the wrong size.
    """
    opt.begin_step()
    expert.for_each_param[target](opt, ctx, String("expert"))
    action_in.for_each_param[target](opt, ctx, String("action_in"))
    time_mlp_in.for_each_param[target](opt, ctx, String("time_mlp_in"))
    time_mlp_out.for_each_param[target](opt, ctx, String("time_mlp_out"))
    action_out.for_each_param[target](opt, ctx, String("action_out"))
    comptime if TRAIN_STATE_PROJ:
        state_proj.for_each_param[target](opt, ctx, String("state_proj"))

    # ⚠ The version bump `Adam.step` does after its walk, which is NOT
    # cosmetic: leaves that cache a derived form of a weight (the bf16 cast,
    # split-K's padded copy) gate that cache on the version, and a bump that
    # never happens leaves the forward reading pre-update weights forever.
    # That exact defect has been shipped here before.
    var bump = ParamVersionBump()
    expert.for_each_param[target](bump, ctx, String("expert"))
    action_in.for_each_param[target](bump, ctx, String("action_in"))
    time_mlp_in.for_each_param[target](bump, ctx, String("time_mlp_in"))
    time_mlp_out.for_each_param[target](bump, ctx, String("time_mlp_out"))
    action_out.for_each_param[target](bump, ctx, String("action_out"))
    comptime if TRAIN_STATE_PROJ:
        state_proj.for_each_param[target](bump, ctx, String("state_proj"))
