# +--------------------------------------------------------------------------+ #
# | ACTTrainer — offline behaviour cloning over ACTLossGraph
# +--------------------------------------------------------------------------+ #
"""Owns the loss graph + one AdamW + the scratch a step needs.

Transposed from `experimental/lewm/trainer.mojo`, which is the established
shape for "one ComputeGraph, one optimizer": a step is

    zero_grad -> set_input x4 -> forward -> seed grad 1/B -> vjp -> clip -> step

On GPU the optimizer runs in GROUPED ARENA mode: `Adam.adopt` packs every
parameter into contiguous val/grd/m/v device buffers, so `zero_grad`, the
grad-norm clip and the update are a handful of kernels over the whole model
instead of one kernel per parameter. On CPU `adopt` is a no-op and `Adam` is
used as a `ParamVisitor` over `graph.for_each_param`, as before.

CPU and GPU: `target` is a comptime parameter. On GPU the sampler still produces
host `List`s, so a step is host-fill -> upload -> graph -> download the three
per-sample loss vectors. Weights, activations and gradients stay on device.

## The arena, and when it is engaged

`Adam.adopt` used to require a `Module`, which a `ComputeGraph` is not, so this
trainer walked `for_each_param` launching one kernel per parameter. The bound is
now `ParamWalkable` (`nn/core/param.mojo`) — the param walk and nothing else —
which a graph satisfies. Measured on `act_so101_profile_gpu.mojo`, the per-param
path was 10.5% of every kernel launch in the run for 1.0% of the kernel time,
and the unconditional host-side grad-norm walk below it was a device
synchronization PER PARAMETER (see `_SumSq`).

⚠ Adoption is LAZY — first `train_step`, not `make`. `ParamArena.adopt` rebinds
each Param's device buffer to a slice of the arena, and `Tensor.upload`
RECREATES a device buffer: anything that uploads into a parameter after
adoption (a checkpoint load, a test injecting reference weights) would silently
detach it, after which the grouped step updates arena memory the model no
longer reads. Loading before the first step is therefore always safe; the
checkpoint and refload paths additionally use `upload_resident`, which reuses
the existing buffer, so a load AFTER adoption is safe too.

## Logging: per-step readback vs device accumulators

There are two families of step method and they must not be mixed in one loop:

  * `train_step` / `train_step_device` / `eval_step*` return an
    `ACTStepResult`, which costs four D2Hs behind two device synchronizations
    every step;
  * `train_step_device_accum` / `eval_step_device_accum` /
    `eval_step_resident_accum` return NOTHING and fold the same four scalars
    into device accumulators, drained by `train_metrics` / `val_metrics` at
    logging cadence.

The second family exists because those four downloads were the last host
traffic in an ACT step — see `docs/ACT_GPU_DATA_PATH.md`. Removing them makes
the per-step sequence pure kernels, which is what a CUDA graph can capture.

⚠ A loop that called both families and then read `train_metrics` would report a
window covering only half its steps, and the number would look plausible.

## Train vs eval

`train_mode(True/False)` sets three things together, because getting one of them
wrong is silent:

  * `set_attr["training"]` — dropout, and BatchNorm batch-vs-running statistics;
  * `zs.multiplier` — 1.0 samples the CVAE latent, 0.0 makes the latent token
    `latent_out_proj(0)`, which is exactly the reference's test-time `z = 0`
    (`detr_vae.py:110`);
  * `z.deterministic` — off in training (the reparameterization is the point),
    on in eval so a validation number is reproducible rather than a fresh draw.
    `set_deterministic_latent(True)` pins it in training too, and STICKS —
    writing the node attribute directly does not, because every mode switch
    rewrites it.

⚠ The reference's model selection is **validation L1**, not the total loss
(`imitate_episodes.py` tracks `min_val_loss` over the summed dict but the L1 term
is what the policy is judged on). `eval_step` returns both.

## Deviations, all deliberate — see `config.mojo` for the full list

* ONE learning rate. The reference gives backbone params `lr_backbone = 1e-5` in
  a second AdamW group; `Adam` here has no name filter, and a 10x-lower rate on
  a FROM-SCRATCH backbone would freeze the vision tower rather than gently
  fine-tune it.
* Gradient clipping IS applied. The reference parses `--clip_max_norm 0.1` and
  never uses it; an unclipped from-scratch ResNet on four episodes is a
  divergence waiting to happen. Set `max_grad_norm = 0.0` to match the
  reference exactly.
"""

from std.math import sqrt

from max.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.constants import DT
from mojo_rl.nn import Adam, Kaiming, Tensor
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.models.resnet18 import (
    RESNET18_OUT_CH,
    ResNet18Backbone,
    ResNet18OutH,
    ResNet18OutW,
)
from mojo_rl.nn.core.checkpoint import (
    BinaryCheckpointReader,
    BinaryCheckpointWriter,
    _is_v3_header,
    _read_file_bytes,
    _write_file_bytes,
)
from mojo_rl.deep_agents.loss.seed_grad_inv_batch import seed_grad_inv_batch
from ..training.device_mean_accum import DeviceMeanAccum

from .config import (
    ACT_CLIP_MAX_NORM,
    ACT_DROPOUT,
    ACT_KL_WEIGHT,
    ACT_LR,
    ACT_WEIGHT_DECAY,
)
from .data_gpu import ACTDeviceDataset
from .loss_graph import ACTLossGraph
from .refload import LoadPrefixedParams, RefDump


# ── grad-norm clip over a graph ──────────────────────────────────────────
# The graph owns every parameter but is not a `Module`, so `Adam.clip_grads`
# (Module-constrained) cannot take it. Two visitor passes: sum of squares, then
# scale. Mirrors `lewm/trainer.mojo`'s `_SumSqV`, CPU-only.


struct _SumSq(ParamVisitor):
    """Sum of squared gradients over every parameter.

    ⚠ CPU ONLY now. On GPU this DOWNLOADS each gradient slab to sum it on the
    host — a device synchronization PER PARAMETER, and (contrary to what this
    note used to claim) it was NOT gated on `max_grad_norm`: only the SCALING
    was conditional, the sum ran on every step because the norm is reported in
    `ACTStepResult`. For an ACT model that is ~150 syncs per step, which is
    most of the 148 synchronizations per pass in `docs/GPU_STEP_PERF.md`. The
    GPU path now uses the optimizer's arena reduction (`clip_grads_device`):
    three kernels over the contiguous grad arena and one D2H for the reported
    norm.
    """

    var sum_sq: Float64

    def __init__(out self):
        self.sum_sq = 0.0

    def __init__(out self, *, deinit move: Self):
        self.sum_sq = move.sum_sq

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target != "cpu":
            grad.download(ctx.value())
        for i in range(N):
            var g = Float64(grad.data[i])
            self.sum_sq += g * g


struct _ScaleGrads(ParamVisitor):
    var scale: Scalar[DT]

    def __init__(out self):
        self.scale = Scalar[DT](1.0)

    def __init__(out self, scale: Scalar[DT]):
        self.scale = scale

    def __init__(out self, *, deinit move: Self):
        self.scale = move.scale

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target != "cpu":
            # ⚠ Unreachable: the GPU clip is the optimizer's arena path. Kept
            # only so the visitor stays target-generic — and `upload` here
            # would DETACH the param from the arena (see the header).
            for i in range(N):
                grad.data[i] = grad.data[i] * self.scale
            grad.upload_resident(ctx.value())
        else:
            for i in range(N):
                grad.data[i] = grad.data[i] * self.scale


@fieldwise_init
struct ACTStepResult(ImplicitlyCopyable):
    """What one step produced. `l1` is the reference's model-selection metric;
    `loss` is what the optimizer actually descended."""

    var loss: Float64
    var l1: Float64
    var kl: Float64
    var grad_norm: Float64


@fieldwise_init
struct ACTWindowMetrics(ImplicitlyCopyable):
    """The same four scalars, MEANED OVER A WINDOW of steps.

    Deliberately not an `ACTStepResult`. The two carry the same field names and
    mean different things, and the way that goes wrong is silent: a window mean
    logged as a step value simply looks like a smoother curve. `n` is how many
    steps the window held — 0 means nothing was accumulated and the means are
    all 0, which is not the same as a loss of 0."""

    var loss: Float64
    var l1: Float64
    var kl: Float64
    var grad_norm: Float64
    var n: Int


struct ACTMetricAccum(Movable & Deinitable):
    """(loss, l1, kl, grad_norm) over a logging window.

    GPU: four `DeviceMeanAccum`s, each a `[2]` device buffer holding
    `[sum_of_batch_means, count]`, folded by a one-block reduction kernel with
    NO D2H — so a step that accumulates into these is capture-safe and the host
    never waits on it. CPU: plain host scalars, because the numbers are already
    on the host there and a device accumulator would buy nothing.

    Drained by `ACTTrainer.train_metrics` / `val_metrics` at logging cadence:
    ONE D2H per accumulator per flush instead of four per step.
    """

    var loss: DeviceMeanAccum
    var l1: DeviceMeanAccum
    var kl: DeviceMeanAccum
    var gn: DeviceMeanAccum
    # CPU mirror. Sums of per-batch means; divided by `n` on read.
    var _h_loss: Float64
    var _h_l1: Float64
    var _h_kl: Float64
    var _h_gn: Float64
    var n: Int
    """Steps folded into this window, counted on the HOST.

    Free — the host calls the step once per step either way — and it answers
    "was this window empty" without a D2H, which the device count cannot."""

    def __init__(out self):
        self.loss = DeviceMeanAccum()
        self.l1 = DeviceMeanAccum()
        self.kl = DeviceMeanAccum()
        self.gn = DeviceMeanAccum()
        self._h_loss = 0.0
        self._h_l1 = 0.0
        self._h_kl = 0.0
        self._h_gn = 0.0
        self.n = 0

    def __init__(out self, *, deinit move: Self):
        self.loss = move.loss^
        self.l1 = move.l1^
        self.kl = move.kl^
        self.gn = move.gn^
        self._h_loss = move._h_loss
        self._h_l1 = move._h_l1
        self._h_kl = move._h_kl
        self._h_gn = move._h_gn
        self.n = move.n

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var a = Self()
        comptime if target != "cpu":
            a.loss = DeviceMeanAccum.make["gpu"](ctx=ctx)
            a.l1 = DeviceMeanAccum.make["gpu"](ctx=ctx)
            a.kl = DeviceMeanAccum.make["gpu"](ctx=ctx)
            a.gn = DeviceMeanAccum.make["gpu"](ctx=ctx)
        return a^

    def read[target: StaticString](mut self) raises -> ACTWindowMetrics:
        """Window means. GPU: four `[2]`-buffer D2Hs behind their own syncs —
        flush cadence only."""
        if self.n == 0:
            return ACTWindowMetrics(0.0, 0.0, 0.0, 0.0, 0)
        comptime if target == "cpu":
            var d = Float64(self.n)
            return ACTWindowMetrics(
                self._h_loss / d, self._h_l1 / d, self._h_kl / d,
                self._h_gn / d, self.n,
            )
        else:
            return ACTWindowMetrics(
                Float64(self.loss.read["gpu"]()),
                Float64(self.l1.read["gpu"]()),
                Float64(self.kl.read["gpu"]()),
                Float64(self.gn.read["gpu"]()),
                self.n,
            )

    def reset[target: StaticString](mut self) raises:
        comptime if target == "cpu":
            self._h_loss = 0.0
            self._h_l1 = 0.0
            self._h_kl = 0.0
            self._h_gn = 0.0
        else:
            self.loss.reset["gpu"]()
            self.l1.reset["gpu"]()
            self.kl.reset["gpu"]()
            self.gn.reset["gpu"]()
        self.n = 0


struct ACTTrainer[
    QPOS: Int,
    ADIM: Int,
    N_CAM: Int,
    IMG_H: Int,
    IMG_W: Int,
    K: Int,
    DIM: Int,
    HEADS: Int,
    FF: Int,
    LATENT: Int,
    N_ENC: Int,
    N_DEC: Int,
    BATCH: Int,
    P: Float64 = ACT_DROPOUT,
    target: StaticString = "cpu",
    # See `loss_graph.mojo` — swappable so a GPU-vs-CPU gate need not
    # instantiate ResNet18's 40 layers twice. Default unchanged.
    FEAT_CH: Int = RESNET18_OUT_CH,
    OH: Int = ResNet18OutH[IMG_H],
    OW: Int = ResNet18OutW[IMG_W],
    BACKBONE: Module = ResNet18Backbone[3, IMG_H, IMG_W],
](Movable & Deinitable):
    comptime LG = ACTLossGraph[
        Self.QPOS,
        Self.ADIM,
        Self.N_CAM,
        Self.IMG_H,
        Self.IMG_W,
        Self.K,
        Self.DIM,
        Self.HEADS,
        Self.FF,
        Self.LATENT,
        Self.N_ENC,
        Self.N_DEC,
        Self.P,
        Self.FEAT_CH,
        Self.OH,
        Self.OW,
        Self.N_CAM * Self.OH * Self.OW,
        2 + Self.N_CAM * Self.OH * Self.OW,
        Self.K + 2,
        Self.BACKBONE,
    ]
    comptime ENC_SEQ: Int = Self.K + 2
    comptime IMG_ELEMS: Int = Self.N_CAM * 3 * Self.IMG_H * Self.IMG_W

    var graph: Self.LG
    var opt: Adam
    var max_grad_norm: Scalar[DT]
    var loss_out: Tensor
    var grad_seed: Tensor
    # Reusable input staging — `set_input` copies into the graph's own pool, so
    # these exist only to bridge the sampler's `List` buffers into a `Tensor`.
    var t_qpos: Tensor
    var t_images: Tensor
    var t_actions: Tensor
    var t_valid: Tensor
    var ctx: Optional[DeviceContext]
    var _adopted: Bool
    """Has `opt.adopt` packed the graph into the grouped arena yet? (GPU only.)

    Lazy on purpose — see "The arena, and when it is engaged" in the header:
    adoption rebinds every parameter's device buffer, so it must happen after
    anything that loads or injects weights by recreating one."""
    var _train_acc: ACTMetricAccum
    """Device-resident logging window for the training steps.

    Fed by `train_step_device_accum`, drained by `train_metrics`. Exists so the
    per-step path contains no D2H: `_read_terms` downloads three `[BATCH]`
    vectors behind a synchronization and `read_clip_norm` a fourth scalar
    behind another, EVERY step, for four numbers the logger averages over
    `LOG_EVERY` of them."""
    var _val_acc: ACTMetricAccum
    """The same, for validation passes — a separate window because folding
    validation batches into the training means would quietly bias the training
    curve toward the eval-mode model."""
    var deterministic_latent: Bool
    """Pin the CVAE draw to its mean in TRAINING mode too.

    ⚠ Exists because `train_mode` sets three coupled flags at once, one of
    which is `z.deterministic`. Without a sticky preference, an explicit
    `set_node_attr["z", "deterministic"](1.0)` is silently undone by the next
    `eval_step`/`predict` (both restore training mode on exit) — the caller's
    override looks applied and is gone one call later. Off by default: sampling
    the latent IS the CVAE."""

    def __init__(out self):
        self.graph = Self.LG()
        self.opt = Adam()
        self.max_grad_norm = Scalar[DT](0.0)
        self.loss_out = Tensor()
        self.grad_seed = Tensor()
        self.t_qpos = Tensor()
        self.t_images = Tensor()
        self.t_actions = Tensor()
        self.t_valid = Tensor()
        self.ctx = None
        self._adopted = False
        self._train_acc = ACTMetricAccum()
        self._val_acc = ACTMetricAccum()
        self.deterministic_latent = False

    def __init__(out self, *, deinit move: Self):
        self.graph = move.graph^
        self.opt = move.opt^
        self.max_grad_norm = move.max_grad_norm
        self.loss_out = move.loss_out^
        self.grad_seed = move.grad_seed^
        self.t_qpos = move.t_qpos^
        self.t_images = move.t_images^
        self.t_actions = move.t_actions^
        self.t_valid = move.t_valid^
        self.ctx = move.ctx^
        self._adopted = move._adopted
        self._train_acc = move._train_acc^
        self._val_acc = move._val_acc^
        self.deterministic_latent = move.deterministic_latent

    @staticmethod
    def make(
        lr: Scalar[DT] = Scalar[DT](ACT_LR),
        kl_weight: Scalar[DT] = Scalar[DT](ACT_KL_WEIGHT),
        weight_decay: Scalar[DT] = Scalar[DT](ACT_WEIGHT_DECAY),
        max_grad_norm: Scalar[DT] = Scalar[DT](ACT_CLIP_MAX_NORM),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime if Self.target != "cpu":
            if not ctx:
                raise Error(
                    "ACTTrainer.make[target='gpu']: a DeviceContext is required"
                )
        var t = Self()
        t.ctx = ctx
        t.graph = Self.LG.make[Self.target, Kaiming](ctx)
        t.graph.set_node_attr["kls", "multiplier"](kl_weight)
        t.opt = Adam(lr=lr, wd=weight_decay)
        t.max_grad_norm = max_grad_norm
        # ⚠ The graph-IO staging tensors keep a HOST copy on both targets: the
        # sampler produces host `List`s, so a GPU step is host-fill then upload.
        # `loss_out` needs both — the graph writes it on device and the step
        # reads batch means from the host side.
        comptime if Self.target == "cpu":
            t.loss_out = Tensor.alloc(Self.BATCH)
            t.grad_seed = Tensor.alloc(Self.BATCH)
        else:
            var c = ctx.value()
            t.loss_out = Tensor.alloc(Self.BATCH)
            t.loss_out.ensure_gpu(c, Self.BATCH)
            t.grad_seed = Tensor.alloc(Self.BATCH)
            t.grad_seed.ensure_gpu(c, Self.BATCH)
        # The backward seed for a mean-over-batch loss is the constant 1/BATCH;
        # nothing in a step mutates it, so it is written once.
        seed_grad_inv_batch[Self.target, Self.BATCH](
            t.grad_seed.lt[Self.target, Layout.row_major(Self.BATCH, 1)](),
            ctx=ctx,
        )
        t.t_qpos = Tensor.alloc(Self.BATCH * Self.QPOS)
        t.t_images = Tensor.alloc(Self.BATCH * Self.IMG_ELEMS)
        t.t_actions = Tensor.alloc(Self.BATCH * Self.K * Self.ADIM)
        t.t_valid = Tensor.alloc(Self.BATCH * Self.ENC_SEQ)
        comptime if Self.target != "cpu":
            var c2 = ctx.value()
            t.t_qpos.ensure_gpu(c2, Self.BATCH * Self.QPOS)
            t.t_images.ensure_gpu(c2, Self.BATCH * Self.IMG_ELEMS)
            t.t_actions.ensure_gpu(c2, Self.BATCH * Self.K * Self.ADIM)
            t.t_valid.ensure_gpu(c2, Self.BATCH * Self.ENC_SEQ)
        t._train_acc = ACTMetricAccum.make[Self.target](ctx)
        t._val_acc = ACTMetricAccum.make[Self.target](ctx)
        t.train_mode(True)
        return t^

    def train_mode(mut self, training: Bool):
        """Flip dropout/BN, latent sampling and latent scaling TOGETHER.

        Splitting these across call sites is how a validation number ends up
        measured on a model that is still sampling its latent and still
        dropping units — a number that is wrong in a direction that looks like
        underfitting.
        """
        self.graph.set_attr["training"](
            Scalar[DT](1.0) if training else Scalar[DT](0.0)
        )
        self.graph.set_node_attr["zs", "multiplier"](
            Scalar[DT](1.0) if training else Scalar[DT](0.0)
        )
        # Eval always pins the draw; training pins it only if the caller asked.
        self.graph.set_node_attr["z", "deterministic"](
            Scalar[DT](0.0) if (
                training and not self.deterministic_latent
            ) else Scalar[DT](1.0)
        )

    def freeze_backbone_norm(mut self, v: Bool = True):
        """`FrozenBatchNorm2d` for the vision backbone — what BOTH ACT
        implementations build theirs with.

        Statistics AND affine become constants, so the ImageNet values survive
        training instead of being EMA'd away in the first few hundred steps
        (momentum 0.1 leaves 2.7e-05 of the original after 100). See
        `BatchNorm2D.frozen`.

        ⚠ ONLY MEANINGFUL WITH PRETRAINED STATISTICS. Frozen at the init values
        — mean 0, var 1, gamma 1, beta 0 — BatchNorm is the identity, so
        freezing a RANDOM backbone does not "hold" a normalization, it deletes
        one. `load_backbone` therefore turns this on itself, and this method is
        for the ablation.

        Broadcast to the whole graph, which is safe because the backbone is the
        only thing in it carrying BatchNorm: the transformer stacks use
        LayerNorm and every other module's `set_attr` ignores an unknown name.
        `frozen` overrides `training`, so a later `train_mode(True)` cannot
        undo this.
        """
        self.graph.set_attr["frozen"](
            Scalar[DT](1.0) if v else Scalar[DT](0.0)
        )

    def set_deterministic_latent(mut self, v: Bool):
        """Pin (or unpin) the CVAE draw in training mode, stickily.

        Use this rather than reaching for `set_node_attr["z", "deterministic"]`
        directly: the node attribute is rewritten on every mode switch, so a
        direct write survives only until the next `eval_step` or `predict`.
        """
        self.deterministic_latent = v
        self.graph.set_node_attr["z", "deterministic"](
            Scalar[DT](1.0) if v else Scalar[DT](0.0)
        )

    def _seed_inputs(
        mut self,
        ref qpos: List[Scalar[DT]],
        ref images: List[Scalar[DT]],
        ref actions: List[Scalar[DT]],
        ref valid: List[Scalar[DT]],
    ) raises:
        """Copy one batch in. `valid` is the sampler's width-K mask; the graph
        wants width K+2, whose two leading entries ([CLS] and qpos) are never
        padding (`detr_vae.py:96`)."""
        for i in range(Self.BATCH * Self.QPOS):
            self.t_qpos.data[i] = qpos[i]
        for i in range(Self.BATCH * Self.IMG_ELEMS):
            self.t_images.data[i] = images[i]
        for i in range(Self.BATCH * Self.K * Self.ADIM):
            self.t_actions.data[i] = actions[i]
        for b in range(Self.BATCH):
            var base = b * Self.ENC_SEQ
            self.t_valid.data[base] = Scalar[DT](1.0)
            self.t_valid.data[base + 1] = Scalar[DT](1.0)
            for t in range(Self.K):
                self.t_valid.data[base + 2 + t] = valid[b * Self.K + t]

        comptime if Self.target != "cpu":
            # ⚠ `upload_resident`, NOT `upload`. `upload` recreates the device
            # buffer on EVERY call and synchronizes TWICE; these four tensors
            # are fixed-size (BATCH x the model dims) and allocated once at
            # `make`, so every one of those reallocations was pure churn.
            #
            # nsys on a 5090, 60 steps: 8,783 device alloc/free pairs, mean
            # `cuMemFree` 310 us (a plain free is 10-20 us — this one drains
            # outstanding work), totalling **3.45 s against 3.43 s of kernel
            # time**. The whole step spent as long managing memory as
            # computing. These four are 4 of the ~70 pairs per pass and 4 of
            # the ~148 synchronizations; the rest is below.
            var c = self.ctx.value()
            self.t_qpos.upload_resident(c)
            self.t_images.upload_resident(c)
            self.t_actions.upload_resident(c)
            self.t_valid.upload_resident(c)
        self.graph.set_input["qpos", Self.BATCH](self.t_qpos, self.ctx)
        self.graph.set_input["images", Self.BATCH](self.t_images, self.ctx)
        self.graph.set_input["actions", Self.BATCH](self.t_actions, self.ctx)
        self.graph.set_input["enc_valid", Self.BATCH](self.t_valid, self.ctx)

    def seed_inputs_device(
        mut self,
        mut ds: ACTDeviceDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
        val: Bool = False,
    ) raises:
        """Draw + gather a batch entirely on the device, then seed the graph.

        The device replacement for `sample_batch` + `_seed_inputs`. Nothing
        crosses the bus: the sampler writes the four staging tensors' DEVICE
        buffers, and `ComputeGraph.set_input` is a device-to-device
        `enqueue_copy` on GPU, so the whole path is kernels.

        What it removes per call, measured on the host path it replaces:
        16.1 ms of `sample_batch` with the GPU idle, two 29.5 MB
        element-by-element fills into pinned memory, the 29.5 MB H2D, and the
        four `upload_resident` device synchronizations.

        ⚠ It does NOT reproduce `sample_batch`'s batches. The device draws with
        Philox and the host with a xorshift, so the two samplers walk different
        streams and no seed reconciles them; `tests/.../test_act_dataset_gpu`
        gates the part that must agree (the gather, given the same rows).
        Anything that needs a SPECIFIC batch — a reference comparison, a
        reproducible eval — must keep using the host path.
        """
        comptime assert Self.target != "cpu", (
            "seed_inputs_device is GPU-only"
        )
        var c = self.ctx.value()
        ds.sample[Self.BATCH, Self.K](
            val,
            self.t_qpos,
            self.t_images,
            self.t_actions,
            self.t_valid,
            c,
        )
        self.graph.set_input["qpos", Self.BATCH](self.t_qpos, self.ctx)
        self.graph.set_input["images", Self.BATCH](self.t_images, self.ctx)
        self.graph.set_input["actions", Self.BATCH](self.t_actions, self.ctx)
        self.graph.set_input["enc_valid", Self.BATCH](self.t_valid, self.ctx)

    def train_step_device(
        mut self,
        mut ds: ACTDeviceDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
    ) raises -> ACTStepResult:
        """`train_step` with the data path on the device.

        Identical to `train_step` from the forward onward — same graph, same
        clip, same optimizer — so the only difference is where the batch came
        from."""
        self._ensure_adopted()
        self.opt.zero_grad[Self.target](self.graph, self.ctx)
        self.seed_inputs_device(ds, False)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        var terms = self._read_terms()
        self.graph.vjp[Self.BATCH, Self.target](self.grad_seed, self.ctx)
        self.opt.clip_grads_device[Self.target](
            self.graph, self.max_grad_norm, self.ctx
        )
        var gn = Float64(self.opt.read_clip_norm(self.ctx.value()))
        self.opt.step[Self.target](self.graph, self.ctx)
        return ACTStepResult(terms.loss, terms.l1, terms.kl, gn)

    def eval_step_device(
        mut self,
        mut ds: ACTDeviceDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
        val: Bool = True,
    ) raises -> ACTStepResult:
        """Forward-only on a FRESH batch drawn on the device.

        The validation counterpart of `train_step_device`. Distinct from
        `eval_step_resident`, which re-scores whatever is already in the input
        slots: a validation pass needs its own batches, so this seeds.

        ⚠ The caller must pin `ds`'s RNG offset around the pass
        (`set_offset`), exactly as the host path pins `ds.rng`. Validation that
        scores different batches every time makes `best_val` the minimum of a
        noisy estimate — it selects the luckiest draw, not the best model."""
        self.train_mode(False)
        self.seed_inputs_device(ds, val)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        var terms = self._read_terms()
        self.train_mode(True)
        return terms

    def eval_step_resident(mut self) raises -> ACTStepResult:
        """Forward-only on the batch ALREADY in the graph's input slots.

        `eval_step` re-seeds from host lists, which under the device data path
        would reintroduce the four `upload_resident` uploads and their syncs
        just to measure a forward. Nothing between `train_step_device` and here
        touches the input slots — `forward` and `vjp` read them, the optimizer
        does not — so the batch is still there and re-seeding is pure cost.

        ⚠ Only valid straight after a `*_device` step. Called on its own it
        measures a forward over whatever happened to be in the slots, which is
        a stale batch, not an error you would notice."""
        self.train_mode(False)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        var terms = self._read_terms()
        self.train_mode(True)
        return terms

    def _read_terms(mut self) raises -> ACTStepResult:
        """Batch means of (loss, l1, kl), read off the graph's own nodes."""
        var lo = Float64(0.0)
        var l1 = Float64(0.0)
        var kl = Float64(0.0)
        ref l1n = self.graph.node_output["l1"]()
        ref kln = self.graph.node_output["kl"]()
        comptime if Self.target != "cpu":
            # ⚠ ONE sync for three reads. `Tensor.download` synchronizes on
            # every call, so the obvious spelling (an explicit `synchronize`
            # then three `download`s) costs FOUR full device drains per call
            # and this runs twice per training iteration — 8 of the ~28
            # synchronizations a step was paying. `download_enqueue` /
            # `download_finalize` exist for exactly this; the enqueues are
            # ordered behind the forward on the same stream, so the leading
            # `synchronize()` was redundant too.
            var c = self.ctx.value()
            self.loss_out.download_enqueue(c)
            l1n.download_enqueue(c)
            kln.download_enqueue(c)
            c.synchronize()
            self.loss_out.download_finalize()
            l1n.download_finalize()
            kln.download_finalize()
        for b in range(Self.BATCH):
            lo += Float64(self.loss_out.data[b])
            l1 += Float64(l1n.data[b])
            kl += Float64(kln.data[b])
        var n = Float64(Self.BATCH)
        return ACTStepResult(lo / n, l1 / n, kl / n, 0.0)

    # ── device-resident logging (no per-step D2H) ────────────────────────
    #
    # `_read_terms` + `read_clip_norm` are what stand between an ACT step and
    # a CUDA-graph capture, and they are ours: four downloads behind two full
    # device synchronizations, every step, for four scalars the logger averages
    # over `LOG_EVERY` steps. The pattern below is SAC's
    # (`sac/trainer.mojo::train_device_kernels`): reduce each `[BATCH]` vector
    # on device into a `[2]` `[sum_of_means, count]` accumulator, and drain the
    # accumulators at flush cadence.
    #
    # ⚠ The two families do NOT mix. `train_step`/`train_step_device` return a
    # per-step `ACTStepResult` and touch no accumulator; `*_accum` folds and
    # returns nothing. A loop that called both and then read `train_metrics`
    # would report a window over only half its steps — so pick one per call
    # site, which is what the examples do under their `GPU_DATA` switch.

    def _accum_terms[VAL: Bool](mut self) raises:
        """Fold this forward's (loss, l1, kl) batch means into a window.

        GPU: three one-block reductions, no bus traffic, capture-safe. CPU: the
        same means computed on the host, where they already live."""
        comptime lb = Layout.row_major(Self.BATCH)
        ref l1n = self.graph.node_output["l1"]()
        ref kln = self.graph.node_output["kl"]()
        comptime if Self.target == "cpu":
            var lo = Float64(0.0)
            var a1 = Float64(0.0)
            var ak = Float64(0.0)
            for b in range(Self.BATCH):
                lo += Float64(self.loss_out.data[b])
                a1 += Float64(l1n.data[b])
                ak += Float64(kln.data[b])
            var d = Float64(Self.BATCH)
            comptime if VAL:
                self._val_acc._h_loss += lo / d
                self._val_acc._h_l1 += a1 / d
                self._val_acc._h_kl += ak / d
                self._val_acc.n += 1
            else:
                self._train_acc._h_loss += lo / d
                self._train_acc._h_l1 += a1 / d
                self._train_acc._h_kl += ak / d
                self._train_acc.n += 1
        else:
            var v1 = l1n.lt["gpu", lb]()
            var vk = kln.lt["gpu", lb]()
            var vl = self.loss_out.lt["gpu", lb]()
            comptime if VAL:
                self._val_acc.loss.accumulate_gpu_lt[Self.BATCH](vl)
                self._val_acc.l1.accumulate_gpu_lt[Self.BATCH](v1)
                self._val_acc.kl.accumulate_gpu_lt[Self.BATCH](vk)
                self._val_acc.n += 1
            else:
                self._train_acc.loss.accumulate_gpu_lt[Self.BATCH](vl)
                self._train_acc.l1.accumulate_gpu_lt[Self.BATCH](v1)
                self._train_acc.kl.accumulate_gpu_lt[Self.BATCH](vk)
                self._train_acc.n += 1

    def _accum_grad_norm(mut self) raises:
        """Fold the pre-clip grad norm `clip_grads_device` just wrote into the
        TRAINING window, off its device buffer.

        ⚠ Silently a no-op when the optimizer has no device norm to offer,
        which is exactly the case where `clip_grads_device` fell back to the
        host path — there the norm was computed on the host and this window is
        not the thing reporting it. `train_step_device_accum` requires the
        arena, so on its path the buffer is always there."""
        comptime if Self.target != "cpu":
            if self.opt.has_clip_norm_dev():
                var v = self.opt.clip_norm_dev()
                self._train_acc.gn.accumulate_gpu_lt[1](v)

    def train_step_device_accum(
        mut self,
        mut ds: ACTDeviceDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
    ) raises:
        """`train_step_device` with NO host readback.

        Identical arithmetic — same draw, same graph, same clip, same optimizer
        — but the four logged scalars are reduced into device accumulators
        instead of downloaded, so the per-step sequence is pure kernels: no
        D2H, no synchronization, no allocation. That is the shape a CUDA graph
        can capture, and it is worth having even while capture is blocked
        (`docs/ACT_GPU_DATA_PATH.md`), because the host stops draining the
        pipeline twice per step.

        ⚠ Returns nothing, deliberately. A per-step value is not available
        without the synchronization this exists to remove, and returning a
        zero or a stale one would read as a training regression rather than as
        a missing number. Read the window with `train_metrics`."""
        self._ensure_adopted()
        self.opt.zero_grad[Self.target](self.graph, self.ctx)
        self.seed_inputs_device(ds, False)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        self._accum_terms[False]()
        self.graph.vjp[Self.BATCH, Self.target](self.grad_seed, self.ctx)
        self.opt.clip_grads_device[Self.target](
            self.graph, self.max_grad_norm, self.ctx
        )
        self._accum_grad_norm()
        self.opt.step[Self.target](self.graph, self.ctx)

    def eval_step_device_accum(
        mut self,
        mut ds: ACTDeviceDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
        val: Bool = True,
    ) raises:
        """`eval_step_device` folding into the VALIDATION window instead of
        downloading. Read it with `val_metrics` after the pass.

        ⚠ Same pinning requirement as `eval_step_device`: pin `ds`'s RNG offset
        around the whole pass (`set_offset`), or every validation scores
        different batches and `best_val` selects the luckiest draw."""
        self.train_mode(False)
        self.seed_inputs_device(ds, val)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        self._accum_terms[True]()
        self.train_mode(True)

    def eval_step_resident_accum(mut self) raises:
        """`eval_step_resident` folding into the validation window.

        ⚠ Only valid straight after a `*_device*` step — see
        `eval_step_resident`. Re-scores the batch already in the input slots."""
        self.train_mode(False)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        self._accum_terms[True]()
        self.train_mode(True)

    def train_metrics(
        mut self, reset: Bool = True
    ) raises -> ACTWindowMetrics:
        """Drain the training window: means since the last reset, and the step
        count. FOUR D2Hs — call at logging cadence, never per step.

        `reset=False` peeks without clearing, for a progress line that must not
        disturb the logger's window."""
        var w = self._train_acc.read[Self.target]()
        if reset:
            self._train_acc.reset[Self.target]()
        return w

    def val_metrics(mut self, reset: Bool = True) raises -> ACTWindowMetrics:
        """Drain the validation window — the mean over the pass's batches.

        ⚠ `grad_norm` is always 0 here: a forward-only pass computes no
        gradients. It is reported rather than omitted so the two windows share
        a type."""
        var w = self._val_acc.read[Self.target]()
        if reset:
            self._val_acc.reset[Self.target]()
        return w

    def _ensure_adopted(mut self) raises:
        """Pack the graph into the optimizer's grouped arena, once, on GPU.

        Deliberately NOT done in `make`: see the header. Everything that seeds
        weights — `load`, `load_reference_params`, a test injecting a CPU
        model's parameters — runs between `make` and the first `train_step`,
        and adoption after that point is only safe because those paths upload
        into the RESIDENT buffer."""
        comptime if Self.target != "cpu":
            if not self._adopted:
                self.opt.adopt[Self.target](self.graph, self.ctx)
                self._adopted = True

    def train_step(
        mut self,
        ref qpos: List[Scalar[DT]],
        ref images: List[Scalar[DT]],
        ref actions: List[Scalar[DT]],
        ref valid: List[Scalar[DT]],
    ) raises -> ACTStepResult:
        self._ensure_adopted()
        # GPU+adopted: ONE fill over the contiguous grad arena. CPU: the
        # per-param walk, unchanged.
        self.opt.zero_grad[Self.target](self.graph, self.ctx)
        self._seed_inputs(qpos, images, actions, valid)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        var terms = self._read_terms()
        self.graph.vjp[Self.BATCH, Self.target](self.grad_seed, self.ctx)

        var gn: Float64
        comptime if Self.target == "cpu":
            var ss = _SumSq()
            self.graph.for_each_param[Self.target](ss, self.ctx, String(""))
            gn = ss.sum_sq ** 0.5
            if self.max_grad_norm > Scalar[DT](0.0) and gn > Float64(
                self.max_grad_norm
            ):
                # A non-finite norm scales to zero rather than propagating NaN
                # into every weight — mirrors
                # `lewm/trainer.mojo::_scale_from_norm`.
                var sc = Scalar[DT](0.0)
                if gn == gn:
                    sc = self.max_grad_norm / Scalar[DT](gn)
                var scaler = _ScaleGrads(sc)
                self.graph.for_each_param[Self.target](
                    scaler, self.ctx, String("")
                )
        else:
            # Same clip, on device: sum-of-squares over the grad arena →
            # `scale = min(1, max_norm/‖g‖)` (non-finite → 0, `max_norm <= 0` →
            # 1) → one scaling pass. Persistent scratch, so no allocation. The
            # ONE D2H is the pre-clip norm, which `ACTStepResult` reports; the
            # host walk it replaces synchronized once per PARAMETER.
            self.opt.clip_grads_device[Self.target](
                self.graph, self.max_grad_norm, self.ctx
            )
            gn = Float64(self.opt.read_clip_norm(self.ctx.value()))

        # GPU+adopted: one grouped kernel over the whole arena. CPU: the
        # per-param `ParamVisitor` walk `begin_step` + `for_each_param` did.
        self.opt.step[Self.target](self.graph, self.ctx)
        return ACTStepResult(terms.loss, terms.l1, terms.kl, gn)

    def eval_step(
        mut self,
        ref qpos: List[Scalar[DT]],
        ref images: List[Scalar[DT]],
        ref actions: List[Scalar[DT]],
        ref valid: List[Scalar[DT]],
    ) raises -> ACTStepResult:
        """Forward only, in eval mode (z = 0, no dropout, BN running stats).

        Restores training mode on the way out, so a validation call inside a
        training loop cannot silently leave the model in eval.
        """
        self.train_mode(False)
        self._seed_inputs(qpos, images, actions, valid)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        var terms = self._read_terms()
        self.train_mode(True)
        return terms

    def predict(
        mut self,
        ref qpos: List[Scalar[DT]],
        ref images: List[Scalar[DT]],
        ref actions: List[Scalar[DT]],
        ref valid: List[Scalar[DT]],
        mut out_actions: List[Scalar[DT]],
    ) raises:
        """The inference path: run in eval mode and read `a_hat`.

        `actions` is still required — the CVAE encoder runs and its output is
        scaled to zero rather than the encoder being skipped (see
        `loss_graph.mojo`). Zeros are a fine argument; nothing downstream of the
        zeroed latent reads them.
        """
        self.train_mode(False)
        self._seed_inputs(qpos, images, actions, valid)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        comptime N = Self.BATCH * Self.K * Self.ADIM
        if len(out_actions) != N:
            out_actions = List[Scalar[DT]](unsafe_uninit_length=N)
        ref ahat = self.graph.node_output["ahat"]()
        comptime if Self.target != "cpu":
            var c = self.ctx.value()
            c.synchronize()
            ahat.download(c)
        for i in range(N):
            out_actions[i] = ahat.data[i]
        self.train_mode(True)

    # ── checkpoints ──────────────────────────────────────────────────────

    def save(mut self, path: String, save_moments: Bool = True) raises:
        """v3 binary named checkpoint: Param sections (+ Adam moments) then
        State sections (BatchNorm running statistics).

        ⚠ The State pass is not optional. Without the running statistics a
        reloaded model runs BatchNorm on whatever its init held (mean 0,
        var 1), and every prediction is wrong in a way that looks like a
        training failure rather than a load failure.
        """
        var w = BinaryCheckpointWriter(save_moments)
        w.mode = 0
        self.graph.for_each_param[Self.target, BinaryCheckpointWriter](w, self.ctx)
        w.mode = 1
        self.graph.for_each_state[Self.target, BinaryCheckpointWriter](w, self.ctx)
        _write_file_bytes(path, w.content)

    def load_backbone(
        mut self, dump_dir: String, freeze_norm: Bool = True
    ) raises -> Int:
        """Fill the vision backbone from a `dump_resnet18_imagenet.py` dump.

        Returns the number of tensors filled. Raises if the dump names a tensor
        the backbone does not have, or sizes it differently — a pretrained
        loader that silently leaves half the network random is worse than none,
        because the run then reports "pretraining did not help".

        ⚠ `feat.0.` is the backbone's path inside the ACT graph:
        `Tokenwise[N_CAM, BACKBONE]` contributes the `.0`. The dump names the
        same tensors `rn18in.*`, backbone-local, which is the mapping
        `dump_act_reference.py:emit_resnet18` writes for the standalone gate —
        so both use one mapping and the gate covers this path too.

        ⚠ TWO WALKS. Weights are parameters; BatchNorm running statistics are
        STATE. Pretrained convolutions carrying init statistics (mean 0, var 1)
        are not the pretrained network, so skipping the second walk would load
        45 MB of weights and still change the function.
        """
        comptime GP = "feat.0."
        comptime DP = "rn18in."
        var wl = LoadPrefixedParams[GP, DP](RefDump(String(dump_dir)))
        self.graph.for_each_param[Self.target, LoadPrefixedParams[GP, DP]](
            wl, self.ctx
        )
        if len(wl.missing) > 0:
            raise Error(
                "load_backbone: " + String(len(wl.missing))
                + " backbone weights absent from the dump, first '"
                + wl.missing[0] + "' — the dump was written for a different"
                " ResNet variant or a different resolution"
            )
        var sl = LoadPrefixedParams[GP, DP](RefDump(String(dump_dir)))
        self.graph.for_each_state[Self.target, LoadPrefixedParams[GP, DP]](
            sl, self.ctx
        )
        if len(sl.missing) > 0:
            raise Error(
                "load_backbone: " + String(len(sl.missing))
                + " BatchNorm running statistics absent, first '"
                + sl.missing[0] + "'"
            )
        if len(wl.loaded) == 0:
            raise Error(
                "load_backbone: matched NOTHING under '" + String(GP)
                + "' — the graph's backbone path changed and this loader did"
                " not, so the weights would have been silently discarded"
            )
        # ⚠ FREEZING IS PART OF LOADING, not a separate decision, because the
        # references never do one without the other: `norm_layer=
        # FrozenBatchNorm2d` is passed at the same `resnet18(pretrained=...)`
        # call. Loading ImageNet statistics and then letting training-mode
        # BatchNorm EMA them away is the worst of both — the cost of the load
        # with none of the benefit, and it reads as "pretraining did not help".
        # `freeze_norm=False` is for the ablation that measures exactly that.
        if freeze_norm:
            self.freeze_backbone_norm(True)
        return len(wl.loaded) + len(sl.loaded)

    def load(mut self, path: String) raises:
        var bytes = _read_file_bytes(path)
        if not _is_v3_header(bytes):
            raise Error(
                "ACTTrainer.load: '" + path + "' is not a v3 binary checkpoint"
            )
        var r = BinaryCheckpointReader(bytes^)
        r.mode = 0
        self.graph.for_each_param[Self.target, BinaryCheckpointReader](r, self.ctx)
        r.mode = 1
        self.graph.for_each_state[Self.target, BinaryCheckpointReader](r, self.ctx)
