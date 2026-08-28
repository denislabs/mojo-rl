# +--------------------------------------------------------------------------+ #
# | ACTTrainer — offline behaviour cloning over ACTLossGraph
# +--------------------------------------------------------------------------+ #
"""Owns the loss graph + one AdamW + the scratch a step needs.

Transposed from `experimental/lewm/trainer.mojo`, which is the established
shape for "one ComputeGraph, one optimizer": a step is

    zero_grad -> set_input x4 -> forward -> seed grad 1/B -> vjp -> clip -> step

`Adam` is itself a `ParamVisitor`, so stepping a graph (which is not a `Module`,
and so cannot use `opt.adopt`/`opt.step`) is `opt.begin_step()` followed by
`graph.for_each_param(opt)`.

CPU and GPU: `target` is a comptime parameter. On GPU the sampler still produces
host `List`s, so a step is host-fill -> upload -> graph -> download the three
per-sample loss vectors. Weights, activations and gradients stay on device.

⚠ `Adam.adopt`'s grouped arena is NOT engaged, because it requires a `Module`
and a `ComputeGraph` is not one — the step walks `for_each_param` with per-param
kernels. Same limitation `lewm/trainer.mojo` records; a graph-aware arena is a
GPU-perf follow-up, not a correctness gap.

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

from .config import (
    ACT_CLIP_MAX_NORM,
    ACT_DROPOUT,
    ACT_KL_WEIGHT,
    ACT_LR,
    ACT_WEIGHT_DECAY,
)
from .loss_graph import ACTLossGraph
from .refload import LoadPrefixedParams, RefDump


# ── grad-norm clip over a graph ──────────────────────────────────────────
# The graph owns every parameter but is not a `Module`, so `Adam.clip_grads`
# (Module-constrained) cannot take it. Two visitor passes: sum of squares, then
# scale. Mirrors `lewm/trainer.mojo`'s `_SumSqV`, CPU-only.


struct _SumSq(ParamVisitor):
    """Sum of squared gradients over every parameter.

    ⚠ On GPU this DOWNLOADS each gradient slab to sum it on the host. That is a
    synchronisation point per parameter and it is the wrong shape for a hot
    loop — `lewm/trainer.mojo` has the device-reduce kernels for exactly this.
    It is correct, and it is only reached when `max_grad_norm > 0`; a training
    run that wants speed on GPU should set `max_grad_norm = 0.0`, which is also
    what the reference effectively does (it parses `clip_max_norm` and never
    applies it).
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
            # `_SumSq` already brought the host copy down this step, so the
            # scale is applied there and re-uploaded. Same caveat as `_SumSq`.
            for i in range(N):
                grad.data[i] = grad.data[i] * self.scale
            grad.upload(ctx.value())
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

    def _read_terms(mut self) raises -> ACTStepResult:
        """Batch means of (loss, l1, kl), read off the graph's own nodes."""
        var lo = Float64(0.0)
        var l1 = Float64(0.0)
        var kl = Float64(0.0)
        ref l1n = self.graph.node_output["l1"]()
        ref kln = self.graph.node_output["kl"]()
        comptime if Self.target != "cpu":
            var c = self.ctx.value()
            c.synchronize()
            self.loss_out.download(c)
            l1n.download(c)
            kln.download(c)
        for b in range(Self.BATCH):
            lo += Float64(self.loss_out.data[b])
            l1 += Float64(l1n.data[b])
            kl += Float64(kln.data[b])
        var n = Float64(Self.BATCH)
        return ACTStepResult(lo / n, l1 / n, kl / n, 0.0)

    def train_step(
        mut self,
        ref qpos: List[Scalar[DT]],
        ref images: List[Scalar[DT]],
        ref actions: List[Scalar[DT]],
        ref valid: List[Scalar[DT]],
    ) raises -> ACTStepResult:
        self.graph.zero_grad[Self.target](self.ctx)
        self._seed_inputs(qpos, images, actions, valid)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        var terms = self._read_terms()
        self.graph.vjp[Self.BATCH, Self.target](self.grad_seed, self.ctx)

        var ss = _SumSq()
        self.graph.for_each_param[Self.target](ss, self.ctx, String(""))
        var gn = ss.sum_sq ** 0.5
        if self.max_grad_norm > Scalar[DT](0.0) and gn > Float64(
            self.max_grad_norm
        ):
            # A non-finite norm scales to zero rather than propagating NaN into
            # every weight — mirrors `lewm/trainer.mojo::_scale_from_norm`.
            var sc = Scalar[DT](0.0)
            if gn == gn:
                sc = self.max_grad_norm / Scalar[DT](gn)
            var scaler = _ScaleGrads(sc)
            self.graph.for_each_param[Self.target](scaler, self.ctx, String(""))

        self.opt.begin_step()
        self.graph.for_each_param[Self.target](self.opt, self.ctx, String(""))
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

    def load_backbone(mut self, dump_dir: String) raises -> Int:
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
