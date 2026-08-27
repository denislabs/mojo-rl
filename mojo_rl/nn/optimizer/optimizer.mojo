"""Optimizer trait — the storage-native optimizer interface.

Mirrors the legacy slim trait (hyperparams are public mut fields on the concrete
struct, poked by external schedules — NOT in the trait), adapted to the storage
optimizers' shape: constructed with hyperparams (`Adam(lr=...)`), then optionally
`adopt`-ed for the contiguous-arena GPU mode. The Trainer is generic over this
trait so optimizers are swappable.

Concrete usage (target-agnostic — `adopt` is a no-op on CPU):
    var opt = Adam(lr=3e-4)
    opt.adopt[target](model, ctx)          # GPU: pack arena;  CPU: nothing
    ...
    opt.zero_grad[target](model, ctx)
    # forward / backward ...
    opt.step[target](model, ctx)

`step` is the only required method. `adopt` defaults to a no-op (optimizers
without an arena mode ignore it); `zero_grad` / `clip_grads` default to the
per-param paths (arena optimizers override for the single-kernel / capture-safe
variants). `set_lr` / `get_lr` default to no-op / 1.0 (schedules read/write them).
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..core.module import Module
from .grad_clip import clip_grad_norm


trait Optimizer(Defaultable & Movable & Deinitable):
    def step[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Apply one optimizer update to every Param of `model`."""
        ...

    def adopt[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Engage a contiguous-arena GPU mode if the optimizer has one (NO-OP on
        CPU). Default: no-op — optimizers without an arena ignore it."""
        pass

    def zero_grad[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Default: per-param zero via the model. Arena optimizers override with a
        single grad-arena fill when adopted."""
        model.zero_grad[target](ctx)

    def clip_grads[
        target: StaticString, M: Module
    ](
        mut self, mut model: M, max_norm: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        """Default: per-param global grad-norm clip (returns the pre-clip norm).
        Arena optimizers override with the capture-safe arena reduction."""
        return clip_grad_norm[target](model, max_norm, ctx)

    def clip_grads_device[
        target: StaticString, M: Module
    ](
        mut self, mut model: M, max_norm: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """CUDA-graph-safe grad-norm clip: on-device kernels, NO allocation and
        NO D2H — drop into a captured train step. Default: falls back to
        `clip_grads` (which may D2H → only correct when NOT capturing). Arena
        optimizers override with the persistent-scratch device path."""
        _ = self.clip_grads[target, M](model, max_norm, ctx)

    def set_lr(mut self, lr: Scalar[DT]):
        """Set the learning rate (for external LR schedules). Default no-op."""
        pass

    def push_lr_device[
        target: StaticString
    ](mut self, lr: Scalar[DT], ctx: Optional[DeviceContext] = None) raises:
        """Capture-safe LR update: write the next LR where a CUDA-graph-captured
        `step` reads it (a device buffer), so an arbitrary host-computed schedule
        survives replay. Default: host `set_lr` — correct when NOT capturing (the
        un-captured kernel reads the host `lr`). Arena optimizers override to also
        push the LR onto the device."""
        self.set_lr(lr)

    def get_lr(self) -> Scalar[DT]:
        """Current learning rate. Default 1.0; optimizers with an `lr` field
        override to return it."""
        return Scalar[DT](1.0)
