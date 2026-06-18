"""nn.storage — the storage-passing neural-network surface.

The successor to the legacy `nn` `Module`/`TensorPack[MutAnyOrigin]` surface
(see docs/STORAGE_NN_MIGRATION_PLAN.md). Leaves take owned `Tensor` storages
by `ref`/`mut` and build their typed views INTERNALLY, so the wildcard origin
is gone from leaf bodies — the only residual erasure is the GPU kernel-arg ABI
(`MutAnyOrigin`) and the one load-bearing `TensorPack.__getitem__` pin.

Promoted from the proven POC (`experimental/nn/storage/`); the design narrative
+ every dead-end is in `docs/ORIGIN_DESIGN_REVIEW.md` §7.7–7.17.
"""

from .tensor import Tensor, TensorImpl
from .tensor_refs import TensorRefs
from .tensor_pack import TensorPack
from .param import ParamS, ParamVisitorS
from .module import ModuleS
from .leaves import LinS, ReLUS, AddS
from .elementwise import ElementwiseS
from .activations import (
    ReLUE, TanhS, SigmoidS, GELUS, MishS, SwishS, SymlogS,
)
from .conv2d import ConvS
from .batch_norm_1d import BatchNorm1DS
from .batch_norm_2d import BatchNorm2DS
from .sequential import SeqS
from .optim_loss import SGDS, mse_forward, mse_backward
