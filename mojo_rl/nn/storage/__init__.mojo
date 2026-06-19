"""nn.storage — the storage-passing neural-network surface.

The successor to the legacy `nn` `Module`/`TensorPack[MutAnyOrigin]` surface
(see docs/STORAGE_NN_MIGRATION_PLAN.md). Leaves take owned `Tensor` storages
by `ref`/`mut` and build their typed views INTERNALLY, so the wildcard origin
is gone from leaf bodies — the only residual erasure is the GPU kernel-arg ABI
(`MutAnyOrigin`) and the one load-bearing `TensorPack.__getitem__` pin.

Subpackage layout mirrors legacy `nn/` (core / primitives / combinators /
optimizer / loss) so the final migration is a directory move. Types carry their
definitive names (no `S` suffix); the legacy surface is aliased in side-by-side
parity tests (`Conv2D as LegacyConv2D`).
"""

from .core.tensor import Tensor, TensorImpl
from .core.tensor_refs import TensorRefs
from .core.tensor_pack import TensorPack
from .core.param import Param, ParamVisitor
from .core.module import Module
from .core.initializer import (
    Initializer, Kaiming, Xavier, Zero, Normal, Deterministic,
)
from .core.checkpoint import save_params, load_params
from .primitives.linear import Linear
from .primitives.linear_relu import LinearReLU
from .primitives.add import Add
from .primitives.elementwise import Elementwise
from .primitives.activations import (
    ReLU, Tanh, Sigmoid, GELU, Mish, Swish, Symlog,
)
from .primitives.conv2d import Conv2D
from .primitives.batch_norm_1d import BatchNorm1D
from .primitives.batch_norm_2d import BatchNorm2D
from .primitives.layer_norm import LayerNorm
from .primitives.rms_norm import RMSNorm
from .primitives.noisy_linear import NoisyLinear
from .primitives.flatten import Flatten
from .primitives.embedding import Embedding
from .primitives.rsample import RSample
from .primitives.binary_elementwise import (
    BinaryElementwise, BinaryElemMin, BinarySub,
)
from .primitives.concat import Concat2
from .combinators.sequential import Sequential
from .combinators.residual import Residual
from .combinators.parallel import Parallel
from .combinators.repeat import Repeat
from .combinators.projected_residual import ProjectedResidual
from .combinators.tokenwise import Tokenwise
from .combinators.skip_concat import SkipConcat
from .combinators.branch_concat import BranchConcat
from .combinators.stop_grad_params import StopGradParams
from .combinators.compute_graph import ComputeGraph
from .optimizer.sgd import SGD
from .optimizer.adam import Adam
from .loss.mse import mse_forward, mse_backward
from .loss.mse_loss import MSELoss
from .loss.sac import polyak_tensor, sac_target_y
from .loss.cross_entropy import CrossEntropyLoss
from .training.trainer import Trainer
