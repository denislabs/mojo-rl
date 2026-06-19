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
from .core.param import Param, ParamVisitor, IsParam
from .core.state import State, IsState
from .core.module import Module
from .core.walkers import for_each_param_auto, zero_grad_auto, join_name
from .core.named_params import NamedParam, named_params, named_states
from .core.initializer import (
    Initializer, Kaiming, Xavier, Zero, Normal, Deterministic,
)
from .core.checkpoint import save_params, load_params
from .primitives.linear import Linear
from .primitives.linear_relu import LinearReLU
from .primitives.linear_act import LinearAct
from .primitives.linear_tanh import LinearTanh
from .primitives.linear_mish import LinearMish
from .primitives.linear_sigmoid import LinearSigmoid
from .primitives.linear_swish import LinearSwish
from .primitives.block_linear import BlockLinear
from .primitives.tied_linear import TiedLinear
from .primitives.gru_cell import GRUCell
from .primitives.lstm_cell import LSTMCell
from .primitives.lstm_seq import LSTMSeq
from .primitives.attention import ScaledDotProductAttention
from .primitives.masked_attention import (
    MaskedAttention, causal_mask, all_allow_mask, build_modality_mask,
)
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
from .primitives.min_max_norm import MinMaxNorm
from .primitives.sim_norm import SimNorm
from .primitives.noisy_linear import NoisyLinear
from .primitives.flatten import Flatten
from .primitives.dropout import Dropout
from .primitives.avg_pool_2d import AvgPool2D
from .primitives.max_pool_2d import MaxPool2D
from .primitives.embedding import Embedding
from .primitives.rsample import RSample
from .primitives.binary_elementwise import (
    BinaryElementwise, BinaryElemMin, BinarySub,
)
from .primitives.concat import Concat2
from .primitives.scale import Scale
from .primitives.clamp import Clamp
from .primitives.gather_cols import GatherCols
from .primitives.dueling_head import DuelingHead
from .primitives.dueling_head_c51 import DuelingHeadC51
# Phase E remainder — RL-utility + LeWM leaves
from .primitives.slice import Slice
from .primitives.reduce import Reduce, Sum, Mean, ReduceOp, SumOp, MeanOp
from .primitives.reduce_max import ReduceMax
from .primitives.gather_action_slice import GatherActionSlice
from .primitives.zero_linear import ZeroLinear
from .primitives.silu import SiLU
from .primitives.stop_grad import StopGrad
from .primitives.sigreg import SIGReg
from .primitives.layer_norm_no_affine import LayerNormNoAffine
from .primitives.mse_per_sample import MSEPerSample
from .primitives.gate import Gate
from .primitives.modulate import Modulate
# Phase E remainder — transformer / Dreamer4 plumbing leaves
from .primitives.bias_add import BiasAdd
from .primitives.transpose_2d import Transpose2D
from .primitives.token_mean import TokenMean
from .primitives.qkv_to_major import QKVToMajor
from .primitives.swiglu import SwiGLU
from .primitives.space_time_transpose import SpaceTimeTranspose
from .primitives.sinusoidal_pos import SinusoidalPosAdd
from .primitives.sinusoidal_pos_bt import SinusoidalPosAddBT
from .primitives.broadcast_tokens import BroadcastTokens
from .primitives.learned_tokens import LearnedTokens
from .primitives.learned_queries import LearnedQueries
from .primitives.mae_replacer import MAEReplacer
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
from .optimizer.optimizer import Optimizer
from .optimizer.param_arena import ParamArena, polyak_arenas
from .optimizer.sgd import SGD
from .optimizer.adam import Adam, AdamW
from .optimizer.schedules import LinearWarmupSchedule
from .optimizer.grad_clip import clip_grad_norm
from .optimizer.scalar_adam import ScalarAdam
from .loss.mse import mse_forward, mse_backward
from .loss.mse_loss import MSELoss
from .loss.sac import polyak_tensor, sac_target_y
from .loss.cross_entropy import CrossEntropyLoss
from .training.trainer import Trainer
