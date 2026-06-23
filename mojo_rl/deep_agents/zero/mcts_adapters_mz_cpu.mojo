"""CPU MuZero adapters for the `planners.tree_search` GenericCPUMCTS surface.

The CPU counterpart of `mcts_adapters_mz.mojo` (GPU). MuZero's latent is the
**learned** hidden state and all three functions are networks — no env in the
loop:

  * `MZRepCPU`  — `Representation`: ``obs → z`` via the rep net (`forward["cpu",1]`).
  * `MZDynCPU`  — `Dynamics`: ``[z ⊕ onehot(a)] → (z', reward)`` via the dyn net.
  * `MZPredCPU` — `Prediction`: ``z → (softmax policy, value)`` via the pred net.

Reward/value decode reuses the **same** `mz_decode_value_batch` (softmax · linear
bins → ``h⁻¹``) the GPU MCTS kernel applies inline and the training targets use.
Single-player: no legal mask. Storage surface: owned `Tensor` I/O (`.data`), no
raw pointers; the non-owning net handle is the `feedback_mojo_set_external_lifetime`
contract (constructed with `.as_unsafe_any_origin()` by the driver).
"""

from std.math import exp
from std.memory import UnsafePointer

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.planners.tree_search import Representation, Dynamics, Prediction
from .twohot_targets import mz_decode_value_batch


def _mz_decode_one[BINS: Int](
    logits: List[Scalar[DT]], off: Int, v_min: Scalar[DT], v_max: Scalar[DT]
) raises -> Float64:
    """Decode one categorical head (``logits[off .. off+BINS)``) to a raw scalar
    via the shared `mz_decode_value_batch` (h-space linear bins → ``h⁻¹``)."""
    var out = List[Scalar[DT]](length=1, fill=0)
    mz_decode_value_batch[1, BINS](logits, off, v_min, v_max, out, 0)
    return Float64(out[0])


@fieldwise_init
struct MZRepCPU[OBS: Int, LATENT: Int, NET: Module](
    Movable, ImplicitlyDeletable, Representation
):
    """H: ``obs → z`` (latent min-max scaled by the net's tail)."""

    comptime OBS_DIM: Int = Self.OBS
    comptime LATENT_DIM: Int = Self.LATENT

    var net: UnsafePointer[Self.NET, MutAnyOrigin]

    def encode_cpu(
        mut self, obs: List[Float64], mut hidden_out: List[Float64]
    ) raises:
        comptime IN = Self.NET.IN_DIMS[0]
        comptime OUT = Self.NET.OUT_DIM
        var ib = Tensor.alloc(IN)
        for i in range(IN):
            ib.data[i] = Scalar[DT](obs[i]) if i < len(obs) else Scalar[DT](0.0)
        var ob = Tensor.alloc(OUT)
        self.net[].forward["cpu", 1](TensorRefs[Self.NET.ARITY](ib), ob, None)
        for i in range(Self.LATENT):
            hidden_out[i] = Float64(ob.data[i])


@fieldwise_init
struct MZDynCPU[LATENT: Int, ACT: Int, BINS: Int, NET: Module](
    Movable, ImplicitlyDeletable, Dynamics
):
    """G: ``[z ⊕ onehot(a)] → (z', reward)``; reward decoded from BINS bins."""

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT

    var net: UnsafePointer[Self.NET, MutAnyOrigin]
    var v_min: Scalar[DT]
    var v_max: Scalar[DT]

    def step_cpu(
        mut self,
        hidden_in: List[Float64],
        action: Int,
        mut hidden_out: List[Float64],
    ) raises -> Float64:
        comptime IN = Self.NET.IN_DIMS[0]     # LATENT + ACT
        comptime OUT = Self.NET.OUT_DIM       # LATENT + BINS
        var ib = Tensor.alloc(IN)
        for i in range(Self.LATENT):
            ib.data[i] = Scalar[DT](hidden_in[i])
        for a in range(Self.ACT):
            ib.data[Self.LATENT + a] = Scalar[DT](0.0)
        ib.data[Self.LATENT + action] = Scalar[DT](1.0)
        var ob = Tensor.alloc(OUT)
        self.net[].forward["cpu", 1](TensorRefs[Self.NET.ARITY](ib), ob, None)
        for i in range(Self.LATENT):
            hidden_out[i] = Float64(ob.data[i])
        return _mz_decode_one[Self.BINS](
            ob.data, Self.LATENT, self.v_min, self.v_max
        )


@fieldwise_init
struct MZPredCPU[LATENT: Int, ACT: Int, BINS: Int, NET: Module](
    Movable, ImplicitlyDeletable, Prediction
):
    """F: ``z → (softmax policy, value)``; value decoded from BINS bins.
    Single-player — all actions legal, softmax over the full policy slice."""

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT

    var net: UnsafePointer[Self.NET, MutAnyOrigin]
    var v_min: Scalar[DT]
    var v_max: Scalar[DT]

    def predict_cpu(
        mut self, hidden: List[Float64], mut policy_out: List[Float64]
    ) raises -> Float64:
        comptime IN = Self.NET.IN_DIMS[0]     # LATENT
        comptime OUT = Self.NET.OUT_DIM       # ACT + BINS
        var ib = Tensor.alloc(IN)
        for i in range(Self.LATENT):
            ib.data[i] = Scalar[DT](hidden[i])
        var ob = Tensor.alloc(OUT)
        self.net[].forward["cpu", 1](TensorRefs[Self.NET.ARITY](ib), ob, None)

        # softmax over the policy slice [0, ACT)
        var max_l = Float64(ob.data[0])
        for a in range(1, Self.ACT):
            var lv = Float64(ob.data[a])
            if lv > max_l:
                max_l = lv
        var s = 0.0
        for a in range(Self.ACT):
            policy_out[a] = exp(Float64(ob.data[a]) - max_l)
            s += policy_out[a]
        for a in range(Self.ACT):
            policy_out[a] /= s

        return _mz_decode_one[Self.BINS](
            ob.data, Self.ACT, self.v_min, self.v_max
        )
