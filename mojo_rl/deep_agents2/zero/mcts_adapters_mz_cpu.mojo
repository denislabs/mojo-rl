"""CPU MuZero adapters for the `planners.tree_search` GenericCPUMCTS surface.

The CPU counterpart of `mcts_adapters_mz.mojo` (GPU). Where the AlphaZero CPU
adapters (`mcts_adapters_cpu.mojo`) thread the env's serialized *state* as the
latent (dynamics = true `env.step`), MuZero's latent is the **learned** hidden
state and all three functions are networks — no env in the loop:

  * `MZRepCPU`  — `Representation`: ``obs → z`` via the rep net (`forward["cpu",1]`).
    The latent is already min-max scaled by the net's `MinMaxNorm` tail.
  * `MZDynCPU`  — `Dynamics`: ``[z ⊕ onehot(a)] → (z', reward)`` via the dyn net;
    splits the output, decodes the categorical reward head to a scalar.
  * `MZPredCPU` — `Prediction`: ``z → (softmax policy, value)`` via the pred net;
    decodes the categorical value head to a scalar.

Reward/value decode reuses the **same** `mz_decode_value_batch` (softmax · linear
bins → ``h⁻¹``) the GPU MCTS kernel applies inline and the training targets use,
so CPU search, GPU search, and training all share one numeric contract. Single-
player: no legal mask (softmax over all actions). dtype bridge: planner Lists are
``Float64``; nn2 forwards are ``DT`` (float32) — convert at the boundary.
"""

from std.math import exp
from std.memory import alloc, UnsafePointer
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.planners.tree_search import Representation, Dynamics, Prediction
from .twohot_targets import mz_decode_value_batch


def _mz_decode_one[BINS: Int](
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    off: Int,
    v_min: Scalar[DT],
    v_max: Scalar[DT],
) -> Float64:
    """Decode one categorical head (``logits[off..off+BINS)``) to a raw scalar,
    via the shared `mz_decode_value_batch` (h-space linear bins → ``h⁻¹``)."""
    var buf = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](BINS)
    )
    for i in range(BINS):
        buf[i] = logits[off + i]
    var out = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](1)
    )
    mz_decode_value_batch[1, BINS](buf, v_min, v_max, out)
    var v = Float64(out[0])
    buf.free()
    out.free()
    return v


@fieldwise_init
struct MZRepCPU[OBS: Int, LATENT: Int, NET: Module](
    Movable, ImplicitlyDestructible, Representation
):
    """h: ``obs → z`` (latent min-max scaled by the net's tail)."""

    comptime OBS_DIM: Int = Self.OBS
    comptime LATENT_DIM: Int = Self.LATENT

    var net: UnsafePointer[Self.NET, MutAnyOrigin]

    def encode_cpu(
        mut self, obs: List[Float64], mut hidden_out: List[Float64]
    ) raises:
        comptime IN = Self.NET.IN_DIMS[0]
        comptime OUT = Self.NET.OUT_DIM
        var ib = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            alloc[Scalar[DT]](IN)
        )
        for i in range(IN):
            ib[i] = Scalar[DT](obs[i]) if i < len(obs) else Scalar[DT](0.0)
        var ob = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            alloc[Scalar[DT]](OUT)
        )
        var it = TileTensor(ib, row_major[1, IN]())
        var ot = TileTensor(ob, row_major[1, OUT]())
        self.net[].forward["cpu", 1](it, output=ot)
        for i in range(Self.LATENT):
            hidden_out[i] = Float64(ob[i])
        ib.free()
        ob.free()


@fieldwise_init
struct MZDynCPU[LATENT: Int, ACT: Int, BINS: Int, NET: Module](
    Movable, ImplicitlyDestructible, Dynamics
):
    """g: ``[z ⊕ onehot(a)] → (z', reward)``; reward decoded from BINS bins."""

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
        var ib = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            alloc[Scalar[DT]](IN)
        )
        for i in range(Self.LATENT):
            ib[i] = Scalar[DT](hidden_in[i])
        for a in range(Self.ACT):
            ib[Self.LATENT + a] = Scalar[DT](0.0)
        ib[Self.LATENT + action] = Scalar[DT](1.0)
        var ob = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            alloc[Scalar[DT]](OUT)
        )
        var it = TileTensor(ib, row_major[1, IN]())
        var ot = TileTensor(ob, row_major[1, OUT]())
        self.net[].forward["cpu", 1](it, output=ot)
        for i in range(Self.LATENT):
            hidden_out[i] = Float64(ob[i])
        var reward = _mz_decode_one[Self.BINS](
            ob, Self.LATENT, self.v_min, self.v_max
        )
        ib.free()
        ob.free()
        return reward


@fieldwise_init
struct MZPredCPU[LATENT: Int, ACT: Int, BINS: Int, NET: Module](
    Movable, ImplicitlyDestructible, Prediction
):
    """f: ``z → (softmax policy, value)``; value decoded from BINS bins.
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
        var ib = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            alloc[Scalar[DT]](IN)
        )
        for i in range(Self.LATENT):
            ib[i] = Scalar[DT](hidden[i])
        var ob = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            alloc[Scalar[DT]](OUT)
        )
        var it = TileTensor(ib, row_major[1, IN]())
        var ot = TileTensor(ob, row_major[1, OUT]())
        self.net[].forward["cpu", 1](it, output=ot)

        # softmax over the policy slice [0, ACT)
        var max_l = Float64(ob[0])
        for a in range(1, Self.ACT):
            var lv = Float64(ob[a])
            if lv > max_l:
                max_l = lv
        var s = 0.0
        for a in range(Self.ACT):
            policy_out[a] = exp(Float64(ob[a]) - max_l)
            s += policy_out[a]
        for a in range(Self.ACT):
            policy_out[a] /= s

        var value = _mz_decode_one[Self.BINS](
            ob, Self.ACT, self.v_min, self.v_max
        )
        ib.free()
        ob.free()
        return value
