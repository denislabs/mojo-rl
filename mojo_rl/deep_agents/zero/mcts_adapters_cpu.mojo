"""CPU AlphaZero adapters for the `planners.tree_search` GenericCPUMCTS surface.

The CPU counterpart to `mcts_adapters.mojo`'s GPU `AZPredGPU` / `AZEnvGPU`.
`GenericCPUMCTS.search[REP, DYN, PRED]` threads a *latent* through three
trait-implementing adapters; for AlphaZero (no learned model, true game rules)
the latent is the env's serialized state and the three adapters are:

  * `AZRepCPU`  — `Representation`: snapshot the live env's state into the latent
    (identity encoder; AlphaZero has no representation network).
  * `AZDynCPU`  — `Dynamics`: load a latent into a workspace env, apply the true
    `env.step` (real game rules), save the resulting state back. Per-edge reward
    is 0 — under `SelfPlay` only the terminal value matters and backup negates it
    up the tree.
  * `AZPredCPU` — `Prediction`: load the latent, and at a terminal return the
    zero-sum outcome from the leaf-mover's perspective (-1 for a win-terminal,
    since the move that ended the game was the *opponent's*; 0 for a draw);
    otherwise run the nn net (`forward["cpu", 1]`), softmax the legal-masked
    policy logits, and tanh-squash the raw value head.

These mirror the legacy `gpu_trait_adapters.mojo` CPU adapters, retargeted from
the old `nn.Network` to an nn `Module` (`forward["cpu", B]`). The env must be
`TwoPlayerDiscreteEnv & Saveable`.
"""

from std.math import exp
from std.memory import UnsafePointer

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.call import call_forward
from mojo_rl.core import TwoPlayerDiscreteEnv, Saveable
from mojo_rl.planners.tree_search import Representation, Dynamics, Prediction


@fieldwise_init
struct AZRepCPU[
    E: TwoPlayerDiscreteEnv & Saveable,
    OBS: Int,
](Movable, ImplicitlyDeletable, Representation):
    """Identity representation: snapshot the live env's serialized state as the
    latent the planner threads through dynamics + prediction. The `obs` argument
    is ignored — the env's own state is the source of truth (the caller positions
    the env at the root before `search`)."""

    comptime OBS_DIM: Int = Self.OBS
    comptime LATENT_DIM: Int = Self.E.SAVE_SIZE

    var env: UnsafePointer[Self.E, MutAnyOrigin]

    def encode_cpu(
        mut self, obs: List[Float64], mut hidden_out: List[Float64]
    ) raises:
        _ = obs  # unused — read env state directly
        var tmp = List[Scalar[DT]](length=Self.E.SAVE_SIZE, fill=0)
        self.env[].save_env_state(tmp)
        for i in range(Self.E.SAVE_SIZE):
            hidden_out[i] = Float64(tmp[i])


@fieldwise_init
struct AZDynCPU[
    E: TwoPlayerDiscreteEnv & Saveable,
    ACT: Int,
](Movable, ImplicitlyDeletable, Dynamics):
    """True game rules as `Dynamics`: load latent → `env.step` → save latent.
    Returns per-edge reward 0.0 (SelfPlay: only terminal value matters)."""

    comptime LATENT_DIM: Int = Self.E.SAVE_SIZE
    comptime ACTION_DIM: Int = Self.ACT

    var env: UnsafePointer[Self.E, MutAnyOrigin]

    def step_cpu(
        mut self,
        hidden_in: List[Float64],
        action: Int,
        mut hidden_out: List[Float64],
    ) raises -> Float64:
        var tmp = List[Scalar[DT]](length=Self.E.SAVE_SIZE, fill=0)
        for i in range(Self.E.SAVE_SIZE):
            tmp[i] = Scalar[DT](hidden_in[i])
        self.env[].load_env_state(tmp)
        _ = self.env[].step(self.env[].action_from_index(action))
        self.env[].save_env_state(tmp)
        for i in range(Self.E.SAVE_SIZE):
            hidden_out[i] = Float64(tmp[i])
        return 0.0


@fieldwise_init
struct AZPredCPU[
    E: TwoPlayerDiscreteEnv & Saveable,
    OBS: Int,
    ACT: Int,
    NET: Module,
](Movable, ImplicitlyDeletable, Prediction):
    """Prediction adapter: load latent, run the policy/value net (or short-circuit
    terminals). On a terminal (`env.game_result() != 0`) the net is skipped and
    the leaf value is the zero-sum outcome from the leaf-mover's perspective
    (-1 win-terminal, 0 draw); SelfPlay backup negates per level so the parent
    gets the right sign. Otherwise: net `forward["cpu", 1]`, legal-masked softmax
    prior, tanh-squashed value."""

    comptime LATENT_DIM: Int = Self.E.SAVE_SIZE
    comptime ACTION_DIM: Int = Self.ACT

    var env: UnsafePointer[Self.E, MutAnyOrigin]
    var net: UnsafePointer[Self.NET, MutAnyOrigin]

    def predict_cpu(
        mut self, hidden: List[Float64], mut policy_out: List[Float64]
    ) raises -> Float64:
        var tmp = List[Scalar[DT]](length=Self.E.SAVE_SIZE, fill=0)
        for i in range(Self.E.SAVE_SIZE):
            tmp[i] = Scalar[DT](hidden[i])
        self.env[].load_env_state(tmp)

        var game_result = self.env[].game_result()
        if game_result != 0:
            for a in range(Self.ACT):
                policy_out[a] = 0.0
            if game_result == 3:
                return 0.0
            return -1.0  # win-terminal: the player to move just lost

        # Non-terminal: run the net on the canonical obs (storage surface —
        # owned host `Tensor`s, no raw pointers).
        comptime IN = Self.NET.IN_DIMS[0]
        comptime OUT = Self.NET.OUT_DIM
        var obs_raw = self.env[].get_obs_list()
        var obs_t = Tensor.alloc(IN)
        for i in range(IN):
            obs_t.data[i] = (
                Scalar[DT](obs_raw[i]) if i < len(obs_raw) else Scalar[DT](0.0)
            )
        var pred_t = Tensor.alloc(OUT)
        call_forward["cpu", 1](self.net[], TensorRefs[Self.NET.ARITY](obs_t), pred_t, None)

        # Legal-masked softmax over the policy logits (illegal → 0, renormalize).
        var legal = self.env[].legal_action_mask()
        var max_l: Float64 = -1e18
        for a in range(Self.ACT):
            var lv = Float64(pred_t.data[a])
            if a < len(legal) and legal[a] and lv > max_l:
                max_l = lv
        var sum_e: Float64 = 0.0
        for a in range(Self.ACT):
            if a < len(legal) and legal[a]:
                policy_out[a] = exp(Float64(pred_t.data[a]) - max_l)
                sum_e += policy_out[a]
            else:
                policy_out[a] = 0.0
        if sum_e > 0.0:
            for a in range(Self.ACT):
                policy_out[a] /= sum_e

        var raw_v = Float64(pred_t.data[Self.ACT])
        if raw_v > 15.0:
            return 1.0
        if raw_v < -15.0:
            return -1.0
        var ev = exp(2.0 * raw_v)
        return (ev - 1.0) / (ev + 1.0)
