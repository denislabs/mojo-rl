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
    otherwise run the nn2 net (`forward["cpu", 1]`), softmax the legal-masked
    policy logits, and tanh-squash the raw value head.

These mirror the legacy `gpu_trait_adapters.mojo` CPU adapters, retargeted from
the old `nn.Network` to an nn2 `Module` (`forward["cpu", B]`). The env must be
`TwoPlayerDiscreteEnv & Saveable`.
"""

from std.math import exp
from std.memory import alloc, UnsafePointer
from layout import TileTensor, row_major

from mojo_rl.nn.constants import dtype
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.core import TwoPlayerDiscreteEnv, Saveable
from mojo_rl.planners.tree_search import Representation, Dynamics, Prediction


@fieldwise_init
struct AZRepCPU[
    E: TwoPlayerDiscreteEnv & Saveable,
    OBS: Int,
](Movable, ImplicitlyDestructible, Representation):
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
        var tmp = alloc[Scalar[dtype]](Self.E.SAVE_SIZE)
        self.env[].save_env_state(tmp)
        for i in range(Self.E.SAVE_SIZE):
            hidden_out[i] = Float64(tmp[i])
        tmp.free()


@fieldwise_init
struct AZDynCPU[
    E: TwoPlayerDiscreteEnv & Saveable,
    ACT: Int,
](Movable, ImplicitlyDestructible, Dynamics):
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
        var tmp = alloc[Scalar[dtype]](Self.E.SAVE_SIZE)
        for i in range(Self.E.SAVE_SIZE):
            tmp[i] = Scalar[dtype](hidden_in[i])
        self.env[].load_env_state(tmp)
        _ = self.env[].step(self.env[].action_from_index(action))
        self.env[].save_env_state(tmp)
        for i in range(Self.E.SAVE_SIZE):
            hidden_out[i] = Float64(tmp[i])
        tmp.free()
        return 0.0


@fieldwise_init
struct AZPredCPU[
    E: TwoPlayerDiscreteEnv & Saveable,
    OBS: Int,
    ACT: Int,
    NET: Module,
](Movable, ImplicitlyDestructible, Prediction):
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
        var tmp = alloc[Scalar[dtype]](Self.E.SAVE_SIZE)
        for i in range(Self.E.SAVE_SIZE):
            tmp[i] = Scalar[dtype](hidden[i])
        self.env[].load_env_state(tmp)
        tmp.free()

        var game_result = self.env[].game_result()
        if game_result != 0:
            for a in range(Self.ACT):
                policy_out[a] = 0.0
            if game_result == 3:
                return 0.0
            return -1.0  # win-terminal: the player to move just lost

        # Non-terminal: run the net on the canonical obs.
        comptime IN = Self.NET.IN_DIMS[0]
        comptime OUT = Self.NET.OUT_DIM
        var obs_raw = self.env[].get_obs_list()
        var obs_buf = alloc[Scalar[DT]](IN)
        for i in range(IN):
            obs_buf[i] = Scalar[DT](obs_raw[i]) if i < len(obs_raw) else Scalar[
                DT
            ](0.0)
        var pred_buf = alloc[Scalar[DT]](OUT)
        var obs_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](obs_buf),
            row_major[1, IN](),
        )
        var pred_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](pred_buf),
            row_major[1, OUT](),
        )
        self.net[].forward["cpu", 1](obs_t, output=pred_t)

        # Legal-masked softmax over the policy logits (illegal → 0, renormalize).
        var legal = self.env[].legal_action_mask()
        var max_l: Float64 = -1e18
        for a in range(Self.ACT):
            var lv = Float64(pred_buf[a])
            if a < len(legal) and legal[a] and lv > max_l:
                max_l = lv
        var sum_e: Float64 = 0.0
        for a in range(Self.ACT):
            if a < len(legal) and legal[a]:
                policy_out[a] = exp(Float64(pred_buf[a]) - max_l)
                sum_e += policy_out[a]
            else:
                policy_out[a] = 0.0
        if sum_e > 0.0:
            for a in range(Self.ACT):
                policy_out[a] /= sum_e

        var raw_v = Float64(pred_buf[Self.ACT])
        obs_buf.free()
        pred_buf.free()
        if raw_v > 15.0:
            return 1.0
        if raw_v < -15.0:
            return -1.0
        var ev = exp(2.0 * raw_v)
        return (ev - 1.0) / (ev + 1.0)
