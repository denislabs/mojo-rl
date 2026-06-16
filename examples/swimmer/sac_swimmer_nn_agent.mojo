"""SAC training on Swimmer (CPU) via the new `SACAgent` facade.

Swimmer counterpart of `examples/half_cheetah/sac_half_cheetah_nn_agent.mojo`.
Uses the new `deep_agents/` surface:

  * `SACAgent[...]` — facade over `SACTrainer` + the single-env off-policy
    driver.
  * `RemoteLogger` — streams metrics to a dashboard at every chunk boundary
    AND at the driver's `print_every` cadence. Config (server URL + API key)
    read from a `.env` via `mojo_rl.core.dotenv`.
  * Single-file checkpointing — `agent.save(CHECKPOINT_PATH)` writes ONE
    `.ckpt` file (overwritten each chunk) under a single `nn-ckpt v2`
    envelope containing actor + twin critics + their Adam states +
    `alpha_opt` ScalarAdam.

After training, the final checkpoint is reloaded into the same agent and a
greedy probe confirms the action vector reproduces dimension-by-dimension to
`|diff| < 1e-5`.

Swimmer (Phyics3dEnv, MuJoCo-style):
  * 8D observation (qpos[2:5] + qvel[0:5])
  * 2D continuous action (2 rotational motor torques)
  * Reward ≈ x_velocity - 0.0001·||action||²
  * No early termination (`TERMINATE_ON_UNHEALTHY=False`).

Run:
    pixi run mojo run -I . examples/swimmer/sac_swimmer_nn_agent.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep
from mojo_rl.envs.swimmer import Swimmer


# =============================================================================
# Architecture
# =============================================================================

comptime EnvT = Swimmer[DT, TERMINATE_ON_UNHEALTHY=False]
comptime OBS_DIM = EnvT.OBS_DIM  # 8
comptime ACT_DIM = EnvT.ACTION_DIM  # 2
comptime HIDDEN = 256
comptime BATCH = 256
comptime REPLAY_CAPACITY = 100_000

# Training duration. CPU single-env; drop NUM_STEPS to 20_000 for a smoke run.
comptime NUM_STEPS = 200_000
comptime PRINT_EVERY = 5_000
comptime DIAG_EVERY = 5_000
comptime CHECKPOINT_EVERY = 50_000

comptime CHECKPOINT_PATH = "sac_swimmer_nn.ckpt"


comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents) — Swimmer CPU + checkpoints + logger")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN             =", HIDDEN)
    print("  BATCH              =", BATCH)
    print("  REPLAY_CAPACITY    =", REPLAY_CAPACITY)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  PRINT_EVERY        =", PRINT_EVERY)
    print("  DIAG_EVERY         =", DIAG_EVERY)
    print("  CHECKPOINT_EVERY   =", CHECKPOINT_EVERY)
    print("  Checkpoint path    =", CHECKPOINT_PATH)
    print("=" * 70)

    # ─── Logger (remote) ───────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="SAC Swimmer NN2 (CPU)",
        buffer_size=200,
        api_key=api_key,
    )
    logger.set_config("algorithm", "SAC")
    logger.set_config("env", "Swimmer")
    logger.set_config("hidden", String(HIDDEN))
    logger.set_config("batch", String(BATCH))

    var logger_ptr = UnsafePointer(to=logger)

    # ─── Agent + env ─────────────────────────────────────────────────────
    var agent = SACAgent[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet,
        CriticNet,
    ](
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        init_alpha=0.2,
        target_entropy=-Scalar[DT](ACT_DIM),  # SAC default heuristic
        learning_starts=1_000,
        window_size=100,
        initial_episode_fill=0.0,
        use_ere=False,
        ere_eta=0.996,
    )
    var env = EnvT()

    # ─── Single train() call — auto-flush + auto-checkpoint ──────────────
    var t_start = perf_counter_ns()
    _ = agent.train_single[
        EnvT,
        L=RemoteLogger,
    ](
        env,
        NUM_STEPS,
        print_every=PRINT_EVERY,
        verbose=True,
        logger=logger_ptr,
        diag_every=DIAG_EVERY,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_every=CHECKPOINT_EVERY,
    )
    var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
    var total = NUM_STEPS
    logger.close()
    _ = logger  # lifetime extender for logger_ptr

    # ─── Summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("Training complete")
    print("  total env_steps        =", total)
    print("  elapsed                =", elapsed_s, "s")
    print("  mean ep return (last 100) =", agent.mean_return())
    print("  episodes completed     =", agent.ep_count())
    print("  remote points sent     =", logger.total_logged())
    print("=" * 70)

    var final_avg = Float64(agent.mean_return())
    if final_avg > 100.0:
        print("EXCELLENT — swimming fast (mean > 100).")
    elif final_avg > 40.0:
        print("STRONG — learned to swim (mean > 40).")
    elif final_avg > 20.0:
        print("PROGRESS — early locomotion (mean > 20).")
    elif final_avg > 0.0:
        print("LEARNING — positive return (mean > 0).")
    else:
        print("EARLY — still exploring (mean < 0).")
    print("=" * 70)

    # ─── Save/load round-trip smoke test ─────────────────────────────────
    var probe_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    for d in range(OBS_DIM):
        probe_obs[d] = Scalar[DT](0.1 * Float64(d - OBS_DIM // 2))
    var act_before = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe_obs, act_before)

    agent.load(CHECKPOINT_PATH)
    var act_after = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe_obs, act_after)

    print("Save/load round-trip on probe obs:")
    var ok = True
    for j in range(ACT_DIM):
        var diff = Float64(act_after[j] - act_before[j])
        if diff < 0:
            diff = -diff
        print(
            "  dim",
            j,
            " before =",
            act_before[j],
            " after =",
            act_after[j],
            " |diff| =",
            diff,
        )
        if diff > 1e-5:
            ok = False
    if ok:
        print("Round-trip OK (max |diff| < 1e-5 on every action dim).")
    else:
        print("Round-trip MISMATCH — investigate save/load semantics.")
    print("=" * 70)
