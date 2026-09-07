"""A3 — ONLINE off-policy FB on dm_control walker, batched GPU env, NO dataset.

`docs/BFM_ZERO_SHOT_RL.md` §18.3 step A3. The question this run answers is
one sentence long: does `pi_z` trained with NO dataset — generating its own
coverage in `N_ENVS` parallel walkers — recover more of the SAC experts than
the ~19 / 5 / 5 % the offline runs are stuck at (§13)? §16.2 says the
offline number is the coverage limit; this is the experiment that isolates
that claim, because it changes nothing else: same `FBTrainer`, same nets as
`fb_train_gpu.mojo`, same eval protocol.

    pixi run -e nvidia mojo run -I . examples/fb/fb_online_walker_gpu.mojo \
        [--steps N] [--ups K] [--warmup N] [--z-hold N] [--bc X] [--ortho X] \
        [--lr-b X] [--expl-std X] [--act-l2 X] [--tag NAME]

Then score it — the online arm has its OWN eval, because the batched env's
observation is dm_control's 24-D vector and not the store's `[qpos | qvel]`:

    pixi run mojo run -I . examples/fb/fb_eval_walker_online.mojo <ckpt>

## What the numbers in the log mean

  * `mean_ret` is the env's OWN reward (walker `walk`) under the exploring,
    z-conditioned policy. FB never trains on it. It is a COVERAGE signal —
    "does the population ever walk" — not a zero-shot score.
  * `fb/*` at each diag flush are the collapse detectors §14 insists on:
    `|B|` must read sqrt(d) = 11.314 at every flush (hard invariant under
    `LayerNormNoAffine`), `ortho_Q` must not grow without bound, `|F|` and
    `grad_norm_f1` must settle, `mean|a|` must stay well below 1.
  * `replay_size` shows the ring filling; nothing is sampled below one batch.

## Knobs, and where their defaults come from

  N_ENVS 256 / `--ups` 8 / BATCH 1024: BFM-Zero runs 1024 envs and 16
  updates per env step (Table 1). At tau = 0.01, 16 updates move the Polyak
  target 15 % per iteration; 8 moves it 7.7 %. The walker SAC ladder measured
  64 updates/iter as a target that chased itself (§13's SAC note), so this
  starts at 8 and exposes it.
  `--z-hold` 150 and the 10 k ZBuffer are BFM-Zero's rollout rule.
  `--bc` 0.0 and `--act-l2` 1.0: the FIRST run of this script (bc 0, no
  penalty, 4.9 M env steps) went bang-bang from the first flush — mean|a|
  0.82 → 0.88, 82–90 % saturated at eval, walk 0.71x / run 0.48x random.
  The prediction that on-policy data corrects the corner was wrong; see
  `FBTrainer.act_l2_weight`. Watch `mean|a|` at the first flush: at 1.0 it
  must sit well below 0.8, or the penalty is in the wrong decade.
  `--ortho` 100 and `--lr-b` 1e-5: the A2 winner (§18.6.1), which is the
  reference's own PAIR — each was null alone, together stand 1.57 / walk
  1.92 / run 1.63x random offline with every rung SIGNAL. That offline arm
  (`ortho100_lrb1e5_u`) is the number this run has to beat with no dataset.

⚠ Segmented, with STEP-STAMPED checkpoints — the driver overwrites its
`checkpoint_path` on every save, and §13 records a good early checkpoint
being destroyed that way. One agent, `N_SEGMENTS` calls into the driver.

⚠ `USE_ENV_CUDA_GRAPH=False`: the fields path's blocked Newton kernel does not
replay (same as the SAC walker script). The TRAIN step is captured.
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.sys import argv
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import CsvLogger, RemoteLogger, CompositeLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh
from mojo_rl.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from mojo_rl.deep_agents.fb.online import FBOnlineAgent
from mojo_rl.deep_agents.training.driver_offpolicy import (
    run_offpolicy_train_batched,
)
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig


# ── the env ──────────────────────────────────────────────────────────────
# `walk` for the coverage readout only; FB never reads the reward.
comptime WalkerCfg = DMWalkerConfig[1.0]
comptime OBS: Int = DMWalkerModel.OBS_DIM      # 24 — dm_control's vector
comptime NACT: Int = DMWalkerModel.ACTION_DIM  # 6
comptime N_ENVS: Int = 256
comptime EnvT = Phyics3dBatchedEnv[
    DMWalkerModel, WalkerCfg, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

# ── the model — MUST match fb_eval_walker_online.mojo ────────────────────
comptime D: Int = 128
comptime BATCH: Int = 1024
comptime HID: Int = 1024
comptime CAP: Int = 1_000_000
comptime ZBUF: Int = 10_000
comptime F_IN = OBS + NACT + D
comptime A_IN = OBS + D
comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, 256], ReLU[256], Linear[256, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, NACT], Tanh[NACT]
]
comptime Agent = FBOnlineAgent[
    FNet, BNet, ANet, OBS, NACT, D, BATCH, CAP, N_ENVS, ZBUF
]

# ── the run ──────────────────────────────────────────────────────────────
comptime TOTAL_ENV_STEPS: Int = 5_000_000
comptime SEGMENT_STEPS: Int = 250_000
comptime UPDATES_PER_ITER: Int = 8
comptime WARMUP_STEPS: Int = 25_600      # 100 iterations of random actions
comptime Z_HOLD: Int = 150
comptime EXPL_STD: Float64 = 0.2
comptime BC_WEIGHT: Float64 = 0.0
comptime ACT_L2: Float64 = 1.0
comptime ORTHO_WEIGHT: Float64 = 100.0
comptime LR_B: Float64 = 1e-5
comptime MAX_GRAD_NORM: Float64 = 1.0
comptime DIAG_EVERY: Int = N_ENVS * 100  # 100 iterations
comptime PRINT_EVERY: Int = N_ENVS * 500
comptime USE_TRAIN_CUDA_GRAPH: Bool = True
comptime CKPT_PATH: StaticString = "fb_online_walker_d128.ckpt"
comptime CSV_PATH: StaticString = "fb_online_walker_d128_metrics.csv"
comptime RUN_NAME: StaticString = "FB online walker d128"
comptime SEED: Int = 20260907

comptime LoggerT = CompositeLogger[CsvLogger, RemoteLogger]


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def main() raises:
    var total = atol(_flag(String("--steps"), String(TOTAL_ENV_STEPS)))
    var ups = atol(_flag(String("--ups"), String(UPDATES_PER_ITER)))
    var warmup = atol(_flag(String("--warmup"), String(WARMUP_STEPS)))
    var z_hold = atol(_flag(String("--z-hold"), String(Z_HOLD)))
    var bc_w = atof(_flag(String("--bc"), String(BC_WEIGHT)))
    var act_l2 = atof(_flag(String("--act-l2"), String(ACT_L2)))
    var ortho_w = atof(_flag(String("--ortho"), String(ORTHO_WEIGHT)))
    var lr_b = atof(_flag(String("--lr-b"), String(LR_B)))
    var expl = atof(_flag(String("--expl-std"), String(EXPL_STD)))
    var tag = _flag(String("--tag"), String(""))
    var ckpt = String(CKPT_PATH)
    var csv_path = String(CSV_PATH)
    var run_name = String(RUN_NAME)
    if tag.byte_length() > 0:
        ckpt = "fb_online_walker_" + tag + ".ckpt"
        csv_path = "fb_online_walker_" + tag + "_metrics.csv"
        run_name = String(RUN_NAME) + " [" + tag + "]"
    if warmup < BATCH:
        raise Error("--warmup must be >= BATCH (" + String(BATCH) + ")")
    var seg = SEGMENT_STEPS if SEGMENT_STEPS < total else total
    var n_segments = (total + seg - 1) // seg

    seed(SEED)
    print("=" * 70)
    print("FB ONLINE — dm_control walker, batched GPU, no dataset")
    print("=" * 70)
    print("  OBS / NACT / D      =", OBS, "/", NACT, "/", D)
    print("  N_ENVS / BATCH      =", N_ENVS, "/", BATCH)
    print("  env steps           =", total, " in", n_segments, "segments of", seg)
    print("  updates / iteration =", ups, " (", ups * BATCH // N_ENVS, "samples per env step )")
    print("  warmup env steps    =", warmup)
    print("  z_hold / ZBUF       =", z_hold, "/", ZBUF)
    print("  expl_std / bc / act_l2 / ortho / lr_b =", expl, "/", bc_w, "/", act_l2, "/", ortho_w, "/", lr_b)
    print("  CUDA graph (train)  =", USE_TRAIN_CUDA_GRAPH)
    print("  tag                 = '", tag, "'")
    print("=" * 70)

    with DeviceContext() as ctx:
        var env_vars = load_dotenv()
        var logger = CompositeLogger(
            CsvLogger(csv_path, buffer_size=64),
            RemoteLogger(
                server_url=env_vars.get("RL_MONITOR_URL", ""),
                run_name=run_name,
                buffer_size=64,
                api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
            ),
        )
        logger.set_config("algorithm", "FB-online")
        logger.set_config("env", "dm_control/walker-walk (coverage readout)")
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("d", String(D))
        logger.set_config("batch", String(BATCH))
        logger.set_config("hidden", String(HID))
        logger.set_config("replay_cap", String(CAP))
        logger.set_config("updates_per_iter", String(ups))
        logger.set_config("warmup", String(warmup))
        logger.set_config("z_hold", String(z_hold))
        logger.set_config("expl_std", String(expl))
        logger.set_config("bc_weight", String(bc_w))
        logger.set_config("act_l2", String(act_l2))
        logger.set_config("ortho_weight", String(ortho_w))
        logger.set_config("lr_b", String(lr_b if lr_b >= 0.0 else 3e-4))
        logger.set_config("max_grad_norm", String(MAX_GRAD_NORM))
        logger.set_config("tag", tag)
        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        var agent = Agent.make(
            ctx,
            lr=3e-4,
            lr_b=lr_b,
            ortho_weight=ortho_w,
            max_grad_norm=MAX_GRAD_NORM,
            bc_weight=bc_w,
            act_l2_weight=act_l2,
            learning_starts=warmup,
            action_scale=1.0,
            expl_std=expl,
            z_hold=z_hold,
            window_size=100,
            initial_episode_fill=0.0,
            seed=UInt64(SEED),
        )
        var env = EnvT(ctx)

        var t_start = perf_counter_ns()
        for s in range(n_segments):
            var done_steps = s * seg
            var this_seg = seg if done_steps + seg <= total else total - done_steps
            _ = run_offpolicy_train_batched[
                Agent, EnvT, N_ENVS=N_ENVS,
                USE_TRAIN_CUDA_GRAPH=USE_TRAIN_CUDA_GRAPH,
                USE_ENV_CUDA_GRAPH=False,
                L=LoggerT,
            ](
                Optional(ctx), agent, env, this_seg,
                rng_seed=UInt64(SEED + s),
                updates_per_step=ups,
                print_every=PRINT_EVERY,
                verbose=True,
                logger=logger_ptr,
                diag_every=DIAG_EVERY,
                episode_sync_every=32,
                base_step=done_steps,
                progress_label="fb-online",
            )
            var at = done_steps + this_seg
            var path = ckpt + "." + String(at)
            agent.save_state(path)
            var el = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "  [segment", s + 1, "/", n_segments, "]  env steps", at,
                "  train steps", agent.total_train_steps(),
                "  mean_ret(walk)", agent.mean_return(),
                "  replay", agent.size, "  ", Float64(at) / el, "env st/s",
                " ->", path,
            )
        var pf = ckpt + ".final"
        agent.save_state(pf)
        logger.close()
        _ = logger
        print("=" * 70)
        print("done. final checkpoint ->", pf, "   metrics ->", csv_path)
        print("next:  pixi run mojo run -I . examples/fb/fb_eval_walker_online.mojo", pf)
        print("=" * 70)
