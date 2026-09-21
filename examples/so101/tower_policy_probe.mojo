"""Where does a `--bc-weight` policy leave the expert? — a checkpoint probe.

    pixi run mojo run -I . examples/so101/tower_policy_probe.mojo \\
        --ckpt projects/so101-tower/runs/<run>/checkpoints/last.ckpt \\
        --demos projects/so101-tower/demos/expert_lift_brick_50.demo \\
        [--episodes 8] [--seed 5000] [--steps 300]
        [--replay-episode K [--open-loop]]   # leg 3: drive demo episode K's placement, print the drift
                                             # (open loop = the RECORDED actions: the env's reproducibility)

Two legs, both on the CPU env (float64 physics, the family's own reward on
the host — `tasks/host_reward.mojo`):

1. **TEACHER FORCING** on the demo file: the checkpoint's greedy action at
   every recorded observation against the recorded action, as a mean L1 per
   action dimension and per third of the episode (approach / grasp / lift).
   This is the number the trainer's BC term minimises — on the demos it was
   trained on it should be ~0.003 (run b7a3c65d), and on a file the run never
   saw it says whether the fit is a POLICY over the placement region or 50
   memorised trajectories.

2. **CLOSED LOOP** from fresh placements (`--seed`, far from every demo
   seed): the checkpoint drives for `--steps` steps; per episode the return,
   the rows paying the closing bonus (> 1.24) and the rung (> 1.5), the
   closest the jaw came to the brick and the highest the brick rose. The
   greedy eval in the training log is this number averaged over 32 lanes on
   the GPU — here it comes with WHY.

⚠ THE SEEDS ARE THE COVERAGE. `expert_lift_brick_50.demo` is seeds 0..49,
`expert_lift_brick_500.demo` seeds 1000..1499 (`posed_qpos`'s draw is the
seed). A closed-loop leg on a seed the demos hold is a memorisation check,
not a generalisation one.
"""

from std.math import sqrt
from std.sys import argv
from std.pathlib import Path

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.core.cont_action import ContAction
from noeira.deep_agents.data.any_replay import AnyReplay
from noeira.deep_agents.demos.file import DemoSet, read_demo_file
from noeira.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from noeira.tasks.sac_family_policy import (
    SacFamilyPolicy, HIDDEN as FAMILY_HIDDEN, POLICY_BATCH, POLICY_CAP,
)
from noeira.deep_agents.training.blocks import ReplaySampleStep
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.physics3d.fields import actuator_column
from noeira.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig, So101TowerTeleopConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.host_reward import family_reward_host
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.utils.fmt import fixed

comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime TASK = "so101_tower_lift_brick"
comptime CFG = So101TowerTeleopConfig
comptime E = Phyics3dEnv[So101TowerModel, CFG, DType.float64, False]
comptime NV = So101TowerModel.NV
comptime OBS_DIM = So101TowerModel.OBS_DIM
comptime ACT = 6
comptime HIDDEN = FAMILY_HIDDEN
"""The family SAC widths, from `noeira/tasks/sac_family_policy.mojo` — the
driver trains with the same constants, so a `--policy` checkpoint loads."""
comptime BATCH = POLICY_BATCH
comptime CAP = POLICY_CAP
comptime GB = So101TowerConfig.OBS_GOAL_BASE

comptime Agent = SacFamilyPolicy[OBS_DIM, ACT]


def _usage():
    print("usage: tower_policy_probe.mojo --ckpt F [--demos F] [--episodes N]"
          " [--seed S] [--steps N] [--replay-episode K [--open-loop]]")


def _reach_mm(ref obs: List[Float64]) -> Float64:
    var x = obs[GB + 3]
    var y = obs[GB + 4]
    var z = obs[GB + 5]
    return sqrt(x * x + y * y + z * z) * 1000.0


def teacher_forcing(mut agent: Agent, ref d: DemoSet) raises:
    """Leg 1: greedy action vs recorded action, on every row."""
    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0))
    var a_rec = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    var a_pol = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    var per_dim = List[Float64](length=ACT, fill=0.0)
    var per_third = List[Float64](length=3, fill=0.0)
    var n_third = List[Int](length=3, fill=0)
    var total = 0.0
    var n_rows = 0
    var worst_ep = -1
    var worst_ep_l1 = -1.0
    for e in range(d.n_episodes()):
        var start = d.ep_start[e]
        var ln = d.ep_len[e]
        var ep_sum = 0.0
        for k in range(ln):
            var r = start + k
            d.row_obs[DT](r, obs)
            d.row_act[DT](r, a_rec)
            agent.select_greedy_action(obs, a_pol)
            var l1 = 0.0
            for j in range(ACT):
                var diff = abs(Float64(a_pol[j]) - Float64(a_rec[j]))
                per_dim[j] += diff
                l1 += diff
            l1 /= Float64(ACT)
            var third = (3 * k) // ln
            if third > 2:
                third = 2
            per_third[third] += l1
            n_third[third] += 1
            ep_sum += l1
            total += l1
            n_rows += 1
        var ep_l1 = ep_sum / Float64(ln)
        if ep_l1 > worst_ep_l1:
            worst_ep_l1 = ep_l1
            worst_ep = e
    print("  rows", n_rows, " episodes", d.n_episodes())
    print("  mean L1 (all rows)     :", fixed(total / Float64(n_rows), 4))
    var dims = String("  per dimension          :")
    for j in range(ACT):
        dims += " " + fixed(per_dim[j] / Float64(n_rows), 4)
    print(dims)
    print("  per third (approach / grasp / lift):",
          fixed(per_third[0] / Float64(max(n_third[0], 1)), 4),
          fixed(per_third[1] / Float64(max(n_third[1], 1)), 4),
          fixed(per_third[2] / Float64(max(n_third[2], 1)), 4))
    print("  worst episode          :", worst_ep, " L1", fixed(worst_ep_l1, 4))


def closed_loop(
    mut agent: Agent, mut env: E, n_episodes: Int, seed: Int, n_steps: Int,
) raises:
    """Leg 2: the checkpoint drives from fresh placements."""
    var sf = So101TowerModel.make_spec_fields[DType.float64]()
    var lo = actuator_column(sf, ACT_IDX_CTRL_MIN, ACT)
    var hi = actuator_column(sf, ACT_IDX_CTRL_MAX, ACT)

    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DType.float64](cw[k])
    var mw = task_meta_words(
        String(TASK), String(FAMILY), CFG.SHAPE_W_GOAL, CFG.SHAPE_W_REACH,
        CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
    )
    var brick = -1
    for b in range(len(fmd.body_names)):
        if String(fmd.body_names[b]) == "brick_brick":
            brick = b
    if brick < 0:
        raise Error("brick_brick not found in the composed scene")

    var frame_skip = CFG.FRAME_SKIP
    var timestep = So101TowerModel.TIMESTEP
    var obs64 = List[Float64](length=OBS_DIM, fill=0.0)
    var obs32 = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0))
    var a32 = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    var act_l = List[Float64](length=ACT, fill=0.0)
    var sum_return = 0.0
    var n_rung_eps = 0
    for ep in range(n_episodes):
        _ = env.reset()
        for k in range(len(mw[0])):
            env.d.meta.data[mw[0][k]] = Scalar[DType.float64](mw[1][k])
        var q0 = posed_qpos[So101TowerPlacement](
            String(TASK), String(FAMILY), So101TowerConfig.SLOT_RADIUS,
            UInt64(seed + ep),
        )
        var v0 = List[Float64](length=NV, fill=0.0)
        var s0 = env.obs_at(q0, v0)
        for k in range(OBS_DIM):
            obs64[k] = s0.data[k]
        # let the props settle on the desk, the arm holding its pose (the
        # recorder's own preamble)
        var hold = ContAction[ACT]()
        for k in range(ACT):
            var span = Float64(hi[k]) - Float64(lo[k])
            var q = Float64(env.d.qpos.data[k])
            var a = 2.0 * (q - Float64(lo[k])) / span - 1.0 if span != 0.0 else 0.0
            hold.data[k] = a
        for _ in range(5):
            var o = env.step(hold)
            for k in range(OBS_DIM):
                obs64[k] = o[0].data[k]
        var ret = 0.0
        var n_close = 0
        var n_rung = 0
        var reach_min = 1e9
        var z_max = -1.0
        var first_rung = -1
        var brick_x0 = Float64(env.d.xpos.data[brick * 3 + 0])
        var brick_y0 = Float64(env.d.xpos.data[brick * 3 + 1])
        for t in range(n_steps):
            for k in range(OBS_DIM):
                obs32[k] = Scalar[DT](obs64[k])
            agent.select_greedy_action(obs32, a32)
            var action = ContAction[ACT]()
            for k in range(ACT):
                action.data[k] = Float64(a32[k])
                act_l[k] = Float64(a32[k])
            var o = env.step(action)
            for k in range(OBS_DIM):
                obs64[k] = o[0].data[k]
            var rd = family_reward_host[CFG, DType.float64, E.MD, ACT](
                env.d, env.mf, act_l, t + 1, frame_skip, timestep,
            )
            var r = Float64(rd[0])
            ret += r
            if r > 1.24:
                n_close += 1
            if r > 1.5:
                n_rung += 1
                if first_rung < 0:
                    first_rung = t
            var reach = _reach_mm(obs64)
            if reach < reach_min:
                reach_min = reach
            var z = Float64(env.d.xpos.data[brick * 3 + 2])
            if z > z_max:
                z_max = z
        sum_return += ret
        if n_rung > 0:
            n_rung_eps += 1
        print("  ep", ep, " seed", seed + ep, " brick at (",
              fixed(brick_x0, 3), ",", fixed(brick_y0, 3), ")  return",
              fixed(ret, 1), " rows>1.24", n_close, " rows>1.5", n_rung,
              " first rung", first_rung, " reach min", fixed(reach_min, 1),
              "mm  brick z max", fixed(z_max, 3))
    print("  mean return", fixed(sum_return / Float64(n_episodes), 1),
          " episodes with the rung", n_rung_eps, "of", n_episodes)


def replay_episode(
    mut agent: Agent, mut env: E, ref d: DemoSet, ep: Int, seed0: Int,
    open_loop: Bool,
) raises:
    """Leg 3: drive from demo episode `ep`'s placement (seed `seed0 + ep`)
    and print, step by step, how far the closed loop drifts from the
    recording: the largest observation gap (and which word), the policy's
    action against the recorded one, and the teacher-forced action at the
    RECORDED observation against the recorded one."""
    var sf = So101TowerModel.make_spec_fields[DType.float64]()
    var lo = actuator_column(sf, ACT_IDX_CTRL_MIN, ACT)
    var hi = actuator_column(sf, ACT_IDX_CTRL_MAX, ACT)
    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DType.float64](cw[k])
    var mw = task_meta_words(
        String(TASK), String(FAMILY), CFG.SHAPE_W_GOAL, CFG.SHAPE_W_REACH,
        CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
    )
    var frame_skip = CFG.FRAME_SKIP
    var timestep = So101TowerModel.TIMESTEP
    _ = env.reset()
    for k in range(len(mw[0])):
        env.d.meta.data[mw[0][k]] = Scalar[DType.float64](mw[1][k])
    var q0 = posed_qpos[So101TowerPlacement](
        String(TASK), String(FAMILY), So101TowerConfig.SLOT_RADIUS,
        UInt64(seed0 + ep),
    )
    var v0 = List[Float64](length=NV, fill=0.0)
    var obs64 = List[Float64](length=OBS_DIM, fill=0.0)
    var s0 = env.obs_at(q0, v0)
    for k in range(OBS_DIM):
        obs64[k] = s0.data[k]
    var hold = ContAction[ACT]()
    for k in range(ACT):
        var span = Float64(hi[k]) - Float64(lo[k])
        var q = Float64(env.d.qpos.data[k])
        hold.data[k] = 2.0 * (q - Float64(lo[k])) / span - 1.0 if span != 0.0 else 0.0
    for _ in range(5):
        var o = env.step(hold)
        for k in range(OBS_DIM):
            obs64[k] = o[0].data[k]

    var start = d.ep_start[ep]
    var ln = d.ep_len[ep]
    var o_rec = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0))
    var a_rec = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    var a_tf = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    var obs32 = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0))
    var a32 = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    var act_l = List[Float64](length=ACT, fill=0.0)
    print("  episode", ep, " seed", seed0 + ep, " recorded length", ln)
    print("   t | max|obs'-obs| (word) | mean|a'-a_rec| | mean|pi(obs)-a_rec| | r'   r_rec")
    for t in range(ln):
        var r = start + t
        d.row_obs[DT](r, o_rec)
        d.row_act[DT](r, a_rec)
        agent.select_greedy_action(o_rec, a_tf)
        var gap = 0.0
        var gap_w = -1
        for k in range(OBS_DIM):
            var g = abs(obs64[k] - Float64(o_rec[k]))
            if g > gap:
                gap = g
                gap_w = k
        for k in range(OBS_DIM):
            obs32[k] = Scalar[DT](obs64[k])
        agent.select_greedy_action(obs32, a32)
        var da = 0.0
        var dtf = 0.0
        for k in range(ACT):
            da += abs(Float64(a32[k]) - Float64(a_rec[k]))
            dtf += abs(Float64(a_tf[k]) - Float64(a_rec[k]))
        da /= Float64(ACT)
        dtf /= Float64(ACT)
        var action = ContAction[ACT]()
        for k in range(ACT):
            # ⚠ OPEN LOOP replays the RECORDED action: the gap it prints is
            # the env's own reproducibility of the recording, not the policy.
            var v = Float64(a_rec[k]) if open_loop else Float64(a32[k])
            action.data[k] = v
            act_l[k] = v
        if t < 4:
            var line = String("      a_pol - a_rec per dim:")
            for k in range(ACT):
                line += " " + fixed(Float64(a32[k]) - Float64(a_rec[k]), 4)
            print(line)
        var o = env.step(action)
        for k in range(OBS_DIM):
            obs64[k] = o[0].data[k]
        var rd = family_reward_host[CFG, DType.float64, E.MD, ACT](
            env.d, env.mf, act_l, t + 1, frame_skip, timestep,
        )
        if t < 24 or t % 10 == 0 or t == ln - 1:
            print("  ", t, "|", fixed(gap, 4), "(", gap_w, ") |", fixed(da, 4),
                  "|", fixed(dtf, 4), "|", fixed(Float64(rd[0]), 3),
                  fixed(Float64(d.rew[r]), 3))


def main() raises:
    var args = argv()
    var ckpt = String("")
    var demos = String("")
    var n_episodes = 8
    var seed = 5000
    var n_steps = 300
    var replay_ep = -1
    var open_loop = False
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--ckpt" and i + 1 < len(args):
            ckpt = String(args[i + 1])
            i += 2
        elif a == "--demos" and i + 1 < len(args):
            demos = String(args[i + 1])
            i += 2
        elif a == "--episodes" and i + 1 < len(args):
            n_episodes = Int(String(args[i + 1]))
            i += 2
        elif a == "--seed" and i + 1 < len(args):
            seed = Int(String(args[i + 1]))
            i += 2
        elif a == "--steps" and i + 1 < len(args):
            n_steps = Int(String(args[i + 1]))
            i += 2
        elif a == "--replay-episode" and i + 1 < len(args):
            replay_ep = Int(String(args[i + 1]))
            i += 2
        elif a == "--open-loop":
            open_loop = True
            i += 1
        elif a == "--help" or a == "-h":
            _usage()
            return
        else:
            _usage()
            raise Error("unrecognised argument: " + a)
    if ckpt.byte_length() == 0 or not Path(ckpt).exists():
        _usage()
        raise Error("--ckpt: no such checkpoint: " + ckpt)

    print("=" * 66)
    print("so101_tower — policy probe:", ckpt)
    print("=" * 66)
    var agent: Agent = SAC["cpu", OBS_DIM, ACT, BATCH, CAP, HIDDEN](
        action_scale=1.0, learning_starts=0,
    )
    agent.load(ckpt)

    var ctx = DeviceContext()
    var env = E(ctx)
    if replay_ep >= 0:
        if demos.byte_length() == 0:
            raise Error("--replay-episode needs --demos")
        var d = read_demo_file(demos)
        print("-- 3. replay of demo episode", replay_ep, "of", demos,
              "(its seed = --seed + episode)")
        replay_episode(agent, env, d, replay_ep, seed, open_loop)
        return

    if demos.byte_length() > 0:
        print("-- 1. teacher forcing on", demos)
        var d = read_demo_file(demos)
        teacher_forcing(agent, d)

    print("-- 2. closed loop,", n_episodes, "episodes from seed", seed, ",",
          n_steps, "steps")
    closed_loop(agent, env, n_episodes, seed, n_steps)
