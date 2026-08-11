"""TD-MPC2 checkpoint driving dm_control `walker` — watch the gait.

    pixi run build-imgui                                        # ONCE
    pixi run -e apple mojo run -I . examples/dm_control/tdmpc2_dm_walker_policy_viewer.mojo
    pixi run -e apple mojo run -I . examples/dm_control/tdmpc2_dm_walker_policy_viewer.mojo walker_run

argv picks the task that opens first, then the drive mode (zero | random |
sweep | policy) and an action scale; every task and mode stays selectable in
the window. Defaults: `walker_walk`, `policy`.

The TD-MPC2 counterpart of `dm_walker_policy_viewer.mojo` (the SAC ladder one),
simplified: TD-MPC2 writes ONE checkpoint per run, not a 20-rung ladder — the
driver overwrites `checkpoint_path` on every save — so there is no rung axis to
sweep. The axis that replaces it is the ACTING MODE, and that one matters more.

## ⚠ MPC vs the policy prior — the two are NOT the same agent

TD-MPC2 acts by PLANNING: MPPI rolls candidate action sequences through the
learned world model and ranks them by predicted reward + terminal Q. The policy
network `π` is a prior that seeds and regularizes that search; it is not the
deployed controller. `train_batched`'s eval inherits the training `USE_MPC`, so
a reported `eval/mean_return` from an MPC run is the PLANNER's score.

Both are offered here as variants, and they will not look alike:

  * `MPC`   — `select_action_mpc`, the thing the eval return measured. GPU
    only, and it runs the full MPPI budget PER FRAME, so expect single-digit
    to low-tens Hz on Apple/Metal, not 60. Use pause/step for a close look.
  * `prior` — `a = π(encode(obs))`, real-time, and typically WEAKER. If the
    prior looks much worse than the eval number led you to expect, that is the
    expected MPC-vs-prior gap, not a broken checkpoint.

Reading the two against each other is the point: a large gap means the world
model + value function are carrying the performance; a small one means π has
absorbed the plan and could be deployed without a planner.

## ⚠ THE DIMS BELOW MUST MATCH THE TRAINING SCRIPT

`load_state` restores parameter slabs BY NAME. Architecture dims (ENC / LATENT
/ MLP / BINS / SN) are part of those shapes, so a mismatch fails loudly — but
`VMIN`/`VMAX`/`H` and the MPPI budget are NOT stored, and a mismatch there
loads clean and plans differently. Keep them in step with
`tdmpc2_dm_walker_batched_gpu.mojo` (or `tdmpc2_dm_walker_gpu.mojo`).

`B` and `CAP` are deliberately TINY here: they size the training batch scratch
and the replay ring, neither of which a viewer uses, and neither is in the
checkpoint. The 1 M-slot ring the training script asks for would cost ~136 MB
of host memory to open a window with.

## ⚠ THE CHECKPOINT IS NOT IN THE REPO

`CKPT_DIRS` probes `checkpoints/` then the project root, for both the batched
and single-env filenames of all three tasks. A variant whose file is absent is
labelled `(missing)` in the combo and left un-driven rather than silently
driving an uninitialised net.

⚠ CPU PHYSICS + GPU AGENT. The viewer's env is the CPU float32 path while the
checkpoint trained on the GPU batched path, so treat small differences as path
noise and only large ones as a finding. The agent keeps its OWN DeviceContext
(the viewer builds one for rendering); nothing crosses between them — the
`ActionSource` interface is host `List`s.

⚠ MPPI WARM-START SURVIVES AN ENV RESET. `ActionSource` has no reset hook, so
the planner's shifted mean carries one step across a reset before the first
refit corrects it. Harmless for watching; do not read the first frame of an
episode as policy behaviour.
"""

from std.pathlib import Path
from std.random import seed
from std.sys import argv
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.config import TDMPC2
from mojo_rl.deep_agents.tdmpc2.agent import TDMPC2Agent
from mojo_rl.envs.dm_control.viewer_core import (
    ActionSource, ViewerState, run_view, task_index, parse_drive, DRIVE_POLICY,
)
from mojo_rl.render.imgui import imgui_shim_available
from mojo_rl.render.renderer3d import Renderer3D

from mojo_rl.envs.dm_control.walker.walker_xml import DMWalkerModel
from mojo_rl.envs.dm_control.walker.walker_config import DMWalkerConfig

comptime SEED: Int = 0

comptime OBS_DIM = DMWalkerModel.OBS_DIM      # 24
comptime ACT_DIM = DMWalkerModel.ACTION_DIM   #  6

# ══ ARCHITECTURE — MUST MATCH THE TRAINING SCRIPT ═══════════════════════
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime H = 3
# MPPI budget — also the training script's. Lower these if the MPC variant is
# too slow to watch; the gait degrades gracefully as the budget shrinks, which
# is itself worth seeing.
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

# Viewer-only sizes (see the header): never trains, never fills a replay.
comptime B = 8
comptime CAP = 1024

comptime AgentT = TDMPC2Agent[
    "gpu", OBS_DIM, ENC, ACT_DIM, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H,
    CAP, MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
]


def ckpt_dirs() -> List[String]:
    """Where a checkpoint might live, in probe order (training writes to CWD;
    the ladders on this machine were moved into `checkpoints/`)."""
    var d = List[String]()
    d.append(String("checkpoints/"))
    d.append(String(""))
    return d^


def ckpt_names() -> List[String]:
    """Every filename the two TD-MPC2 walker training scripts can write.

    ⚠ POSITIONALLY COUPLED to the `_mpc` / `_mpcoff` suffix those scripts pick
    from their own `USE_MPC`. A run trained MPC-off still loads and still
    plans here — the checkpoint holds no acting mode — but its world model was
    never shaped by planning data, so expect the MPC variant to disappoint.
    """
    var tasks = List[String]()
    tasks.append(String("stand"))
    tasks.append(String("walk"))
    tasks.append(String("run"))

    var n = List[String]()
    for t in range(len(tasks)):
        n.append("tdmpc2_dm_walker_batched_" + tasks[t] + "_mpc.ckpt")
        n.append("tdmpc2_dm_walker_batched_" + tasks[t] + "_mpcoff.ckpt")
        n.append("tdmpc2_dm_walker_" + tasks[t] + "_mpc.ckpt")
        n.append("tdmpc2_dm_walker_" + tasks[t] + "_mpcoff.ckpt")
    return n^


# ═══════════════════════════════════════════════════════════════════════════
# the ActionSource — one checkpoint, two acting modes
# ═══════════════════════════════════════════════════════════════════════════


struct TDMPC2Walker(ActionSource, Movable, Deinitable):
    """A TD-MPC2 checkpoint as a selectable policy, in both acting modes.

    ONE agent, reloaded in place: `load_state` overwrites the world model +
    Q ensemble + policy, so switching checkpoint is a few milliseconds and the
    nets (and the MPPI planner's device scratch) are built exactly once.

    VARIANTS ARE FLAT AND CHECKPOINT-MAJOR — `<ckpt> MPC`, `<ckpt> prior`,
    next checkpoint, … — so the sidebar's +/- buttons toggle the ACTING MODE
    on one checkpoint, which is the comparison worth stepping through.
    """

    var agent: AgentT
    var paths: List[String]
    """Resolved path per variant, or "" for a checkpoint no directory holds."""
    var labels: List[String]
    var use_mpc: List[Bool]
    """Acting mode per variant, parallel to `paths`."""
    var current: Int
    var loaded: Bool

    def __init__(out self) raises:
        # The agent owns its own context — `run_view` builds a separate one for
        # the renderer. Safe because the two never share a buffer: this trait
        # speaks host Lists.
        var ctx = DeviceContext()
        self.agent = TDMPC2[
            "gpu", OBS_DIM, ACT_DIM, B, CAP, ENC, LATENT, MLP, BINS, SN,
            VMIN, VMAX, H, MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
        ](ctx=ctx, action_scale=Scalar[DT](1.0), learning_starts=0)
        self.paths = List[String]()
        self.labels = List[String]()
        self.use_mpc = List[Bool]()
        self.current = -1
        self.loaded = False

        var names = ckpt_names()
        var dirs = ckpt_dirs()
        var n_found = 0
        for i in range(len(names)):
            var found = String("")
            for d in range(len(dirs)):
                var cand = dirs[d] + names[i]
                if Path(cand).exists():
                    found = cand
                    break
            # A missing checkpoint contributes NO variants at all — with 12
            # candidate filenames x 2 modes, listing the absent ones would bury
            # the one that exists in 22 lines of "(missing)".
            if not found:
                continue
            n_found += 1
            var short = names[i].replace("tdmpc2_dm_walker_", "").replace(
                ".ckpt", ""
            )
            self.paths.append(found)
            self.labels.append(short + " MPC")
            self.use_mpc.append(True)
            self.paths.append(found)
            self.labels.append(short + " prior")
            self.use_mpc.append(False)

        print("  checkpoints found:", n_found, "→", len(self.paths), "variants")
        for i in range(len(self.labels)):
            print("    ", self.labels[i], " ←", self.paths[i])
        if n_found == 0:
            print("  ⚠ NO checkpoint found. Looked in checkpoints/ and ./ for")
            print("      tdmpc2_dm_walker_batched_<task>_{mpc,mpcoff}.ckpt")
            print("      tdmpc2_dm_walker_<task>_{mpc,mpcoff}.ckpt")
            print("    Train one first:")
            print(
                "      pixi run -e nvidia mojo run -I ."
                " examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo"
            )
            print("    The walker will stand inert until then.")

    # ─── ActionSource ───────────────────────────────────────────────────

    def obs_dim(self) -> Int:
        return OBS_DIM

    def act_dim(self) -> Int:
        return ACT_DIM

    def variant_labels(self) -> List[String]:
        return self.labels.copy()

    def choose(mut self, i: Int) raises:
        """Load variant `i`, or raise WITHOUT SIDE EFFECTS — a failed switch
        leaves the previously loaded checkpoint driving, so the sidebar, the
        status line and the torque never disagree about which policy is live.
        """
        if i < 0 or i >= len(self.paths):
            raise Error("variant out of range: " + String(i))
        if not self.paths[i]:
            raise Error("checkpoint not on disk: " + self.labels[i])
        # Re-reading the same file when only the MODE changed is a few ms and
        # keeps this branch-free; the planner's warm-start is reset either way.
        self.agent.load_state(self.paths[i])
        self.agent.mpc_start_episode()
        self.current = i
        self.loaded = True

    def status(self) -> String:
        if self.current < 0:
            return String("no ckpt selected")
        if not self.loaded:
            return String("MISSING ") + self.labels[self.current]
        return self.labels[self.current] + String(" loaded")

    def act(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        greedy: Bool,
    ) raises:
        if not self.loaded:
            # Zero torque rather than an untrained net: a jittering walker
            # reads as "this checkpoint is bad", not "there is no checkpoint".
            for j in range(ACT_DIM):
                action_out[j] = Scalar[DT](0)
            return
        if self.use_mpc[self.current]:
            # `greedy` → the eval-time planner (no exploration noise on the
            # selected action), which is what produced the eval return.
            self.agent.select_action_mpc(obs, action_out, explore=not greedy)
        else:
            self.agent.select_action(obs, action_out, explore=not greedy)


# ═══════════════════════════════════════════════════════════════════════════
# the three-task front end
# ═══════════════════════════════════════════════════════════════════════════


def task_names() -> List[String]:
    """⚠ POSITIONALLY COUPLED TO `dispatch` — index i here is the arm
    `st.task == i` there."""
    var t = List[String]()
    t.append(String("walker_stand"))
    t.append(String("walker_walk"))
    t.append(String("walker_run"))
    return t^


def domain_names() -> List[String]:
    var d = List[String]()
    d.append(String("walker"))
    return d^


def task_domain() -> List[Int]:
    var t = List[Int]()
    for _ in range(3):
        t.append(0)
    return t^


def dispatch(
    mut st: ViewerState, policy: Pointer[TDMPC2Walker, MutAnyOrigin]
) raises:
    """Run whichever task `st.task` names, and return when it wants another.

    ⚠ All three tasks share ONE model and ONE observation — only the REWARD
    differs — so any checkpoint drives any of the three envs. The sidebar
    deliberately lets the checkpoint's task and the env's task disagree:
    "drive the walk policy, read the stand reward" is a real question and the
    reward sparkline answers it. The MOVE_SPEED values are dm_control's and
    must match the training script's.
    """
    var name = task_names()[st.task]
    if st.task == 0:
        run_view[DMWalkerModel, DMWalkerConfig[0.0], TDMPC2Walker](
            name, st, policy
        )
    elif st.task == 1:
        run_view[DMWalkerModel, DMWalkerConfig[1.0], TDMPC2Walker](
            name, st, policy
        )
    elif st.task == 2:
        run_view[DMWalkerModel, DMWalkerConfig[8.0], TDMPC2Walker](
            name, st, policy
        )
    else:
        print("unknown task index:", st.task)
        st.quit = True


def main() raises:
    seed(SEED)
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    var args = argv()
    var start = String(args[1]) if len(args) > 1 else String("walker_walk")
    var task = task_index(start, task_names())
    if task < 0:
        print("unknown task:", start, "— this front end registers:")
        var names = task_names()
        for i in range(len(names)):
            print("   ", names[i])
        return

    var drive = parse_drive(String(args[2])) if len(args) > 2 else DRIVE_POLICY
    var scale = Float64(1.0)
    if len(args) > 3:
        try:
            scale = Float64(String(args[3]))
        except:
            print("bad scale, using 1.0")

    print("=" * 66)
    print("dm_control walker x 3 tasks + TD-MPC2 checkpoint")
    print("=" * 66)
    print("  MPPI budget:", MPC_SAMPLES, "+", MPC_PI_TRAJS, "trajs,",
          MPC_ITERS, "iters, horizon", H)
    print("  ⚠ the MPC variant plans EVERY frame — expect a few Hz, not 60")
    var walker = TDMPC2Walker()

    var st = ViewerState(
        task, drive, scale, task_names(), domain_names(), task_domain()
    )
    # Open on variant 0 — the first checkpoint found, in MPC mode, i.e. the
    # configuration whose return the training run actually reported.
    st.policy_variant = Int32(0)

    # ⚠ THE POLICY OUTLIVES EVERY ENV, and must: it holds the loaded weights
    # and the planner's device scratch, so a task switch reuses them instead of
    # re-reading the checkpoint and re-allocating the MPPI buffers.
    var pol = Pointer(to=walker).as_unsafe_any_origin()
    while not st.quit:
        dispatch(st, pol)
    _ = walker  # lifetime extender for `pol`

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
