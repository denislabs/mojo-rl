"""Watch a trained SO-ARM101 reach policy — ImGui sidebar, free camera.

    pixi run build-imgui                                              # ONCE
    pixi run mojo run -I . examples/so101/sac_so_arm101_reach_policy_viewer.mojo
    pixi run mojo run -I . examples/so101/sac_so_arm101_reach_policy_viewer.mojo my.ckpt

The interactive counterpart of `sac_so_arm101_reach_eval_cpu.mojo`. That
script prints a mean return over ten episodes and renders with no controls;
this one is the same checkpoint, the same env and the same greedy action with
the viewer's sidebar around it — per-step reward as a live sparkline, the
running return, pause / single-step, screenshot, recording, and a camera the
mouse actually owns.

⚠⚠ **IT OPENS ON THE FREE CAMERA, AND THAT IS THE POINT.** `so_arm101.xml`
ships exactly one camera — `<camera name="wrist_cam">`, bolted to the wrist —
and the model renderer starts at `active_camera = 0`, so every other entry
point into this model opens looking down the gripper at whatever the gripper
happens to face. It whips around with the wrist, and dragging cannot fix it:
a body-attached camera is re-aimed EVERY frame (`model_renderer.render`), so
the mouse fights the model and loses. `ViewerState.free_camera` asks for
dm_control's camera -1 instead, which is the absence of a model camera and
therefore the only one orbit/pan/zoom fully control. Press `1` — or the
sidebar's camera button — for the wrist view when you want it.

WHAT THE SIDEBAR IS FOR HERE. Reward is a shaped `tolerance` in [0, 1] per
step, so the sparkline is the whole task in one glance: it should climb as the
jaw closes on the mocap target and then FLATTEN NEAR 1.0 and stay there. A
policy that reaches and drifts off looks completely different from one that
reaches and holds, and the two can share a mean return. The policy status line
adds the distance the reward is computed from, in millimetres, against the
task's own 20 mm success radius.

⚠ WATCH THE SHAPE, NOT THE NUMBER. An untrained actor once measured 45.9 mean
over 11 episodes here — a shaped reward's floor is not zero, so a plausible
number is not evidence. ⚠ THAT FIGURE IS STALE: it predates the stillness
term, `REWARD_MARGIN` 0.25 -> 0.05 and the normalized action space, all three
of which changed the scale. Re-measure it. A sparkline that PINS near 1.0 is
the thing to look for either way.

⚠ THE DRIVE COMBO STILL OFFERS zero / random / sweep. They drive the MODEL,
not the agent — useful for telling "the policy is bad" apart from "the model
moved", which is the same question `examples/robots/so_arm_viewer_imgui.mojo`
exists to answer. The action here is NORMALIZED [-1, 1] per joint (the env
maps it onto each `ctrlrange`), so `scale` reaches the joint limits at 1.0
rather than meaning a torque; policy mode ignores it entirely.

⚠ CPU PHYSICS + CPU POLICY, on purpose: one arm at 60 Hz needs no GPU and a
256x256 MLP forward pass is free beside the physics step. Run it on the LAPTOP
— it opens an SDL3 window and blocks on it — and FROM THE REPO ROOT, since the
model reaches its meshes by repo-root-relative path.

⚠ SO-101 IS THE SLOW ONE TO COMPILE: 33 280 hull vertices.
"""

from std.pathlib import Path
from std.random import seed
from std.sys import argv

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.data.any_replay import AnyReplay
from mojo_rl.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from mojo_rl.deep_agents.training.blocks import ReplaySampleStep
from mojo_rl.envs.dm_control.viewer_core import (
    ActionSource, DRIVE_POLICY, ViewerState, run_view,
)
from mojo_rl.envs.robots.so_arm101 import SoArm101ReachConfig
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model
from mojo_rl.render.imgui import imgui_shim_available
from mojo_rl.render.renderer3d import Renderer3D
from mojo_rl.utils.fmt import fixed

comptime SEED: Int = 0

comptime OBS_DIM = 21  # qpos(6) + qvel(6) + ee(3) + target(3) + ee_to_target(3)
comptime ACT_DIM = 6
comptime HIDDEN = 256
"""⚠ ARCHITECTURE ONLY, AND IT MUST MATCH THE TRAINER'S — `nn-ckpt v2` loads by
parameter layout, so a different `HIDDEN` is a load error rather than a wrong
policy, which is the good outcome. `BATCH`/`CAP` size a replay buffer this
agent never fills; the checkpoint holds no replay."""
comptime BATCH = 256
comptime CAP = 1000

comptime ACTION_SCALE = 1.0
"""⚠⚠ MUST MATCH `sac_so_arm101_reach_training_gpu.mojo`. The greedy action is
`tanh(mu) * action_scale`, and this env's action space is NORMALIZED — [-1, 1]
per joint, mapped affinely onto each joint's `ctrlrange` by the env
(`SoArmReachConfig.NORMALIZED_ACTIONS`). A mismatched scale does not weaken
the policy, it commands a different pose: at 2.0 the useful band would sit
back inside the tanh rails and half the range would be unreachable."""

comptime CKPT_NAME = "sac_so_arm101_reach.ckpt"
"""What the trainer's `checkpoint_path` writes. A single file, overwritten at
`checkpoint_every` and again at the end — NOT the stamped ladder
`sac_dm_walker_training_gpu.mojo` produces, which is why this viewer's variant
list is one entry rather than twenty rungs."""

comptime TARGET_RADIUS_MM = 20.0
"""`SoArmReachConfig.TARGET_RADIUS` = 0.02 m — the radius inside which
`tolerance` returns 1.0. Repeated here only to label the status line."""


def ckpt_dirs() -> List[String]:
    """Where a checkpoint might live, in probe order — same convention as
    `dm_walker_policy_viewer.mojo`: the trainer writes to the CWD, and the
    ladders on this machine were moved into `checkpoints/`."""
    var d = List[String]()
    d.append(String("checkpoints/"))
    d.append(String(""))
    return d^


# ═══════════════════════════════════════════════════════════════════════════
# the ActionSource — one SAC checkpoint, driving the arm
# ═══════════════════════════════════════════════════════════════════════════


struct ReachPolicy(ActionSource, Movable):
    """A trained SAC actor as the viewer's fourth drive mode.

    Built through the `SAC[...]` preset rather than by naming the nets by hand
    so the parameter layout is the trainer's by construction — the trainer
    spells its actor and critic out (`StochasticActor[...]` / `Sequential`),
    and those are exactly what `SACActorNet` / `SACCriticNet` expand to. Two
    spellings of one architecture; a checkpoint is the gate that catches a
    drift between them.

    ⚠ `learning_starts=0` IS LOAD-BEARING for the non-greedy path.
    `select_action` takes the uniform-random WARMUP branch below that
    threshold, so with the default 1000 the sidebar's "greedy" checkbox would
    be a random-action toggle for the first thousand frames — the trap
    `dm_walker_policy_viewer.mojo` and `collect_walker_sac.mojo` both document.
    """

    var agent: SACAgent[
        "cpu",
        ReplaySampleStep[AnyReplay["cpu", OBS_DIM, ACT_DIM, CAP], BATCH],
        SACActorNet[OBS_DIM, ACT_DIM, HIDDEN],
        SACCriticNet[OBS_DIM, ACT_DIM, HIDDEN],
    ]
    var paths: List[String]
    var labels: List[String]
    var current: Int
    var loaded: Bool
    var step_idx: Int
    var dist: Float64
    """Jaw-to-target distance from the LAST observation, metres.

    Read out of `obs[18..20]` — the `ee_to_target` block — rather than
    recomputed, so the status line reports the number the reward was actually
    computed from and cannot drift from it."""

    def __init__(out self, var explicit: String) raises:
        self.agent = SAC["cpu", OBS_DIM, ACT_DIM, BATCH, CAP, HIDDEN](
            action_scale=ACTION_SCALE,
            learning_starts=0,
        )
        self.paths = List[String]()
        self.labels = List[String]()
        self.current = -1
        self.loaded = False
        self.step_idx = 1
        self.dist = -1.0

        # An explicit argv path is taken AS GIVEN and not probed: naming a file
        # and being handed a different one that happened to exist is worse than
        # a clean "not on disk".
        if explicit:
            self.paths.append(explicit.copy())
            self.labels.append(
                explicit.copy() if Path(explicit).exists()
                else explicit + " (missing)"
            )
        else:
            var dirs = ckpt_dirs()
            for d in range(len(dirs)):
                var cand = dirs[d] + CKPT_NAME
                if Path(cand).exists():
                    self.paths.append(cand.copy())
                    self.labels.append(cand.copy())
        if len(self.paths) == 0:
            print("  ⚠ no checkpoint found. Looked for:")
            var dirs = ckpt_dirs()
            for d in range(len(dirs)):
                print("     " + dirs[d] + CKPT_NAME)
            print("    Train one first:")
            print(
                "     pixi run -e nvidia mojo run -I ."
                " examples/so101/sac_so_arm101_reach_training_gpu.mojo"
            )
        else:
            print("  checkpoint:", self.paths[0])

    # ─── ActionSource ───────────────────────────────────────────────────

    def obs_dim(self) -> Int:
        return OBS_DIM

    def act_dim(self) -> Int:
        return ACT_DIM

    def variant_labels(self) -> List[String]:
        return self.labels.copy()

    def choose(mut self, i: Int) raises:
        """Load variant `i`, or raise WITHOUT SIDE EFFECTS.

        All-or-nothing for the reason `WalkerLadder.choose` spells out: marking
        a variant selected and then failing to load it leaves the sidebar
        naming one policy while another drives.
        """
        if i < 0 or i >= len(self.paths):
            raise Error("variant out of range: " + String(i))
        self.agent.load(self.paths[i])
        self.current = i
        self.loaded = True

    def status(self) -> String:
        if self.current < 0 or not self.loaded:
            return String("no checkpoint loaded — drive modes still work")
        if self.dist < 0.0:
            return self.labels[self.current] + String(" loaded")
        var mm = self.dist * 1000.0
        var mark = String(" ✓ inside") if mm <= TARGET_RADIUS_MM else String("")
        return (
            String("ee->target ") + fixed(mm, 1) + " mm / "
            + fixed(TARGET_RADIUS_MM, 0) + " mm" + mark
        )

    def act(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        greedy: Bool,
    ) raises:
        if not self.loaded:
            # Zero, not an untrained forward pass: a jittering arm would read
            # as "this checkpoint is bad" rather than "there is no checkpoint".
            for j in range(ACT_DIM):
                action_out[j] = Scalar[DT](0)
            return
        # obs[18..20] is `ee_to_target` — see `so_arm_reach_config`'s
        # `custom_extract_obs_cpu`, whose ordering this indexes into.
        var dx = Float64(obs[18])
        var dy = Float64(obs[19])
        var dz = Float64(obs[20])
        self.dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if greedy:
            self.agent.select_greedy_action(obs, action_out)
        else:
            self.agent.select_action(obs, action_out, self.step_idx)
            self.step_idx += 1


def main() raises:
    seed(SEED)
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    var args = argv()
    var explicit = String(args[1]) if len(args) > 1 else String("")

    print("=" * 66)
    print("SO-ARM101 reach — trained policy, ImGui viewer")
    print("=" * 66)
    var pol_src = ReachPolicy(explicit^)
    print(
        "  reward is a shaped `tolerance` in [0, 1] per step: the sparkline"
        "\n  should climb as the jaw closes and then FLATTEN near 1.0."
        "\n  camera: opens FREE (mouse-controlled). `1` selects the model's"
        "\n  wrist_cam, which is an onboard view and re-aimed every frame."
        "\n  `n` starts a fresh episode and draws a fresh target."
    )
    print("=" * 66)

    var names = List[String]()
    names.append(String("so_arm101_reach"))
    var domains = List[String]()
    domains.append(String("so_arm101"))
    var td = List[Int]()
    td.append(0)

    var first = names[0].copy()
    # scale 1.0 is inert in policy mode and is the sane starting point for the
    # other three; see `ui_drive_controls`.
    var st = ViewerState(0, DRIVE_POLICY, 1.0, names^, domains^, td^)
    st.free_camera = True
    # The task ends itself at `SoArmReachConfig.MAX_STEPS` (500 control steps),
    # so the viewer's own 1000-step limit would only ever fire on a config that
    # removed that one. Left at the default rather than duplicated here.

    # ⚠ THE POLICY OUTLIVES THE ENV, and must: it holds the loaded weights, so
    # a task switch reuses them instead of re-reading the checkpoint.
    var pol = Pointer(to=pol_src).as_unsafe_any_origin()
    run_view[SoArm101Model, SoArm101ReachConfig, ReachPolicy](first, st, pol)
    _ = pol_src  # lifetime extender for `pol`

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
