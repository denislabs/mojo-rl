# +--------------------------------------------------------------------------+ #
# | The real leader arm drives the SIMULATED SO-101
# +--------------------------------------------------------------------------+ #
"""Read the physical leader over serial, command the physics3d model, render.

    pixi run build-imgui          # ONCE
    pixi run build-serial         # ONCE
    pixi run mojo run -I . examples/so101/teleop_sim.mojo

Pick "policy" in the viewer's drive combo to hand control to the arm; the
other drive modes still work, so you can compare a swept joint against a
hand-moved one without restarting.

**Episodes never time out here** and the reset noise is off — see
`SoArm101TeleopConfig`. Press **`n`** in the window for a fresh episode and a
fresh reach target. ⚠ A reset still folds the arm to HOME before the leader
pulls it back, because the reset hook is static and cannot see the leader.

**WHAT THIS IS FOR — and it is not a demo.** `docs/SO101_SERIAL_LAYER.md` §4
proves our servo ticks equal lerobot's ticks. It proves nothing about whether
a servo angle means the right thing in `so_arm101.xml`. Three unknowns sit
between the two, and `mojo_rl/robot/so101/sim_map.mojo` documents all three:
per-joint **zero**, per-joint **sign**, and a **range** disagreement that is
already measurable. Moving the real arm by hand makes a sign error obvious in
one second, where a numeric gate would need you to know the answer first.

⚠ **The mapping starts as the identity and is therefore WRONG.** That is the
point: run it, watch which joints go backwards, and write the signs and
offsets into `SimJointMap`. The startup banner says so, and the sidebar
`status` line keeps saying so until someone measures them.

⚠ THE LEADER IS READ-ONLY HERE. Its torque is disabled at startup so it can
be backdriven by hand, and nothing is ever written to the follower — this
program cannot move a physical arm. That is why it needs no clamp, no arming
sequence and no `soarm-torque-off` afterwards.

⚠ RUN FROM THE REPO ROOT (mesh paths) and ON THE LAPTOP (it opens a window).
SO-101 is the slow one to compile — 33 280 hull vertices.
"""

from std.random import seed

from mojo_rl.envs.dm_control.viewer_core import (
    ActionSource,
    DRIVE_POLICY,
    ViewerState,
    run_view,
)
from mojo_rl.envs.robots.so_arm101 import SoArm101TeleopConfig
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model, SO_ARM101_OBS_DIM
from mojo_rl.nn.constants import DT
from mojo_rl.physics3d.fields import actuator_column
from mojo_rl.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from mojo_rl.render.imgui import imgui_shim_available
from mojo_rl.render.renderer3d import Renderer3D
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name
from mojo_rl.robot.so101.sim_map import SimJointMap
from mojo_rl.utils.fmt import col, fixed

comptime LEADER_PORT = "/dev/cu.usbmodem5B910455171"
comptime SEED: Int = 0


struct LeaderArmSource(ActionSource, Movable):
    """An `ActionSource` whose "policy" is a human moving a real arm.

    Plugs into the EXISTING viewer with no change to `viewer_core` — the
    policy hook already takes an observation and writes actions, and this one
    simply ignores the observation. That is the whole integration.
    """

    var arm: SO101Arm
    var map: SimJointMap
    var _raw: InlineArray[Int32, SO101_N]
    var _last_ok: Int
    var _clamped: InlineArray[Float64, SO101_N]

    def __init__(out self, var port: String) raises:
        # max_step_ticks=0: nothing here ever writes a goal, so the step clamp
        # would only cost an extra round trip per tick.
        self.arm = SO101Arm(port^, max_step_ticks=0)
        self.arm.bus.timeout_ms = 20
        self.arm.set_torque(False)  # backdriven by hand

        var sf = SoArm101Model.make_spec_fields[DType.float64]()
        var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, SO101_N)
        var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, SO101_N)
        var lo = InlineArray[Float64, SO101_N](fill=0.0)
        var hi = InlineArray[Float64, SO101_N](fill=0.0)
        for i in range(SO101_N):
            lo[i] = Float64(lo_col[i])
            hi[i] = Float64(hi_col[i])
        self.map = SimJointMap.identity(lo^, hi^)

        self._raw = InlineArray[Int32, SO101_N](fill=0)
        # -1, not 0: `run_view` prints `status()` ONCE before the first `act`,
        # and "0 of 6 motors answered" there reads as a dead bus when it only
        # means "not read yet".
        self._last_ok = -1
        self._clamped = InlineArray[Float64, SO101_N](fill=0.0)

    # ── ActionSource ───────────────────────────────────────────────────────

    def obs_dim(self) -> Int:
        # Must equal the task's, or `run_view` disables policy mode. The
        # observation is unused; a human is the policy.
        return SO_ARM101_OBS_DIM

    def act_dim(self) -> Int:
        return SO101_N

    def variant_labels(self) -> List[String]:
        var v = List[String]()
        v.append(String("leader arm (live)"))
        return v^

    def choose(mut self, i: Int) raises:
        pass

    def status(self) -> String:
        """One cheap line — reads the CACHED frame, never the bus.

        The trait says this runs every frame; hitting the serial port here
        would double the bus traffic for a label.
        """
        if self._last_ok < 0:
            return String("leader: connected, waiting for the first read")
        if self._last_ok != SO101_N:
            return (
                String("leader: only ")
                + String(self._last_ok)
                + " of 6 motors answered"
            )
        var out = String("")
        var any_clamped = False
        for i in range(SO101_N):
            out += String(joint_name(i)[byte=0:4]) + "=" + fixed(
                self.map.to_sim(self.arm.cal, i, self._raw[i]), 2
            ) + " "
            if self._clamped[i] > 0.0:
                any_clamped = True
        if any_clamped:
            out += " ⚠CLAMPED"
        if not self.map.measured():
            out += "  [identity map — UNMEASURED]"
        return out^

    def act(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        greedy: Bool,
    ) raises:
        """Leader ticks -> model radians. `obs` and `greedy` are unused."""
        var n = self.arm.read_positions(Span(self._raw))
        self._last_ok = n
        if n != SO101_N:
            # Hold the last command rather than driving a half-updated pose —
            # the same rule the hardware teleop follows.
            return
        for i in range(SO101_N):
            self._clamped[i] = self.map.clamped_by(self.arm.cal, i, self._raw[i])
            action_out[i] = Scalar[DT](
                self.map.to_sim(self.arm.cal, i, self._raw[i])
            )


def main() raises:
    seed(SEED)
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    print("opening leader:", LEADER_PORT)
    var src = LeaderArmSource(String(LEADER_PORT))

    print("\n" + src.map.range_report(src.arm.cal))
    print(src.map.describe())
    print(
        "\n⚠ The mapping above is the IDENTITY and has not been measured."
        "\n  Move each joint in turn and watch the simulated arm:"
        "\n    - moves the wrong way   -> flip `sign[i]` in SimJointMap"
        "\n    - offset from the real  -> set `offset_rad[i]`"
        "\n    - pins at a limit       -> the `gap` column above, not a bug\n"
    )

    var names = List[String]()
    names.append(String("so_arm101_reach"))
    var domains = List[String]()
    domains.append(String("so_arm101"))
    var td = List[Int]()
    td.append(0)

    var first = names[0].copy()
    var st = ViewerState(0, DRIVE_POLICY, 1.0, names^, domains^, td^)
    # No timeout: a human driving an arm has no episode. `SoArm101TeleopConfig`
    # removes the task's own 500-step limit; this removes the viewer's.
    # Press `n` in the window for a fresh episode (and a fresh target).
    st.episode_steps = 0
    # Same handoff the dm_control policy viewers use: the source is owned HERE
    # and lent to the loop, so its serial port outlives a task switch.
    var pol = Pointer(to=src).as_unsafe_any_origin()
    run_view[SoArm101Model, SoArm101TeleopConfig, LeaderArmSource](
        first, st, pol
    )

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
