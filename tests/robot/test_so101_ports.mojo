# +--------------------------------------------------------------------------+ #
# | Which serial port is which arm — the resolution order, and the refusals
# +--------------------------------------------------------------------------+ #
"""Gate `robot/so101/ports.mojo`.

    pixi run mojo run -I . tests/robot/test_so101_ports.mojo

No hardware: every path here is chosen to NOT exist, which is exactly the
interesting case — the arm is named before it is opened.

⚠ WHY THIS EXISTS. The ports were `comptime` constants in ten files, so the
Jetson could not run a binary the Mac could build. The value of replacing them
is entirely in the PRECEDENCE — a CLI argument that loses to an environment
variable would silently run the wrong arm on a machine that has both plugged
in — and precedence is the one thing a compile does not check.

⚠ THE ENVIRONMENT LEG RUNS LAST, and that ordering is load-bearing: `setenv`
changes this process for good, so every check that depends on the DEFAULT has
to have run already. Putting it first would make the default checks pass
against the environment's value and prove nothing.
"""

from std.os import setenv
from std.sys import CompilationTarget

from mojo_rl.robot.so101.ports import (
    LINUX_FOLLOWER_PORT,
    LINUX_LEADER_PORT,
    MAC_FOLLOWER_PORT,
    MAC_LEADER_PORT,
    default_follower_port,
    default_leader_port,
    follower_port,
    leader_port,
    port_refusal,
)


def main() raises:
    print("[so101-ports] gate")
    var n = 0

    # ── the platform default ─────────────────────────────────────────────
    # ⚠ BOTH BRANCHES ARE NAMED, so this test fails on the machine where the
    # mapping is wrong rather than only on the one it was written on.
    comptime if CompilationTarget.is_macos():
        if default_follower_port() != String(MAC_FOLLOWER_PORT):
            raise Error("macOS must default the follower to the cu.usbmodem path")
        if default_leader_port() != String(MAC_LEADER_PORT):
            raise Error("macOS must default the leader to the cu.usbmodem path")
    else:
        if default_follower_port() != String(LINUX_FOLLOWER_PORT):
            raise Error("Linux must default the follower to /dev/soarm_follower")
        if default_leader_port() != String(LINUX_LEADER_PORT):
            raise Error("Linux must default the leader to /dev/soarm_leader")
    n += 2
    print(
        "  default: follower " + default_follower_port() + ", leader "
        + default_leader_port()
    )

    # ⚠⚠ THE TWO ARMS MUST NOT RESOLVE TO THE SAME PORT. Teleop reads one and
    # writes the other; one path for both means the leader is driven by the
    # policy and the operator's arm fights it.
    if default_follower_port() == default_leader_port():
        raise Error("the follower and the leader must not be the same port")
    n += 1

    # ── the CLI wins ─────────────────────────────────────────────────────
    if follower_port(String("/dev/explicit")) != "/dev/explicit":
        raise Error("an explicit follower port must win")
    if leader_port(String("/dev/explicit")) != "/dev/explicit":
        raise Error("an explicit leader port must win")
    # An empty argument is "not given", not "the empty port".
    if follower_port(String("")) != default_follower_port():
        raise Error("an empty argument must fall through to the default")
    n += 3
    print("  precedence: an explicit argument wins; \"\" falls through")

    # ── refusals ─────────────────────────────────────────────────────────
    if port_refusal(String(""), String("follower")).byte_length() == 0:
        raise Error("an empty port must be refused")
    var missing = port_refusal(String("/dev/definitely_not_here"), String("follower"))
    if missing.byte_length() == 0:
        raise Error("a path that does not exist must be refused")
    # ⚠ THE MESSAGE HAS TO NAME THE PATH. "no such file" with no path is the
    # failure this replaced.
    if "/dev/definitely_not_here" not in missing:
        raise Error("the refusal must name the port it refused")
    if "follower" not in missing:
        raise Error("the refusal must name the ROLE, so an operator knows which arm")
    n += 4
    print("  refusal: " + missing[byte=0:58] + " ...")

    # A port that DOES exist is not refused. `/dev/null` is a device node on
    # every platform this runs on — not an arm, but this check is about the
    # existence test, and nothing is opened.
    if port_refusal(String("/dev/null"), String("follower")).byte_length() != 0:
        raise Error("an existing path must not be refused")
    n += 1
    print("  an existing device node is accepted (nothing is opened)")

    # ── the environment, BELOW the CLI and ABOVE the default ─────────────
    # ⚠ LAST, ON PURPOSE — see the header. Nothing after this line may assume
    # the platform default.
    var before = default_follower_port()
    _ = setenv("SOARM_FOLLOWER_PORT", "/dev/from_env", True)
    _ = setenv("SOARM_LEADER_PORT", "/dev/leader_from_env", True)
    if follower_port() != "/dev/from_env":
        raise Error("$SOARM_FOLLOWER_PORT must beat the platform default")
    if leader_port() != "/dev/leader_from_env":
        raise Error("$SOARM_LEADER_PORT must beat the platform default")
    if follower_port(String("/dev/explicit")) != "/dev/explicit":
        raise Error("an explicit argument must still beat the environment")
    # ⚠ AND THE DEFAULT IS STILL THE DEFAULT. `default_follower_port` reports
    # the platform's answer, not the resolved one; a caller printing "the
    # default is X" while running against Y would be worse than silent.
    if default_follower_port() != before:
        raise Error("the environment must not change the PLATFORM default")
    n += 4
    print("  precedence: $SOARM_*_PORT beats the default, loses to an argument")

    print("  " + String(n) + " checks, 0 failures")
    print("[PASS] so101-ports")
