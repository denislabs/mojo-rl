# +--------------------------------------------------------------------------+ #
# | Which serial port is which arm — one answer, resolved at RUNTIME
# +--------------------------------------------------------------------------+ #
"""`follower_port()` / `leader_port()`: the SO-101 bus paths, per machine.

    var port = follower_port(cli_port)   # --port wins, then the env, then the
                                         # platform's own default

⚠⚠ THESE USED TO BE `comptime` CONSTANTS IN TEN FILES. A compile-time
`"/dev/cu.usbmodem5B8E1139971"` names nothing on the Jetson, and being
compile-time it could not be overridden without a rebuild — so the board could
not run a binary the Mac could build, which defeats the point of moving the
stack onto the board at all (`docs/JETSON_DEPLOYMENT.md` §4.1). The rule this
file exists to keep is the one that was broken: THE ARM-TO-PORT MAPPING IS
WRITTEN ONCE.

Resolution order, highest first:

| source             | follower                    | leader                    |
|--------------------|-----------------------------|---------------------------|
| CLI argument       | `--port` (tools that take it) | —                       |
| environment        | `SOARM_FOLLOWER_PORT`       | `SOARM_LEADER_PORT`       |
| default, Linux     | `/dev/soarm_follower`       | `/dev/soarm_leader`       |
| default, macOS     | `/dev/cu.usbmodem5B8E11...` | `/dev/cu.usbmodem5B9104...`|

⚠ THE LINUX DEFAULTS ARE udev SYMLINKS, NOT DEVICE NODES. `/dev/ttyACM0` and
`/dev/ttyACM1` swap between boots — both adapters are CH9102 (`1a86:55d3`), so
VID:PID discriminates nothing and only the SERIAL does.
`/etc/udev/rules.d/99-soarm.rules` keys on it, which is what makes a stable
name possible; see `docs/JETSON_DEPLOYMENT.md` §3. On macOS the `cu.usbmodem`
path already embeds the serial, so it is stable for free.

⚠ THE macOS DEFAULTS ARE AUTHORITATIVE ABOUT WHICH ARM IS WHICH: `5B8E113997`
is the FOLLOWER and `5B910455171` the LEADER. The udev rules were written from
these, not the other way round. Getting the pair backwards means teleop drives
the arm you are holding.
"""

from std.os import getenv
from std.os.path import exists

from std.sys import CompilationTarget


comptime MAC_FOLLOWER_PORT = "/dev/cu.usbmodem5B8E1139971"
comptime MAC_LEADER_PORT = "/dev/cu.usbmodem5B910455171"
comptime LINUX_FOLLOWER_PORT = "/dev/soarm_follower"
comptime LINUX_LEADER_PORT = "/dev/soarm_leader"

comptime FOLLOWER_ENV = "SOARM_FOLLOWER_PORT"
comptime LEADER_ENV = "SOARM_LEADER_PORT"


def default_follower_port() -> String:
    """The follower's path on THIS machine, ignoring flag and environment."""
    comptime if CompilationTarget.is_macos():
        return String(MAC_FOLLOWER_PORT)
    else:
        return String(LINUX_FOLLOWER_PORT)


def default_leader_port() -> String:
    """The leader's path on THIS machine, ignoring flag and environment."""
    comptime if CompilationTarget.is_macos():
        return String(MAC_LEADER_PORT)
    else:
        return String(LINUX_LEADER_PORT)


def follower_port(cli: String = String("")) -> String:
    """Resolve the follower's port: `cli`, else the environment, else default.
    """
    return _resolve(cli, String(FOLLOWER_ENV), default_follower_port())


def leader_port(cli: String = String("")) -> String:
    """Resolve the leader's port: `cli`, else the environment, else default."""
    return _resolve(cli, String(LEADER_ENV), default_leader_port())


def _resolve(cli: String, env_name: String, fallback: String) -> String:
    if cli.byte_length() > 0:
        return cli
    var e = getenv(env_name, String(""))
    if e.byte_length() > 0:
        return e^
    return fallback


def port_refusal(path: String, role: String) -> String:
    """Why this port cannot be a live arm, or "" if it looks usable.

    ⚠ CHECKED BEFORE THE OPEN, because the failure the open produces names only
    the path. On the board a missing `/dev/soarm_follower` almost always means
    the udev rule did not fire — the arm is plugged into a socket the rule does
    not cover, or the rules were never installed — and "no such file" does not
    say that to anyone who has not read §3 of the Jetson document.
    """
    if path.byte_length() == 0:
        return String("no port given for the ") + role
    if exists(path):
        return String("")
    var why = String("the ") + role + " port " + path + " does not exist"
    comptime if not CompilationTarget.is_macos():
        if path.startswith("/dev/soarm_"):
            return (
                why
                + " — that name is a udev symlink, so either the arm is"
                " unplugged or /etc/udev/rules.d/99-soarm.rules did not fire"
                " (docs/JETSON_DEPLOYMENT.md §3). `ls -l /dev/soarm_*` and"
                " `ls /dev/ttyACM*` say which."
            )
    return (
        why
        + " — plug the arm in, or name the right one with the flag or $"
        + (FOLLOWER_ENV if role == "follower" else LEADER_ENV)
    )
