# +--------------------------------------------------------------------------+ #
# | `--devices`: indices, device paths, or a mix — one parser for all of them
# +--------------------------------------------------------------------------+ #
"""Gate `vision/camera_thread.mojo`: `parse_camera_specs`, `camera_spec_is_path`
and the fourcc packing that carries a negotiated format off the camera thread.

    pixi run mojo run -I . tests/vision/test_camera_spec.mojo

No camera is opened. Everything here is the string layer that decides WHICH
camera gets opened, which is the layer that can be wrong silently: opening the
wrong camera is not a crash, it is a well-formed observation of the wrong
scene, and the policy acts on it confidently.

⚠ WHY THIS EXISTS. `record`, `record_ui`, `act_so101_deploy_real` and
`record_budget` had FOUR hand-rolled `--devices` splitters between them, and
they disagreed: one accepted a trailing comma that another turned into a camera
named "". That is `_a_rule_written_inline_twice_drifts` with four copies.
"""

from std.sys import CompilationTarget

from mojo_rl.vision.camera_thread import (
    CameraReader,
    _pack_fourcc,
    default_fourcc,
    _unpack_fourcc,
    camera_spec_is_path,
    parse_camera_specs,
)


def main() raises:
    print("[camera-spec] gate")
    var n = 0

    # ── index vs path ────────────────────────────────────────────────────
    if camera_spec_is_path(String("0")):
        raise Error("'0' is an index")
    if camera_spec_is_path(String("12")):
        raise Error("'12' is an index")
    if not camera_spec_is_path(String("/dev/soarm_cam_wrist")):
        raise Error("an absolute device path is a path")
    if not camera_spec_is_path(String("video0")):
        raise Error("anything not all-digits is a path")
    # ⚠ THE RULE IS "NOT ALL DIGITS", NOT "STARTS WITH /". A relative path is
    # still a path, and erring this way makes the failure name what the
    # operator typed instead of silently opening camera 0.
    if not camera_spec_is_path(String("0a")):
        raise Error("'0a' must not be read as index 0")
    n += 5
    print("  index vs path: digits are an index, anything else is a path")

    # ── the split ────────────────────────────────────────────────────────
    var two = parse_camera_specs(String("0,1"))
    if len(two) != 2 or two[0] != "0" or two[1] != "1":
        raise Error("'0,1' must give exactly two specs")
    var paths = parse_camera_specs(
        String("/dev/soarm_cam_overhead,/dev/soarm_cam_wrist")
    )
    if len(paths) != 2 or paths[0] != "/dev/soarm_cam_overhead":
        raise Error("two paths must survive the split, in order")
    # A mix is legal: one camera by path, one by index.
    var mixed = parse_camera_specs(String("/dev/soarm_cam_overhead,1"))
    if len(mixed) != 2 or mixed[1] != "1":
        raise Error("a mix of a path and an index must parse")
    n += 3

    # ⚠ THE DISAGREEMENT THAT MOTIVATED ONE PARSER: a trailing comma, and
    # spaces around an entry. Both used to produce a camera named "".
    var trailing = parse_camera_specs(String("0,1,"))
    if len(trailing) != 2:
        raise Error("a trailing comma must not add an empty camera")
    var spaced = parse_camera_specs(String(" 0 , 1 "))
    if len(spaced) != 2 or spaced[0] != "0" or spaced[1] != "1":
        raise Error("spaces around an entry must be stripped, not kept")
    n += 2
    print("  split: order kept; a trailing comma and stray spaces add nothing")

    # ── refusals ─────────────────────────────────────────────────────────
    # ⚠ AN EMPTY LIST MUST RAISE, NOT RETURN EMPTY. Returning empty lands as
    # "the policy takes 2 cameras but 0 were given" several checks later,
    # naming the wrong thing.
    var refused = 0
    for bad in [String(""), String(","), String("  "), String(",,,")]:
        try:
            var got = parse_camera_specs(bad)
            print("  ⚠ '" + bad + "' gave " + String(len(got)) + " spec(s)")
        except:
            refused += 1
    if refused != 4:
        raise Error(
            "every input naming no camera must be refused; "
            + String(4 - refused) + " were not"
        )
    n += 4
    print("  refusal: '', ',', '  ' and ',,,' all raise")

    # ── the fourcc cell ──────────────────────────────────────────────────
    # It crosses a thread boundary as an Int64 because the camera worker can
    # neither raise nor print. A round trip that loses the code would report
    # the wrong pixel format, which is the one thing it exists to report.
    if _unpack_fourcc(_pack_fourcc(String("MJPG"))) != "MJPG":
        raise Error("MJPG must survive the pack/unpack round trip")
    if _unpack_fourcc(_pack_fourcc(String("YUYV"))) != "YUYV":
        raise Error("YUYV must survive the pack/unpack round trip")
    # ⚠ AND THE TWO MUST NOT PACK THE SAME. A round trip through a constant
    # would pass both checks above.
    if _pack_fourcc(String("MJPG")) == _pack_fourcc(String("YUYV")):
        raise Error("two different formats must not pack to the same value")
    if _unpack_fourcc(Int64(0)) != "":
        raise Error("0 means 'the device reported nothing', not a format")
    if _pack_fourcc(String("MJP")) != Int64(0):
        raise Error("a code that is not four characters is not a format")
    n += 5
    print(
        "  fourcc: MJPG/YUYV round-trip and differ ("
        + String(_pack_fourcc(String("MJPG"))) + " vs "
        + String(_pack_fourcc(String("YUYV"))) + ")"
    )

    # ── the requested format ─────────────────────────────────────────────
    # ⚠ NO CAMERA IS OPENED. This gates the REQUEST the reader will make,
    # which is the part that was silently absent: every V4L2 camera came up
    # YUYV because nothing asked for anything.
    comptime if CompilationTarget.is_macos():
        if default_fourcc() != "":
            raise Error("macOS opens by index; it must request no format")
    else:
        if default_fourcc() != "MJPG":
            raise Error(
                "a path-opened V4L2 camera must ask for MJPG — YUYV at 30 fps"
                " is 147 Mbit/s per camera on a shared USB 2.0 bus"
            )
    n += 1

    # `none` is the escape hatch, and it must differ from the empty default.
    var free = CameraReader.at_path(
        String("/dev/nope"), 640, 480, 30.0, fourcc=String("none")
    )
    if free.fourcc != "":
        raise Error("`none` must leave the device's own format alone")
    var defaulted = CameraReader.at_path(String("/dev/nope"), 640, 480, 30.0)
    if defaulted.fourcc != default_fourcc():
        raise Error("an unspecified format must become the platform default")
    # ⚠ AND AN INDEX-OPENED CAMERA ASKS FOR NOTHING: the default is about
    # V4L2's per-format frame-size table, not about an AVFoundation index.
    #
    # ⚠⚠ THIS LEG IS VACUOUS WHERE `default_fourcc()` IS EMPTY — that is,
    # on macOS, where both sides are "" and deleting the guard changes
    # nothing. Verified by mutation: removing `and path.byte_length() > 0`
    # SURVIVES here and is killed only on Linux. It is kept because the board
    # is where it decides something, and stated because a check that cannot
    # fail on the machine you are running is not a check on that machine.
    var by_index = CameraReader(0, 640, 480, 30.0)
    if by_index.fourcc != "":
        raise Error("an index-opened camera must not acquire a format request")
    n += 3
    print(
        "  format: `" + default_fourcc() + "` by default for a path, `none`"
        " opts out, an index asks for nothing"
    )

    print("  " + String(n) + " checks, 0 failures")
    print("[PASS] camera-spec")
