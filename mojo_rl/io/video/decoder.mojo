# +--------------------------------------------------------------------------+ #
# | Sequential video decode, without Python and without an FFI struct layout
# +--------------------------------------------------------------------------+ #
"""Decode an mp4 to RGB24 frames by piping `ffmpeg`'s raw output.

    var d = VideoDecoder(path)                  # 640x480, say
    var row = List[UInt8](unsafe_uninit_length = d.frame_bytes)
    while d.next_into(mptr(row)):
        ...                                     # d.index-1 frames consumed

## Why a pipe and not libavcodec

`libavformat`/`libavcodec`/`libswscale` are all in the pixi environment with
plain C ABIs, so binding them looks like the obvious move. It is not, for one
reason: **every useful FFmpeg call needs a struct field.** `AVFrame.data[0]`,
`AVFrame.linesize[0]`, `AVCodecContext.pix_fmt`, `AVStream.codecpar` — none of
them have accessor functions, so an FFI binding has to hardcode byte offsets
into public structs that change between major sonames. A wrong offset does not
crash; it reads a neighbouring field and decodes plausible-looking garbage.

Piping `-f rawvideo -pix_fmt rgb24` moves all of that inside a binary that
already agrees with its own headers. **Verified byte-identical to `imageio`'s
FFMPEG plugin** on the first five frames of a LeRobot recording — which is
unsurprising, since imageio shells out to the same `ffmpeg` with the same
arguments. `tests/io/test_video_decode.mojo` re-checks that over a whole file.

⚠ `-fps_mode passthrough` IS LOAD-BEARING, and it must come AFTER `-i`
(it is an output option; before `-i` ffmpeg rejects it outright). Without it
ffmpeg is free to duplicate or drop frames to hit a constant output rate, and
the frame at index `i` in the pipe stops being the frame at index `i` in the
container — which is exactly the assumption the LeRobot episode routing rests
on (`round(from_timestamp * fps)` is a container frame index).

⚠ THE PIPE IS SEQUENTIAL, BY DESIGN. There is no seek. LeRobot packs many
episodes into one mp4 and gives each a `from_timestamp`, so a whole file is
decoded once and its frames are routed to their rows as they arrive.
"""

from std.memory import Pointer

from ..proc import Pipe, quote_arg, run_capture


comptime FFMPEG = "ffmpeg"
comptime FFPROBE = "ffprobe"


@fieldwise_init
struct VideoInfo(Copyable, ImplicitlyCopyable, Movable):
    var width: Int
    var height: Int
    var n_frames: Int
    """From the container index. -1 when it does not carry one."""
    var pix_fmt: String


def _field(text: String, key: String) raises -> String:
    """Pull `key=value` out of ffprobe's `default=noprint_wrappers=1` output."""
    for line in text.splitlines():
        var s = String(line)
        if s.startswith(key + "="):
            return String(s.removeprefix(key + "="))
    raise Error(
        "video: ffprobe did not report '" + key + "'; it said:\n" + text
    )


def probe_video(path: String) raises -> VideoInfo:
    """Geometry and frame count, without decoding.

    Deliberately does NOT pass `-count_frames`: that decodes the entire file,
    which for a 200 MB LeRobot camera file costs as much as the real decode
    and would double the import time to learn a number the caller checks
    anyway as it streams.
    """
    var cmd = (
        String(FFPROBE) + " -v error -select_streams v:0 -show_entries"
        " stream=width,height,nb_frames,pix_fmt"
        " -of default=noprint_wrappers=1:nokey=0 " + quote_arg(path)
    )
    var text = run_capture(cmd^)
    var w = atol(_field(text, String("width")))
    var h = atol(_field(text, String("height")))
    var pf = _field(text, String("pix_fmt"))
    var n = -1
    try:
        var s = _field(text, String("nb_frames"))
        if s != "N/A":
            n = atol(s)
    except:
        n = -1
    if w <= 0 or h <= 0:
        raise Error(
            "video: '" + path + "' reports a " + String(w) + "x" + String(h)
            + " video stream"
        )
    return VideoInfo(w, h, n, pf^)


struct VideoDecoder(Movable):
    """A running `ffmpeg` emitting RGB24 frames at the source resolution."""

    var pipe: Pipe
    var width: Int
    var height: Int
    var frame_bytes: Int
    var declared_frames: Int
    var index: Int
    """Frames delivered so far."""
    var eof: Bool
    var path: String

    def __init__(out self, var path: String) raises:
        var info = probe_video(path)
        self.width = info.width
        self.height = info.height
        self.frame_bytes = info.width * info.height * 3
        self.declared_frames = info.n_frames
        self.index = 0
        self.eof = False
        # `-v error` keeps the banner and per-frame chatter out of stderr while
        # leaving real failures visible; stderr is inherited, not captured.
        var cmd = (
            String(FFMPEG) + " -v error -i " + quote_arg(path)
            + " -fps_mode passthrough -f rawvideo -pix_fmt rgb24 -"
        )
        self.pipe = Pipe(cmd^)
        self.path = path^

    def __init__(out self, *, deinit move: Self):
        self.pipe = move.pipe^
        self.width = move.width
        self.height = move.height
        self.frame_bytes = move.frame_bytes
        self.declared_frames = move.declared_frames
        self.index = move.index
        self.eof = move.eof
        self.path = move.path^

    def next_into(
        mut self, dst: Pointer[Scalar[DType.uint8], MutAnyOrigin]
    ) raises -> Bool:
        """Fill `dst` with one HWC RGB24 frame. False at end of stream.

        A partial frame is an error, not an end: it means ffmpeg died
        mid-frame, and silently accepting it would hand the caller a row that
        is half the previous frame.
        """
        if self.eof:
            return False
        var got = 0
        while got < self.frame_bytes:
            var n = self.pipe.read_into(
                dst.unsafe_offset(got), self.frame_bytes - got
            )
            if n <= 0:
                break
            got += n
        if got == 0:
            self.eof = True
            return False
        if got != self.frame_bytes:
            raise Error(
                "video: '" + self.path + "' ended " + String(got) + " bytes"
                " into frame " + String(self.index) + " of "
                + String(self.frame_bytes) + " — ffmpeg died mid-frame"
            )
        self.index += 1
        return True

    def close(mut self) raises:
        """Reap ffmpeg and raise if it failed.

        ⚠ CALL THIS. An ffmpeg that failed *after* producing some frames looks
        exactly like a short video until its exit status is read.

        Closing before end of stream is legitimate and common — LeRobot packs
        several episodes per file and the importer moves on once it has the
        ones it wants — so `SIGPIPE` is excused in that case and only that
        case. Once EOF has been seen, a clean exit is required.
        """
        _ = self.pipe.close(not self.eof)
