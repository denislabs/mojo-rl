# +--------------------------------------------------------------------------+ #
# | Encoding an mp4 the way LeRobot does, by piping `ffmpeg`
# +--------------------------------------------------------------------------+ #
"""The inverse of `decoder.mojo`: RGB24 frames in, one H.264 mp4 out.

    var enc = VideoEncoder(path, 640, 480, fps=30)
    for each frame:
        enc.add_frame(ptr, n_bytes)             # rgb24, row-major, tightly packed
    var n = enc.close()                         # WAITS for ffmpeg

Same argument as the decoder for using a pipe rather than binding libavcodec:
every useful FFmpeg call needs a struct field, and none of them have accessor
functions. RGB24 is deliberately the same pixel format both directions, so a
decode of what this wrote is comparable to what went in.

## The settings are LeRobot's, read off a real recording

`ffprobe` on `record-test_20260828_092736`: `h264`, `640x480`, `yuv420p`,
`30/1`, and `meta/info.json` records `video.crf 30` and **`video.g 2`**.

⚠ **`-g 2` IS NOT A TYPO AND IT IS EXPENSIVE.** A keyframe every two frames is
what makes an episode inside a packed mp4 cheap to reach; LeRobot pays real
file size for it (194 MB for 8,370 frames). Encoding a dataset with a default
GOP of 250 produces a much smaller file that is much slower to sample from,
which will look like a win right up until training reads it.

⚠ **`-fps_mode passthrough`, AFTER `-i`.** The decoder's header explains why
it is load-bearing there; it is the same property from the other side. Without
it ffmpeg may duplicate or drop frames to hit a constant rate, and the frame
this writes at index `i` stops being the frame at index `i` in the container —
which is exactly what `round(from_timestamp * fps)` assumes when routing an
episode's frames back out.

⚠ **ODD DIMENSIONS RAISE.** `yuv420p` subsamples chroma by two, so an odd
width or height cannot be encoded and ffmpeg would either fail or, with a
scale filter, silently resize. For a DATASET a silent resize is the worse
outcome by far: `meta/info.json` would declare a shape the frames do not have.

⚠ **`close()` WAITS.** The `moov` atom is written when the input ends, so
returning before ffmpeg exits hands back a truncated file that looks finished.
`WritePipe.close` already does this; it is repeated here because a recorder
that forgets is left with a directory of unplayable videos.
"""

from std.memory import Pointer

from ..proc import WritePipe, quote_arg


comptime FFMPEG = "ffmpeg"

comptime LEROBOT_CRF = 30
comptime LEROBOT_GOP = 2
"""`video.g` in a LeRobot `meta/info.json`. See the warning above."""


struct VideoEncoder(Movable):
    """One mp4 being written, frame by frame."""

    var _pipe: WritePipe
    var path: String
    var width: Int
    var height: Int
    var fps: Int
    var crf: Int
    var gop: Int
    var frames: Int
    var closed: Bool

    def __init__(
        out self,
        var path: String,
        width: Int,
        height: Int,
        fps: Int = 30,
        crf: Int = LEROBOT_CRF,
        gop: Int = LEROBOT_GOP,
    ) raises:
        if width <= 0 or height <= 0:
            raise Error(
                "video: refusing a " + String(width) + "x" + String(height)
                + " encoder"
            )
        if width % 2 != 0 or height % 2 != 0:
            raise Error(
                "video: " + String(width) + "x" + String(height)
                + " cannot be encoded as yuv420p, which subsamples chroma by"
                " two. Pick even dimensions rather than letting a scale filter"
                " silently resize the frames away from the shape info.json"
                " declares."
            )
        var cmd = (
            String(FFMPEG) + " -y -v error"
            + " -f rawvideo -pix_fmt rgb24 -s " + String(width) + "x"
            + String(height) + " -r " + String(fps) + " -i -"
            + " -an -c:v libx264 -pix_fmt yuv420p"
            + " -crf " + String(crf) + " -g " + String(gop)
            + " -fps_mode passthrough "
            + quote_arg(path)
        )
        self._pipe = WritePipe(cmd^)
        self.path = path^
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.gop = gop
        self.frames = 0
        self.closed = False

    def __init__(out self, *, deinit move: Self):
        self._pipe = move._pipe^
        self.path = move.path^
        self.width = move.width
        self.height = move.height
        self.fps = move.fps
        self.crf = move.crf
        self.gop = move.gop
        self.frames = move.frames
        self.closed = move.closed

    def frame_bytes(self) -> Int:
        return self.width * self.height * 3

    def add_frame(
        mut self, src: Pointer[Scalar[DType.uint8], MutAnyOrigin], count: Int
    ) raises:
        """Append one RGB24 frame.

        ⚠ THE COUNT IS CHECKED, not trusted. A short frame does not fail — it
        shifts every subsequent frame by the shortfall, and the video decodes
        into a slowly sliding image that looks like a camera problem rather
        than a caller bug.
        """
        if self.closed:
            raise Error("video: add_frame on a closed encoder: " + self.path)
        if count != self.frame_bytes():
            raise Error(
                "video: frame " + String(self.frames) + " is " + String(count)
                + " bytes, a " + String(self.width) + "x" + String(self.height)
                + " rgb24 frame is " + String(self.frame_bytes())
            )
        self._pipe.write_all(src, count)
        self.frames += 1

    def add_frame_list(mut self, mut frame: List[UInt8]) raises:
        """`add_frame` over a host byte list."""
        if len(frame) == 0:
            raise Error("video: an empty frame")
        self.add_frame(
            frame.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
            .as_unsafe_any_origin(),
            len(frame),
        )

    def close(mut self) raises -> Int:
        """Close stdin, WAIT for ffmpeg, return the frame count."""
        if self.closed:
            return self.frames
        if self.frames == 0:
            # An mp4 with no frames is not a video; ffmpeg writes a file that
            # `probe_video` reports as 0 frames and every consumer trips over
            # later, far from here.
            _ = self._pipe.close()
            self.closed = True
            raise Error(
                "video: closing " + self.path + " with no frames written"
            )
        _ = self._pipe.close()
        self.closed = True
        return self.frames
