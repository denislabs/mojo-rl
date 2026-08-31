# +--------------------------------------------------------------------------+ #
# | Writing a LeRobot v3.0 dataset, entirely in Mojo
# +--------------------------------------------------------------------------+ #
"""The inverse of `lerobot.mojo`: episodes in, a v3.0 dataset directory out.

    var w = LeRobotWriter(root, fps=30, state_names, action_names,
                          cameras, height=480, width=640)
    for each episode:
        w.begin_episode(String("Grab the green cube"))
        for each frame:
            w.add_frame(state, action, camera_frames)   # rgb24 per camera
        w.end_episode()
    w.close()

Produces exactly the tree `import_lerobot_v3` reads, which is the tree
`lerobot-record` produces:

    meta/info.json
    meta/tasks.parquet
    meta/stats.json
    meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet
    videos/<camera>/chunk-000/file-{000,001,...}.mp4

## Episodes are packed into videos, and that is the whole layout problem

LeRobot concatenates many episodes into one mp4 per camera and locates each by
`from_timestamp` — `round(from_timestamp * fps)` is a frame index inside that
file. So a camera has ONE open encoder at a time, frames go straight into it as
they arrive (never buffered: 480x640x3 is 921 KB a frame), and an episode's
`from_timestamp` is the frame count already in the file divided by the rate.

⚠ **A FILE ROLLS ONLY BETWEEN EPISODES.** `from_timestamp`/`to_timestamp` name
a span inside ONE file, so an episode split across two of them is unlocatable.
`_maybe_roll` therefore runs in `end_episode`, never in `add_frame`.

⚠ **`video_files_size_in_mb` DEFAULTS TO 100 HERE, NOT LeRobot's 200.** It is
the lever that keeps every uploaded object inside HuggingFace's single-part
limit — see `docs/SO101_RECORDING_PLAN.md`. Raising it is legal and costs a
multipart implementation.

⚠ **THE ROLL IS CHECKED AFTER THE FACT.** An encoder's output size is only
known once ffmpeg has flushed, so the size test happens after `close()` and the
file that crossed the limit is already written. That is why the default has
headroom rather than sitting on the limit.

## Statistics

Per-episode stats for every feature, which is most of the 107 columns in
`meta/episodes/*.parquet`. Numeric features accumulate their values and
quantiles come from a sort (an episode is hundreds of rows). Images accumulate
a **256-bin histogram per channel** instead — exact for 8-bit data, constant
memory, and it makes `q01`..`q99` exact rather than sampled.

⚠ **IMAGE STATS ARE NORMALISED TO [0, 1]**, matching LeRobot: a real recording
reports `min 0.0`, `max 1.0`, `mean 0.47`. Writing raw 0..255 there would be
read as a wildly different scale by anything that normalises with them.

⚠ `count` IS A SAMPLE COUNT: the histogram samples every 4th pixel in each
axis, so it reports `length * H * W / 16` — the same rate LeRobot uses. That
is not cosmetic agreement; walking every pixel cost **45.8 ms of a 33.3 ms
tick** in the record loop.

## What is not implemented

⚠ **ONE `data/` FILE AND ONE CHUNK.** `data_files_size_in_mb` rolling is not
implemented — a 15,447-frame recording's data file is about 1.6 MB, so the
limit is nowhere near. `close()` raises rather than writing a file past 2 GB.
The video side DOES roll, because there the limit is reached routinely.

⚠ **UNCOMPRESSED PARQUET**, per `io/parquet/writer.mojo`.
"""

from std.math import sqrt
from std.os import makedirs
from std.os.path import exists

from mojo_rl.io.fileio import file_size, write_file_atomic
from mojo_rl.io.json import JsonWriter
from mojo_rl.io.parquet.writer import (
    ParquetWriter, PQ_F32, PQ_F64, PQ_I64, PQ_STR, PqColumn, pq_list,
    pq_list3, pq_scalar,
)
from mojo_rl.io.video import VideoEncoderThread


comptime CODEBASE_VERSION = "v3.0"
comptime CHUNKS_SIZE = 1000
comptime DEFAULT_VIDEO_MB = 100
comptime DEFAULT_DATA_MB = 100
comptime MAX_DATA_BYTES = 2_000_000_000

comptime STATS_STRIDE = 4
"""Sample every 4th pixel in each axis for the per-episode image histogram —
1/16 of the pixels, the same rate LeRobot uses. See `_ImgStats.add_frame`:
every pixel cost 45.8 ms of a 33.3 ms tick."""

comptime N_STATS = 10
"""min, max, mean, std, count, q01, q10, q50, q90, q99 — in that order, which
is the order a real `meta/episodes/*.parquet` carries them."""


def _stat_name(i: Int) -> String:
    if i == 0: return String("min")
    if i == 1: return String("max")
    if i == 2: return String("mean")
    if i == 3: return String("std")
    if i == 4: return String("count")
    if i == 5: return String("q01")
    if i == 6: return String("q10")
    if i == 7: return String("q50")
    if i == 8: return String("q90")
    return String("q99")


def _quantile_p(i: Int) -> Float64:
    if i == 5: return 0.01
    if i == 6: return 0.10
    if i == 7: return 0.50
    if i == 8: return 0.90
    return 0.99


def _pad3(v: Int) -> String:
    var s = String(v)
    while s.byte_length() < 3:
        s = "0" + s
    return s^


def _sort_f64(mut xs: List[Float64]):
    """Insertion sort. An episode is hundreds of rows, once per feature dim."""
    for i in range(1, len(xs)):
        var v = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > v:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = v


def _quantile_sorted(ref xs: List[Float64], p: Float64) -> Float64:
    """Linear-interpolated quantile of an already sorted list — numpy's
    default method, which is what produced the reference values."""
    var n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return xs[0]
    var pos = p * Float64(n - 1)
    var lo = Int(pos)
    if lo >= n - 1:
        return xs[n - 1]
    var frac = pos - Float64(lo)
    return xs[lo] + (xs[lo + 1] - xs[lo]) * frac


struct _NumStats(Movable):
    """One numeric feature's values for ONE episode, `[row * dim + d]`."""

    var dim: Int
    var vals: List[Float64]

    def __init__(out self, dim: Int):
        self.dim = dim
        self.vals = List[Float64]()

    def __init__(out self, *, deinit move: Self):
        self.dim = move.dim
        self.vals = move.vals^

    def n_rows(self) -> Int:
        return len(self.vals) // self.dim if self.dim > 0 else 0

    def stat(self, which: Int, d: Int) raises -> Float64:
        """One statistic of dimension `d`."""
        var n = self.n_rows()
        if n == 0:
            return 0.0
        if which == 4:
            return Float64(n)
        var col = List[Float64]()
        for r in range(n):
            col.append(self.vals[r * self.dim + d])
        if which == 2 or which == 3:
            var s = 0.0
            for i in range(len(col)):
                s += col[i]
            var mean = s / Float64(n)
            if which == 2:
                return mean
            # ⚠ POPULATION std (ddof=0), which is what LeRobot reports. The
            # ACT normalisation in `lerobot.mojo` uses ddof=1 for a DIFFERENT
            # purpose; they are not the same number and neither is wrong.
            var acc = 0.0
            for i in range(len(col)):
                var dv = col[i] - mean
                acc += dv * dv
            return sqrt(acc / Float64(n))
        _sort_f64(col)
        if which == 0:
            return col[0]
        if which == 1:
            return col[len(col) - 1]
        return _quantile_sorted(col, _quantile_p(which))


struct _ImgStats(Movable):
    """A 256-bin histogram per channel, for one camera over one episode."""

    var hist: List[Int]
    """`[channel * 256 + bin]`."""
    var samples: Int
    """Pixels counted per channel."""

    def __init__(out self):
        self.hist = List[Int]()
        for _ in range(3 * 256):
            self.hist.append(0)
        self.samples = 0

    def __init__(out self, *, deinit move: Self):
        self.hist = move.hist^
        self.samples = move.samples

    def reset(mut self):
        for i in range(len(self.hist)):
            self.hist[i] = 0
        self.samples = 0

    def add_frame(
        mut self, ref frame: List[UInt8], width: Int, height: Int
    ):
        """Accumulate one RGB24 frame, sampling every `STATS_STRIDE`th pixel.

        ⚠ **EVERY PIXEL IS TOO EXPENSIVE TO DO IN THE RECORD LOOP.** At
        640x480x2 cameras that is 614,400 histogram increments per tick, and
        it took the recorder's worst per-tick WORK to 45.8 ms against a
        33.3 ms budget — on a loop the budget tool had measured at 5.43 ms
        before this function existed. Sampling 4x4 is 1/16 the work.

        This also makes `count` match LeRobot's, which samples the same way:
        `length * H * W / 16`. The statistic is a summary of a camera's
        exposure over an episode; a sixteenth of a 480p frame is 19,200
        samples per frame, which is not the limiting factor in its accuracy.
        """
        var stride = STATS_STRIDE
        var n = 0
        for y in range(0, height, stride):
            var row = y * width
            for x in range(0, width, stride):
                var b = (row + x) * 3
                self.hist[Int(frame[b])] += 1
                self.hist[256 + Int(frame[b + 1])] += 1
                self.hist[512 + Int(frame[b + 2])] += 1
                n += 1
        self.samples += n

    def stat(self, which: Int, c: Int) -> Float64:
        """One statistic of channel `c`, normalised to [0, 1]."""
        if self.samples == 0:
            return 0.0
        if which == 4:
            return Float64(self.samples)
        var base = c * 256
        if which == 0:
            for b in range(256):
                if self.hist[base + b] > 0:
                    return Float64(b) / 255.0
            return 0.0
        if which == 1:
            for b in range(255, -1, -1):
                if self.hist[base + b] > 0:
                    return Float64(b) / 255.0
            return 0.0
        if which == 2 or which == 3:
            var s = 0.0
            for b in range(256):
                s += Float64(self.hist[base + b]) * Float64(b)
            var mean = s / Float64(self.samples)
            if which == 2:
                return mean / 255.0
            var acc = 0.0
            for b in range(256):
                var dv = Float64(b) - mean
                acc += Float64(self.hist[base + b]) * dv * dv
            return sqrt(acc / Float64(self.samples)) / 255.0
        # A quantile, read straight off the cumulative histogram. Exact for
        # 8-bit data, which is why the histogram is worth keeping at all.
        var target = _quantile_p(which) * Float64(self.samples - 1)
        var seen = 0
        for b in range(256):
            seen += self.hist[base + b]
            if Float64(seen) > target:
                return Float64(b) / 255.0
        return 1.0


struct LeRobotWriter(Movable):
    """Accumulates episodes and writes a v3.0 dataset directory."""

    var root: String
    var fps: Int
    var robot_type: String
    var state_names: List[String]
    var action_names: List[String]
    var cameras: List[String]
    var height: Int
    var width: Int
    var video_mb: Int

    # ── dataset-wide row storage (small: floats, not images) ──────────
    var state: List[Float64]
    var action: List[Float64]
    var tasks: List[String]

    # ── per-episode bookkeeping ───────────────────────────────────────
    var ep_length: List[Int]
    var ep_task: List[Int]
    var ep_from: List[Int]
    var ep_to: List[Int]
    var ep_vid_file: List[Int]
    """`[camera * n_episodes + ep]`."""
    var ep_vid_from: List[Float64]
    var ep_stats: List[Float64]
    """Flattened per-episode numeric stats; see `_stat_slot`."""
    var ep_img_stats: List[Float64]
    """`[(ep * n_cam + cam) * (N_STATS * 3) + which * 3 + channel]`."""

    # ── in-flight episode ─────────────────────────────────────────────
    var _open: Bool
    var _cur_len: Int
    var _cur_task: Int
    var _cur_state: _NumStats
    var _cur_action: _NumStats
    var _img: List[_ImgStats]

    # ── video encoders, one per camera ────────────────────────────────
    var _enc: List[VideoEncoderThread]
    var _enc_file: List[Int]
    var _enc_frames: List[Int]
    var _enc_pending: List[Bool]
    """The camera's encoder is CLOSED and the next one is not open yet."""
    var _submitted: List[Int]
    """Frames handed to each camera's encoders, across rolls."""
    var _accepted: List[Int]
    """Frames ffmpeg accepted, accumulated as encoders are closed."""
    var closed: Bool

    def __init__(
        out self,
        var root: String,
        fps: Int,
        var state_names: List[String],
        var action_names: List[String],
        var cameras: List[String],
        height: Int,
        width: Int,
        var robot_type: String = String("so_follower"),
        video_mb: Int = DEFAULT_VIDEO_MB,
    ) raises:
        if len(cameras) == 0:
            raise Error(
                "lerobot_write: a v3 dataset needs at least one camera —"
                " `import_lerobot_v3` refuses a dataset with no video feature"
            )
        if len(state_names) == 0 or len(action_names) == 0:
            raise Error("lerobot_write: state and action need names")
        self.root = root^
        self.fps = fps
        self.robot_type = robot_type^
        self.state_names = state_names^
        self.action_names = action_names^
        self.cameras = cameras^
        self.height = height
        self.width = width
        self.video_mb = video_mb

        self.state = List[Float64]()
        self.action = List[Float64]()
        self.tasks = List[String]()
        self.ep_length = List[Int]()
        self.ep_task = List[Int]()
        self.ep_from = List[Int]()
        self.ep_to = List[Int]()
        self.ep_vid_file = List[Int]()
        self.ep_vid_from = List[Float64]()
        self.ep_stats = List[Float64]()
        self.ep_img_stats = List[Float64]()

        self._open = False
        self._cur_len = 0
        self._cur_task = -1
        self._cur_state = _NumStats(len(self.state_names))
        self._cur_action = _NumStats(len(self.action_names))
        self._img = List[_ImgStats]()
        self._enc = List[VideoEncoderThread]()
        self._enc_file = List[Int]()
        self._enc_frames = List[Int]()
        self._enc_pending = List[Bool]()
        self._submitted = List[Int]()
        self._accepted = List[Int]()
        self.closed = False

        makedirs(self.root + "/meta/episodes/chunk-000", exist_ok=True)
        makedirs(self.root + "/data/chunk-000", exist_ok=True)
        for c in range(len(self.cameras)):
            makedirs(
                self.root + "/videos/" + self.cameras[c] + "/chunk-000",
                exist_ok=True,
            )
            self._img.append(_ImgStats())
            self._enc_file.append(0)
            self._enc_frames.append(0)
            self._enc_pending.append(False)
            var e = VideoEncoderThread(
                self._video_path(c, 0), self.width, self.height, self.fps
            )
            e.start()
            self._enc.append(e^)
            self._submitted.append(0)
            self._accepted.append(0)

    def __init__(out self, *, deinit move: Self):
        self.root = move.root^
        self.fps = move.fps
        self.robot_type = move.robot_type^
        self.state_names = move.state_names^
        self.action_names = move.action_names^
        self.cameras = move.cameras^
        self.height = move.height
        self.width = move.width
        self.video_mb = move.video_mb
        self.state = move.state^
        self.action = move.action^
        self.tasks = move.tasks^
        self.ep_length = move.ep_length^
        self.ep_task = move.ep_task^
        self.ep_from = move.ep_from^
        self.ep_to = move.ep_to^
        self.ep_vid_file = move.ep_vid_file^
        self.ep_vid_from = move.ep_vid_from^
        self.ep_stats = move.ep_stats^
        self.ep_img_stats = move.ep_img_stats^
        self._open = move._open
        self._cur_len = move._cur_len
        self._cur_task = move._cur_task
        self._cur_state = move._cur_state^
        self._cur_action = move._cur_action^
        self._img = move._img^
        self._enc = move._enc^
        self._enc_file = move._enc_file^
        self._enc_frames = move._enc_frames^
        self._enc_pending = move._enc_pending^
        self._submitted = move._submitted^
        self._accepted = move._accepted^
        self.closed = move.closed

    def _video_path(self, cam: Int, file_index: Int) -> String:
        return (
            self.root + "/videos/" + self.cameras[cam] + "/chunk-000/file-"
            + _pad3(file_index) + ".mp4"
        )

    def n_episodes(self) -> Int:
        return len(self.ep_length)

    def n_rows(self) -> Int:
        var t = 0
        for i in range(len(self.ep_length)):
            t += self.ep_length[i]
        return t

    def _task_index(mut self, var task: String) -> Int:
        for i in range(len(self.tasks)):
            if self.tasks[i] == task:
                return i
        self.tasks.append(task^)
        return len(self.tasks) - 1

    # ── recording ─────────────────────────────────────────────────────

    def begin_episode(mut self, var task: String) raises:
        if self.closed:
            raise Error("lerobot_write: the dataset is closed")
        if self._open:
            raise Error(
                "lerobot_write: begin_episode while episode "
                + String(self.n_episodes()) + " is still open"
            )
        self._open = True
        self._cur_len = 0
        self._cur_task = self._task_index(task^)
        self._cur_state = _NumStats(len(self.state_names))
        self._cur_action = _NumStats(len(self.action_names))
        for c in range(len(self.cameras)):
            self._img[c].reset()

        # ⚠ THE ROLLED-TO ENCODER IS OPENED HERE, NOT AT THE ROLL. Opening it
        # in `_maybe_roll` creates a file for an episode that may never come:
        # rolling after the LAST episode then leaves an empty mp4, which
        # `VideoEncoder.close` rightly refuses to write. Deferring to the next
        # `begin_episode` means a file exists only once it has a frame.
        for c in range(len(self.cameras)):
            if self._enc_pending[c]:
                var e = VideoEncoderThread(
                    self._video_path(c, self._enc_file[c]),
                    self.width,
                    self.height,
                    self.fps,
                )
                e.start()
                self._enc[c] = e^
                self._enc_pending[c] = False

        # Where this episode starts inside each camera's CURRENT file.
        for c in range(len(self.cameras)):
            self.ep_vid_file.append(self._enc_file[c])
            self.ep_vid_from.append(
                Float64(self._enc_frames[c]) / Float64(self.fps)
            )

    def add_frame(
        mut self,
        ref state: List[Float64],
        ref action: List[Float64],
        mut frames: List[List[UInt8]],
    ) raises:
        """One timestep: the observed state, the commanded action, one RGB24
        frame per camera in `self.cameras` order."""
        if not self._open:
            raise Error("lerobot_write: add_frame outside an episode")
        if len(state) != len(self.state_names):
            raise Error(
                "lerobot_write: state has " + String(len(state))
                + " values, the schema declares " + String(len(self.state_names))
            )
        if len(action) != len(self.action_names):
            raise Error(
                "lerobot_write: action has " + String(len(action))
                + " values, the schema declares "
                + String(len(self.action_names))
            )
        if len(frames) != len(self.cameras):
            raise Error(
                "lerobot_write: " + String(len(frames)) + " camera frames for "
                + String(len(self.cameras)) + " cameras"
            )

        var need = self.width * self.height * 3
        for c in range(len(self.cameras)):
            if len(frames[c]) != need:
                raise Error(
                    "lerobot_write: camera '" + self.cameras[c] + "' frame is "
                    + String(len(frames[c])) + " bytes, a " + String(self.width)
                    + "x" + String(self.height) + " rgb24 frame is "
                    + String(need)
                )

        for i in range(len(state)):
            self.state.append(state[i])
            self._cur_state.vals.append(state[i])
        for i in range(len(action)):
            self.action.append(action[i])
            self._cur_action.vals.append(action[i])

        for c in range(len(self.cameras)):
            # ⚠ BLOCKING, NOT DROPPING. A dropped frame leaves the mp4 with
            # fewer frames than the parquet has rows, which offsets every
            # episode after it in that file. See `submit_blocking`.
            if not self._enc[c].submit_blocking(frames[c]):
                raise Error(
                    "lerobot_write: the encoder for '" + self.cameras[c]
                    + "' did not accept a frame within its timeout — ffmpeg"
                    " is wedged and this recording cannot be completed"
                )
            self._submitted[c] += 1
            self._enc_frames[c] += 1
            self._img[c].add_frame(frames[c], self.width, self.height)

        self._cur_len += 1

    def end_episode(mut self) raises:
        if not self._open:
            raise Error("lerobot_write: end_episode with no episode open")
        if self._cur_len == 0:
            raise Error(
                "lerobot_write: refusing to record an episode with no frames"
                " — an empty episode is a length-0 span every consumer has to"
                " special-case"
            )
        var start = self.n_rows()
        self.ep_length.append(self._cur_len)
        self.ep_task.append(self._cur_task)
        self.ep_from.append(start)
        self.ep_to.append(start + self._cur_len)

        self._collect_stats()
        self._open = False
        self._maybe_roll()

    def _collect_stats(mut self) raises:
        """Freeze the open episode's statistics, in the file's feature order."""
        var n = self._cur_len
        var ep = self.n_episodes() - 1

        # The scalar features are derived, not stored: they are exactly what
        # the data file will carry for these rows.
        var ts = _NumStats(1)
        var fi = _NumStats(1)
        var ei = _NumStats(1)
        var gi = _NumStats(1)
        var ti = _NumStats(1)
        for r in range(n):
            ts.vals.append(Float64(r) / Float64(self.fps))
            fi.vals.append(Float64(r))
            ei.vals.append(Float64(ep))
            gi.vals.append(Float64(self.ep_from[ep] + r))
            ti.vals.append(Float64(self._cur_task))

        for which in range(N_STATS):
            for d in range(self._cur_action.dim):
                self.ep_stats.append(self._cur_action.stat(which, d))
        for which in range(N_STATS):
            for d in range(self._cur_state.dim):
                self.ep_stats.append(self._cur_state.stat(which, d))
        for which in range(N_STATS):
            self.ep_stats.append(ts.stat(which, 0))
        for which in range(N_STATS):
            self.ep_stats.append(fi.stat(which, 0))
        for which in range(N_STATS):
            self.ep_stats.append(ei.stat(which, 0))
        for which in range(N_STATS):
            self.ep_stats.append(gi.stat(which, 0))
        for which in range(N_STATS):
            self.ep_stats.append(ti.stat(which, 0))

        for c in range(len(self.cameras)):
            for which in range(N_STATS):
                for ch in range(3):
                    self.ep_img_stats.append(self._img[c].stat(which, ch))

    def _maybe_roll(mut self) raises:
        """Start a new video file per camera that has grown past the limit.

        ⚠ BETWEEN EPISODES ONLY — see the module docstring. And the size is
        only knowable after `close()` flushes, so the file that crossed the
        limit has already been written; the limit is a target with headroom,
        not a cap.
        """
        var limit = self.video_mb * 1000 * 1000
        for c in range(len(self.cameras)):
            var path = self._video_path(c, self._enc_file[c])
            var frames_here = self._enc_frames[c]
            if frames_here == 0:
                continue
            # Closing to measure would end the file, so measure what ffmpeg
            # has flushed so far: an mp4 being written grows on disk.
            var so_far: Int
            try:
                so_far = file_size(path)
            except:
                so_far = 0
            if so_far < limit:
                continue
            self._accepted[c] += self._enc[c].stop()
            self._enc_file[c] += 1
            self._enc_frames[c] = 0
            self._enc_pending[c] = True

    # ── writing the dataset out ───────────────────────────────────────

    def close(mut self, verbose: Bool = True) raises:
        if self.closed:
            raise Error("lerobot_write: already closed")
        if self._open:
            raise Error(
                "lerobot_write: close() with episode "
                + String(self.n_episodes()) + " still open — call end_episode"
            )
        if self.n_episodes() == 0:
            raise Error(
                "lerobot_write: refusing to write a dataset with no episodes"
            )

        for c in range(len(self.cameras)):
            # A pending camera's encoder was already stopped by the roll.
            if not self._enc_pending[c]:
                self._accepted[c] += self._enc[c].stop()

        # ⚠ THE VIDEO MUST HOLD EXACTLY THE ROWS THE PARQUET CLAIMS. A frame
        # lost between here and ffmpeg is not a degraded recording, it is a
        # MISALIGNED one: every episode after the gap reads another episode's
        # frames, and nothing downstream can detect it. Checked once, here,
        # where the counts are finally settled.
        for c in range(len(self.cameras)):
            if self._submitted[c] != self._accepted[c]:
                raise Error(
                    "lerobot_write: camera '" + self.cameras[c] + "' was sent "
                    + String(self._submitted[c]) + " frames but ffmpeg"
                    " accepted " + String(self._accepted[c])
                    + " — the video and the data would not line up"
                )
            if self._submitted[c] != self.n_rows():
                raise Error(
                    "lerobot_write: camera '" + self.cameras[c] + "' has "
                    + String(self._submitted[c]) + " frames for "
                    + String(self.n_rows()) + " rows"
                )

        self._write_data()
        self._write_tasks()
        self._write_episodes()
        self._write_info()
        self._write_stats_json()
        self.closed = True
        if verbose:
            print(
                "wrote " + String(self.n_episodes()) + " episodes / "
                + String(self.n_rows()) + " frames to " + self.root
            )

    def _write_data(mut self) raises:
        var sdim = len(self.state_names)
        var adim = len(self.action_names)
        var cols = List[PqColumn]()
        cols.append(pq_list(String("action"), PQ_F32, adim))
        cols.append(pq_list(String("observation.state"), PQ_F32, sdim))
        cols.append(pq_scalar(String("timestamp"), PQ_F32))
        cols.append(pq_scalar(String("frame_index"), PQ_I64))
        cols.append(pq_scalar(String("episode_index"), PQ_I64))
        cols.append(pq_scalar(String("index"), PQ_I64))
        cols.append(pq_scalar(String("task_index"), PQ_I64))

        var w = ParquetWriter(cols^)
        w.add_metadata(String("huggingface"), self._hf_features_json())

        var row = 0
        for ep in range(self.n_episodes()):
            var vals = w.new_values()
            for r in range(self.ep_length[ep]):
                for d in range(adim):
                    vals[0].push_f64(self.action[row * adim + d])
                for d in range(sdim):
                    vals[1].push_f64(self.state[row * sdim + d])
                vals[2].push_f64(Float64(r) / Float64(self.fps))
                vals[3].push_i64(r)
                vals[4].push_i64(ep)
                vals[5].push_i64(row)
                vals[6].push_i64(self.ep_task[ep])
                row += 1
            # ⚠ ONE ROW GROUP PER EPISODE, like LeRobot. A reader that wants
            # one episode then reads one row group instead of scanning.
            w.write_row_group(vals, self.ep_length[ep])

        var n = w.close(self.root + "/data/chunk-000/file-000.parquet")
        if n > MAX_DATA_BYTES:
            raise Error(
                "lerobot_write: the data file is " + String(n // 1000000)
                + " MB; rolling across data files is not implemented"
            )

    def _write_tasks(mut self) raises:
        var cols = List[PqColumn]()
        cols.append(pq_scalar(String("task_index"), PQ_I64))
        cols.append(pq_scalar(String("task"), PQ_STR))
        var w = ParquetWriter(cols^)
        var vals = w.new_values()
        for i in range(len(self.tasks)):
            vals[0].push_i64(i)
            vals[1].push_str(self.tasks[i])
        w.write_row_group(vals, len(self.tasks))
        _ = w.close(self.root + "/meta/tasks.parquet")

    def _write_episodes(mut self) raises:
        var n_cam = len(self.cameras)
        var sdim = len(self.state_names)
        var adim = len(self.action_names)

        var cols = List[PqColumn]()
        cols.append(pq_scalar(String("episode_index"), PQ_I64))
        cols.append(pq_list(String("tasks"), PQ_STR))
        cols.append(pq_scalar(String("length"), PQ_I64))
        cols.append(pq_scalar(String("data/chunk_index"), PQ_I64))
        cols.append(pq_scalar(String("data/file_index"), PQ_I64))
        cols.append(pq_scalar(String("dataset_from_index"), PQ_I64))
        cols.append(pq_scalar(String("dataset_to_index"), PQ_I64))
        for c in range(n_cam):
            var pre = String("videos/") + self.cameras[c] + "/"
            cols.append(pq_scalar(pre + "chunk_index", PQ_I64))
            cols.append(pq_scalar(pre + "file_index", PQ_I64))
            cols.append(pq_scalar(pre + "from_timestamp", PQ_F64))
            cols.append(pq_scalar(pre + "to_timestamp", PQ_F64))

        # Numeric feature stats, in the file's order.
        var feat_names = List[String]()
        var feat_dims = List[Int]()
        feat_names.append(String("action"))
        feat_dims.append(adim)
        feat_names.append(String("observation.state"))
        feat_dims.append(sdim)
        for nm in [
            String("timestamp"), String("frame_index"),
            String("episode_index"), String("index"), String("task_index"),
        ]:
            feat_names.append(nm)
            feat_dims.append(1)

        for f in range(len(feat_names)):
            for s in range(N_STATS):
                var nm = String("stats/") + feat_names[f] + "/" + _stat_name(s)
                if s == 4:
                    cols.append(pq_list(nm, PQ_I64, 1))
                else:
                    cols.append(pq_list(nm, PQ_F64, feat_dims[f]))
        for c in range(n_cam):
            for s in range(N_STATS):
                var nm = String("stats/") + self.cameras[c] + "/" + _stat_name(s)
                if s == 4:
                    cols.append(pq_list(nm, PQ_I64, 1))
                else:
                    cols.append(pq_list3(nm, PQ_F64, 3, 1, 1))
        cols.append(pq_scalar(String("meta/episodes/chunk_index"), PQ_I64))
        cols.append(pq_scalar(String("meta/episodes/file_index"), PQ_I64))

        var w = ParquetWriter(cols^)
        var vals = w.new_values()
        var n_ep = self.n_episodes()

        # Stats stride per episode, matching `_collect_stats`' append order.
        var num_stride = N_STATS * (adim + sdim + 5)

        for ep in range(n_ep):
            var k = 0
            vals[k].push_i64(ep); k += 1
            vals[k].push_str(self.tasks[self.ep_task[ep]]); vals[k].push_count(1); k += 1
            vals[k].push_i64(self.ep_length[ep]); k += 1
            vals[k].push_i64(0); k += 1
            vals[k].push_i64(0); k += 1
            vals[k].push_i64(self.ep_from[ep]); k += 1
            vals[k].push_i64(self.ep_to[ep]); k += 1
            for c in range(n_cam):
                var slot = ep * n_cam + c
                vals[k].push_i64(0); k += 1
                vals[k].push_i64(self.ep_vid_file[slot]); k += 1
                vals[k].push_f64(self.ep_vid_from[slot]); k += 1
                vals[k].push_f64(
                    self.ep_vid_from[slot]
                    + Float64(self.ep_length[ep]) / Float64(self.fps)
                ); k += 1

            var base = ep * num_stride
            var off = 0
            for f in range(len(feat_names)):
                var d = feat_dims[f]
                for s in range(N_STATS):
                    if s == 4:
                        vals[k].push_i64(Int(self.ep_stats[base + off]))
                        off += d
                    else:
                        for j in range(d):
                            vals[k].push_f64(self.ep_stats[base + off + j])
                        off += d
                    k += 1

            var ibase = ep * n_cam * N_STATS * 3
            for c in range(n_cam):
                for s in range(N_STATS):
                    var o = ibase + c * N_STATS * 3 + s * 3
                    if s == 4:
                        vals[k].push_i64(Int(self.ep_img_stats[o]))
                    else:
                        for ch in range(3):
                            vals[k].push_f64(self.ep_img_stats[o + ch])
                    k += 1
            vals[k].push_i64(0); k += 1
            vals[k].push_i64(0); k += 1

        w.write_row_group(vals, n_ep)
        _ = w.close(
            self.root + "/meta/episodes/chunk-000/file-000.parquet"
        )

    def _hf_features_json(self) raises -> String:
        """The `huggingface` key-value entry the Hub's viewer reads.

        ⚠ NOT `ARROW:schema`. That second entry is a base64 Arrow flatbuffer,
        which is where pyarrow's `fixed_size_list[6]` comes from and which
        would need a flatbuffer encoder. This one is plain JSON and is what
        LeRobot itself reads — see `docs/SO101_RECORDING_PLAN.md`.
        """
        var w = JsonWriter()
        w.begin_object()
        w.key(String("info"))
        w.begin_object()
        w.key(String("features"))
        w.begin_object()
        for nm_dim in [
            (String("action"), len(self.action_names)),
            (String("observation.state"), len(self.state_names)),
        ]:
            w.key(nm_dim[0])
            w.begin_object()
            w.key(String("feature"))
            w.begin_object()
            w.member(String("dtype"), String("float32"))
            w.member(String("_type"), String("Value"))
            w.end_object()
            w.member(String("length"), nm_dim[1])
            w.member(String("_type"), String("List"))
            w.end_object()
        for nm_ty in [
            (String("timestamp"), String("float32")),
            (String("frame_index"), String("int64")),
            (String("episode_index"), String("int64")),
            (String("index"), String("int64")),
            (String("task_index"), String("int64")),
        ]:
            w.key(nm_ty[0])
            w.begin_object()
            w.member(String("dtype"), nm_ty[1])
            w.member(String("_type"), String("Value"))
            w.end_object()
        w.end_object()
        w.end_object()
        w.end_object()
        return w.done()

    def _write_info(mut self) raises:
        var w = JsonWriter()
        w.begin_object()
        w.member(String("codebase_version"), String(CODEBASE_VERSION))
        w.member(String("robot_type"), self.robot_type)
        w.member(String("fps"), self.fps)
        w.member(String("total_episodes"), self.n_episodes())
        w.member(String("total_frames"), self.n_rows())
        w.member(String("total_tasks"), len(self.tasks))
        w.member(String("chunks_size"), CHUNKS_SIZE)
        w.member(String("data_files_size_in_mb"), DEFAULT_DATA_MB)
        w.member(String("video_files_size_in_mb"), self.video_mb)
        w.member(
            String("data_path"),
            String("data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"),
        )
        w.member(
            String("video_path"),
            String(
                "videos/{video_key}/chunk-{chunk_index:03d}/"
                "file-{file_index:03d}.mp4"
            ),
        )
        w.key(String("splits"))
        w.begin_object()
        w.member(
            String("train"), String("0:") + String(self.n_episodes())
        )
        w.end_object()

        w.key(String("features"))
        w.begin_object()
        for is_action in [True, False]:
            w.key(
                String("action") if is_action
                else String("observation.state")
            )
            w.begin_object()
            w.member(String("dtype"), String("float32"))
            w.key(String("names"))
            w.begin_array()
            ref names = self.action_names if is_action else self.state_names
            for i in range(len(names)):
                w.string(names[i])
            w.end_array()
            w.key(String("shape"))
            w.begin_array()
            w.integer(len(names))
            w.end_array()
            w.end_object()

        for c in range(len(self.cameras)):
            w.key(self.cameras[c])
            w.begin_object()
            w.member(String("dtype"), String("video"))
            w.key(String("shape"))
            w.begin_array()
            w.integer(self.height)
            w.integer(self.width)
            w.integer(3)
            w.end_array()
            w.key(String("names"))
            w.begin_array()
            w.string(String("height"))
            w.string(String("width"))
            w.string(String("channels"))
            w.end_array()
            w.key(String("info"))
            w.begin_object()
            w.key(String("is_depth_map"))
            w.boolean(False)
            w.member(String("video.height"), self.height)
            w.member(String("video.width"), self.width)
            w.member(String("video.codec"), String("h264"))
            w.member(String("video.pix_fmt"), String("yuv420p"))
            w.member(String("video.fps"), self.fps)
            w.member(String("video.channels"), 3)
            w.key(String("has_audio"))
            w.boolean(False)
            w.member(String("video.g"), 2)
            w.member(String("video.crf"), 30)
            w.member(String("video.video_backend"), String("ffmpeg-pipe"))
            w.end_object()
            w.end_object()

        for nm_ty in [
            (String("timestamp"), String("float32")),
            (String("frame_index"), String("int64")),
            (String("episode_index"), String("int64")),
            (String("index"), String("int64")),
            (String("task_index"), String("int64")),
        ]:
            w.key(nm_ty[0])
            w.begin_object()
            w.member(String("dtype"), nm_ty[1])
            w.key(String("shape"))
            w.begin_array()
            w.integer(1)
            w.end_array()
            w.key(String("names"))
            w.null()
            w.end_object()
        w.end_object()
        w.end_object()

        var text = w.done()
        var bytes = List[UInt8]()
        for i in range(text.byte_length()):
            bytes.append(text.as_bytes()[i])
        write_file_atomic(self.root + "/meta/info.json", bytes)

    def _write_stats_json(mut self) raises:
        """Dataset-level aggregate stats.

        ⚠ AGGREGATED FROM THE PER-EPISODE MIN/MAX/MEAN, not recomputed. The
        mean is length-weighted; min and max are exact. `std` is NOT
        aggregated exactly — it is the root of the length-weighted mean of the
        per-episode variances, which understates the between-episode spread.
        Nothing in this repo reads `meta/stats.json`; it exists because
        LeRobot writes one, and this note exists so nobody trusts that `std`
        for normalisation without recomputing it.
        """
        var w = JsonWriter()
        w.begin_object()
        var adim = len(self.action_names)
        var sdim = len(self.state_names)
        var num_stride = N_STATS * (adim + sdim + 5)
        var total = Float64(self.n_rows())

        for f in range(2):
            var dim = adim if f == 0 else sdim
            var off = 0 if f == 0 else N_STATS * adim
            w.key(String("action") if f == 0 else String("observation.state"))
            w.begin_object()
            for which in [0, 1, 2]:
                w.key(_stat_name(which))
                w.begin_array()
                for d in range(dim):
                    var acc = 0.0
                    var first = True
                    for ep in range(self.n_episodes()):
                        var v = self.ep_stats[
                            ep * num_stride + off + which * dim + d
                        ]
                        if which == 0:
                            acc = v if first else (v if v < acc else acc)
                        elif which == 1:
                            acc = v if first else (v if v > acc else acc)
                        else:
                            acc += v * Float64(self.ep_length[ep])
                        first = False
                    w.number(acc / total if which == 2 else acc)
                w.end_array()
            w.end_object()
        w.end_object()

        var text = w.done()
        var bytes = List[UInt8]()
        for i in range(text.byte_length()):
            bytes.append(text.as_bytes()[i])
        write_file_atomic(self.root + "/meta/stats.json", bytes)
