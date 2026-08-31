"""Video decode and encode for datasets, both by piping `ffmpeg`.

See `decoder.mojo` for why a pipe rather than a libavcodec binding, and
`encoder.mojo` for the LeRobot encoding settings (`-g 2` in particular).
RGB24 is the pixel format in both directions.

`frame_pipe_thread.mojo` is the format-agnostic half: fixed-size frames into an
`ffmpeg` pipe from a worker thread, which is what `render/video_recorder.mojo`
feeds bgra through.
"""

from .decoder import VideoDecoder, VideoInfo, probe_video
from .encoder import LEROBOT_CRF, LEROBOT_GOP, VideoEncoder
from .encoder_thread import VideoEncoderThread
from .frame_pipe_thread import FramePipeThread, slots_for
