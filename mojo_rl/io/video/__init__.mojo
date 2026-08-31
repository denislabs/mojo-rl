"""Video decode and encode for datasets, both by piping `ffmpeg`.

See `decoder.mojo` for why a pipe rather than a libavcodec binding, and
`encoder.mojo` for the LeRobot encoding settings (`-g 2` in particular).
RGB24 is the pixel format in both directions.
"""

from .decoder import VideoDecoder, VideoInfo, probe_video
from .encoder import LEROBOT_CRF, LEROBOT_GOP, VideoEncoder
