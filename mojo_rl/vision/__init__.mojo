"""Vision — camera capture, fiducial detection, pose and calibration.

A new package rather than a corner of `mojo_rl/io/`: capture is I/O, but
detection, pose and camera calibration are not, and the three want to sit
together. `docs/OPENCV_SHIM_SCOPE.md` is the plan this is being built to.
"""

from .camera_thread import CameraReader
