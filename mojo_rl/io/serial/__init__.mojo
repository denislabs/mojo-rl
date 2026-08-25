# +--------------------------------------------------------------------------+ #
# | mojo-rl serial I/O
# +--------------------------------------------------------------------------+ #
"""Raw-mode serial ports over libc, for talking to hardware.

`SerialPort` is device-agnostic: it opens a tty, configures 8N1 raw at an
arbitrary baud, and moves bytes against a deadline. Protocol lives on top —
see `mojo_rl/robot/feetech/` for the SO-101's bus servos.

⚠ Requires the shim: `pixi run build-serial`. `native.mojo` explains why six
lines of C are unavoidable (`ioctl` is C-variadic; Mojo's `external_call`
cannot express that, and the failure is a silent EFAULT).
"""

from mojo_rl.io.serial.port import SerialPort, errno
from mojo_rl.io.serial.native import set_speed
