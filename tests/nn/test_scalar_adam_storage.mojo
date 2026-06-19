"""ScalarAdam CPU↔GPU parity (storage surface).

The CPU host-scalar `step(grad)` and the GPU 1-thread `step_device` (reading
grad = -(lp_mean + H_target) on-device) must track to fp tolerance over several
steps — both run the same incremental-bias-correction Adam on log_alpha and
expose α = exp(log_alpha).

Run: pixi run -e apple mojo run -I . tests/nn/test_scalar_adam_storage.mojo
"""

from std.math import exp as fexp
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.optimizer.scalar_adam import ScalarAdam


def main() raises:
    print("ScalarAdam CPU↔GPU parity")
    comptime STEPS = 8
    var lp = Scalar[DT](0.5)
    var H = Scalar[DT](-3.0)
    var grad = -(lp + H)  # the AlphaUpdateStep grad expression
    var lr = Scalar[DT](1e-3)

    var cpu = ScalarAdam.new(Scalar[DT](0.0), lr)
    for _ in range(STEPS):
        cpu.step(grad)
    var cpu_alpha = fexp(cpu.value)

    var ctx = DeviceContext()
    var gpu = ScalarAdam.new_device(ctx, Scalar[DT](0.0), lr)
    var lp_buf = ctx.enqueue_create_buffer[DT](1)
    var lp_host = ctx.enqueue_create_host_buffer[DT](1)
    ctx.synchronize()
    lp_host[0] = lp
    ctx.enqueue_copy(lp_buf, lp_host)
    ctx.synchronize()
    for _ in range(STEPS):
        gpu.step_device(ctx, lp_buf, H)
    var gpu_alpha = gpu.read_alpha()

    print("  cpu alpha:", cpu_alpha, " gpu alpha:", gpu_alpha)
    var ok = abs(cpu_alpha - gpu_alpha) < Scalar[DT](1e-5)
    assert_true(ok, "ScalarAdam CPU/GPU parity")
    print("SCALAR ADAM OK")
