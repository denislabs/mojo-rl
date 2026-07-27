"""TD-MPC2 multi-task GPU smoke (Apple Metal) — item C, §14.3.

GPU counterpart of `test_tdmpc2_multitask_smoke.mojo`. The CPU smoke covers the
multi-task train stack on the host; this one drives the SAME synthetic two-task
data through the `"gpu"` agent so the multi-task GPU train blocks actually run —
specifically `TDTargetStepMT._td_gpu` and `PolicyStepMT._pol_gpu`, whose
persistent-scratch refactor (no per-step `enqueue_create_buffer`) was otherwise
untested (the only prior MT-GPU test, `test_tdmpc2_mt_wm_gpu_parity`, exercises
the WM step alone).

NOTE: the GPU agent REQUIRES `ctx=ctx` at construction (the preset's `ctx`
defaults to `None`; without it the GPU net factories raise "ctx required").

Random actions throughout (no `select_action`) keep the test focused on
`train_step`. Asserts the WM loss stays finite and decreases over training.

Run: `pixi run -e apple mojo run -I . tests/deep_agents/test_tdmpc2_multitask_gpu_smoke.mojo`
"""

from std.random import random_float64, seed
from std.math import isfinite
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.config_mt import TDMPC2MultiTask

comptime MAX_OBS = 4
comptime MAX_ACT = 1
comptime NUM_TASKS = 2
comptime TASK_EMB = 8
comptime B = 4
comptime CAP = 4096
comptime ENC = 64
comptime LATENT = 64
comptime MLP = 64
comptime BINS = 51


def main() raises:
    print("=" * 70)
    print("TD-MPC2 multi-task GPU smoke (Apple) — MT td-target + policy steps")
    print("=" * 70)
    seed(11)

    var ctx = DeviceContext()
    var ag = TDMPC2MultiTask[
        "gpu", MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
        ENC, LATENT, MLP, BINS,
    ](ctx=ctx, lr=Scalar[DT](1e-3), learning_starts=64)

    var obsbuf = List[Scalar[DT]](length=MAX_OBS, fill=Scalar[DT](0))
    var actbuf = List[Scalar[DT]](length=MAX_ACT, fill=Scalar[DT](0))

    comptime TOTAL = 400
    comptime LEARN_START = 64
    comptime TRAIN_EVERY = 6
    var first_wm: Scalar[DT] = 0.0
    var last_wm: Scalar[DT] = 0.0
    var n_train = 0

    for step in range(TOTAL):
        var task = step % NUM_TASKS
        ag.set_task(task)
        # synthetic obs: task 0 fills [0:3], leaves [3]=0; task 1 fills all 4.
        var ndim = 3 if task == 0 else 4
        for i in range(MAX_OBS):
            obsbuf[i] = Scalar[DT](0.0)
        for i in range(ndim):
            obsbuf[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        actbuf[0] = Scalar[DT](random_float64() * 2.0 - 1.0)
        var r = Scalar[DT](random_float64() - 0.5)
        ag.record(obsbuf, actbuf, r, Scalar[DT](0.0))
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            if ag.train_step():
                var wm = ag.last_wm_loss()
                assert_true(isfinite(wm), "WM finite")
                if n_train == 0:
                    first_wm = wm
                last_wm = wm
                n_train += 1

    print("  trained", n_train, "steps; WM:", first_wm, "->", last_wm)
    assert_true(n_train > 0, "should have trained")
    assert_true(last_wm < first_wm, "WM loss should decrease")
    print("=" * 70)
    print("MT GPU SMOKE PASSED — MT td-target + policy train blocks run on GPU")
    print("=" * 70)
