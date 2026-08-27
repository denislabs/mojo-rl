"""`TaskEmbedding` row-norm ceiling — reference `nn.Embedding(..., max_norm=1)`.

The reference bounds every task-embedding row to unit norm
(`references/tdmpc2-main/tdmpc2/common/world_model.py:21`). Ours did not, and
nothing downstream pushes back: the encoder's first layer is
`Linear -> LayerNorm -> Mish`, so the loss is largely invariant to a shared
component in the embedding and there is no restoring force on its magnitude.

Measured on a 130k-step walker stand+walk+run run before the fix: the three
rows grew 22x (norm 0.55 -> 11.9) while their pairwise cosines converged to
0.9995 — about 96% of every row was one SHARED direction carrying no task
information at all.

Gated on both targets because the projection is written twice (a host loop and
`_renorm_k`), and the GPU path is the one that actually runs in training.

Run:
  pixi run mojo run -I . tests/deep_agents/test_tdmpc2_task_embedding_max_norm.mojo
  pixi run -e apple mojo run -I . tests/deep_agents/test_tdmpc2_task_embedding_max_norm.mojo
"""

from std.math import sqrt, abs
from std.testing import assert_true, assert_almost_equal, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.task_embedding import TaskEmbedding

comptime NUM_TASKS = 3
comptime TASK_EMB = 8
comptime Emb = TaskEmbedding[NUM_TASKS, TASK_EMB]


def _norm(ref d: List[Scalar[DT]], t: Int) -> Float64:
    var s = 0.0
    for e in range(TASK_EMB):
        var x = Float64(d[t * TASK_EMB + e])
        s += x * x
    return sqrt(s)


def _set_row(mut te: Emb, t: Int, val: Float64):
    for e in range(TASK_EMB):
        te.param.data[t * TASK_EMB + e] = Scalar[DT](val)


def test_cpu_projects_long_rows_and_leaves_short_ones() raises:
    var te = Emb.make["cpu"]()
    # row 0: norm = 4*sqrt(8) ~ 11.3, far outside — the shape the walker run
    # actually drifted into. row 1: norm = sqrt(8)/8 ~ 0.35, inside. row 2:
    # exactly at the boundary.
    _set_row(te, 0, 4.0)
    _set_row(te, 1, 0.125)
    _set_row(te, 2, 1.0 / sqrt(Float64(TASK_EMB)))
    var before1 = _norm(te.param.data, 1)
    var before2 = _norm(te.param.data, 2)

    # grad is zero, so Adam moves nothing — any change is the projection.
    te.step["cpu"]()

    print("  cpu norms:", _norm(te.param.data, 0), _norm(te.param.data, 1),
          _norm(te.param.data, 2))
    assert_almost_equal(
        Scalar[DT](_norm(te.param.data, 0)), Scalar[DT](1.0), atol=1e-5,
        msg="a row past the ceiling must be projected back onto it",
    )
    assert_almost_equal(
        Scalar[DT](_norm(te.param.data, 1)), Scalar[DT](before1), atol=1e-6,
        msg="a row inside the ball must be left alone",
    )
    assert_almost_equal(
        Scalar[DT](_norm(te.param.data, 2)), Scalar[DT](before2), atol=1e-6,
        msg="a row exactly at the ceiling must be left alone",
    )
    # Projection scales, it does not rotate: every element of the uniform row
    # must remain equal.
    for e in range(TASK_EMB):
        assert_almost_equal(
            te.param.data[e], te.param.data[0], atol=1e-6,
            msg="projection must preserve direction",
        )


def test_cpu_disabled_when_max_norm_non_positive() raises:
    """`max_norm <= 0` opts out — the escape hatch must actually escape."""
    var te = Emb.make["cpu"]()
    te.max_norm = Scalar[DT](0.0)
    _set_row(te, 0, 4.0)
    var before = _norm(te.param.data, 0)
    te.step["cpu"]()
    assert_almost_equal(
        Scalar[DT](_norm(te.param.data, 0)), Scalar[DT](before), atol=1e-6,
        msg="max_norm <= 0 must leave the table untouched",
    )
    print("  disabled → norm stays", _norm(te.param.data, 0))


def test_gpu_projects_long_rows() raises:
    """The path that runs in training — written separately as `_renorm_k`."""
    var ctx = DeviceContext()
    var te = Emb.make["gpu"](ctx=ctx)
    _set_row(te, 0, 4.0)
    _set_row(te, 1, 0.125)
    te.upload_from_host()
    var before1 = _norm(te.param.data, 1)

    te.step["gpu"]()
    te.sync_to_host()

    print("  gpu norms:", _norm(te.param.data, 0), _norm(te.param.data, 1))
    assert_almost_equal(
        Scalar[DT](_norm(te.param.data, 0)), Scalar[DT](1.0), atol=1e-4,
        msg="GPU projection must clamp a long row to the ceiling",
    )
    assert_almost_equal(
        Scalar[DT](_norm(te.param.data, 1)), Scalar[DT](before1), atol=1e-5,
        msg="GPU projection must leave a short row alone",
    )
    for e in range(TASK_EMB):
        assert_almost_equal(
            te.param.data[e], te.param.data[0], atol=1e-5,
            msg="GPU projection must preserve direction",
        )


def test_repeated_steps_stay_bounded() raises:
    """The failure being prevented is DRIFT, not a single large value — so the
    bound has to hold under repeated updates, which is how the 22x growth
    accumulated in the first place."""
    var ctx = DeviceContext()
    var te = Emb.make["gpu"](ctx=ctx)
    for t in range(NUM_TASKS):
        _set_row(te, t, 0.3)
    te.upload_from_host()
    # A constant gradient every step is the pathological case: Adam's update is
    # ~lr in magnitude per element and never decays, so the row walks outward
    # without bound unless something projects it back.
    for i in range(NUM_TASKS * TASK_EMB):
        te.grad.data[i] = Scalar[DT](-1.0)
    te.grad.upload(ctx)
    for _ in range(200):
        te.step["gpu"]()
    te.sync_to_host()
    var n0 = _norm(te.param.data, 0)
    print("  after 200 steps of constant gradient: norm =", n0)
    assert_true(
        n0 <= 1.0 + 1e-4,
        "the row escaped the ceiling under repeated updates — this is exactly"
        " the unbounded drift the projection exists to stop",
    )


def main() raises:
    print("=" * 70)
    print("TaskEmbedding max_norm projection")
    print("=" * 70)
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("=" * 70)
    print("MAX_NORM GATE PASSED")
    print("=" * 70)
