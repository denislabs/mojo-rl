"""Test — Logger plumbing on the SAC trainer.

Phase A.5 validation. Three sub-tests:

  1. `log_bundle` correctly emits one log_scalar per Metric field of a
     SACMetrics bundle. Field names match the struct's source order.
  2. `SACTrainer.flush_metrics` (no logger) returns the bundle and
     resets accumulators. Bit-identity gate (no log calls, no D2H).
  3. `SACTrainer.flush_metrics` with a real ListLogger pointer emits
     one log_scalar call per SACMetrics field at the specified step.
     After K flushes there are exactly K × n_fields recorded calls.

This file is fast (no actual training). The full bit-identity check
(SAC Pendulum 30k → mean_ret(10) = -167.572) runs separately.
"""

from std.random import seed
from std.testing import assert_equal, assert_true

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.metric import LogScalar
from mojo_rl.nn.core.log_bundle import log_bundle
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac.trainer import SACTrainer
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents.sac.metrics import SACMetrics


# ──────────────────────────────────────────────────────────────────────
# ListLogger — records every log_scalar call so we can assert on counts
# + names + values + steps.
# ──────────────────────────────────────────────────────────────────────


struct ListLogger(Logger):
    var names: List[String]
    var values: List[Float64]
    var steps: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.values = List[Float64]()
        self.steps = List[Int]()

    def __init__(out self, *, copy: Self):
        self.names = copy.names.copy()
        self.values = copy.values.copy()
        self.steps = copy.steps.copy()

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        self.names.append(name)
        self.values.append(value)
        self.steps.append(step)

    def log_scalars(
        mut self,
        names: List[String],
        values: List[Float64],
        step: Int,
    ) raises:
        for i in range(len(names)):
            self.log_scalar(names[i], values[i], step)

    def flush(mut self) raises:
        pass

    def close(mut self) raises:
        pass

    def set_config(mut self, key: String, value: String):
        pass

    def is_active(self) -> Bool:
        return True


# ──────────────────────────────────────────────────────────────────────
# Test 1: log_bundle walks SACMetrics → ListLogger.
# ──────────────────────────────────────────────────────────────────────


def test_log_bundle_walks_sac_metrics() raises:
    print("test_log_bundle_walks_sac_metrics ...")
    var m = SACMetrics(
        actor_loss=LogScalar[DT](Scalar[DT](0.42)),
        critic_loss=LogScalar[DT](Scalar[DT](-1.7)),
        alpha=LogScalar[DT](Scalar[DT](0.2)),
        mean_q=LogScalar[DT](Scalar[DT](1.0)),
        mean_target=LogScalar[DT](Scalar[DT](1.1)),
        mean_reward=LogScalar[DT](Scalar[DT](-0.5)),
        mean_next_q=LogScalar[DT](Scalar[DT](0.9)),
        mean_done=LogScalar[DT](Scalar[DT](0.0)),
        mean_abs_action=LogScalar[DT](Scalar[DT](0.3)),
        train_steps=LogScalar[DT](Scalar[DT](1024.0)),
        n_updates=LogScalar[DT](Scalar[DT](256.0)),
    )
    var logger = ListLogger()
    log_bundle(logger, m, 7)

    assert_equal(len(logger.names), 11, "SACMetrics has 11 Metric fields")
    assert_equal(logger.names[0], String("actor_loss"))
    assert_equal(logger.names[1], String("critic_loss"))
    assert_equal(logger.names[2], String("alpha"))
    assert_equal(logger.names[3], String("mean_q"))
    assert_equal(logger.names[4], String("mean_target"))
    assert_equal(logger.names[5], String("mean_reward"))
    assert_equal(logger.names[6], String("mean_next_q"))
    assert_equal(logger.names[7], String("mean_done"))
    assert_equal(logger.names[8], String("mean_abs_action"))
    assert_equal(logger.names[9], String("train_steps"))
    assert_equal(logger.names[10], String("n_updates"))
    for i in range(11):
        assert_equal(logger.steps[i], 7)
    print("  ok (11 log_scalar calls)")


# ──────────────────────────────────────────────────────────────────────
# Test 2: NoOpLogger short-circuits at comptime — no log calls.
# ──────────────────────────────────────────────────────────────────────


def test_noop_logger_short_circuits() raises:
    """The `comptime if not L.ENABLED: return` at the top of log_bundle
    means a NoOpLogger reaches no field reads. Nothing to assert beyond
    'this compiles and runs without error' — the real verification is
    via the bit-identity gate (no D2H, no perf shift)."""
    print("test_noop_logger_short_circuits ...")
    var m = SACMetrics(
        actor_loss=LogScalar[DT](Scalar[DT](0.0)),
        critic_loss=LogScalar[DT](Scalar[DT](0.0)),
        alpha=LogScalar[DT](Scalar[DT](0.0)),
        mean_q=LogScalar[DT](Scalar[DT](0.0)),
        mean_target=LogScalar[DT](Scalar[DT](0.0)),
        mean_reward=LogScalar[DT](Scalar[DT](0.0)),
        mean_next_q=LogScalar[DT](Scalar[DT](0.0)),
        mean_done=LogScalar[DT](Scalar[DT](0.0)),
        mean_abs_action=LogScalar[DT](Scalar[DT](0.0)),
        train_steps=LogScalar[DT](Scalar[DT](0.0)),
        n_updates=LogScalar[DT](Scalar[DT](0.0)),
    )
    var logger = NoOpLogger()
    log_bundle(logger, m, 0)
    print("  ok (NoOpLogger comptime-elided)")


# ──────────────────────────────────────────────────────────────────────
# Test 3: SACTrainer.flush_metrics emits via wired Logger pointer.
# Builds a real SAC trainer, hand-injects fake accumulator values,
# flushes through a ListLogger, asserts the emitted lines.
# ──────────────────────────────────────────────────────────────────────


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY = 50_000
comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime SACT = SACTrainer[
    "cpu",
    UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY],
    ActorNet,
    CriticNet,
]


def test_trainer_flush_metrics_emits() raises:
    print("test_trainer_flush_metrics_emits ...")
    seed(42)
    var trainer = SACT.make()

    # Hand-inject 3 fake "updates" worth of accumulated losses.
    trainer._actor_L_accum = Scalar[DT](6.0)  # mean over 3 → 2.0
    trainer._critic_L_accum = Scalar[DT](3.0)  # mean over 3 → 1.0
    trainer._alpha_accum = Scalar[DT](0.6)  # mean over 3 → 0.2
    trainer._update_count = 3

    var logger = ListLogger()
    var bundle = trainer.flush_metrics(
        logger=Optional[UnsafePointer[ListLogger, MutAnyOrigin]](
            UnsafePointer(to=logger)
        ),
        step=500,
    )
    _ = logger  # lifetime extender — trainer holds raw pointer.

    # 11 SACMetrics fields → 11 log_scalar calls.
    assert_equal(len(logger.names), 11)
    assert_equal(logger.names[0], String("actor_loss"))
    assert_equal(logger.names[1], String("critic_loss"))
    assert_equal(logger.names[2], String("alpha"))
    assert_equal(logger.names[10], String("n_updates"))

    # Values match the means. The diag-walk fields (mean_q ... train_steps)
    # are 0 here because no train_step ran — only loss accumulators were
    # hand-injected.
    assert_true((logger.values[0] - 2.0).__abs__() < 1e-5, "actor_loss=2.0")
    assert_true((logger.values[1] - 1.0).__abs__() < 1e-5, "critic_loss=1.0")
    assert_true((logger.values[2] - 0.2).__abs__() < 1e-5, "alpha=0.2")
    assert_true((logger.values[10] - 3.0).__abs__() < 1e-9, "n_updates=3")

    # Step plumbed through.
    for i in range(11):
        assert_equal(logger.steps[i], 500)

    # Returned bundle matches the emitted values.
    assert_true(
        (bundle.actor_loss.v - Scalar[DT](2.0)).__abs__() < Scalar[DT](1e-5)
    )
    assert_true(
        (bundle.critic_loss.v - Scalar[DT](1.0)).__abs__() < Scalar[DT](1e-5)
    )
    assert_true((bundle.alpha.v - Scalar[DT](0.2)).__abs__() < Scalar[DT](1e-5))
    assert_equal(Int(bundle.n_updates.v), 3)

    # Accumulators reset.
    assert_equal(trainer._update_count, 0)
    assert_true(trainer._actor_L_accum == Scalar[DT](0.0))
    assert_true(trainer._critic_L_accum == Scalar[DT](0.0))
    assert_true(trainer._alpha_accum == Scalar[DT](0.0))

    print("  ok (11 log_scalar calls, accumulators reset)")


# ──────────────────────────────────────────────────────────────────────
# Test 4: Multiple flushes accumulate logger entries.
# K=5 flushes × 11 fields = 55 log_scalar calls, monotonically
# increasing steps.
# ──────────────────────────────────────────────────────────────────────


def test_multiple_flushes_accumulate() raises:
    print("test_multiple_flushes_accumulate ...")
    seed(42)
    var trainer = SACT.make()
    var logger = ListLogger()
    var logger_ptr = Optional[UnsafePointer[ListLogger, MutAnyOrigin]](
        UnsafePointer(to=logger)
    )

    comptime K = 5
    for i in range(K):
        trainer._actor_L_accum = Scalar[DT](Float64(i + 1))
        trainer._critic_L_accum = Scalar[DT](Float64(i + 1))
        trainer._alpha_accum = Scalar[DT](Float64(i + 1) * 0.1)
        trainer._update_count = 1
        _ = trainer.flush_metrics(logger=logger_ptr, step=100 * (i + 1))
    _ = logger  # lifetime extender

    assert_equal(len(logger.names), K * 11)
    # Check that step values cycle 100×11, 200×11, ...
    for i in range(K):
        for f in range(11):
            assert_equal(logger.steps[i * 11 + f], 100 * (i + 1))
    print("  ok (", K * 11, "log_scalar calls across", K, "flushes)")


def main() raises:
    print("=" * 70)
    print("Metrics + Logger plumbing (Phase A.5)")
    print("=" * 70)
    test_log_bundle_walks_sac_metrics()
    test_noop_logger_short_circuits()
    test_trainer_flush_metrics_emits()
    test_multiple_flushes_accumulate()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
