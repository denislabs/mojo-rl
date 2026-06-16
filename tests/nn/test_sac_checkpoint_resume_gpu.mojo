"""SAC GPU checkpoint save/resume + GPU→CPU interchange (Phase 2).

Replaces the former placeholder (GPU checkpointing now implemented).
Builds a GPU SAC trainer (actor + twin critics + 3 Adam optimizers +
alpha ScalarAdam), saves, and verifies:
  1. GPU save → fresh-GPU load → re-save is byte-identical.
  2. loaded GPU trainer agrees on greedy (deterministic) actions.
  3. GPU→CPU interchange: a CPU trainer loads the GPU checkpoint and
     agrees on greedy actions (train-on-GPU → eval-on-CPU).

The post-training Adam-moment round-trip rigor (m/v/bias-correction
across many moments) is covered by `test_dqn_checkpoint_gpu.mojo`'s
8770-moment invariant; SAC reuses the same shared GPU Adam helpers.
SAC's alpha (ScalarAdam) round-trips α/m/v exactly; its bias-correction
`t` resets on the GPU path (documented accepted gap — eval ignores α).

Guards on DeviceContext (no-op on CPU-only CI).
Run: pixi run -e apple mojo run -I . tests/nn/test_sac_checkpoint_resume_gpu.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.deep_agents.sac.trainer import SACTrainer
from mojo_rl.deep_agents.training.blocks import (
    UniformSampleGpuStep, UniformSampleCpuStep,
)


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime BATCH = 64
comptime CAP = 10_000
comptime WARMUP = 256

comptime ACTOR = Sequential[
    Linear[OBS_DIM, 16],
    ReLU[16],
    Linear[16, 2 * ACT_DIM],
]
comptime CRITIC = Sequential[
    Linear[OBS_DIM + ACT_DIM, 16],
    ReLU[16],
    Linear[16, 1],
]

comptime GPU_SAMPLE = UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, CAP]
comptime CPU_SAMPLE = UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, CAP]

comptime CKPT = String("/tmp/sac_gpu_ckpt.ckpt")
comptime CKPT_B = String("/tmp/sac_gpu_ckpt_b.ckpt")


def _read_file(path: String) raises -> String:
    with open(path, "r") as f:
        return String(f.read())


def _obs(f: Scalar[DT]) -> List[Scalar[DT]]:
    var o = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    o[0] = f
    o[1] = -f
    o[2] = f * Scalar[DT](0.5)
    return o^


def _greedy_gpu(
    mut t: SACTrainer["gpu", GPU_SAMPLE, ACTOR, CRITIC],
    f: Scalar[DT],
) raises -> Scalar[DT]:
    var obs = _obs(f)
    var a = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    t.select_greedy_action(obs, a)
    return a[0]


def _greedy_cpu(
    mut t: SACTrainer["cpu", CPU_SAMPLE, ACTOR, CRITIC],
    f: Scalar[DT],
) raises -> Scalar[DT]:
    var obs = _obs(f)
    var a = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    t.select_greedy_action(obs, a)
    return a[0]


def test_sac_gpu_checkpoint() raises:
    print("--- SAC GPU checkpoint save/resume ---")
    try:
        var _probe = DeviceContext()
    except:
        print("  no accelerator — skipping")
        return

    var trainer = SACTrainer["gpu", GPU_SAMPLE, ACTOR, CRITIC].make(
        ctx=DeviceContext(), actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](3e-4), learning_starts=WARMUP,
    )
    trainer.save_state(CKPT)
    print("  saved GPU SAC trainer ->", CKPT)

    var loaded = SACTrainer["gpu", GPU_SAMPLE, ACTOR, CRITIC].make(
        ctx=DeviceContext(), actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](3e-4), learning_starts=WARMUP,
    )
    loaded.load_state(CKPT)
    loaded.save_state(CKPT_B)
    assert_equal(
        _read_file(CKPT), _read_file(CKPT_B),
        "SAC GPU checkpoint not byte-identical after save→load→save",
    )
    print("  [1] GPU re-save byte-identity OK")

    # Continuous actions: the v2 text format is not bit-exact (it
    # serializes fp32 via String(float)), so loaded params differ from
    # the original in low bits → use a tolerance, not exact equality.
    # (DQN/C51 use exact discrete-argmax equality; that's robust.)
    # 1e-2: the v2 text format keeps ~7 sig figs; through two Linear
    # layers + tanh the worst-case continuous-action drift is a few e-3.
    comptime TOL = Scalar[DT](1e-2)
    var max_dev = Scalar[DT](0.0)
    for i in range(20):
        var f = Scalar[DT](i) * Scalar[DT](0.1) - Scalar[DT](1.0)
        var d = abs(_greedy_gpu(trainer, f) - _greedy_gpu(loaded, f))
        if d > max_dev:
            max_dev = d
    assert_true(
        max_dev < TOL,
        "loaded SAC GPU greedy deviates beyond tol (max="
        + String(max_dev) + ")",
    )
    print("  [2] greedy-action agreement OK ( max dev", max_dev, "< 1e-2 )")

    var cpu = SACTrainer["cpu", CPU_SAMPLE, ACTOR, CRITIC].make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](3e-4), learning_starts=WARMUP,
    )
    cpu.load_state(CKPT)
    var max_dev_x = Scalar[DT](0.0)
    for i in range(20):
        var f = Scalar[DT](i) * Scalar[DT](0.1) - Scalar[DT](1.0)
        var d = abs(_greedy_gpu(trainer, f) - _greedy_cpu(cpu, f))
        if d > max_dev_x:
            max_dev_x = d
    assert_true(
        max_dev_x < TOL,
        "SAC GPU→CPU interchange: greedy deviates beyond tol (max="
        + String(max_dev_x) + ")",
    )
    print("  [3] GPU→CPU interchange OK ( CPU eval max dev", max_dev_x,
          "< 1e-2 )")
    print("PASS")


def main() raises:
    test_sac_gpu_checkpoint()
