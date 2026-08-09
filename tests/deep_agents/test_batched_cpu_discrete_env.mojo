"""BatchedCpuDiscreteEnv — parallel-step parity + no-op starts + resets.

Uses the RAM-mode Atari Pong emulator as the env: it is fully
deterministic (no RNG anywhere), so the multi-core `step_batch` has an
exact serial oracle — N independent envs stepped one by one with the
same action schedule must produce bit-identical obs/reward/done/term.

Three checks:
  (A) parallel step_batch ≡ serial per-env stepping (150 steps, 4 envs).
  (B) noop_max > 0 decorrelates the reset states (lanes differ).
  (C) max_frames truncation: done=1 but terminated=0 (bootstrap kept),
      and selective_reset_batch restarts only the done lanes.

Requires `roms/pong.bin` (run from the repo root).

Run:
    pixi run mojo run -I . tests/deep_agents/test_batched_cpu_discrete_env.mojo
"""

from std.testing import assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.training.batched_env import BatchedCpuDiscreteEnv
from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


comptime N = 4
comptime OBS = 128  # RAM mode
comptime STEPS = 150

comptime PongRam = AtariEnv[0]


def _make_envs(
    rom: Pointer[UInt8, MutAnyOrigin],
    rom_size: Int,
    max_frames: Int = 108000,
) -> List[PongRam]:
    var envs = List[PongRam]()
    for _ in range(N):
        envs.append(
            PongRam(AtariGame.PONG, rom, rom_size, max_frames=max_frames)
        )
    return envs^


def main() raises:
    var rom = load_rom("roms/pong.bin")

    # ── (A) parallel step_batch ≡ serial per-env stepping ──
    var batched = BatchedCpuDiscreteEnv[PongRam, N, OBS](
        _make_envs(rom.data.value().as_unsafe_any_origin(), rom.size)
    )
    batched.reset_batch[N](ctx=None, rng_seed=UInt64(7))

    var serial = _make_envs(rom.data.value().as_unsafe_any_origin(), rom.size)
    for i in range(N):
        _ = serial[i].reset()

    var act = batched.action_ptr()
    var obs = batched.obs_ptr()
    var rew = batched.reward_ptr()
    var done = batched.done_ptr()
    for t in range(STEPS):
        for i in range(N):
            act[i] = Scalar[DT]((t + i) % 6)
        batched.step_batch[N](ctx=None, rng_seed=UInt64(t))
        for i in range(N):
            var res = serial[i].step_obs((t + i) % 6)
            for d in range(OBS):
                assert_equal(
                    obs[i * OBS + d],
                    Scalar[DT](res[0][d]),
                    String("obs mismatch env ") + String(i),
                )
            assert_equal(rew[i], Scalar[DT](res[1]))
            assert_equal(
                done[i],
                Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
            )
    print("(A) parallel step parity vs serial oracle: OK")

    # ── (B) no-op starts decorrelate the reset lanes ──
    var noop_env = BatchedCpuDiscreteEnv[PongRam, N, OBS](
        _make_envs(rom.data.value().as_unsafe_any_origin(), rom.size), noop_max=30
    )
    noop_env.reset_batch[N](ctx=None, rng_seed=UInt64(42))
    var nobs = noop_env.obs_ptr()
    var any_differ = False
    for i in range(N):
        for j in range(i + 1, N):
            for d in range(OBS):
                if nobs[i * OBS + d] != nobs[j * OBS + d]:
                    any_differ = True
                    break
    assert_true(any_differ, "noop_max=30 reset lanes must differ")
    print("(B) noop starts decorrelate lanes: OK")

    # ── (C) max_frames truncation + selective reset ──
    # Reset itself burns ~70-100 frames (title screen + RESET hold), then
    # frame_skip=4 per step → max_frames=400 truncates within ~80 steps.
    var trunc_env = BatchedCpuDiscreteEnv[PongRam, N, OBS](
        _make_envs(rom.data.value().as_unsafe_any_origin(), rom.size, max_frames=400)
    )
    trunc_env.reset_batch[N](ctx=None, rng_seed=UInt64(3))
    var tact = trunc_env.action_ptr()
    var tdone = trunc_env.done_ptr()
    var tterm = trunc_env.terminated_ptr()
    var saw_done = False
    for t in range(120):
        for i in range(N):
            tact[i] = Scalar[DT](0)
        trunc_env.step_batch[N](ctx=None, rng_seed=UInt64(t))
        for i in range(N):
            if tdone[i] > Scalar[DT](0.5):
                saw_done = True
                # Time-limit truncation is NOT a natural termination.
                assert_equal(
                    tterm[i],
                    Scalar[DT](0.0),
                    "truncation must not set terminated",
                )
        if saw_done:
            break
    assert_true(saw_done, "max_frames=400 must truncate within 120 steps")

    # Selective reset restarts the done lanes; another step succeeds and
    # leaves every lane un-done (fresh episodes).
    trunc_env.selective_reset_batch[N](ctx=None, rng_seed=UInt64(9))
    for i in range(N):
        tact[i] = Scalar[DT](0)
    trunc_env.step_batch[N](ctx=None, rng_seed=UInt64(99))
    for i in range(N):
        assert_equal(
            tdone[i], Scalar[DT](0.0), "post-reset lane must be un-done"
        )
    print("(C) truncation flags + selective reset: OK")

    print("test_batched_cpu_discrete_env: ALL OK")
