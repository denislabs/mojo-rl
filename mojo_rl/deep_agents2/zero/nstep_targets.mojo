"""MuZero n-step value targets with two-player sign flips (the legacy P0 bug).

For each unroll position ``k`` the value target is the n-step bootstrap

    V_target[k] = Σ_{i=0}^{n-1} γⁱ · r̃_{k+i}  +  γⁿ · Ṽ_root(s_{k+n})

computed **from the perspective of the player to move at step k**. In a
two-player zero-sum game every reward and the bootstrap value is negated when
the player to move at that future step differs from the perspective player
(``replay_buffer.py:243-260``). The legacy MuZero **omitted this flip**
(``docs/MUZERO_AUDIT.md`` P0) — it is centralized here via
`zero/signs.mojo::flip_for_perspective` and pinned by a hand-computed
two-player unit test.

Single-player envs (CartPole) pass ``to_play`` all-zero, so every
`zero_sum_sign` is +1 and the flips vanish — the same code path serves both.

The accumulation stops at the first terminal inside the n-window (terminal
states have value 0, so there is no bootstrap past them). Inputs are one
trajectory window; the self-play driver loops this over the sampled batch.
Stored ``root_values`` are real-scalar (the planner's visit-weighted Q, already
``h⁻¹``-decoded); the ``h``/two-hot transform is applied later at encode time.
"""

from mojo_rl.nn2.constants import DT
from .signs import flip_for_perspective


def compute_nstep_value_targets[
    K: Int, N: Int,
](
    rewards: UnsafePointer[Scalar[DT], MutAnyOrigin],
    dones: UnsafePointer[Scalar[DT], MutAnyOrigin],
    root_values: UnsafePointer[Scalar[DT], MutAnyOrigin],
    to_play: UnsafePointer[Scalar[DT], MutAnyOrigin],
    gamma: Scalar[DT],
    mut value_targets: UnsafePointer[Scalar[DT], MutAnyOrigin],
    last_valid: Int = K + N + 1,
):
    """Fill ``value_targets[0..K]`` for one trajectory window.

    Window sizing (the self-play driver guarantees these lengths):
      * ``rewards[K+N]``, ``dones[K+N]``    — per-step reward / terminal flag
      * ``root_values[K+N+1]``               — MCTS root value per position (real)
      * ``to_play[K+N+1]``                    — player to move per position (0/1)

    ``dones[i] == 1`` marks step ``i`` as terminal. ``gamma`` is the discount.

    ``last_valid`` handles **time-limit truncation**: it is the last window
    index that still carries real data (a stored root value). When the n-step
    window would read past it, the reward sum stops there and the bootstrap
    uses ``root_values[last_valid]`` — truncation is NOT a terminal (a state
    cut off by a time limit does not have value 0, and labelling it so
    corrupts the value head for every state that looks like it). The default
    (``K+N+1``, beyond the window) never caps — terminal ``dones`` handle
    naturally-ended episodes.
    """
    for k in range(K + 1):
        var perspective = Int(to_play[k])
        var n_return = Scalar[DT](0.0)
        var gamma_pow = Scalar[DT](1.0)
        var hit_terminal = False
        # cap the reward sum at the truncation boundary (no-op by default).
        var n_lim = N
        if k + n_lim > last_valid:
            n_lim = last_valid - k
            if n_lim < 0:
                n_lim = 0
        for i in range(n_lim):
            var step_idx = k + i
            var r = rewards[step_idx]
            # Flip the reward into the perspective player's frame.
            var r_flipped = Scalar[DT](
                flip_for_perspective(
                    Float64(r), Int(to_play[step_idx]), perspective
                )
            )
            n_return += gamma_pow * r_flipped
            gamma_pow *= gamma
            if dones[step_idx] > Scalar[DT](0.5):
                hit_terminal = True
                break
        # Bootstrap with the MCTS root value n steps out, unless a terminal
        # cut the sum short (terminal value == 0). On truncation the bootstrap
        # index is clamped to the last position with a stored root value.
        if not hit_terminal:
            var boot_idx = k + n_lim
            if boot_idx > last_valid:
                boot_idx = last_valid
            var boot_v = root_values[boot_idx]
            var boot_flipped = Scalar[DT](
                flip_for_perspective(
                    Float64(boot_v), Int(to_play[boot_idx]), perspective
                )
            )
            n_return += gamma_pow * boot_flipped
        value_targets[k] = n_return


def extract_reward_targets[
    K: Int, N: Int,
](
    rewards: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mut reward_targets: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Reward targets for the ``K`` dynamics steps: the raw reward received
    after each unrolled action (``reward_targets[k] = rewards[k]``). MuZero
    predicts the reward in the acting player's own frame, so — unlike the
    value target — no perspective flip is applied (legacy parity). The ``h`` /
    two-hot transform is applied downstream at encode time.
    """
    for k in range(K):
        reward_targets[k] = rewards[k]
