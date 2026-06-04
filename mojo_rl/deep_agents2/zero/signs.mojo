"""Two-player zero-sum sign conventions for zero-series self-play targets.

Hardcoded 2-player zero-sum (decision D5). Centralizes the value-sign logic
that the legacy MuZero got wrong (``docs/MUZERO_AUDIT.md`` P0: the two-player
value-target sign flip was missing) and that AlphaZero applies at
data-generation time (``Coach.py:69``).

Convention: **all value targets are stored from the perspective of the player
to move at that timestep.** A value or reward observed from a different
player's perspective is negated (zero-sum). Result codes match the board envs:
0=ongoing, 1=P0 wins, 2=P1 wins, 3=draw.
"""

comptime RESULT_ONGOING: Int = 0
comptime RESULT_P0_WINS: Int = 1
comptime RESULT_P1_WINS: Int = 2
comptime RESULT_DRAW: Int = 3


@always_inline
def zero_sum_sign(player_a: Int, player_b: Int) -> Float64:
    """+1.0 if the two players match, -1.0 otherwise.

    Multiply any value/reward by this to transport it between two players'
    perspectives in a zero-sum game.
    """
    return 1.0 if player_a == player_b else -1.0


@always_inline
def az_value_target(result: Int, player_t: Int) -> Float64:
    """AlphaZero outcome target ``z_t`` for the example at timestep ``t``, from
    the perspective of ``player_t`` (the player to move at ``t``).

    +1 if ``player_t`` is the winner, -1 if the loser, 0 for draw / ongoing.
    Equivalent to ``Coach.py:69``: ``z = r·(-1)^(player_t != winner)``.
    """
    if result == RESULT_DRAW or result == RESULT_ONGOING:
        return 0.0
    var winner = result - 1  # 1→P0 (0), 2→P1 (1)
    return 1.0 if player_t == winner else -1.0


@always_inline
def flip_for_perspective(
    value: Float64, to_play_k: Int, to_play_t: Int
) -> Float64:
    """Transport a value/reward observed at step ``k`` into the perspective of
    the player to move at step ``t`` (zero-sum negation).

    Used by the MuZero n-step bootstrap (``replay_buffer.py:243-260``): the
    bootstrap value and each accumulated reward flip sign when
    ``to_play[k] != to_play[t]``. The legacy P0 bug was omitting exactly this.
    """
    return value * zero_sum_sign(to_play_k, to_play_t)
