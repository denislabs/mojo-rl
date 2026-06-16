"""Visit-count sampling temperature — the zero-series exploration schedule.

Self-play drivers sample actions ∝ visits^(1/T). T starts at 1.0 (pure
visit-proportional exploration) and steps down as training progresses so the
behavior policy sharpens and the replay buffer fills with near-greedy episodes
— without it the buffer stays full of exploratory deaths and the training
return badly understates the policy (legacy MuZero schedule, muzero.mojo:2334).

The *stored* policy target must stay the untempered visit distribution; only
the action SAMPLING is tempered.
"""


def visit_temperature(it: Int, decay_steps: Int) -> Float64:
    """Legacy piecewise schedule over ``decay_steps``:
    T = 1.0 → 0.5 (at 50% progress) → 0.25 (at 75%).
    ``decay_steps <= 0`` disables the schedule (always 1.0)."""
    if decay_steps <= 0:
        return 1.0
    var progress = Float64(it) / Float64(decay_steps)
    if progress >= 0.75:
        return 0.25
    if progress >= 0.5:
        return 0.5
    return 1.0
