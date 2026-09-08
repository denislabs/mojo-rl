"""THE SHAPED REWARD'S PARAMETERS, PER LANE — the host half.

    var sw = shaping_words(w_goal, w_reach, goal_margin, reach_margin)
    for j in range(len(sw)):
        meta[env * METADATA_SIZE + META_IDX_SHAPE_W_GOAL + j] = sw[j]

## ⚠⚠ WHY PER LANE AND NOT PER RUN

They lived in `curriculum`, which is `[1, MODEL_CURRICULUM_SIZE]` — one row for
the whole batch. That is right for the region table, because a region belongs
to the FAMILY, and wrong for shaping, because what a weight is WORTH depends on
the task's own distance scale. Measured on `so101_tabletop` at identical
weights (1.0 / 0.5) and identical margins (0.10 / 0.20):

    task     op      goal dist   goal term   total reward
    gather   Near        0.139       0.011          0.292
    lift     Above       0.030       0.811          1.171
    settle   On          0.000       1.000          1.360

A 4.7x spread in the reward and 91x in the goal term, from the same four
numbers. `Near`'s distance is a separation between two props that starts near
0.14 m; `Above`'s is a z-shortfall of 0.03 m; `On`'s is zero at reset by
construction. One margin cannot serve all three, and a two-task batch under one
would hand its lanes a bimodal reward — which on this family is how a critic
gets destabilised.

## ⚠ ZERO IS "NO SHAPING", WHICH IS WHAT AN UNTOUCHED `meta` HOLDS

`Data` uploads a zero-filled `meta`, so a driver that never writes these gets
the SPARSE reward — the goal bit and nothing else — instead of a shaped one
with meaningless parameters. Same bias-toward-safe as the init-region words.
"""

from mojo_rl.physics3d.gpu.constants import (
    META_IDX_SHAPE_W_GOAL, META_IDX_SHAPE_W_REACH,
    META_IDX_GOAL_MARGIN, META_IDX_REACH_MARGIN,
)


comptime SHAPING_WORDS: Int = 4
"""How many `meta` words `shaping_words` returns, starting at
`META_IDX_SHAPE_W_GOAL`. ⚠ THE FOUR ARE CONTIGUOUS AND IN THIS ORDER so a
caller can write them with one loop; `constants.mojo` is where that adjacency
is declared and this only restates the count."""


def shaping_words(
    w_goal: Float64, w_reach: Float64,
    goal_margin: Float64, reach_margin: Float64,
) raises -> List[Float64]:
    """`[w_goal, w_reach, goal_margin, reach_margin]`, validated.

    ⚠ NEGATIVE WEIGHTS ARE REFUSED. They multiply a `tolerance` that is LARGER
    nearer the goal, so a negative one pays the policy to move away — and it
    would train toward exactly that, with a perfectly healthy critic.

    ⚠⚠ A NONZERO WEIGHT WITH A ZERO MARGIN IS REFUSED, and it is the trap this
    function exists for. `tolerance` with `margin == 0` is a HARD INDICATOR: 1
    inside the bounds, 0 outside. So the term silently becomes SPARSE — no
    gradient anywhere — which is the one thing the shaping was added to avoid,
    and it reads in a log as a shaped run that would not learn.

    ⚠ A ZERO WEIGHT WITH ANY MARGIN IS FINE: the term is off, which is how a
    caller turns one half of the shaping off deliberately.
    """
    if w_goal < 0.0 or w_reach < 0.0:
        raise Error(
            "tasks: negative shaping weight (" + String(w_goal) + ", "
            + String(w_reach) + "). These multiply a `tolerance` that REWARDS"
            " proximity, so a negative one pays the policy to move away from"
            " the goal — and it would learn that."
        )
    if w_goal > 0.0 and goal_margin <= 0.0:
        raise Error(
            "tasks: goal weight " + String(w_goal) + " with margin "
            + String(goal_margin) + ". `tolerance` with a zero margin is a"
            " HARD INDICATOR — 1 inside the bounds, 0 outside — so the term"
            " has no gradient anywhere and the run is sparse while looking"
            " shaped."
        )
    if w_reach > 0.0 and reach_margin <= 0.0:
        raise Error(
            "tasks: reach weight " + String(w_reach) + " with margin "
            + String(reach_margin) + ". See the goal-margin error above:"
            " a zero margin makes the term a hard indicator."
        )
    var out = List[Float64]()
    out.append(w_goal)
    out.append(w_reach)
    out.append(goal_margin)
    out.append(reach_margin)
    return out^
