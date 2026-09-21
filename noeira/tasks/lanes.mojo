"""WHICH LANE RUNS WHICH TASK, in a multi-task batch.

    var t = lane_task(e, n_tasks)          # the task index for lane `e`
    var n = lanes_for_task(t, N_ENVS, n_tasks)   # how many lanes it gets

## ⚠⚠ ONE RULE, TWO CALLERS, WHICH IS WHY IT IS A MODULE

The driver writes each lane's tape, mask, init-region and shaping words by
this mapping, and the greedy evaluation reads a lane's success back by the
same mapping. If those two disagree by so much as an offset, every per-task
rate is a rate for the wrong task — and it still prints, sums correctly, and
looks entirely reasonable. That is the failure shape this tree keeps paying
for, so the rule is written once.

## ⚠ INTERLEAVED (`lane % n_tasks`), NOT BLOCKED

A block assignment (`lane * n_tasks / N_ENVS`) ties the task to a contiguous
run of lane indices, and the lane index is the Philox axis the device-side
placement sampler draws on — so a task would see a systematically different
slice of the placement distribution than its neighbour. Interleaving gives
each task the same spread of lane indices.

⚠ THE SPLIT IS EVEN ONLY WHEN `n_tasks` DIVIDES `N_ENVS`. It is off by at
most one lane otherwise, and `lanes_for_task` is what the caller must use to
normalise a per-task rate — dividing by `N_ENVS / n_tasks` would be wrong for
the tail tasks and wrong in a way no total would reveal.
"""


def lane_task(lane: Int, n_tasks: Int) raises -> Int:
    """The task index lane `lane` runs. Interleaved."""
    if n_tasks <= 0:
        raise Error("tasks: lane_task with n_tasks=" + String(n_tasks))
    if lane < 0:
        raise Error("tasks: lane_task with lane=" + String(lane))
    return lane % n_tasks


def lanes_for_task(task: Int, n_lanes: Int, n_tasks: Int) raises -> Int:
    """How many of `n_lanes` lanes run task `task`.

    ⚠ THE DENOMINATOR OF A PER-TASK RATE. With 32 lanes and 3 tasks the
    counts are 11/11/10, and dividing all three by 32/3 would report the last
    task's rate about 10% low while the three still 'summed' plausibly.
    """
    if n_tasks <= 0 or task < 0 or task >= n_tasks:
        raise Error(
            "tasks: lanes_for_task(task=" + String(task) + ", n_tasks="
            + String(n_tasks) + ")"
        )
    var n = 0
    for e in range(n_lanes):
        if e % n_tasks == task:
            n += 1
    return n
