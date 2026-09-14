"""THE LANE->TASK MAPPING IS A PARTITION, AND ITS COUNTS ARE THE DENOMINATORS.

    pixi run mojo run -I . tests/tasks/test_lanes.mojo

## ⚠⚠ WHAT THIS CATCHES

The driver writes a lane's task words by this mapping and the greedy
evaluation reads that lane's success back by the same mapping. An offset
between them reports every task's rate under another task's name — and the
numbers still sum, still look plausible, and name no error. So the gate
checks the two properties a mapping used from both sides must have: every
lane belongs to exactly one task, and the per-task counts are the real
denominators rather than `n_lanes / n_tasks`.
"""

from mojo_rl.tasks.lanes import lane_task, lanes_for_task


def main() raises:
    print("=== lane_task / lanes_for_task ===")
    var fails = 0

    for n_tasks in range(1, 6):
        for n_lanes in [8, 32, 64]:
            # ⚠ EVERY LANE EXACTLY ONCE. Counted by walking the lanes and
            # tallying per task, then comparing to `lanes_for_task` — the
            # two are computed differently on purpose.
            var tally = List[Int](length=n_tasks, fill=0)
            for e in range(n_lanes):
                var t = lane_task(e, n_tasks)
                if t < 0 or t >= n_tasks:
                    print("  FAIL: lane", e, "-> task", t, "out of range")
                    fails += 1
                else:
                    tally[t] += 1
            var total = 0
            for t in range(n_tasks):
                var claimed = lanes_for_task(t, n_lanes, n_tasks)
                total += claimed
                if claimed != tally[t]:
                    print("  FAIL: n_tasks", n_tasks, "lanes", n_lanes,
                          "task", t, "counted", tally[t], "claimed", claimed)
                    fails += 1
            if total != n_lanes:
                print("  FAIL: counts sum to", total, "not", n_lanes)
                fails += 1
            # ⚠ AND BALANCED TO WITHIN ONE, or a task is quietly starved of
            # data while its rate is still reported as if comparable.
            var lo = tally[0]
            var hi = tally[0]
            for t in range(n_tasks):
                if tally[t] < lo:
                    lo = tally[t]
                if tally[t] > hi:
                    hi = tally[t]
            if hi - lo > 1:
                print("  FAIL: n_tasks", n_tasks, "lanes", n_lanes,
                      "split", lo, "..", hi, "differs by more than one lane")
                fails += 1
    print("  ok: partition, denominators and balance over 15 combinations")

    # ⚠ THE UNEVEN CASE IS THE ONE THAT MATTERS, so it is asserted by hand
    # rather than only by the loop above: 32 lanes over 3 tasks is 11/11/10,
    # and `n_lanes / n_tasks` would be 10 for all three.
    var c0 = lanes_for_task(0, 32, 3)
    var c1 = lanes_for_task(1, 32, 3)
    var c2 = lanes_for_task(2, 32, 3)
    print("  32 lanes over 3 tasks ->", c0, c1, c2)
    if c0 != 11 or c1 != 11 or c2 != 10:
        print("  FAIL: expected 11/11/10")
        fails += 1

    # ⚠ SINGLE TASK IS THE EXISTING BEHAVIOUR and must be untouched: every
    # lane on task 0, so the multi-task path is not a second code path.
    for e in range(64):
        if lane_task(e, 1) != 0:
            print("  FAIL: single-task run put lane", e, "elsewhere")
            fails += 1
    if lanes_for_task(0, 64, 1) != 64:
        print("  FAIL: single task does not claim all 64 lanes")
        fails += 1
    print("  ok: a single task still owns every lane")

    var raised = False
    try:
        _ = lane_task(0, 0)
    except:
        raised = True
    if not raised:
        print("  FAIL: n_tasks=0 returned a task index")
        fails += 1

    print()
    if fails == 0:
        print("=== PASS ===")
    else:
        raise Error("lanes: " + String(fails) + " check(s) failed")
