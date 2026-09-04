"""Placement sampling — `init=brick@table_top` is a DISTRIBUTION — P2b.

    var r = SampleReport()
    var placed = sample_placements(task, family, frames, radii, seed, lane, r)

`TASK_LAYER_PLAN.md` §6.1: rejection sampling over the declared regions,
seeded, deterministic given `(task, seed, lane)`, bounded retries, and **RAISE
on exhaustion rather than silently returning an overlapping scene** — the
silent version is a bug that surfaces as a policy that cannot learn.

## ⚠⚠ PURE GEOMETRY, ON PURPOSE — IT HAS TO RUN ON DEVICE

This file takes region FRAMES and slot RADII as plain numbers and returns
poses. It never touches `Data`, `Model` or MuJoCo. That is not minimalism: P3
resets 1024 lanes on device, so the sampler must be callable from a kernel,
and a version that reached into `Data` here would be rewritten there — two
implementations of one distribution, drifting.

The caller supplies the frames because IT knows where the sites are: on the
CPU authoring path from `Data.site_xpos` after FK, on the GPU path from the
same tensor, already per-lane.

## ⚠⚠ THE SEED IS SALTED AWAY FROM THE ENV'S RESET NOISE, DELIBERATELY

`ModelDefFromXML.reset_env_gpu` seeds its joint-noise Philox with
`seed * 2654435761 + env * 12345` (`model_def_from_xml.mojo:1742`). Reusing
that spelling here would draw placements from THE SAME STREAM as the arm's
joint noise, so a lane's object layout and its starting pose would be
perfectly correlated across the whole batch — a hidden confound that no test
would report and that reads, much later, as a policy that has learned the
correlation instead of the task.

`PLACEMENT_SALT` is what makes them independent streams. ⚠ It is a magic
number and it is SUPPOSED to be: the requirement is only that it differ, and
recording why is worth more than choosing it cleverly.

## ⚠ COUNTER-BASED, NOT A STATEFUL STREAM

`std.random.philox` is seeded per draw from `(seed, lane, slot, attempt)`
rather than advanced. A stateful stream gives lane 7 a different draw
depending on how many attempts lanes 0-6 needed — so a scene would depend on
its neighbours, reruns would not reproduce, and the CPU and GPU legs could not
be compared at all. Counter-based, every draw is a pure function of its
coordinates.
"""

from std.random.philox import Random as PhiloxRandom

from .spec import FamilySpec, TaskSpec, SLOT_FREE


# ⚠ BOUNDED, AND EXHAUSTION RAISES. An unbounded retry loop on an
# over-constrained region hangs a training run at reset with no diagnostic;
# a loop that gives up and returns the last draw produces an overlapping
# scene, which MuJoCo resolves by launching the objects apart on step 1.
comptime MAX_PLACE_ATTEMPTS: Int = 64

# See the module header. Any value that is not the env's own works.
comptime PLACEMENT_SALT: UInt64 = 0x9E3779B97F4A7C15


struct Placement(Copyable, ImplicitlyCopyable, Movable):
    """Where one slot starts this episode, in world coordinates."""

    var slot: Int
    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, slot: Int, x: Float64, y: Float64, z: Float64):
        self.slot = slot
        self.x = x
        self.y = y
        self.z = z


struct RegionFrame(Copyable, ImplicitlyCopyable, Movable):
    """A region's site, in WORLD coordinates, this episode.

    ⚠ RESOLVED PER EPISODE BY THE CALLER, not stored on the family. A region
    attached to a movable slot's site TRAVELS with it — that is the whole
    reason regions are site-relative — so its world frame is only known after
    forward kinematics.
    """

    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64, y: Float64, z: Float64):
        self.x = x
        self.y = y
        self.z = z


struct SampleReport(Copyable, ImplicitlyCopyable, Movable):
    """Draws attempted vs accepted.

    ⚠⚠ THE ATTEMPT COUNT IS NOT DIAGNOSTICS, IT IS THE ANTI-VACUITY CHECK. A
    rejection sampler whose rejection branch never runs is indistinguishable
    from one with no rejection at all, and both pass "it returned poses". The
    gate asserts this counter MOVES on a crowded region.
    """

    var attempts: Int
    var accepted: Int

    def __init__(out self):
        self.attempts = 0
        self.accepted = 0

    def rejected(self) -> Int:
        return self.attempts - self.accepted


@always_inline
def _uniform01(seed: UInt64, lane: Int, axis: Int, attempt: Int) -> Float64:
    """One uniform draw, a PURE FUNCTION of its coordinates.

    ⚠ USES PHILOX'S OWN COUNTERS — `subsequence` and `offset` — rather than
    hashing everything into `seed`. That is what the generator is FOR: the
    three are independent counter axes with guaranteed-decorrelated streams,
    whereas XOR-folding coordinates into a seed can collide two different
    (lane, axis) pairs onto one stream and nothing would report it.

    ⚠ One value per call, not four. `step_uniform` returns four, but reusing
    them would make a draw depend on which SLOT of a batch it came from, and
    the coordinates would have to encode that. One value per
    (lane, axis, attempt) keeps every draw reproducible even when attempt
    counts differ between runs — which is exactly what rejection sampling
    makes happen.
    """
    var rng = PhiloxRandom(
        seed=seed ^ PLACEMENT_SALT,
        subsequence=(UInt64(lane) << 16) | UInt64(axis),
        offset=UInt64(attempt),
    )
    var v = rng.step_uniform()
    return Float64(v[0])


def sample_placements(
    t: TaskSpec,
    f: FamilySpec,
    frames: List[RegionFrame],
    radii: List[Float64],
    seed: UInt64,
    lane: Int,
    mut report: SampleReport,
) raises -> List[Placement]:
    """Every `init=` in the task, placed. Deterministic in `(seed, lane)`.

    `frames` is indexed by FAMILY REGION INDEX; `radii` by FAMILY SLOT INDEX.

    ⚠ INITS ARE PLACED IN TASK ORDER and each is rejected against the ones
    ALREADY placed. That makes the result order-dependent — which is correct
    and must stay stable, because changing the order changes every episode of
    every seed. `spec.mojo` preserves file order for exactly this reason.
    """
    if len(frames) != len(f.regions):
        raise Error(
            "tasks: sampler got " + String(len(frames)) + " region frames for"
            " a family with " + String(len(f.regions)) + " regions. The"
            " caller must resolve EVERY region, in family order."
        )
    if len(radii) != len(f.slots):
        raise Error(
            "tasks: sampler got " + String(len(radii)) + " slot radii for a"
            " family with " + String(len(f.slots)) + " slots."
        )

    var out = List[Placement]()
    for i in range(len(t.inits)):
        var si = f.slot_index(t.inits[i].slot)
        var ri = f.region_index(t.inits[i].region)
        if si < 0 or ri < 0:
            # `validate_task_against_family` refuses this long before here;
            # re-checked because the sampler indexes with the results.
            raise Error(
                "tasks: init '" + t.inits[i].describe() + "' does not resolve"
                " against family '" + f.name + "'. Run"
                " validate_task_against_family first."
            )

        ref reg = f.regions[ri]
        ref fr = frames[ri]
        var placed = False
        for attempt in range(MAX_PLACE_ATTEMPTS):
            report.attempts += 1
            var x = fr.x
            var y = fr.y
            if reg.has_rect:
                var u = _uniform01(seed, lane, si * 2, attempt)
                var v = _uniform01(seed, lane, si * 2 + 1, attempt)
                x = fr.x + reg.x_min + u * (reg.x_max - reg.x_min)
                y = fr.y + reg.y_min + v * (reg.y_max - reg.y_min)
            # ⚠ RESTING ON THE SURFACE, not centred in it. The site is on the
            # face a region describes, so an object's CENTRE sits one radius
            # above it. Placing it AT the site starts every episode with the
            # prop half inside the table, which the solver resolves by
            # ejecting it — a scene that looks sampled and is not.
            var z = fr.z + radii[si]

            var clash = False
            for j in range(len(out)):
                var dx = out[j].x - x
                var dy = out[j].y - y
                var rr = radii[si] + radii[out[j].slot]
                if dx * dx + dy * dy < rr * rr:
                    clash = True
            if not clash:
                out.append(Placement(si, x, y, z))
                report.accepted += 1
                placed = True
                break

        if not placed:
            raise Error(
                "tasks: could not place slot '" + t.inits[i].slot + "' in"
                " region '" + t.inits[i].region + "' after "
                + String(MAX_PLACE_ATTEMPTS) + " attempts — every draw"
                " overlapped an object already placed. The region is too"
                " small for the objects the task puts in it. ⚠ This RAISES"
                " rather than returning an overlapping scene: an overlap is"
                " resolved by the solver ejecting the props on step 1, which"
                " reads as a policy that cannot learn, far from the cause."
            )
    return out^


def parked_pose(f: FamilySpec, slot_index: Int) -> List[Float64]:
    """Where an INACTIVE slot goes. Re-exported from `family` so that a caller
    resetting a lane has one import and one spelling of "parked"."""
    from .family import park_pos

    return park_pos(f, slot_index)
