"""Recording demonstrations an episode at a time — and DAgger's handover.

`EpisodeRecorder` is the bookkeeping every recorder repeats: rows go into a
`DemoSet`, success is the task's goal HELD for `hold_steps` consecutive
steps, a finished episode is kept (success, or `keep_failures`) or dropped,
and the file is REWRITTEN after every kept episode so a crash loses at most
the episode in progress.

DAgger (Ross et al., 2011) is a mode of recording, not a trainer here: a
policy drives, a human or an expert takes over, and the rows after the
takeover carry `intervened=True`. `Handover` is the takeover rule a scripted
expert uses; a human's is a key. Aggregation is `--demos a.demo,b.demo`.
"""

from .file import DemoSet, write_demo_file


struct EpisodeRecorder(Movable):
    var demos: DemoSet
    var out_path: String
    var keep_failures: Bool
    var hold_steps: Int
    """The goal held this many consecutive steps = success."""

    var held: Int
    """Consecutive steps the goal has held, this episode."""
    var rows: Int
    var intervened_rows: Int
    var ret: Float64
    """This episode's return."""
    var n_saved: Int
    var n_dropped: Int

    def __init__(
        out self, obs_dim: Int, act_dim: Int, out_path: String,
        keep_failures: Bool, hold_steps: Int,
    ):
        self.demos = DemoSet(obs_dim, act_dim)
        self.out_path = out_path
        self.keep_failures = keep_failures
        self.hold_steps = hold_steps
        self.held = 0
        self.rows = 0
        self.intervened_rows = 0
        self.ret = 0.0
        self.n_saved = 0
        self.n_dropped = 0

    def begin(mut self):
        self.demos.begin_episode()
        self.held = 0
        self.rows = 0
        self.intervened_rows = 0
        self.ret = 0.0

    def record[T: DType, TA: DType](
        mut self,
        ref prev_obs: List[Scalar[T]],
        ref action: List[Scalar[TA]],
        reward: Float64,
        ref obs: List[Scalar[T]],
        holds: Bool,
        intervened: Bool,
    ) raises -> Bool:
        """One transition. Returns True once the goal has held `hold_steps`
        steps — the episode is a success and should end now.

        ⚠ `done` IS WRITTEN 0: an episode's end here is a truncation (the
        horizon, a success, a key), and the online rows of the drivers these
        files feed never carry a terminal either."""
        self.demos.add(prev_obs, action, reward, obs, 0.0, intervened=intervened)
        self.rows += 1
        if intervened:
            self.intervened_rows += 1
        self.ret += reward
        if holds:
            self.held += 1
        else:
            self.held = 0
        return self.held >= self.hold_steps

    def end(mut self, success: Bool, discard: Bool = False) raises -> Bool:
        """Close the episode: kept (and the file rewritten) when it succeeded
        or failures are kept, and it has rows and was not discarded; dropped
        otherwise. Returns whether it was kept."""
        var keep = (not discard) and (success or self.keep_failures) and self.rows > 0
        if keep:
            self.demos.end_episode(success=success)
            write_demo_file(self.out_path, self.demos)
            self.n_saved += 1
        else:
            self.demos.discard_episode()
            self.n_dropped += 1
        return keep


struct Handover(Copyable, Movable, ImplicitlyCopyable):
    """When a driving policy hands control to the expert: once it has
    `arrived` (the caller's test — near the object, arm settled) after at
    least `min_steps` steps, or when `max_steps` have passed without."""

    var min_steps: Int
    var max_steps: Int

    def __init__(out self, min_steps: Int, max_steps: Int):
        self.min_steps = min_steps
        self.max_steps = max_steps

    def arrived_at(self, k: Int, arrived: Bool) -> Bool:
        """Step `k` (0-based) ended with the policy `arrived`: hand over?"""
        return k + 1 >= self.min_steps and arrived
