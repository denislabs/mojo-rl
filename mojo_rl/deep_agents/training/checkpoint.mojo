# +--------------------------------------------------------------------------+ #
# | Telling the artifact sink that a checkpoint just landed
# +--------------------------------------------------------------------------+ #
"""One place where "a checkpoint was written" becomes "upload it".

    trainer.save_state(checkpoint_path)
    announce_checkpoint(checkpoint_path, artifacts, run_dir)

## ⚠⚠ Why this is a function and not two lines at each site

`trainer.save_state(checkpoint_path)` appears **eighteen times** across four
driver files. Writing the offer inline at each of them is the defect shape this
tree pays for most often — a rule written in eighteen places drifts, and here
the drift is silent: a site that saves and forgets to announce produces an
artifact that simply never leaves the box, with nothing to see in the output.

So the rule lives here, and `tests/deep_agents/test_checkpoints_announce.mojo`
is a SOURCE gate that reads the four drivers and fails if any
`trainer.save_state(` is not immediately followed by an `announce_checkpoint(`.
That is the same shape as `tests/core/test_drivers_use_runcontext.mojo`, and it
exists because the previous version of this mistake — five hand-rolled `--tag`
blocks in the FB family — was found by accident rather than by a gate.

## ⚠ Why the offer is not inside `save_state`

`save_state` is a trait method on the trainers, implemented per algorithm and
defaulting to a no-op. Putting an upload behind it would make every trainer
depend on `io/artifact_sink` and would fire on paths that are not run
artifacts at all (a probe, a unit test's scratch file). The driver knows it is
running a RUN; the trainer does not.
"""

from ...io.artifact_sink import ArtifactSink, KIND_CHECKPOINT


def announce_checkpoint(
    path: String,
    artifacts: Optional[ArtifactSink],
    run_dir: String,
) raises:
    """Offer a just-written checkpoint to the sink. A no-op without one.

    ⚠ NEVER RAISES IN PRACTICE AND NEVER BLOCKS. `offer` is a memcpy onto a
    ring; the transfer happens on the sink's own thread. A driver must not pay
    for the dashboard being slow, and must not stop because it is down.

    ⚠ THE PATH IS MADE RELATIVE TO THE RUN DIRECTORY HERE. The drivers work in
    absolute paths — `checkpoint_path` is whatever the caller passed — while
    the artifact's identity is its path WITHIN the run, because that is what
    the monitor keys on and what `run.kv` records. Converting at one place
    means a driver never has to know the difference.

    ⚠ A PATH OUTSIDE THE RUN DIRECTORY IS DROPPED, NOT UPLOADED. A driver still
    writing to a `comptime` constant (one that P0d did not retrofit) would
    otherwise land its checkpoint under a run it does not belong to, and a
    misfiled artifact is worse than an absent one — the whole value of the
    layer is that `run.kv` can be trusted.
    """
    if not artifacts:
        return
    if path.byte_length() == 0 or run_dir.byte_length() == 0:
        return
    var prefix = run_dir + "/"
    if not path.startswith(prefix):
        return
    var rel = String(path[byte = prefix.byte_length() :])
    if rel.byte_length() == 0:
        return
    var sink = artifacts.value()
    _ = sink.offer(rel, String(KIND_CHECKPOINT))
