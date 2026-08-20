"""Undo across STRUCTURAL edits — the document snapshot stack. V2.9.

## ⚠⚠ WHAT WAS BROKEN

V2 shipped `delete body`, which prunes the actuators, tendons, equalities,
sensors and keyframes that referenced what it removed — and then reset the
`EditLog`, because a replay of dims-preserving edits cannot express a delete.
So the single most destructive operation in the studio had **no undo at all**.
Delete the wrong link on a forty-body robot and the only way back was to
reopen the file and lose everything else done since.

## THE SNAPSHOT IS THE DOCUMENT

Plan §4 asked for snapshots rather than command/inverse pairs. `EditLog` was a
compromise around one fact: `FlatModelDef` is `Movable` and not `Copyable`, so
there was nothing cheap to snapshot. V2.4 removed that constraint by making
the **document** authoritative — every edit, fast or structural, now lands in
`Loaded.flat`. A document is a `String`, and

    Loaded(path, doc, base_dir)

is a pure function of one. So a stack of documents IS the undo stack, it costs
a `String` copy to push, and it covers both tiers with no case analysis.

⚠ THE ENTRY CARRIES `base_dir` AND THE SCENE, NOT JUST THE TEXT. Two of the
three are easy to forget:

* **`base_dir`** varies *during a session*. A structural edit rebuilds against
  the model's directory; a prop edit rebuilds against the CWD, because the
  scene's `<asset><model file=>` paths are relative to where the process was
  started (`physics_studio` documents that at length — it is the oldest bug
  shape in that file). Restoring a document against the wrong base resolves
  its meshes against the wrong directory.
* **the `SceneDoc`** is separate state that the document cannot reconstruct:
  `to_mjcf` is one-way. Restoring the text but not the scene would leave the
  next `add prop` regenerating from a composition the user had already undone,
  silently re-applying it.

## COALESCING — why a drag is ONE undo step

`ImGui`'s drag widgets emit a value every frame they are held, so a single
pull on the size slider produces on the order of a hundred edits. Pushing a
snapshot each time is both 33 KB × 100 of text and, far worse, an undo stack
where "undo" appears to do nothing a hundred times running.

So a push carries a `key`. A push whose key is non-empty and equal to the top
entry's REPLACES it instead of growing the stack. The key is the edit's
identity — target, index, field — so dragging size, then mass, then size again
gives three steps, which is what the hand remembers doing. A structural edit
passes `""`, which never coalesces.

⚠ `""` IS NOT A KEY, IT IS THE ABSENCE OF ONE. Two structural edits in a row
must both be undoable; if the empty string compared equal to itself here, a
delete followed by a rename would leave one step and lose the delete.

## ⚠ THE CAP DROPS THE OLDEST, AND ENTRY 0 IS NOT SPECIAL

`docs[0]` is the file as opened, and the cap can evict it like any other. That
is deliberate: the alternative — pinning it — makes "undo all the way back"
mean two different things depending on how much was done since. The stack
reports its own floor through `can_undo`, and `File > Open` reloads the file
if that is what the user wants.
"""

from .remap import PoseSnapshot
from .scene import SceneDoc


comptime HISTORY_CAP: Int = 64
"""How many documents to keep. A flattened Menagerie scene measures ~33 KB
(`hello_robot_stretch_3`), so the ceiling here is a few megabytes — chosen
because it is invisible, not because a deeper stack would be wrong."""


struct HistoryEntry(Copyable, Movable):
    """One restorable state: the document, where it resolves, and the scene."""

    var doc: String
    """The EXPANDED MJCF. `Loaded` re-expands on the way in, which is a no-op
    on already-expanded text — the structural path has relied on that since
    V2.1, so this adds no new assumption."""
    var base_dir: String
    var scene: SceneDoc
    var label: String
    """What the edit that PRODUCED this state was — "deleted 'bthigh'". Shown
    on the Undo menu item, so the user reads what will come back rather than
    guessing."""
    var key: String
    """Coalescing identity, or "" for an edit that must stand alone."""
    var pose: PoseSnapshot
    """This state's joints as they stood when it STOPPED being live.

    ⚠⚠ WRITTEN ON THE WAY OUT, NOT ON THE WAY IN. An entry's pose is not
    known when it is pushed — it is whatever the user left it at when the
    next edit, undo or redo moved off it. So every transition below stamps
    the OUTGOING entry, and `pose` is empty only for a state never departed
    from (the live one, and the redo tail). Filling it at push time would
    record the pose from BEFORE the edit that created the entry, which is a
    different state's pose wearing this one's label."""

    def __init__(out self, doc: String, base_dir: String, scene: SceneDoc,
                 label: String, key: String):
        self.doc = doc
        self.base_dir = base_dir
        self.scene = scene.copy()
        self.label = label
        self.key = key
        self.pose = PoseSnapshot()


struct History(Movable):
    """A stack of document snapshots with a cursor. Redo is the tail above it.

    ⚠ THE CURSOR INDEXES THE **LIVE** ENTRY, not the number of live entries.
    `EditLog.cursor` counted edits, and off-by-one between "how many are
    applied" and "which one am I on" is the classic undo bug. Here `cursor`
    always names a real entry, `undo` decrements it and `entries[cursor]` is
    always exactly what is on screen.
    """

    var entries: List[HistoryEntry]
    var cursor: Int

    def __init__(out self):
        self.entries = List[HistoryEntry]()
        self.cursor = 0

    def depth(self) -> Int:
        return len(self.entries)

    def can_undo(self) -> Bool:
        return self.cursor > 0

    def can_redo(self) -> Bool:
        return self.cursor + 1 < len(self.entries)

    def undo_label(self) -> String:
        """What undoing would take back — the LIVE entry's label."""
        if not self.can_undo():
            return String("")
        return self.entries[self.cursor].label

    def redo_label(self) -> String:
        if not self.can_redo():
            return String("")
        return self.entries[self.cursor + 1].label

    def push(mut self, doc: String, base_dir: String, scene: SceneDoc,
             label: String, key: String = String(""),
             outgoing: PoseSnapshot = PoseSnapshot()) raises:
        """Record a new state. Discards the redo tail; may coalesce; may evict.

        `outgoing` is the pose of the state being LEFT — see `HistoryEntry`.

        ⚠ COALESCING REPLACES THE TOP, IT DOES NOT SKIP THE PUSH. The new
        text has to land — it is the current one — while the step count stays
        put. An implementation that returned early on a matching key would
        leave the stack holding the FIRST frame of the drag and restore it on
        the next undo, which reads as "undo went too far".
        """
        var leaving = len(self.entries) > 0
        if leaving:
            self.entries[self.cursor].pose = outgoing.copy()
        # A new edit after an undo discards the redo tail, as every editor does.
        if self.cursor + 1 < len(self.entries):
            var keep = List[HistoryEntry]()
            for i in range(self.cursor + 1):
                keep.append(self.entries[i].copy())
            self.entries = keep^
            self.cursor = len(self.entries) - 1

        if key.byte_length() > 0 and len(self.entries) > 0 \
                and self.entries[len(self.entries) - 1].key == key:
            self.entries[len(self.entries) - 1] = HistoryEntry(
                doc, base_dir, scene, label, key
            )
            self.cursor = len(self.entries) - 1
            return

        self.entries.append(HistoryEntry(doc, base_dir, scene, label, key))
        if len(self.entries) > HISTORY_CAP:
            var trimmed = List[HistoryEntry]()
            for i in range(len(self.entries) - HISTORY_CAP, len(self.entries)):
                trimmed.append(self.entries[i].copy())
            self.entries = trimmed^
        self.cursor = len(self.entries) - 1

    def undo(mut self, outgoing: PoseSnapshot = PoseSnapshot()) -> Bool:
        """Step back one. False when already at the floor — the caller must
        NOT rebuild on False, or an undo at the bottom would rebuild the same
        document and throw the pose away for nothing.

        ⚠ THE OUTGOING POSE IS STAMPED EVEN THOUGH WE ARE LEAVING. Redo comes
        back here, and without it a redo restores the structure and drops the
        pose — the same bug this fixes, in the other direction."""
        if not self.can_undo():
            return False
        self.entries[self.cursor].pose = outgoing.copy()
        self.cursor -= 1
        return True

    def redo(mut self, outgoing: PoseSnapshot = PoseSnapshot()) -> Bool:
        if not self.can_redo():
            return False
        self.entries[self.cursor].pose = outgoing.copy()
        self.cursor += 1
        return True

    def doc(self) -> String:
        if len(self.entries) == 0:
            return String("")
        return self.entries[self.cursor].doc

    def base_dir(self) -> String:
        if len(self.entries) == 0:
            return String("")
        return self.entries[self.cursor].base_dir

    def scene(self) -> SceneDoc:
        if len(self.entries) == 0:
            return SceneDoc()
        return self.entries[self.cursor].scene.copy()

    def pose(self) -> PoseSnapshot:
        """The live entry's snapshot — empty if this state was never left."""
        if len(self.entries) == 0:
            return PoseSnapshot()
        return self.entries[self.cursor].pose.copy()

    def label(self) -> String:
        if len(self.entries) == 0:
            return String("")
        return self.entries[self.cursor].label


def edit_key(target: Int, index: Int, field: Int) -> String:
    """The coalescing identity of a fast-path field edit.

    ⚠ ALL THREE PARTS. Dragging geom 4's size and then geom 7's size are two
    edits the user made separately; a key of the field alone would fold them
    into one and the first geom would never come back.
    """
    return String("f:", target, ":", index, ":", field)
