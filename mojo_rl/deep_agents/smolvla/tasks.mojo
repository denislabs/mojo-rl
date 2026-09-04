"""Pre-tokenised instructions: `task_index` -> the ids the prefix embeds.

`SmolVLAPrefixEmbed.run` takes ids, not text, and that is a decision rather than
an omission. `tokenizer_max_length` is 48 and `pad_language_to` is "longest", so
for a fixed instruction set the ids are a **constant** -- recomputing them at
50 Hz would mean shipping a tokeniser (and CPython) back into a training binary
that has just been cleared of one. `tools/vla/dump_smolvla_tasks.py` produces
the table offline; this reads it.

⚠ **The trailing newline is a pipeline step, not part of the string.**
`processor_smolvla.py` runs `NewLineTaskProcessorStep()` before the tokeniser,
so "Grab the green cube" tokenises as six ids ending in 198 (`'\\n'`), not five.
Drop it and the state token, `P`, and every mask built from `P` shift by one,
silently.

⚠ **`N_LANG` is a comptime parameter here and a runtime one in the reference.**
`pad_language_to: "longest"` pads to the longest instruction *in the batch*, so
the reference's language block grows and shrinks; ours is fixed at compile time.
For a single-task dataset those coincide and ours is the stricter of the two.
For several instructions of different lengths they do not, and the value must be
pinned to whatever the fine-tune used -- `n_tokens` is carried per row so a
mismatch can be named instead of inferred.

Format, `#` comments skipped:

    <task_index>\\t<n_tokens>\\t<id,id,...>\\t<task text>
"""

from mojo_rl.io.fileio import read_file_bytes


struct TaskTokens(Movable):
    """One dataset's instruction table."""

    var indices: List[Int]
    var ids: List[List[Int]]
    var texts: List[String]

    def __init__(out self, path: String) raises:
        self.indices = List[Int]()
        self.ids = List[List[Int]]()
        self.texts = List[String]()
        var raw = read_file_bytes(path)
        var text = String(from_utf8=Span(raw))
        for line in text.split(String("\n")):
            if line.byte_length() == 0 or line.startswith(String("#")):
                continue
            var parts = line.split(String("\t"))
            if len(parts) < 3:
                continue
            var n = Int(parts[1])
            var row = List[Int]()
            for tok in parts[2].split(String(",")):
                row.append(Int(tok))
            # The count is written beside the ids on purpose: it is the one
            # field a hand edit gets wrong, and a short language block is a
            # working policy that attends to a truncated instruction.
            if len(row) != n:
                raise Error(
                    "smolvla tasks: row "
                    + String(parts[0])
                    + " declares "
                    + String(n)
                    + " tokens but lists "
                    + String(len(row))
                )
            self.indices.append(Int(parts[0]))
            self.ids.append(row^)
            self.texts.append(
                String(parts[3]) if len(parts) > 3 else String("")
            )

    def __init__(out self, *, deinit move: Self):
        self.indices = move.indices^
        self.ids = move.ids^
        self.texts = move.texts^

    def size(self) -> Int:
        return len(self.indices)

    def n_lang(self) raises -> Int:
        """The single token count, or an error naming the disagreement.

        A comptime `N_LANG` cannot represent two lengths, so a table holding
        two is a decision the caller has to make, not a default this can pick.
        """
        if len(self.ids) == 0:
            raise Error("smolvla tasks: the table is empty")
        var n = len(self.ids[0])
        for i in range(1, len(self.ids)):
            if len(self.ids[i]) != n:
                raise Error(
                    "smolvla tasks: task "
                    + String(self.indices[0])
                    + " is "
                    + String(n)
                    + " tokens and task "
                    + String(self.indices[i])
                    + " is "
                    + String(len(self.ids[i]))
                    + " — a comptime N_LANG must be pinned to the one the"
                    " fine-tune used"
                )
        return n

    def for_index(self, task_index: Int) raises -> List[Int]:
        for i in range(len(self.indices)):
            if self.indices[i] == task_index:
                return self.ids[i].copy()
        raise Error(
            "smolvla tasks: no task_index " + String(task_index) + " in the table"
        )
