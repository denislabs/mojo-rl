# Test fixtures

## `act_ref_xattn.tgz` (16 KB)

The cross-attention slice of the ACT reference dump — the 18 arrays
`tests/nn/test_cross_attention_gpu.mojo` reads, and nothing else.

    mkdir -p /tmp/act_ref && tar xzf tests/fixtures/act_ref_xattn.tgz -C /tmp/act_ref

**Why it is committed.** Regenerating it needs the `act-ref` pixi environment
(PyTorch + torchvision), whose dependency tree is multi-GB and has filled the
disk on a fresh box. That is a lot of setup for 16 KB of fixed arrays, and it
put the ONLY GPU gate with a prerequisite behind a torch install. The other
three GPU gates (`test_act_gpu_vs_cpu`, `test_resnet18_gpu`,
`test_batch_norm_2d_eval_backward_gpu`) need nothing at all, so with this the
whole GPU suite is `git clone` + run.

**It is a SUBSET, on purpose.** The full dump is ~130 MB, dominated by the
DETRVAE and ResNet18 parameter blobs the CPU reference gates load. Those stay
regenerated rather than committed — they are large, and they must track
`references/act-main/` rather than drift as a stale binary.

    pixi run -e act-ref python tools/act/dump_act_reference.py --out /tmp/act_ref

`manifest.txt` here lists only the `xattn_*` arrays. `RefDump` reads names and
sizes from the manifest and opens a `.bin` only when asked for it, so a subset
manifest is consistent — a gate asking for anything outside it fails loudly with
the missing name rather than reading garbage.

## `parquet/golden_v3_shapes.parquet` (13 KB)

26 rows in 4 unequal row groups, in every column shape a LeRobot v3 dataset
uses, written by **Arrow** (`parquet-cpp-arrow`) rather than by this repo.

    pixi run python tools/io/make_parquet_golden.py

**Why it is committed.** `tests/io/test_parquet_write.mojo` has to answer "is
what we wrote a real Parquet file". A round trip through our own reader cannot:
the pair would share any misunderstanding and agree with itself while nothing
else could open the result. Arrow is the independent party, and committing its
output means the gate needs no `pyarrow` at test time — the same trade as
`test_sha256.mojo` pinning digests that came out of `hashlib` once.

**Two of its columns exist only to defeat a passing gate.** `tasks` has
variable-length rows, because a constant width makes repetition levels
predictable from the row index. `nested_2x3x2` is not a LeRobot shape at all:
LeRobot's nested statistic is `[3,1,1]`, whose inner dimensions are both 1, so
repetition levels 2 and 3 are NEVER EMITTED and the code that produces them is
dead. A deliberate corruption of those levels survived every other check until
this column existed.
