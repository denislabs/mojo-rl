# Test tree compliance

Status of the `tests/` tree against three independent axes, and the plan for
closing the gaps. Updated as sweeps land — **the tables are measurements, not
aspirations; do not edit a number without re-running the command that produced
it** (every one is given below).

Why this file exists: `pixi run build` compiles the **package**, not the tests.
Nothing gates the test tree, so a toolchain bump can leave whole directories
uncompilable and silent for weeks. The Mojo 1.0.0rc2 bump did exactly that —
10 files in `physics3d`/`dm_control` had been dark since it landed.

---

## The three axes

A file is **compliant** when all three hold. They are independent and worth
tracking separately, because they fail for different reasons and cost different
amounts to fix.

| Axis | Question | Failure mode |
|---|---|---|
| **Compiles** | does `mojo build` succeed? | dark since the last toolchain bump — the test does not run *at all* |
| **Gated** | does it `assert_`, `raise Error`, or use `TestSuite`? | runs, prints, and passes unconditionally — measures nothing |
| **Discoverable** | does `main` use `TestSuite.discover_tests[__functions_in_module()]()`? | a test the author forgot to call in `main` is silently skipped |

⚠ **Compiling is not passing.** The sweeps below establish that files *build*.
Whether they *pass* is a separate and much more expensive question — a single
GPU parity test (`test_fish_swimmer_gpu_vs_cpu`) takes 4.5 hours. Do not read
a green compile column as a green suite.

---

## Status

706 `.mojo` files. `SWEPT` = a `mojo build` sweep has been run and its failures
resolved or classified.

| Directory | Files | Swept | Discoverable | Ungated |
|---|---:|:---:|---:|---:|
| `nn`            | 214 | ✗ |   0 | 13 |
| `deep_agents`   | 143 | ✗ |   4 |  9 |
| `physics3d`     |  78 | ✅ |  38 |  6 |
| `pcn`           |  64 | ✗ |   0 | 21 |
| `envs`          |  63 | ✗ |  39 |  0 |
| `dm_control`    |  46 | ✅ |  44 |  0 |
| `experimental`  |  31 | ✗ |   0 |  0 |
| `planners`      |  16 | ✗ |   0 |  1 |
| `arcade_games`  |  13 | ✗ |   0 |  5 |
| `data`          |   7 | ✗ |   0 |  0 |
| `cuda`          |   5 | ✗ |   1 |  3 |
| `board_games`   |   4 | ✗ |   0 |  4 |
| `atari`         |   4 | ✗ |   0 |  2 |
| `render`        |   3 | ✗ |   2 |  1 |
| `physics2d`     |   3 | ✗ |   0 |  1 |
| `io`            |   3 | ✗ |   0 |  1 |
| `core`          |   2 | ✗ |   0 |  1 |
| *(tests/ root)* |   7 | ✗ |   0 |  1 |
| **Total**       | **706** | **124** | **128** | **69** |

Two things the table says out loud:

* **Discoverability tracks attention, not age.** `dm_control` (44/46), `envs`
  (39/63) and `physics3d` (38/78) have adopted it; `nn` (0/214),
  `deep_agents` (4/143) and `pcn` (0/64) have not. Those three directories are
  421 files — the entire migration, essentially.
* **`pcn` is the outlier on gating**: 21 of its 64 files assert nothing at all.
  It is experimental code (`mojo_rl/experimental/pcn`), and it is the first
  place to ask whether files should be repaired or removed.

### Commands behind the table

```bash
# files
find tests/<dir> -name '*.mojo' | wc -l
# discoverable
grep -rl 'discover_tests' --include='*.mojo' tests/<dir> | wc -l
# ungated  (⚠ `assert_` ALONE is wrong — many physics3d tests gate via `raise Error`;
#           counting only assert_ overstated this 3x, 189 vs the real 69)
for f in $(find tests/<dir> -name '*.mojo'); do \
  grep -qE "assert_|TestSuite|raise Error" "$f" || echo "$f"; done
```

---

## Sweep results

### physics3d + dm_control — done, commit `1251c264`

124 files, 14 failures: **10 real compile errors** (fixed; all 10 verified to
compile *and* pass — 56 tests, 0 failures) and **4 link-only** (not source
defects, see below).

Three rc2 rules accounted for all ten:

**1. `Array` is no longer `ImplicitlyCopyable`** — a returned local must transfer.

```mojo
var qpos = InlineArray[Float64, NQ](fill=0.0)
return qpos      # ✗ rc2: cannot be implicitly copied
return qpos^     # ✓
```

**2. Subscripting a *comptime* `InlineArray` materializes the WHOLE array**,
which rc2 then refuses. A comptime index is **not** enough — the *element* must
be materialized, or the binding made a runtime local, or the values hoisted
into a `List` before any runtime loop reads them.

```mojo
comptime acd = materialize[M._acd]()
acd.motor_kp[i]                        # ✗ runtime i — materializes the array
var acd = materialize[M._acd]()        # ✓ runtime local, index freely
materialize[M._acd.motor_kp[a]]()      # ✓ comptime a — element, not array
```

**3. A bare `[a, b, c]` literal now infers `Array`, not `List`** — so it has no
`.append` and will not convert to a `List` parameter. Annotate the binding:
`var far: List[Float64] = [-1.7, -1.0, 0.0]`.

Notable finds:

* **`test_dog_actuator_transmission` was a latent bug rc2 promoted to an
  error.** Its own docstring warned "`_acd` RE-MATERIALIZES ON EVERY READ … §8
  requires one explicit materialize into a local" — and the binding was
  `comptime acd =`, exactly the per-read form the warning forbids. Now passes,
  transmission matching MuJoCo to 2e-16.
* **`test_finger_vs_dm_control` had been cited as gating the mocap-weld path
  while not compiling.** It now compiles and passes 10/10.
* **`test_comptime_parser_io` could not be repaired, only repurposed.** It read
  an MJCF at comptime — impossible, `open` is an `external_call` the comptime
  interpreter refuses — from `test.xml`, *a path that exists nowhere in the
  tree*, with the failure swallowed into `""`. It asserted nothing and had
  never run. Rewritten as a runtime file → `parse_xml_full` round trip.

### Link-only failures — not source defects

Six files fail **only at the link step**, on Apple's new linker:

```
ld: Assertion failed: (name.size() <= maxLength),
    function makeSymbolStringInPlace, file SymbolString.cpp, line 74
```

The mangled name of a deeply-parameterized physics3d generic overruns ld's
symbol-name limit — a dimension spelled `parse_xml(<whole XML string>).NQ`
keeps the entire XML inside the type. Affected: `test_cg_fields`,
`test_newton_solve_fields`, `test_newton_blocked_fields`,
`test_newton_blocked_tendon_fields`, `test_dog_fetch_vs_dm_control`,
`test_fish_swimmer_gpu_vs_cpu`.

They compile fine, link with `-Xlinker -ld_classic`, and run green under
`mojo run` (which JITs — no ld). Verified by running `test_cg_fields`,
`test_newton_solve_fields` and `test_fish_swimmer_gpu_vs_cpu` to completion.

**Classify before fixing:** a log containing `failed to link executable` and
**no** `*.mojo:<line>:<col>: error:` is link-only. Do not read the first error
and generalise.

---

## Plan

Three passes, deliberately **not** combined. A commit that both fixes a compile
error and restructures the file to `discover_tests` is unreviewable: if the
test then fails, nothing says which change did it.

**Pass 1 — inventory, repair nothing.** Sweep all 582 remaining files, classify
every failure (real / link-only / clean), fix none. Cheap and judgment-free;
it is what lets the other two passes be scoped from data instead of guesses.

**Pass 2 — repair, ordered by roadmap value.** Not alphabetically. `envs`, the
`nn` core, and the `deep_agents` families the demo actually uses first; `pcn`,
Dreamer4/EZv2/MuZero/Atari/Procgen last — those are also the likeliest deletion
candidates, so doing them last may mean not doing them at all.

**Pass 3 — `discover_tests` migration.** 578 files; by far the largest and least
urgent. ⚠ More than cosmetic: a file whose `main` forgot to call `test_c()`
currently skips it silently, and this pass makes it run. Expect it to surface
failures — which is the point, and the reason it must be isolated so those
failures are attributable.

### Cost

Measured, not guessed. Ten files sampled across `nn`, `deep_agents`, `pcn`,
`envs`, `planners`, `arcade_games`: median ~22 s, mean ~42 s
(2 s … 105 s). Against **3–7 min** per file in `physics3d`/`dm_control`, which
are the pathological case precisely because they lean on the comptime
metaprogramming rc2 broke.

⇒ **582 files ≈ 7 h serial, ~2–3 h wall clock at 3-way parallel.** One
overnight run. The compute is cheap; the *decisions* in passes 2 and 3 are the
multi-day part.

⚠ All 10 sampled files compiled clean. That is **not** evidence the rest of the
tree is fine: if the true failure rate matched physics3d/dm_control's 8%, a
10-file sample comes back all-green 43 % of the time. The sample prices the
sweep; it cannot predict what the sweep finds.

---

## Deletion criteria

Some files are stale rather than broken. A file is a **deletion candidate**
when one or more holds — and the evidence is quoted in the proposal, per file:

1. **Gates nothing** — no `assert_`, no `raise Error`, no `TestSuite`. A
   print-probe, not a test.
2. **References something gone** — a fixture, path, or symbol that no longer
   exists (`test_comptime_parser_io` read `test.xml`, absent from the tree).
3. **Strictly superseded** — another file covers the same surface with more
   assertions.
4. **Tests a removed component** — e.g. the legacy `nn` / `deep_agents` sunset.

Process: **propose with evidence, then delete on approval.** Criterion 1 alone
is not sufficient grounds — a probe can still be a useful diagnostic; it just
must not be counted as a gate.

---

## Running a sweep

```bash
# per file — build only, never run; classify from the log
pixi run mojo build -I . tests/<dir>/<file>.mojo -o /tmp/out.bin
```

Gotchas paid for once already:

* **Run it DETACHED** (`nohup … & disown`). A sweep killed by a tool/shell
  timeout leaves truncated logs that read as failures — two of the first
  sweep's "failures" were merely interrupted builds.
* **Write results incrementally**, one line per file as it finishes. Piping
  through `sort` buffers everything and yields nothing if the run is cut short.
* **3-way parallel** is the ceiling on an 8-core / 16 GB box; `mojo build` is
  memory-hungry.
* **`mojo run` is the gate that works today**, not `mojo build` — see the
  link-only section. A `mojo build`-based CI needs `-Xlinker -ld_classic`.

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-11 | `physics3d` + `dm_control` swept (124 files). 10 real errors fixed and verified passing, 4 link-only classified — commit `1251c264`. This file created. |
