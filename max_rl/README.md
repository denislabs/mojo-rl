# max_rl — MAX-as-inference-backend prototype

A small prototype probing **MAX as an *inference* backend for mojo-rl**, driven from Mojo
via Python interop. Companion to `docs/MAX_TRAINING_ASSESSMENT.md` (which covers why
*training* on MAX is blocked today). v1 scope is **MLP inference only**.

It answers three questions for "should mojo-rl incorporate MAX?":

1. How fast is MAX device compute on a realistic RL MLP?
2. What is the host↔device data-transfer cost (H2D / D2H)?
3. What does the Mojo↔Python interop bridge actually cost on the real call path?

## Layout

| File | What |
|---|---|
| `mlp_inference.py` | `MLPInference` — configurable MLP (dims/batch/device all variables) built+compiled once on MAX, plus timing primitives. Inference only; weights are random. |
| `benchmark_interop.mojo` | Mojo driver that imports the package via Python interop and times the MAX decomposition across a batch/shape sweep. |
| `benchmark_nn2_baseline.mojo` | Pure-nn2 native GPU forward on the SAME shapes — the apples-to-apples "why incorporate MAX?" baseline. |
| `probe_c_api.sh` | Path-B feasibility probe: is the MAX C API linkable + is there a MEF-export path? Prints GO/NO-GO. Run under `-e nvidia`. |
| `graph_mlp_example.py`, `graph_relu_example.py` | Original MAX reference snippets (kept for reference). |

`MLPInference(input_dim, hidden, output_dim, batch, device="gpu", seed=0)` — `hidden` is a
list `[256, 256]` or a Mojo-friendly string `"256,256"`. `device="gpu"` → Metal on Apple,
CUDA on NVIDIA (same Python, different backend); falls back to CPU if no accelerator.

## How to run

**Build to a binary — do NOT `mojo run`** (JIT triggers an `M::Context` clash with MAX's
Python engine; a compiled binary has no JIT context). **And run the binary *inside* the
activated env** — the embedded Python needs `MOJO_PYTHON_LIBRARY` set, which `pixi run` only
provides for the command it wraps. Build + run in one pixi invocation:

```bash
# Apple (Metal)
pixi run -e apple  bash -c 'mojo build -I . max_rl/benchmark_interop.mojo -o /tmp/bench && /tmp/bench'
# NVIDIA (CUDA)
pixi run -e nvidia bash -c 'mojo build -I . max_rl/benchmark_interop.mojo -o /tmp/bench && /tmp/bench'
```
(Running the bare `/tmp/bench` outside `pixi run` fails with "No module named 'max'", because
activation env vars aren't set in your shell.)

nn2 baseline (pure nn2, no Python — plain `mojo run` is fine):
```bash
pixi run -e apple  mojo run -I . max_rl/benchmark_nn2_baseline.mojo
pixi run -e nvidia mojo run -I . max_rl/benchmark_nn2_baseline.mojo
```

You can also drive the Python package directly:
```bash
pixi run -e apple python -c "from max_rl import MLPInference; m=MLPInference(17,'256,256',6,64); print(m.info())"
```

## Findings so far (Apple M-series / Metal, 2026-06-03)

### ⚠️ Footgun: `mojo run` + MAX Python engine clash at the runtime-context level
JIT `mojo run` creates an `M::Context` whose `Init::Options` conflict with the one
`max.engine` wants → `LLVM ERROR: Init::getOrCreateContext() requested an M::Context with
different Init::Options`. **A compiled binary has no JIT context, so the Python engine
initializes cleanly.** Always `mojo build` then run the executable. (Worth re-checking on
NVIDIA — this is your part.)

### The Mojo↔Python FFI crossing is essentially free
The per-call interop floor (a Mojo loop over a Python `noop()`) is **~0.15 µs/call**. The
"Python in the hot loop" tax people worry about is *not* the FFI boundary. Mojo-side
end-to-end `infer()` ≈ Python-side end-to-end `full()` to within run-to-run noise.

### The real costs are MAX compute, data transfer, and Python *glue*
For each call the decomposition is: `MAX device compute` + `H2D` + `D2H` + `Python glue`
(numpy contiguity checks, `Buffer` object creation, attribute lookups). On Metal the
**Python glue per call (~130–170 µs)** dwarfs the FFI crossing (0.15 µs) — i.e. *what you
do in Python per call matters far more than crossing the Mojo/Python line.* This is the
argument for the production path (B): Mojo → MAX **C API** on a precompiled MEF, which
removes the Python glue entirely.

### Numbers are path (A), a pessimistic upper bound
This prototype is **path (A): Mojo → CPython → MAX**, Python in the hot loop. The
production-realistic **path (B): Mojo → MAX C API** (`M_executeModelSync` on a precompiled
MEF) removes Python from the loop and is strictly faster. Read (A) as an upper bound: *if
MAX wins even here, path (B) wins by more.*

### Metal compile time is high (~15 s even for a tiny MLP)
One-time per graph shape, excluded from per-call numbers, but relevant for research
iteration that sweeps many shapes. NVIDIA compile times are the ones that matter for you.

### nn2 vs MAX head-to-head (Apple/Metal, µs per call)

| Shape | nn2 forward (compute) | MAX device compute | MAX end-to-end from Mojo |
|---|---|---|---|
| actor-b1    (17→256→256→6, b=1)     | 658  | **284**   | 507   |
| actor-b64   (b=64)                  | 618  | **309**   | 978   |
| actor-b1024 (b=1024)                | **1357** | 1663  | 4484  |
| wide-b1     (256→512→512→64, b=1)   | 699  | **287**   | 868   |
| wide-b1024  (b=1024)                | **8913** | 12455 | 15050 |

### nn2 vs MAX head-to-head (NVIDIA / CUDA, µs per call) — the decisive run

| Shape | nn2 forward (delivered) | MAX raw compute | MAX delivered (e2e) | nn2 vs MAX-delivered |
|---|---|---|---|---|
| actor-b1    | **26.6** | 45.3 | 73.8  | nn2 2.8× |
| actor-b64   | **36.9** | 71.1 | 99.3  | nn2 2.7× |
| actor-b1024 | **38.9** | 55.5 | 104.5 | nn2 2.7× |
| wide-b1     | **18.7** | 30.8 | 75.2  | nn2 4.0× |
| wide-b1024  | **68.0** | 51.2 | 202.5 | nn2 3.0× |

Reading (CUDA — **this is the verdict**):
- **nn2 wins delivered latency everywhere, ~2.7–4×.** H2D+D2H+Python glue (30–150 µs) dwarfs
  compute at RL-MLP scale.
- **nn2 wins even raw compute in 4/5 shapes.** MAX's compiler only leads at the widest matmul
  (wide-b1024) — the large/transformer regime it's built for, not small RL MLPs.
- **Interop bridge is free (0.18 µs/call)** on CUDA too — the cost is transfer + Python glue.
- nn2 here is **unoptimized** (plain `Linear+ReLU`, no fused `LinearReLU`, no CUDA-graph
  capture) — a *ceiling*; the real nn2 is faster still.
- **MAX compile cost on CUDA is ~46–52 s per shape** (vs ~15 s Metal) — a real RL shape-sweep tax.
- **This bounds path B too:** path B's best case ≈ MAX raw compute (45–71 µs for actor) still
  loses to nn2 delivered (27–39 µs) except at wide-b1024 (where nn2 is unoptimized). A perfect
  no-Python path B can't flip the RL-scale verdict.

**Bottom line for "why don't I incorporate MAX?": at RL-MLP inference scale on NVIDIA, nn2 is
~3× faster delivered and competitive-to-better on raw compute, with no interop tax and no
per-shape compile wall. MAX pays off at large/transformer-scale graphs, not here.**

### (Earlier) Apple/Metal numbers — for reference

Reading (Metal only):
- **MAX raw compute wins at small batch** (~2× faster, 284 vs 658 µs at b=1) but **loses
  end-to-end** once you add H2D+D2H+Python glue — the *delivered* MAX latency to a Mojo
  caller (507–978 µs) is at or above nn2's (618–699 µs).
- **nn2 wins outright at large batch**, on raw compute *and* end-to-end (its native kernels
  beat MAX's here on Metal, and it pays zero transfer/Python tax).
- nn2's number is the **full delivered latency** to a Mojo caller (data already in Mojo GPU
  buffers); MAX must overcome its transfer+glue tax to be worth it. On Metal it generally
  isn't; whether MAX's compute edge widens enough on CUDA to flip the end-to-end verdict is
  exactly what the NVIDIA run answers.
- Caveat: nn2 small-batch numbers include nn2's per-call overhead (Sequential mid-buffer
  handling); both columns are Metal and not the target platform. Treat the *shape of the
  story* (compute vs delivered, small vs large batch) as the takeaway, not absolute µs.

## Path B (no-Python hot path): investigated — blocked on Apple, GO/NO-GO probe for NVIDIA

Path B = Mojo → MAX **C API** (`M_compileModel` → `M_initModel` → `M_executeModelSync`) on a
precompiled artifact, removing Python (and its ~hundreds-of-µs glue) from the hot loop.

**Finding on Apple (2026-06-03): blocked at the linker level.**
- The C API is **header-only** here: `include/max/c/*.h` ship, but a scan of *every*
  `.dylib`/`.so`/`.a` in the env finds **zero** exported `M_compileModel` /
  `M_executeModelSync` / `M_newRuntimeContext` symbols. The engine is compiled *into* the
  Python extension `max/_core*.so` with hidden visibility — only reachable through the
  Python bindings, not linkable or `dlopen`-able from Mojo.
- There is also **no exposed Python path to emit a `.mef`** (no save/export/serialize on
  `CompiledModel` or `max._core`), so you can't even hand the C API a precompiled artifact.
- ⇒ Path B is not buildable on this Apple install. (This *extends* the training assessment:
  even the "Mojo in the hot path via C API" fallback is unavailable here.)

**The C API library genuinely might ship in the linux/NVIDIA MAX package** (the C API and its
`examples/capi` are linux-oriented). Run the probe there — it prints a GO/NO-GO verdict:
```bash
pixi run -e nvidia bash max_rl/probe_c_api.sh
```
(`probe_c_api.sh` checks headers, scans all libs for the exported C-ABI symbols, and checks
for a MEF-export path. On Apple it correctly reports NO-GO.)

**If the probe says GO on NVIDIA, the FFI binding is mechanical** — Mojo `external_call` over
this sequence (signatures already mapped from `include/max/c/{common,context,device,model,tensor}.h`):
```
M_newStatus → M_newRuntimeConfig → (M_newDevice|M_createAcceleratorDevice) →
M_runtimeConfigAddDevice → M_newRuntimeContext →
M_newCompileConfig → M_setModelPath(<artifact>) → M_compileModel → M_waitForCompilation →
M_initModel → M_waitForModel →
[hot loop] M_newAsyncTensorMap → M_newTensorSpec → M_borrowTensorInto →
           M_executeModelSync → M_getTensorByNameFrom → M_getTensorData
```
Open sub-question even on GO: what file `M_setModelPath` accepts. If no `.mef` export exists,
serialize the graph's MLIR (`Graph.module`) from Python and let the C API compile it once at
startup — Python stays out of the *execution* loop, which is the point.

## Not yet done (next steps)
- **NVIDIA run** (the decisive measurement — your part): the interop decomposition + nn2
  baseline on CUDA, re-verify the `mojo run`/`M::Context` clash, and run `probe_c_api.sh`.
- If probe = GO: implement the Mojo C-API FFI binding above and measure the no-Python path.
- bf16 / fp16, CUDA-graph capture/replay (`M_captureModelSync` / `M_replayModelSync`).

## Done
- ✅ `MLPInference` MAX package (configurable dims/batch/device) + interop decomposition.
- ✅ nn2 native baseline on identical shapes (`benchmark_nn2_baseline.mojo`).
- ✅ Path B feasibility probe (`probe_c_api.sh`) + finding (blocked on Apple; GO/NO-GO for NVIDIA).
