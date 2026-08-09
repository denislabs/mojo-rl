/*
 * CUDA kernel launch interceptor via dlsym hooking.
 *
 * Mojo loads libcuda.so via dlopen/dlsym, which bypasses LD_PRELOAD.
 * We intercept dlsym itself to return our wrappers when Mojo requests
 * cuLaunchKernel or cuLaunchKernelEx.
 *
 * Build (handled by pixi activation script):
 *   gcc -shared -fPIC -o mojo_rl/cuda/libcuda_intercept.so mojo_rl/cuda/cuda_intercept.c -ldl
 *
 * Loaded automatically via LD_PRELOAD in the nvidia pixi environment.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Opaque CUDA types */
typedef void* CUfunction;
typedef void* CUstream;
typedef int CUresult;

/* CUlaunchConfig for cuLaunchKernelEx (CUDA 12+) */
typedef struct {
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void **attrs;       /* CUlaunchAttribute array */
    unsigned int numAttrs;
} CUlaunchConfig;

/* ---- Recording state ---- */

typedef struct {
    CUfunction func;
    unsigned int gridX, gridY, gridZ;
    unsigned int blockX, blockY, blockZ;
    unsigned int sharedMem;
    CUstream stream;
} KernelRecord;

#define MAX_RECORDS 256
static KernelRecord g_records[MAX_RECORDS];
static int g_num_records = 0;
static int g_recording = 0;
static int g_logging = 0;  /* quiet by default — enable via intercept_set_logging(1) */
static int g_launch_count = 0;
static CUstream g_mojo_stream = NULL;  /* the stream Mojo actually uses */

/* ---- Real function pointers (resolved from libcuda.so) ---- */

typedef CUresult (*cuLaunchKernel_t)(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra);

typedef CUresult (*cuLaunchKernelEx_t)(
    const CUlaunchConfig *config, CUfunction f,
    void **kernelParams, void **extra);

static cuLaunchKernel_t real_cuLaunchKernel = NULL;
static cuLaunchKernelEx_t real_cuLaunchKernelEx = NULL;

/* ---- Our wrapper functions ---- */

static CUresult wrapped_cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra)
{
    g_launch_count++;

    if (g_logging) {
        fprintf(stderr, "[intercept] cuLaunchKernel #%d: func=%p grid=(%u,%u,%u) "
                "block=(%u,%u,%u) shm=%u stream=%p\n",
                g_launch_count, f, gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream);
    }

    if (g_recording && g_num_records < MAX_RECORDS) {
        KernelRecord *r = &g_records[g_num_records++];
        r->func = f;
        r->gridX = gridDimX; r->gridY = gridDimY; r->gridZ = gridDimZ;
        r->blockX = blockDimX; r->blockY = blockDimY; r->blockZ = blockDimZ;
        r->sharedMem = sharedMemBytes;
        r->stream = hStream;
    }

    return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                blockDimX, blockDimY, blockDimZ,
                                sharedMemBytes, hStream,
                                kernelParams, extra);
}

static CUresult wrapped_cuLaunchKernelEx(
    const CUlaunchConfig *config, CUfunction f,
    void **kernelParams, void **extra)
{
    g_launch_count++;

    if (g_logging) {
        fprintf(stderr, "[intercept] cuLaunchKernelEx #%d: func=%p grid=(%u,%u,%u) "
                "block=(%u,%u,%u) shm=%u stream=%p\n",
                g_launch_count, f,
                config->gridDimX, config->gridDimY, config->gridDimZ,
                config->blockDimX, config->blockDimY, config->blockDimZ,
                config->sharedMemBytes, config->hStream);
    }

    /* Capture the stream Mojo uses */
    if (!g_mojo_stream && config->hStream) {
        g_mojo_stream = config->hStream;
        fprintf(stderr, "[intercept] Captured Mojo stream: %p\n", g_mojo_stream);
    }

    if (g_recording && g_num_records < MAX_RECORDS) {
        KernelRecord *r = &g_records[g_num_records++];
        r->func = f;
        r->gridX = config->gridDimX; r->gridY = config->gridDimY; r->gridZ = config->gridDimZ;
        r->blockX = config->blockDimX; r->blockY = config->blockDimY; r->blockZ = config->blockDimZ;
        r->sharedMem = config->sharedMemBytes;
        r->stream = config->hStream;
    }

    return real_cuLaunchKernelEx(config, f, kernelParams, extra);
}

/* ---- Unsafe-during-capture call tracing --------------------------------
 *
 * ⚠ WHY THIS EXISTS: EVERY MOJO `print()` IS LOST WHEN THE RUN ABORTS.
 * Mojo's stdout is block-buffered and an abort never flushes it, so a
 * crashing capture shows ONLY the `[intercept]` lines — these go to stderr,
 * which is unbuffered. That is why three separate NVIDIA runs told us
 * nothing about how far the test got: the test's own prints never made it
 * out. Anything we want to see from a crashing run must be printed HERE.
 *
 * WHAT WE ARE HUNTING. `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` is raised by
 * MAX (its own error formatter, not ours -- our wrappers only return rc
 * codes), and it survives capture modes GLOBAL, THREAD_LOCAL and RELAXED
 * alike. Mode only governs calls on OTHER threads/resources, so a failure in
 * all three modes means the call is on the CAPTURED STREAM ITSELF, which is
 * illegal unconditionally. These wrappers name it instead of inferring it.
 *
 * ⚠ HOOKED VIA BOTH PATHS ON PURPOSE. A dlsym hook alone is not enough:
 * modern CUDA prefers `cuGetProcAddress`, and anything MAX resolves that way
 * is invisible to LD_PRELOAD. Both routes funnel through `maybe_wrap`, so if
 * NOTHING prints below, that is itself the finding -- it means the offending
 * call reaches the driver by a third route, and the next move is to widen
 * the hook rather than to keep editing the capture code.
 */

static int g_capturing = 0;
static CUstream g_capture_stream = NULL;

typedef CUresult (*fn_ptr1_t)(void*);
typedef CUresult (*fn_void_t)(void);
typedef CUresult (*fn_alloc_t)(void**, size_t);

static fn_ptr1_t  real_cuStreamSynchronize = NULL;
static fn_ptr1_t  real_cuStreamQuery       = NULL;
static fn_ptr1_t  real_cuEventSynchronize  = NULL;
static fn_ptr1_t  real_cuEventQuery        = NULL;
static fn_void_t  real_cuCtxSynchronize    = NULL;
static fn_alloc_t real_cuMemAlloc          = NULL;

static void note_unsafe(const char *name, void *arg) {
    if (!g_capturing) return;
    fprintf(stderr, "[intercept] !! %s(%p) DURING CAPTURE%s\n", name, arg,
            (arg && arg == g_capture_stream)
                ? "  <-- ON THE CAPTURED STREAM: illegal in EVERY mode"
                : "  (other resource; legal under RELAXED)");
}

#define DEFINE_PTR1_WRAPPER(NAME)                                             \
    static CUresult wrapped_##NAME(void *a) {                                 \
        note_unsafe(#NAME, a);                                                \
        return real_##NAME(a);                                                \
    }

DEFINE_PTR1_WRAPPER(cuEventSynchronize)
DEFINE_PTR1_WRAPPER(cuEventQuery)

/* ---- Neutralizing MAX's sync of the stream it is capturing --------------
 *
 * MEASURED (NVIDIA, MAX 26.5.0rc2, 2026-08-09): between our successful
 * `cuStreamBeginCapture` and the first captured kernel, MAX calls
 * `cuStreamSynchronize` ON THE VERY STREAM BEING CAPTURED -- same handle in
 * both trace lines. That is illegal in EVERY capture mode, which is exactly
 * why GLOBAL / THREAD_LOCAL / RELAXED all aborted identically. It is MAX's
 * call, not ours: our own wrappers only ever return rc codes, and the abort
 * carries MAX's error text.
 *
 * ⚠ WHY ANSWERING "ALREADY DONE" IS CORRECT, NOT A LIE. Capture RECORDS work
 * instead of executing it, so while a stream is capturing there is by
 * construction nothing running on it to wait for. The precondition that
 * makes this airtight is ours to keep, and we do keep it: the stream is
 * fully drained before capture opens -- `CUDAGraph.__init__` calls
 * `ctx.synchronize()`, and `maybe_capture_replay` synchronizes again before
 * constructing the graph. So no work enqueued BEFORE the window can still be
 * in flight, and no work enqueued DURING it executes. "Wait until idle" is
 * already true. CUDA rejects the call because its meaning is ambiguous mid
 * capture, not because there is anything pending.
 *
 * ⚠ WHAT WOULD MAKE IT A LIE. If a future caller opens a capture WITHOUT
 * draining first, this turns a real wait into a no-op and reads back garbage
 * silently. Any new capture site must sync before `begin_capture`. That is
 * the invariant this whole comment exists to protect.
 *
 * ⚠ THIS IS NOT SELF-VALIDATING -- the gate is. Suppressing an error can
 * always convert a loud failure into a quiet wrong answer, so it is only
 * defensible because `test_cuda_graph_minimal` checks that the graph is
 * non-empty (`num_nodes() > 0`) AND that replay produces exact counter
 * arithmetic. If capture were silently recording nothing, both fail.
 *
 * Set MOJO_RL_CAPTURE_NEUTRALIZE_SYNC=0 to restore the raw behaviour and get
 * the abort back (i.e. to re-measure this whenever MAX is upgraded).
 */

static int g_neutralized = 0;   /* count, per capture window */

static int neutralize_enabled(void) {
    const char *e = getenv("MOJO_RL_CAPTURE_NEUTRALIZE_SYNC");
    return !(e && e[0] == '0');
}

static CUresult wrapped_cuStreamSynchronize(void *a) {
    if (g_capturing && a == g_capture_stream && neutralize_enabled()) {
        /* Log the first few only: a capture window in a real trainer holds
           many kernels, and one line per call would drown the output. */
        if (g_neutralized < 3) {
            fprintf(stderr,
                    "[intercept] cuStreamSynchronize(%p) on the CAPTURING "
                    "stream -> answered SUCCESS without calling the driver "
                    "(nothing executes during capture)\n", a);
        }
        g_neutralized++;
        return 0;  /* CUDA_SUCCESS */
    }
    note_unsafe("cuStreamSynchronize", a);
    return real_cuStreamSynchronize(a);
}

static CUresult wrapped_cuStreamQuery(void *a) {
    /* Same argument: "is this stream idle?" is trivially yes during capture. */
    if (g_capturing && a == g_capture_stream && neutralize_enabled()) {
        g_neutralized++;
        return 0;  /* CUDA_SUCCESS == all work complete */
    }
    note_unsafe("cuStreamQuery", a);
    return real_cuStreamQuery(a);
}

static CUresult wrapped_cuCtxSynchronize(void) {
    /* Context-wide sync: hits the captured stream by definition. */
    note_unsafe("cuCtxSynchronize", g_capture_stream);
    return real_cuCtxSynchronize();
}

static CUresult wrapped_cuMemAlloc(void **p, size_t n) {
    if (g_capturing)
        fprintf(stderr,
                "[intercept] !! cuMemAlloc(%zu bytes) DURING CAPTURE\n", n);
    return real_cuMemAlloc(p, n);
}

/* Shared by the dlsym hook and the cuGetProcAddress hook. */
static void *maybe_wrap(const char *symbol, void *real_fn) {
    if (!symbol || !real_fn) return real_fn;

#define HOOK_PTR1(NAME)                                                       \
    if (strcmp(symbol, #NAME) == 0) {                                         \
        if (!real_##NAME) real_##NAME = (fn_ptr1_t)real_fn;                   \
        return (void*)wrapped_##NAME;                                         \
    }
    HOOK_PTR1(cuStreamSynchronize)
    HOOK_PTR1(cuStreamQuery)
    HOOK_PTR1(cuEventSynchronize)
    HOOK_PTR1(cuEventQuery)
#undef HOOK_PTR1

    if (strcmp(symbol, "cuCtxSynchronize") == 0) {
        if (!real_cuCtxSynchronize) real_cuCtxSynchronize = (fn_void_t)real_fn;
        return (void*)wrapped_cuCtxSynchronize;
    }
    /* `_v2` is the ABI the driver actually exports for cuMemAlloc. */
    if (strcmp(symbol, "cuMemAlloc") == 0
        || strcmp(symbol, "cuMemAlloc_v2") == 0) {
        if (!real_cuMemAlloc) real_cuMemAlloc = (fn_alloc_t)real_fn;
        return (void*)wrapped_cuMemAlloc;
    }
    return real_fn;
}

/* cuGetProcAddress: the route a dlsym hook cannot see. We let the real one
   resolve, then swap our wrapper into the caller's out-pointer. */
typedef CUresult (*cuGetProcAddress_t)(
    const char*, void**, int, unsigned long long);
typedef CUresult (*cuGetProcAddress_v2_t)(
    const char*, void**, int, unsigned long long, int*);

static cuGetProcAddress_t    real_cuGetProcAddress    = NULL;
static cuGetProcAddress_v2_t real_cuGetProcAddress_v2 = NULL;

static CUresult wrapped_cuGetProcAddress(
    const char *sym, void **pfn, int ver, unsigned long long flags)
{
    CUresult r = real_cuGetProcAddress(sym, pfn, ver, flags);
    if (r == 0 && pfn && *pfn) *pfn = maybe_wrap(sym, *pfn);
    return r;
}

static CUresult wrapped_cuGetProcAddress_v2(
    const char *sym, void **pfn, int ver, unsigned long long flags, int *status)
{
    CUresult r = real_cuGetProcAddress_v2(sym, pfn, ver, flags, status);
    if (r == 0 && pfn && *pfn) *pfn = maybe_wrap(sym, *pfn);
    return r;
}

/* ---- dlsym interception ---- */

/* We need the real dlsym to resolve everything else */
typedef void* (*dlsym_t)(void *handle, const char *symbol);
static dlsym_t real_dlsym = NULL;

static void ensure_real_dlsym(void) {
    if (!real_dlsym) {
        /* Use dlvsym to get the real dlsym from glibc */
        real_dlsym = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym) {
            /* Fallback: try without version */
            real_dlsym = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17");
        }
        if (!real_dlsym) {
            fprintf(stderr, "[intercept] WARNING: could not resolve real dlsym via dlvsym, "
                    "trying __libc_dlsym\n");
            /* Last resort */
            void *libc = dlopen("libc.so.6", RTLD_LAZY | RTLD_NOLOAD);
            if (libc) {
                real_dlsym = (dlsym_t)dlvsym(libc, "dlsym", "GLIBC_2.2.5");
            }
        }
        if (real_dlsym) {
            fprintf(stderr, "[intercept] dlsym hook installed successfully\n");
        } else {
            fprintf(stderr, "[intercept] FATAL: cannot find real dlsym\n");
        }
    }
}

/* Override dlsym — this IS picked up by LD_PRELOAD since dlsym is called
   through the normal dynamic linker path */
void* dlsym(void *handle, const char *symbol) {
    ensure_real_dlsym();

    if (!symbol || !real_dlsym) {
        return real_dlsym ? real_dlsym(handle, symbol) : NULL;
    }

    /* Intercept cuLaunchKernel */
    if (strcmp(symbol, "cuLaunchKernel") == 0) {
        /* Save the real function pointer */
        void *real_fn = real_dlsym(handle, symbol);
        if (real_fn && !real_cuLaunchKernel) {
            real_cuLaunchKernel = (cuLaunchKernel_t)real_fn;
            fprintf(stderr, "[intercept] Hooked cuLaunchKernel -> %p\n", real_fn);
        }
        return (void*)wrapped_cuLaunchKernel;
    }

    /* Intercept cuLaunchKernelEx */
    if (strcmp(symbol, "cuLaunchKernelEx") == 0) {
        void *real_fn = real_dlsym(handle, symbol);
        if (real_fn && !real_cuLaunchKernelEx) {
            real_cuLaunchKernelEx = (cuLaunchKernelEx_t)real_fn;
            fprintf(stderr, "[intercept] Hooked cuLaunchKernelEx -> %p\n", real_fn);
        }
        return (void*)wrapped_cuLaunchKernelEx;
    }

    /* Intercept cuGetProcAddress so symbols resolved through it are wrapped
       too — see the note above `maybe_wrap`. */
    if (strcmp(symbol, "cuGetProcAddress") == 0) {
        void *real_fn = real_dlsym(handle, symbol);
        if (real_fn && !real_cuGetProcAddress) {
            real_cuGetProcAddress = (cuGetProcAddress_t)real_fn;
            fprintf(stderr, "[intercept] Hooked cuGetProcAddress -> %p\n",
                    real_fn);
        }
        return real_fn ? (void*)wrapped_cuGetProcAddress : NULL;
    }
    if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        void *real_fn = real_dlsym(handle, symbol);
        if (real_fn && !real_cuGetProcAddress_v2) {
            real_cuGetProcAddress_v2 = (cuGetProcAddress_v2_t)real_fn;
            fprintf(stderr, "[intercept] Hooked cuGetProcAddress_v2 -> %p\n",
                    real_fn);
        }
        return real_fn ? (void*)wrapped_cuGetProcAddress_v2 : NULL;
    }

    return maybe_wrap(symbol, real_dlsym(handle, symbol));
}

/* ---- API callable from Mojo via FFI ---- */

void intercept_set_logging(int enabled) {
    g_logging = enabled;
}

int intercept_get_launch_count(void) {
    return g_launch_count;
}

/* Returns the stream handle that Mojo's AsyncRT actually dispatches on.
   Call after at least one warmup kernel launch. */
void* intercept_get_mojo_stream(void) {
    return (void*)g_mojo_stream;
}

/* ---- CUDA Graph API wrappers ----
   These resolve CUDA functions via the already-loaded libcuda.so,
   avoiding the need for Mojo to dlopen("libcuda.so") separately
   (which can cause re-entrant crashes with the dlsym hook). */

static void *g_libcuda = NULL;

static void* cuda_fn(const char *name) {
    if (!g_libcuda) {
        g_libcuda = dlopen("libcuda.so", RTLD_LAZY | RTLD_NOLOAD);
        if (!g_libcuda) g_libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (!g_libcuda || !real_dlsym) return NULL;
    return real_dlsym(g_libcuda, name);
}

CUresult intercept_stream_create(void **out) {
    typedef CUresult (*fn_t)(void**, unsigned int);
    fn_t f = (fn_t)cuda_fn("cuStreamCreate");
    if (!f) return 1;
    return f(out, 0);
}

/*
 * Capture mode. CUDA defines three, in decreasing strictness:
 *
 *   0 GLOBAL       any "unsafe" CUDA call, ON ANY THREAD, in ANY resource,
 *                  is an error for the whole duration of the capture.
 *   1 THREAD_LOCAL the prohibition applies only to the capturing thread.
 *   2 RELAXED      only actions on the capturing stream itself are policed.
 *
 * ⚠ WE USED TO PASS GLOBAL, AND THAT MADE US HOSTAGE TO MAX'S INTERNALS.
 * We do not own every CUDA call in this process — MAX/AsyncRT is
 * multithreaded and makes its own driver calls. Under GLOBAL, one
 * synchronize or allocation ANYWHERE, on a stream we have nothing to do
 * with, aborts the run with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED. Measured
 * on 2026-08-09: MAX (26.5.0rc2) trips this reliably during a capture of a
 * single one-thread kernel, and the abort comes from MAX's error handler,
 * not from any interceptor call — our cuStreamBeginCapture had already
 * returned 0. GLOBAL was policing code we neither wrote nor can fix.
 *
 * RELAXED is what we actually mean: "nothing else may touch THIS stream
 * while we record it". Overridable so the three modes are one env var apart
 * rather than a rebuild apart, and because the choice is a DISCRIMINATING
 * test, not just a workaround:
 *
 *   fails at GLOBAL, works at THREAD_LOCAL -> the offending call is on
 *                                             another MAX thread
 *   fails at THREAD_LOCAL, works at RELAXED -> it is on our thread, but on
 *                                              some other stream/resource
 *   fails at RELAXED                        -> something really is touching
 *                                              the captured stream; that
 *                                              WOULD be our bug
 */
static int capture_mode(void) {
    const char *e = getenv("MOJO_RL_CAPTURE_MODE");
    if (e && *e) {
        int m = atoi(e);
        if (m >= 0 && m <= 2) return m;
    }
    return 2;  /* CU_STREAM_CAPTURE_MODE_RELAXED */
}

CUresult intercept_stream_begin_capture(void *stream) {
    typedef CUresult (*fn_t)(void*, int);
    fn_t f = (fn_t)cuda_fn("cuStreamBeginCapture");
    if (!f) return 1;
    int mode = capture_mode();
    fprintf(stderr, "[intercept] cuStreamBeginCapture stream=%p mode=%d (%s)\n",
            stream, mode,
            mode == 0 ? "GLOBAL" : (mode == 1 ? "THREAD_LOCAL" : "RELAXED"));
    CUresult r = f(stream, mode);
    if (r == 0) {
        g_capturing = 1;
        g_capture_stream = stream;
        g_neutralized = 0;
    }
    fprintf(stderr, "[intercept] cuStreamBeginCapture rc=%d\n", (int)r);
    return r;
}

CUresult intercept_stream_end_capture(void *stream, void **graph_out) {
    typedef CUresult (*fn_t)(void*, void**);
    fn_t f = (fn_t)cuda_fn("cuStreamEndCapture");
    if (!f) return 1;
    /* Clear the flag BEFORE the call: cuStreamEndCapture is itself an
       operation on the capturing stream, and tracing it would be noise. */
    g_capturing = 0;
    g_capture_stream = NULL;
    CUresult r = f(stream, graph_out);
    fprintf(stderr,
            "[intercept] cuStreamEndCapture rc=%d graph=%p "
            "(neutralized %d sync/query call(s) on the captured stream)\n",
            (int)r, graph_out ? *graph_out : NULL, g_neutralized);
    return r;
}

CUresult intercept_graph_instantiate(void **exec_out, void *graph) {
    typedef CUresult (*fn_t)(void**, void*, unsigned long long);
    fn_t f = (fn_t)cuda_fn("cuGraphInstantiate");
    if (!f) return 1;
    return f(exec_out, graph, 0ULL);
}

CUresult intercept_graph_launch(void *exec, void *stream) {
    typedef CUresult (*fn_t)(void*, void*);
    fn_t f = (fn_t)cuda_fn("cuGraphLaunch");
    if (!f) return 1;
    return f(exec, stream);
}

CUresult intercept_stream_synchronize(void *stream) {
    typedef CUresult (*fn_t)(void*);
    fn_t f = (fn_t)cuda_fn("cuStreamSynchronize");
    if (!f) return 1;
    return f(stream);
}

CUresult intercept_graph_destroy(void *graph) {
    typedef CUresult (*fn_t)(void*);
    fn_t f = (fn_t)cuda_fn("cuGraphDestroy");
    if (!f) return 1;
    return f(graph);
}

CUresult intercept_graph_exec_destroy(void *exec) {
    typedef CUresult (*fn_t)(void*);
    fn_t f = (fn_t)cuda_fn("cuGraphExecDestroy");
    if (!f) return 1;
    return f(exec);
}

CUresult intercept_graph_get_nodes(void *graph, unsigned long long *num_nodes) {
    typedef CUresult (*fn_t)(void*, void*, unsigned long long*);
    fn_t f = (fn_t)cuda_fn("cuGraphGetNodes");
    if (!f) return 1;
    return f(graph, NULL, num_nodes);
}

/* Constructor: print banner on load */
__attribute__((constructor))
static void on_load(void) {
    fprintf(stderr, "[intercept] CUDA interceptor loaded (dlsym hooking mode)\n");
}
