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
static int g_stream_changes = 0;   /* rate limit: hot path */
static int g_destroy_logged = 0;   /* rate limit: fires every synchronize */
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

    /* Capture the stream Mojo uses -- and the CONTEXT it was bound to. This
       hook runs on MAX's OWN thread, so this is the one place we can observe
       a context that is known-good for driver calls. */
    /* ⚠ TRACK THE CURRENT STREAM, DO NOT CACHE THE FIRST ONE. MAX destroys
       its stream (measured: cuStreamDestroy right after ctx.synchronize()),
       so a handle saved once is a dangling pointer minutes later and every
       driver call we make with it is a use-after-free. Re-recording on every
       launch means `intercept_get_mojo_stream` answers with a stream that was
       alive as of the last launch, and the cuStreamDestroy hook NULLs it the
       moment it dies -- so callers get NULL (a clean, checkable answer)
       rather than a freed pointer that faults somewhere inside libcuda. */
    /* ⚠ THIS IS THE HOT PATH — EVERY KERNEL LAUNCH. Keep it to one pointer
       compare and one store. It previously logged and called cuCtxGetCurrent
       on each CHANGE, which looked cheap until we learned MAX destroys and
       recreates its stream on EVERY ctx.synchronize(): that turned into an
       unbuffered stderr write plus a dlopen+dlsym+driver round trip per sync,
       and it measurably slowed training down. Diagnostics here are rate
       limited and must stay that way. */
    if (config->hStream && config->hStream != g_mojo_stream) {
        if (g_logging || g_stream_changes < 4) {
            fprintf(stderr, "[intercept] Mojo stream: %p%s\n",
                    config->hStream,
                    g_mojo_stream ? " (changed)" : "");
            if (g_stream_changes == 3) {
                fprintf(stderr, "[intercept] (further stream changes silent —"
                        " MAX recreates its stream on every synchronize)\n");
            }
        }
        g_stream_changes++;
        g_mojo_stream = config->hStream;
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

/* ⚠ THE "NEUTRALIZE MAX'S MID-CAPTURE SYNC" MACHINERY WAS REMOVED.
 * It existed because `cuStreamSynchronize` was seen on the capturing stream
 * and read as MAX bookkeeping we had to suppress. It was not: it was MAX's
 * DeviceContext DESTRUCTOR, running because Mojo destroys a value at its last
 * use and `CUDAGraph` did not hold the context. With the context held, a
 * capture reports `neutralized 0` — MAX does not synchronize inside the
 * window at all. Suppressing a real synchronize is a lie about work being
 * complete, so the code is gone rather than left switchable.
 *
 * The wrappers below still TRACE such calls, which costs nothing outside a
 * capture and would immediately show a regression here.
 */
static CUresult wrapped_cuStreamSynchronize(void *a) {
    note_unsafe("cuStreamSynchronize", a);
    return real_cuStreamSynchronize(a);
}

static CUresult wrapped_cuStreamQuery(void *a) {
    note_unsafe("cuStreamQuery", a);
    return real_cuStreamQuery(a);
}

static CUresult wrapped_cuCtxSynchronize(void) {
    /* Context-wide sync: reaches the captured stream by definition. */
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
static fn_ptr1_t real_cuStreamDestroy = NULL;

/* ⚠ THE SMOKING GUN, IF IT FIRES. We hold MAX's stream handle for the whole
   program. If MAX destroys it, every later intercept_* call is a
   use-after-free on a driver object -- which faults erratically rather than
   returning an error, and would explain rc=0 from one entry point and a
   SIGSEGV from another. */
static CUresult wrapped_cuStreamDestroy(void *a) {
    /* ⚠ RATE LIMITED: MAX destroys its stream on EVERY ctx.synchronize(), so
       an unconditional log here is one stderr syscall per sync in the
       training loop. NULLing the handle is the part that must always run —
       it is what turns a later use-after-free into a checkable NULL. */
    if (g_logging || g_destroy_logged < 2) {
        fprintf(stderr, "[intercept] cuStreamDestroy(%p)%s\n", a,
                (a && a == g_mojo_stream) ? "  <-- the stream we track" : "");
        g_destroy_logged++;
    }
    if (a && a == g_mojo_stream) g_mojo_stream = NULL;
    return real_cuStreamDestroy(a);
}

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

    if (strcmp(symbol, "cuStreamDestroy") == 0
        || strcmp(symbol, "cuStreamDestroy_v2") == 0) {
        if (!real_cuStreamDestroy) real_cuStreamDestroy = (fn_ptr1_t)real_fn;
        return (void*)wrapped_cuStreamDestroy;
    }
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

/* ⚠⚠ THE BASE SYMBOL NAME IS NOT THE ABI THE CUDA HEADER GIVES YOU.
 *
 * `cuda.h` is a nest of macros: `cuStreamBeginCapture` expands to
 * `cuStreamBeginCapture_v2`, `cuGraphInstantiate` to `cuGraphInstantiate_v2`,
 * `cuStreamGetCaptureInfo` to `..._v2`, and so on. libcuda.so.1 exports BOTH
 * — the base name keeps the OLD ABI forever for binary compatibility. So
 * `dlsym(lib, "cuStreamBeginCapture")` does NOT return what a program
 * compiled against the header calls; it returns the ANCIENT one, with a
 * DIFFERENT ARITY.
 *
 * On x86-64 SysV, calling with too FEW arguments is silent memory
 * corruption: the callee reads whatever junk is in the remaining registers
 * and, for out-parameters, WRITES THROUGH IT. Too many is harmless (extras
 * are ignored). Measured here, all three from this file:
 *
 *   cuStreamGetCaptureInfo   base = 3 args, _v2 = 6. Called with 3
 *                            -> driver wrote through 3 garbage pointers
 *                            -> SIGSEGV inside cuStreamGetCaptureInfo.
 *   cuGraphInstantiate       base = 5 args (exec, graph, phErrorNode,
 *                            logBuffer, bufferSize). Called with 3
 *                            -> logBuffer/bufferSize are garbage and the
 *                            driver writes its error log into them.
 *   cuStreamBeginCapture     base = 1 arg (v1 predates capture modes).
 *                            Called with 2 -> THE MODE WAS SILENTLY
 *                            DISCARDED. Every "capture mode" experiment we
 *                            ran was really GLOBAL, which is exactly why
 *                            GLOBAL/THREAD_LOCAL/RELAXED were byte-identical.
 *
 * ⚠ SO: RESOLVE THE VERSIONED NAME EXPLICITLY, and match its arity. Never
 * add a `cuda_fn("cuSomething")` here without checking the base symbol's
 * arity against the `_v2`/`_v3` the header actually selects. Symbols with a
 * single lifelong ABI (cuStreamCreate, cuStreamEndCapture, cuGraphLaunch,
 * cuGraphDestroy, cuGraphExecDestroy, cuGraphGetNodes, cuStreamSynchronize)
 * are fine under the base name and are left alone.
 */
static void* cuda_fn_versioned(const char *versioned, const char *base) {
    void *f = cuda_fn(versioned);
    if (f) return f;
    fprintf(stderr, "[intercept] WARNING: %s unavailable, falling back to %s "
            "— CHECK THE ARITY MATCHES\n", versioned, base);
    return cuda_fn(base);
}

/* ⚠ THE CUDA-CONTEXT-AFFINITY DIAGNOSTIC WAS REMOVED — THE THEORY WAS
 * REFUTED. It checked whether a context was current on our thread, on the
 * idea that MAX drove the driver from a worker thread where one was bound
 * and we did not. Measured: the current context was IDENTICAL to MAX's, and
 * binding it explicitly changed nothing. The faults were a freed stream, not
 * a missing context. Removing it also removes a dlsym + driver round trip
 * from every entry point.
 */

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
    /* ⚠ _v2 IS THE ONE THAT TAKES A MODE. The base symbol is v1, which
       predates capture modes entirely and takes ONLY the stream — passing a
       mode to it is discarded in silence. See the note on cuda_fn_versioned. */
    typedef CUresult (*fn_t)(void*, int);
    fn_t f = (fn_t)cuda_fn_versioned("cuStreamBeginCapture_v2",
                                     "cuStreamBeginCapture");
    if (!f) return 1;
    int mode = capture_mode();
    fprintf(stderr, "[intercept] cuStreamBeginCapture stream=%p mode=%d (%s)\n",
            stream, mode,
            mode == 0 ? "GLOBAL" : (mode == 1 ? "THREAD_LOCAL" : "RELAXED"));
    CUresult r = f(stream, mode);
    if (r == 0) {
        g_capturing = 1;
        g_capture_stream = stream;
    }
    fprintf(stderr, "[intercept] cuStreamBeginCapture rc=%d\n", (int)r);
    return r;
}

/* Ask the driver what it thinks the capture's state is.
   Returns the CUstreamCaptureStatus (0 NONE / 1 ACTIVE / 2 INVALIDATED), or
   -1 if the symbol is unavailable. */
static int query_capture_status(void *stream, unsigned long long *id_out) {
    /* ⚠ SIX ARGUMENTS. The `_v2` ABI is
         (stream, status_out, id_out, graph_out, deps_out, num_deps_out)
       and the base symbol takes only the first three. Calling the v2 symbol
       with three arguments made the driver write through three garbage
       registers -- that is the SIGSEGV inside cuStreamGetCaptureInfo we saw.
       Every out-param below is a real local, so the call is safe whichever
       ABI we land on: extra arguments are ignored by an older callee. */
    typedef CUresult (*fn_t)(void*, int*, unsigned long long*,
                             void**, const void**, size_t*);
    fn_t f = (fn_t)cuda_fn_versioned("cuStreamGetCaptureInfo_v2",
                                     "cuStreamGetCaptureInfo");
    if (!f) return -1;

    int status = -1;
    unsigned long long id = 0;
    void *graph = NULL;
    const void *deps = NULL;
    size_t num_deps = 0;
    CUresult r = f(stream, &status, &id, &graph, &deps, &num_deps);
    if (id_out) *id_out = id;
    if (r != 0) {
        fprintf(stderr, "[intercept] cuStreamGetCaptureInfo rc=%d\n", (int)r);
        return -1;
    }
    return status;
}

CUresult intercept_stream_end_capture(void *stream, void **graph_out) {
    typedef CUresult (*fn_t)(void*, void**);
    fn_t f = (fn_t)cuda_fn("cuStreamEndCapture");
    if (!f) return 1;
    /* Clear the flag BEFORE the call: cuStreamEndCapture is itself an
       operation on the capturing stream, and tracing it would be noise. */
    g_capturing = 0;
    g_capture_stream = NULL;

    /* ⚠ ASK BEFORE ENDING. `cuStreamEndCapture` SEGFAULTED INSIDE THE DRIVER
       here (measured NVIDIA 2026-08-09, stack: cuStreamEndCapture ->
       libcuda+0x38895e -> fault), which is not a behaviour the API documents
       — it should return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED. A crash
       inside the driver usually means we are ending a capture the driver no
       longer considers ours, so establish that BEFORE calling in, while we
       can still print. Refusing to call on a non-ACTIVE capture converts a
       segfault into an rc that `graph.mojo` raises as a clean Mojo error. */
    /* ⚠ OPT-IN, BECAUSE THE PROBE ITSELF CRASHED THE DRIVER. Even called
       with the full 6-argument `_v2` ABI and real locals for every
       out-param, `cuStreamGetCaptureInfo_v2` faulted inside libcuda on this
       driver. Whatever that means, a diagnostic that aborts the run is worse
       than no diagnostic -- it displaced the real failure and cost a round
       trip. Off by default; MOJO_RL_CAPTURE_PROBE=1 to re-enable. */
    const char *probe = getenv("MOJO_RL_CAPTURE_PROBE");
    unsigned long long cap_id = 0;
    int status = (probe && probe[0] != '0')
                     ? query_capture_status(stream, &cap_id)
                     : 1 /* assume ACTIVE; do not touch the driver */;
    (void)cap_id;
    if (probe && probe[0] != '0') {
        fprintf(stderr,
                "[intercept] pre-end capture status=%d (%s) id=%llu stream=%p\n",
                status,
                status == 0 ? "NONE — the stream is NOT capturing"
                            : (status == 1 ? "ACTIVE"
                            : (status == 2 ? "INVALIDATED" : "unavailable")),
                cap_id, stream);
    }
    if (status == 0 || status == 2) {
        fprintf(stderr,
                "[intercept] refusing to call cuStreamEndCapture on a "
                "non-ACTIVE capture (that is what crashed the driver)\n");
        if (graph_out) *graph_out = NULL;
        return 900;  /* distinct rc; surfaces as a [CUDAGraph] Mojo error */
    }

    CUresult r = f(stream, graph_out);
    fprintf(stderr, "[intercept] cuStreamEndCapture rc=%d graph=%p\n",
            (int)r, graph_out ? *graph_out : NULL);
    return r;
}

CUresult intercept_graph_instantiate(void **exec_out, void *graph) {
    /* ⚠ THE 3-ARG FORM IS `cuGraphInstantiateWithFlags`, NOT
       `cuGraphInstantiate`. The base symbol is the FIVE-argument
       (exec, graph, phErrorNode, logBuffer, bufferSize) form, and calling it
       with three left logBuffer + bufferSize as garbage registers that the
       driver writes its error log into. Prime suspect for the original
       NVIDIA crash, which struck at the first graph instantiation. */
    typedef CUresult (*fn_t)(void**, void*, unsigned long long);
    fn_t f = (fn_t)cuda_fn("cuGraphInstantiateWithFlags");
    if (f) return f(exec_out, graph, 0ULL);

    /* Fallback: the genuine 5-arg form, called with all five. */
    typedef CUresult (*fn5_t)(void**, void*, void**, char*, size_t);
    fn5_t f5 = (fn5_t)cuda_fn("cuGraphInstantiate_v2");
    if (!f5) f5 = (fn5_t)cuda_fn("cuGraphInstantiate");
    if (!f5) return 1;
    fprintf(stderr, "[intercept] cuGraphInstantiateWithFlags unavailable; "
            "using the 5-arg cuGraphInstantiate\n");
    void *err_node = NULL;
    return f5(exec_out, graph, &err_node, NULL, 0);
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

/* Constructor: print banner on load.
   MOJO_RL_INTERCEPT_LOG=1 turns on per-launch tracing from the very first
   launch — `intercept_set_logging` can only be called from Mojo, i.e. too
   late to see anything that happens during startup or inside a capture that
   crashes before Mojo regains control. */
__attribute__((constructor))
static void on_load(void) {
    const char *e = getenv("MOJO_RL_INTERCEPT_LOG");
    if (e && e[0] != '0') g_logging = 1;
    fprintf(stderr, "[intercept] CUDA interceptor loaded (dlsym hooking mode)%s\n",
            g_logging ? " [launch logging ON]" : "");
}
