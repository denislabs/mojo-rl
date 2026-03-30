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

    return real_dlsym(handle, symbol);
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

CUresult intercept_stream_begin_capture(void *stream) {
    typedef CUresult (*fn_t)(void*, int);
    fn_t f = (fn_t)cuda_fn("cuStreamBeginCapture");
    if (!f) return 1;
    return f(stream, 0);  /* CU_STREAM_CAPTURE_MODE_GLOBAL */
}

CUresult intercept_stream_end_capture(void *stream, void **graph_out) {
    typedef CUresult (*fn_t)(void*, void**);
    fn_t f = (fn_t)cuda_fn("cuStreamEndCapture");
    if (!f) return 1;
    return f(stream, graph_out);
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
