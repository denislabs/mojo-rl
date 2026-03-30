/*
 * CUDA kernel launch interceptor via dlsym hooking.
 *
 * Mojo loads libcuda.so via dlopen/dlsym, which bypasses LD_PRELOAD.
 * We intercept dlsym itself to return our wrappers when Mojo requests
 * cuLaunchKernel or cuLaunchKernelEx.
 *
 * Build:
 *   gcc -shared -fPIC -o libcuda_intercept.so cuda_intercept.c -ldl
 *
 * Use:
 *   LD_PRELOAD=$PWD/benchmarks/libcuda_intercept.so pixi run -e nvidia \
 *     bash -c 'mojo run -I . benchmarks/_test_cuda_ffi.mojo' 2>&1
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
static int g_logging = 1;
static int g_launch_count = 0;

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

void intercept_start_recording(void) {
    g_num_records = 0;
    g_recording = 1;
    fprintf(stderr, "[intercept] Recording started\n");
}

void intercept_stop_recording(void) {
    g_recording = 0;
    fprintf(stderr, "[intercept] Recording stopped. %d launches captured.\n", g_num_records);
}

int intercept_get_num_records(void) {
    return g_num_records;
}

void* intercept_get_records(void) {
    return (void*)g_records;
}

void intercept_set_logging(int enabled) {
    g_logging = enabled;
}

int intercept_get_launch_count(void) {
    return g_launch_count;
}

void intercept_print_summary(void) {
    fprintf(stderr, "\n[intercept] === Launch Summary ===\n");
    fprintf(stderr, "[intercept] Total intercepted launches: %d\n", g_launch_count);
    fprintf(stderr, "[intercept] Recorded launches: %d\n", g_num_records);

    /* Report unique streams */
    CUstream unique_streams[64];
    int num_unique = 0;
    for (int i = 0; i < g_num_records; i++) {
        int found = 0;
        for (int j = 0; j < num_unique; j++) {
            if (unique_streams[j] == g_records[i].stream) { found = 1; break; }
        }
        if (!found && num_unique < 64) {
            unique_streams[num_unique++] = g_records[i].stream;
        }
    }
    fprintf(stderr, "[intercept] Unique streams used: %d\n", num_unique);
    for (int i = 0; i < num_unique; i++) {
        fprintf(stderr, "[intercept]   stream %p", unique_streams[i]);
        if (unique_streams[i] == (CUstream)0) fprintf(stderr, " (NULL/default)");
        if (unique_streams[i] == (CUstream)1) fprintf(stderr, " (CU_STREAM_LEGACY)");
        if (unique_streams[i] == (CUstream)2) fprintf(stderr, " (CU_STREAM_PER_THREAD)");
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "[intercept] ==================\n\n");
}

/* Constructor: print banner on load */
__attribute__((constructor))
static void on_load(void) {
    fprintf(stderr, "[intercept] CUDA interceptor loaded (dlsym hooking mode)\n");
}
