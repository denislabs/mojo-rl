/*
 * CUDA kernel launch interceptor via LD_PRELOAD.
 *
 * Intercepts cuLaunchKernel and cuLaunchKernelEx to log what stream
 * Mojo's AsyncRT actually dispatches on, and records kernel launches
 * for later CUDA graph construction.
 *
 * Build:
 *   gcc -shared -fPIC -o libcuda_intercept.so cuda_intercept.c -ldl
 *
 * Use:
 *   LD_PRELOAD=./libcuda_intercept.so pixi run -e nvidia mojo run -I . benchmarks/_test_cuda_ffi.mojo
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Opaque CUDA types — we only need the handles as pointers */
typedef void* CUfunction;
typedef void* CUstream;
typedef int CUresult;

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
static int g_recording = 0;  /* controlled from Mojo via exported functions */
static int g_logging = 1;    /* always log by default */

/* ---- Intercept cuLaunchKernel ---- */

typedef CUresult (*cuLaunchKernel_t)(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra);

static cuLaunchKernel_t real_cuLaunchKernel = NULL;

CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra)
{
    if (!real_cuLaunchKernel) {
        real_cuLaunchKernel = (cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
        if (!real_cuLaunchKernel) {
            fprintf(stderr, "[intercept] FATAL: cannot find real cuLaunchKernel\n");
            return 1;
        }
    }

    if (g_logging) {
        fprintf(stderr, "[intercept] cuLaunchKernel: func=%p grid=(%u,%u,%u) "
                "block=(%u,%u,%u) shm=%u stream=%p\n",
                f, gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream);
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

/* ---- Intercept cuLaunchKernelEx (CUDA 12+) ---- */

/* CUlaunchConfig is a struct but we just need to forward it */
typedef CUresult (*cuLaunchKernelEx_t)(
    void *config, CUfunction f, void **kernelParams, void **extra);

static cuLaunchKernelEx_t real_cuLaunchKernelEx = NULL;

CUresult cuLaunchKernelEx(
    void *config, CUfunction f, void **kernelParams, void **extra)
{
    if (!real_cuLaunchKernelEx) {
        real_cuLaunchKernelEx = (cuLaunchKernelEx_t)dlsym(RTLD_NEXT, "cuLaunchKernelEx");
        if (!real_cuLaunchKernelEx) {
            fprintf(stderr, "[intercept] FATAL: cannot find real cuLaunchKernelEx\n");
            return 1;
        }
    }

    if (g_logging) {
        fprintf(stderr, "[intercept] cuLaunchKernelEx: func=%p config=%p\n", f, config);
    }

    return real_cuLaunchKernelEx(config, f, kernelParams, extra);
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

/* Returns pointer to the records array (for Mojo to read) */
void* intercept_get_records(void) {
    return (void*)g_records;
}

void intercept_set_logging(int enabled) {
    g_logging = enabled;
}

void intercept_print_summary(void) {
    fprintf(stderr, "\n[intercept] === Launch Summary ===\n");
    fprintf(stderr, "[intercept] Total launches: %d\n", g_num_records);
    for (int i = 0; i < g_num_records; i++) {
        KernelRecord *r = &g_records[i];
        fprintf(stderr, "[intercept]   #%d: func=%p grid=(%u,%u,%u) block=(%u,%u,%u) stream=%p\n",
                i, r->func, r->gridX, r->gridY, r->gridZ,
                r->blockX, r->blockY, r->blockZ, r->stream);
    }

    /* Report unique streams used */
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
