/* mrl_http.c — libcurl behind a NON-VARIADIC C API.
 *
 * Why there is C here at all
 * --------------------------
 * `curl_easy_setopt(CURL*, CURLoption, ...)` is C-variadic, and Mojo's
 * `external_call` emits a FIXED prototype: the value lands in a register
 * while the callee's `va_arg` reads the stack (Apple arm64 passes variadic
 * arguments on the stack). That is the same trap `mojo_rl/io/serial/` records
 * for `ioctl` — a silent EFAULT-class failure at runtime, not a compile
 * error. Every entry point below takes a fixed argument list, so the Mojo
 * side never touches a variadic symbol.
 *
 * Having crossed the boundary once, the shim also owns the three things that
 * are genuinely easier in C than over FFI: the write callback, the growable
 * response buffer, and the progress line.
 *
 * Scope: HTTP/1.1+ CLIENT only — GET / POST / PUT / DELETE / HEAD, TLS,
 * redirects, `Range` resume, file upload. No server, no async. That is the
 * whole of what `mojo_rl` asks the network for.
 *
 * Built by scripts/build_http.sh into libmrl_http.dylib / .so, which
 * `mojo_rl/io/http.mojo` dlopens through `_get_dylib_function` — the path
 * `render/imgui`, `vision/opencv` and `io/serial` all use. Not tracked in git.
 *
 * Conventions, mirrored on the Mojo side:
 *   - a handle is an opaque pointer, passed to Mojo as an address (Int);
 *     0 means "allocation failed".
 *   - `mrl_http_perform` returns 0 on a completed transfer and a CURLcode
 *     otherwise. THE HTTP STATUS IS NOT AN ERROR HERE: a clean 404 returns 0
 *     and `mrl_http_status()` says 404. The caller decides what is a failure,
 *     because `hf.mojo` and `remote.mojo` disagree about that.
 */

#define _GNU_SOURCE

#include <curl/curl.h>
#include <zstd.h>

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define MRL_DEFAULT_MAX_BODY (64LL << 20)
/* How much lands in the page cache before we push it out again. The Python
 * downloader this replaces did the same thing every 32 chunks of 8 MiB, and
 * the comment it carried is worth keeping: a multi-GB write with no fsync +
 * DONTNEED evicts everything else on the machine. */
#define MRL_SYNC_EVERY (256LL << 20)

/* ── error slots ────────────────────────────────────────────────────────── */

struct mrl_http {
    CURL *curl;
    struct curl_slist *headers;

    /* response body, when no output file is set */
    unsigned char *buf;
    size_t buf_len;
    size_t buf_cap;
    long long max_body;

    /* output file, opened LAZILY in the write callback — see wr_cb */
    char *out_path;
    FILE *fp;
    long long since_sync;

    /* upload */
    FILE *up_fp;
    long long up_size;

    /* request body held in memory (POST/PUT of JSON) */
    unsigned char *req_body;
    long req_body_len;

    long long resume_from;
    long status;
    int first_write_done;

    /* failure flags the write callback cannot report any other way: aborting
     * a transfer from a write callback yields a bare CURLE_WRITE_ERROR, which
     * would otherwise be indistinguishable from a full disk. */
    int range_ignored;
    int open_failed;
    int write_failed;
    int too_big;

    /* progress */
    int progress;
    int progress_done;
    int last_decile;   /* non-TTY throttling; see xfer_cb */
    char label[64];
    double t0;
    double t_last;

    /* zstd: a DStream that OUTLIVES a single perform, so a dropped
     * connection can be resumed at the compressed offset without restarting
     * a 13 GB transfer. See mrl_http_zstd_enable. */
    void *zds;
    unsigned char *z_out;
    size_t z_out_cap;
    long long z_in_total;
    int z_failed;

    char *cainfo;
    int accept_encoding;
    long timeout_ms;
    long connect_timeout_ms;
    long low_speed_limit;
    long low_speed_time;
    int follow;
    long max_redirs;

    char err[CURL_ERROR_SIZE];
};

typedef struct mrl_http mrl_http;

/* ── one-time global init ───────────────────────────────────────────────── */

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static CURLcode g_init_rc = CURLE_OK;

static void do_global_init(void) {
    g_init_rc = curl_global_init(CURL_GLOBAL_DEFAULT);
}

int mrl_http_init(void) {
    pthread_once(&g_once, do_global_init);
    return (int)g_init_rc;
}

const char *mrl_http_curl_version(void) { return curl_version(); }

/* ── helpers ────────────────────────────────────────────────────────────── */

static double now_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static void free_str(char **p) {
    if (*p) {
        free(*p);
        *p = NULL;
    }
}

static void drop_files(mrl_http *h) {
    if (h->fp) {
        fclose(h->fp);
        h->fp = NULL;
    }
    if (h->up_fp) {
        fclose(h->up_fp);
        h->up_fp = NULL;
    }
}

/* ── lifecycle ──────────────────────────────────────────────────────────── */

mrl_http *mrl_http_new(void) {
    if (mrl_http_init() != 0) return NULL;
    mrl_http *h = (mrl_http *)calloc(1, sizeof(mrl_http));
    if (!h) return NULL;
    h->curl = curl_easy_init();
    if (!h->curl) {
        free(h);
        return NULL;
    }
    h->max_body = MRL_DEFAULT_MAX_BODY;
    h->accept_encoding = 1;
    h->follow = 1;
    h->max_redirs = 20;
    h->connect_timeout_ms = 30000;
    return h;
}

void mrl_http_free(mrl_http *h) {
    if (!h) return;
    drop_files(h);
    if (h->zds) ZSTD_freeDStream((ZSTD_DStream *)h->zds);
    free(h->z_out);
    if (h->headers) curl_slist_free_all(h->headers);
    if (h->curl) curl_easy_cleanup(h->curl);
    free(h->buf);
    free(h->req_body);
    free(h->out_path);
    free(h->cainfo);
    free(h);
}

/* Clear per-REQUEST state. The CURL handle survives, and with it the live
 * connection, the TLS session cache and the DNS cache — which is the whole
 * reason `RemoteLogger` keeps one client rather than making one per flush. */
void mrl_http_reset(mrl_http *h) {
    if (!h) return;
    drop_files(h);
    if (h->headers) {
        curl_slist_free_all(h->headers);
        h->headers = NULL;
    }
    free_str(&h->out_path);
    free(h->req_body);
    h->req_body = NULL;
    h->req_body_len = 0;
    h->buf_len = 0;
    h->since_sync = 0;
    h->resume_from = 0;
    h->up_size = 0;
    h->status = 0;
    h->first_write_done = 0;
    h->range_ignored = 0;
    h->open_failed = 0;
    h->write_failed = 0;
    h->too_big = 0;
    h->progress = 0;
    h->last_decile = -1;
    h->accept_encoding = 1;
    h->label[0] = 0;
    h->err[0] = 0;
}

/* ── configuration (all fixed-arity) ────────────────────────────────────── */

int mrl_http_add_header(mrl_http *h, const char *line) {
    if (!h || !line) return -1;
    struct curl_slist *n = curl_slist_append(h->headers, line);
    if (!n) return -1;
    h->headers = n;
    return 0;
}

void mrl_http_set_timeout_ms(mrl_http *h, long total_ms, long connect_ms) {
    if (!h) return;
    h->timeout_ms = total_ms;
    h->connect_timeout_ms = connect_ms;
}

/* The right stall guard for a multi-GB download: a total timeout would have
 * to be sized for the slowest acceptable link and then still kills a healthy
 * transfer of a bigger file. */
void mrl_http_set_low_speed(mrl_http *h, long bytes_per_s, long seconds) {
    if (!h) return;
    h->low_speed_limit = bytes_per_s;
    h->low_speed_time = seconds;
}

void mrl_http_set_follow(mrl_http *h, int on, long max_redirs) {
    if (!h) return;
    h->follow = on;
    h->max_redirs = max_redirs;
}

void mrl_http_set_resume_from(mrl_http *h, long long off) {
    if (!h) return;
    h->resume_from = off;
}

int mrl_http_set_out_file(mrl_http *h, const char *path) {
    if (!h) return -1;
    free_str(&h->out_path);
    if (!path || !*path) return 0;
    h->out_path = strdup(path);
    return h->out_path ? 0 : -1;
}

int mrl_http_set_body(mrl_http *h, const unsigned char *data, long len) {
    if (!h) return -1;
    free(h->req_body);
    h->req_body = NULL;
    h->req_body_len = 0;
    if (!data || len <= 0) return 0;
    h->req_body = (unsigned char *)malloc((size_t)len);
    if (!h->req_body) return -1;
    memcpy(h->req_body, data, (size_t)len);
    h->req_body_len = len;
    return 0;
}

int mrl_http_set_upload_file(mrl_http *h, const char *path) {
    if (!h || !path) return -1;
    if (h->up_fp) fclose(h->up_fp);
    h->up_fp = fopen(path, "rb");
    if (!h->up_fp) return -1;
    if (fseeko(h->up_fp, 0, SEEK_END) != 0) return -1;
    h->up_size = (long long)ftello(h->up_fp);
    rewind(h->up_fp);
    return 0;
}

void mrl_http_set_progress(mrl_http *h, int on, const char *label) {
    if (!h) return;
    h->progress = on;
    h->label[0] = 0;
    if (label) {
        strncpy(h->label, label, sizeof(h->label) - 1);
        h->label[sizeof(h->label) - 1] = 0;
    }
}

int mrl_http_set_cainfo(mrl_http *h, const char *path) {
    if (!h) return -1;
    free_str(&h->cainfo);
    if (!path || !*path) return 0;
    h->cainfo = strdup(path);
    return h->cainfo ? 0 : -1;
}

void mrl_http_set_max_body(mrl_http *h, long long bytes) {
    if (h) h->max_body = bytes;
}

/* Decode the response body through zstd on its way to the output file.
 *
 * ⚠ THE DSTREAM SURVIVES `mrl_http_reset`, AND THAT IS THE WHOLE POINT. The
 * one caller (`nn/datasets/lewm_pusht.mojo`) streams a 13 GB `.zst` into a
 * 47 GB `.h5` and must never land the compressed file on disk. A dropped
 * connection therefore has to resume the DOWNLOAD at a compressed byte offset
 * while the DECOMPRESSOR carries on from exactly where it stopped — which is
 * what the Python version this replaces did by keeping one `decompressobj`
 * across reconnects. Freeing the stream per request would make a retry
 * restart 13 GB of transfer.
 *
 * `mrl_http_zstd_read` is the compressed offset to resume from: bytes FULLY
 * fed to the decompressor, so it can never point into a half-consumed frame.
 *
 * Pass on=0 to tear the stream down (also done by mrl_http_free).
 */
int mrl_http_zstd_enable(mrl_http *h, int on) {
    if (!h) return -1;
    if (!on) {
        if (h->zds) {
            ZSTD_freeDStream((ZSTD_DStream *)h->zds);
            h->zds = NULL;
        }
        free(h->z_out);
        h->z_out = NULL;
        h->z_out_cap = 0;
        h->z_in_total = 0;
        h->z_failed = 0;
        return 0;
    }
    if (h->zds) return 0; /* already streaming: keep the state */
    h->zds = ZSTD_createDStream();
    if (!h->zds) return -1;
    if (ZSTD_isError(ZSTD_initDStream((ZSTD_DStream *)h->zds))) {
        ZSTD_freeDStream((ZSTD_DStream *)h->zds);
        h->zds = NULL;
        return -1;
    }
    h->z_out_cap = ZSTD_DStreamOutSize();
    h->z_out = (unsigned char *)malloc(h->z_out_cap);
    if (!h->z_out) {
        ZSTD_freeDStream((ZSTD_DStream *)h->zds);
        h->zds = NULL;
        return -1;
    }
    h->z_in_total = 0;
    h->z_failed = 0;
    return 0;
}

long long mrl_http_zstd_read(mrl_http *h) { return h ? h->z_in_total : 0; }

/* ⚠ TRANSPARENT DECOMPRESSION AND A BYTE COUNT ARE INCOMPATIBLE, and both
 * callers of the download path count bytes: `hf.mojo` checks the size the Hub
 * listing gave, `fetch_to_cache` checks the catalog's. With
 * `Accept-Encoding` on, `Content-Length` describes the COMPRESSED stream
 * while the file on disk holds the decompressed one — the probe measured
 * 2017 against 6034 on a README. Worse, a `Range` resume would restart a
 * gzip stream at a byte offset and append its output to a decompressed
 * prefix, which is silent corruption rather than an error.
 *
 * So this is off for any transfer with an output file, and `perform` enforces
 * that rather than trusting the call site. It stays on for JSON responses
 * read into memory, where it is a free win and nothing counts bytes. */
void mrl_http_set_accept_encoding(mrl_http *h, int on) {
    if (h) h->accept_encoding = on;
}

/* ── callbacks ──────────────────────────────────────────────────────────── */

/* ⚠ THE STATUS LINE IS WHAT DELIMITS A RESPONSE, NOT THE TRANSFER.
 * With FOLLOWLOCATION on, libcurl hands us the body of every 3xx in the chain
 * as well as the final one. Resetting on each `HTTP/` line is what keeps a
 * redirect's body out of the answer — and what makes "first write of THIS
 * response" a meaningful moment for the lazy file open below. */
static size_t hdr_cb(char *b, size_t sz, size_t n, void *ud) {
    mrl_http *h = (mrl_http *)ud;
    size_t len = sz * n;
    if (len >= 5 && strncmp(b, "HTTP/", 5) == 0) {
        h->first_write_done = 0;
        /* ⚠ A REDIRECT IS A COMPLETE RESPONSE, and its tiny body finishes
         * instantly — which printed a "100%" bar before the real transfer had
         * sent a byte. Every status line restarts the progress state, and the
         * bar is suppressed for a 3xx entirely. */
        h->progress_done = 0;
        h->last_decile = -1;
        h->t_last = 0.0;
        if (!h->fp) h->buf_len = 0;
        char *sp = (char *)memchr(b, ' ', len);
        if (sp) h->status = strtol(sp + 1, NULL, 10);
    }
    return len;
}

static size_t wr_cb(char *ptr, size_t sz, size_t n, void *ud) {
    mrl_http *h = (mrl_http *)ud;
    size_t len = sz * n;

    if (!h->first_write_done) {
        h->first_write_done = 1;
        /* An output file is opened ONLY for a successful final response.
         * A 404's body must not land on top of a half-downloaded .part —
         * it goes to the memory buffer instead, where it becomes the error
         * message the caller prints. */
        if (h->out_path && h->status >= 200 && h->status < 300) {
            /* ⚠ A SERVER THAT IGNORES `Range` ANSWERS 200 AND SENDS THE WHOLE
             * FILE. Appending that onto a partial download is silent
             * corruption — the exact hazard the Python downloader guarded.
             * Refuse, and let the caller restart from zero. */
            if (h->resume_from > 0 && h->status != 206) {
                h->range_ignored = 1;
                return 0;
            }
            h->fp = fopen(h->out_path, h->resume_from > 0 ? "ab" : "wb");
            if (!h->fp) {
                h->open_failed = 1;
                return 0;
            }
#if defined(__APPLE__)
            (void)fcntl(fileno(h->fp), F_NOCACHE, 1);
#endif
        }
    }

    if (h->fp && h->zds) {
        /* Compressed in, decompressed out — neither side ever fully resident. */
        ZSTD_inBuffer in;
        in.src = ptr;
        in.size = len;
        in.pos = 0;
        while (in.pos < in.size) {
            ZSTD_outBuffer out;
            out.dst = h->z_out;
            out.size = h->z_out_cap;
            out.pos = 0;
            size_t rc2 = ZSTD_decompressStream((ZSTD_DStream *)h->zds, &out, &in);
            if (ZSTD_isError(rc2)) {
                h->z_failed = 1;
                return 0;
            }
            if (out.pos && fwrite(h->z_out, 1, out.pos, h->fp) != out.pos) {
                h->write_failed = 1;
                return 0;
            }
            h->since_sync += (long long)out.pos;
        }
        /* Only now: the whole chunk is inside the decompressor, so this offset
         * can be handed to `Range` on a retry without losing a frame. */
        h->z_in_total += (long long)len;
        if (h->since_sync >= MRL_SYNC_EVERY) {
            fflush(h->fp);
            int fd = fileno(h->fp);
            fsync(fd);
#if defined(__linux__)
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
            h->since_sync = 0;
        }
        return len;
    }

    if (h->fp) {
        if (fwrite(ptr, 1, len, h->fp) != len) {
            h->write_failed = 1;
            return 0;
        }
        h->since_sync += (long long)len;
        if (h->since_sync >= MRL_SYNC_EVERY) {
            fflush(h->fp);
            int fd = fileno(h->fp);
            fsync(fd);
#if defined(__linux__)
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
            h->since_sync = 0;
        }
        return len;
    }

    if ((long long)(h->buf_len + len) > h->max_body) {
        h->too_big = 1;
        return 0;
    }
    if (h->buf_len + len > h->buf_cap) {
        size_t cap = h->buf_cap ? h->buf_cap * 2 : 8192;
        while (cap < h->buf_len + len) cap *= 2;
        unsigned char *nb = (unsigned char *)realloc(h->buf, cap);
        if (!nb) {
            h->write_failed = 1;
            return 0;
        }
        h->buf = nb;
        h->buf_cap = cap;
    }
    memcpy(h->buf + h->buf_len, ptr, len);
    h->buf_len += len;
    return len;
}

/* Is stderr a terminal? Cached: isatty() is a syscall and this is called on
 * every progress tick.
 *
 * A `\r`-redrawn bar is right on a terminal and wrong everywhere else: piped
 * into a file or a CI log there is no cursor to move, so every tick appends
 * and a 200 MB upload becomes one enormous unreadable line. Off a terminal we
 * emit a plain line every 10% instead. */
static int stderr_is_tty(void) {
    static int cached = -1;
    if (cached < 0) cached = isatty(fileno(stderr)) ? 1 : 0;
    return cached;
}

static int xfer_cb(void *ud, curl_off_t dltotal, curl_off_t dlnow,
                   curl_off_t ultotal, curl_off_t ulnow) {
    mrl_http *h = (mrl_http *)ud;
    if (!h->progress) return 0;
    if (h->status >= 300 && h->status < 400) return 0; /* a redirect hop */

    int up = (ultotal > 0 || ulnow > 0);
    curl_off_t now = up ? ulnow : dlnow;
    curl_off_t total = up ? ultotal : dltotal;
    /* Range resume: libcurl counts from the resume point, the user counts
     * from the start of the file. */
    long long done = (long long)now + h->resume_from;
    long long full = total > 0 ? (long long)total + h->resume_from : 0;

    if (now == 0 && total == 0) return 0; /* nothing has moved yet */
    double t = now_s();
    /* The throttle is bypassed only for the FINAL tick, so the bar always
     * lands on 100%. `now == total` is also true before anything moves, which
     * is what made the first probe print thirteen empty bars. */
    int final = (total > 0 && now == total);
    if (final && h->progress_done) return 0; /* libcurl ticks several times at the end */
    int tty = stderr_is_tty();
    if (!tty) {
        /* Every 10% of the transfer, not every 0.2 s. `progress_done` also
         * gates here: libcurl ticks several times at the end, and without
         * this the 100% line prints twice. */
        if (h->progress_done) return 0;
        int pct10 = full > 0 ? (int)(done * 10 / full) : -1;
        if (!final && (pct10 < 0 || pct10 == h->last_decile)) return 0;
        h->last_decile = pct10;
    } else if (t - h->t_last < 0.2 && !final) {
        return 0;
    }
    if (final) h->progress_done = 1;
    h->t_last = t;
    double el = t - h->t0;
    double mbs = el > 0 ? (double)now / 1e6 / el : 0.0;

    if (full > 0) {
        int pct = (int)(done * 100 / full);
        int filled = (int)(done * 30 / full);
        char bar[31];
        for (int i = 0; i < 30; i++) bar[i] = i < filled ? '#' : '.';
        bar[30] = 0;
        int eta = mbs > 0 ? (int)((double)(full - done) / 1e6 / mbs) : 0;
        if (tty) {
            fprintf(stderr, "\r  [%s] [%s] %d%% %.0f MB/s ETA %ds   ", h->label,
                    bar, pct, mbs, eta);
        } else {
            fprintf(stderr, "  [%s] %d%% of %lld MB, %.0f MB/s\n", h->label,
                    pct, full / 1000000, mbs);
        }
    } else if (tty) {
        fprintf(stderr, "\r  [%s] %lld MB %.0f MB/s   ", h->label,
                done / 1000000, mbs);
    } else {
        fprintf(stderr, "  [%s] %lld MB, %.0f MB/s\n", h->label,
                done / 1000000, mbs);
    }
    fflush(stderr);
    return 0;
}

/* ── perform ────────────────────────────────────────────────────────────── */

int mrl_http_perform(mrl_http *h, const char *method, const char *url) {
    if (!h || !method || !url) return -1;

    CURL *c = h->curl;
    curl_easy_reset(c); /* options only — keeps connections, TLS + DNS caches */

    h->err[0] = 0;
    h->buf_len = 0;
    h->since_sync = 0;
    h->status = 0;
    h->first_write_done = 0;
    h->range_ignored = 0;
    h->open_failed = 0;
    h->write_failed = 0;
    h->too_big = 0;
    h->t0 = now_s();
    h->t_last = 0.0;
    h->progress_done = 0;

    curl_easy_setopt(c, CURLOPT_URL, url);
    curl_easy_setopt(c, CURLOPT_ERRORBUFFER, h->err);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, wr_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, h);
    curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, hdr_cb);
    curl_easy_setopt(c, CURLOPT_HEADERDATA, h);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, (long)(h->follow ? 1 : 0));
    curl_easy_setopt(c, CURLOPT_MAXREDIRS, h->max_redirs);
    /* See mrl_http_set_accept_encoding: never for a file, no matter what the
     * caller asked for. */
    if (h->accept_encoding && !h->out_path)
        curl_easy_setopt(c, CURLOPT_ACCEPT_ENCODING, "");
    if (h->headers) curl_easy_setopt(c, CURLOPT_HTTPHEADER, h->headers);
    if (h->cainfo) curl_easy_setopt(c, CURLOPT_CAINFO, h->cainfo);
    if (h->timeout_ms > 0) curl_easy_setopt(c, CURLOPT_TIMEOUT_MS, h->timeout_ms);
    if (h->connect_timeout_ms > 0)
        curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS, h->connect_timeout_ms);
    if (h->low_speed_time > 0) {
        curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, h->low_speed_limit);
        curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, h->low_speed_time);
    }
    if (h->resume_from > 0)
        curl_easy_setopt(c, CURLOPT_RESUME_FROM_LARGE,
                         (curl_off_t)h->resume_from);
    if (h->progress) {
        curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, xfer_cb);
        curl_easy_setopt(c, CURLOPT_XFERINFODATA, h);
    }

    if (strcmp(method, "HEAD") == 0) {
        curl_easy_setopt(c, CURLOPT_NOBODY, 1L);
    } else if (h->up_fp) {
        /* PUT (or any method) streaming a file: libcurl's default read
         * callback is fread, so READDATA takes the FILE* directly. */
        curl_easy_setopt(c, CURLOPT_UPLOAD, 1L);
        curl_easy_setopt(c, CURLOPT_READDATA, h->up_fp);
        curl_easy_setopt(c, CURLOPT_INFILESIZE_LARGE, (curl_off_t)h->up_size);
        if (strcmp(method, "PUT") != 0)
            curl_easy_setopt(c, CURLOPT_CUSTOMREQUEST, method);
    } else if (strcmp(method, "POST") == 0) {
        curl_easy_setopt(c, CURLOPT_POST, 1L);
        curl_easy_setopt(c, CURLOPT_POSTFIELDS,
                         h->req_body ? (const char *)h->req_body : "");
        curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE, h->req_body_len);
    } else if (h->req_body_len > 0) {
        curl_easy_setopt(c, CURLOPT_CUSTOMREQUEST, method);
        curl_easy_setopt(c, CURLOPT_POSTFIELDS, (const char *)h->req_body);
        curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE, h->req_body_len);
    } else if (strcmp(method, "GET") != 0) {
        curl_easy_setopt(c, CURLOPT_CUSTOMREQUEST, method);
    }

    CURLcode rc = curl_easy_perform(c);

    if (h->fp) {
        fflush(h->fp);
        fsync(fileno(h->fp));
        fclose(h->fp);
        h->fp = NULL;
    }
    if (h->up_fp) {
        fclose(h->up_fp);
        h->up_fp = NULL;
        h->up_size = 0;
    }
    if (h->progress && stderr_is_tty()) fprintf(stderr, "\n");

    if (h->status == 0) {
        long code = 0;
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
        h->status = code;
    }

    /* ⚠ TWO PATHS REACH ONE CONCLUSION. libcurl checks for `Content-Range`
     * in the HEADER phase and fails with CURLE_RANGE_ERROR before the write
     * callback ever runs, so the guard in wr_cb only fires when a server
     * answers 206-without-honouring or streams before the check. Both mean
     * "the resume did not happen"; a caller keying off `range_ignored` to
     * restart from zero must see them as one flag, or the retry loop spins
     * out its budget instead of restarting. Measured against
     * tools/io/mock_http_server.py's /blob-norange. */
    if (rc == CURLE_RANGE_ERROR) h->range_ignored = 1;

    /* A write callback that returned short shows up as CURLE_WRITE_ERROR with
     * an empty error buffer. Name the actual cause. */
    if (rc == CURLE_WRITE_ERROR || h->err[0] == 0) {
        const char *why = NULL;
        if (h->range_ignored) why = "server ignored Range and answered 200";
        else if (h->open_failed) why = "cannot open the output file";
        else if (h->write_failed) why = "write to the output file failed";
        else if (h->too_big) why = "response body exceeds the configured cap";
        else if (h->z_failed) why = "the zstd stream is corrupt";
        if (why) {
            strncpy(h->err, why, sizeof(h->err) - 1);
            h->err[sizeof(h->err) - 1] = 0;
        } else if (h->err[0] == 0 && rc != CURLE_OK) {
            strncpy(h->err, curl_easy_strerror(rc), sizeof(h->err) - 1);
            h->err[sizeof(h->err) - 1] = 0;
        }
    }
    return (int)rc;
}

/* ── results ────────────────────────────────────────────────────────────── */

long mrl_http_status(mrl_http *h) { return h ? h->status : 0; }
int mrl_http_range_ignored(mrl_http *h) { return h ? h->range_ignored : 0; }
const char *mrl_http_error(mrl_http *h) { return h ? h->err : ""; }
long mrl_http_body_len(mrl_http *h) { return h ? (long)h->buf_len : 0; }

long mrl_http_body_copy(mrl_http *h, unsigned char *dst, long cap) {
    if (!h || !dst || cap <= 0) return 0;
    long n = (long)h->buf_len < cap ? (long)h->buf_len : cap;
    memcpy(dst, h->buf, (size_t)n);
    return n;
}

long long mrl_http_downloaded(mrl_http *h) {
    if (!h) return 0;
    curl_off_t n = 0;
    curl_easy_getinfo(h->curl, CURLINFO_SIZE_DOWNLOAD_T, &n);
    return (long long)n;
}

long long mrl_http_content_length(mrl_http *h) {
    if (!h) return -1;
    curl_off_t n = -1;
    curl_easy_getinfo(h->curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &n);
    return (long long)n;
}

/* ── gzip, because some bytes arrive as a .gz URL ───────────────────────── */
/*
 * Content decoding IS an HTTP concern; the only thing different about these
 * bytes is that the compression is named by the file extension instead of a
 * `Content-Encoding` header, so libcurl will not touch it. zlib is already
 * linked (libcurl uses it), so this is two entry points rather than a second
 * shim.
 *
 * ⚠ THE TRAILER'S ISIZE IS MOD 2^32. Correct for every file under 4 GiB,
 * which is every file that reaches this path (the largest is MNIST's 47 MB
 * training set). A bigger one must stream instead, and `mrl_gzip_inflate`
 * reports the shortfall rather than truncating.
 */

#include <zlib.h>

long mrl_gzip_isize(const unsigned char *src, long n) {
    if (!src || n < 18) return -1;                 /* header + trailer */
    if (src[0] != 0x1F || src[1] != 0x8B) return -1; /* not gzip */
    return (long)((unsigned long)src[n - 4] | ((unsigned long)src[n - 3] << 8)
                  | ((unsigned long)src[n - 2] << 16)
                  | ((unsigned long)src[n - 1] << 24));
}

/* Streaming gzip: file in, file out, bounded memory.
 *
 * ⚠ THE WHOLE-BUFFER FORM DOES NOT SCALE. `mrl_gzip_inflate` needs the
 * compressed input AND the decompressed output resident at once — 350 MB for
 * CIFAR-10's 162 MB tarball, and unbounded in general. This one holds two
 * 1 MB buffers regardless of the file.
 *
 * Returns the decompressed byte count, or a negative error:
 *   -1 cannot open the input   -2 cannot open the output
 *   -3 zlib init failed        -4 corrupt or truncated stream
 *   -5 a write failed
 */
long long mrl_gzip_inflate_file(const char *src_path, const char *dst_path) {
    if (!src_path || !dst_path) return -1;
    FILE *fi = fopen(src_path, "rb");
    if (!fi) return -1;
    FILE *fo = fopen(dst_path, "wb");
    if (!fo) {
        fclose(fi);
        return -2;
    }
    z_stream s;
    memset(&s, 0, sizeof(s));
    if (inflateInit2(&s, 15 + 32) != Z_OK) {
        fclose(fi);
        fclose(fo);
        return -3;
    }

    const size_t BUF = 1 << 20;
    unsigned char *in = (unsigned char *)malloc(BUF);
    unsigned char *out = (unsigned char *)malloc(BUF);
    long long total = 0;
    int rc = Z_OK;
    if (!in || !out) {
        total = -3;
        goto done;
    }
    for (;;) {
        size_t n = fread(in, 1, BUF, fi);
        if (n == 0) {
            /* EOF before Z_STREAM_END means the archive is truncated. */
            if (rc != Z_STREAM_END) total = -4;
            break;
        }
        s.next_in = in;
        s.avail_in = (uInt)n;
        while (s.avail_in > 0) {
            s.next_out = out;
            s.avail_out = (uInt)BUF;
            rc = inflate(&s, Z_NO_FLUSH);
            if (rc != Z_OK && rc != Z_STREAM_END) {
                total = -4;
                goto done;
            }
            size_t have = BUF - s.avail_out;
            if (have && fwrite(out, 1, have, fo) != have) {
                total = -5;
                goto done;
            }
            total += (long long)have;
            if (rc == Z_STREAM_END) break;
        }
        if (rc == Z_STREAM_END) break;
    }
done:
    inflateEnd(&s);
    free(in);
    free(out);
    fclose(fi);
    if (fflush(fo) != 0 || fclose(fo) != 0) return -5;
    return total;
}

long mrl_gzip_inflate(const unsigned char *src, long src_len,
                      unsigned char *dst, long dst_cap) {
    if (!src || !dst || src_len <= 0 || dst_cap <= 0) return -1;
    z_stream s;
    memset(&s, 0, sizeof(s));
    /* 15 + 32: the window size, plus "detect gzip or zlib from the header". */
    if (inflateInit2(&s, 15 + 32) != Z_OK) return -2;
    s.next_in = (Bytef *)src;
    s.avail_in = (uInt)src_len;
    s.next_out = (Bytef *)dst;
    s.avail_out = (uInt)dst_cap;
    int rc = inflate(&s, Z_FINISH);
    long out = (long)s.total_out;
    inflateEnd(&s);
    if (rc != Z_STREAM_END) return -3; /* truncated input, or dst too small */
    return out;
}


/* ── deflate + crc32, for writing a PNG ─────────────────────────────────── */
/*
 * The other direction of the same library. `mojo_rl/io/png.mojo` needs a zlib
 * stream and a CRC-32 to emit a PNG, and both are one zlib call — there is no
 * reason to transcribe either.
 */

long mrl_zlib_compress_bound(long n) {
    return (long)compressBound((uLong)n);
}

/* Returns the compressed length, or a negative zlib error. */
long mrl_zlib_compress(const unsigned char *src, long src_len,
                       unsigned char *dst, long dst_cap, int level) {
    if (!src || !dst || src_len < 0 || dst_cap <= 0) return -1;
    uLongf out = (uLongf)dst_cap;
    int rc = compress2(dst, &out, src, (uLong)src_len, level);
    if (rc != Z_OK) return -2;
    return (long)out;
}

/* Rolling CRC-32; pass 0 for the first call, then the previous result. */
unsigned long mrl_crc32(unsigned long crc, const unsigned char *buf, long n) {
    return (unsigned long)crc32((uLong)crc, buf, (uInt)n);
}
