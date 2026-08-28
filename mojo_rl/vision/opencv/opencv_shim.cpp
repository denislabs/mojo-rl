// A flat C API over OpenCV, sized for Mojo FFI.  Groups A (lifecycle) and
// B (capture) of `docs/OPENCV_SHIM_SCOPE.md`.
//
// WHY A SHIM AT ALL.  SDL3 is the precedent that does NOT transfer: it is a C
// library with a stable C ABI, so `external_call` reaches it directly.  OpenCV
// has had no C API since 4.x — it is C++ with mangled names, `cv::Mat`
// refcounting, `std::vector<std::vector<cv::Point2f>>` and exceptions, none of
// which Mojo can express.  The precedent that DOES transfer is Dear ImGui,
// also C++, already shimmed here as `render/imgui/imgui_shim.cpp`; this file
// follows its conventions deliberately.
//
// THE FOUR MARSHALLING RULES.  They are the whole design:
//
//   1. No `cv::Mat` ever crosses.  Images cross as
//      (data, width, height, channels, stride) and are wrapped in a
//      NON-OWNING `cv::Mat` header on this side.
//   2. Ragged outputs are caller-allocated: the caller passes a capacity and
//      the shim writes a count.  Nothing is allocated across the boundary and
//      nothing has to be freed across it.
//   3. C++ objects are opaque handles with a magic word, so a stale or wrong
//      pointer is a status code instead of a jump into freed memory.
//   4. NO EXCEPTION MAY CROSS `extern "C"`.  OpenCV throws `cv::Exception`
//      freely — an unreadable device, an empty frame, a bad matrix size — and
//      an exception escaping this boundary is undefined behaviour, not a
//      debuggable crash.  Every body below is wrapped by MRL_CV_TRY.
//
// ⚠ EVERY `mrl_cv_*` NAME IS AN FFI CONTRACT.  `mojo_rl/vision/opencv/
// __init__.mojo` looks these up by string through dlsym.  Renaming one here
// without renaming it there is a LOAD-TIME ABORT, not a compile error.

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>
#include <cstring>

// ─── status codes ───────────────────────────────────────────────────────────
//
// ⚠ NEGATIVE IS AN ERROR, ZERO IS OK, and POSITIVE IS NEVER RETURNED as a
// status.  Anything a caller wants to know beyond success goes out through a
// pointer, so a status can never be confused with a value.
#define MRL_CV_OK             0
#define MRL_CV_ERR_CV        -1   // cv::Exception
#define MRL_CV_ERR_STD       -2   // std::exception
#define MRL_CV_ERR_UNKNOWN   -3   // something else entirely
#define MRL_CV_ERR_ARG       -4   // null pointer, bad handle, nonsense size
#define MRL_CV_ERR_CAPACITY  -5   // caller's buffer is too small
#define MRL_CV_ERR_NO_FRAME  -6   // read returned nothing: EOF, or a dead device

// ⚠ THREAD_LOCAL, NOT GLOBAL.  A shared error string would be a data race the
// moment two threads call in, and the failure would look like a wrong message
// rather than like a race.
static thread_local std::string g_last_error;

static void mrl_cv_set_error(const char* what) {
    g_last_error.assign(what ? what : "unknown");
}

// The one place an exception is allowed to stop.  Every entry point that can
// throw is written as `MRL_CV_TRY({ ... })`.
#define MRL_CV_TRY(BODY)                                                      \
    try {                                                                     \
        BODY                                                                  \
    } catch (const cv::Exception& e) {                                        \
        mrl_cv_set_error(e.what());                                           \
        return MRL_CV_ERR_CV;                                                 \
    } catch (const std::exception& e) {                                       \
        mrl_cv_set_error(e.what());                                           \
        return MRL_CV_ERR_STD;                                                \
    } catch (...) {                                                           \
        mrl_cv_set_error("unknown C++ exception");                            \
        return MRL_CV_ERR_UNKNOWN;                                            \
    }

// ─── handles ────────────────────────────────────────────────────────────────
//
// Rule 3.  The magic word costs four bytes and turns "the caller passed a
// closed capture" from a segfault into MRL_CV_ERR_ARG.
static const unsigned int MRL_CAP_MAGIC = 0x4d435031u;  // "MCP1"

struct MrlCapture {
    unsigned int   magic;
    cv::VideoCapture cap;
    MrlCapture() : magic(MRL_CAP_MAGIC) {}
};

static MrlCapture* as_capture(void* h) {
    MrlCapture* c = static_cast<MrlCapture*>(h);
    if (c == nullptr || c->magic != MRL_CAP_MAGIC) return nullptr;
    return c;
}

extern "C" {

// ═══════════════════════════════════════════════════════════════════════════
// A — lifecycle
// ═══════════════════════════════════════════════════════════════════════════

// Availability + ABI probe.  A caller that gets 5/0 back knows the dylib
// loaded AND that it is the OpenCV this project was scoped against; OpenCV 5
// moved `solvePnP` and `calibrateCamera` between headers and DELETED
// `estimatePoseSingleMarkers`, `calibrateCameraCharuco` and
// `calibrateHandEye`, so the major version is a real compatibility question.
int mrl_cv_version(int* major, int* minor) {
    if (major == nullptr || minor == nullptr) return MRL_CV_ERR_ARG;
    MRL_CV_TRY({
        *major = CV_VERSION_MAJOR;
        *minor = CV_VERSION_MINOR;
        return MRL_CV_OK;
    })
}

// Never NULL, so a caller can print it unconditionally.
const char* mrl_cv_last_error(void) {
    return g_last_error.empty() ? "" : g_last_error.c_str();
}

// ⚠ NOT A PERFORMANCE KNOB — IT IS A GATE REQUIREMENT.  `calibrateCamera` is
// an iterative LM fit and OpenCV's `parallel_for_` can change reduction order
// with the thread count, so bit-equality against Python cv2 is a claim about
// identical inputs AND identical scheduling.  Both sides call this with 1.
int mrl_cv_set_num_threads(int n) {
    MRL_CV_TRY({
        cv::setNumThreads(n);
        return MRL_CV_OK;
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// B — capture
// ═══════════════════════════════════════════════════════════════════════════
//
// ⚠ THIS GROUP EXISTS TO UNBLOCK SOMETHING ALREADY BLOCKED.  Phase 2's
// closed-loop ACT rollout is gated on "a camera capture path in Mojo, which
// does not exist yet" (docs/SO101_MANIPULATION_PLAN_2026_08_26.md).  Phase 3's
// ArUco perception needs the same four calls.

// Live device.  `width`/`height`/`fps` are REQUESTS: a camera is free to
// ignore them, which is why `mrl_cv_cap_props` reports what was actually
// negotiated instead of echoing back what was asked for.  Pass 0 to leave a
// property alone.
void* mrl_cv_cap_open(int index, int width, int height, double fps) {
    try {
        MrlCapture* c = new MrlCapture();
        if (!c->cap.open(index)) {
            mrl_cv_set_error("VideoCapture: device did not open");
            delete c;
            return nullptr;
        }
        if (width  > 0) c->cap.set(cv::CAP_PROP_FRAME_WIDTH,  width);
        if (height > 0) c->cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        if (fps    > 0) c->cap.set(cv::CAP_PROP_FPS,          fps);
        return c;
    } catch (const cv::Exception& e) {
        mrl_cv_set_error(e.what());
        return nullptr;
    } catch (...) {
        mrl_cv_set_error("VideoCapture: unknown C++ exception");
        return nullptr;
    }
}

// A file, which is what the GATE uses.  A capture path tested only against a
// live camera is a capture path with no gate at all — the frames are never the
// same twice, so there is nothing to compare against.  Decoding a committed
// .mp4 through the same dylib on both sides IS comparable, bit for bit.
void* mrl_cv_cap_open_file(const char* path) {
    if (path == nullptr) {
        mrl_cv_set_error("VideoCapture: null path");
        return nullptr;
    }
    try {
        MrlCapture* c = new MrlCapture();
        if (!c->cap.open(std::string(path))) {
            mrl_cv_set_error((std::string("VideoCapture: cannot open ") + path).c_str());
            delete c;
            return nullptr;
        }
        return c;
    } catch (const cv::Exception& e) {
        mrl_cv_set_error(e.what());
        return nullptr;
    } catch (...) {
        mrl_cv_set_error("VideoCapture: unknown C++ exception");
        return nullptr;
    }
}

// What the source actually gives, as opposed to what was asked for.  A live
// camera that silently substituted 640x480 for a requested 1920x1080 is not an
// error anywhere in OpenCV, and reading 640x480 bytes into a buffer sized for
// 1080p is the kind of defect that shows up as a wrong ANSWER later.
// `frame_count` is 0 for a live device and is only meaningful for a file.
int mrl_cv_cap_props(void* h, int* w, int* height, double* fps,
                     int* frame_count) {
    MrlCapture* c = as_capture(h);
    if (c == nullptr) { mrl_cv_set_error("bad capture handle"); return MRL_CV_ERR_ARG; }
    if (w == nullptr || height == nullptr || fps == nullptr ||
        frame_count == nullptr) return MRL_CV_ERR_ARG;
    MRL_CV_TRY({
        *w           = static_cast<int>(c->cap.get(cv::CAP_PROP_FRAME_WIDTH));
        *height      = static_cast<int>(c->cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        *fps         = c->cap.get(cv::CAP_PROP_FPS);
        double n     = c->cap.get(cv::CAP_PROP_FRAME_COUNT);
        *frame_count = (n > 0.0) ? static_cast<int>(n) : 0;
        return MRL_CV_OK;
    })
}

// One frame, BGR, HWC, contiguous.
//
// ⚠⚠ BGR AND HWC, WHICH IS NOT WHAT THE REST OF THIS PROJECT USES.  The ACT
// store holds images CHW and RGB.  A silent channel swap raises no error
// anywhere and simply produces wrong answers — the exact class of defect this
// tree keeps recording.  The conversion lives in ONE named function on the
// Mojo side with its own gate; this function is deliberately the raw thing
// OpenCV hands back, so the conversion has a fixed point to be gated against.
//
// Rule 2: the caller sizes the buffer and this reports what it filled.
int mrl_cv_cap_read(void* h, unsigned char* out, int max_bytes,
                    int* w, int* height, int* channels) {
    MrlCapture* c = as_capture(h);
    if (c == nullptr) { mrl_cv_set_error("bad capture handle"); return MRL_CV_ERR_ARG; }
    if (out == nullptr || w == nullptr || height == nullptr ||
        channels == nullptr) return MRL_CV_ERR_ARG;
    MRL_CV_TRY({
        cv::Mat frame;
        if (!c->cap.read(frame) || frame.empty()) {
            mrl_cv_set_error("VideoCapture: no frame (end of stream?)");
            return MRL_CV_ERR_NO_FRAME;
        }
        const int need = frame.rows * frame.cols * frame.channels();
        // Reported BEFORE the capacity check, so a caller that gets
        // ERR_CAPACITY learns the size it should have allocated instead of
        // having to guess.
        *w        = frame.cols;
        *height   = frame.rows;
        *channels = frame.channels();
        if (need > max_bytes) {
            mrl_cv_set_error("cap_read: output buffer too small");
            return MRL_CV_ERR_CAPACITY;
        }
        // ⚠ `isContinuous` IS NOT A FORMALITY.  A decoder is free to hand back
        // a Mat with padded rows, and a flat memcpy of a padded Mat copies the
        // padding into the image.  The row loop is the correct copy; the fast
        // path is only taken when it is provably identical.
        //
        // ⚠⚠ AND THE ROW LOOP IS CURRENTLY UNGATED.  Measured: forcing the
        // memcpy branch unconditionally still passes the capture gate, because
        // every frame the committed fixture decodes to IS continuous.  A hit
        // count is not coverage of a branch.  Gating it needs a source that
        // yields a padded Mat (a cropped ROI, or some camera backends), and
        // until one exists this path is reasoned-correct, not measured.
        if (frame.isContinuous()) {
            std::memcpy(out, frame.data, static_cast<size_t>(need));
        } else {
            const size_t row_bytes =
                static_cast<size_t>(frame.cols) * frame.channels();
            for (int r = 0; r < frame.rows; ++r) {
                std::memcpy(out + static_cast<size_t>(r) * row_bytes,
                            frame.ptr(r), row_bytes);
            }
        }
        return MRL_CV_OK;
    })
}

// Idempotent: closing twice is not an error, because the second call sees a
// pointer whose magic no longer matches and does nothing.  A capture left open
// holds a camera device against the whole machine.
void mrl_cv_cap_close(void* h) {
    MrlCapture* c = as_capture(h);
    if (c == nullptr) return;
    try {
        c->cap.release();
    } catch (...) {
        // A destructor path has nowhere to report to, and leaking the handle
        // would be worse than dropping the message.
    }
    c->magic = 0;
    delete c;
}

}  // extern "C"
