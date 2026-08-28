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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/calib.hpp>
#include <opencv2/geometry/3d.hpp>

#include <vector>

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
//
// ⚠ A BARE COMMA INSIDE THE BRACES SPLITS THE MACRO ARGUMENT.  The
// preprocessor only protects commas inside PARENTHESES, not inside braces, so
// `cv::Mat rvec, tvec;` in a body reads as two macro arguments and fails with
// "too many arguments provided to function-like macro invocation" pointing at
// a line that looks fine.  Declare one variable per statement in here.
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

static const unsigned int MRL_DET_MAGIC = 0x4d444531u;  // "MDE1"

struct MrlDetector {
    unsigned int magic;
    cv::aruco::ArucoDetector det;
    MrlDetector(const cv::aruco::Dictionary& d)
        : magic(MRL_DET_MAGIC), det(d) {}
};

static MrlDetector* as_detector(void* h) {
    MrlDetector* d = static_cast<MrlDetector*>(h);
    if (d == nullptr || d->magic != MRL_DET_MAGIC) return nullptr;
    return d;
}

static const unsigned int MRL_CHB_MAGIC = 0x4d434231u;  // "MCB1"

// ⚠ THE BOARD AND ITS DETECTOR LIVE IN ONE STRUCT ON PURPOSE.
// `CharucoDetector` is constructed FROM a board and the caller must keep that
// board alive; two handles would let a caller free the board and leave the
// detector reading freed memory, with no diagnostic.
struct MrlCharuco {
    unsigned int magic;
    cv::aruco::CharucoBoard board;
    cv::aruco::CharucoDetector det;
    MrlCharuco(const cv::Size& sz, float sq, float mk,
               const cv::aruco::Dictionary& d)
        : magic(MRL_CHB_MAGIC), board(sz, sq, mk, d), det(board) {}
};

static MrlCharuco* as_charuco(void* h) {
    MrlCharuco* c = static_cast<MrlCharuco*>(h);
    if (c == nullptr || c->magic != MRL_CHB_MAGIC) return nullptr;
    return c;
}

// Rule 1: images arrive as raw bytes and are wrapped in a NON-OWNING header.
// `stride` is the byte distance between rows; pass 0 for "tightly packed".
static cv::Mat wrap_image(const unsigned char* data, int w, int h,
                          int channels, int stride) {
    const int type = (channels == 1) ? CV_8UC1
                   : (channels == 3) ? CV_8UC3
                   : (channels == 4) ? CV_8UC4 : -1;
    if (type < 0 || data == nullptr || w <= 0 || h <= 0) return cv::Mat();
    const size_t step = (stride > 0) ? static_cast<size_t>(stride)
                                     : cv::Mat::AUTO_STEP;
    return cv::Mat(h, w, type, const_cast<unsigned char*>(data), step);
}

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

// ═══════════════════════════════════════════════════════════════════════════
// C — marker detection
// ═══════════════════════════════════════════════════════════════════════════

// `dict_id` is one of OpenCV's `cv::aruco::PredefinedDictionaryType` values —
// DICT_4X4_50 is 0.  Passing the int rather than a string keeps the boundary
// free of allocation and matches what `cv2.aruco.DICT_*` already is.
void* mrl_cv_aruco_create(int dict_id) {
    try {
        return new MrlDetector(cv::aruco::getPredefinedDictionary(dict_id));
    } catch (const cv::Exception& e) {
        mrl_cv_set_error(e.what());
        return nullptr;
    } catch (...) {
        mrl_cv_set_error("ArucoDetector: unknown C++ exception");
        return nullptr;
    }
}

// Rule 2: the caller sizes the outputs and this reports what it filled.
//
// `corners_out` receives 8 floats per marker — four corners, xy, in OpenCV's
// own CLOCKWISE order.  ⚠ THAT ORDER IS PART OF THE CONTRACT, not an
// implementation detail: solvePnP pairs image points with object points
// POSITIONALLY, so a caller that reorders the corners without reordering the
// object points gets a plausible pose that is silently rotated.
//
// ⚠ FINDING NOTHING IS NOT AN ERROR.  An image with no marker returns
// MRL_CV_OK with *n_out == 0, because "no marker in view" is the normal state
// of a camera and an exception would make the caller's loop the wrong shape.
int mrl_cv_aruco_detect(void* h, const unsigned char* img, int w, int height,
                        int channels, int stride, int max_markers,
                        int* ids_out, float* corners_out, int* n_out) {
    MrlDetector* d = as_detector(h);
    if (d == nullptr) { mrl_cv_set_error("bad detector handle"); return MRL_CV_ERR_ARG; }
    if (ids_out == nullptr || corners_out == nullptr || n_out == nullptr ||
        max_markers < 0) return MRL_CV_ERR_ARG;
    MRL_CV_TRY({
        cv::Mat image = wrap_image(img, w, height, channels, stride);
        if (image.empty()) {
            mrl_cv_set_error("aruco_detect: bad image geometry");
            return MRL_CV_ERR_ARG;
        }
        std::vector<std::vector<cv::Point2f> > corners;
        std::vector<int> ids;
        d->det.detectMarkers(image, corners, ids);

        *n_out = static_cast<int>(ids.size());
        if (static_cast<int>(ids.size()) > max_markers) {
            // Reported before the refusal, as in cap_read: a caller that hits
            // this learns how many it should have made room for.
            mrl_cv_set_error("aruco_detect: more markers than max_markers");
            return MRL_CV_ERR_CAPACITY;
        }
        for (size_t m = 0; m < ids.size(); ++m) {
            ids_out[m] = ids[m];
            for (int k = 0; k < 4; ++k) {
                corners_out[m * 8 + k * 2 + 0] = corners[m][k].x;
                corners_out[m * 8 + k * 2 + 1] = corners[m][k].y;
            }
        }
        return MRL_CV_OK;
    })
}

void mrl_cv_aruco_destroy(void* h) {
    MrlDetector* d = as_detector(h);
    if (d == nullptr) return;
    d->magic = 0;
    delete d;
}

// An image from disk, BGR HWC — the detection gate's input, and the only way
// to test detection without a camera.  Same output contract as `cap_read`.
int mrl_cv_imread(const char* path, unsigned char* out, int max_bytes,
                  int* w, int* height, int* channels) {
    if (path == nullptr || out == nullptr || w == nullptr ||
        height == nullptr || channels == nullptr) return MRL_CV_ERR_ARG;
    MRL_CV_TRY({
        cv::Mat img = cv::imread(std::string(path), cv::IMREAD_COLOR);
        if (img.empty()) {
            mrl_cv_set_error((std::string("imread: cannot read ") + path).c_str());
            return MRL_CV_ERR_ARG;
        }
        const int need = img.rows * img.cols * img.channels();
        *w        = img.cols;
        *height   = img.rows;
        *channels = img.channels();
        if (need > max_bytes) {
            mrl_cv_set_error("imread: output buffer too small");
            return MRL_CV_ERR_CAPACITY;
        }
        if (img.isContinuous()) {
            std::memcpy(out, img.data, static_cast<size_t>(need));
        } else {
            const size_t row_bytes =
                static_cast<size_t>(img.cols) * img.channels();
            for (int r = 0; r < img.rows; ++r) {
                std::memcpy(out + static_cast<size_t>(r) * row_bytes,
                            img.ptr(r), row_bytes);
            }
        }
        return MRL_CV_OK;
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// D — pose
// ═══════════════════════════════════════════════════════════════════════════

// `obj_xyz` is 3n doubles, `img_xy` is 2n.  `K` is 9 (row-major 3x3).
// `dist` is `n_dist` doubles; pass n_dist = 0 for an undistorted image.
//
// ⚠ `flags` IS NOT COSMETIC.  SOLVEPNP_IPPE_SQUARE (7) is the four-coplanar-
// corner case a fiducial actually is, and it is what the ArUco path wants;
// SOLVEPNP_ITERATIVE (0) on four coplanar points is a different, worse
// estimator.  Passing the wrong one produces a pose, not an error.
//
// ⚠ AND A SQUARE MARKER HAS A GENUINE TWO-FOLD POSE AMBIGUITY NEAR HEAD-ON.
// Position stays solid; ORIENTATION can flip between two solutions frame to
// frame.  That is the geometry, not a bug here — design against `tvec`, and
// treat `rvec` near head-on with suspicion.
int mrl_cv_solve_pnp(const double* obj_xyz, const double* img_xy, int n,
                     const double* K, const double* dist, int n_dist,
                     int flags, double* rvec_out, double* tvec_out) {
    if (obj_xyz == nullptr || img_xy == nullptr || K == nullptr ||
        rvec_out == nullptr || tvec_out == nullptr || n < 4)
        return MRL_CV_ERR_ARG;
    if (n_dist < 0 || (n_dist > 0 && dist == nullptr)) return MRL_CV_ERR_ARG;
    MRL_CV_TRY({
        // Non-owning headers over the caller's memory: rule 1 again.
        const cv::Mat obj(n, 3, CV_64F, const_cast<double*>(obj_xyz));
        const cv::Mat img(n, 2, CV_64F, const_cast<double*>(img_xy));
        const cv::Mat cam(3, 3, CV_64F, const_cast<double*>(K));
        cv::Mat dc = (n_dist > 0)
            ? cv::Mat(1, n_dist, CV_64F, const_cast<double*>(dist))
            : cv::Mat::zeros(1, 4, CV_64F);

        cv::Mat rvec;
        cv::Mat tvec;
        if (!cv::solvePnP(obj, img, cam, dc, rvec, tvec, false, flags)) {
            mrl_cv_set_error("solvePnP: no solution");
            return MRL_CV_ERR_CV;
        }
        for (int i = 0; i < 3; ++i) {
            rvec_out[i] = rvec.at<double>(i);
            tvec_out[i] = tvec.at<double>(i);
        }
        return MRL_CV_OK;
    })
}

// Rotation vector -> 3x3 row-major rotation matrix.  Needed because a pose is
// only useful once it composes with the robot's frames, and `rvec` is an
// axis-angle, not a matrix.
int mrl_cv_rodrigues(const double* rvec, double* R9) {
    if (rvec == nullptr || R9 == nullptr) return MRL_CV_ERR_ARG;
    MRL_CV_TRY({
        const cv::Mat r(3, 1, CV_64F, const_cast<double*>(rvec));
        cv::Mat R;
        cv::Rodrigues(r, R);
        for (int i = 0; i < 9; ++i) R9[i] = R.at<double>(i / 3, i % 3);
        return MRL_CV_OK;
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// E — calibration
// ═══════════════════════════════════════════════════════════════════════════
//
// ⚠ WRAPPED NOW EVEN THOUGH THE DRIVER IS STILL PYTHON.  The scope document
// argues this at length: covering the whole capability costs two extra entry
// points today and lets the prep script be DELETED later without redrawing the
// boundary.  A shim that stopped at detection would have needed a second
// design pass to reach "no Python in the project".

void* mrl_cv_charuco_create(int squares_x, int squares_y, float square_len,
                            float marker_len, int dict_id) {
    try {
        return new MrlCharuco(cv::Size(squares_x, squares_y),
                              square_len, marker_len,
                              cv::aruco::getPredefinedDictionary(dict_id));
    } catch (const cv::Exception& e) {
        mrl_cv_set_error(e.what());
        return nullptr;
    } catch (...) {
        mrl_cv_set_error("CharucoBoard: unknown C++ exception");
        return nullptr;
    }
}

// The board's chessboard corners in BOARD coordinates (metres, Z = 0), indexed
// by charuco id — 3 floats each.
//
// ⚠ THIS IS NOT A CONVENIENCE.  `calibrateCamera` needs the 3D point for every
// detected corner, and the caller cannot compute it: the ChArUco corner layout
// CHANGED in OpenCV 4.6 for even row counts (see `setLegacyPattern`).  Deriving
// it in Mojo from squares_x/squares_y would be a second implementation of a
// convention that has already moved once, and it would be silently wrong on
// exactly one board shape.  Ask the board.
int mrl_cv_charuco_board_corners(void* h, float* xyz_out, int max_corners,
                                 int* n_out) {
    MrlCharuco* c = as_charuco(h);
    if (c == nullptr) { mrl_cv_set_error("bad charuco handle"); return MRL_CV_ERR_ARG; }
    if (xyz_out == nullptr || n_out == nullptr) return MRL_CV_ERR_ARG;
    MRL_CV_TRY({
        const std::vector<cv::Point3f> pts = c->board.getChessboardCorners();
        *n_out = static_cast<int>(pts.size());
        if (static_cast<int>(pts.size()) > max_corners) {
            mrl_cv_set_error("charuco_board_corners: buffer too small");
            return MRL_CV_ERR_CAPACITY;
        }
        for (size_t i = 0; i < pts.size(); ++i) {
            xyz_out[i * 3 + 0] = pts[i].x;
            xyz_out[i * 3 + 1] = pts[i].y;
            xyz_out[i * 3 + 2] = pts[i].z;
        }
        return MRL_CV_OK;
    })
}

// ⚠ ONLY VISIBLE CORNERS COME BACK, and `ids_out` says which ones.  A view of
// a partially occluded or off-frame board yields fewer corners than the board
// has, which is normal — pairing them with the board's 3D points POSITIONALLY
// instead of BY ID is a silent mismatch that calibrates to nonsense.
int mrl_cv_charuco_detect(void* h, const unsigned char* img, int w, int height,
                          int channels, int stride, int max_corners,
                          float* corners_xy, int* ids_out, int* n_out) {
    MrlCharuco* c = as_charuco(h);
    if (c == nullptr) { mrl_cv_set_error("bad charuco handle"); return MRL_CV_ERR_ARG; }
    if (corners_xy == nullptr || ids_out == nullptr || n_out == nullptr)
        return MRL_CV_ERR_ARG;
    MRL_CV_TRY({
        cv::Mat image = wrap_image(img, w, height, channels, stride);
        if (image.empty()) {
            mrl_cv_set_error("charuco_detect: bad image geometry");
            return MRL_CV_ERR_ARG;
        }
        cv::Mat corners;
        cv::Mat ids;
        c->det.detectBoard(image, corners, ids);

        const int n = ids.empty() ? 0 : static_cast<int>(ids.total());
        *n_out = n;
        if (n > max_corners) {
            mrl_cv_set_error("charuco_detect: more corners than max_corners");
            return MRL_CV_ERR_CAPACITY;
        }
        for (int i = 0; i < n; ++i) {
            const cv::Point2f p = corners.at<cv::Point2f>(i);
            corners_xy[i * 2 + 0] = p.x;
            corners_xy[i * 2 + 1] = p.y;
            ids_out[i] = ids.at<int>(i);
        }
        return MRL_CV_OK;
    })
}

void mrl_cv_charuco_destroy(void* h) {
    MrlCharuco* c = as_charuco(h);
    if (c == nullptr) return;
    c->magic = 0;
    delete c;
}

// Intrinsics from several views.  `obj_xyz` and `img_xy` are the views'
// correspondences CONCATENATED, and `counts[v]` says how many belong to view v.
//
// ⚠⚠ `n_dist_out` IS AN OUTPUT, AND ASSUMING 5 IS A SILENT TRUNCATION.  The
// distortion vector is 4, 5, 8, 12 or 14 long depending on `flags`; a caller
// that copies a fixed 5 out of a 14-long answer keeps a lens model that is
// wrong in a way nothing reports.  `dist_out` must have room for 14.
//
// ⚠ THIS IS ITERATIVE (Levenberg-Marquardt), so a bit-equality gate against
// Python cv2 also requires the SAME THREAD COUNT — see mrl_cv_set_num_threads.
int mrl_cv_calibrate_camera(const double* obj_xyz, const double* img_xy,
                            const int* counts, int n_views,
                            int img_w, int img_h, int flags,
                            double* K_out, double* dist_out, int* n_dist_out,
                            double* rms_out) {
    if (obj_xyz == nullptr || img_xy == nullptr || counts == nullptr ||
        K_out == nullptr || dist_out == nullptr || n_dist_out == nullptr ||
        rms_out == nullptr || n_views < 1) return MRL_CV_ERR_ARG;
    MRL_CV_TRY({
        std::vector<std::vector<cv::Point3f> > obj;
        std::vector<std::vector<cv::Point2f> > img;
        int off = 0;
        for (int v = 0; v < n_views; ++v) {
            const int n = counts[v];
            if (n < 4) {
                mrl_cv_set_error("calibrate_camera: a view has < 4 points");
                return MRL_CV_ERR_ARG;
            }
            std::vector<cv::Point3f> o;
            std::vector<cv::Point2f> p;
            o.reserve(n);
            p.reserve(n);
            for (int i = 0; i < n; ++i) {
                o.push_back(cv::Point3f(
                    static_cast<float>(obj_xyz[(off + i) * 3 + 0]),
                    static_cast<float>(obj_xyz[(off + i) * 3 + 1]),
                    static_cast<float>(obj_xyz[(off + i) * 3 + 2])));
                p.push_back(cv::Point2f(
                    static_cast<float>(img_xy[(off + i) * 2 + 0]),
                    static_cast<float>(img_xy[(off + i) * 2 + 1])));
            }
            obj.push_back(o);
            img.push_back(p);
            off += n;
        }

        cv::Mat K;
        cv::Mat dist;
        std::vector<cv::Mat> rvecs;
        std::vector<cv::Mat> tvecs;
        const double rms = cv::calibrateCamera(obj, img, cv::Size(img_w, img_h),
                                               K, dist, rvecs, tvecs, flags);
        for (int i = 0; i < 9; ++i) K_out[i] = K.at<double>(i / 3, i % 3);
        const int nd = static_cast<int>(dist.total());
        *n_dist_out = nd;
        for (int i = 0; i < nd; ++i) dist_out[i] = dist.at<double>(i);
        *rms_out = rms;
        return MRL_CV_OK;
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// F — the one piece of linear algebra we would otherwise have to write
// ═══════════════════════════════════════════════════════════════════════════

// ⚠ THIS EXISTS SO THAT KABSCH DOES NOT NEED A JACOBI IMPLEMENTATION.  The
// camera->robot-base extrinsics fit is Kabsch/Umeyama, which needs a 3x3 SVD,
// and `math3d` has no SVD or eigensolver.  `cv::SVD` is already linked, so
// this is one entry point instead of an algorithm plus its own gate.
//
// All three outputs are row-major.  `Vt` is ALREADY TRANSPOSED, as OpenCV
// names it: R = U * Vt is the Kabsch rotation, with the usual determinant
// correction if det < 0.
int mrl_cv_svd_3x3(const double* A9, double* U9, double* S3, double* Vt9) {
    if (A9 == nullptr || U9 == nullptr || S3 == nullptr || Vt9 == nullptr)
        return MRL_CV_ERR_ARG;
    MRL_CV_TRY({
        const cv::Mat A(3, 3, CV_64F, const_cast<double*>(A9));
        cv::Mat w;
        cv::Mat u;
        cv::Mat vt;
        cv::SVD::compute(A, w, u, vt);
        for (int i = 0; i < 9; ++i) {
            U9[i]  = u.at<double>(i / 3, i % 3);
            Vt9[i] = vt.at<double>(i / 3, i % 3);
        }
        for (int i = 0; i < 3; ++i) S3[i] = w.at<double>(i);
        return MRL_CV_OK;
    })
}

}  // extern "C"
