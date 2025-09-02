#include "DualCam.h"

#include <QLabel>
#include <QTimer>
#include <QVBoxLayout>
#include <QImage>
#include <QPixmap>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <tuple>
#include <optional>

using namespace std;
using namespace cv;

// Helper to convert environment variable to lower-case string
static inline std::string getenv_lower(const char* key) {
    const char* v = std::getenv(key);
    if (!v) return std::string();
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

// Static: build a list of candidate capture specifications for the given camera
std::vector<std::tuple<std::string, bool, bool>> DualCam::buildCandidates(int index) {
    std::vector<std::tuple<std::string, bool, bool>> cands;
    // Determine CAP_PRIORITY; if it contains "v4l2", we will not push any GStreamer pipelines.
    std::string prioStr = getenv_lower("CAP_PRIORITY");
    bool preferV4L2Only = (!prioStr.empty() && prioStr.find("v4l2") != std::string::npos);

    // Check for explicit pipeline override (gst pipelines still supported) unless V4L2 is explicitly requested
    if (!preferV4L2Only) {
        std::string key = std::string("GST_PIPELINE_CAM") + std::to_string(index);
        const char* val = std::getenv(key.c_str());
        if (val && *val) {
            // If the value starts with "gst:", strip that prefix
            std::string spec(val);
            if (spec.rfind("gst:", 0) == 0) spec = spec.substr(4);
            cands.emplace_back(spec, /*isGst*/true, /*isDevice*/false);
        }
    }

    // Check for explicit device path override (direct V4L2 capture)
    {
        std::string key = std::string("DEV_VIDEO_CAM") + std::to_string(index);
        const char* val = std::getenv(key.c_str());
        if (val && *val) {
            // When a device path is specified, add it as a candidate.  This allows
            // direct V4L2 capture on Linux or path-based capture on other OSes.
            cands.emplace_back(std::string(val), /*isGst*/false, /*isDevice*/true);
        }
    }
    // Check for camera-name override (libcamera), unless V4L2 is explicitly requested
    if (!preferV4L2Only) {
        std::string key = std::string("GST_CAMERA_NAME_CAM") + std::to_string(index);
        const char* val = std::getenv(key.c_str());
        if (val && *val) {
            std::string name(val);
            // Use defaults or environment-specified dimensions
            int w = 640, h = 480, f = 30;
            if (const char* wenv = std::getenv("GST_WIDTH")) {
                int wi = atoi(wenv); if (wi > 0) w = wi;
            }
            if (const char* henv = std::getenv("GST_HEIGHT")) {
                int hi = atoi(henv); if (hi > 0) h = hi;
            }
            if (const char* fpsenv = std::getenv("GST_FPS")) {
                int fi = atoi(fpsenv); if (fi > 0) f = fi;
            }
            std::string pipeline = std::string("libcamerasrc camera-name=") + name +
                " ! video/x-raw,width=" + std::to_string(w) + ",height=" + std::to_string(h) + ",format=YUY2"
                " ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true sync=false";
            cands.emplace_back(pipeline, true, false);
        }
    }
    // Decide priority: prefer libcamera if CAP_PRIORITY contains libcamera/gst/gstreamer;
    // explicitly prefer v4l2 if CAP_PRIORITY contains v4l2
    bool preferLibcamera = false;
    {
        std::string pr = getenv_lower("CAP_PRIORITY");
        if (!pr.empty()) {
            // If v4l2 is requested, force libcamera preference off
            if (pr.find("v4l2") != std::string::npos) {
                preferLibcamera = false;
            } else if (pr.find("libcamera") != std::string::npos || pr.find("gst") != std::string::npos || pr.find("gstreamer") != std::string::npos) {
                preferLibcamera = true;
            }
        }
    }
    // Precompute width/height/fps for camera-id pipelines
    int w = 640, h = 480, f = 30;
    if (const char* wenv = std::getenv("GST_WIDTH")) {
        int wi = atoi(wenv); if (wi > 0) w = wi;
    }
    if (const char* henv = std::getenv("GST_HEIGHT")) {
        int hi = atoi(henv); if (hi > 0) h = hi;
    }
    if (const char* fpsenv = std::getenv("GST_FPS")) {
        int fi = atoi(fpsenv); if (fi > 0) f = fi;
    }
    // If prefer libcamera, push camera-id pipeline first
    if (preferLibcamera) {
        std::string pipe = std::string("libcamerasrc camera-id=") + std::to_string(index) +
            " ! video/x-raw,width=" + std::to_string(w) + ",height=" + std::to_string(h) + ",format=YUY2"
            " ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true sync=false";
        cands.emplace_back(pipe, true, false);
    }
#if defined(__linux__)
    // On Linux, allow V4L2 device and index candidates
    if (!preferLibcamera) {
        // index string (numeric)
        cands.emplace_back(std::to_string(index), false, false);
        // device path
        cands.emplace_back(std::string("/dev/video") + std::to_string(index), false, true);
    } else {
        // even if preferring libcamera, still push index as a fallback
        cands.emplace_back(std::to_string(index), false, false);
    }
#else
    // On non-Linux platforms, only index is meaningful
    cands.emplace_back(std::to_string(index), false, false);
#endif
    // If not preferring libcamera, append camera-id pipeline as ultimate fallback,
    // unless V4L2-only mode was requested.  In V4L2-only mode, do not push any
    // GStreamer/libcamera pipeline here.
    if (!preferLibcamera && !preferV4L2Only) {
        std::string pipe = std::string("libcamerasrc camera-id=") + std::to_string(index) +
            " ! video/x-raw,width=" + std::to_string(w) + ",height=" + std::to_string(h) + ",format=YUY2"
            " ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true sync=false";
        cands.emplace_back(pipe, true, false);
    }
    return cands;
}

// Warm up capture by reading frames for ms milliseconds
void DualCam::warmup(cv::VideoCapture& cap, int ms) {
    using namespace std::chrono;
    auto t0 = steady_clock::now();
    cv::Mat tmp;
    while (true) {
        cap.read(tmp);
        auto dt = duration_cast<milliseconds>(steady_clock::now() - t0);
        if (dt.count() >= ms) break;
        std::this_thread::sleep_for(milliseconds(10));
    }
}

// Attempt to read one frame, with timeout
bool DualCam::readOnce(cv::VideoCapture& cap, cv::Mat& out) {
    using namespace std::chrono;
    auto t0 = steady_clock::now();
    while (true) {
        if (cap.read(out) && !out.empty()) return true;
        auto dt = duration_cast<milliseconds>(steady_clock::now() - t0);
        if (dt.count() > 500) return false;
        std::this_thread::sleep_for(milliseconds(10));
    }
}

// Open first working candidate
std::optional<cv::VideoCapture> DualCam::openWithCandidates(const std::vector<std::tuple<std::string, bool, bool>>& cands,
                                                            int width, int height, double fps) {
    for (const auto& cand : cands) {
        const std::string& spec = std::get<0>(cand);
        bool isGst = std::get<1>(cand);
        bool isDev = std::get<2>(cand);
        cv::VideoCapture cap;
        if (isGst) {
            cap.open(spec, cv::CAP_GSTREAMER);
        } else {
            // Non-GStreamer; decide whether numeric index or path
            bool numeric = !spec.empty();
            for (size_t i = 0; i < spec.size(); ++i) {
                char c = spec[i];
                if (!std::isdigit((unsigned char)c) && !(i == 0 && (c == '+' || c == '-'))) {
                    numeric = false; break;
                }
            }
            if (numeric) {
                int idx = 0;
                try { idx = std::stoi(spec); } catch (...) { numeric = false; }
                if (numeric) {
                    cap.open(idx, cv::CAP_ANY);
                }
            }
            if (!numeric) {
#if defined(__linux__)
                if (isDev) {
                    cap.open(spec, cv::CAP_V4L2);
                } else {
                    // treat unknown string as device path
                    cap.open(spec, cv::CAP_V4L2);
                }
#else
                cap.open(spec, cv::CAP_ANY);
#endif
            }
        }
        if (!cap.isOpened()) continue;
        // Set properties for non-pipeline sources
        if (!isGst) {
            if (width > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
            if (height > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            if (fps > 0.0) cap.set(cv::CAP_PROP_FPS, fps);
        }
        warmup(cap, 600);
        cv::Mat f;
        if (readOnce(cap, f)) {
            return cap;
        }
    }
    return std::nullopt;
}

// Compute a simple variance-of-Laplacian focus metric
double DualCam::focusMeasure(const cv::Mat& bgr) {
    if (bgr.empty()) return 0.0;
    cv::Mat g;
    if (bgr.channels() == 3) {
        cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
    } else if (bgr.channels() == 1) {
        g = bgr;
    } else {
        cv::Mat tmp;
        bgr.convertTo(tmp, CV_8U);
        cv::cvtColor(tmp, g, cv::COLOR_BGR2GRAY);
    }
    cv::Mat lap;
    cv::Laplacian(g, lap, CV_64F);
    cv::Scalar m, s;
    cv::meanStdDev(lap, m, s);
    return s[0] * s[0];
}

// Find last crossing index between two history sequences
double DualCam::findEqualIndex(const std::deque<double>& a, const std::deque<double>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n < 2) return -1.0;
    // Work on the last n samples
    size_t sa = a.size() - n;
    size_t sb = b.size() - n;
    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
        double d0 = a[sa + i] - b[sb + i];
        double d1 = a[sa + i + 1] - b[sb + i + 1];
        if ((d0 <= 0 && d1 >= 0) || (d0 >= 0 && d1 <= 0)) {
            double t = d0 / (d0 - d1 + 1e-12);
            return i + t;
        }
    }
    return -1.0;
}

// Draw a focus history plot
void DualCam::drawFocusPlot(cv::Mat& plotImg,
                            const std::deque<double>& a,
                            const std::deque<double>& b,
                            int margin,
                            int thickness,
                            double eqIdx) {
    plotImg.setTo(cv::Scalar(32, 32, 32));
    int w = plotImg.cols;
    int h = plotImg.rows;
    size_t n = std::min(a.size(), b.size());
    if (n < 2) return;
    size_t sa = a.size() - n;
    size_t sb = b.size() - n;
    double maxv = 1.0;
    // Find maximum value across both histories for normalization
    for (size_t i = 0; i < n; ++i) {
        if (a[sa + i] > maxv) maxv = a[sa + i];
        if (b[sb + i] > maxv) maxv = b[sb + i];
    }
    auto toX = [&](double idx){ return margin + (w - 2*margin) * (idx / (double)(n - 1)); };
    auto toY = [&](double v){ return h - margin - (h - 2*margin) * (v / maxv); };
    // Draw lines for a (green) and b (red)
    for (size_t i = 1; i < n; ++i) {
        double x0 = toX((double)(i - 1));
        double x1 = toX((double)i);
        double y0a = toY(a[sa + i - 1]);
        double y1a = toY(a[sa + i]);
        double y0b = toY(b[sb + i - 1]);
        double y1b = toY(b[sb + i]);
        cv::line(plotImg, cv::Point((int)x0, (int)y0a), cv::Point((int)x1, (int)y1a), cv::Scalar(0,255,0), thickness, cv::LINE_AA);
        cv::line(plotImg, cv::Point((int)x0, (int)y0b), cv::Point((int)x1, (int)y1b), cv::Scalar(255,0,0), thickness, cv::LINE_AA);
    }
    // Draw equal index marker if valid
    if (eqIdx >= 0.0) {
        double xt = toX(eqIdx);
        int xi = (int)std::round(xt);
        cv::line(plotImg, cv::Point(xi, margin), cv::Point(xi, h - margin), cv::Scalar(200,200,200), 1, cv::LINE_AA);
    }
}

// Resize src to fit inside (W,H) with aspect ratio, letterbox with black
cv::Mat DualCam::resizeFit(const cv::Mat& src, int W, int H) {
    if (src.empty()) {
        return cv::Mat(H, W, CV_8UC3, cv::Scalar(0,0,0));
    }
    double ar = src.cols / (double)src.rows;
    double target = W / (double)H;
    cv::Mat dst;
    if (ar > target) {
        int nh = (int)std::round(W / ar);
        cv::resize(src, dst, cv::Size(W, nh), 0, 0, cv::INTER_AREA);
    } else {
        int nw = (int)std::round(H * ar);
        cv::resize(src, dst, cv::Size(nw, H), 0, 0, cv::INTER_AREA);
    }
    cv::Mat out(H, W, dst.type(), cv::Scalar(0,0,0));
    int x = (W - dst.cols) / 2;
    int y = (H - dst.rows) / 2;
    dst.copyTo(out(cv::Rect(x, y, dst.cols, dst.rows)));
    return out;
}

// Convert BGR (or grayscale) to RGB QImage
QImage DualCam::toQImageRGB(const cv::Mat& bgr) {
    if (bgr.empty()) return QImage();
    cv::Mat rgb;
    if (bgr.channels() == 3) {
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    } else if (bgr.channels() == 1) {
        cv::cvtColor(bgr, rgb, cv::COLOR_GRAY2RGB);
    } else if (bgr.channels() == 4) {
        cv::cvtColor(bgr, rgb, cv::COLOR_BGRA2RGB);
    } else {
        bgr.convertTo(rgb, CV_8UC1);
        cv::cvtColor(rgb, rgb, cv::COLOR_GRAY2RGB);
    }
    return QImage(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888).copy();
}

// Constructor: initialize UI and open cameras
DualCam::DualCam(QWidget* parent) : QWidget(parent) {
    // Determine capture resolution and fps from environment or defaults
    int envW = 640;
    int envH = 480;
    double envFps = 30.0;
    if (const char* wenv = std::getenv("CAP_WIDTH")) {
        int wi = atoi(wenv);
        if (wi > 0) envW = wi;
    }
    if (const char* henv = std::getenv("CAP_HEIGHT")) {
        int hi = atoi(henv);
        if (hi > 0) envH = hi;
    }
    if (const char* fenv = std::getenv("CAP_FPS")) {
        double fi = atof(fenv);
        if (fi > 1.0) envFps = fi;
    }
    // Use environment dimensions for tile size if larger than defaults
    tileW_ = envW;
    tileH_ = envH;
    fps_   = envFps;
    histCap_ = 120;

    view_ = new QLabel(this);
    timer_ = new QTimer(this);
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0,0,0,0);
    layout->addWidget(view_);
    setLayout(layout);

    // Open cameras
    {
        auto cands0 = buildCandidates(0);
        auto capOpt0 = openWithCandidates(cands0, tileW_, tileH_, fps_);
        if (capOpt0) {
            cap0_ = std::move(*capOpt0);
            ok0_ = true;
        } else {
            ok0_ = false;
        }
    }
    {
        auto cands1 = buildCandidates(1);
        auto capOpt1 = openWithCandidates(cands1, tileW_, tileH_, fps_);
        if (capOpt1) {
            cap1_ = std::move(*capOpt1);
            ok1_ = true;
        } else {
            ok1_ = false;
        }
    }

    // Initialize images
    black_ = cv::Mat(tileH_, tileW_, CV_8UC3, cv::Scalar(0,0,0));
    grid_ = cv::Mat(tileH_ * 2, tileW_ * 2, CV_8UC3, cv::Scalar(0,0,0));
    plotImg_ = cv::Mat(tileH_, tileW_, CV_8UC3, cv::Scalar(0,0,0));

    // Setup timer
    connect(timer_, &QTimer::timeout, this, &DualCam::tick);
    int interval = (int)std::round(1000.0 / std::max(1.0, fps_));
    timer_->start(interval);
}

DualCam::~DualCam() {
    // QObjects (timer_, view_) will be deleted automatically as children
    // Caputures release themselves on destruction
}

// Handle keypresses: ESC closes, F freezes history, T toggles graph, S toggles view mode
void DualCam::keyPressEvent(QKeyEvent* event) {
    switch (event->key()) {
    case Qt::Key_Escape:
        close();
        break;
    case Qt::Key_F:
        freezeHistory_ = !freezeHistory_;
        break;
    case Qt::Key_T:
        showGraph_ = !showGraph_;
        break;
    case Qt::Key_S:
        mode_ = (mode_ == ViewMode::Analytics4Q ? ViewMode::SideBySide : ViewMode::Analytics4Q);
        break;
    default:
        QWidget::keyPressEvent(event);
    }
}

// Periodic update: read frames, compute metrics, update display
void DualCam::tick() {
    // Update tile size if window has changed
    int winW = this->width();
    int winH = this->height();
    int newTileW;
    int newTileH;
    if (mode_ == ViewMode::Analytics4Q) {
        newTileW = std::max(1, winW / 2);
        newTileH = std::max(1, winH / 2);
    } else {
        // side by side: height remains tileH_, width per tile is half the window
        newTileW = std::max(1, winW / 2);
        newTileH = std::max(1, winH);
    }
    if (newTileW != tileW_ || newTileH != tileH_) {
        tileW_ = newTileW;
        tileH_ = newTileH;
        histCap_ = std::max(50, tileW_);
        grid_ = cv::Mat(tileH_ * (mode_ == ViewMode::Analytics4Q ? 2 : 1), tileW_ * 2, CV_8UC3, cv::Scalar(0,0,0));
        plotImg_ = cv::Mat(tileH_, tileW_, CV_8UC3, cv::Scalar(0,0,0));
        black_ = cv::Mat(tileH_, tileW_, CV_8UC3, cv::Scalar(0,0,0));
    }
    cv::Mat f0, f1;
    bool got0 = ok0_ && cap0_.read(f0);
    bool got1 = ok1_ && cap1_.read(f1);
    // If frames are 16-bit (e.g. Y16 or 16-bit grayscale), convert them down to 8-bit
    if (got0 && !f0.empty() && f0.depth() == CV_16U) {
        // Scale 16-bit to 8-bit by shifting right 8 bits.  Use per-channel conversion.
        double scale = 1.0 / 256.0;
        int type = CV_MAKETYPE(CV_8U, f0.channels());
        cv::Mat tmp;
        f0.convertTo(tmp, type, scale);
        f0 = tmp;
    }
    if (got1 && !f1.empty() && f1.depth() == CV_16U) {
        double scale = 1.0 / 256.0;
        int type = CV_MAKETYPE(CV_8U, f1.channels());
        cv::Mat tmp;
        f1.convertTo(tmp, type, scale);
        f1 = tmp;
    }
    cv::Mat r0b = got0 ? f0 : black_;
    cv::Mat r1b = got1 ? f1 : black_;
    // Convert grayscale to BGR
    if (!r0b.empty() && r0b.channels() == 1) cv::cvtColor(r0b, r0b, cv::COLOR_GRAY2BGR);
    if (!r1b.empty() && r1b.channels() == 1) cv::cvtColor(r1b, r1b, cv::COLOR_GRAY2BGR);
    // Resize to fit tiles
    r0b = resizeFit(r0b, tileW_, tileH_);
    r1b = resizeFit(r1b, tileW_, tileH_);
    // Compute difference heatmap
    cv::Mat diffColor;
    if (got0 && got1) {
        cv::Mat g0, g1, diff;
        cv::cvtColor(r0b, g0, cv::COLOR_BGR2GRAY);
        cv::cvtColor(r1b, g1, cv::COLOR_BGR2GRAY);
        cv::absdiff(g0, g1, diff);
        cv::applyColorMap(diff, diffColor, cv::COLORMAP_MAGMA);
    } else {
        diffColor = black_.clone();
    }
    // Compute focus measures
    double s0 = got0 ? focusMeasure(r0b) : 0.0;
    double s1 = got1 ? focusMeasure(r1b) : 0.0;
    // Update history if not frozen
    if (!freezeHistory_) {
        h0_.push_back(s0);
        h1_.push_back(s1);
        if (h0_.size() > (size_t)histCap_) h0_.pop_front();
        if (h1_.size() > (size_t)histCap_) h1_.pop_front();
    }
    double eqIdx = findEqualIndex(h0_, h1_);
    // Draw plot if analytics mode
    if (mode_ == ViewMode::Analytics4Q) {
        drawFocusPlot(plotImg_, h0_, h1_, 24, 2, eqIdx);
    }
    // Assemble grid
    if (mode_ == ViewMode::Analytics4Q) {
        // Top-left: camera 0
        r0b.copyTo(grid_(cv::Rect(0, 0, tileW_, tileH_)));
        // Top-right: camera 1
        r1b.copyTo(grid_(cv::Rect(tileW_, 0, tileW_, tileH_)));
        // Bottom-left: difference
        diffColor.copyTo(grid_(cv::Rect(0, tileH_, tileW_, tileH_)));
        // Bottom-right: plot or blank
        if (showGraph_) {
            plotImg_.copyTo(grid_(cv::Rect(tileW_, tileH_, tileW_, tileH_)));
        } else {
            cv::Mat blank(tileH_, tileW_, CV_8UC3, cv::Scalar(0,0,0));
            blank.copyTo(grid_(cv::Rect(tileW_, tileH_, tileW_, tileH_)));
        }
    } else {
        // Side by side: single row
        // Ensure grid has correct size
        if (grid_.rows != tileH_ || grid_.cols != tileW_ * 2) {
            grid_ = cv::Mat(tileH_, tileW_ * 2, CV_8UC3, cv::Scalar(0,0,0));
        }
        r0b.copyTo(grid_(cv::Rect(0, 0, tileW_, tileH_)));
        r1b.copyTo(grid_(cv::Rect(tileW_, 0, tileW_, tileH_)));
    }
    // Overlay info text
    std::string info = std::string("S0:") + std::to_string((int)s0) + " S1:" + std::to_string((int)s1) +
        " D:" + std::to_string((int)std::llabs((long long) (s0 - s1)));
    cv::putText(grid_, info, cv::Point(16, 32), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    // Convert to QImage and display
    QImage img = toQImageRGB(grid_);
    view_->setPixmap(QPixmap::fromImage(img));
}