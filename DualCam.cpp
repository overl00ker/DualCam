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

static inline std::string getenv_lower(const char* key) 
{
    const char* v = std::getenv(key);
    if (!v) return std::string();
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

std::vector<std::tuple<std::string, bool, bool>> DualCam::buildCandidates(int index) 
{
    std::vector<std::tuple<std::string, bool, bool>> cands;
    std::string prioStr = getenv_lower("CAP_PRIORITY");
    bool preferV4L2Only = (!prioStr.empty() && prioStr.find("v4l2") != std::string::npos);

    if (!preferV4L2Only) 
    {
        std::string key = std::string("GST_PIPELINE_CAM") + std::to_string(index);
        const char* val = std::getenv(key.c_str());
        if (val && *val) 
        {
            std::string spec(val);
            if (spec.rfind("gst:", 0) == 0) spec = spec.substr(4);
            cands.emplace_back(spec, /*isGst*/true, /*isDevice*/false);
        }
    }

    {
        std::string key = std::string("DEV_VIDEO_CAM") + std::to_string(index);
        const char* val = std::getenv(key.c_str());
        if (val && *val) 
        {
            cands.emplace_back(std::string(val), /*isGst*/false, /*isDevice*/true);
        }
    }
    if (!preferV4L2Only) 
    {
        std::string key = std::string("GST_CAMERA_NAME_CAM") + std::to_string(index);
        const char* val = std::getenv(key.c_str());
        if (val && *val) 
        {
            std::string name(val);
            int w = 1440, h = 1088, f = 60;
            if (const char* wenv = std::getenv("GST_WIDTH")) 
            {
                int wi = atoi(wenv); if (wi > 0) w = wi;
            }
            if (const char* henv = std::getenv("GST_HEIGHT")) 
            {
                int hi = atoi(henv); if (hi > 0) h = hi;
            }
            if (const char* fpsenv = std::getenv("GST_FPS")) 
            {
                int fi = atoi(fpsenv); if (fi > 0) f = fi;
            }
            std::string pipeline = std::string("libcamerasrc camera-name=") + name +
                " ! video/x-raw,width=" + std::to_string(w) + ",height=" + std::to_string(h) + ",format=GRAY8"
                " ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true sync=false";
            cands.emplace_back(pipeline, true, false);
        }
    }
    bool preferLibcamera = false;
    {
        std::string pr = getenv_lower("CAP_PRIORITY");
        if (!pr.empty()) 
        {
            if (pr.find("v4l2") != std::string::npos) 
            {
                preferLibcamera = false;
            } else if (pr.find("libcamera") != std::string::npos || pr.find("gst") != std::string::npos || pr.find("gstreamer") != std::string::npos) 
            {
                preferLibcamera = true;
            }
        }
    }
    int w = 1400, h = 1088, f = 60;
    if (const char* wenv = std::getenv("GST_WIDTH")) 
    {
        int wi = atoi(wenv); if (wi > 0) w = wi;
    }
    if (const char* henv = std::getenv("GST_HEIGHT")) 
    {
        int hi = atoi(henv); if (hi > 0) h = hi;
    }
    if (const char* fpsenv = std::getenv("GST_FPS")) 
    {
        int fi = atoi(fpsenv); if (fi > 0) f = fi;
    }
    if (preferLibcamera) {
        std::string pipe = std::string("libcamerasrc camera-id=") + std::to_string(index) +
            " ! video/x-raw,width=" + std::to_string(w) + ",height=" + std::to_string(h) + ",format=YUY2"
            " ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true sync=false";
        cands.emplace_back(pipe, true, false);
    }
#if defined(__linux__)
    if (!preferLibcamera) 
    {
        cands.emplace_back(std::to_string(index), false, false);
        // device path
        cands.emplace_back(std::string("/dev/video") + std::to_string(index), false, true);
    } else 
    {
        cands.emplace_back(std::to_string(index), false, false);
    }
#else
    cands.emplace_back(std::to_string(index), false, false);
#endif
    if (!preferLibcamera && !preferV4L2Only) 
    {
        std::string pipe = std::string("libcamerasrc camera-id=") + std::to_string(index) +
            " ! video/x-raw,width=" + std::to_string(w) + ",height=" + std::to_string(h) + ",format=YUY2"
            " ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true sync=false";
        cands.emplace_back(pipe, true, false);
    }
    return cands;
}

void DualCam::warmup(cv::VideoCapture& cap, int ms) 
{
    using namespace std::chrono;
    auto t0 = steady_clock::now();
    cv::Mat tmp;
    while (true) 
    {
        cap.read(tmp);
        auto dt = duration_cast<milliseconds>(steady_clock::now() - t0);
        if (dt.count() >= ms) break;
        std::this_thread::sleep_for(milliseconds(10));
    }
}

bool DualCam::readOnce(cv::VideoCapture& cap, cv::Mat& out) 
{
    using namespace std::chrono;
    auto t0 = steady_clock::now();
    while (true) {
        if (cap.read(out) && !out.empty()) return true;
        auto dt = duration_cast<milliseconds>(steady_clock::now() - t0);
        if (dt.count() > 500) return false;
        std::this_thread::sleep_for(milliseconds(10));
    }
}

std::optional<cv::VideoCapture> DualCam::openWithCandidates(const std::vector<std::tuple<std::string, bool, bool>>& cands,
                                                            int width, int height, double fps) 
{
    for (const auto& cand : cands) 
    {
        const std::string& spec = std::get<0>(cand);
        bool isGst = std::get<1>(cand);
        bool isDev = std::get<2>(cand);
        cv::VideoCapture cap;
        if (isGst) {
            cap.open(spec, cv::CAP_GSTREAMER);
        } else 
        {
            bool numeric = !spec.empty();
            for (size_t i = 0; i < spec.size(); ++i) 
            {
                char c = spec[i];
                if (!std::isdigit((unsigned char)c) && !(i == 0 && (c == '+' || c == '-'))) {
                    numeric = false; break;
                }
            }
            if (numeric) 
            {
                int idx = 0;
                try { idx = std::stoi(spec); } catch (...) { numeric = false; }
                if (numeric) 
                {
                    cap.open(idx, cv::CAP_ANY);
                }
            }
            if (!numeric) 
            {
#if defined(__linux__)
                if (isDev) 
                {
                    cap.open(spec, cv::CAP_V4L2);
                } else 
                {
                    cap.open(spec, cv::CAP_V4L2);
                }
#else
                cap.open(spec, cv::CAP_ANY);
#endif
            }
        }
        if (!cap.isOpened()) continue;
        if (!isGst) 
        {
            if (width > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
            if (height > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            if (fps > 0.0) cap.set(cv::CAP_PROP_FPS, fps);
        }
        warmup(cap, 600);
        cv::Mat f;
        if (readOnce(cap, f)) 
        {
            return cap;
        }
    }
    return std::nullopt;
}

double DualCam::focusMeasure(const cv::Mat& bgr) 
{
    if (bgr.empty()) return 0.0;
    cv::Mat g;
    if (bgr.channels() == 3) 
    {
        cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
    } else if (bgr.channels() == 1) 
    {
        g = bgr;
    } else 
    {
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

double DualCam::findEqualIndex(const std::deque<double>& a, const std::deque<double>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n < 2) return -1.0;
    size_t sa = a.size() - n;
    size_t sb = b.size() - n;
    for (int i = static_cast<int>(n) - 2; i >= 0; --i) 
    {
        double d0 = a[sa + i] - b[sb + i];
        double d1 = a[sa + i + 1] - b[sb + i + 1];
        if ((d0 <= 0 && d1 >= 0) || (d0 >= 0 && d1 <= 0)) 
        {
            double t = d0 / (d0 - d1 + 1e-12);
            return i + t;
        }
    }
    return -1.0;
}

void DualCam::drawFocusPlot(cv::Mat& plotImg,
                            const std::deque<double>& a,
                            const std::deque<double>& b,
                            int margin,
                            int thickness,
                            double eqIdx) 
{
    plotImg.setTo(cv::Scalar(32, 32, 32));
    int w = plotImg.cols;
    int h = plotImg.rows;
    size_t n = std::min(a.size(), b.size());
    if (n < 2) return;
    size_t sa = a.size() - n;
    size_t sb = b.size() - n;
    double maxv = 1.0;
    for (size_t i = 0; i < n; ++i) 
    {
        if (a[sa + i] > maxv) maxv = a[sa + i];
        if (b[sb + i] > maxv) maxv = b[sb + i];
    }
    auto toX = [&](double idx){ return margin + (w - 2*margin) * (idx / (double)(n - 1)); };
    auto toY = [&](double v){ return h - margin - (h - 2*margin) * (v / maxv); };
    for (size_t i = 1; i < n; ++i) 
    {
        double x0 = toX((double)(i - 1));
        double x1 = toX((double)i);
        double y0a = toY(a[sa + i - 1]);
        double y1a = toY(a[sa + i]);
        double y0b = toY(b[sb + i - 1]);
        double y1b = toY(b[sb + i]);
        cv::line(plotImg, cv::Point((int)x0, (int)y0a), cv::Point((int)x1, (int)y1a), cv::Scalar(0,255,0), thickness, cv::LINE_AA);
        cv::line(plotImg, cv::Point((int)x0, (int)y0b), cv::Point((int)x1, (int)y1b), cv::Scalar(255,0,0), thickness, cv::LINE_AA);
    }
    if (eqIdx >= 0.0) 
    {
        double xt = toX(eqIdx);
        int xi = (int)std::round(xt);
        cv::line(plotImg, cv::Point(xi, margin), cv::Point(xi, h - margin), cv::Scalar(200,200,200), 1, cv::LINE_AA);
    }
}

cv::Mat DualCam::resizeFit(const cv::Mat& src, int W, int H) {
    if (src.empty()) 
    {
        return cv::Mat(H, W, CV_8UC3, cv::Scalar(0,0,0));
    }
    double ar = src.cols / (double)src.rows;
    double target = W / (double)H;
    cv::Mat dst;
    if (ar > target) 
    {
        int nh = (int)std::round(W / ar);
        cv::resize(src, dst, cv::Size(W, nh), 0, 0, cv::INTER_AREA);
    } else 
    {
        int nw = (int)std::round(H * ar);
        cv::resize(src, dst, cv::Size(nw, H), 0, 0, cv::INTER_AREA);
    }
    cv::Mat out(H, W, dst.type(), cv::Scalar(0,0,0));
    int x = (W - dst.cols) / 2;
    int y = (H - dst.rows) / 2;
    dst.copyTo(out(cv::Rect(x, y, dst.cols, dst.rows)));
    return out;
}

QImage DualCam::toQImageRGB(const cv::Mat& bgr) 
{
    if (bgr.empty()) return QImage();
    cv::Mat rgb;
    if (bgr.channels() == 3) 
    {
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    } else if (bgr.channels() == 1) 
    {
        cv::cvtColor(bgr, rgb, cv::COLOR_GRAY2RGB);
    } else if (bgr.channels() == 4) 
    {
        cv::cvtColor(bgr, rgb, cv::COLOR_BGRA2RGB);
    } else {
        bgr.convertTo(rgb, CV_8UC1);
        cv::cvtColor(rgb, rgb, cv::COLOR_GRAY2RGB);
    }
    return QImage(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888).copy();
}

DualCam::DualCam(QWidget* parent) : QWidget(parent) 
{
    int envW = 640;
    int envH = 480;
    double envFps = 30.0;
    if (const char* wenv = std::getenv("CAP_WIDTH")) 
    {
        int wi = atoi(wenv);
        if (wi > 0) envW = wi;
    }
    if (const char* henv = std::getenv("CAP_HEIGHT")) 
    {
        int hi = atoi(henv);
        if (hi > 0) envH = hi;
    }
    if (const char* fenv = std::getenv("CAP_FPS")) 
    {
        double fi = atof(fenv);
        if (fi > 1.0) envFps = fi;
    }
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

    {
        auto cands0 = buildCandidates(0);
        auto capOpt0 = openWithCandidates(cands0, tileW_, tileH_, fps_);
        if (capOpt0) 
        {
            cap0_ = std::move(*capOpt0);
            ok0_ = true;
        } else {
            ok0_ = false;
        }
    }
    {
        auto cands1 = buildCandidates(1);
        auto capOpt1 = openWithCandidates(cands1, tileW_, tileH_, fps_);
        if (capOpt1) 
        {
            cap1_ = std::move(*capOpt1);
            ok1_ = true;
        } else 
        {
            ok1_ = false;
        }
    }

    black_ = cv::Mat(tileH_, tileW_, CV_8UC3, cv::Scalar(0,0,0));
    grid_ = cv::Mat(tileH_ * 2, tileW_ * 2, CV_8UC3, cv::Scalar(0,0,0));
    plotImg_ = cv::Mat(tileH_, tileW_, CV_8UC3, cv::Scalar(0,0,0));

    connect(timer_, &QTimer::timeout, this, &DualCam::tick);
    int interval = (int)std::round(1000.0 / std::max(1.0, fps_));
    timer_->start(interval);
}

DualCam::~DualCam() 
{
}
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

void DualCam::tick() 
{
    int winW = this->width();
    int winH = this->height();
    int newTileW;
    int newTileH;
    if (mode_ == ViewMode::Analytics4Q) {
        newTileW = std::max(1, winW / 2);
        newTileH = std::max(1, winH / 2);
    } else 
    {
        newTileW = std::max(1, winW / 2);
        newTileH = std::max(1, winH);
    }
    if (newTileW != tileW_ || newTileH != tileH_) 
    {
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
    if (got0 && !f0.empty() && f0.depth() == CV_16U) 
    {
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
    if (!r0b.empty() && r0b.channels() == 1) cv::cvtColor(r0b, r0b, cv::COLOR_GRAY2BGR);
    if (!r1b.empty() && r1b.channels() == 1) cv::cvtColor(r1b, r1b, cv::COLOR_GRAY2BGR);
    r0b = resizeFit(r0b, tileW_, tileH_);
    r1b = resizeFit(r1b, tileW_, tileH_);
    cv::Mat diffColor;
    if (got0 && got1) 
    {
        cv::Mat g0, g1, diff;
        cv::cvtColor(r0b, g0, cv::COLOR_BGR2GRAY);
        cv::cvtColor(r1b, g1, cv::COLOR_BGR2GRAY);
        cv::absdiff(g0, g1, diff);
        cv::applyColorMap(diff, diffColor, cv::COLORMAP_MAGMA);
    } else 
    {
        diffColor = black_.clone();
    }
    double s0 = got0 ? focusMeasure(r0b) : 0.0;
    double s1 = got1 ? focusMeasure(r1b) : 0.0;
    if (!freezeHistory_) 
    {
        h0_.push_back(s0);
        h1_.push_back(s1);
        if (h0_.size() > (size_t)histCap_) h0_.pop_front();
        if (h1_.size() > (size_t)histCap_) h1_.pop_front();
    }
    double eqIdx = findEqualIndex(h0_, h1_);
    if (mode_ == ViewMode::Analytics4Q) 
    {
        drawFocusPlot(plotImg_, h0_, h1_, 24, 2, eqIdx);
    }
    if (mode_ == ViewMode::Analytics4Q) 
    {
        r0b.copyTo(grid_(cv::Rect(0, 0, tileW_, tileH_)));
        r1b.copyTo(grid_(cv::Rect(tileW_, 0, tileW_, tileH_)));
        diffColor.copyTo(grid_(cv::Rect(0, tileH_, tileW_, tileH_)));
        if (showGraph_) {
            plotImg_.copyTo(grid_(cv::Rect(tileW_, tileH_, tileW_, tileH_)));
        } else {
            cv::Mat blank(tileH_, tileW_, CV_8UC3, cv::Scalar(0,0,0));
            blank.copyTo(grid_(cv::Rect(tileW_, tileH_, tileW_, tileH_)));
        }
    } else 
    {
        if (grid_.rows != tileH_ || grid_.cols != tileW_ * 2) 
        {
            grid_ = cv::Mat(tileH_, tileW_ * 2, CV_8UC3, cv::Scalar(0,0,0));
        }
        r0b.copyTo(grid_(cv::Rect(0, 0, tileW_, tileH_)));
        r1b.copyTo(grid_(cv::Rect(tileW_, 0, tileW_, tileH_)));
    }
    std::string info = std::string("S0:") + std::to_string((int)s0) + " S1:" + std::to_string((int)s1) + " D:" + std::to_string((int)std::llabs((long long) (s0 - s1)));
    cv::putText(grid_, info, cv::Point(16, 32), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    QImage img = toQImageRGB(grid_);
    view_->setPixmap(QPixmap::fromImage(img));
}
