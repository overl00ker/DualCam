#include "mainwindow.h"
#ifndef DUALCAM_SEPARATE_VIEWER
#include "viewer_dialogs.h"
#endif

#include <QPropertyAnimation>
#include <QParallelAnimationGroup>
#include <QGraphicsOpacityEffect>
#include <QVariantAnimation>

#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <QImage>
#include <QPixmap>
#include <QStatusBar>
#include <QComboBox>
#include <QCheckBox>
#include <QSplitter>
#include <QSpinBox>
#include <QSlider>
#include <QProcess>
#include <QRegularExpression>
#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QGroupBox>
#include <QTabWidget>
#include <QScrollArea>
#include <QStackedWidget>
#include <QSettings>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QSignalBlocker>
#include <QDialog>
#include <QListWidget>
#include <QListWidgetItem>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QCloseEvent>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>
#include <QMenu>
#include <QAction>
#include <QInputDialog>
#include <QMessageBox>
#include <QKeySequenceEdit>
#include <QFrame>
#include <QStyle>
#include <QButtonGroup>
#include <QDesktopServices>
#include <QUrl>
#include <QtCharts/QValueAxis>
#include <QtCharts/QChart>
#include <QtCharts/QLegend>
#include <QScreen>
#include <QStandardPaths>
#include <QThread>
#include <QMutexLocker>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <vector>
#include <cmath>

#include <QPainter>
#include <QPaintEvent>
#include <QShortcut>
#include <QToolBar>
#include <QToolButton>
#include <QPointer>
#include <QPropertyAnimation>
#include <QEasingCurve>
#include <QWidgetAction>
#include <QColorDialog>
#include <QFileDialog>
#include <QTextStream>
#include <QProcess>
#include <QLinearGradient>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>

GpuImageView::GpuImageView(QWidget* parent)
    : QOpenGLWidget(parent)
{
    setAttribute(Qt::WA_OpaquePaintEvent);
    setAutoFillBackground(false);
    setMinimumSize(160, 120);
}

void GpuImageView::initializeGL()
{
    initializeOpenGLFunctions();
}

void GpuImageView::resizeGL(int , int ) {}

void GpuImageView::setImage(const QImage& img)
{
    m_image = img;
    update();
}

void GpuImageView::setPlaceholder(const QString& text)
{
    m_placeholder = text;
    m_image = QImage();
    update();
}

void GpuImageView::setOverlayText(const QString& text, bool rightAlign)
{
    m_overlayText = text;
    m_overlayRight = rightAlign;
    update();
}

void GpuImageView::paintGL()
{
    QPainter p(this);
    p.fillRect(rect(), QColor("#000000"));

    if (m_image.isNull()) {
        if (!m_placeholder.isEmpty()) {

            p.setRenderHint(QPainter::Antialiasing, true);

            QFont fId("Space Mono");
            fId.setPointSize(11);
            fId.setBold(true);
            fId.setLetterSpacing(QFont::AbsoluteSpacing, 2.0);
            QFontMetrics fmId(fId);
            int idH = fmId.height();

            QFont fStat("Space Mono");
            fStat.setPointSize(8);
            fStat.setLetterSpacing(QFont::AbsoluteSpacing, 1.5);
            QFontMetrics fmStat(fStat);
            int statH = fmStat.height();

            const QString id   = m_placeholder;
            const QString stat = "NO SIGNAL";
            int blockH = idH + 6 + 1 + 6 + statH;
            int cy = (height() - blockH) / 2;
            int cx = width() / 2;

            p.setFont(fId);
            p.setPen(QColor(0xa3, 0x8b, 0x89));
            p.drawText(QRect(0, cy, width(), idH), Qt::AlignHCenter | Qt::AlignVCenter, id);

            int ruleY = cy + idH + 6;
            p.fillRect(QRect(cx - 12, ruleY, 24, 1), QColor(0x50, 0x0b, 0x0b));

            p.setFont(fStat);
            p.setPen(QColor(0x8a, 0x84, 0x82));
            p.drawText(QRect(0, ruleY + 6, width(), statH), Qt::AlignHCenter | Qt::AlignVCenter, stat);
        }
        return;
    }

    {
        const QSize is = m_image.size();
        const QSize ws = size();
        const double kx = static_cast<double>(ws.width())  / is.width();
        const double ky = static_cast<double>(ws.height()) / is.height();
        const double k  = m_stretch ? std::max(kx, ky) : std::min(kx, ky);
        const int dw = static_cast<int>(is.width()  * k);
        const int dh = static_cast<int>(is.height() * k);
        const int dx = (ws.width()  - dw) / 2;
        const int dy = (ws.height() - dh) / 2;
        p.setRenderHint(QPainter::SmoothPixmapTransform, false);
        p.drawImage(QRect(dx, dy, dw, dh), m_image);
    }

    if (!m_overlayText.isEmpty()) {

        QFont f = p.font();
        f.setFamily("Space Mono");
        f.setPointSize(9);
        f.setBold(true);
        f.setLetterSpacing(QFont::AbsoluteSpacing, 1.0);
        p.setFont(f);
        QFontMetrics fm(f);
        QRect tr = fm.boundingRect(m_overlayText).adjusted(-8, -3, 8, 3);
        int x = m_overlayRight ? (width() - 8 - tr.width()) : 8;
        int y = 8;
        QRect bg(x, y, tr.width(), tr.height());
        p.fillRect(bg, QColor(10, 10, 10, 220));
        p.fillRect(QRect(bg.left(), bg.bottom(), bg.width(), 1), QColor(0x50, 0x0b, 0x0b));
        p.setPen(m_overlayColor);
        p.drawText(bg, Qt::AlignCenter, m_overlayText);
    }
}

CameraWorker::CameraWorker(QObject* parent) : QThread(parent) {
    m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16.0, false);
}
CameraWorker::~CameraWorker() {
    stopCameras();
}
void CameraWorker::startCameras(const std::string& pipe1, const std::string& pipe2, int w, int h, int fps) {
    m_cap1.open(pipe1, cv::CAP_GSTREAMER);
    m_cap2.open(pipe2, cv::CAP_GSTREAMER);
    if (!m_cap1.isOpened() || !m_cap2.isOpened()) {
        emit cameraError("Failed to open GStreamer pipelines");
        return;
    }
    m_running = true;
    start();
}
void CameraWorker::startCamerasV4L2(int id1, int id2, int w, int h, int fps) {
#ifdef Q_OS_LINUX
    m_cap1.open(id1, cv::CAP_V4L2);
    m_cap2.open(id2, cv::CAP_V4L2);
#else
    m_cap1.open(id1, cv::CAP_DSHOW);
    m_cap2.open(id2, cv::CAP_DSHOW);
#endif
    if (!m_cap1.isOpened() || !m_cap2.isOpened()) {
        emit cameraError("Failed to open one or both cameras (V4L2/DSHOW fallback)");
        if (m_cap1.isOpened()) m_cap1.release();
        if (m_cap2.isOpened()) m_cap2.release();
        return;
    }
    m_cap1.set(cv::CAP_PROP_FRAME_WIDTH, w);
    m_cap1.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    m_cap1.set(cv::CAP_PROP_FPS, fps);
    m_cap2.set(cv::CAP_PROP_FRAME_WIDTH, w);
    m_cap2.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    m_cap2.set(cv::CAP_PROP_FPS, fps);
    m_running = true;
    start();
}
void CameraWorker::stopCameras() {
    m_running = false;
    if (m_cap1.isOpened()) m_cap1.release();
    if (m_cap2.isOpened()) m_cap2.release();
    if (!wait(2000)) {
        requestInterruption();
        if (!wait(1000)) {
            terminate();
            wait();
        }
    }
}
void CameraWorker::setParams(const WorkerParams& p) {
    m_paramMutex.lock();
    m_params = p;
    m_paramMutex.unlock();
}
cv::Mat CameraWorker::toWorkingFormat(const cv::Mat& frame, ColorMode mode) {
    if (frame.empty()) return frame;
    cv::Mat res;
    if (mode == ColorMode::GRAY_CV) {
        if (frame.channels() == 3) cv::cvtColor(frame, res, cv::COLOR_BGR2GRAY);
        else res = frame.clone();
    } else {
        if (frame.channels() == 1 && mode == ColorMode::COLOR) cv::cvtColor(frame, res, cv::COLOR_GRAY2BGR);
        else res = frame.clone();
    }
    return res;
}
double CameraWorker::detectMotion(const cv::Mat& frame, double /*thr*/) {

    if (frame.empty()) return 0.0;
    cv::Mat fgMask;
    m_bgSubtractor->apply(frame, fgMask, 0.01);
    double nonZero = cv::countNonZero(fgMask);
    return nonZero / (frame.cols * frame.rows);
}
cv::Mat CameraWorker::applyTemporalDenoise(cv::Mat& frame, cv::Mat& ema, int bufferSize) {
    if (bufferSize <= 1 || frame.empty()) return frame;
    double alpha = 1.0 / static_cast<double>(bufferSize);
    int cvType = (frame.channels() == 3) ? CV_32FC3 : CV_32F;
    if (ema.empty() || ema.size() != frame.size() || ema.type() != cvType) {
        frame.convertTo(ema, cvType);
    } else {
        cv::Mat tmp;
        frame.convertTo(tmp, cvType);
        cv::accumulateWeighted(tmp, ema, alpha);
    }
    cv::Mat res;
    ema.convertTo(res, frame.type());
    return res;
}
double CameraWorker::calculateFocus(const cv::Mat& frame) {
    if (frame.empty()) return 0.0;
    cv::Mat gray;
    if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else gray = frame;
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    return stddev.val[0] * stddev.val[0];
}
void CameraWorker::run() {
    cv::Mat f1, f2;
    while (m_running) {
        if (!m_cap1.grab() || !m_cap2.grab()) continue;
        m_cap1.retrieve(f1);
        m_cap2.retrieve(f2);

        if (f1.empty() || f2.empty()) continue;

        m_paramMutex.lock();
        WorkerParams p = m_params;
        m_paramMutex.unlock();

        if (p.flipHor2 && p.flipVer2) cv::flip(f2, f2, -1);
        else if (p.flipHor2) cv::flip(f2, f2, 1);
        else if (p.flipVer2) cv::flip(f2, f2, 0);

        f1 = toWorkingFormat(f1, p.colorMode);
        f2 = toWorkingFormat(f2, p.colorMode);

        double motion = detectMotion(f2, p.motionThr);
        bool motionDetected = (motion > p.motionThr);

        if (motionDetected) {
            m_ema1.release();
            m_ema2.release();
        }

        f1 = applyTemporalDenoise(f1, m_ema1, p.bufferSize);
        f2 = applyTemporalDenoise(f2, m_ema2, p.bufferSize);

        if (p.applyBilateral) {
            const int depth = f1.depth();
            if (depth == CV_8U || depth == CV_32F) {
                int s = std::max(1, std::min(20, p.bilateralStrength));
                int d = (s % 2 == 0) ? s + 1 : s;
                double sigma = 10.0 * s;
                cv::Mat b1, b2;
                cv::bilateralFilter(f1, b1, d, sigma, sigma);
                cv::bilateralFilter(f2, b2, d, sigma, sigma);
                f1 = b1; f2 = b2;
            }
        }

        double focus1 = calculateFocus(f1);
        double focus2 = calculateFocus(f2);

        m_frameCount++;

        if (m_pendingFrames.load() >= 2) {

            continue;
        }
        m_pendingFrames.fetch_add(1);
        emit framesProcessed(f1.clone(), f2.clone(), focus1, focus2, motionDetected, m_frameCount);
    }
}

namespace {
class AspectImageLabel : public QLabel {
public:
    explicit AspectImageLabel(QWidget* parent = nullptr) : QLabel(parent) {
        setAlignment(Qt::AlignCenter);
        setMinimumSize(1, 1);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }
    void setOriginal(const QPixmap& p) {
        m_orig = p;
        rescale();
    }
protected:
    void resizeEvent(QResizeEvent* ev) override {
        rescale();
        QLabel::resizeEvent(ev);
    }
private:
    void rescale() {
        if (m_orig.isNull()) return;
        QLabel::setPixmap(m_orig.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    QPixmap m_orig;
};

static QRect displayRectFor(const QSize& widget, const QSize& image) {
    if (image.isEmpty() || widget.isEmpty()) return QRect();
    const double sx = double(widget.width()) / image.width();
    const double sy = double(widget.height()) / image.height();
    const double s = std::min(sx, sy);
    const int w = int(std::round(image.width()  * s));
    const int h = int(std::round(image.height() * s));
    return QRect((widget.width() - w) / 2, (widget.height() - h) / 2, w, h);
}

static QLabel* makeSectionLabel(const QString& t, QWidget* parent) {
    QLabel* l = new QLabel(t.toUpper(), parent);
    l->setProperty("role", "section");
    return l;
}
}

class AnalysisCanvas : public QWidget {
public:
    enum class Mode { Line, Rect };
    explicit AnalysisCanvas(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumSize(1, 1);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setMouseTracking(true);
        setFocusPolicy(Qt::StrongFocus);
        setCursor(Qt::CrossCursor);
    }
    void setImage(const QImage& img) { m_image = img; update(); }
    void setMode(Mode m) { m_mode = m; m_hasSelection = false; update(); }
    Mode mode() const { return m_mode; }
    void setAllowedAngles(const QList<int>& deg) { m_angles = deg; }
    void onLineSelected(std::function<void(QPoint,QPoint)> cb) { m_lineCb = std::move(cb); }
    void onRectSelected(std::function<void(QRect)> cb) { m_rectCb = std::move(cb); }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.fillRect(rect(), QColor("#0e0e0e"));
        if (m_image.isNull()) return;
        const QRect dr = displayRectFor(size(), m_image.size());
        p.drawImage(dr, m_image);
        if (!m_hasSelection && !m_dragging) return;

        QPen pen(QColor(0xff, 0xb1, 0x9a));
        pen.setWidth(2);
        p.setPen(pen);
        if (m_mode == Mode::Line) {
            QPoint a = imageToWidget(m_a);
            QPoint b = imageToWidget(m_b);
            p.drawLine(a, b);
            p.setBrush(QColor(0xff, 0xb1, 0x9a));
            p.drawEllipse(a, 3, 3);
            p.drawEllipse(b, 3, 3);
        } else {
            QRect imgR = QRect(m_a, m_b).normalized();
            QRect wr(imageToWidget(imgR.topLeft()), imageToWidget(imgR.bottomRight()));
            p.setBrush(QColor(255, 177, 154, 40));
            p.drawRect(wr.normalized());
        }
    }

    void mousePressEvent(QMouseEvent* ev) override {
        if (ev->button() != Qt::LeftButton || m_image.isNull()) return;
        m_dragging = true;
        m_a = widgetToImage(ev->pos());
        m_b = m_a;
        update();
    }

    void mouseMoveEvent(QMouseEvent* ev) override {
        if (!m_dragging) return;
        QPoint p = widgetToImage(ev->pos());
        if (m_mode == Mode::Line && (ev->modifiers() & Qt::ShiftModifier) && !m_angles.isEmpty()) {
            p = snapToAngle(m_a, p);
        }
        if (m_mode == Mode::Rect && (ev->modifiers() & Qt::ShiftModifier)) {
            int s = std::max(std::abs(p.x() - m_a.x()), std::abs(p.y() - m_a.y()));
            p = QPoint(m_a.x() + (p.x() >= m_a.x() ? s : -s),
                       m_a.y() + (p.y() >= m_a.y() ? s : -s));
        }
        m_b = p;
        update();
    }

    void mouseReleaseEvent(QMouseEvent* ev) override {
        if (ev->button() != Qt::LeftButton || !m_dragging) return;
        m_dragging = false;
        m_hasSelection = true;
        update();
        if (m_mode == Mode::Line) { if (m_lineCb) m_lineCb(m_a, m_b); }
        else                      { if (m_rectCb) m_rectCb(QRect(m_a, m_b).normalized()); }
    }

private:
    QPoint widgetToImage(const QPoint& w) const {
        const QRect dr = displayRectFor(size(), m_image.size());
        if (dr.width() == 0 || dr.height() == 0) return QPoint();
        const double sx = double(m_image.width())  / dr.width();
        const double sy = double(m_image.height()) / dr.height();
        int x = int(std::round((w.x() - dr.x()) * sx));
        int y = int(std::round((w.y() - dr.y()) * sy));
        x = std::clamp(x, 0, m_image.width()  - 1);
        y = std::clamp(y, 0, m_image.height() - 1);
        return QPoint(x, y);
    }
    QPoint imageToWidget(const QPoint& i) const {
        const QRect dr = displayRectFor(size(), m_image.size());
        if (m_image.width() == 0 || m_image.height() == 0) return QPoint();
        const double sx = double(dr.width())  / m_image.width();
        const double sy = double(dr.height()) / m_image.height();
        return QPoint(dr.x() + int(std::round(i.x() * sx)),
                      dr.y() + int(std::round(i.y() * sy)));
    }
    QPoint snapToAngle(const QPoint& a, const QPoint& b) const {
        const double dx = b.x() - a.x();
        const double dy = b.y() - a.y();
        const double len = std::hypot(dx, dy);
        if (len < 1.0) return b;
        const double curDeg = std::atan2(dy, dx) * 180.0 / M_PI;
        int best = m_angles.first();
        double bestDelta = 1e9;
        for (int deg : m_angles) {
            for (int sign = -1; sign <= 1; sign += 2) {
                for (int k = 0; k < 360; k += deg) {
                    double cand = sign * double(k);
                    double diff = std::abs(std::fmod(curDeg - cand + 540.0, 360.0) - 180.0);
                    if (diff < bestDelta) { bestDelta = diff; best = int(cand); }
                }
            }
        }
        const double rad = best * M_PI / 180.0;
        return QPoint(a.x() + int(std::round(len * std::cos(rad))),
                      a.y() + int(std::round(len * std::sin(rad))));
    }

    QImage m_image;
    Mode m_mode = Mode::Line;
    QList<int> m_angles{45, 90};
    bool m_dragging = false;
    bool m_hasSelection = false;
    QPoint m_a, m_b;
    std::function<void(QPoint,QPoint)> m_lineCb;
    std::function<void(QRect)>         m_rectCb;
};

class ModeToggle : public QWidget {
    Q_OBJECT
    Q_PROPERTY(qreal pos READ pos WRITE setPos)
public:
    explicit ModeToggle(QWidget* parent = nullptr) : QWidget(parent) {
        setFixedSize(120, 28);
        setCursor(Qt::PointingHandCursor);
        m_anim = new QPropertyAnimation(this, "pos", this);
        m_anim->setDuration(180);
        m_anim->setEasingCurve(QEasingCurve::OutCubic);
    }
    void setLabels(const QString& left, const QString& right) {
        m_left = left; m_right = right; update();
    }
    qreal pos() const { return m_pos; }
    void setPos(qreal p) { m_pos = p; update(); }
    int value() const { return m_target; }
    void setValue(int v, bool animated = true) {
        v = std::clamp(v, 0, 1);
        if (v == m_target) return;
        m_target = v;
        if (animated) {
            m_anim->stop();
            m_anim->setStartValue(m_pos);
            m_anim->setEndValue(qreal(v));
            m_anim->start();
        } else {
            m_pos = v;
            update();
        }
        emit valueChanged(v);
    }

signals:
    void valueChanged(int v);

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const QRectF bg = rect().adjusted(1, 1, -1, -1);
        p.setPen(QPen(QColor(255, 255, 255, 25), 1));
        p.setBrush(QColor("#1a1a1a"));
        p.drawRoundedRect(bg, height() / 2.0, height() / 2.0);

        const qreal halfW = bg.width() / 2.0;
        const QRectF knob(bg.x() + m_pos * halfW, bg.y(), halfW, bg.height());
        p.setPen(Qt::NoPen);
        p.setBrush(QColor("#7d2929"));
        p.drawRoundedRect(knob.adjusted(2, 2, -2, -2), (knob.height() - 4) / 2.0, (knob.height() - 4) / 2.0);

        QFont f = font(); f.setBold(true); p.setFont(f);
        p.setPen(QColor(m_pos < 0.5 ? "#e5e2e1" : "#a0a0a0"));
        p.drawText(QRectF(bg.x(), bg.y(), halfW, bg.height()), Qt::AlignCenter, m_left);
        p.setPen(QColor(m_pos > 0.5 ? "#e5e2e1" : "#a0a0a0"));
        p.drawText(QRectF(bg.x() + halfW, bg.y(), halfW, bg.height()), Qt::AlignCenter, m_right);
    }
    void mousePressEvent(QMouseEvent* ev) override {
        if (ev->button() != Qt::LeftButton) return;
        const qreal halfW = width() / 2.0;
        setValue(ev->position().x() < halfW ? 0 : 1, true);
    }

private:
    QString m_left = "2D";
    QString m_right = "3D";
    qreal m_pos = 0.0;
    int m_target = 0;
    QPropertyAnimation* m_anim = nullptr;
};

static cv::Mat qImageToBGR(const QImage& src) {
    if (src.isNull()) return cv::Mat();
    QImage img = src.format() == QImage::Format_Grayscale8
        ? src
        : src.convertToFormat(QImage::Format_RGB888);
    if (img.format() == QImage::Format_Grayscale8) {
        cv::Mat m(img.height(), img.width(), CV_8UC1,
                  const_cast<uchar*>(img.constBits()), img.bytesPerLine());
        return m.clone();
    }
    cv::Mat rgb(img.height(), img.width(), CV_8UC3,
                const_cast<uchar*>(img.constBits()), img.bytesPerLine());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}


namespace T {
    static const char* bg0 = "#050505";
    static const char* bg1 = "#0e0e0e";
    static const char* bg2 = "#1a1a1a";
    static const char* bg3 = "#2a2a2a";
    static const char* bgHover = "#353534";
    static const char* border = "rgba(255, 255, 255, 0.05)";
    static const char* borderStrong = "rgba(255, 255, 255, 0.1)";
    static const char* text = "#e5e2e1";
    static const char* textDim = "#c8c6c5";
    static const char* textFaint = "#b7b5b4";
    static const char* accent = "#500b0b";
    static const char* accentDim = "#450d0d";
    static const char* ok = "#ffdad6";
    static const char* warn = "#ffb3ae";
    static const char* err = "#ffb4ab";
    static const char* cam1 = "#c8c6c5";
    static const char* cam2 = "#a38b89";
}

static void setPillState(QLabel* lbl, const char* state, const QString& text = QString())
{
    if (!lbl) return;
    lbl->setProperty("role", "pill");
    lbl->setProperty("pillState", state);
    if (!text.isNull()) lbl->setText(text);
    lbl->style()->unpolish(lbl);
    lbl->style()->polish(lbl);
    lbl->update();
}

static QString settingsPath()
{
    QString dir = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
    QDir().mkpath(dir);
    return dir + "/dualcam.ini";
}

struct FilenameParamSpec {
    const char* key;
    const char* label;
};

static const FilenameParamSpec kFilenameParamSpecs[] = {
    {"BF",  "Bilateral (BF#)"},
    {"TB",  "Time Buffer (TB#)"},
    {"MT",  "Motion Threshold (MT#)"},
    {"FU",  "Fusion (FU)"},
    {"ECC", "Align (ECC)"},
    {"NF",  "Noise Floor (NF#)"},
    {"ST",  "Intensity Stretch (ST)"},
    {"G",   "Gain (G#x)"},
    {"SH",  "Shutter (SH#us)"},
    {"CM",  "Color Mode (RGB/GRY)"},
    {"FH",  "Flip Horizontal (FH)"},
    {"FV",  "Flip Vertical (FV)"},
};

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    cv::setUseOptimized(true);
    cv::setNumThreads(cv::getNumberOfCPUs());

    QApplication::setLayoutDirection(Qt::LeftToRight);

    m_worker = new CameraWorker(this);
    connect(m_worker, &CameraWorker::framesProcessed, this, &MainWindow::onFramesProcessed, Qt::QueuedConnection);
    connect(m_worker, &CameraWorker::cameraError, this, [this](const QString& msg) {
        if (m_statusBar) m_statusBar->showMessage(msg, 5000);
    });

    for (const auto& spec : kFilenameParamSpecs) {
        m_paramInName[QString::fromLatin1(spec.key)] = true;
    }

    initUI();
    initCommands();
    loadSettings();
    refreshPresetList();
}

MainWindow::~MainWindow()
{
    saveSettings();
    closeCameras();
    if (m_calibThread.joinable()) m_calibThread.join();
}

QString MainWindow::styleSheetText() const
{

    return QString(R"(
        /* Foundation */
        QMainWindow, QWidget#root { background: %1; color: %3; }
        QDialog { background: %2; color: %3; }
        QDialog QLabel { color: %3; background: transparent; }
        QWidget { color: %3; font-family: 'Inter','Segoe UI Variable','Segoe UI',sans-serif; font-size: 12px; }
        QToolTip {
            background: %6; color: %3; border: 1px solid %9; padding: 4px 8px;
            border-radius: 0px; font-size: 11px; font-weight: 500;
        }

        QMenu {
            background: %2; color: %3; border: 1px solid %7;
            padding: 4px 0px; font-size: 12px;
        }
        QMenu::item {
            background: transparent; color: %3;
            padding: 6px 22px 6px 18px;
        }
        QMenu::item:selected   { background: %6; color: %3; }
        QMenu::item:disabled   { color: %10; }
        QMenu::separator {
            height: 1px; background: %7; margin: 4px 8px;
        }
        QMenu::indicator {
            width: 14px; height: 14px; margin-left: 4px;
        }
        QMenu::indicator:non-exclusive:checked   { background: %5; border: 1px solid %5; }
        QMenu::indicator:non-exclusive:unchecked { background: transparent; border: 1px solid %9; }

        /* Typography roles */
        QLabel { color: %3; }
        QLabel[role="dim"]     { color: %4;  font-size: 11px; }
        QLabel[role="faint"]   { color: %10; font-size: 11px; }
        QLabel[role="title"]   { color: %3;  font-size: 13px; font-weight: 600; }
        QLabel[role="section"] { color: %4;  font-size: 10px; font-weight: 600; letter-spacing: 0.10em; }
        QLabel[role="hero"]    { color: %3;  font-family: 'Space Mono','JetBrains Mono',monospace; font-size: 22px; font-weight: 700; }
        QLabel[role="mono"]    { color: %3;  font-family: 'Space Mono','JetBrains Mono',monospace; font-size: 11px; }

        /* Pill (status indicator) -- sourced from Mario Guzman segmented bottom bar idiom,
           recolored for light-on-dark Adrenalin */
        QLabel[role="pill"] {
            background: %6; color: %4; border: 1px solid %7;
            border-radius: 0px; padding: 2px 8px;
            font-family: 'Space Mono','JetBrains Mono',monospace;
            font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
        }
        QLabel[role="pill"][pillState="ok"]   { color: #cde6c8; background: #143022; border-color: #1f5036; }
        QLabel[role="pill"][pillState="warn"] { color: #f5c98a; background: #2a2310; border-color: #5a4823; }
        QLabel[role="pill"][pillState="err"]  { color: #f5b3a8; background: #2a1414; border-color: %5; }
        QLabel[role="pill"][pillState="info"] { color: %3;      background: %6;      border-color: %5; }
        QLabel[role="pill"][pillState="idle"] { color: %4;      background: %6;      border-color: %7; }

        /* Frames */
        QFrame[role="card"]  { background: %6; border: 1px solid %7; border-radius: 0px; }
        QFrame[role="panel"] { background: %2; border: 1px solid %7; border-radius: 0px; }
        QFrame[role="hud"]   { background: rgba(10,10,10,210); border: 1px solid %7; border-radius: 0px; }

        /* Tabs -- top-tab navigation idiom from AMD Adrenalin (calibration.md Source 8),
           hugging-left per skill anti-pattern fix (no setExpanding) */
        QTabWidget::pane { border: none; background: %2; }
        QTabBar { background: %2; qproperty-drawBase: 0; }
        QTabBar::tab {
            background: transparent; color: %4;
            padding: 8px 18px; min-width: 78px;
            font-size: 11px; font-weight: 600;
            letter-spacing: 0.10em;
            border-top: 2px solid transparent;
            border-bottom: 2px solid transparent;
        }
        QTabBar::tab:selected         { color: %3; border-bottom: 2px solid %5; }
        QTabBar::tab:hover:!selected  { color: %3; }

        /* Buttons -- Apple HIG "Regular" approximation (tokens.md controlH 28),
           but tightened to 22 min-height for Compact tier */
        QPushButton {
            background: %6; color: %3; border: 1px solid %7; border-radius: 0px;
            padding: 5px 12px; min-height: 22px;
            font-size: 12px; font-weight: 500;
        }
        QPushButton:hover    { background: %8; border-color: %9; }
        QPushButton:pressed  { background: %7; }
        QPushButton:disabled { color: %10; background: %2; border-color: %7; }
        QPushButton:checked  { background: %11; border-color: %5; color: %3; }

        QToolButton {
            background: %6; color: %3; border: 1px solid %7; border-radius: 0px;
            padding: 5px 12px; min-height: 22px;
            font-size: 12px; font-weight: 500;
        }
        QToolButton:hover    { background: %8; border-color: %9; color: %3; }
        QToolButton:pressed  { background: %7; color: %3; }
        QToolButton:checked  { background: %11; border-color: %5; color: %3; }
        QToolButton:disabled { color: %10; background: %2; border-color: %7; }
        QToolButton::menu-indicator { image: none; }

        QPushButton[kind="primary"] {
            background: %5; color: %3; border: 1px solid %5;
            font-weight: 600; padding: 6px 18px; min-height: 24px;
        }
        QPushButton[kind="primary"]:hover    { background: %11; border-color: %11; }
        QPushButton[kind="primary"]:pressed  { background: %5; }
        QPushButton[kind="primary"]:disabled { background: %11; color: %10; border-color: %7; }

        QPushButton[kind="ghost"] {
            background: transparent; border: 1px solid transparent; color: %4;
        }
        QPushButton[kind="ghost"]:hover    { color: %3; background: %6; }
        QPushButton[kind="ghost"]:checked  { color: %3; background: %6; border-color: %5; }

        QPushButton[kind="quiet"] {
            background: transparent; border: none; color: %4;
            padding: 4px 8px;
        }
        QPushButton[kind="quiet"]:hover { color: %3; }

        /* Sheet handle -- at Mario Guzman's small bottom-bar spec (22pt) */
        QPushButton#sheetHandle {
            background: %2; border: none;
            border-top: 1px solid %7; border-bottom: 1px solid %7;
            color: %4; padding: 0; min-height: 22px; max-height: 22px;
        }
        QPushButton#sheetHandle:hover { background: %6; color: %3; }

        /* Inputs -- monospace, Adrenalin "telemetry" treatment */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background: %8; color: %3; border: 1px solid %7; border-radius: 0px;
            padding: 4px 8px; min-height: 22px;
            selection-background-color: %5; selection-color: %3;
            font-family: 'Space Mono','JetBrains Mono',monospace; font-size: 11px;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: %5; }
        QComboBox::drop-down { border: none; width: 18px; }
        QComboBox QAbstractItemView {
            background: %2; color: %3; border: 1px solid %7;
            selection-background-color: %11; selection-color: %3;
            font-family: 'Space Mono','JetBrains Mono',monospace; font-size: 11px;
        }

        /* Checkboxes -- small for Compact tier */
        QCheckBox { color: %3; spacing: 6px; font-size: 12px; }
        QCheckBox::indicator {
            width: 14px; height: 14px; border-radius: 0px;
            background: %8; border: 1px solid %9;
        }
        QCheckBox::indicator:hover    { border-color: %5; }
        QCheckBox::indicator:checked  { background: %5; border-color: %5; }
        QCheckBox::indicator:disabled { background: %2; border-color: %7; }

        /* Sliders -- rectangle handle sized to align with button baseline (22-28px),
           accent-fill sub-page. Handle height matches QPushButton min-height so
           sliders read as peers of the 2D/3D toggle rather than as decoration. */
        QSlider::horizontal { min-height: 22px; }
        QSlider::groove:horizontal { background: %8; height: 4px; border: 1px solid %7; border-radius: 0px; }
        QSlider::sub-page:horizontal { background: %5; }
        QSlider::add-page:horizontal { background: %8; }
        QSlider::handle:horizontal {
            background: %3; width: 8px; height: 22px; margin: -9px 0;
            border: 1px solid %5; border-radius: 0px;
        }
        QSlider::handle:horizontal:hover    { background: %5; border-color: %3; }
        QSlider::handle:horizontal:pressed  { background: %11; border-color: %3; }
        QSlider::handle:horizontal:disabled { background: %8; border-color: %7; }

        QSlider::vertical { min-width: 22px; }
        QSlider::groove:vertical { background: %8; width: 4px; border: 1px solid %7; border-radius: 0px; }
        QSlider::sub-page:vertical { background: %8; }
        QSlider::add-page:vertical { background: %5; }
        QSlider::handle:vertical {
            background: %3; height: 8px; width: 22px; margin: 0 -9px;
            border: 1px solid %5; border-radius: 0px;
        }
        QSlider::handle:vertical:hover { background: %5; border-color: %3; }

        /* Lists */
        QListWidget {
            background: %2; color: %3; border: 1px solid %7; border-radius: 0px;
            outline: 0; font-size: 12px;
        }
        QListWidget::item {
            padding: 6px 10px; border-bottom: 1px solid %7;
        }
        QListWidget::item:selected { background: %6; color: %3; border-left: 2px solid %5; padding-left: 8px; }
        QListWidget::item:hover    { background: %6; }

        /* Status bar — Mario small bottom-bar 22pt, Space Mono telemetry */
        QStatusBar {
            background: %2; color: %4;
            border-top: 1px solid %7;
            font-family: 'Space Mono','JetBrains Mono',monospace; font-size: 10px;
            min-height: 22px;
        }
        QStatusBar::item { border: none; }

        /* Splitter — 2px gap, DESIGN.md "pitch black background as natural border" */
        QSplitter::handle           { background: %1; }
        QSplitter::handle:horizontal { width: 2px; }
        QSplitter::handle:vertical   { height: 2px; }
        QSplitter::handle:hover      { background: %5; }

        /* Scrollbars — slim, hover-revealed accent */
        QScrollBar:vertical   { background: %2; width: 8px; }
        QScrollBar:horizontal { background: %2; height: 8px; }
        QScrollBar::handle:vertical   { background: %9; min-height: 24px; border-radius: 0px; }
        QScrollBar::handle:horizontal { background: %9; min-width:  24px; border-radius: 0px; }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: %5; }
        QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
        QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

        /* Group box (rare) */
        QGroupBox { border: 1px solid %7; border-radius: 0px; margin-top: 14px; padding: 10px; font-size: 11px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; color: %4; }
    )")
        .arg(T::bg0)
        .arg(T::bg1)
        .arg(T::text)
        .arg(T::textDim)
        .arg(T::accent)
        .arg(T::bg2)
        .arg(T::border)
        .arg(T::bg3)
        .arg(T::borderStrong)
        .arg(T::textFaint)
        .arg(T::accentDim);
}

void MainWindow::initUI()
{
    setWindowTitle("DualCam Analysis Tool");
    setMinimumSize(560, 420);
    setStyleSheet(styleSheetText());

    m_centralWidget = new QWidget(this);
    m_centralWidget->setObjectName("root");
    m_centralWidget->setFocusPolicy(Qt::ClickFocus);
    setCentralWidget(m_centralWidget);

    QVBoxLayout* rootLayout = new QVBoxLayout(m_centralWidget);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    QWidget* topStrip = new QWidget(this);
    topStrip->setFixedHeight(40);
    topStrip->setStyleSheet(QString("background:%1; border-bottom:1px solid %2;")
                                .arg(T::bg1).arg(T::border));
    QHBoxLayout* topLay = new QHBoxLayout(topStrip);
    topLay->setContentsMargins(8, 0, 8, 0);
    topLay->setSpacing(6);

    m_comboCamSet = new QComboBox(this);
    m_comboCamSet->setToolTip("Camera Resolution & FPS (Fetches from system)");
    m_comboCamSet->setMinimumWidth(160);

    refreshCameraModes();
    connect(m_comboCamSet, QOverload<int>::of(&QComboBox::activated), this, [this](int index) {
        QVariantList v = m_comboCamSet->itemData(index).toList();
        if (v.size() == 3 && v[0].toInt() == 0) {
            QDialog dlg(this);
            dlg.setWindowTitle("Custom Resolution");
            dlg.setStyleSheet(styleSheetText());
            QVBoxLayout* lay = new QVBoxLayout(&dlg);
            QSpinBox* spinW = new QSpinBox(&dlg); spinW->setRange(320, 3840); spinW->setValue(1920);
            QSpinBox* spinH = new QSpinBox(&dlg); spinH->setRange(240, 2160); spinH->setValue(1080);
            QSpinBox* spinF = new QSpinBox(&dlg); spinF->setRange(1, 240); spinF->setValue(30);
            QFormLayout* form = new QFormLayout();
            form->addRow("Width:", spinW);
            form->addRow("Height:", spinH);
            form->addRow("FPS:", spinF);
            lay->addLayout(form);
            QDialogButtonBox* box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
            connect(box, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
            connect(box, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
            lay->addWidget(box);

            if (m_animDialogsEnabled) {
                animateDialogEntry(&dlg, m_comboCamSet, m_animSpeedMs);
            }
            if (dlg.exec() == QDialog::Accepted) {
                QString label = QString("%1x%2 @ %3 FPS \u2605").arg(spinW->value()).arg(spinH->value()).arg(spinF->value());
                m_comboCamSet->setItemText(index, label);
                m_comboCamSet->setItemData(index, QVariantList{spinW->value(), spinH->value(), spinF->value()});
            }
        }
    });

    m_fpsPill = new QLabel("IDLE", this);
    m_fpsPill->setProperty("role", "pill");
    m_fpsPill->setProperty("pillState", "idle");
    m_eccPill = new QLabel("NO ALIGN", this);
    m_eccPill->setProperty("role", "pill");
    m_eccPill->setProperty("pillState", "err");

    m_btnGallery = new QPushButton("Snapshots", this);
    m_btnGallery->setCursor(Qt::PointingHandCursor);
    m_btnGallery->setToolTip("Open snapshots folder in OS");
    connect(m_btnGallery, &QPushButton::clicked, this, [this]() {
        QString dir = QCoreApplication::applicationDirPath() + "/metrics";
        QDir().mkpath(dir);
        QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
        m_statusBar->showMessage("Snapshots folder: " + dir, 4000);
    });

    m_btnHelp = new QPushButton("?", this);
    m_btnHelp->setProperty("kind", "ghost");
    m_btnHelp->setFixedWidth(28);
    m_btnHelp->setCursor(Qt::PointingHandCursor);
    m_btnHelp->setToolTip("Actions & Hotkeys Menu");
    connect(m_btnHelp, &QPushButton::clicked, this, &MainWindow::showActionsMenu);

    topLay->addWidget(m_comboCamSet);

    m_btnExpToggle = new QPushButton("AUTO", this);
    m_btnExpToggle->setCheckable(true);
    m_btnExpToggle->setCursor(Qt::PointingHandCursor);
    m_btnExpToggle->setToolTip("Toggle automatic / manual exposure");
    m_btnExpToggle->setFixedWidth(56);
    m_btnExpToggle->setStyleSheet(QString(
        "QPushButton { background: %1; color: %2; border: 1px solid %3; border-radius: 0px;"
        " padding: 3px 8px; font-weight: 700; font-size: 10px; letter-spacing: 0.10em; font-family: 'Space Mono',monospace; }"
        "QPushButton:checked { background: %4; color: %2; border-color: %4; }")
        .arg(T::bg2).arg(T::text).arg(T::border).arg(T::accent));

    auto buildGainSpin = [this](int q8) {
        auto* s = new QDoubleSpinBox(this);
        s->setRange(0.1, 10.0);
        s->setDecimals(2);
        s->setSingleStep(0.1);
        s->setSuffix("x");
        s->setValue(qBound(0.1, q8 / 256.0, 10.0));
        s->setKeyboardTracking(false);
        s->hide();
        return s;
    };
    auto buildGainSlider = [this](double initVal) {
        auto* s = new QSlider(Qt::Horizontal, this);
        s->setRange(10, 1000);
        s->setSingleStep(10);
        s->setPageStep(50);
        s->setValue(qBound(10, static_cast<int>(initVal * 100.0 + 0.5), 1000));
        s->hide();
        return s;
    };
    auto buildShutterSpin = [this](int us) {
        auto* s = new QSpinBox(this);
        s->setRange(50, 30000);
        s->setSingleStep(100);
        s->setSuffix(" us");
        s->setValue(qBound(50, us, 30000));
        s->setKeyboardTracking(false);
        s->hide();
        return s;
    };
    auto buildShutterSlider = [this](int initVal) {
        auto* s = new QSlider(Qt::Horizontal, this);
        s->setRange(50, 30000);
        s->setSingleStep(100);
        s->setPageStep(1000);
        s->setValue(qBound(50, initVal, 30000));
        s->hide();
        return s;
    };

    m_spnGain     = buildGainSpin(m_gainQ8);
    m_sldGain     = buildGainSlider(m_spnGain->value());
    m_spnShutter  = buildShutterSpin(m_shutterUs);
    m_sldShutter  = buildShutterSlider(m_spnShutter->value());

    m_spnGain2    = buildGainSpin(m_gain2Q8);
    m_sldGain2    = buildGainSlider(m_spnGain2->value());
    m_spnShutter2 = buildShutterSpin(m_shutter2Us);
    m_sldShutter2 = buildShutterSlider(m_spnShutter2->value());

    m_lblExposureVals = new QLabel(this);
    m_lblExposureVals->setProperty("role", "mono");
    m_lblExposureVals->setStyleSheet(QString(
        "QLabel { background: %1; color: %2; border: 1px solid %3; border-radius: 0px;"
        " padding: 3px 8px; font-family: 'Space Mono',monospace; font-size: 11px; }")
        .arg(T::bg2).arg(T::text).arg(T::border));
    m_lblExposureVals->setToolTip("Current gain * shutter (manual mode)");
    m_lblExposureVals->setMinimumWidth(160);
    m_lblExposureVals->setAlignment(Qt::AlignCenter);

    m_btnExpChange = new QPushButton("Detail...", this);
    m_btnExpChange->setCursor(Qt::PointingHandCursor);
    m_btnExpChange->setToolTip("Detailed gain/shutter parameters (manual mode only)");
    m_btnExpChange->setEnabled(false);
    connect(m_btnExpChange, &QPushButton::clicked, this, &MainWindow::showExposureDialog);

    auto refreshExposureLabel = [this]() {
        if (!m_lblExposureVals) return;
        double g1 = m_spnGain     ? m_spnGain->value()     : m_gainQ8    / 256.0;
        int    s1 = m_spnShutter  ? m_spnShutter->value()  : m_shutterUs;
        if (m_perCameraExposure) {
            double g2 = m_spnGain2    ? m_spnGain2->value()    : m_gain2Q8 / 256.0;
            int    s2 = m_spnShutter2 ? m_spnShutter2->value() : m_shutter2Us;
            m_lblExposureVals->setText(QString("%1x * %2 // %3x * %4")
                .arg(g1, 0, 'f', 2).arg(s1)
                .arg(g2, 0, 'f', 2).arg(s2));
        } else {
            m_lblExposureVals->setText(QString("%1x * %2")
                .arg(g1, 0, 'f', 2).arg(s1));
        }
    };
    refreshExposureLabel();

    connect(m_btnExpToggle, &QPushButton::toggled, this, [this](bool on) {
        m_manualExposure = on;
        m_btnExpToggle->setText(on ? "MAN" : "AUTO");
        const bool en = on;
        if (m_spnGain)     m_spnGain->setEnabled(en);
        if (m_spnShutter)  m_spnShutter->setEnabled(en);
        if (m_sldGain)     m_sldGain->setEnabled(en);
        if (m_sldShutter)  m_sldShutter->setEnabled(en);
        if (m_spnGain2)    m_spnGain2->setEnabled(en);
        if (m_spnShutter2) m_spnShutter2->setEnabled(en);
        if (m_sldGain2)    m_sldGain2->setEnabled(en);
        if (m_sldShutter2) m_sldShutter2->setEnabled(en);
        if (m_btnExpChange) m_btnExpChange->setEnabled(en);
        applyExposureControls();
    });

    auto wireGainPair = [this, refreshExposureLabel](QDoubleSpinBox* spn, QSlider* sld, int* backing) {
        connect(spn, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [this, sld, backing, refreshExposureLabel](double v) {
            if (sld) {
                QSignalBlocker b(sld);
                sld->setValue(qBound(sld->minimum(), static_cast<int>(v * 100.0 + 0.5), sld->maximum()));
            }
            *backing = static_cast<int>(v * 256.0 + 0.5);
            refreshExposureLabel();
            if (m_manualExposure) applyExposureControls();
        });
        connect(sld, &QSlider::valueChanged, this,
                [this, spn, backing, refreshExposureLabel](int v) {
            if (spn) {
                QSignalBlocker b(spn);
                spn->setValue(v / 100.0);
            }
            *backing = static_cast<int>((v / 100.0) * 256.0 + 0.5);
            refreshExposureLabel();
            if (m_manualExposure) applyExposureControls();
        });
    };
    auto wireShutterPair = [this, refreshExposureLabel](QSpinBox* spn, QSlider* sld, int* backing) {
        connect(spn, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [this, sld, backing, refreshExposureLabel](int v) {
            if (sld) {
                QSignalBlocker b(sld);
                sld->setValue(qBound(sld->minimum(), v, sld->maximum()));
            }
            *backing = v;
            refreshExposureLabel();
            if (m_manualExposure) applyExposureControls();
        });
        connect(sld, &QSlider::valueChanged, this,
                [this, spn, backing, refreshExposureLabel](int v) {
            if (spn) {
                QSignalBlocker b(spn);
                spn->setValue(v);
            }
            *backing = v;
            refreshExposureLabel();
            if (m_manualExposure) applyExposureControls();
        });
    };

    wireGainPair(m_spnGain,    m_sldGain,    &m_gainQ8);
    wireShutterPair(m_spnShutter, m_sldShutter, &m_shutterUs);
    wireGainPair(m_spnGain2,   m_sldGain2,   &m_gain2Q8);
    wireShutterPair(m_spnShutter2, m_sldShutter2, &m_shutter2Us);

    topLay->addSpacing(4);
    topLay->addWidget(m_btnExpToggle);
    topLay->addWidget(m_lblExposureVals);
    topLay->addWidget(m_btnExpChange);

    topLay->addStretch();
    topLay->addWidget(m_fpsPill);
    topLay->addWidget(m_eccPill);
    topLay->addSpacing(4);
    topLay->addWidget(m_btnGallery);
    topLay->addWidget(m_btnHelp);

    rootLayout->addWidget(topStrip);

    m_videoArea = new QWidget(this);
    m_videoArea->setStyleSheet(QString("background:%1;").arg(T::bg0));
    QVBoxLayout* vidLay = new QVBoxLayout(m_videoArea);
    vidLay->setContentsMargins(2, 2, 2, 2);
    vidLay->setSpacing(2);

    m_splitter = new QSplitter(Qt::Horizontal, m_videoArea);

    m_splitter->setHandleWidth(2);

    auto makeView = [this](const QString& placeholder) {
        GpuImageView* v = new GpuImageView(this);
        v->setMinimumSize(160, 120);
        v->setPlaceholder(placeholder);
        return v;
    };

    m_view1 = makeView("CAM 1");
    m_view2 = makeView("CAM 2");
    m_splitter->addWidget(m_view1);
    m_splitter->addWidget(m_view2);

    m_resultView = makeView("RESULT");
    m_resultView->hide();

    vidLay->addWidget(m_splitter, 1);
    vidLay->addWidget(m_resultView, 1);

    buildFocusChart();

    m_btnFloatingPreviewToggle = new QPushButton("[]", m_videoArea);
    m_btnFloatingPreviewToggle->setCursor(Qt::PointingHandCursor);
    m_btnFloatingPreviewToggle->setFixedSize(32, 32);
    m_btnFloatingPreviewToggle->setStyleSheet(QString(
        "QPushButton { background: rgba(10,10,10,210); color: %1; border: 1px solid %2; border-radius: 0px; font-weight: bold; font-family: 'Space Mono',monospace; }"
        "QPushButton:hover { background: rgba(26,26,26,230); color: %3; border-color: %4; }")
        .arg(T::textDim, T::border, T::text, T::accent));
    m_btnFloatingPreviewToggle->setToolTip("Show Sidebar");
    connect(m_btnFloatingPreviewToggle, &QPushButton::clicked, this, &MainWindow::togglePreviewMode);
    m_btnFloatingPreviewToggle->hide();

    m_btnFabStream = new QPushButton(m_videoArea);
    m_btnFabStream->setText("");
    m_btnFabStream->setFixedSize(40, 40);
    m_btnFabStream->setCursor(Qt::PointingHandCursor);
    m_btnFabStream->setToolTip("Start streaming");
    m_btnFabStream->setStyleSheet(QString(
        "QPushButton { background: rgba(20,20,20,230); border: 1px solid %1; border-radius: 0px; }"
        "QPushButton:hover { background: rgba(40,12,12,230); border: 1px solid %2; }")
        .arg(T::border).arg(T::accent));
    auto addFabIcon = [](QPushButton* btn, const QString& text, const QString& style) -> QLabel* {
        QLabel* icon = new QLabel(text, btn);
        icon->setAttribute(Qt::WA_TransparentForMouseEvents);
        icon->setAlignment(Qt::AlignCenter);
        icon->setStyleSheet(style);
        QVBoxLayout* lay = new QVBoxLayout(btn);
        lay->setContentsMargins(0, 0, 0, 0);
        lay->addWidget(icon, 0, Qt::AlignCenter);
        return icon;
    };
    m_fabStreamIcon = addFabIcon(m_btnFabStream, QString::fromUtf8("\u25B6"), QString(
        "QLabel { background: transparent; color: %1; border: none;"
        " font-family: 'Segoe UI Symbol','Segoe UI Emoji','Arial Unicode MS';"
        " font-size: 14px; font-weight: 700; }").arg(T::text));
    connect(m_btnFabStream, &QPushButton::clicked, this, [this]() {
        if (m_camerasOpen) closeCameras();
        else openCameras();
    });
    m_btnToggleCameras = m_btnFabStream;

    m_btnModeToggle = new QPushButton("DUAL", m_videoArea);
    m_btnModeToggle->setFixedHeight(26);
    m_btnModeToggle->setCursor(Qt::PointingHandCursor);
    m_btnModeToggle->setStyleSheet(QString(
        "QPushButton { background: %1; color: %2; border: 1px solid %3;"
        " border-radius: 0px; padding: 3px 18px; font-weight: 700; font-size: 10px;"
        " letter-spacing: 0.12em; min-width: 68px; font-family: 'Space Mono',monospace; }"
        "QPushButton:hover { background: %4; }")
        .arg(T::accent).arg(T::text).arg(T::accent).arg(T::accentDim));
    connect(m_btnModeToggle, &QPushButton::clicked, this, [this]() {
        setDiffMode(!m_isDiffMode);
    });

    m_btnFabSnapshot = new QPushButton(m_videoArea);
    m_btnFabSnapshot->setText("");
    m_btnFabSnapshot->setFixedSize(40, 40);
    m_btnFabSnapshot->setCursor(Qt::PointingHandCursor);
    m_btnFabSnapshot->setToolTip("Save snapshot");
    m_btnFabSnapshot->setStyleSheet(QString(
        "QPushButton { background: %1; border: 1px solid %2; border-radius: 0px; }"
        "QPushButton:hover { background: %3; border-color: %4; }"
        "QPushButton:pressed { background: %2; }")
        .arg(T::accent).arg(T::accentDim).arg(T::accentDim).arg(T::text));
    m_fabSnapIcon = addFabIcon(m_btnFabSnapshot, "SNAP", QString(
        "QLabel { background: transparent; color: %1; border: none;"
        " font-family: 'Space Mono','JetBrains Mono',monospace;"
        " font-size: 11px; font-weight: 800; letter-spacing: 0.10em; }").arg(T::text));
    connect(m_btnFabSnapshot, &QPushButton::clicked, this, [this]() {
        saveSnapshot();
        refreshSnapshotPreview();
    });

    m_btnFabStream->hide();
    m_btnModeToggle->hide();
    m_btnFabSnapshot->hide();

    m_videoArea->installEventFilter(this);

    m_mainSplit = new QSplitter(Qt::Horizontal, this);
    m_mainSplit->setHandleWidth(2);
    m_mainSplit->setChildrenCollapsible(false);

    m_sideNav = buildSideNav();
    m_mainSplit->addWidget(m_sideNav);

    m_workStack = new QStackedWidget(this);

    m_workSplit = new QSplitter(Qt::Vertical, m_workStack);
    m_workSplit->setHandleWidth(2);
    m_workSplit->setChildrenCollapsible(false);
    m_workSplit->setStyleSheet(QString("QSplitter::handle { background: %1; }").arg(T::accent));
    m_workSplit->addWidget(m_videoArea);

    m_bottomStack = new QStackedWidget(m_workSplit);
    m_bottomStack->addWidget(buildCaptureTab());
    m_bottomStack->addWidget(buildPipelineTab());
    m_bottomStack->addWidget(buildDiffTab());
    m_focusDataWidget = buildFocusDataPanel();
    m_bottomStack->addWidget(m_focusDataWidget);
    m_bottomStack->addWidget(buildSnapshotTab());
    m_bottomStack->addWidget(buildSettingsTab());
    m_workSplit->addWidget(m_bottomStack);

    m_workSplit->setStretchFactor(0, 1);
    m_workSplit->setStretchFactor(1, 0);

    m_workStack->addWidget(m_workSplit);

    m_galleryView = buildGalleryView();
    m_workStack->addWidget(m_galleryView);

    connect(m_bottomStack, &QStackedWidget::currentChanged, this, [this](int index) {
        QWidget* w = m_bottomStack->widget(index);
        if (!w) return;
        if (!m_animTabsEnabled) {
            QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(w->graphicsEffect());
            if (effect) effect->setOpacity(1.0);
            return;
        }
        QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(w->graphicsEffect());
        if (!effect) {
            effect = new QGraphicsOpacityEffect(w);
            w->setGraphicsEffect(effect);
        }
        QPropertyAnimation* anim = new QPropertyAnimation(effect, "opacity", w);
        anim->setDuration(m_animSpeedMs * 8 / 10);
        anim->setStartValue(0.0);
        anim->setEndValue(1.0);
        anim->setEasingCurve(QEasingCurve::OutQuad);
        anim->start(QAbstractAnimation::DeleteWhenStopped);
    });

    connect(m_workStack, &QStackedWidget::currentChanged, this, [this](int index) {
        QWidget* w = m_workStack->widget(index);
        if (!w) return;
        if (!m_animTabsEnabled) {
            QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(w->graphicsEffect());
            if (effect) effect->setOpacity(1.0);
            return;
        }
        QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(w->graphicsEffect());
        if (!effect) {
            effect = new QGraphicsOpacityEffect(w);
            w->setGraphicsEffect(effect);
        }
        QPropertyAnimation* anim = new QPropertyAnimation(effect, "opacity", w);
        anim->setDuration(m_animSpeedMs);
        anim->setStartValue(0.0);
        anim->setEndValue(1.0);
        anim->setEasingCurve(QEasingCurve::OutQuad);
        anim->start(QAbstractAnimation::DeleteWhenStopped);
    });

    m_mainSplit->addWidget(m_workStack);
    m_mainSplit->setStretchFactor(0, 0);
    m_mainSplit->setStretchFactor(1, 1);

    QTimer::singleShot(0, this, [this]() {
        m_mainSplit->setSizes({200, std::max(600, width() - 200)});
        m_workSplit->setSizes({height(), 1});
    });

    rootLayout->addWidget(m_mainSplit, 1);

    setNavItem(NavItem::Capture);
    applyNavLabels();

    buildAlignDialog();

    m_statusBar = new QStatusBar(this);
    setStatusBar(m_statusBar);
    m_statusBar->showMessage("Ready. Press \u25B6 to open cameras.");

    QTimer::singleShot(0, this, [this]() {
        positionFloatingButtons();
    });
}

QWidget* MainWindow::buildCaptureTab()
{
    QWidget* page = new QWidget(this);
    QGridLayout* grid = new QGridLayout(page);

    grid->setContentsMargins(14, 12, 14, 12);
    grid->setHorizontalSpacing(20);
    grid->setVerticalSpacing(6);

    auto sectionLabel = [this](const QString& t) { return makeSectionLabel(t, this); };

    grid->addWidget(sectionLabel("COLOR"), 0, 0);
    m_comboColorMode = new QComboBox(this);
    m_comboColorMode->addItems({ "Gray Native", "Gray CV", "Color" });
    m_comboColorMode->setCurrentIndex(1);
    connect(m_comboColorMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int i) {
        m_colorMode = static_cast<ColorMode>(i);
        pushWorkerParams();
    });
    grid->addWidget(m_comboColorMode, 1, 0);

    grid->addWidget(sectionLabel("FLIP CAM2"), 0, 1);
    QWidget* flipBox = new QWidget(this);
    QHBoxLayout* flipLay = new QHBoxLayout(flipBox);
    flipLay->setContentsMargins(0, 0, 0, 0);
    m_chkFlipHor2 = new QCheckBox("Horizontal", this);
    m_chkFlipVer2 = new QCheckBox("Vertical", this);
    connect(m_chkFlipHor2, &QCheckBox::stateChanged, this, [this](int) { pushWorkerParams(); });
    connect(m_chkFlipVer2, &QCheckBox::stateChanged, this, [this](int) { pushWorkerParams(); });
    flipLay->addWidget(m_chkFlipHor2);
    flipLay->addWidget(m_chkFlipVer2);
    flipLay->addStretch();
    grid->addWidget(flipBox, 1, 1);

    grid->addWidget(sectionLabel("BILATERAL FILTER"), 0, 2);
    QWidget* bfBox = new QWidget(this);
    QHBoxLayout* bfLay = new QHBoxLayout(bfBox);
    bfLay->setContentsMargins(0, 0, 0, 0);
    bfLay->setSpacing(6);
    m_chkBilateral = new QCheckBox("On", this);
    connect(m_chkBilateral, &QCheckBox::stateChanged, this, [this](int) { pushWorkerParams(); });
    m_bilateralSlider = new QSlider(Qt::Horizontal, this);
    m_bilateralSlider->setRange(1, 20);
    m_bilateralSlider->setValue(m_bilateralStrength);
    m_bilateralSlider->setToolTip("Filter strength (radius & sigma)");
    m_bilateralLabel = new QLabel(QString::number(m_bilateralStrength), this);
    m_bilateralLabel->setFixedWidth(28);
    m_bilateralLabel->setAlignment(Qt::AlignCenter);
    connect(m_bilateralSlider, &QSlider::valueChanged, this, [this](int v) {
        m_bilateralStrength = v;
        m_bilateralLabel->setText(QString::number(v));
        pushWorkerParams();
    });
    bfLay->addWidget(m_chkBilateral);
    bfLay->addWidget(m_bilateralSlider, 1);
    bfLay->addWidget(m_bilateralLabel);
    grid->addWidget(bfBox, 1, 2);

    grid->addWidget(sectionLabel("VIEW"), 2, 0);
    m_chkStretchView = new QCheckBox("Adaptive view", this);
    m_chkStretchView->setToolTip("Scale images to fill the viewport (maintains aspect ratio, crops edges)");
    connect(m_chkStretchView, &QCheckBox::stateChanged, this, [this](int state) {
        bool stretch = (state == Qt::Checked);
        if (m_view1) m_view1->setStretch(stretch);
        if (m_view2) m_view2->setStretch(stretch);
        if (m_resultView) m_resultView->setStretch(stretch);
    });
    grid->addWidget(m_chkStretchView, 3, 0);

    grid->setColumnStretch(0, 1);
    grid->setColumnStretch(1, 1);
    grid->setColumnStretch(2, 1);
    grid->setRowStretch(4, 1);
    return page;
}

QWidget* MainWindow::buildPipelineTab()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* root = new QVBoxLayout(page);
    root->setContentsMargins(14, 12, 14, 12);
    root->setSpacing(10);

    QGridLayout* row1 = new QGridLayout();
    row1->setHorizontalSpacing(20);
    row1->setVerticalSpacing(4);

    auto sectionLabel = [this](const QString& t) { return makeSectionLabel(t, this); };

    row1->addWidget(sectionLabel("T-BUFFER"), 0, 0);
    QWidget* bufBox = new QWidget(this);
    QHBoxLayout* bufLay = new QHBoxLayout(bufBox);
    bufLay->setContentsMargins(0, 0, 0, 0);
    m_bufferSlider = new QSlider(Qt::Horizontal, this);
    m_bufferSlider->setRange(1, 30);
    m_bufferSlider->setValue(m_bufferSize);
    m_bufferLabel = new QLabel(QString::number(m_bufferSize), this);
    m_bufferLabel->setFixedWidth(28);
    m_bufferLabel->setAlignment(Qt::AlignCenter);
    connect(m_bufferSlider, &QSlider::valueChanged, this, [this](int v) {
        m_bufferSize = v;
        m_bufferLabel->setText(QString::number(v));
        pushWorkerParams();
    });
    bufLay->addWidget(m_bufferSlider, 1);
    bufLay->addWidget(m_bufferLabel);
    row1->addWidget(bufBox, 1, 0);

    row1->addWidget(sectionLabel("MOTION THR"), 0, 1);
    QWidget* motBox = new QWidget(this);
    QHBoxLayout* motLay = new QHBoxLayout(motBox);
    motLay->setContentsMargins(0, 0, 0, 0);
    m_motionThresholdSlider = new QSlider(Qt::Horizontal, this);
    m_motionThresholdSlider->setRange(1, 50);
    m_motionThresholdSlider->setValue(static_cast<int>(m_motionThreshold * 100));
    m_motionThresholdLabel = new QLabel(QString("%1%").arg(m_motionThreshold * 100, 0, 'f', 0), this);
    m_motionThresholdLabel->setFixedWidth(40);
    m_motionThresholdLabel->setAlignment(Qt::AlignCenter);
    connect(m_motionThresholdSlider, &QSlider::valueChanged, this, [this](int v) {
        m_motionThreshold = static_cast<double>(v) / 100.0;
        m_motionThresholdLabel->setText(QString("%1%").arg(v));
        pushWorkerParams();
    });
    motLay->addWidget(m_motionThresholdSlider, 1);
    motLay->addWidget(m_motionThresholdLabel);
    row1->addWidget(motBox, 1, 1);

    row1->addWidget(sectionLabel("FUSION"), 0, 2);
    QWidget* fuseBox = new QWidget(this);
    QHBoxLayout* fuseLay = new QHBoxLayout(fuseBox);
    fuseLay->setContentsMargins(0, 0, 0, 0);
    m_chkFusion = new QCheckBox("Blend CAM1+CAM2", this);
    m_motionIndicator = new QLabel(QString::fromUtf8("\u25CF Idle"), this);
    m_motionIndicator->setStyleSheet(QString("color:%1; font-weight:600;").arg(T::textDim));
    fuseLay->addWidget(m_chkFusion);
    fuseLay->addStretch();
    fuseLay->addWidget(m_motionIndicator);
    row1->addWidget(fuseBox, 1, 2);

    row1->setColumnStretch(0, 1);
    row1->setColumnStretch(1, 1);
    row1->setColumnStretch(2, 1);
    root->addLayout(row1);

    QFrame* eccCard = new QFrame(this);
    eccCard->setProperty("role", "card");
    QVBoxLayout* eccLay = new QVBoxLayout(eccCard);
    eccLay->setContentsMargins(12, 10, 12, 12);
    eccLay->setSpacing(10);

    QHBoxLayout* eccHeader = new QHBoxLayout();
    QLabel* eccTitle = new QLabel("ALIGNMENT", this);
    eccTitle->setProperty("role", "section");
    m_eccIndicator = new QLabel("NOT CALIBRATED", this);
    setPillState(m_eccIndicator, "err");
    eccHeader->addWidget(eccTitle);
    eccHeader->addWidget(m_eccIndicator);
    eccHeader->addStretch();
    eccLay->addLayout(eccHeader);

    QHBoxLayout* eccButtons = new QHBoxLayout();
    eccButtons->setSpacing(6);
    m_btnCalibrateAlign = new QPushButton("Calibrate", this);
    m_btnCalibrateAlign->setMinimumHeight(28);
    connect(m_btnCalibrateAlign, &QPushButton::clicked, this, &MainWindow::calibrateAlignment);

    m_chkAlign = new QCheckBox("Apply to CAM2", this);
    m_chkAlign->setMinimumHeight(28);
    connect(m_chkAlign, &QCheckBox::stateChanged, this, &MainWindow::updateView);

    m_btnOpenManualAlign = new QPushButton("Manual 6-DOF...", this);
    m_btnOpenManualAlign->setMinimumHeight(28);
    connect(m_btnOpenManualAlign, &QPushButton::clicked, this, [this]() {
        if (m_alignDialog) {
            applyAdjustToWidgets();
            if (m_animDialogsEnabled) {
                animateDialogEntry(m_alignDialog, m_btnOpenManualAlign, m_animSpeedMs);
            } else {
                m_alignDialog->show();
                m_alignDialog->raise();
            }
        }
    });

    eccButtons->addWidget(m_btnCalibrateAlign, 1);
    eccButtons->addWidget(m_chkAlign, 1);
    eccButtons->addWidget(m_btnOpenManualAlign, 1);
    eccLay->addLayout(eccButtons);

    QLabel* eccHint = new QLabel(
        "Calibrate solves the warp matrix once (SIFT + FLANN + RANSAC). "
        "Apply reuses that matrix on every frame.",
        this);
    eccHint->setWordWrap(true);
    eccHint->setProperty("role", "faint");
    eccLay->addWidget(eccHint);

    root->addWidget(eccCard);
    root->addStretch();
    return page;
}

QWidget* MainWindow::buildDiffTab()
{
    QWidget* page = new QWidget(this);
    QGridLayout* grid = new QGridLayout(page);
    grid->setContentsMargins(14, 12, 14, 12);
    grid->setHorizontalSpacing(20);
    grid->setVerticalSpacing(6);

    auto sectionLabel = [this](const QString& t) { return makeSectionLabel(t, this); };

    grid->addWidget(sectionLabel("NOISE FLOOR"), 0, 0);
    QWidget* nfBox = new QWidget(this);
    QHBoxLayout* nfLay = new QHBoxLayout(nfBox);
    nfLay->setContentsMargins(0, 0, 0, 0);
    m_noiseFloorSlider = new QSlider(Qt::Horizontal, this);
    m_noiseFloorSlider->setRange(0, 50);
    m_noiseFloorSlider->setValue(m_noiseFloor);
    m_noiseFloorLabel = new QLabel(QString::number(m_noiseFloor), this);
    m_noiseFloorLabel->setFixedWidth(28);
    m_noiseFloorLabel->setAlignment(Qt::AlignCenter);
    connect(m_noiseFloorSlider, &QSlider::valueChanged, this, [this](int v) {
        m_noiseFloor = v;
        m_noiseFloorLabel->setText(QString::number(v));
        pushWorkerParams();
    });
    nfLay->addWidget(m_noiseFloorSlider, 1);
    nfLay->addWidget(m_noiseFloorLabel);
    grid->addWidget(nfBox, 1, 0);

    grid->addWidget(sectionLabel("STRETCH"), 0, 1);
    m_chkStretch = new QCheckBox("Stretch intensity range", this);
    grid->addWidget(m_chkStretch, 1, 1);

    grid->addWidget(sectionLabel("PEAKS"), 0, 2);
    m_btnPeakIntensities = new QPushButton("Track peaks", this);
    m_btnPeakIntensities->setCheckable(true);
    connect(m_btnPeakIntensities, &QPushButton::toggled, this, [this](bool checked) {
        if (!checked) m_lblPeakInfo->clear();
        updateView();
    });
    grid->addWidget(m_btnPeakIntensities, 1, 2);

    m_lblPeakInfo = new QLabel("", this);
    m_lblPeakInfo->setStyleSheet(QString("color:%1; font-family:'Space Mono',monospace;").arg(T::warn));
    grid->addWidget(m_lblPeakInfo, 2, 0, 1, 3);

    grid->setColumnStretch(0, 1);
    grid->setColumnStretch(1, 1);
    grid->setColumnStretch(2, 1);
    grid->setRowStretch(3, 1);
    return page;
}

void MainWindow::buildFocusChart()
{
    m_chart = new QChart();
    m_chart->setTitle("");
    m_chart->setBackgroundBrush(QBrush(QColor(T::bg1)));
    m_chart->setPlotAreaBackgroundBrush(QBrush(QColor(T::bg2)));
    m_chart->setPlotAreaBackgroundVisible(true);
    m_chart->setTitleBrush(QBrush(QColor(T::text)));
    m_chartView = new QChartView(m_chart);
    m_chartView->setRenderHint(QPainter::Antialiasing);
    m_chartView->setMinimumHeight(160);
    m_chartView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    m_seriesCam1 = new QLineSeries();
    m_seriesCam1->setName("Camera 1 Focus");
    m_seriesCam1->setColor(QColor(T::cam1));
    m_seriesCam2 = new QLineSeries();
    m_seriesCam2->setName("Camera 2 Focus");
    m_seriesCam2->setColor(QColor(T::cam2));

    m_chart->addSeries(m_seriesCam1);
    m_chart->addSeries(m_seriesCam2);

    QValueAxis* axisX = new QValueAxis;
    axisX->setTitleText("Time (Frames)");
    axisX->setRange(0, m_maxHistory);
    axisX->setTickCount(10);
    axisX->setLabelsColor(QColor(T::text));
    axisX->setTitleBrush(QBrush(QColor(T::textDim)));
    axisX->setGridLineColor(QColor(T::border));
    m_chart->addAxis(axisX, Qt::AlignBottom);

    QValueAxis* axisY = new QValueAxis;
    axisY->setTitleText("Focus Score");

    axisY->setRange(0, 100);
    axisY->setLabelsColor(QColor(T::text));
    axisY->setTitleBrush(QBrush(QColor(T::textDim)));
    axisY->setGridLineColor(QColor(T::border));
    m_chart->addAxis(axisY, Qt::AlignLeft);

    m_seriesCam1->attachAxis(axisX);
    m_seriesCam1->attachAxis(axisY);
    m_seriesCam2->attachAxis(axisX);
    m_seriesCam2->attachAxis(axisY);

    m_chart->legend()->setVisible(true);
    m_chart->legend()->setAlignment(Qt::AlignTop);
    m_chart->legend()->setLabelColor(QColor(T::text));
}

QWidget* MainWindow::buildFocusDataPanel()
{
    QWidget* panel = new QWidget(this);
    panel->setStyleSheet(QString("background: %1;").arg(T::bg1));
    QVBoxLayout* rootLay = new QVBoxLayout(panel);
    rootLay->setContentsMargins(0, 0, 0, 0);
    rootLay->setSpacing(0);

    m_chartView->setParent(panel);
    m_chartView->show();
    rootLay->addWidget(m_chartView, 1);

    QWidget* cardsWidget = new QWidget(panel);
    QHBoxLayout* lay = new QHBoxLayout(cardsWidget);
    lay->setContentsMargins(14, 10, 14, 10);
    lay->setSpacing(8);

    auto makeCard = [this](const QString& title, const char* titleColor, QLabel*& valueLabel) {
        QFrame* card = new QFrame(this);
        card->setProperty("role", "card");
        QVBoxLayout* cardLay = new QVBoxLayout(card);
        cardLay->setContentsMargins(10, 8, 10, 8);
        cardLay->setSpacing(2);
        QLabel* t = new QLabel(title, card);

        t->setStyleSheet(QString(
            "color:%1; font-size:10px; font-weight:600; letter-spacing:0.10em;").arg(titleColor));
        valueLabel = new QLabel("0", card);
        valueLabel->setProperty("role", "hero");
        cardLay->addWidget(t);
        cardLay->addWidget(valueLabel);
        return card;
    };

    lay->addWidget(makeCard("CAM1 FOCUS", T::cam1, m_lblFocus1Big), 1);
    lay->addWidget(makeCard("CAM2 FOCUS", T::cam2, m_lblFocus2Big), 1);

    QFrame* histCard = new QFrame(this);
    histCard->setProperty("role", "card");
    QVBoxLayout* histLay = new QVBoxLayout(histCard);
    histLay->setContentsMargins(10, 8, 10, 8);
    histLay->setSpacing(6);
    QLabel* histTitle = new QLabel("GRAPH HISTORY", histCard);
    histTitle->setProperty("role", "section");
    histLay->addWidget(histTitle);

    QHBoxLayout* histControls = new QHBoxLayout();
    histControls->setSpacing(8);
    m_historySlider = new QSlider(Qt::Horizontal, this);
    m_historySlider->setRange(50, 2000);
    m_historySlider->setSingleStep(50);
    m_historySlider->setPageStep(50);
    m_historySlider->setValue(m_maxHistory);
    m_historySpinBox = new QSpinBox(this);
    m_historySpinBox->setRange(50, 2000);
    m_historySpinBox->setSingleStep(50);
    m_historySpinBox->setValue(m_maxHistory);
    m_historySpinBox->setSuffix(" f");
    m_historySpinBox->setFixedWidth(80);

    auto applyHistory = [this](int value) {
        int snapped = ((value + 25) / 50) * 50;
        if (snapped < 50) snapped = 50;
        if (snapped > 2000) snapped = 2000;
        m_maxHistory = snapped;
        { QSignalBlocker a(m_historySlider); m_historySlider->setValue(snapped); }
        { QSignalBlocker b(m_historySpinBox); m_historySpinBox->setValue(snapped); }
        if (m_camerasOpen) {
            auto ax = m_chart->axes(Qt::Horizontal).first();
            ax->setRange(std::max(0LL, m_frameCount - m_maxHistory), m_frameCount);
        } else {
            m_chart->axes(Qt::Horizontal).first()->setRange(0, m_maxHistory);
        }
    };
    connect(m_historySlider, &QSlider::valueChanged, this, applyHistory);
    connect(m_historySpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, applyHistory);

    histControls->addWidget(m_historySlider, 1);
    histControls->addWidget(m_historySpinBox);
    histLay->addLayout(histControls);

    lay->addWidget(histCard, 1);
    rootLay->addWidget(cardsWidget, 0);
    return panel;
}

QWidget* MainWindow::buildSnapshotTab()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* root = new QVBoxLayout(page);
    root->setContentsMargins(14, 12, 14, 12);
    root->setSpacing(8);

    QFrame* saveCard = new QFrame(this);
    saveCard->setProperty("role", "card");
    QHBoxLayout* saveLay = new QHBoxLayout(saveCard);
    saveLay->setContentsMargins(10, 8, 10, 8);
    saveLay->setSpacing(8);

    QLabel* nameLabel = new QLabel("NAME", this);
    nameLabel->setProperty("role", "section");
    saveLay->addWidget(nameLabel);

    m_snapshotNameEdit = new QLineEdit(this);
    m_snapshotNameEdit->setPlaceholderText("Auto-generated...");
    saveLay->addWidget(m_snapshotNameEdit, 1);

    m_chkAppendParams = new QCheckBox("Append params", this);
    m_chkAppendParams->setToolTip("Add active filter values to filename (e.g. _BF5_NF15)");
    saveLay->addWidget(m_chkAppendParams);

    m_btnConfigParams = new QPushButton("Configure...", this);
    m_btnConfigParams->setProperty("kind", "ghost");
    m_btnConfigParams->setToolTip("Choose which parameters are written into the filename");
    m_btnConfigParams->setCursor(Qt::PointingHandCursor);
    connect(m_btnConfigParams, &QPushButton::clicked, this, &MainWindow::showFilenameParamsDialog);
    saveLay->addWidget(m_btnConfigParams);

    m_btnSaveSnapshot = new QPushButton("Save", this);
    m_btnSaveSnapshot->setProperty("kind", "primary");
    m_btnSaveSnapshot->setMinimumHeight(28);
    connect(m_btnSaveSnapshot, &QPushButton::clicked, this, [this]() {
        saveSnapshot();
        refreshSnapshotPreview();
    });
    saveLay->addWidget(m_btnSaveSnapshot);

    root->addWidget(saveCard);

    QHBoxLayout* recentHead = new QHBoxLayout();
    QLabel* recentTitle = new QLabel("RECENT", this);
    recentTitle->setProperty("role", "section");
    recentHead->addWidget(recentTitle);
    recentHead->addStretch();
    QPushButton* btnSeeAll = new QPushButton("See all", this);
    btnSeeAll->setProperty("kind", "ghost");
    connect(btnSeeAll, &QPushButton::clicked, this, [this]() {
        setNavItem(NavItem::Gallery);
    });
    recentHead->addWidget(btnSeeAll);
    root->addLayout(recentHead);

    m_snapshotRecent = new QListWidget(this);
    m_snapshotRecent->setViewMode(QListWidget::IconMode);
    m_snapshotRecent->setIconSize(QSize(110, 80));
    m_snapshotRecent->setGridSize(QSize(126, 110));
    m_snapshotRecent->setResizeMode(QListWidget::Adjust);
    m_snapshotRecent->setFlow(QListWidget::LeftToRight);
    m_snapshotRecent->setWrapping(false);
    m_snapshotRecent->setUniformItemSizes(true);
    m_snapshotRecent->setSpacing(6);
    m_snapshotRecent->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_snapshotRecent->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_snapshotRecent->setFixedHeight(120);
    m_snapshotRecent->setTextElideMode(Qt::ElideMiddle);
    m_snapshotRecent->setStyleSheet(QString(
        "QListWidget { background: %1; border: 1px solid %2; border-radius: 0px; }"
        "QListWidget::item { padding: 4px; border-bottom: none; }"
        "QListWidget::item:selected { background: %3; border-left: 2px solid %4; padding-left: 2px; }"
        "QListWidget::item:hover { background: %3; }")
        .arg(T::bg1).arg(T::border).arg(T::bg2).arg(T::accent));
    root->addWidget(m_snapshotRecent);

    connect(m_snapshotRecent, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* item) {
        if (!item) return;
        const QString fileBase = item->text();
        const QString imagePath = QCoreApplication::applicationDirPath() + "/metrics/" + fileBase;
        openPreviewWindow(imagePath, fileBase);
    });

    m_snapshotRecent->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_snapshotRecent, &QListWidget::customContextMenuRequested, this, [this](const QPoint& pos) {
        QListWidgetItem* item = m_snapshotRecent->itemAt(pos);
        if (!item) return;
        QMenu menu(this);
        menu.setStyleSheet(styleSheetText());
        QAction* openAct = menu.addAction("Preview");
        QAction* renameAct = menu.addAction("Rename");
        QAction* dltAct = menu.addAction("Delete");
        QAction* res = menu.exec(m_snapshotRecent->mapToGlobal(pos));
        if (!res) return;
        QString dir = QCoreApplication::applicationDirPath() + "/metrics/";
        QString oldPath = dir + item->text();
        if (res == openAct) {
            emit m_snapshotRecent->itemDoubleClicked(item);
        } else if (res == renameAct) {
            bool ok;
            QString newName = QInputDialog::getText(this, "Rename", "New name:", QLineEdit::Normal, item->text(), &ok);
            if (ok && !newName.isEmpty()) {
                QString newPath = dir + newName;
                QFile::rename(oldPath, newPath);
                refreshSnapshotPreview();
            }
        } else if (res == dltAct) {
            if (QMessageBox::question(this, "Delete", "Delete " + item->text() + "?") == QMessageBox::Yes) {
                QFile::remove(oldPath);
                refreshSnapshotPreview();
            }
        }
    });

    root->addStretch(1);
    refreshSnapshotPreview();
    return page;
}

QWidget* MainWindow::buildSettingsTab()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* root = new QVBoxLayout(page);
    root->setContentsMargins(14, 14, 14, 14);
    root->setSpacing(10);

    // Left card for Presets save
    QFrame* saveCard = new QFrame(this);
    saveCard->setProperty("role", "card");
    QHBoxLayout* saveLay = new QHBoxLayout(saveCard);
    saveLay->setContentsMargins(10, 10, 10, 10);
    saveLay->setSpacing(8);

    m_presetNameEdit = new QLineEdit(this);
    m_presetNameEdit->setPlaceholderText("New preset name\u2026");
    m_btnSavePreset = new QPushButton("Save current", this);
    m_btnSavePreset->setProperty("kind", "primary");
    auto saveNow = [this]() {
        QString name = m_presetNameEdit->text().trimmed();
        if (name.isEmpty()) name = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        savePreset(name);
        m_presetNameEdit->clear();
        m_statusBar->showMessage("Preset '" + name + "' saved.", 3000);
    };
    connect(m_btnSavePreset, &QPushButton::clicked, this, saveNow);
    connect(m_presetNameEdit, &QLineEdit::returnPressed, this, saveNow);

    saveLay->addWidget(m_presetNameEdit, 1);
    saveLay->addWidget(m_btnSavePreset);

    m_presetList = new QListWidget(this);
    m_presetList->setSelectionMode(QAbstractItemView::SingleSelection);

    m_btnLoadPreset = new QPushButton("Load", this);
    m_btnDeletePreset = new QPushButton("Delete", this);
    m_btnDeletePreset->setProperty("kind", "ghost");

    connect(m_btnLoadPreset, &QPushButton::clicked, this, [this]() {
        QListWidgetItem* it = m_presetList->currentItem();
        if (it) loadPreset(it->text());
    });
    connect(m_btnDeletePreset, &QPushButton::clicked, this, [this]() {
        QListWidgetItem* it = m_presetList->currentItem();
        if (it) deletePreset(it->text());
    });
    connect(m_presetList, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* it) {
        if (it) loadPreset(it->text());
    });

    QHBoxLayout* listRow = new QHBoxLayout();
    listRow->setSpacing(8);
    listRow->addWidget(m_presetList, 1);
    QVBoxLayout* listBtnCol = new QVBoxLayout();
    listBtnCol->setSpacing(6);
    listBtnCol->addWidget(m_btnLoadPreset);
    listBtnCol->addWidget(m_btnDeletePreset);
    listBtnCol->addStretch();
    listRow->addLayout(listBtnCol);

    // Right card for Animations
    QFrame* animCard = new QFrame(this);
    animCard->setProperty("role", "card");
    QVBoxLayout* animLay = new QVBoxLayout(animCard);
    animLay->setContentsMargins(14, 14, 14, 14);
    animLay->setSpacing(8);

    QLabel* animTitle = new QLabel("Animation Options", animCard);
    animTitle->setStyleSheet(QString("font-weight: bold; font-size: 13px; color: %1;").arg(T::accent));
    animLay->addWidget(animTitle);

    QHBoxLayout* speedRow = new QHBoxLayout();
    QLabel* speedLbl = new QLabel("Transition Speed:", animCard);
    QSlider* speedSlider = new QSlider(Qt::Horizontal, animCard);
    speedSlider->setRange(50, 1000);
    speedSlider->setValue(m_animSpeedMs);
    QLabel* speedValLbl = new QLabel(QString("%1 ms").arg(m_animSpeedMs), animCard);
    speedValLbl->setFixedWidth(60);
    
    connect(speedSlider, &QSlider::valueChanged, this, [this, speedValLbl](int val) {
        m_animSpeedMs = val;
        speedValLbl->setText(QString("%1 ms").arg(val));
    });
    
    speedRow->addWidget(speedLbl);
    speedRow->addWidget(speedSlider, 1);
    speedRow->addWidget(speedValLbl);
    animLay->addLayout(speedRow);

    QCheckBox* chkSidebar = new QCheckBox("Animate Sidebar", animCard);
    chkSidebar->setChecked(m_animSidebarEnabled);
    connect(chkSidebar, &QCheckBox::toggled, this, [this](bool checked) { m_animSidebarEnabled = checked; });
    animLay->addWidget(chkSidebar);

    QCheckBox* chkFABs = new QCheckBox("Animate Floating Buttons", animCard);
    chkFABs->setChecked(m_animFABsEnabled);
    connect(chkFABs, &QCheckBox::toggled, this, [this](bool checked) { m_animFABsEnabled = checked; });
    animLay->addWidget(chkFABs);

    QCheckBox* chkBottom = new QCheckBox("Animate Bottom Panel", animCard);
    chkBottom->setChecked(m_animBottomPanelEnabled);
    connect(chkBottom, &QCheckBox::toggled, this, [this](bool checked) { m_animBottomPanelEnabled = checked; });
    animLay->addWidget(chkBottom);

    QCheckBox* chkTabs = new QCheckBox("Animate Tab Fade-ins", animCard);
    chkTabs->setChecked(m_animTabsEnabled);
    connect(chkTabs, &QCheckBox::toggled, this, [this](bool checked) { m_animTabsEnabled = checked; });
    animLay->addWidget(chkTabs);

    QCheckBox* chkDialogs = new QCheckBox("Animate Pop-up Dialogs", animCard);
    chkDialogs->setChecked(m_animDialogsEnabled);
    connect(chkDialogs, &QCheckBox::toggled, this, [this](bool checked) { m_animDialogsEnabled = checked; });
    animLay->addWidget(chkDialogs);

    animLay->addStretch();

    // Side-by-side layout
    QHBoxLayout* mainRow = new QHBoxLayout();
    mainRow->setSpacing(14);

    QVBoxLayout* leftCol = new QVBoxLayout();
    leftCol->setSpacing(10);
    leftCol->addWidget(saveCard);
    leftCol->addLayout(listRow, 1);

    mainRow->addLayout(leftCol, 1);
    mainRow->addWidget(animCard, 1);

    root->addLayout(mainRow, 1);

    return page;
}

struct NavSpec {
    NavItem  id;
    const char* glyph;
    const char* label;
    const char* tip;
};

static const NavSpec kNavSpecs[] = {
    { NavItem::Capture,  "C",  "Capture",   "Camera & color setup" },
    { NavItem::Pipeline, "P",  "Pipeline",  "Filter chain & alignment" },
    { NavItem::Diff,     "D",  "Diff",      "Diff mode parameters" },
    { NavItem::Focus,    "F",  "Focus",     "Focus telemetry & chart" },
    { NavItem::Snapshot, "S",  "Snapshot",  "Capture & recent shots" },
    { NavItem::Presets,  "R",  "Settings",   "Presets & animation settings" },
};

static int navIndexInBottomStack(NavItem item)
{

    switch (item) {
        case NavItem::Capture:  return 0;
        case NavItem::Pipeline: return 1;
        case NavItem::Diff:     return 2;
        case NavItem::Focus:    return 3;
        case NavItem::Snapshot: return 4;
        case NavItem::Presets:  return 5;
        default:                return -1;
    }
}

QWidget* MainWindow::buildSideNav()
{
    QWidget* nav = new QWidget(this);
    nav->setObjectName("sideNav");
    nav->setStyleSheet(QString("QWidget#sideNav { background: %1; border-right: 1px solid %2; }")
                       .arg(T::bg1).arg(T::border));
    nav->setMinimumWidth(56);
    nav->setMaximumWidth(280);

    QVBoxLayout* lay = new QVBoxLayout(nav);
    lay->setContentsMargins(6, 8, 6, 8);
    lay->setSpacing(2);

    auto makeRailToggle = [this](const QString& glyph, const QString& tip) {
        QPushButton* b = new QPushButton(glyph, this);
        b->setProperty("kind", "ghost");
        b->setCheckable(false);
        b->setCursor(Qt::PointingHandCursor);
        b->setToolTip(tip);
        b->setFixedHeight(32);
        b->setStyleSheet(QString(
            "QPushButton { background: transparent; color: %1; border: none;"
            " font-size: 14px; font-weight: 600; padding: 0 8px; text-align: left; }"
            "QPushButton:hover { color: %2; }")
            .arg(T::textDim).arg(T::text));
        return b;
    };

    m_btnNavCollapse = makeRailToggle("<<", "Collapse / expand side rail");
    connect(m_btnNavCollapse, &QPushButton::clicked, this, [this]() {
        setNavExpanded(!m_navExpanded);
    });
    lay->addWidget(m_btnNavCollapse);

    m_btnPreviewMode = makeRailToggle("[]",
                                      "Preview mode (hide chrome, cameras-only)");
    m_btnPreviewMode->setCheckable(true);
    m_btnPreviewMode->setStyleSheet(m_btnPreviewMode->styleSheet() +
        QString("QPushButton:checked { color: %1; }").arg(T::accent));
    connect(m_btnPreviewMode, &QPushButton::clicked, this, &MainWindow::togglePreviewMode);
    lay->addWidget(m_btnPreviewMode);

    QFrame* sep1 = new QFrame(nav);
    sep1->setFrameShape(QFrame::HLine);
    sep1->setStyleSheet(QString("color:%1; background:%1; max-height:1px;").arg(T::border));
    lay->addSpacing(4);
    lay->addWidget(sep1);
    lay->addSpacing(4);

    auto makeNavButton = [this](const NavSpec& spec) {
        QPushButton* b = new QPushButton(this);
        b->setCheckable(true);
        b->setAutoExclusive(false);
        b->setCursor(Qt::PointingHandCursor);
        b->setFixedHeight(32);
        b->setProperty("navGlyph", QString::fromLatin1(spec.glyph));
        b->setProperty("navLabel", QString::fromLatin1(spec.label));
        b->setText(QString("%1   %2")
                   .arg(QString::fromLatin1(spec.glyph),
                        QString::fromLatin1(spec.label)));
        b->setToolTip(QString("%1 — %2")
                      .arg(QString::fromLatin1(spec.label),
                           QString::fromLatin1(spec.tip)));
        b->setStyleSheet(QString(
            "QPushButton { background: transparent; color: %1; border: none;"
            " border-left: 2px solid transparent;"
            " text-align: left; padding: 0 10px; font-size: 12px; font-weight: 500;"
            " min-height: 28px; }"
            "QPushButton:hover { background: %2; color: %3; }"
            "QPushButton:checked { background: %4; color: %3;"
            " border-left: 2px solid %5; font-weight: 600; }")
            .arg(T::textDim)
            .arg(T::bg2)
            .arg(T::text)
            .arg(T::bg2)
            .arg(T::accent));
        return b;
    };

    for (const NavSpec& spec : kNavSpecs) {
        QPushButton* b = makeNavButton(spec);
        const NavItem id = spec.id;
        connect(b, &QPushButton::clicked, this, [this, id]() {
            toggleNavItem(id);
        });
        m_navButtons.insert(id, b);
        lay->addWidget(b);
    }

    lay->addStretch(1);

    QFrame* sep2 = new QFrame(nav);
    sep2->setFrameShape(QFrame::HLine);
    sep2->setStyleSheet(QString("color:%1; background:%1; max-height:1px;").arg(T::border));
    lay->addWidget(sep2);
    lay->addSpacing(4);

    NavSpec gallerySpec{ NavItem::Gallery, "G", "Gallery", "All snapshots, full view" };
    QPushButton* btnGallery = makeNavButton(gallerySpec);
    connect(btnGallery, &QPushButton::clicked, this, [this]() {
        toggleNavItem(NavItem::Gallery);
    });
    m_navButtons.insert(NavItem::Gallery, btnGallery);
    lay->addWidget(btnGallery);

    return nav;
}

void MainWindow::applyNavLabels()
{

    for (auto it = m_navButtons.begin(); it != m_navButtons.end(); ++it) {
        QPushButton* b = it.value();
        QString g = b->property("navGlyph").toString();
        QString l = b->property("navLabel").toString();
        if (m_navExpanded) {
            b->setText(QString("%1   %2").arg(g, l));
        } else {
            b->setText(g);
        }
    }

    if (m_btnNavCollapse) {
        m_btnNavCollapse->setText(m_navExpanded ? "<<   Close sidebar" : "<<");
    }
    if (m_btnPreviewMode) {
        m_btnPreviewMode->setText(m_navExpanded ? "[]   Hide sidebar" : "[]");
    }
}

void MainWindow::setNavExpanded(bool expanded)
{
    m_navExpanded = expanded;
    if (!m_mainSplit) {
        applyNavLabels();
        return;
    }
    QList<int> sizes = m_mainSplit->sizes();
    if (sizes.size() < 2) {
        applyNavLabels();
        return;
    }
    int total = sizes[0] + sizes[1];
    int startW = sizes[0];
    int endW = expanded ? 200 : 56;

    if (!m_animSidebarEnabled) {
        m_mainSplit->setSizes({ endW, std::max(300, total - endW) });
        applyNavLabels();
        return;
    }

    QVariantAnimation* anim = new QVariantAnimation(this);
    anim->setDuration(m_animSpeedMs);
    anim->setStartValue(startW);
    anim->setEndValue(endW);
    anim->setEasingCurve(QEasingCurve::OutQuad);
    connect(anim, &QVariantAnimation::valueChanged, this, [this, total](const QVariant& value) {
        int w = value.toInt();
        m_mainSplit->setSizes({ w, std::max(300, total - w) });
    });
    connect(anim, &QVariantAnimation::finished, this, &MainWindow::applyNavLabels);
    if (expanded) {
        applyNavLabels();
    }
    anim->start(QAbstractAnimation::DeleteWhenStopped);
}

void MainWindow::setNavItem(NavItem item)
{
    m_currentNav = item;

    for (auto it = m_navButtons.begin(); it != m_navButtons.end(); ++it) {
        it.value()->setChecked(it.key() == item);
    }

    auto animateBottomStack = [this](bool show) {
        if (!m_bottomStack || !m_workSplit) return;
        
        QList<int> sizes = m_workSplit->sizes();
        if (sizes.size() < 2) {
            m_bottomStack->setVisible(show);
            return;
        }

        int totalH = sizes[0] + sizes[1];
        int startH = sizes[1];
        int targetH = 260;

        if (!m_animBottomPanelEnabled) {
            m_bottomStack->setVisible(show);
            if (show) {
                m_workSplit->setSizes({ totalH - targetH, targetH });
            } else {
                m_workSplit->setSizes({ totalH, 0 });
            }
            return;
        }

        if (show) {
            if (!m_bottomStack->isVisible()) {
                m_bottomStack->setMinimumHeight(0);
                m_bottomStack->setVisible(true);
                QVariantAnimation* anim = new QVariantAnimation(this);
                anim->setDuration(m_animSpeedMs);
                anim->setStartValue(0);
                anim->setEndValue(targetH);
                anim->setEasingCurve(QEasingCurve::OutQuad);
                connect(anim, &QVariantAnimation::valueChanged, this, [this, totalH](const QVariant& val) {
                    int h = val.toInt();
                    m_workSplit->setSizes({ totalH - h, h });
                });
                anim->start(QAbstractAnimation::DeleteWhenStopped);
            }
        } else {
            if (m_bottomStack->isVisible()) {
                m_bottomStack->setMinimumHeight(0);
                QVariantAnimation* anim = new QVariantAnimation(this);
                anim->setDuration(m_animSpeedMs);
                anim->setStartValue(startH);
                anim->setEndValue(0);
                anim->setEasingCurve(QEasingCurve::OutQuad);
                connect(anim, &QVariantAnimation::valueChanged, this, [this, totalH](const QVariant& val) {
                    int h = val.toInt();
                    m_workSplit->setSizes({ totalH - h, h });
                });
                connect(anim, &QVariantAnimation::finished, this, [this]() {
                    m_bottomStack->setVisible(false);
                });
                anim->start(QAbstractAnimation::DeleteWhenStopped);
            }
        }
    };

    if (item == NavItem::Gallery) {
        if (m_workStack) m_workStack->setCurrentIndex(1);
        animateBottomStack(true);

        if (m_btnFabStream)   m_btnFabStream->hide();
        if (m_btnFabSnapshot) m_btnFabSnapshot->hide();
        if (m_btnModeToggle)  m_btnModeToggle->hide();
        refreshSnapshotPreview();
        return;
    }

    if (m_workStack) m_workStack->setCurrentIndex(0);

    if (item == NavItem::None) {
        animateBottomStack(false);
    } else {
        if (m_bottomStack) {
            int idx = navIndexInBottomStack(item);
            if (idx >= 0) m_bottomStack->setCurrentIndex(idx);
        }
        animateBottomStack(true);
        m_lastNonGalleryNav = item;
    }

    const bool focus = (item == NavItem::Focus);
    m_focusViewActive = focus;

    if (m_isDiffMode) {
        if (m_splitter)   m_splitter->hide();
        if (m_resultView) m_resultView->show();
    } else {
        if (m_splitter)   m_splitter->show();
        if (m_resultView) m_resultView->hide();
    }

    positionFloatingButtons();
}

void MainWindow::toggleNavItem(NavItem item)
{

    if (item == NavItem::Gallery) {
        if (m_currentNav == NavItem::Gallery) setNavItem(m_lastNonGalleryNav);
        else                                  setNavItem(NavItem::Gallery);
        return;
    }
    if (m_currentNav == item) {
        setNavItem(NavItem::None);
    } else {
        setNavItem(item);
    }
}

void MainWindow::togglePreviewMode()
{
    m_previewMode = !m_previewMode;
    if (m_btnPreviewMode) m_btnPreviewMode->setChecked(m_previewMode);

    if (!m_animSidebarEnabled) {
        if (m_previewMode) {
            if (m_sideNav) m_sideNav->hide();
            if (m_bottomStack) m_bottomStack->hide();
            if (m_statusBar) m_statusBar->hide();
        } else {
            if (m_sideNav) m_sideNav->show();
            if (m_bottomStack) m_bottomStack->setVisible(m_currentNav != NavItem::None);
            if (m_statusBar) m_statusBar->show();
        }
        positionFloatingButtons();
        return;
    }

    if (m_previewMode) {
        if (m_mainSplit && m_sideNav && m_sideNav->isVisible()) {
            m_sideNav->setMinimumWidth(0);
            QList<int> sizes = m_mainSplit->sizes();
            if (sizes.size() >= 2) {
                int startW = sizes[0];
                int totalW = sizes[0] + sizes[1];
                QVariantAnimation* anim = new QVariantAnimation(this);
                anim->setDuration(m_animSpeedMs);
                anim->setStartValue(startW);
                anim->setEndValue(0);
                anim->setEasingCurve(QEasingCurve::OutQuad);
                connect(anim, &QVariantAnimation::valueChanged, this, [this, totalW](const QVariant& val) {
                    int w = val.toInt();
                    m_mainSplit->setSizes({ w, totalW - w });
                });
                connect(anim, &QVariantAnimation::finished, this, [this]() {
                    m_sideNav->hide();
                });
                anim->start(QAbstractAnimation::DeleteWhenStopped);
            }
        }

        if (m_workSplit && m_bottomStack && m_bottomStack->isVisible()) {
            m_bottomStack->setMinimumHeight(0);
            QList<int> sizes = m_workSplit->sizes();
            if (sizes.size() >= 2) {
                int startH = sizes[1];
                int totalH = sizes[0] + sizes[1];
                QVariantAnimation* anim = new QVariantAnimation(this);
                anim->setDuration(m_animSpeedMs);
                anim->setStartValue(startH);
                anim->setEndValue(0);
                anim->setEasingCurve(QEasingCurve::OutQuad);
                connect(anim, &QVariantAnimation::valueChanged, this, [this, totalH](const QVariant& val) {
                    int h = val.toInt();
                    m_workSplit->setSizes({ totalH - h, h });
                });
                connect(anim, &QVariantAnimation::finished, this, [this]() {
                    m_bottomStack->hide();
                });
                anim->start(QAbstractAnimation::DeleteWhenStopped);
            }
        }

        if (m_statusBar && m_statusBar->isVisible()) {
            QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(m_statusBar->graphicsEffect());
            if (!effect) {
                effect = new QGraphicsOpacityEffect(m_statusBar);
                m_statusBar->setGraphicsEffect(effect);
            }
            QPropertyAnimation* fade = new QPropertyAnimation(effect, "opacity", m_statusBar);
            fade->setDuration(m_animSpeedMs);
            fade->setStartValue(1.0);
            fade->setEndValue(0.0);
            connect(fade, &QPropertyAnimation::finished, this, [this]() {
                m_statusBar->hide();
            });
            fade->start(QAbstractAnimation::DeleteWhenStopped);
        }
    } else {
        if (m_mainSplit && m_sideNav) {
            m_sideNav->setMinimumWidth(0);
            m_sideNav->show();
            QList<int> sizes = m_mainSplit->sizes();
            if (sizes.size() >= 2) {
                int totalW = sizes[0] + sizes[1];
                int targetW = m_navExpanded ? 200 : 56;
                QVariantAnimation* anim = new QVariantAnimation(this);
                anim->setDuration(m_animSpeedMs);
                anim->setStartValue(0);
                anim->setEndValue(targetW);
                anim->setEasingCurve(QEasingCurve::OutQuad);
                connect(anim, &QVariantAnimation::valueChanged, this, [this, totalW](const QVariant& val) {
                    int w = val.toInt();
                    m_mainSplit->setSizes({ w, totalW - w });
                });
                connect(anim, &QVariantAnimation::finished, this, [this]() {
                    m_sideNav->setMinimumWidth(56);
                });
                anim->start(QAbstractAnimation::DeleteWhenStopped);
            }
        }

        if (m_workSplit && m_bottomStack && m_currentNav != NavItem::None) {
            m_bottomStack->setMinimumHeight(0);
            m_bottomStack->show();
            QList<int> sizes = m_workSplit->sizes();
            if (sizes.size() >= 2) {
                int totalH = sizes[0] + sizes[1];
                int targetH = 260;
                QVariantAnimation* anim = new QVariantAnimation(this);
                anim->setDuration(m_animSpeedMs);
                anim->setStartValue(0);
                anim->setEndValue(targetH);
                anim->setEasingCurve(QEasingCurve::OutQuad);
                connect(anim, &QVariantAnimation::valueChanged, this, [this, totalH](const QVariant& val) {
                    int h = val.toInt();
                    m_workSplit->setSizes({ totalH - h, h });
                });
                anim->start(QAbstractAnimation::DeleteWhenStopped);
            }
        }

        if (m_statusBar) {
            m_statusBar->show();
            QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(m_statusBar->graphicsEffect());
            if (!effect) {
                effect = new QGraphicsOpacityEffect(m_statusBar);
                m_statusBar->setGraphicsEffect(effect);
            }
            QPropertyAnimation* fade = new QPropertyAnimation(effect, "opacity", m_statusBar);
            fade->setDuration(m_animSpeedMs);
            fade->setStartValue(0.0);
            fade->setEndValue(1.0);
            fade->start(QAbstractAnimation::DeleteWhenStopped);
        }
    }

    positionFloatingButtons();
}

QWidget* MainWindow::buildGalleryView()
{
    QWidget* page = new QWidget(this);
    page->setStyleSheet(QString("background: %1;").arg(T::bg1));
    QVBoxLayout* root = new QVBoxLayout(page);
    root->setContentsMargins(16, 12, 16, 12);
    root->setSpacing(8);

    QHBoxLayout* head = new QHBoxLayout();
    QLabel* title = new QLabel("SNAPSHOTS", page);
    title->setProperty("role", "section");
    head->addWidget(title);
    head->addStretch();
    QPushButton* btnOpenFolder = new QPushButton("Open folder", page);
    btnOpenFolder->setProperty("kind", "ghost");
    connect(btnOpenFolder, &QPushButton::clicked, this, [this]() {
        QString dir = QCoreApplication::applicationDirPath() + "/metrics";
        QDir().mkpath(dir);
        QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
    });
    head->addWidget(btnOpenFolder);
    root->addLayout(head);

    m_snapshotPreview = new QListWidget(page);
    m_snapshotPreview->setViewMode(QListWidget::IconMode);
    m_snapshotPreview->setIconSize(QSize(180, 130));
    m_snapshotPreview->setGridSize(QSize(200, 210));
    m_snapshotPreview->setResizeMode(QListWidget::Adjust);
    m_snapshotPreview->setSpacing(8);
    m_snapshotPreview->setFlow(QListWidget::LeftToRight);
    m_snapshotPreview->setWrapping(true);
    m_snapshotPreview->setUniformItemSizes(true);
    m_snapshotPreview->setWordWrap(true);
    m_snapshotPreview->setTextElideMode(Qt::ElideMiddle);
    root->addWidget(m_snapshotPreview, 1);

    m_snapshotPreview->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_snapshotPreview, &QListWidget::customContextMenuRequested, this, [this](const QPoint& pos) {
        QListWidgetItem* item = m_snapshotPreview->itemAt(pos);
        if (!item) return;
        QMenu menu(this);
        menu.setStyleSheet(styleSheetText());
        QAction* openAct = menu.addAction("Preview");
        QAction* renameAct = menu.addAction("Rename");
        QAction* dltAct = menu.addAction("Delete");
        QAction* res = menu.exec(m_snapshotPreview->mapToGlobal(pos));
        if (!res) return;
        QString dir = QCoreApplication::applicationDirPath() + "/metrics/";
        QString oldPath = dir + item->text();
        if (res == openAct) {
            emit m_snapshotPreview->itemDoubleClicked(item);
        } else if (res == renameAct) {
            bool ok;
            QString newName = QInputDialog::getText(this, "Rename", "New name:", QLineEdit::Normal, item->text(), &ok);
            if (ok && !newName.isEmpty()) {
                QString newPath = dir + newName;
                QFile::rename(oldPath, newPath);
                refreshSnapshotPreview();
            }
        } else if (res == dltAct) {
            if (QMessageBox::question(this, "Delete", "Delete " + item->text() + "?") == QMessageBox::Yes) {
                QFile::remove(oldPath);
                refreshSnapshotPreview();
            }
        }
    });

    connect(m_snapshotPreview, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* item) {
        if (!item) return;
        QString fileBase = item->text();
        QString imagePath = QCoreApplication::applicationDirPath() + "/metrics/" + fileBase;
        openPreviewWindow(imagePath, fileBase);
    });

    return page;
}

void MainWindow::recreateGpuViews()
{
    if (!m_splitter || !m_videoArea) return;

    QList<int> sizes = m_splitter->sizes();
    int resultIdx = -1;
    if (QLayout* lay = m_videoArea->layout()) {
        for (int i = 0; i < lay->count(); ++i) {
            if (lay->itemAt(i)->widget() == m_resultView) { resultIdx = i; break; }
        }
    }

    bool resultVisible = m_resultView && m_resultView->isVisible();

    auto* oldView1 = m_view1;
    auto* oldView2 = m_view2;
    auto* oldResult = m_resultView;

    auto makeFresh = [this](const QString& placeholder) {
        GpuImageView* v = new GpuImageView(this);
        v->setMinimumSize(160, 120);
        v->setPlaceholder(placeholder);
        return v;
    };
    m_view1     = makeFresh("CAM 1");
    m_view2     = makeFresh("CAM 2");
    m_resultView = makeFresh("RESULT");

    m_splitter->insertWidget(0, m_view1);
    m_splitter->insertWidget(1, m_view2);
    if (sizes.size() >= 2) m_splitter->setSizes(sizes);

    if (resultIdx >= 0 && m_videoArea->layout()) {
        QBoxLayout* bl = qobject_cast<QBoxLayout*>(m_videoArea->layout());
        if (bl) bl->insertWidget(resultIdx, m_resultView, 1);
    }
    m_resultView->setVisible(resultVisible);

    if (oldView1)  oldView1->deleteLater();
    if (oldView2)  oldView2->deleteLater();
    if (oldResult) oldResult->deleteLater();

    if (m_isDiffMode && !m_lastDiffResult.empty()) {
        m_resultView->setOverlayColor(QColor(0xff, 0xff, 0xff));
        m_resultView->setOverlayText("DIFF", false);
        displayMat(m_resultView, m_lastDiffResult);
    } else if (!m_frame1.empty() && !m_frame2.empty()) {
        m_view1->setOverlayColor(QColor(0x4e, 0xc9, 0xb0));
        m_view1->setOverlayText(QString("CAM1  Focus %1").arg(static_cast<int>(m_lastFocus1)), false);
        m_view2->setOverlayColor(QColor(0xce, 0x91, 0x78));
        m_view2->setOverlayText(QString("CAM2  Focus %1").arg(static_cast<int>(m_lastFocus2)), true);
        displayMat(m_view1, m_frame1);
        displayMat(m_view2, m_frame2);
    }
}

#ifdef DUALCAM_SEPARATE_VIEWER
static QString viewerExecutablePath()
{
    QString dir = QCoreApplication::applicationDirPath();
    QString exe = dir + "/DualCamViewer";
#ifdef Q_OS_WIN
    exe += ".exe";
#endif
    return exe;
}

static bool spawnViewer(const QString& mode, const QJsonObject& obj, QString& errOut)
{
    QString tmpDir = QDir::tempPath();
    QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss_zzz");
    QString payloadPath = QString("%1/dualcam_%2_%3.json").arg(tmpDir, mode, stamp);

    QFile f(payloadPath);
    if (!f.open(QIODevice::WriteOnly)) {
        errOut = "Cannot write payload to " + payloadPath;
        return false;
    }
    f.write(QJsonDocument(obj).toJson(QJsonDocument::Compact));
    f.close();

    QString exe = viewerExecutablePath();
    if (!QFile::exists(exe)) {
        QFile::remove(payloadPath);
        errOut = "Viewer not found: " + exe;
        return false;
    }
    QStringList args { "--mode", mode, "--payload", payloadPath };
    if (!QProcess::startDetached(exe, args)) {
        QFile::remove(payloadPath);
        errOut = "QProcess::startDetached failed.";
        return false;
    }
    return true;
}
#endif

void MainWindow::launchProfileViewer(const cv::Mat& img, QPoint a, QPoint b,
                                     const QString& title, bool darkTheme,
                                     QWidget* parent)
{
    const double dx = b.x() - a.x();
    const double dy = b.y() - a.y();
    const double length = std::hypot(dx, dy);
    const int steps = std::max(2, int(std::ceil(length)));

    auto bilinear = [&](const cv::Mat& m, int channel, double x, double y) -> double {
        if (x < 0) x = 0; if (y < 0) y = 0;
        if (x > m.cols - 1) x = m.cols - 1;
        if (y > m.rows - 1) y = m.rows - 1;
        int x0 = int(std::floor(x)), x1 = std::min(x0 + 1, m.cols - 1);
        int y0 = int(std::floor(y)), y1 = std::min(y0 + 1, m.rows - 1);
        double fx = x - x0, fy = y - y0;
        auto px = [&](int xx, int yy) -> double {
            if (m.channels() == 1) return m.at<uchar>(yy, xx);
            return m.at<cv::Vec3b>(yy, xx)[channel];
        };
        double v00 = px(x0, y0), v10 = px(x1, y0), v01 = px(x0, y1), v11 = px(x1, y1);
        return (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10
             + (1 - fx) * fy * v01 + fx * fy * v11;
    };
    auto sampleLuma = [&](double x, double y) -> double {
        if (img.channels() == 1) return bilinear(img, 0, x, y);
        const double bb = bilinear(img, 0, x, y);
        const double gg = bilinear(img, 1, x, y);
        const double rr = bilinear(img, 2, x, y);
        return 0.299 * rr + 0.587 * gg + 0.114 * bb;
    };

    QJsonArray samples;
    for (int i = 0; i < steps; ++i) {
        const double t = double(i) / (steps - 1);
        const double x = a.x() + t * dx;
        const double y = a.y() + t * dy;
        const double luma = sampleLuma(x, y);
        QJsonArray row { t * length, x, y, luma };
        samples.append(row);
    }

    QJsonObject obj;
    obj["title"]   = title;
    obj["dark"]    = darkTheme;
    obj["ax"]      = a.x();
    obj["ay"]      = a.y();
    obj["bx"]      = b.x();
    obj["by"]      = b.y();
    obj["length"]  = length;
    obj["samples"] = samples;

#ifndef DUALCAM_SEPARATE_VIEWER
    QDialog* dlg = makeProfileDialog(obj, parent ? parent : this);
    if (m_animDialogsEnabled) {
        animateDialogEntry(dlg, nullptr, m_animSpeedMs);
    } else {
        dlg->show();
    }
#else
    QString err;
    if (!spawnViewer("profile", obj, err)) {
        if (m_statusBar) m_statusBar->showMessage("Viewer error: " + err, 5000);
    }
#endif
}

void MainWindow::launchSurfaceViewer(const cv::Mat& img, QRect roi,
                                     const QString& title, bool darkTheme,
                                     QWidget* parent)
{
    cv::Rect bounds(0, 0, img.cols, img.rows);
    cv::Rect r(roi.x(), roi.y(), roi.width(), roi.height());
    r = r & bounds;
    if (r.width < 2 || r.height < 2) {
        if (m_statusBar) m_statusBar->showMessage("ROI too small.", 3000);
        return;
    }

    cv::Mat sub = img(r);
    cv::Mat gray;
    if (sub.channels() == 1) gray = sub;
    else cv::cvtColor(sub, gray, cv::COLOR_BGR2GRAY);

    const int maxSide = 96;
    cv::Mat g;
    if (gray.cols > maxSide || gray.rows > maxSide) {
        double k = double(maxSide) / std::max(gray.cols, gray.rows);
        cv::resize(gray, g, cv::Size(), k, k, cv::INTER_AREA);
    } else g = gray.clone();

    const int cols = g.cols;
    const int rows = g.rows;
    QByteArray gridBytes;
    gridBytes.resize(cols * rows);
    for (int y = 0; y < rows; ++y) {
        std::memcpy(gridBytes.data() + y * cols, g.ptr<uchar>(y), cols);
    }

    const double kBackX = (cols > 0) ? (double(r.width)  / double(cols)) : 1.0;
    const double kBackY = (rows > 0) ? (double(r.height) / double(rows)) : 1.0;

    QJsonObject obj;
    obj["title"] = title;
    obj["dark"]  = darkTheme;
    obj["roiX"]  = r.x;
    obj["roiY"]  = r.y;
    obj["kx"]    = kBackX;
    obj["ky"]    = kBackY;
    obj["cols"]  = cols;
    obj["rows"]  = rows;
    obj["grid"]  = QString::fromLatin1(gridBytes.toBase64());

#ifndef DUALCAM_SEPARATE_VIEWER
    QDialog* dlg = makeSurfaceDialog(obj, parent ? parent : this);
    if (m_animDialogsEnabled) {
        animateDialogEntry(dlg, nullptr, m_animSpeedMs);
    } else {
        dlg->show();
    }
#else
    QString err;
    if (!spawnViewer("surface", obj, err)) {
        if (m_statusBar) m_statusBar->showMessage("Viewer error: " + err, 5000);
    }
#endif
}

void MainWindow::restoreMainViewAfterChildClose()
{
    QTimer::singleShot(0, this, [this]() {
        auto kick = [](QOpenGLWidget* v) {
            if (!v) return;
            v->hide();
            v->show();
            v->update();
        };
        kick(m_view1);
        kick(m_view2);
        kick(m_resultView);
        this->update();
        this->activateWindow();
        this->raise();

        QTimer::singleShot(50, this, [this]() {
            if (m_isDiffMode && !m_lastDiffResult.empty()) {
                displayMat(m_resultView, m_lastDiffResult);
            } else if (!m_frame1.empty() && !m_frame2.empty()) {
                displayMat(m_view1, m_frame1);
                displayMat(m_view2, m_frame2);
            }
        });
    });
}

void MainWindow::openPreviewWindow(const QString& imagePath, const QString& fileBase)
{
    QDialog* dlg = new QDialog(nullptr, Qt::Window);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle("Preview: " + fileBase);
    dlg->setStyleSheet(styleSheetText());
    dlg->resize(this->width() * 2 / 3, this->height() * 2 / 3);

    connect(dlg, &QObject::destroyed, this, [this]() {
        restoreMainViewAfterChildClose();
    });

    QVBoxLayout* root = new QVBoxLayout(dlg);
    root->setContentsMargins(8, 8, 8, 8);
    root->setSpacing(8);

    QFrame* toolbar = new QFrame(dlg);
    toolbar->setProperty("role", "card");
    QHBoxLayout* tlay = new QHBoxLayout(toolbar);
    tlay->setContentsMargins(10, 6, 10, 6);
    tlay->setSpacing(10);

    ModeToggle* modeBtn = new ModeToggle(toolbar);

    QToolButton* depsBtn = new QToolButton(toolbar);
    depsBtn->setText("Dependencies");
    depsBtn->setCursor(Qt::PointingHandCursor);
    depsBtn->setPopupMode(QToolButton::InstantPopup);
    QMenu* depsMenu = new QMenu(depsBtn);
    depsMenu->setStyleSheet(styleSheetText());
    QAction* act45  = depsMenu->addAction("45°");
    QAction* act90  = depsMenu->addAction("90°");
    act45->setCheckable(true); act45->setChecked(true);
    act90->setCheckable(true); act90->setChecked(true);
    depsMenu->addSeparator();
    QAction* actCustom = depsMenu->addAction("Custom");
    actCustom->setCheckable(true);
    QWidgetAction* spinAct = new QWidgetAction(depsMenu);
    QWidget* spinRow = new QWidget(depsMenu);
    QHBoxLayout* spinLay = new QHBoxLayout(spinRow);
    spinLay->setContentsMargins(24, 2, 12, 6);
    spinLay->setSpacing(6);
    QSpinBox* spinCustom = new QSpinBox(spinRow);
    spinCustom->setRange(1, 89);
    spinCustom->setValue(30);
    spinCustom->setSuffix("°");
    spinLay->addWidget(new QLabel("Angle:", spinRow));
    spinLay->addWidget(spinCustom, 1);
    spinAct->setDefaultWidget(spinRow);
    depsMenu->addAction(spinAct);
    depsBtn->setMenu(depsMenu);

    ModeToggle* themeBtn = new ModeToggle(toolbar);
    themeBtn->setLabels("Light", "Dark");
    themeBtn->setValue(0, false);

    QLabel* hint = new QLabel("Hold Shift to snap. Drag to select region.", toolbar);
    hint->setProperty("role", "faint");

    tlay->addWidget(modeBtn);
    tlay->addSpacing(12);
    tlay->addWidget(depsBtn);
    tlay->addSpacing(12);
    tlay->addWidget(themeBtn);
    tlay->addSpacing(16);
    tlay->addWidget(hint, 1);
    root->addWidget(toolbar);

    QHBoxLayout* body = new QHBoxLayout();
    body->setContentsMargins(0, 0, 0, 0);
    body->setSpacing(8);
    root->addLayout(body, 1);

    cv::Mat raw = cv::imread(imagePath.toStdString(), cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        body->addWidget(new QLabel("Failed to load image."));
    }

    AnalysisCanvas* canvas = new AnalysisCanvas(dlg);
    if (!raw.empty()) {
        cv::Mat disp;
        if (raw.channels() == 1) {
            QImage qimg(raw.data, raw.cols, raw.rows, static_cast<int>(raw.step), QImage::Format_Grayscale8);
            canvas->setImage(qimg.copy());
        } else {
            cv::cvtColor(raw, disp, cv::COLOR_BGR2RGB);
            QImage qimg(disp.data, disp.cols, disp.rows, static_cast<int>(disp.step), QImage::Format_RGB888);
            canvas->setImage(qimg.copy());
        }
    }
    body->addWidget(canvas, 3);

    cv::Mat analysisSrc = raw.clone();

    auto updateAngles = [canvas, act45, act90, actCustom, spinCustom]() {
        QList<int> a;
        if (act45->isChecked())  a << 45;
        if (act90->isChecked())  a << 90;
        if (actCustom->isChecked()) a << spinCustom->value();
        if (a.isEmpty()) a << 90;
        canvas->setAllowedAngles(a);
    };
    updateAngles();
    connect(act45,      &QAction::toggled,         dlg, [updateAngles](bool){ updateAngles(); });
    connect(act90,      &QAction::toggled,         dlg, [updateAngles](bool){ updateAngles(); });
    connect(actCustom,  &QAction::toggled,         dlg, [updateAngles](bool){ updateAngles(); });
    connect(spinCustom, QOverload<int>::of(&QSpinBox::valueChanged), dlg, [updateAngles](int){ updateAngles(); });
    connect(modeBtn, &ModeToggle::valueChanged, dlg, [canvas](int v) {
        canvas->setMode(v == 0 ? AnalysisCanvas::Mode::Line : AnalysisCanvas::Mode::Rect);
    });

    QPointer<QDialog> dlgPtr(dlg);
    canvas->onLineSelected([this, analysisSrc, fileBase, dlgPtr, themeBtn](QPoint a, QPoint b) {
        const bool dark = themeBtn->value() == 1;
        launchProfileViewer(analysisSrc, a, b, fileBase, dark, dlgPtr.data());
    });
    canvas->onRectSelected([this, analysisSrc, fileBase, dlgPtr, themeBtn](QRect r) {
        const bool dark = themeBtn->value() == 1;
        launchSurfaceViewer(analysisSrc, r, fileBase, dark, dlgPtr.data());
    });

    QFrame* metaFrame = new QFrame(dlg);
    metaFrame->setProperty("role", "card");
    QVBoxLayout* metaLay = new QVBoxLayout(metaFrame);
    metaLay->setContentsMargins(8, 8, 8, 8);
    metaLay->setSpacing(6);
    QLabel* lblTitle = new QLabel("METADATA", dlg);
    lblTitle->setProperty("role", "section");
    metaLay->addWidget(lblTitle);

    QListWidget* metaList = new QListWidget(dlg);
    metaList->setSelectionMode(QAbstractItemView::SingleSelection);
    metaList->setUniformItemSizes(true);
    metaList->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    metaList->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    metaList->setTextElideMode(Qt::ElideRight);
    metaLay->addWidget(metaList, 1);

    std::vector<uint8_t> jpegBytes;
    QFile jf(imagePath);
    if (jf.open(QIODevice::ReadOnly)) {
        QByteArray ba = jf.readAll();
        jpegBytes.assign(reinterpret_cast<const uint8_t*>(ba.constData()),
                         reinterpret_cast<const uint8_t*>(ba.constData()) + ba.size());
    }
    std::string desc = readExifDescription(jpegBytes);
    QJsonDocument jdoc = desc.empty()
        ? QJsonDocument()
        : QJsonDocument::fromJson(QByteArray::fromStdString(desc));
    if (!jdoc.isNull() && jdoc.isObject()) {
        std::function<void(const QJsonObject&, const QString&)> emitObj =
            [&](const QJsonObject& obj, const QString& prefix) {
                for (const QString& key : obj.keys()) {
                    const QJsonValue v = obj.value(key);
                    const QString full = prefix.isEmpty() ? key : (prefix + "." + key);
                    QString line;
                    if (v.isObject()) { emitObj(v.toObject(), full); continue; }
                    else if (v.isArray())  line = full + ": [array]";
                    else if (v.isString()) line = full + ": " + v.toString();
                    else if (v.isBool())   line = full + ": " + (v.toBool() ? "true" : "false");
                    else if (v.isDouble()) {
                        const double d = v.toDouble();
                        QString s = (d == static_cast<qint64>(d))
                            ? QString::number(static_cast<qint64>(d))
                            : QString::number(d, 'g', 6);
                        line = full + ": " + s;
                    } else if (v.isNull()) line = full + ": null";
                    QListWidgetItem* it = new QListWidgetItem(line, metaList);
                    it->setToolTip(line);
                }
            };
        emitObj(jdoc.object(), QString());
    } else {
        new QListWidgetItem("No metadata found for this snapshot.", metaList);
    }
    body->addWidget(metaFrame, 1);

    if (m_animDialogsEnabled) {
        animateDialogEntry(dlg, m_snapshotPreview, m_animSpeedMs);
    } else {
        dlg->show();
        dlg->raise();
        dlg->activateWindow();
    }
}

void MainWindow::minimizeAllDialogs()
{
    int count = 0;
    for (QWidget* w : QApplication::topLevelWidgets()) {
        QDialog* d = qobject_cast<QDialog*>(w);
        if (!d) continue;
        if (!d->isVisible()) continue;
        d->showMinimized();
        ++count;
    }
    if (m_statusBar) m_statusBar->showMessage(QString("Minimized %1 dialog(s).").arg(count), 1500);
}

void MainWindow::animateDialogEntry(QDialog* dlg, QWidget* triggerWidget, int durationMs)
{
    if (!dlg) return;

    QWidget* parent = dlg->parentWidget();
    if (!parent) parent = QApplication::activeWindow();

    QSize targetSize = dlg->sizeHint();
    if (targetSize.width() < 100 || targetSize.height() < 100) {
        targetSize = dlg->size();
    }
    QRect targetGeo(QPoint(0, 0), targetSize);
    if (parent) {
        QPoint center = parent->geometry().center();
        targetGeo.moveCenter(center);
    } else {
        QPoint center = QApplication::primaryScreen()->geometry().center();
        targetGeo.moveCenter(center);
    }

    QPoint startPos;
    if (triggerWidget) {
        startPos = triggerWidget->mapToGlobal(triggerWidget->rect().center());
    } else {
        startPos = QCursor::pos();
    }

    QRect startGeo(startPos, QSize(10, 10));

    const QObjectList children = dlg->children();
    QList<QWidget*> childWidgets;
    for (QObject* child : children) {
        QWidget* cw = qobject_cast<QWidget*>(child);
        if (cw && cw->parentWidget() == dlg && !cw->isHidden()) {
            QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(cw->graphicsEffect());
            if (!effect) {
                effect = new QGraphicsOpacityEffect(cw);
                cw->setGraphicsEffect(effect);
            }
            effect->setOpacity(0.0);
            cw->setProperty("isFaded", false);
            childWidgets.append(cw);
        }
    }

    dlg->setGeometry(startGeo);
    dlg->show();
    dlg->raise();
    dlg->activateWindow();

    QPropertyAnimation* geomAnim = new QPropertyAnimation(dlg, "geometry", dlg);
    geomAnim->setDuration(durationMs);
    geomAnim->setStartValue(startGeo);
    geomAnim->setEndValue(targetGeo);
    geomAnim->setEasingCurve(QEasingCurve::OutQuad);

    QObject::connect(geomAnim, &QPropertyAnimation::valueChanged, dlg, [childWidgets, targetGeo](const QVariant& value) {
        QRect current = value.toRect();
        double progress = double(current.width()) / targetGeo.width();
        if (progress >= 0.33) {
            for (QWidget* cw : childWidgets) {
                if (cw && !cw->property("isFaded").toBool()) {
                    cw->setProperty("isFaded", true);
                    QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(cw->graphicsEffect());
                    if (effect) {
                        QPropertyAnimation* fade = new QPropertyAnimation(effect, "opacity", cw);
                        fade->setDuration(150);
                        fade->setStartValue(0.0);
                        fade->setEndValue(1.0);
                        connect(fade, &QPropertyAnimation::finished, cw, [cw]() {
                            cw->setGraphicsEffect(nullptr);
                        });
                        fade->start(QAbstractAnimation::DeleteWhenStopped);
                    }
                }
            }
        }
    });

    geomAnim->start(QAbstractAnimation::DeleteWhenStopped);
}

void MainWindow::buildAlignDialog()
{
    m_alignDialog = new QDialog(this);
    m_alignDialog->setWindowTitle("Manual Align (6 DOF)");
    m_alignDialog->setModal(false);
    m_alignDialog->resize(620, 460);


    m_alignDialog->setStyleSheet(styleSheetText() + QString(R"(
        QDialog QLabel { font-size: 12px; font-weight: 600; }
        QDialog QComboBox { font-size: 12px; padding: 4px 10px; min-height: 24px; }
        QDialog QDoubleSpinBox { font-size: 12px; padding: 4px 8px; min-height: 24px; }
        QDialog QPushButton { font-size: 12px; font-weight: 600; padding: 6px 16px; min-height: 26px; }
    )"));

    QVBoxLayout* root = new QVBoxLayout(m_alignDialog);
    root->setContentsMargins(14, 14, 14, 14);
    root->setSpacing(10);

    QHBoxLayout* headerRow = new QHBoxLayout();
    m_comboAdjCam = new QComboBox(m_alignDialog);
    m_comboAdjCam->addItems({ "CAM 1", "CAM 2" });
    m_comboAdjCam->setCurrentIndex(m_activeAdjCam - 1);
    connect(m_comboAdjCam, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int i) {
        m_activeAdjCam = i + 1;
        applyAdjustToWidgets();
    });
    m_btnResetAdj = new QPushButton("Reset", m_alignDialog);
    connect(m_btnResetAdj, &QPushButton::clicked, this, &MainWindow::resetActiveAdjust);

    headerRow->addWidget(new QLabel("Apply to:", m_alignDialog));
    headerRow->addWidget(m_comboAdjCam);
    headerRow->addStretch();
    headerRow->addWidget(m_btnResetAdj);
    root->addLayout(headerRow);

    auto makeAxisRow = [this, root](const QString& label,
                                    QSlider*& slider, QDoubleSpinBox*& spin,
                                    double minVal, double maxVal, double step,
                                    double sliderScale, const QString& suffix, int decimals)
    {
        QHBoxLayout* row = new QHBoxLayout();
        QLabel* lbl = new QLabel(label, m_alignDialog);
        lbl->setFixedWidth(70);

        slider = new QSlider(Qt::Horizontal, m_alignDialog);
        slider->setInvertedAppearance(false);
        slider->setInvertedControls(false);
        slider->setRange(static_cast<int>(minVal * sliderScale),
                         static_cast<int>(maxVal * sliderScale));

        spin = new QDoubleSpinBox(m_alignDialog);
        spin->setRange(minVal, maxVal);
        spin->setSingleStep(step);
        spin->setDecimals(decimals);
        spin->setSuffix(suffix);
        spin->setFixedWidth(130);
        spin->setKeyboardTracking(false);

        connect(slider, &QSlider::valueChanged, this, [this, spin, sliderScale](int v) {
            if (m_updatingAdjUI) return;
            QSignalBlocker b(spin);
            spin->setValue(static_cast<double>(v) / sliderScale);
            applyWidgetsToAdjust();
        });
        connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [this, slider, sliderScale](double v) {
            if (m_updatingAdjUI) return;
            QSignalBlocker b(slider);
            slider->setValue(static_cast<int>(v * sliderScale));
            applyWidgetsToAdjust();
        });

        row->addWidget(lbl);
        row->addWidget(slider, 1);
        row->addWidget(spin);
        root->addLayout(row);
    };

    makeAxisRow("Tx:",    m_sldAdjTx,    m_spnAdjTx,    -500.0,  500.0,  0.5, 1.0,    " px", 1);
    makeAxisRow("Ty:",    m_sldAdjTy,    m_spnAdjTy,    -500.0,  500.0,  0.5, 1.0,    " px", 1);
    makeAxisRow("Zoom:",  m_sldAdjScale, m_spnAdjScale,    0.2,    3.0,  0.01, 100.0, " x",  3);
    makeAxisRow("Pitch:", m_sldAdjRx,    m_spnAdjRx,     -45.0,   45.0,  0.1, 10.0,   " \u00B0", 2);
    makeAxisRow("Yaw:",   m_sldAdjRy,    m_spnAdjRy,     -45.0,   45.0,  0.1, 10.0,   " \u00B0", 2);
    makeAxisRow("Roll:",  m_sldAdjRz,    m_spnAdjRz,    -180.0,  180.0,  0.1, 10.0,   " \u00B0", 2);

    applyAdjustToWidgets();
    root->addStretch();
}

void MainWindow::positionFloatingButtons()
{
    if (!m_videoArea) return;
    const int w = m_videoArea->width();
    const int h = m_videoArea->height();
    const int pad = 10;
    const bool camerasHidden = (m_currentNav == NavItem::Gallery);

    auto animateButton = [this](QWidget* btn, bool show, const QPoint& targetPos) {
        if (!btn) return;

        bool currentlyVisible = btn->isVisible() && (btn->graphicsEffect() ? qobject_cast<QGraphicsOpacityEffect*>(btn->graphicsEffect())->opacity() > 0.1 : true);

        if (!m_animFABsEnabled) {
            btn->move(targetPos);
            btn->setVisible(show);
            QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(btn->graphicsEffect());
            if (effect) effect->setOpacity(1.0);
            return;
        }

        if (show) {
            if (currentlyVisible) {
                if (btn->pos() != targetPos) {
                    QPropertyAnimation* posAnim = new QPropertyAnimation(btn, "pos", btn);
                    posAnim->setDuration(m_animSpeedMs);
                    posAnim->setStartValue(btn->pos());
                    posAnim->setEndValue(targetPos);
                    posAnim->setEasingCurve(QEasingCurve::OutQuad);
                    posAnim->start(QAbstractAnimation::DeleteWhenStopped);
                }
                QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(btn->graphicsEffect());
                if (effect) effect->setOpacity(1.0);
            } else {
                btn->show();
                btn->raise();

                QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(btn->graphicsEffect());
                if (!effect) {
                    effect = new QGraphicsOpacityEffect(btn);
                    btn->setGraphicsEffect(effect);
                }
                effect->setOpacity(0.0);

                QPropertyAnimation* opacityAnim = new QPropertyAnimation(effect, "opacity", btn);
                opacityAnim->setDuration(m_animSpeedMs);
                opacityAnim->setStartValue(0.0);
                opacityAnim->setEndValue(1.0);
                opacityAnim->setEasingCurve(QEasingCurve::OutQuad);

                QPropertyAnimation* posAnim = new QPropertyAnimation(btn, "pos", btn);
                posAnim->setDuration(m_animSpeedMs);
                QPoint startPos = targetPos + QPoint(0, 25);
                posAnim->setStartValue(startPos);
                posAnim->setEndValue(targetPos);
                posAnim->setEasingCurve(QEasingCurve::OutQuad);

                QParallelAnimationGroup* group = new QParallelAnimationGroup(btn);
                group->addAnimation(opacityAnim);
                group->addAnimation(posAnim);
                group->start(QAbstractAnimation::DeleteWhenStopped);
            }
        } else {
            if (currentlyVisible) {
                QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(btn->graphicsEffect());
                if (!effect) {
                    effect = new QGraphicsOpacityEffect(btn);
                    btn->setGraphicsEffect(effect);
                }

                QPropertyAnimation* opacityAnim = new QPropertyAnimation(effect, "opacity", btn);
                opacityAnim->setDuration(m_animSpeedMs);
                opacityAnim->setStartValue(effect->opacity());
                opacityAnim->setEndValue(0.0);
                opacityAnim->setEasingCurve(QEasingCurve::OutQuad);

                QPropertyAnimation* posAnim = new QPropertyAnimation(btn, "pos", btn);
                posAnim->setDuration(m_animSpeedMs);
                QPoint endPos = btn->pos() + QPoint(0, 25);
                posAnim->setStartValue(btn->pos());
                posAnim->setEndValue(endPos);
                posAnim->setEasingCurve(QEasingCurve::OutQuad);

                QParallelAnimationGroup* group = new QParallelAnimationGroup(btn);
                group->addAnimation(opacityAnim);
                group->addAnimation(posAnim);

                QObject::connect(group, &QParallelAnimationGroup::finished, btn, [btn]() {
                    btn->hide();
                });
                group->start(QAbstractAnimation::DeleteWhenStopped);
            } else {
                btn->hide();
            }
        }
    };

    if (m_btnModeToggle) {
        QPoint target(w / 2 - m_btnModeToggle->width() / 2, h - pad - m_btnModeToggle->height());
        animateButton(m_btnModeToggle, !camerasHidden, target);
    }
    if (m_btnFabStream) {
        QPoint target(pad, h - pad - m_btnFabStream->height());
        animateButton(m_btnFabStream, !camerasHidden, target);
    }
    if (m_btnFabSnapshot) {
        QPoint target(w - pad - m_btnFabSnapshot->width(), h - pad - m_btnFabSnapshot->height());
        animateButton(m_btnFabSnapshot, !camerasHidden, target);
    }
    if (m_btnFloatingPreviewToggle) {
        QPoint target(pad, pad);
        animateButton(m_btnFloatingPreviewToggle, m_previewMode, target);
    }
}

void MainWindow::toggleSheet()
{

    if (m_currentNav == NavItem::None) {
        setNavItem(m_lastNonGalleryNav == NavItem::None
                   ? NavItem::Capture : m_lastNonGalleryNav);
    } else {
        setNavItem(NavItem::None);
    }
}

void MainWindow::setDiffMode(bool on)
{
    m_isDiffMode = on;
    if (m_btnModeToggle) {
        m_btnModeToggle->setText(on ? "Diff" : "Dual");
    }
    if (!on && m_lblPeakInfo) m_lblPeakInfo->clear();
    updateView();
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    saveSettings();
    QMainWindow::closeEvent(event);
}

void MainWindow::mousePressEvent(QMouseEvent* event)
{
    QWidget* focused = QApplication::focusWidget();
    if (qobject_cast<QLineEdit*>(focused)) {
        focused->clearFocus();
    }
    QMainWindow::mousePressEvent(event);
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Escape) {
        if (m_previewMode) {
            togglePreviewMode();
            return;
        }
    }
    QMainWindow::keyPressEvent(event);
}

void MainWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);

    const int w = width();
    const bool compact = w < 720;
    const bool ultraCompact = w < 560;

    if (m_fpsPill) m_fpsPill->setVisible(!ultraCompact);
    if (m_eccPill) m_eccPill->setVisible(!ultraCompact);

    if (m_btnGallery) m_btnGallery->setText(compact ? QStringLiteral("▣") : QStringLiteral("Snapshots"));
}

void MainWindow::toggleFocusView()
{

    if (m_currentNav == NavItem::Focus) {
        setNavItem(m_lastNonGalleryNav == NavItem::Focus
                   ? NavItem::Capture : m_lastNonGalleryNav);
    } else {
        setNavItem(NavItem::Focus);
    }
}

void MainWindow::refreshSnapshotPreview()
{
    QString dir = QCoreApplication::applicationDirPath() + "/metrics";
    QDir metricsDir(dir);
    if (!metricsDir.exists()) {
        if (m_snapshotPreview) m_snapshotPreview->clear();
        if (m_snapshotRecent)  m_snapshotRecent->clear();
        return;
    }

    QStringList filters = {"*.png", "*.jpg", "*.jpeg", "*.bmp"};
    QFileInfoList files = metricsDir.entryInfoList(filters, QDir::Files, QDir::Time);

    auto fillList = [](QListWidget* lw, const QFileInfoList& src, int maxItems,
                       const QSize& thumbSize) {
        if (!lw) return;
        lw->clear();
        const QSize grid = lw->gridSize();
        int n = 0;
        for (const QFileInfo& fi : src) {
            if (maxItems > 0 && n >= maxItems) break;
            cv::Mat raw = cv::imread(fi.absoluteFilePath().toStdString(), cv::IMREAD_COLOR);
            if (raw.empty()) continue;
            cv::Mat rgb;
            cv::cvtColor(raw, rgb, cv::COLOR_BGR2RGB);
            QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
            QPixmap pix = QPixmap::fromImage(qimg.copy());
            if (pix.isNull()) continue;
            QListWidgetItem* item = new QListWidgetItem(
                QIcon(pix.scaled(thumbSize, Qt::KeepAspectRatio, Qt::SmoothTransformation)),
                fi.fileName());
            item->setToolTip(fi.fileName());
            item->setTextAlignment(Qt::AlignHCenter | Qt::AlignTop);
            if (grid.isValid()) item->setSizeHint(grid);
            lw->addItem(item);
            ++n;
        }
    };

    fillList(m_snapshotPreview, files, 0,  QSize(180, 130));
    fillList(m_snapshotRecent,  files, 5,  QSize(110, 80));
}

void MainWindow::updateEccPill()
{
    if (!m_eccPill) return;
    if (m_isAligned && !m_eccWarpMatrix.empty()) {
        if (m_chkAlign && m_chkAlign->isChecked()) {
            setPillState(m_eccPill, "ok", "ALIGN ON");
        } else {
            setPillState(m_eccPill, "info", "ALIGN READY");
        }
    } else if (m_calibrating) {
        setPillState(m_eccPill, "warn", "CALIBRATING\u2026");
    } else {
        setPillState(m_eccPill, "err", "NO ALIGN");
    }
}

void MainWindow::updateFpsPill()
{
    if (!m_fpsPill) return;
    if (m_camerasOpen) {
        setPillState(m_fpsPill, "ok", "STREAMING");
    } else {
        setPillState(m_fpsPill, "idle", "IDLE");
    }
}

void MainWindow::refreshCameraModes()
{
    m_comboCamSet->clear();
    m_comboCamSet->addItem("640x480 @ 30 FPS", QVariantList{640, 480, 30});
    m_comboCamSet->addItem("640x480 @ 60 FPS", QVariantList{640, 480, 60});
    m_comboCamSet->addItem("1280x720 @ 30 FPS", QVariantList{1280, 720, 30});
    m_comboCamSet->addItem("1280x720 @ 60 FPS", QVariantList{1280, 720, 60});
    m_comboCamSet->addItem("1920x1080 @ 30 FPS", QVariantList{1920, 1080, 30});
    m_comboCamSet->addItem("1920x1080 @ 60 FPS", QVariantList{1920, 1080, 60});
    m_comboCamSet->addItem("Custom...", QVariantList{0, 0, 0});
}

QStringList MainWindow::getLibCameraIds()
{
    QStringList cameraPaths;
    QProcess process;

    QString program = "rpicam-hello";
    process.start(program, QStringList() << "--list-cameras");
    if (!process.waitForFinished() || process.exitCode() != 0) {
        program = "libcamera-hello";
        process.start(program, QStringList() << "--list-cameras");
        process.waitForFinished();
    }

    QString output = process.readAllStandardOutput();

    QRegularExpression re("\\((/base/[^)]+)\\)");
    QRegularExpressionMatchIterator i = re.globalMatch(output);

    while (i.hasNext()) {
        QRegularExpressionMatch match = i.next();
        if (match.hasMatch()) {
            cameraPaths << match.captured(1);
        }
    }

    return cameraPaths;
}

std::string MainWindow::makeGStreamerPipeline(const QString& cameraId, int width, int height, int fps, int camIndex)
{
    QString controls;
    if (m_manualExposure) {
        int gainQ8     = (m_perCameraExposure && camIndex == 1) ? m_gain2Q8    : m_gainQ8;
        int shutterUs  = (m_perCameraExposure && camIndex == 1) ? m_shutter2Us : m_shutterUs;
        double gain = gainQ8 / 256.0;
        if (gain < 1.0) gain = 1.0;
        controls = QString(" ae-enable=false analogue-gain-mode=manual exposure-time-mode=manual analogue-gain=%1 exposure-time=%2")
            .arg(gain, 0, 'f', 3)
            .arg(shutterUs);
    } else {
        controls = " ae-enable=true";
    }

    return QString("libcamerasrc camera-name=%1%5 ! "
        "video/x-raw, width=%2, height=%3, framerate=%4/1, format=NV12 ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=1 sync=false")
        .arg(cameraId)
        .arg(width)
        .arg(height)
        .arg(fps)
        .arg(controls)
        .toStdString();
}

void MainWindow::applyExposureControls()
{

    if (m_camerasOpen) {
        m_statusBar->showMessage("Restarting cameras to apply exposure...", 2000);
        closeCameras();
        openCameras();
    }
}

void MainWindow::openCameras()
{
    m_statusBar->showMessage("Detecting cameras...");
    QApplication::processEvents();

    int reqW = 640, reqH = 480, reqFps = 30;
    if (m_comboCamSet->count() > 0) {
        QVariantList v = m_comboCamSet->currentData().toList();
        if (v.size() == 3) {
            reqW = v[0].toInt();
            reqH = v[1].toInt();
            reqFps = v[2].toInt();
        }
    }

    QStringList camPaths = getLibCameraIds();
    if (camPaths.size() < 2) {
        m_statusBar->showMessage("Warning: < 2 libcameras found. Attempting fallback.", 3000);
        int c1 = -1, c2 = -1;
        for (int i = 0; i < 10; ++i) {
            cv::VideoCapture tmp;
#ifdef Q_OS_LINUX
            tmp.open(i, cv::CAP_V4L2);
#else
            tmp.open(i, cv::CAP_DSHOW);
#endif
            if (tmp.isOpened()) {
                if (c1 == -1) c1 = i;
                else if (c2 == -1) { c2 = i; break; }
            }
        }
        if (c1 != -1 && c2 != -1) {
            m_worker->startCamerasV4L2(c1, c2, reqW, reqH, reqFps);
        } else {
            m_statusBar->showMessage("Error: Could not find two fallback cameras.", 3000);
            return;
        }
    } else {
        std::string p1 = makeGStreamerPipeline(camPaths[0], reqW, reqH, reqFps, 0);
        std::string p2 = makeGStreamerPipeline(camPaths[1], reqW, reqH, reqFps, 1);
        m_worker->startCameras(p1, p2, reqW, reqH, reqFps);
    }

    m_camerasOpen = true;
    m_isAligned = false;
    m_eccWarpMatrix.release();

    if (m_comboCamSet) m_comboCamSet->setEnabled(false);

    m_btnFabStream->setToolTip("Stop streaming");
    m_btnFabStream->setStyleSheet(QString(
        "QPushButton { background: %1; border: 1px solid %1; border-radius: 0px; }"
        "QPushButton:hover { background: %2; border-color: %2; }")
        .arg(T::accent).arg(T::accentDim));
    if (m_fabStreamIcon) {
        m_fabStreamIcon->setText(QString::fromUtf8("\u25A0"));
        m_fabStreamIcon->setStyleSheet(QString(
            "QLabel { background: transparent; color: %1; border: none;"
            " font-family: 'Segoe UI Symbol','Segoe UI Emoji','Arial Unicode MS';"
            " font-size: 14px; font-weight: 800; }").arg(T::text));
    }

    updateEccPill();
    updateFpsPill();

    m_motionActive = false;

    m_frameCount = 0;
    m_seriesCam1->clear();
    m_seriesCam2->clear();

    m_chart->axes(Qt::Horizontal).first()->setRange(0, m_maxHistory);

    m_chart->axes(Qt::Vertical).first()->setRange(0, 100);

    m_btnFabStream->setObjectName("fabStreamStop");
    m_btnFabStream->style()->unpolish(m_btnFabStream);
    m_btnFabStream->style()->polish(m_btnFabStream);

    m_statusBar->showMessage("Cameras OK. Noise suppression active.", 4000);
}

void MainWindow::closeCameras()
{
    m_worker->stopCameras();
    m_camerasOpen = false;

    if (m_comboCamSet) m_comboCamSet->setEnabled(true);

    if (m_btnFabStream) {

        m_btnFabStream->setToolTip("Start streaming");
        m_btnFabStream->setStyleSheet(QString(
            "QPushButton { background: rgba(20,20,20,230); border: 1px solid %1; border-radius: 0px; }"
            "QPushButton:hover { background: rgba(40,12,12,230); border: 1px solid %2; }")
            .arg(T::border).arg(T::accent));
        if (m_fabStreamIcon) {
            m_fabStreamIcon->setText(QString::fromUtf8("\u25B6"));
            m_fabStreamIcon->setStyleSheet(QString(
                "QLabel { background: transparent; color: %1; border: none;"
                " font-family: 'Segoe UI Symbol','Segoe UI Emoji','Arial Unicode MS';"
                " font-size: 14px; font-weight: 700; }").arg(T::text));
        }
    }

    m_frame1.release();
    m_frame2.release();

    if (m_view1) { m_view1->setOverlayText("", false); m_view1->setPlaceholder("CAM 1"); }
    if (m_view2) { m_view2->setOverlayText("", false); m_view2->setPlaceholder("CAM 2"); }
    if (m_resultView) { m_resultView->setOverlayText("", false); m_resultView->setPlaceholder("DIFF"); }

    if (m_motionIndicator) {
        m_motionIndicator->setText(QString::fromUtf8("\u25CF Idle"));
        m_motionIndicator->setStyleSheet(QString("color:%1; font-weight:600;").arg(T::textDim));
    }
    updateEccPill();
    updateFpsPill();

    m_statusBar->showMessage("Cameras closed.", 2000);
}

cv::Mat MainWindow::fuseCameras(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat out;
    cv::addWeighted(a, 0.5, b, 0.5, 0.0, out);
    return out;
}

cv::Mat MainWindow::applyDiffView(const cv::Mat& d1, const cv::Mat& d2)
{
    cv::Mat diff;

    cv::Mat a = d1, b = d2;
    if (b.size() != a.size()) cv::resize(b, b, a.size());

    cv::Mat ga, gb;
    if (a.channels() == 3) cv::cvtColor(a, ga, cv::COLOR_BGR2GRAY);
    else ga = a;
    if (b.channels() == 3) cv::cvtColor(b, gb, cv::COLOR_BGR2GRAY);
    else gb = b;

    const double f1 = m_lastFocus1;
    const double f2 = m_lastFocus2;
    if (f1 > 1.0 && f2 > 1.0) {
        const double ratio = (f1 > f2) ? (f1 / f2) : (f2 / f1);
        if (ratio > 1.15) {
            const double sigma = std::min(6.0, 0.6 * std::sqrt(ratio - 1.0));
            int k = static_cast<int>(std::ceil(sigma * 3.0)) | 1;
            if (k < 3) k = 3;
            if (k > 31) k = 31;
            if (f1 > f2) cv::GaussianBlur(ga, ga, cv::Size(k, k), sigma);
            else         cv::GaussianBlur(gb, gb, cv::Size(k, k), sigma);
        }
    }

    cv::absdiff(ga, gb, diff);

    if (m_noiseFloor > 0) {
        cv::threshold(diff, diff, m_noiseFloor, 255, cv::THRESH_TOZERO);
    }

    if (m_chkStretch && m_chkStretch->isChecked()) {
        cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX);
    }

    cv::applyColorMap(diff, diff, cv::COLORMAP_JET);

    return diff;
}

bool MainWindow::eventFilter(QObject* obj, QEvent* event)
{
    if (obj == m_videoArea && event->type() == QEvent::Resize) {
        positionFloatingButtons();
    }
    return QMainWindow::eventFilter(obj, event);
}

void MainWindow::onFramesProcessed(cv::Mat f1, cv::Mat f2, double focus1, double focus2, bool motionDetected, qint64 frameCount)
{
    if (m_worker) m_worker->m_pendingFrames.fetch_sub(1);

    if (!m_camerasOpen) return;

    m_frame1 = f1;
    m_frame2 = f2;
    m_lastFocus1 = focus1;
    m_lastFocus2 = focus2;
    m_frameCount = frameCount;

    if (m_focusViewActive) {
        if (m_lblFocus1Big) m_lblFocus1Big->setText(QString::number(static_cast<int>(focus1)));
        if (m_lblFocus2Big) m_lblFocus2Big->setText(QString::number(static_cast<int>(focus2)));

        if (m_seriesCam1 && m_seriesCam2 && m_chart) {
            m_seriesCam1->append(frameCount, focus1);
            m_seriesCam2->append(frameCount, focus2);

            int diff1 = m_seriesCam1->count() - m_maxHistory;
            if (diff1 > 0) m_seriesCam1->removePoints(0, diff1);
            int diff2 = m_seriesCam2->count() - m_maxHistory;
            if (diff2 > 0) m_seriesCam2->removePoints(0, diff2);

            auto axes = m_chart->axes(Qt::Horizontal);
            if (!axes.isEmpty()) {
                axes.first()->setRange(std::max(0LL, frameCount - m_maxHistory), frameCount);
            }


            auto seriesMax = [](QLineSeries* s) {
                double m = 0.0;
                const auto pts = s->points();
                for (const QPointF& p : pts) if (p.y() > m) m = p.y();
                return m;
            };
            const double rawMax = std::max(seriesMax(m_seriesCam1), seriesMax(m_seriesCam2));
            auto niceCeil = [](double v) {
                if (v <= 0.0) return 100.0;
                const double padded = v * 1.10;
                const double scale  = std::pow(10.0, std::floor(std::log10(padded)));
                const double m      = padded / scale;
                double rounded;
                if      (m <= 1.0) rounded = 1.0;
                else if (m <= 2.0) rounded = 2.0;
                else if (m <= 5.0) rounded = 5.0;
                else               rounded = 10.0;
                return rounded * scale;
            };
            const double targetMax = niceCeil(rawMax);
            auto vaxes = m_chart->axes(Qt::Vertical);
            if (!vaxes.isEmpty()) {
                QValueAxis* yAx = qobject_cast<QValueAxis*>(vaxes.first());
                if (yAx) {
                    const double currentMax = yAx->max();
                    if (targetMax > currentMax || targetMax < currentMax * 0.7) {
                        yAx->setRange(0.0, targetMax);
                    }
                }
            }
        }
    }

    if (m_motionActive != motionDetected) {
        m_motionActive = motionDetected;
        if (m_motionIndicator) {
            if (m_motionActive) {
                m_motionIndicator->setText(QString::fromUtf8("\u25CF Motion"));
                m_motionIndicator->setStyleSheet(QString("color:%1; font-weight:600;").arg(T::err));
            } else {
                m_motionIndicator->setText(QString::fromUtf8("\u25CF Stable"));
                m_motionIndicator->setStyleSheet(QString("color:%1; font-weight:600;").arg(T::ok));
            }
        }
    }

    updateView();
}

void MainWindow::updateView()
{
    updateEccPill();

    if (m_frame1.empty() || m_frame2.empty()) return;

    cv::Mat f1 = m_frame1.clone();
    cv::Mat f2 = m_frame2.clone();

    if (!m_manualAdj1.isIdentity()) {
        cv::Mat H = buildManualHomography(m_manualAdj1, f1.size());
        cv::Mat warped;
        cv::warpPerspective(f1, warped, H, f1.size(), cv::INTER_LINEAR);
        f1 = warped;
    }
    if (!m_manualAdj2.isIdentity()) {
        cv::Mat H = buildManualHomography(m_manualAdj2, f2.size());
        cv::Mat warped;
        cv::warpPerspective(f2, warped, H, f2.size(), cv::INTER_LINEAR);
        f2 = warped;
    }

    const bool eccReady = m_isAligned && !m_eccWarpMatrix.empty();
    const bool wantAlign = m_chkAlign && m_chkAlign->isChecked() && eccReady;
    const bool wantFusion = m_chkFusion && m_chkFusion->isChecked() && eccReady;

    cv::Mat warpedF2 = f2;
    if (wantAlign || wantFusion) {
        try {
            cv::warpPerspective(f2, warpedF2, m_eccWarpMatrix, f1.size(), cv::INTER_LINEAR);
        }
        catch (const cv::Exception& e) {
            std::cerr << "Warp error: " << e.what() << std::endl;
            warpedF2 = f2;
        }
    }

    cv::Mat alignedF2 = wantAlign ? warpedF2 : f2;
    if (wantFusion) {
        f1 = fuseCameras(f1, warpedF2);
    }

    bool showPeaks = m_btnPeakIntensities && m_btnPeakIntensities->isChecked();
    cv::Point maxLoc1, maxLoc2;
    double maxVal1 = 0, maxVal2 = 0;

    if (showPeaks) {
        cv::Mat g1, g2;
        if (f1.channels() == 3) cv::cvtColor(f1, g1, cv::COLOR_BGR2GRAY);
        else g1 = f1;

        if (alignedF2.channels() == 3) cv::cvtColor(alignedF2, g2, cv::COLOR_BGR2GRAY);
        else g2 = alignedF2;

        cv::Mat blur1, blur2;
        cv::blur(g1, blur1, cv::Size(31, 31));
        cv::blur(g2, blur2, cv::Size(31, 31));

        cv::minMaxLoc(blur1, nullptr, nullptr, nullptr, &maxLoc1);
        cv::minMaxLoc(blur2, nullptr, nullptr, nullptr, &maxLoc2);

        cv::minMaxLoc(g1, nullptr, &maxVal1, nullptr, nullptr);
        cv::minMaxLoc(g2, nullptr, &maxVal2, nullptr, nullptr);

        m_lblPeakInfo->setText(QString("Peak 1: %1 | Peak 2: %2")
            .arg(static_cast<int>(maxVal1))
            .arg(static_cast<int>(maxVal2)));
    }

    auto drawTarget = [](cv::Mat& img, cv::Point pt, const cv::Scalar& color, const std::string& label) {
        cv::circle(img, pt, 20, color, 2);
        cv::line(img, cv::Point(pt.x - 10, pt.y), cv::Point(pt.x + 10, pt.y), color, 2);
        cv::line(img, cv::Point(pt.x, pt.y - 10), cv::Point(pt.x, pt.y + 10), color, 2);
        cv::putText(img, label, cv::Point(pt.x + 25, pt.y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    };

    if (!m_isDiffMode)
    {
        m_resultView->hide();
        m_splitter->show();

        if (showPeaks) {
            if (f1.channels() == 1) cv::cvtColor(f1, f1, cv::COLOR_GRAY2BGR);
            if (alignedF2.channels() == 1) cv::cvtColor(alignedF2, alignedF2, cv::COLOR_GRAY2BGR);
            drawTarget(f1, maxLoc1, cv::Scalar(0, 255, 255), "Max 1");
            drawTarget(alignedF2, maxLoc2, cv::Scalar(0, 255, 255), "Max 2");
        }

        m_view1->setOverlayColor(QColor(0x4e, 0xc9, 0xb0));
        m_view1->setOverlayText(QString("CAM1  Focus %1").arg(static_cast<int>(m_lastFocus1)), false);
        m_view2->setOverlayColor(QColor(0xce, 0x91, 0x78));
        m_view2->setOverlayText(QString("CAM2  Focus %1").arg(static_cast<int>(m_lastFocus2)), true);

        displayMat(m_view1, f1);
        displayMat(m_view2, alignedF2);
    }
    else
    {
        m_splitter->hide();
        m_resultView->show();

        cv::Mat diff = applyDiffView(f1, alignedF2);

        if (showPeaks) {
            drawTarget(diff, maxLoc1, cv::Scalar(0, 255, 255), "P1");
            drawTarget(diff, maxLoc2, cv::Scalar(255, 0, 255), "P2");
        }

        m_resultView->setOverlayColor(QColor(0xff, 0xff, 0xff));
        m_resultView->setOverlayText("DIFF", false);

        m_lastDiffResult = diff.clone();
        displayMat(m_resultView, diff);
    }
}

void MainWindow::calibrateAlignment()
{
    if (!m_camerasOpen) {
        m_statusBar->showMessage("Cameras closed.", 2000);
        return;
    }

    if (m_calibrating) {
        m_statusBar->showMessage("Calibration already in progress...", 2000);
        return;
    }

    cv::Mat f1 = m_frame1.clone();
    cv::Mat f2 = m_frame2.clone();

    if (f1.empty() || f2.empty()) {
        m_statusBar->showMessage("Calibration failed: Empty frames.", 3000);
        return;
    }

    if (!m_manualAdj1.isIdentity()) {
        cv::Mat H = buildManualHomography(m_manualAdj1, f1.size());
        cv::Mat warped;
        cv::warpPerspective(f1, warped, H, f1.size(), cv::INTER_LINEAR);
        f1 = warped;
    }
    if (!m_manualAdj2.isIdentity()) {
        cv::Mat H = buildManualHomography(m_manualAdj2, f2.size());
        cv::Mat warped;
        cv::warpPerspective(f2, warped, H, f2.size(), cv::INTER_LINEAR);
        f2 = warped;
    }

    cv::Mat gray1, gray2;
    if (f1.channels() == 3) cv::cvtColor(f1, gray1, cv::COLOR_BGR2GRAY);
    else gray1 = f1.clone();
    if (f2.channels() == 3) cv::cvtColor(f2, gray2, cv::COLOR_BGR2GRAY);
    else gray2 = f2.clone();

    if (calculateFocus(gray1) < 2.0) {
        m_statusBar->showMessage("Error: Too dark for calibration!", 4000);
        return;
    }

    m_calibrating = true;
    m_btnCalibrateAlign->setEnabled(false);
    setPillState(m_eccIndicator, "warn", "CALIBRATING\u2026");
    updateEccPill();
    m_statusBar->showMessage("Calibrating (SIFT)... please wait.");

    if (m_calibThread.joinable()) m_calibThread.join();

    m_calibThread = std::thread([this, gray1 = std::move(gray1), gray2 = std::move(gray2)]() {
        QString errorMsg;
        QString modelUsed;
        cv::Mat warpMatrix;
        bool success = false;

        auto isHomographySane = [](const cv::Mat& H, const cv::Size& imgSz) -> bool {
            if (H.empty() || H.rows != 3 || H.cols != 3) return false;
            cv::Mat Hd;
            H.convertTo(Hd, CV_64F);
            double det = cv::determinant(Hd);
            if (!std::isfinite(det) || std::abs(det) < 1e-4 || std::abs(det) > 1e4) return false;

            std::vector<cv::Point2f> corners = {
                {0.f, 0.f},
                {static_cast<float>(imgSz.width - 1), 0.f},
                {static_cast<float>(imgSz.width - 1), static_cast<float>(imgSz.height - 1)},
                {0.f, static_cast<float>(imgSz.height - 1)}
            };
            std::vector<cv::Point2f> warped;
            cv::perspectiveTransform(corners, warped, Hd);
            for (const auto& p : warped) {
                if (!std::isfinite(p.x) || !std::isfinite(p.y)) return false;
                if (std::abs(p.x) > 4.0 * imgSz.width || std::abs(p.y) > 4.0 * imgSz.height) return false;
            }
            auto edgeLen = [](cv::Point2f a, cv::Point2f b) {
                return std::hypot(a.x - b.x, a.y - b.y);
            };
            double e0 = edgeLen(warped[0], warped[1]);
            double e1 = edgeLen(warped[1], warped[2]);
            double e2 = edgeLen(warped[2], warped[3]);
            double e3 = edgeLen(warped[3], warped[0]);
            double srcE0 = imgSz.width;
            double srcE1 = imgSz.height;
            auto ratio = [](double a, double b) { return (a > b) ? (a / b) : (b / a); };
            if (ratio(e0, srcE0) > 5.0 || ratio(e2, srcE0) > 5.0) return false;
            if (ratio(e1, srcE1) > 5.0 || ratio(e3, srcE1) > 5.0) return false;
            if (ratio(e0, e2) > 4.0 || ratio(e1, e3) > 4.0) return false;
            return true;
        };

        const cv::Size imgSz = gray1.size();

        auto runWithDetector = [&](cv::Ptr<cv::Feature2D> detector,
                                   cv::Ptr<cv::DescriptorMatcher> matcher,
                                   float loweRatio,
                                   const QString& tag,
                                   QString& outErr,
                                   QString& outModel,
                                   cv::Mat& outWarp) -> bool {
            std::vector<cv::KeyPoint> kp1, kp2;
            cv::Mat des1, des2;
            detector->detectAndCompute(gray1, cv::noArray(), kp1, des1);
            detector->detectAndCompute(gray2, cv::noArray(), kp2, des2);

            if (des1.empty() || des2.empty() || kp1.size() < 16 || kp2.size() < 16) {
                outErr = tag + ": insufficient keypoints";
                return false;
            }

            std::vector<std::vector<cv::DMatch>> knn;
            matcher->knnMatch(des2, des1, knn, 2);

            std::vector<cv::Point2f> ptsSrc, ptsDst;
            ptsSrc.reserve(knn.size());
            ptsDst.reserve(knn.size());
            for (const auto& pair : knn) {
                if (pair.size() < 2) continue;
                if (pair[0].distance < loweRatio * pair[1].distance) {
                    ptsSrc.push_back(kp2[pair[0].queryIdx].pt);
                    ptsDst.push_back(kp1[pair[0].trainIdx].pt);
                }
            }

            if (ptsSrc.size() < 12) {
                outErr = QString("%1: only %2 good matches").arg(tag).arg(static_cast<int>(ptsSrc.size()));
                return false;
            }

            cv::Mat inliers;
            cv::Mat H = cv::findHomography(ptsSrc, ptsDst, cv::RANSAC, 3.0, inliers, 5000, 0.999);
            int inlierCount = cv::countNonZero(inliers);
            double inlierRatio = static_cast<double>(inlierCount) / static_cast<double>(ptsSrc.size());

            if (H.empty() || inlierCount < 10 || inlierRatio < 0.30) {
                outErr = QString("%1: %2/%3 inliers (%4%)")
                           .arg(tag).arg(inlierCount).arg(static_cast<int>(ptsSrc.size()))
                           .arg(static_cast<int>(inlierRatio * 100));
                return false;
            }

            if (!isHomographySane(H, imgSz)) {
                outErr = QString("%1: homography rejected (geometry sanity check failed — likely repetitive pattern)").arg(tag);
                return false;
            }

            H.convertTo(outWarp, CV_32F);
            outModel = QString("%1 %2/%3 in (%4%)")
                         .arg(tag).arg(inlierCount).arg(static_cast<int>(ptsSrc.size()))
                         .arg(static_cast<int>(inlierRatio * 100));
            std::cerr << "[align] " << tag.toStdString()
                      << " H =\n" << H << "\n"
                      << "  inliers " << inlierCount << "/" << ptsSrc.size()
                      << ", img " << imgSz.width << "x" << imgSz.height << std::endl;
            return true;
        };

        try {
            auto sift = cv::SIFT::create(0, 3, 0.04, 10.0, 1.6);
            cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(5);
            cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
            cv::Ptr<cv::DescriptorMatcher> flann = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);

            success = runWithDetector(sift, flann, 0.75f, "SIFT", errorMsg, modelUsed, warpMatrix);

            if (!success) {
                auto orb = cv::ORB::create(4000);
                cv::Ptr<cv::DescriptorMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING, false);
                QString orbErr, orbModel;
                cv::Mat orbWarp;
                if (runWithDetector(orb, bf, 0.75f, "ORB", orbErr, orbModel, orbWarp)) {
                    success = true;
                    warpMatrix = orbWarp;
                    modelUsed = orbModel;
                } else {
                    errorMsg += " | " + orbErr;
                }
            }
        }
        catch (const cv::Exception& e) {
            errorMsg = QString("Feature stage failed: ") + QString::fromStdString(e.what());
        }

        cv::Mat resultMatrix = success ? warpMatrix : cv::Mat();
        QMetaObject::invokeMethod(this, [this, success, resultMatrix, errorMsg, modelUsed]() {
            if (success) {
                m_eccWarpMatrix = resultMatrix;
                m_isAligned = true;
                setPillState(m_eccIndicator, "ok", "MATRIX READY");
                m_statusBar->showMessage("Alignment OK [" + modelUsed + "]", 3000);
            }
            else {
                m_eccWarpMatrix.release();
                m_isAligned = false;
                setPillState(m_eccIndicator, "err", "NOT CALIBRATED");
                m_statusBar->showMessage("Alignment failed (low texture / too different views): " + errorMsg, 5000);
            }
            m_btnCalibrateAlign->setEnabled(true);
            m_calibrating = false;
            updateEccPill();
        });
    });
}

double MainWindow::calculateFocus(const cv::Mat& frame)
{
    if (frame.empty()) return 0.0;
    cv::Mat gray, lap;
    if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else gray = frame;

    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev.val[0] * stddev.val[0];
}

void MainWindow::pushWorkerParams()
{
    if (!m_worker) return;
    WorkerParams p;
    p.colorMode         = m_colorMode;
    p.flipHor2          = m_chkFlipHor2 && m_chkFlipHor2->isChecked();
    p.flipVer2          = m_chkFlipVer2 && m_chkFlipVer2->isChecked();
    p.motionThr         = m_motionThreshold;
    p.bufferSize        = m_bufferSize;
    p.applyBilateral    = m_chkBilateral && m_chkBilateral->isChecked();
    p.bilateralStrength = m_bilateralStrength;
    p.noiseFloor        = m_noiseFloor;
    m_worker->setParams(p);
}

void MainWindow::displayMat(GpuImageView* view, const cv::Mat& mat)
{
    if (mat.empty() || view == nullptr) return;

    cv::Mat src = mat;
    const int targetW = std::max(1, view->width());
    const int targetH = std::max(1, view->height());
    if (src.cols > targetW * 2 || src.rows > targetH * 2) {
        double k = std::min(static_cast<double>(targetW) / src.cols,
                            static_cast<double>(targetH) / src.rows);
        if (k > 0.0 && k < 1.0) {
            cv::Mat small;
            cv::resize(src, small, cv::Size(), k, k, cv::INTER_AREA);
            src = small;
        }
    }

    if (src.channels() == 1) {
        QImage img(src.data, src.cols, src.rows, static_cast<int>(src.step), QImage::Format_Grayscale8);
        view->setImage(img.copy());
    }
    else if (src.channels() == 3) {
        cv::Mat rgb;
        cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
        QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
        view->setImage(img.copy());
    }
}

QString MainWindow::buildSnapshotBaseName(const QString& prefix) const
{
    QString user = m_snapshotNameEdit ? m_snapshotNameEdit->text().trimmed() : QString();
    QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");

    QStringList parts;
    auto enabled = [this](const QString& k, bool dflt = true) {
        return m_paramInName.value(k, dflt);
    };

    if (m_chkAppendParams && m_chkAppendParams->isChecked()) {
        if (enabled("BF") && m_chkBilateral && m_chkBilateral->isChecked()) {
            parts << QString("BF%1").arg(m_bilateralSlider ? m_bilateralSlider->value() : m_bilateralStrength);
        }
        if (enabled("TB") && m_bufferSlider && m_bufferSlider->value() > 1) {
            parts << QString("TB%1").arg(m_bufferSlider->value());
        }
        if (enabled("MT") && m_motionThresholdSlider && m_motionThresholdSlider->value() > 0) {
            parts << QString("MT%1").arg(m_motionThresholdSlider->value());
        }
        if (enabled("FU") && m_chkFusion && m_chkFusion->isChecked()) parts << "FU";
        if (enabled("ECC") && m_isAligned && m_chkAlign && m_chkAlign->isChecked()) parts << "ECC";
        if (enabled("NF") && m_noiseFloorSlider && m_noiseFloorSlider->value() > 0) {
            parts << QString("NF%1").arg(m_noiseFloorSlider->value());
        }
        if (enabled("ST") && m_chkStretch && m_chkStretch->isChecked()) parts << "ST";
        if (enabled("G") && m_manualExposure) {
            if (m_perCameraExposure) {
                parts << QString("G1_%1x").arg(m_gainQ8  / 256.0, 0, 'f', 2);
                parts << QString("G2_%1x").arg(m_gain2Q8 / 256.0, 0, 'f', 2);
            } else {
                parts << QString("G%1x").arg(m_gainQ8 / 256.0, 0, 'f', 2);
            }
        }
        if (enabled("SH") && m_manualExposure) {
            if (m_perCameraExposure) {
                parts << QString("SH1_%1us").arg(m_shutterUs);
                parts << QString("SH2_%1us").arg(m_shutter2Us);
            } else {
                parts << QString("SH%1us").arg(m_shutterUs);
            }
        }
        if (enabled("CM")) {
            switch (m_colorMode) {
                case ColorMode::COLOR:      parts << "RGB"; break;
                case ColorMode::GRAY_CV:    parts << "GRY"; break;
                case ColorMode::GRAY_NATIVE: parts << "GRYN"; break;
            }
        }
        if (enabled("FH") && m_chkFlipHor2 && m_chkFlipHor2->isChecked()) parts << "FH";
        if (enabled("FV") && m_chkFlipVer2 && m_chkFlipVer2->isChecked()) parts << "FV";
    }

    QString result = prefix;
    if (!user.isEmpty()) result += "_" + user;
    if (!parts.isEmpty()) result += "_" + parts.join("_");
    result += "_" + stamp;
    return result;
}

void MainWindow::showFilenameParamsDialog()
{
    QDialog dlg(this);
    dlg.setWindowTitle("Filename parameters");
    dlg.setStyleSheet(styleSheetText());
    dlg.resize(440, 520);

    QVBoxLayout* root = new QVBoxLayout(&dlg);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    QLabel* title = new QLabel("Filename parameters", &dlg);
    title->setProperty("role", "title");
    title->setStyleSheet(QString("padding: 16px 16px 4px 16px;"));
    root->addWidget(title);

    QLabel* hint = new QLabel("Only checked parameters appear in the filename, and only when their effect is active.", &dlg);
    hint->setProperty("role", "faint");
    hint->setStyleSheet("padding: 0px 16px 12px 16px;");
    hint->setWordWrap(true);
    root->addWidget(hint);

    QScrollArea* scroll = new QScrollArea(&dlg);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setStyleSheet("QScrollArea { background: transparent; } QWidget#fpScrollContent { background: transparent; }");

    QWidget* scrollContent = new QWidget();
    scrollContent->setObjectName("fpScrollContent");
    QVBoxLayout* listLay = new QVBoxLayout(scrollContent);
    listLay->setContentsMargins(20, 0, 20, 20);
    listLay->setSpacing(6);

    QMap<QString, QCheckBox*> boxes;
    for (const auto& spec : kFilenameParamSpecs) {
        QString key = QString::fromLatin1(spec.key);
        QFrame* row = new QFrame();
        row->setProperty("role", "panel");
        QHBoxLayout* rowLay = new QHBoxLayout(row);
        rowLay->setContentsMargins(12, 6, 12, 6);

        QCheckBox* cb = new QCheckBox(spec.label, row);
        cb->setChecked(m_paramInName.value(key, true));
        cb->setStyleSheet("font-size: 13px;");
        rowLay->addWidget(cb, 1);

        listLay->addWidget(row);
        boxes.insert(key, cb);
    }
    listLay->addStretch();

    scroll->setWidget(scrollContent);
    root->addWidget(scroll, 1);

    QFrame* bottomBar = new QFrame(&dlg);

    bottomBar->setStyleSheet(QString("background: %1; border-top: 1px solid %2;")
                             .arg(T::bg2).arg(T::border));
    QHBoxLayout* btns = new QHBoxLayout(bottomBar);
    btns->setContentsMargins(16, 10, 16, 10);

    QPushButton* btnAll  = new QPushButton("All", bottomBar);
    btnAll->setProperty("kind", "ghost");
    QPushButton* btnNone = new QPushButton("None", bottomBar);
    btnNone->setProperty("kind", "ghost");
    QPushButton* btnOk   = new QPushButton("OK", bottomBar);
    btnOk->setProperty("kind", "primary");
    QPushButton* btnCancel = new QPushButton("Cancel", bottomBar);
    btnCancel->setProperty("kind", "ghost");
    btns->addWidget(btnAll);
    btns->addWidget(btnNone);
    btns->addStretch();
    btns->addWidget(btnCancel);
    btns->addWidget(btnOk);
    root->addWidget(bottomBar);

    connect(btnAll, &QPushButton::clicked, &dlg, [&boxes]() {
        for (auto* cb : boxes) cb->setChecked(true);
    });
    connect(btnNone, &QPushButton::clicked, &dlg, [&boxes]() {
        for (auto* cb : boxes) cb->setChecked(false);
    });
    connect(btnCancel, &QPushButton::clicked, &dlg, &QDialog::reject);
    connect(btnOk, &QPushButton::clicked, &dlg, &QDialog::accept);

    if (m_animDialogsEnabled) {
        animateDialogEntry(&dlg, m_btnConfigParams, m_animSpeedMs);
    }
    if (dlg.exec() != QDialog::Accepted) return;

    for (auto it = boxes.constBegin(); it != boxes.constEnd(); ++it) {
        m_paramInName[it.key()] = it.value()->isChecked();
    }

    QSettings s(settingsPath(), QSettings::IniFormat);
    s.beginGroup("FilenameParams");
    for (auto it = m_paramInName.constBegin(); it != m_paramInName.constEnd(); ++it) {
        s.setValue(it.key(), it.value());
    }
    s.endGroup();

    if (m_statusBar) m_statusBar->showMessage("Filename parameter selection saved.", 3000);
}

void MainWindow::showExposureDialog()
{
    if (!m_manualExposure) return;

    QDialog dlg(this);
    dlg.setWindowTitle("Manual exposure");
    dlg.setStyleSheet(styleSheetText());
    dlg.resize(480, 320);

    QVBoxLayout* root = new QVBoxLayout(&dlg);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(10);

    QCheckBox* chkPerCam = new QCheckBox("Per-camera exposure (separate Cam1 / Cam2)", &dlg);
    chkPerCam->setChecked(m_perCameraExposure);
    root->addWidget(chkPerCam);

    QFrame* camToggleRow = new QFrame(&dlg);
    camToggleRow->setProperty("role", "panel");
    QHBoxLayout* camToggleLay = new QHBoxLayout(camToggleRow);
    camToggleLay->setContentsMargins(12, 6, 12, 6);
    camToggleLay->setSpacing(6);
    QLabel* camLbl = new QLabel("CAMERA", camToggleRow);
    camLbl->setProperty("role", "section");
    camToggleLay->addWidget(camLbl);
    QPushButton* btnCam1 = new QPushButton("Cam1", camToggleRow);
    QPushButton* btnCam2 = new QPushButton("Cam2", camToggleRow);
    btnCam1->setCheckable(true);
    btnCam2->setCheckable(true);
    btnCam1->setCursor(Qt::PointingHandCursor);
    btnCam2->setCursor(Qt::PointingHandCursor);

    QString camBtnSty = QString(
        "QPushButton { background: %1; color: %2; border: 1px solid %3; border-radius: 0px;"
        " padding: 4px 14px; font-weight: 600; font-size: 11px;"
        " font-family: 'Space Mono','JetBrains Mono',monospace; letter-spacing: 0.06em; }"
        "QPushButton:checked { background: %4; color: %2; border-color: %4; }")
        .arg(T::bg2).arg(T::text).arg(T::border).arg(T::accent);
    btnCam1->setStyleSheet(camBtnSty);
    btnCam2->setStyleSheet(camBtnSty);
    btnCam1->setChecked(true);
    camToggleLay->addWidget(btnCam1);
    camToggleLay->addWidget(btnCam2);
    camToggleLay->addStretch();
    root->addWidget(camToggleRow);
    camToggleRow->setVisible(m_perCameraExposure);

    auto buildRow = [&dlg](const QString& title, QWidget* spin, QSlider* slider) -> QFrame* {
        QFrame* row = new QFrame(&dlg);
        row->setProperty("role", "panel");
        QVBoxLayout* lay = new QVBoxLayout(row);
        lay->setContentsMargins(14, 10, 14, 10);
        lay->setSpacing(6);

        QHBoxLayout* head = new QHBoxLayout();
        QLabel* lbl = new QLabel(title.toUpper(), row);
        lbl->setProperty("role", "section");
        head->addWidget(lbl);
        head->addStretch();
        head->addWidget(spin);
        lay->addLayout(head);
        lay->addWidget(slider);
        return row;
    };

    QDoubleSpinBox* spnG = new QDoubleSpinBox(&dlg);
    spnG->setRange(0.1, 10.0);
    spnG->setDecimals(2);
    spnG->setSingleStep(0.1);
    spnG->setSuffix("x");
    spnG->setKeyboardTracking(false);
    spnG->setFixedWidth(100);

    QSlider* sldG = new QSlider(Qt::Horizontal, &dlg);
    sldG->setRange(10, 1000);
    sldG->setSingleStep(10);
    sldG->setPageStep(50);
    sldG->setMinimumWidth(280);

    QSpinBox* spnS = new QSpinBox(&dlg);
    spnS->setRange(50, 30000);
    spnS->setSingleStep(100);
    spnS->setSuffix(" us");
    spnS->setKeyboardTracking(false);
    spnS->setFixedWidth(120);

    QSlider* sldS = new QSlider(Qt::Horizontal, &dlg);
    sldS->setRange(50, 30000);
    sldS->setSingleStep(100);
    sldS->setPageStep(1000);
    sldS->setMinimumWidth(280);

    QFrame* rowG = buildRow("Gain (0.10x – 10.00x)",   spnG, sldG);
    QFrame* rowS = buildRow("Shutter (50 – 30000 µs)", spnS, sldS);
    root->addWidget(rowG);
    root->addWidget(rowS);

    bool* activeCam2 = new bool(false);
    dlg.connect(&dlg, &QDialog::destroyed, [activeCam2]{ delete activeCam2; });

    auto loadValuesIntoEditors = [this, spnG, sldG, spnS, sldS, activeCam2]() {
        double g; int sh;
        if (m_perCameraExposure && *activeCam2) {
            g = m_gain2Q8 / 256.0;  sh = m_shutter2Us;
        } else {
            g = m_gainQ8  / 256.0;  sh = m_shutterUs;
        }
        g  = qBound(0.1, g, 10.0);
        sh = qBound(50, sh, 30000);
        QSignalBlocker bg(spnG), bgs(sldG), bs(spnS), bss(sldS);
        spnG->setValue(g);
        sldG->setValue(static_cast<int>(g * 100.0 + 0.5));
        spnS->setValue(sh);
        sldS->setValue(sh);
    };
    loadValuesIntoEditors();

    auto pushFromEditors = [this, spnG, spnS, activeCam2]() {
        int  gQ8 = static_cast<int>(spnG->value() * 256.0 + 0.5);
        int  shU = spnS->value();
        if (m_perCameraExposure && *activeCam2) {
            m_gain2Q8    = gQ8;
            m_shutter2Us = shU;

            if (m_spnGain2)    { QSignalBlocker b(m_spnGain2);    m_spnGain2->setValue(gQ8 / 256.0); }
            if (m_sldGain2)    { QSignalBlocker b(m_sldGain2);    m_sldGain2->setValue(static_cast<int>((gQ8 / 256.0) * 100.0 + 0.5)); }
            if (m_spnShutter2) { QSignalBlocker b(m_spnShutter2); m_spnShutter2->setValue(shU); }
            if (m_sldShutter2) { QSignalBlocker b(m_sldShutter2); m_sldShutter2->setValue(shU); }
        } else {
            m_gainQ8    = gQ8;
            m_shutterUs = shU;
            if (m_spnGain)    { QSignalBlocker b(m_spnGain);    m_spnGain->setValue(gQ8 / 256.0); }
            if (m_sldGain)    { QSignalBlocker b(m_sldGain);    m_sldGain->setValue(static_cast<int>((gQ8 / 256.0) * 100.0 + 0.5)); }
            if (m_spnShutter) { QSignalBlocker b(m_spnShutter); m_spnShutter->setValue(shU); }
            if (m_sldShutter) { QSignalBlocker b(m_sldShutter); m_sldShutter->setValue(shU); }
        }

        if (m_lblExposureVals) {
            double g1 = m_gainQ8 / 256.0;
            int    s1 = m_shutterUs;
            if (m_perCameraExposure) {
                double g2 = m_gain2Q8 / 256.0;
                int    s2 = m_shutter2Us;
                m_lblExposureVals->setText(QString("%1x * %2 // %3x * %4")
                    .arg(g1, 0, 'f', 2).arg(s1).arg(g2, 0, 'f', 2).arg(s2));
            } else {
                m_lblExposureVals->setText(QString("%1x * %2").arg(g1, 0, 'f', 2).arg(s1));
            }
        }
        applyExposureControls();
    };

    connect(spnG, QOverload<double>::of(&QDoubleSpinBox::valueChanged), &dlg, [sldG, pushFromEditors](double v) {
        QSignalBlocker b(sldG);
        sldG->setValue(qBound(sldG->minimum(), static_cast<int>(v * 100.0 + 0.5), sldG->maximum()));
        pushFromEditors();
    });
    connect(sldG, &QSlider::valueChanged, &dlg, [spnG, pushFromEditors](int v) {
        QSignalBlocker b(spnG);
        spnG->setValue(v / 100.0);
        pushFromEditors();
    });
    connect(spnS, QOverload<int>::of(&QSpinBox::valueChanged), &dlg, [sldS, pushFromEditors](int v) {
        QSignalBlocker b(sldS);
        sldS->setValue(qBound(sldS->minimum(), v, sldS->maximum()));
        pushFromEditors();
    });
    connect(sldS, &QSlider::valueChanged, &dlg, [spnS, pushFromEditors](int v) {
        QSignalBlocker b(spnS);
        spnS->setValue(v);
        pushFromEditors();
    });

    auto selectCam = [activeCam2, btnCam1, btnCam2, loadValuesIntoEditors](bool cam2) {
        *activeCam2 = cam2;
        QSignalBlocker b1(btnCam1), b2(btnCam2);
        btnCam1->setChecked(!cam2);
        btnCam2->setChecked(cam2);
        loadValuesIntoEditors();
    };
    connect(btnCam1, &QPushButton::clicked, &dlg, [selectCam]() { selectCam(false); });
    connect(btnCam2, &QPushButton::clicked, &dlg, [selectCam]() { selectCam(true);  });

    connect(chkPerCam, &QCheckBox::toggled, &dlg, [this, camToggleRow, selectCam](bool on) {
        m_perCameraExposure = on;
        camToggleRow->setVisible(on);
        selectCam(false);
        applyExposureControls();
    });

    QHBoxLayout* btns = new QHBoxLayout();
    btns->addStretch();
    QPushButton* btnClose = new QPushButton("Close", &dlg);
    btnClose->setProperty("kind", "primary");
    btns->addWidget(btnClose);
    root->addLayout(btns);
    connect(btnClose, &QPushButton::clicked, &dlg, &QDialog::accept);

    if (m_animDialogsEnabled) {
        animateDialogEntry(&dlg, m_btnExpChange, m_animSpeedMs);
        dlg.exec();
    } else {
        dlg.exec();
    }
}

void MainWindow::saveSnapshot()
{
    if (m_isDiffMode) {
        saveDiffSnapshot();
    } else {
        saveDualSnapshot(true);
    }
}

static bool saveWithExif(const QString& path, const cv::Mat& mat, const ExifParams& params) {
    std::vector<uint8_t> buf;
    if (!cv::imencode(".jpg", mat, buf)) return false;

    std::vector<uint8_t> finalData = insertExif(buf, params);

    QFile f(path);
    if (f.open(QIODevice::WriteOnly)) {
        f.write(reinterpret_cast<const char*>(finalData.data()), finalData.size());
        f.close();
        return true;
    }
    return false;
}

void MainWindow::saveDiffSnapshot()
{
    if (m_lastDiffResult.empty()) {
        m_statusBar->showMessage("No difference image available to save.", 3000);
        return;
    }

    QString metricsDir = QCoreApplication::applicationDirPath() + "/metrics";
    QDir().mkpath(metricsDir);

    QString filename = buildSnapshotBaseName("diff");
    QString filePath = metricsDir + "/" + filename + ".jpg";

    ExifParams p = buildExifParams("diff");
    bool success = saveWithExif(filePath, m_lastDiffResult, p);

    if (success) {
        m_statusBar->showMessage("Saved snapshot: " + filePath, 4000);
    }
    else {
        m_statusBar->showMessage("Error saving snapshot to: " + filePath, 4000);
    }
}

void MainWindow::saveDualSnapshot(bool combined)
{
    if (m_frame1.empty() || m_frame2.empty()) {
        m_statusBar->showMessage("No frames available to save.", 3000);
        return;
    }

    QString metricsDir = QCoreApplication::applicationDirPath() + "/metrics";
    QDir().mkpath(metricsDir);

    QString baseName = buildSnapshotBaseName("dual");

    cv::Mat f1 = m_frame1.clone();
    cv::Mat f2 = m_frame2.clone();
    if (f1.channels() == 1) cv::cvtColor(f1, f1, cv::COLOR_GRAY2BGR);
    if (f2.channels() == 1) cv::cvtColor(f2, f2, cv::COLOR_GRAY2BGR);

    if (combined) {
        if (f1.rows != f2.rows) {
            int newW = f2.cols * f1.rows / f2.rows;
            cv::resize(f2, f2, cv::Size(newW, f1.rows));
        }
        cv::Mat combo;
        cv::hconcat(f1, f2, combo);
        QString filename = baseName;
        QString path = metricsDir + "/" + filename + ".jpg";
        ExifParams p = buildExifParams("dual_combined");
        bool ok = saveWithExif(path, combo, p);
        m_statusBar->showMessage(ok ? "Saved: " + path : "Error saving: " + path, 4000);
    } else {
        QString filename1 = baseName + "_cam1";
        QString filename2 = baseName + "_cam2";
        QString path1 = metricsDir + "/" + filename1 + ".jpg";
        QString path2 = metricsDir + "/" + filename2 + ".jpg";

        ExifParams p1 = buildExifParams("dual_cam1");
        ExifParams p2 = buildExifParams("dual_cam2");

        bool ok1 = saveWithExif(path1, f1, p1);
        bool ok2 = saveWithExif(path2, f2, p2);
        if (ok1 && ok2)
            m_statusBar->showMessage("Saved cam1 and cam2 snapshots.", 4000);
        else
            m_statusBar->showMessage("Error saving dual snapshots.", 4000);
    }
}

ExifParams MainWindow::buildExifParams(const QString& mode) const
{
    ExifParams p;
    p.make = "DualCamQt";
    p.model = mode.toStdString();
    p.software = "DualCam Analysis Tool";

    QJsonObject obj;
    obj["mode"]      = mode;
    obj["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    obj["userName"]  = m_snapshotNameEdit ? m_snapshotNameEdit->text().trimmed() : QString();

    QString colorStr;
    switch (m_colorMode) {
        case ColorMode::COLOR:       colorStr = "RGB";        break;
        case ColorMode::GRAY_CV:     colorStr = "GRAY_CV";    break;
        case ColorMode::GRAY_NATIVE: colorStr = "GRAY_NATIVE"; break;
    }
    obj["colorMode"] = colorStr;
    obj["flipHorizontal2"] = m_chkFlipHor2 && m_chkFlipHor2->isChecked();
    obj["flipVertical2"]   = m_chkFlipVer2 && m_chkFlipVer2->isChecked();

    obj["timeBuffer"]      = m_bufferSlider ? m_bufferSlider->value() : 0;
    obj["motionThreshold"] = m_motionThresholdSlider ? m_motionThresholdSlider->value() : 0;
    obj["fusion"]          = m_chkFusion && m_chkFusion->isChecked();
    obj["bilateralFilter"] = m_chkBilateral && m_chkBilateral->isChecked();
    obj["bilateralStrength"] = m_bilateralSlider ? m_bilateralSlider->value() : m_bilateralStrength;
    obj["noiseFloor"]      = m_noiseFloorSlider ? m_noiseFloorSlider->value() : m_noiseFloor;
    obj["intensityStretch"] = m_chkStretch && m_chkStretch->isChecked();
    obj["trackPeaks"]      = m_btnPeakIntensities && m_btnPeakIntensities->isChecked();

    QJsonObject align;
    align["enabled"]    = m_chkAlign && m_chkAlign->isChecked();
    align["calibrated"] = m_isAligned;
    align["activeCam"]  = m_activeAdjCam;
    auto adjToJson = [](const ManualAdjust& a) {
        QJsonObject o;
        o["tx"] = a.tx; o["ty"] = a.ty; o["scale"] = a.scale;
        o["rx"] = a.rx; o["ry"] = a.ry; o["rz"] = a.rz;
        return o;
    };
    align["manualCam1"] = adjToJson(m_manualAdj1);
    align["manualCam2"] = adjToJson(m_manualAdj2);
    obj["alignment"] = align;

    QJsonObject exposure;
    exposure["mode"]       = m_manualExposure ? "manual" : "auto";
    exposure["perCamera"]  = m_perCameraExposure;
    if (m_perCameraExposure) {
        QJsonObject c1; c1["gain"] = m_gainQ8  / 256.0; c1["shutterUs"] = m_shutterUs;
        QJsonObject c2; c2["gain"] = m_gain2Q8 / 256.0; c2["shutterUs"] = m_shutter2Us;
        exposure["cam1"] = c1;
        exposure["cam2"] = c2;
    } else {
        exposure["gain"]      = m_gainQ8 / 256.0;
        exposure["shutterUs"] = m_shutterUs;
    }
    obj["exposure"] = exposure;

    QJsonObject focus;
    focus["cam1"] = m_lastFocus1;
    focus["cam2"] = m_lastFocus2;
    obj["focus"] = focus;
    obj["motionActive"] = m_motionActive;

    obj["diffMode"]   = m_isDiffMode;
    obj["frameCount"] = m_frameCount;

    QJsonDocument doc(obj);
    p.description = doc.toJson(QJsonDocument::Compact).toStdString();

    p.exposureTimeNum = m_shutterUs;
    p.exposureTimeDen = 1000000;
    p.isoSpeed = static_cast<uint16_t>(m_gainQ8 * 100 / 256);

    return p;
}

void MainWindow::saveSettings()
{
    QSettings s(settingsPath(), QSettings::IniFormat);
    s.setValue("colorMode", m_comboColorMode->currentIndex());
    s.setValue("flipVer2", m_chkFlipVer2->isChecked());
    s.setValue("flipHor2", m_chkFlipHor2->isChecked());
    s.setValue("align", m_chkAlign->isChecked());
    s.setValue("bufferSize", m_bufferSlider->value());
    s.setValue("motionThreshold", m_motionThresholdSlider->value());
    s.setValue("fusion", m_chkFusion->isChecked());
    s.setValue("bilateral", m_chkBilateral->isChecked());
    s.setValue("bilateralStrength", m_bilateralSlider ? m_bilateralSlider->value() : m_bilateralStrength);
    s.setValue("noiseFloor", m_noiseFloorSlider->value());
    s.setValue("stretchIntensity", m_chkStretch->isChecked());
    s.setValue("trackPeaks", m_btnPeakIntensities->isChecked());
    s.setValue("appendParams", m_chkAppendParams ? m_chkAppendParams->isChecked() : false);
    s.beginGroup("FilenameParams");
    for (auto it = m_paramInName.constBegin(); it != m_paramInName.constEnd(); ++it) {
        s.setValue(it.key(), it.value());
    }
    s.endGroup();
    s.setValue("manualExposure", m_manualExposure);
    s.setValue("perCameraExposure", m_perCameraExposure);
    s.setValue("gainQ8", m_gainQ8);
    s.setValue("shutterUs", m_shutterUs);
    s.setValue("gain2Q8", m_gain2Q8);
    s.setValue("shutter2Us", m_shutter2Us);
    s.setValue("maxHistory", m_historySpinBox->value());
    s.setValue("diffMode", m_isDiffMode);
    s.setValue("sheetOpen", m_currentNav != NavItem::None);

    auto writeAdj = [&s](const QString& prefix, const ManualAdjust& a) {
        s.setValue(prefix + "tx", a.tx);
        s.setValue(prefix + "ty", a.ty);
        s.setValue(prefix + "scale", a.scale);
        s.setValue(prefix + "rx", a.rx);
        s.setValue(prefix + "ry", a.ry);
        s.setValue(prefix + "rz", a.rz);
    };
    writeAdj("adj1/", m_manualAdj1);
    writeAdj("adj2/", m_manualAdj2);
    s.setValue("activeAdjCam", m_activeAdjCam);
    s.setValue("animSpeedMs", m_animSpeedMs);
    s.setValue("animSidebarEnabled", m_animSidebarEnabled);
    s.setValue("animFABsEnabled", m_animFABsEnabled);
    s.setValue("animBottomPanelEnabled", m_animBottomPanelEnabled);
    s.setValue("animTabsEnabled", m_animTabsEnabled);
    s.setValue("animDialogsEnabled", m_animDialogsEnabled);
    s.sync();
}

void MainWindow::loadSettings()
{
    QSettings s(settingsPath(), QSettings::IniFormat);
    m_animSpeedMs            = s.value("animSpeedMs", 220).toInt();
    m_animSidebarEnabled     = s.value("animSidebarEnabled", true).toBool();
    m_animFABsEnabled        = s.value("animFABsEnabled", true).toBool();
    m_animBottomPanelEnabled = s.value("animBottomPanelEnabled", true).toBool();
    m_animTabsEnabled        = s.value("animTabsEnabled", true).toBool();
    m_animDialogsEnabled     = s.value("animDialogsEnabled", true).toBool();

    m_comboColorMode->setCurrentIndex(s.value("colorMode", 1).toInt());
    m_chkFlipVer2->setChecked(s.value("flipVer2", false).toBool());
    m_chkFlipHor2->setChecked(s.value("flipHor2", false).toBool());
    m_chkAlign->setChecked(s.value("align", false).toBool());
    m_bufferSlider->setValue(s.value("bufferSize", 8).toInt());
    m_motionThresholdSlider->setValue(s.value("motionThreshold", 5).toInt());
    m_chkFusion->setChecked(s.value("fusion", false).toBool());
    m_chkBilateral->setChecked(s.value("bilateral", false).toBool());
    if (m_bilateralSlider) m_bilateralSlider->setValue(s.value("bilateralStrength", 5).toInt());
    m_noiseFloorSlider->setValue(s.value("noiseFloor", 15).toInt());
    m_chkStretch->setChecked(s.value("stretchIntensity", false).toBool());
    m_btnPeakIntensities->setChecked(s.value("trackPeaks", false).toBool());
    if (m_chkAppendParams) m_chkAppendParams->setChecked(s.value("appendParams", false).toBool());
    s.beginGroup("FilenameParams");
    for (const auto& spec : kFilenameParamSpecs) {
        QString key = QString::fromLatin1(spec.key);
        m_paramInName[key] = s.value(key, true).toBool();
    }
    s.endGroup();
    m_gainQ8     = s.value("gainQ8",     256).toInt();
    m_shutterUs  = s.value("shutterUs",  10000).toInt();
    m_gain2Q8    = s.value("gain2Q8",    m_gainQ8).toInt();
    m_shutter2Us = s.value("shutter2Us", m_shutterUs).toInt();
    m_perCameraExposure = s.value("perCameraExposure", false).toBool();

    double gainVal     = qBound(0.1, m_gainQ8    / 256.0, 10.0);
    double gain2Val    = qBound(0.1, m_gain2Q8   / 256.0, 10.0);
    int    shutterVal  = qBound(50,  m_shutterUs,  30000);
    int    shutter2Val = qBound(50,  m_shutter2Us, 30000);
    m_gainQ8     = static_cast<int>(gainVal  * 256.0 + 0.5);
    m_gain2Q8    = static_cast<int>(gain2Val * 256.0 + 0.5);
    m_shutterUs  = shutterVal;
    m_shutter2Us = shutter2Val;

    if (m_spnGain)     { QSignalBlocker b(m_spnGain);     m_spnGain->setValue(gainVal); }
    if (m_sldGain)     { QSignalBlocker b(m_sldGain);     m_sldGain->setValue(static_cast<int>(gainVal * 100.0 + 0.5)); }
    if (m_spnShutter)  { QSignalBlocker b(m_spnShutter);  m_spnShutter->setValue(shutterVal); }
    if (m_sldShutter)  { QSignalBlocker b(m_sldShutter);  m_sldShutter->setValue(shutterVal); }
    if (m_spnGain2)    { QSignalBlocker b(m_spnGain2);    m_spnGain2->setValue(gain2Val); }
    if (m_sldGain2)    { QSignalBlocker b(m_sldGain2);    m_sldGain2->setValue(static_cast<int>(gain2Val * 100.0 + 0.5)); }
    if (m_spnShutter2) { QSignalBlocker b(m_spnShutter2); m_spnShutter2->setValue(shutter2Val); }
    if (m_sldShutter2) { QSignalBlocker b(m_sldShutter2); m_sldShutter2->setValue(shutter2Val); }

    bool me = s.value("manualExposure", false).toBool();
    if (m_btnExpToggle) m_btnExpToggle->setChecked(me);
    int maxH = s.value("maxHistory", 200).toInt();
    m_historySpinBox->setValue(maxH);
    m_historySlider->setValue(maxH);
    m_maxHistory = maxH;

    bool diffMode = s.value("diffMode", false).toBool();
    setDiffMode(diffMode);

    bool sheetOpen = s.value("sheetOpen", true).toBool();
    if (sheetOpen != (m_currentNav != NavItem::None)) toggleSheet();

    auto readAdj = [&s](const QString& prefix, ManualAdjust& a) {
        a.tx    = s.value(prefix + "tx",    0.0).toDouble();
        a.ty    = s.value(prefix + "ty",    0.0).toDouble();
        a.scale = s.value(prefix + "scale", 1.0).toDouble();
        a.rx    = s.value(prefix + "rx",    0.0).toDouble();
        a.ry    = s.value(prefix + "ry",    0.0).toDouble();
        a.rz    = s.value(prefix + "rz",    0.0).toDouble();
    };
    readAdj("adj1/", m_manualAdj1);
    readAdj("adj2/", m_manualAdj2);
    m_activeAdjCam = s.value("activeAdjCam", 2).toInt();
    if (m_comboAdjCam) {
        QSignalBlocker b(m_comboAdjCam);
        m_comboAdjCam->setCurrentIndex(m_activeAdjCam - 1);
    }
    applyAdjustToWidgets();

    updateEccPill();
    updateFpsPill();
}

void MainWindow::savePreset(const QString& name)
{
    if (name.isEmpty()) return;
    QSettings s(settingsPath(), QSettings::IniFormat);
    s.beginGroup("presets");
    s.beginGroup(name);
    s.setValue("colorMode", m_comboColorMode->currentIndex());
    s.setValue("flipVer2", m_chkFlipVer2->isChecked());
    s.setValue("flipHor2", m_chkFlipHor2->isChecked());
    s.setValue("align", m_chkAlign->isChecked());
    s.setValue("bufferSize", m_bufferSlider->value());
    s.setValue("motionThreshold", m_motionThresholdSlider->value());
    s.setValue("fusion", m_chkFusion->isChecked());
    s.setValue("bilateral", m_chkBilateral->isChecked());
    s.setValue("bilateralStrength", m_bilateralSlider ? m_bilateralSlider->value() : m_bilateralStrength);
    s.setValue("noiseFloor", m_noiseFloorSlider->value());
    s.setValue("stretchIntensity", m_chkStretch->isChecked());
    s.setValue("trackPeaks", m_btnPeakIntensities->isChecked());
    s.setValue("diffMode", m_isDiffMode);
    auto writeAdj = [&s](const QString& prefix, const ManualAdjust& a) {
        s.setValue(prefix + "tx", a.tx);
        s.setValue(prefix + "ty", a.ty);
        s.setValue(prefix + "scale", a.scale);
        s.setValue(prefix + "rx", a.rx);
        s.setValue(prefix + "ry", a.ry);
        s.setValue(prefix + "rz", a.rz);
    };
    writeAdj("adj1_", m_manualAdj1);
    writeAdj("adj2_", m_manualAdj2);
    s.endGroup();
    s.endGroup();
    refreshPresetList();
}

void MainWindow::loadPreset(const QString& name)
{
    if (name.isEmpty()) return;
    QSettings s(settingsPath(), QSettings::IniFormat);
    s.beginGroup("presets");
    s.beginGroup(name);
    m_comboColorMode->setCurrentIndex(s.value("colorMode", 1).toInt());
    m_chkFlipVer2->setChecked(s.value("flipVer2", false).toBool());
    m_chkFlipHor2->setChecked(s.value("flipHor2", false).toBool());
    m_chkAlign->setChecked(s.value("align", false).toBool());
    m_bufferSlider->setValue(s.value("bufferSize", 8).toInt());
    m_motionThresholdSlider->setValue(s.value("motionThreshold", 5).toInt());
    m_chkFusion->setChecked(s.value("fusion", false).toBool());
    m_chkBilateral->setChecked(s.value("bilateral", false).toBool());
    if (m_bilateralSlider) m_bilateralSlider->setValue(s.value("bilateralStrength", 5).toInt());
    m_noiseFloorSlider->setValue(s.value("noiseFloor", 15).toInt());
    m_chkStretch->setChecked(s.value("stretchIntensity", false).toBool());
    m_btnPeakIntensities->setChecked(s.value("trackPeaks", false).toBool());
    bool diffMode = s.value("diffMode", m_isDiffMode).toBool();
    setDiffMode(diffMode);
    auto readAdj = [&s](const QString& prefix, ManualAdjust& a) {
        a.tx    = s.value(prefix + "tx",    0.0).toDouble();
        a.ty    = s.value(prefix + "ty",    0.0).toDouble();
        a.scale = s.value(prefix + "scale", 1.0).toDouble();
        a.rx    = s.value(prefix + "rx",    0.0).toDouble();
        a.ry    = s.value(prefix + "ry",    0.0).toDouble();
        a.rz    = s.value(prefix + "rz",    0.0).toDouble();
    };
    readAdj("adj1_", m_manualAdj1);
    readAdj("adj2_", m_manualAdj2);
    s.endGroup();
    s.endGroup();
    applyAdjustToWidgets();
    updateView();
    m_statusBar->showMessage("Preset '" + name + "' loaded.", 3000);
}

void MainWindow::deletePreset(const QString& name)
{
    if (name.isEmpty()) return;
    QSettings s(settingsPath(), QSettings::IniFormat);
    s.beginGroup("presets");
    s.beginGroup(name);
    s.remove("");
    s.endGroup();
    s.endGroup();
    refreshPresetList();
    m_statusBar->showMessage("Preset '" + name + "' deleted.", 3000);
}

void MainWindow::refreshPresetList()
{
    QSettings s(settingsPath(), QSettings::IniFormat);
    s.beginGroup("presets");
    QStringList presets = s.childGroups();
    s.endGroup();
    if (m_presetList) {
        m_presetList->clear();
        m_presetList->addItems(presets);
    }
}

cv::Mat MainWindow::buildManualHomography(const ManualAdjust& a, const cv::Size& sz) const
{
    const double w = sz.width;
    const double h = sz.height;
    const double cx = w * 0.5;
    const double cy = h * 0.5;
    const double f = std::max(w, h);

    const double rx = a.rx * CV_PI / 180.0;
    const double ry = a.ry * CV_PI / 180.0;
    const double rz = a.rz * CV_PI / 180.0;

    cv::Matx33d Rx(1, 0, 0,
                   0, std::cos(rx), -std::sin(rx),
                   0, std::sin(rx),  std::cos(rx));
    cv::Matx33d Ry( std::cos(ry), 0, std::sin(ry),
                    0,            1, 0,
                   -std::sin(ry), 0, std::cos(ry));
    cv::Matx33d Rz(std::cos(rz), -std::sin(rz), 0,
                   std::sin(rz),  std::cos(rz), 0,
                   0,             0,            1);
    cv::Matx33d R = Rz * Ry * Rx;

    cv::Matx33d K(f, 0, cx,
                  0, f, cy,
                  0, 0, 1);
    cv::Matx33d Kinv = K.inv();
    cv::Matx33d H3D = K * R * Kinv;

    cv::Matx33d Tc1(1, 0, -cx, 0, 1, -cy, 0, 0, 1);
    cv::Matx33d Sc(a.scale, 0, 0, 0, a.scale, 0, 0, 0, 1);
    cv::Matx33d Tc2(1, 0,  cx, 0, 1,  cy, 0, 0, 1);
    cv::Matx33d S = Tc2 * Sc * Tc1;

    cv::Matx33d Tx(1, 0, a.tx, 0, 1, a.ty, 0, 0, 1);

    return cv::Mat(Tx * S * H3D);
}

ManualAdjust& MainWindow::activeManualAdjust()
{
    return (m_activeAdjCam == 1) ? m_manualAdj1 : m_manualAdj2;
}

void MainWindow::applyAdjustToWidgets()
{
    if (!m_spnAdjTx) return;
    m_updatingAdjUI = true;
    const ManualAdjust& a = activeManualAdjust();

    m_spnAdjTx->setValue(a.tx);
    m_sldAdjTx->setValue(static_cast<int>(a.tx));

    m_spnAdjTy->setValue(a.ty);
    m_sldAdjTy->setValue(static_cast<int>(a.ty));

    m_spnAdjScale->setValue(a.scale);
    m_sldAdjScale->setValue(static_cast<int>(a.scale * 100.0));

    m_spnAdjRx->setValue(a.rx);
    m_sldAdjRx->setValue(static_cast<int>(a.rx * 10.0));

    m_spnAdjRy->setValue(a.ry);
    m_sldAdjRy->setValue(static_cast<int>(a.ry * 10.0));

    m_spnAdjRz->setValue(a.rz);
    m_sldAdjRz->setValue(static_cast<int>(a.rz * 10.0));

    m_updatingAdjUI = false;
}

void MainWindow::applyWidgetsToAdjust()
{
    if (m_updatingAdjUI) return;
    ManualAdjust& a = activeManualAdjust();
    a.tx    = m_spnAdjTx->value();
    a.ty    = m_spnAdjTy->value();
    a.scale = m_spnAdjScale->value();
    a.rx    = m_spnAdjRx->value();
    a.ry    = m_spnAdjRy->value();
    a.rz    = m_spnAdjRz->value();
    updateView();
}

void MainWindow::resetActiveAdjust()
{
    activeManualAdjust() = ManualAdjust{};
    applyAdjustToWidgets();
    updateView();
}

void MainWindow::initCommands()
{
    auto createPopup = [this](const QString& title, QSlider* slider) {
        if (!slider) return;

        QDialog* dlg = new QDialog(this, Qt::Popup | Qt::FramelessWindowHint | Qt::NoDropShadowWindowHint);
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->setStyleSheet(styleSheetText() + QString(
            "QDialog { background: %1; border: 1px solid %2; border-radius: 0px; }")
            .arg(T::bg1).arg(T::accent));

        QVBoxLayout* lay = new QVBoxLayout(dlg);
        lay->setContentsMargins(16, 14, 16, 14);
        lay->setSpacing(8);

        QLabel* lbl = new QLabel(title.toUpper(), dlg);
        lbl->setAlignment(Qt::AlignCenter);
        lbl->setProperty("role", "section");
        lay->addWidget(lbl);

        QSlider* sl = new QSlider(Qt::Horizontal, dlg);
        sl->setRange(slider->minimum(), slider->maximum());
        sl->setValue(slider->value());
        sl->setMinimumWidth(240);
        lay->addWidget(sl);

        QLabel* valLbl = new QLabel(QString::number(sl->value()), dlg);
        valLbl->setAlignment(Qt::AlignCenter);
        valLbl->setProperty("role", "mono");
        lay->addWidget(valLbl);

        connect(sl, &QSlider::valueChanged, slider, &QSlider::setValue);
        connect(sl, &QSlider::valueChanged, valLbl, [valLbl](int v){ valLbl->setNum(v); });

        dlg->adjustSize();
        QRect rect = this->geometry();
        dlg->move(rect.center().x() - dlg->width() / 2, rect.center().y() - dlg->height() / 2);
        dlg->show();
    };

    m_commands = {
        {"cmd_stream", "Start / Stop Streams", "Capture", CmdType::Action, [this](){ if (m_camerasOpen) closeCameras(); else openCameras(); }, {}},
        {"cmd_snapshot", "Take Snapshot", "Capture", CmdType::Action, [this](){ saveSnapshot(); refreshSnapshotPreview(); }, {}},
        {"cmd_mode", "Toggle Dual / Diff Mode", "Capture", CmdType::Action, [this](){ setDiffMode(!m_isDiffMode); }, {}},
        {"cmd_focus", "Toggle Focus View", "Capture", CmdType::Action, [this](){ toggleFocusView(); }, {}},
        {"cmd_sheet",   "Toggle Bottom Panel", "Capture", CmdType::Action, [this](){ toggleSheet(); }, {}},
        {"cmd_preview", "Toggle Preview Mode", "Capture", CmdType::Action, [this](){ togglePreviewMode(); }, {}},
        {"cmd_nav_collapse", "Toggle Side Rail", "Capture", CmdType::Action, [this](){ setNavExpanded(!m_navExpanded); }, {}},
        {"cmd_gallery", "Open Gallery", "Snapshot", CmdType::Action, [this](){ setNavItem(NavItem::Gallery); }, {}},
        {"cmd_minimize_dialogs", "Minimize All Dialogs", "Capture", CmdType::Action, [this](){ minimizeAllDialogs(); }, {}},
        {"cmd_color_mode", "Cycle Color Mode", "Capture", CmdType::Action, [this](){
            if (!m_comboColorMode) return;
            int n = m_comboColorMode->count();
            if (n > 0) m_comboColorMode->setCurrentIndex((m_comboColorMode->currentIndex() + 1) % n);
        }, {}},
        {"cmd_flip_h2", "Toggle Flip Horizontal (Cam2)", "Capture", CmdType::Toggle, [this](){ if (m_chkFlipHor2) m_chkFlipHor2->setChecked(!m_chkFlipHor2->isChecked()); }, {}},
        {"cmd_flip_v2", "Toggle Flip Vertical (Cam2)", "Capture", CmdType::Toggle, [this](){ if (m_chkFlipVer2) m_chkFlipVer2->setChecked(!m_chkFlipVer2->isChecked()); }, {}},

        {"cmd_exposure_toggle", "Toggle Auto / Manual Exposure", "Exposure", CmdType::Toggle, [this](){ if (m_btnExpToggle) m_btnExpToggle->setChecked(!m_btnExpToggle->isChecked()); }, {}},
        {"cmd_gain_inc", "Gain +0.1x", "Exposure", CmdType::Action, [this](){
            if (m_spnGain && m_manualExposure) m_spnGain->setValue(qMin(m_spnGain->maximum(), m_spnGain->value() + 0.1));
        }, {}},
        {"cmd_gain_dec", "Gain -0.1x", "Exposure", CmdType::Action, [this](){
            if (m_spnGain && m_manualExposure) m_spnGain->setValue(qMax(m_spnGain->minimum(), m_spnGain->value() - 0.1));
        }, {}},
        {"cmd_shutter_inc", "Shutter +500us", "Exposure", CmdType::Action, [this](){
            if (m_spnShutter && m_manualExposure) m_spnShutter->setValue(qMin(m_spnShutter->maximum(), m_spnShutter->value() + 500));
        }, {}},
        {"cmd_shutter_dec", "Shutter -500us", "Exposure", CmdType::Action, [this](){
            if (m_spnShutter && m_manualExposure) m_spnShutter->setValue(qMax(m_spnShutter->minimum(), m_spnShutter->value() - 500));
        }, {}},

        {"cmd_ecc", "Toggle Alignment", "Pipeline", CmdType::Toggle, [this](){ if (m_chkAlign) m_chkAlign->setChecked(!m_chkAlign->isChecked()); }, {}},
        {"cmd_fusion", "Toggle Fusion", "Pipeline", CmdType::Toggle, [this](){ if (m_chkFusion) m_chkFusion->setChecked(!m_chkFusion->isChecked()); }, {}},
        {"cmd_bilateral", "Toggle Bilateral Filter", "Pipeline", CmdType::Toggle, [this](){ if (m_chkBilateral) m_chkBilateral->setChecked(!m_chkBilateral->isChecked()); }, {}},
        {"cmd_stretch", "Toggle Intensity Stretch", "Pipeline", CmdType::Toggle, [this](){ if (m_chkStretch) m_chkStretch->setChecked(!m_chkStretch->isChecked()); }, {}},
        {"cmd_peaks", "Toggle Tracking Peaks", "Pipeline", CmdType::Toggle, [this](){ if (m_btnPeakIntensities) m_btnPeakIntensities->setChecked(!m_btnPeakIntensities->isChecked()); }, {}},
        {"cmd_calibrate", "Calibrate Alignment", "Pipeline", CmdType::Action, [this](){ calibrateAlignment(); }, {}},
        {"cmd_open_align", "Open Manual Align Dialog", "Pipeline", CmdType::Action, [this](){ if (m_btnOpenManualAlign) m_btnOpenManualAlign->click(); }, {}},

        {"cmd_tbuffer", "Time Buffer Size", "Pipeline Parameters", CmdType::Parameter, {}, [this, createPopup](QWidget*){ createPopup("Time Buffer Size", m_bufferSlider); }},
        {"cmd_motionthr", "Motion Threshold", "Pipeline Parameters", CmdType::Parameter, {}, [this, createPopup](QWidget*){ createPopup("Motion Threshold", m_motionThresholdSlider); }},
        {"cmd_noisefloor", "Noise Floor", "Pipeline Parameters", CmdType::Parameter, {}, [this, createPopup](QWidget*){ createPopup("Noise Floor", m_noiseFloorSlider); }},
        {"cmd_exposure_dialog", "Open Exposure Dialog", "Exposure", CmdType::Action, [this](){ if (m_manualExposure) showExposureDialog(); }, {}},

        {"cmd_rename_snapshot", "Edit Snapshot Name", "Snapshot", CmdType::Action, [this](){
            setNavItem(NavItem::Snapshot);
            if (m_snapshotNameEdit) {
                m_snapshotNameEdit->setFocus();
                m_snapshotNameEdit->selectAll();
            }
        }, {}},
        {"cmd_append_params", "Toggle Append Params", "Snapshot", CmdType::Toggle, [this](){ if (m_chkAppendParams) m_chkAppendParams->setChecked(!m_chkAppendParams->isChecked()); }, {}},
        {"cmd_filename_config", "Configure Filename Params", "Snapshot", CmdType::Action, [this](){ showFilenameParamsDialog(); }, {}},

        {"cmd_save_preset", "Save Current Preset", "Presets", CmdType::Action, [this](){ if (m_btnSavePreset) m_btnSavePreset->click(); }, {}},
        {"cmd_load_preset", "Load Selected Preset", "Presets", CmdType::Action, [this](){ if (m_btnLoadPreset) m_btnLoadPreset->click(); }, {}},
        {"cmd_delete_preset", "Delete Selected Preset", "Presets", CmdType::Action, [this](){ if (m_btnDeletePreset) m_btnDeletePreset->click(); }, {}},

        {"cmd_history", "Chart History Limit", "Options", CmdType::Parameter, {}, [this, createPopup](QWidget*){ createPopup("Chart History Limit", m_historySlider); }}
    };

    QSettings s(settingsPath(), QSettings::IniFormat);
    s.beginGroup("Hotkeys");
    for (const auto& cmd : m_commands) {
        QString keyStr = s.value(cmd.id, "").toString();
        if (!keyStr.isEmpty()) {
            applyShortcut(cmd.id, QKeySequence(keyStr));
        }
    }
    s.endGroup();

    if (!m_hotkeys.contains("cmd_minimize_dialogs") || m_hotkeys.value("cmd_minimize_dialogs").isEmpty()) {
        applyShortcut("cmd_minimize_dialogs", QKeySequence("Ctrl+Shift+M"));
    }
}

void MainWindow::applyShortcut(const QString& id, const QKeySequence& seq)
{
    m_hotkeys[id] = seq;

    if (m_shortcuts.contains(id)) {
        m_shortcuts[id]->deleteLater();
        m_shortcuts.remove(id);
    }

    if (!seq.isEmpty()) {
        QShortcut* shortcut = new QShortcut(seq, this);
        shortcut->setContext(Qt::ApplicationShortcut);
        connect(shortcut, &QShortcut::activated, this, [this, id]() {
            executeCommand(id);
        });
        m_shortcuts[id] = shortcut;
    }

    QSettings s(settingsPath(), QSettings::IniFormat);
    s.beginGroup("Hotkeys");
    if (seq.isEmpty()) s.remove(id);
    else s.setValue(id, seq.toString());
    s.endGroup();
}

void MainWindow::executeCommand(const QString& id)
{
    auto it = std::find_if(m_commands.begin(), m_commands.end(), [&](const AppCommand& c){ return c.id == id; });
    if (it != m_commands.end()) {
        if (it->type == CmdType::Parameter) {
            if (it->showParam) it->showParam(this);
        } else {
            if (it->trigger) it->trigger();
        }
    }
}

class HotkeyButton : public QPushButton {
public:
    HotkeyButton(const QKeySequence& seq, QWidget* parent = nullptr) : QPushButton(parent) {
        setSeq(seq);
        setCheckable(true);

        setStyleSheet(
            "QPushButton:checked { background: #500b0b; color: #e5e2e1; border: 1px solid #500b0b; }");
    }

    void setSeq(const QKeySequence& seq) {
        m_seq = seq;
        setText(m_seq.isEmpty() ? "None" : m_seq.toString(QKeySequence::NativeText));
        setChecked(false);
    }

    std::function<void(QKeySequence)> onSequenceChanged;

protected:
    bool event(QEvent* e) override {
        if (e->type() == QEvent::KeyPress) {
            QKeyEvent* ke = static_cast<QKeyEvent*>(e);
            if (ke->key() == Qt::Key_Tab || ke->key() == Qt::Key_Backtab) {
                if (isChecked()) {
                    keyPressEvent(ke);
                    return true;
                }
            }
        }
        return QPushButton::event(e);
    }

    void keyPressEvent(QKeyEvent* e) override {
        if (!isChecked()) {
            QPushButton::keyPressEvent(e);
            return;
        }
        int key = e->key();
        if (key == Qt::Key_unknown) return;

        Qt::KeyboardModifiers mods = e->modifiers();
        if (key == Qt::Key_Shift || key == Qt::Key_Control || key == Qt::Key_Alt || key == Qt::Key_Meta) {
            setText(QKeySequence(mods).toString() + "...");
            return;
        }

        m_seq = QKeySequence(key | mods);
        setSeq(m_seq);
        if (onSequenceChanged) onSequenceChanged(m_seq);
    }

    void keyReleaseEvent(QKeyEvent* e) override {
        if (!isChecked()) {
            QPushButton::keyReleaseEvent(e);
            return;
        }
        int key = e->key();
        if (key == Qt::Key_Shift || key == Qt::Key_Control || key == Qt::Key_Alt || key == Qt::Key_Meta) {
            m_seq = QKeySequence(key);
            setSeq(m_seq);
            if (onSequenceChanged) onSequenceChanged(m_seq);
        } else {
            QPushButton::keyReleaseEvent(e);
        }
    }

private:
    QKeySequence m_seq;
};

void MainWindow::showActionsMenu()
{
    QDialog dlg(this);
    dlg.setWindowTitle("Actions & Hotkeys");
    dlg.setStyleSheet(styleSheetText());
    dlg.resize(500, 600);

    QVBoxLayout* root = new QVBoxLayout(&dlg);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    QLabel* title = new QLabel("Actions & Keybinds", &dlg);
    title->setProperty("role", "title");
    title->setStyleSheet("padding: 16px 16px 4px 16px;");
    root->addWidget(title);

    QLabel* tip = new QLabel("Click a shortcut box to assign a key. Click 'Execute' to run immediately.", &dlg);
    tip->setProperty("role", "faint");
    tip->setStyleSheet("padding: 0px 16px 12px 16px;");
    root->addWidget(tip);

    QScrollArea* scroll = new QScrollArea(&dlg);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setStyleSheet("QScrollArea { background: transparent; } QWidget#scrollContent { background: transparent; }");

    QWidget* scrollContent = new QWidget();
    scrollContent->setObjectName("scrollContent");
    QVBoxLayout* listLay = new QVBoxLayout(scrollContent);
    listLay->setContentsMargins(16, 0, 16, 16);
    listLay->setSpacing(8);

    QString lastSection = "";

    QMap<QString, HotkeyButton*> editorMap;

    for (const auto& cmd : m_commands) {
        if (cmd.section != lastSection) {
            QLabel* sec = new QLabel(cmd.section.toUpper());
            sec->setProperty("role", "section");
            sec->setStyleSheet("margin-top: 8px;");
            listLay->addWidget(sec);
            lastSection = cmd.section;
        }

        QFrame* row = new QFrame();
        row->setProperty("role", "panel");
        QHBoxLayout* rowLay = new QHBoxLayout(row);
        rowLay->setContentsMargins(12, 6, 12, 6);

        QLabel* name = new QLabel(cmd.name);
        name->setProperty("role", "title");
        rowLay->addWidget(name, 1);

        QHBoxLayout* editorLay = new QHBoxLayout();
        editorLay->setSpacing(4);
        HotkeyButton* editor = new HotkeyButton(m_hotkeys.value(cmd.id));
        editor->setMinimumHeight(24);
        editor->setMaximumWidth(120);
        editor->setToolTip("Click to assign new hotkey.");
        editorMap[cmd.id] = editor;
        editor->onSequenceChanged = [this, id = cmd.id, editor, &editorMap, &dlg](QKeySequence seq) {
            if (!seq.isEmpty()) {
                QString existingId;
                for (auto it = m_hotkeys.begin(); it != m_hotkeys.end(); ++it) {
                    if (it.value() == seq && it.key() != id) {
                        existingId = it.key();
                        break;
                    }
                }
                if (!existingId.isEmpty()) {
                    QMessageBox msgBox(&dlg);
                    msgBox.setWindowTitle("Shortcut Conflict");
                    msgBox.setText("This shortcut is already in use by another action.\nDo you want to replace it?");
                    msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
                    msgBox.setDefaultButton(QMessageBox::No);
                    msgBox.setStyleSheet(styleSheetText());
                    if (msgBox.exec() == QMessageBox::No) {
                        editor->setSeq(m_hotkeys.value(id));
                        return;
                    }
                    applyShortcut(existingId, QKeySequence());
                    if (editorMap.contains(existingId)) {
                        editorMap[existingId]->setSeq(QKeySequence());
                    }
                }
            }
            applyShortcut(id, seq);
            m_statusBar->showMessage("Shortcut saved.", 3000);
        };
        editorLay->addWidget(editor);

        QPushButton* btnClear = new QPushButton(QString::fromUtf8("\u2715"));
        btnClear->setFixedSize(24, 24);
        btnClear->setStyleSheet(QString(
            "QPushButton { background: transparent; color: %1; border: none;"
            " font-weight: 600; font-size: 12px; padding: 0; }"
            "QPushButton:hover { color: %2; }")
            .arg(T::textFaint).arg(T::accent));
        btnClear->setCursor(Qt::PointingHandCursor);
        btnClear->setToolTip("Clear shortcut");
        connect(btnClear, &QPushButton::clicked, this, [editor]() {
            editor->setSeq(QKeySequence());
            if (editor->onSequenceChanged) editor->onSequenceChanged(QKeySequence());
        });
        editorLay->addWidget(btnClear);

        rowLay->addLayout(editorLay);

        QPushButton* btnExec = new QPushButton(cmd.type == CmdType::Parameter ? "Adjust" : "Exec");
        btnExec->setProperty("kind", "ghost");
        btnExec->setFixedWidth(65);
        connect(btnExec, &QPushButton::clicked, this, [this, id = cmd.id]() {
            executeCommand(id);
        });
        rowLay->addWidget(btnExec);

        listLay->addWidget(row);
    }
    listLay->addStretch();

    scroll->setWidget(scrollContent);
    root->addWidget(scroll, 1);

    QFrame* bottomBar = new QFrame(&dlg);
    bottomBar->setStyleSheet(QString("background: %1; border-top: 1px solid %2;")
                             .arg(T::bg2).arg(T::border));
    QHBoxLayout* btnLay = new QHBoxLayout(bottomBar);
    btnLay->setContentsMargins(16, 10, 16, 10);

    btnLay->addStretch();

    QPushButton* resetBtn = new QPushButton("Reset", bottomBar);
    resetBtn->setProperty("kind", "primary");
    resetBtn->setCursor(Qt::PointingHandCursor);
    connect(resetBtn, &QPushButton::clicked, this, [this, &editorMap, &dlg]() {
        QMessageBox msgBox(&dlg);
        msgBox.setWindowTitle("Reset Hotkeys");
        msgBox.setText("Are you sure you want to clear all hotkeys?");
        msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::No);
        msgBox.setStyleSheet(styleSheetText());
        if (msgBox.exec() == QMessageBox::Yes) {
            for (const auto& cmd : m_commands) {
                applyShortcut(cmd.id, QKeySequence());
                if (editorMap.contains(cmd.id)) {
                    editorMap[cmd.id]->setSeq(QKeySequence());
                }
            }
            m_statusBar->showMessage("All shortcuts cleared.", 3000);
        }
    });
    btnLay->addWidget(resetBtn);

    root->addWidget(bottomBar);

    if (m_animDialogsEnabled) {
        animateDialogEntry(&dlg, m_btnHelp, m_animSpeedMs);
    }
    dlg.exec();
}

#include "mainwindow.moc"
