#include "mainwindow.h"
#include "resource_monitor.h"
#ifndef DUALCAM_SEPARATE_VIEWER
#include "viewer_dialogs.h"
#endif

#include <QPropertyAnimation>
#include <QParallelAnimationGroup>
#include <QPainterPath>
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
#ifdef Q_OS_LINUX
#include <unistd.h>
#include <sys/syscall.h>
#endif
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <array>

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
#include <QImageReader>

ImageView::ImageView(QWidget* parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_OpaquePaintEvent);
    setAttribute(Qt::WA_NoSystemBackground);
    setAutoFillBackground(false);
    setMinimumSize(160, 120);
}

void ImageView::setImage(const QImage& img)
{
    m_image = img;
    update();
}

void ImageView::setPlaceholder(const QString& text)
{
    m_placeholder = text;
    m_image = QImage();
    update();
}

void ImageView::setOverlayText(const QString& text, bool rightAlign)
{
    if (m_overlayText == text && m_overlayRight == rightAlign) return;
    m_overlayText = text;
    m_overlayRight = rightAlign;
    update();
}

void ImageView::setOverlayColor(const QColor& c)
{
    if (m_overlayColor == c) return;
    m_overlayColor = c;
    update();
}

void ImageView::setColorScale(const QImage& ramp, double maxValue, const QString& title)
{
    if (m_scaleRamp.cacheKey() == ramp.cacheKey() && m_scaleMax == maxValue && m_scaleTitle == title)
        return;
    m_scaleRamp = ramp;
    m_scaleMax = maxValue;
    m_scaleTitle = title;
    update();
}

QSize ImageView::targetSize(const QSize& imageSize) const
{
    if (imageSize.isEmpty()) return QSize();
    const double dpr = devicePixelRatioF();
    const double kx = width()  * dpr / imageSize.width();
    const double ky = height() * dpr / imageSize.height();
    const double k  = m_stretch ? std::max(kx, ky) : std::min(kx, ky);
    return QSize(std::max(1, int(imageSize.width() * k)),
                 std::max(1, int(imageSize.height() * k)));
}

void ImageView::paintEvent(QPaintEvent*)
{
    QPainter p(this);

    if (m_image.isNull()) {
        p.fillRect(rect(), QColor("#000000"));
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
        const double dpr = devicePixelRatioF();
        const QSize is = m_image.size();
        QSize ds = targetSize(is);
        const bool oneToOne = std::abs(ds.width() - is.width()) <= 1
                           && std::abs(ds.height() - is.height()) <= 1;
        if (oneToOne) ds = is;
        const QSizeF logical(ds.width() / dpr, ds.height() / dpr);
        const QRectF target(QPointF(std::floor((width() - logical.width()) / 2.0),
                                    std::floor((height() - logical.height()) / 2.0)), logical);

        const QRegion bars = QRegion(rect()).subtracted(QRegion(target.toAlignedRect()));
        for (const QRect& r : bars) p.fillRect(r, Qt::black);

        p.setRenderHint(QPainter::SmoothPixmapTransform, false);
        if (oneToOne) {
            p.drawImage(target.topLeft(), m_image);
        } else {
            p.drawImage(target, m_image);
        }
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

    if (!m_scaleRamp.isNull() && m_scaleMax > 0.0) {
        p.setRenderHint(QPainter::Antialiasing, true);
        QFont f = p.font();
        f.setFamily("Space Mono");
        f.setPointSize(8);
        f.setBold(false);
        f.setLetterSpacing(QFont::AbsoluteSpacing, 0.0);
        QFont fBold = f;
        fBold.setBold(true);
        const QFontMetrics fm(f), fmBold(fBold);
        const int pad = 10, barW = 8, tickW = 4, gap = 6;
        const int lineH = fm.height();
        const int barH = std::clamp(height() * 40 / 100, 80, 200);

        const QString title = m_scaleTitle.section(", ", 0, 0);
        const QString unit = m_scaleTitle.section(", ", 1);

        const int prec = m_scaleMax < 10.0 ? 1 : 0;
        auto label = [prec](double v) { return QString::number(v, 'f', prec); };
        double step = m_scaleMax / 4.0;
        const double mag = std::pow(10.0, std::floor(std::log10(step)));
        const double norm = step / mag;
        step = (norm < 1.5 ? 1.0 : norm < 3.0 ? 2.0 : norm < 7.0 ? 5.0 : 10.0) * mag;
        while (step / m_scaleMax * barH < lineH + 2) step *= 2.0;
        QVector<double> ticks;
        for (double v = 0.0; v < m_scaleMax - 0.4 * step; v += step) ticks << v;

        int labelW = fmBold.horizontalAdvance(label(m_scaleMax));
        for (double v : ticks) labelW = std::max(labelW, fm.horizontalAdvance(label(v)));
        int contentW = std::max(barW + tickW + gap + labelW,
                                fmBold.horizontalAdvance(title) + gap + fm.horizontalAdvance(unit));
        const int boxW = contentW + 2 * pad;
        const int boxH = pad + lineH + 10 + barH + lineH / 2 + pad;
        const QRectF box(width() - 12 - boxW, std::max(8, (height() - boxH) / 2), boxW, boxH);
        p.setPen(QPen(QColor(255, 255, 255, 22), 1.0));
        p.setBrush(QColor(12, 12, 12, 215));
        p.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), 8, 8);

        const QColor textCol(0xe5, 0xe2, 0xe1), dimCol(0x9a, 0x97, 0x94);
        const int left = int(box.left()) + pad;
        int y = int(box.top()) + pad;
        p.setFont(fBold);
        p.setPen(textCol);
        p.drawText(QRect(left, y, contentW, lineH), Qt::AlignLeft | Qt::AlignVCenter, title);
        p.setFont(f);
        p.setPen(dimCol);
        p.drawText(QRect(left + fmBold.horizontalAdvance(title) + gap, y, contentW, lineH),
                   Qt::AlignLeft | Qt::AlignVCenter, unit);
        y += lineH + 10;

        const QRect bar(left, y, barW, barH);
        QPainterPath barPath;
        barPath.addRoundedRect(QRectF(bar), 3, 3);
        p.save();
        p.setClipPath(barPath);
        p.setRenderHint(QPainter::SmoothPixmapTransform, true);
        p.drawImage(bar, m_scaleRamp);
        p.restore();

        const int tickX = bar.right() + 2;
        const int lx = tickX + tickW + gap;
        auto drawTick = [&](double v, bool top) {
            const int ty = bar.bottom() - int(std::round(v / m_scaleMax * (barH - 1)));
            p.setPen(QPen(top ? textCol : QColor(255, 255, 255, 110), 1.0));
            p.drawLine(QPointF(tickX, ty + 0.5), QPointF(tickX + tickW, ty + 0.5));
            p.setFont(top ? fBold : f);
            p.setPen(top ? textCol : dimCol);
            p.drawText(QRect(lx, ty - lineH / 2, labelW, lineH), Qt::AlignLeft | Qt::AlignVCenter, label(v));
        };
        for (double v : ticks) drawTick(v, false);
        drawTick(m_scaleMax, true);
    }
}

CameraWorker::CameraWorker(QObject* parent) : QThread(parent) {
    m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16.0, false);
}
CameraWorker::~CameraWorker() {
    stopCameras();
}
bool CameraWorker::startCameras(const std::string& pipe1, const std::string& pipe2, int w, int h, int fps) {
    Q_UNUSED(w); Q_UNUSED(h); Q_UNUSED(fps);
    stopCameras();
    m_cap1.open(pipe1, cv::CAP_GSTREAMER);
    m_cap2.open(pipe2, cv::CAP_GSTREAMER);
    if (!m_cap1.isOpened() || !m_cap2.isOpened()) {
        emit cameraError("Failed to open GStreamer pipelines");
        m_cap1.release();
        m_cap2.release();
        return false;
    }
    m_frameCount = 0;
    m_running = true;
    start();
    return true;
}
bool CameraWorker::startCamerasV4L2(int id1, int id2, int w, int h, int fps) {
    stopCameras();
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
        return false;
    }
    m_cap1.set(cv::CAP_PROP_FRAME_WIDTH, w);
    m_cap1.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    m_cap1.set(cv::CAP_PROP_FPS, fps);
    m_cap2.set(cv::CAP_PROP_FRAME_WIDTH, w);
    m_cap2.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    m_cap2.set(cv::CAP_PROP_FPS, fps);
    m_frameCount = 0;
    m_running = true;
    start();
    return true;
}
void CameraWorker::stopCameras() {
    m_running = false;

    if (isRunning() && !wait(2000)) {
        m_cap1.release();
        m_cap2.release();
        if (!wait(1000)) {
            terminate();
            wait();
        }
    }
    if (m_cap1.isOpened()) m_cap1.release();
    if (m_cap2.isOpened()) m_cap2.release();
}
void CameraWorker::setParams(const WorkerParams& p) {
    m_paramMutex.lock();
    m_params = p;
    m_paramMutex.unlock();
}
cv::Mat CameraWorker::toWorkingFormat(const cv::Mat& frame, ColorMode mode) {
    if (frame.empty()) return frame;
    cv::Mat res = frame;
    if (mode == ColorMode::GRAY_CV || mode == ColorMode::GRAY_NATIVE) {
        if (frame.channels() == 3) cv::cvtColor(frame, res, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 1 && mode == ColorMode::COLOR) {
        cv::cvtColor(frame, res, cv::COLOR_GRAY2BGR);
    }
    return res;
}
double CameraWorker::detectMotion(const cv::Mat& frame) {

    if (frame.empty()) return 0.0;

    constexpr int kProxySide = 320;
    cv::Mat src = frame;
    const int side = std::max(frame.cols, frame.rows);
    if (side > kProxySide) {
        const double k = double(kProxySide) / side;
        cv::resize(frame, src, cv::Size(), k, k, cv::INTER_AREA);
    }
    cv::Mat fgMask;
    m_bgSubtractor->apply(src, fgMask, 0.01);
    double nonZero = cv::countNonZero(fgMask);
    return nonZero / (src.cols * src.rows);
}
cv::Mat CameraWorker::applyTemporalDenoise(cv::Mat& frame, cv::Mat& ema, int bufferSize) {
    if (bufferSize <= 1 || frame.empty()) return frame;
    double alpha = 1.0 / static_cast<double>(bufferSize);
    int cvType = (frame.channels() == 3) ? CV_32FC3 : CV_32F;
    if (ema.empty() || ema.size() != frame.size() || ema.type() != cvType) {
        frame.convertTo(ema, cvType);
    } else {
        cv::accumulateWeighted(frame, ema, alpha);
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
    cv::Laplacian(gray, laplacian, gray.depth() == CV_8U ? CV_16S : CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    return stddev.val[0] * stddev.val[0];
}
void CameraWorker::run() {
#ifdef Q_OS_LINUX
    m_threadId = static_cast<int>(::syscall(SYS_gettid));
#elif defined(Q_OS_WIN)
    m_threadId = static_cast<int>(reinterpret_cast<quintptr>(QThread::currentThreadId()));
#endif
    int failedGrabs = 0;
    while (m_running) {
        cv::Mat f1, f2;
        if (!m_cap1.grab() || !m_cap2.grab()) {
            if (++failedGrabs == 200) emit cameraError("Camera stream stalled: no frames are arriving");
            QThread::msleep(10);
            continue;
        }
        failedGrabs = 0;
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

        double motion = detectMotion(f2);
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
        m_statProcessed.fetch_add(1);

        if (m_pendingFrames.load() >= 1) {
            m_statDropped.fetch_add(1);
            continue;
        }
        m_pendingFrames.fetch_add(1);
        emit framesProcessed(f1, f2, focus1, focus2, motionDetected, m_frameCount);
    }
    m_threadId = 0;
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
    static const char* bg3 = "#262626";
    static const char* bgHover = "#303030";
    static const char* border = "rgba(255, 255, 255, 0.06)";
    static const char* borderStrong = "rgba(255, 255, 255, 0.14)";
    static const char* text = "#f2efee";
    static const char* textDim = "#b8b5b3";
    static const char* textFaint = "#8a8784";
    static const char* accent = "#d4564a";
    static const char* accentDim = "#7e2c24";
    static const char* onAccent = "#1f0a08";
    static const char* ok = "#7ed492";
    static const char* warn = "#f0b860";
    static const char* err = "#f07d72";
    static const char* cam1 = "#5dcaa5";
    static const char* cam2 = "#f0997b";
}

static void setPillState(QLabel* lbl, const char* state, const QString& text = QString())
{
    if (!lbl) return;
    if (!text.isNull()) lbl->setText(text);

    if (lbl->property("role").toString() == QLatin1String("pill")
        && lbl->property("pillState").toString() == QLatin1String(state))
        return;
    lbl->setProperty("role", "pill");
    lbl->setProperty("pillState", state);
    lbl->style()->unpolish(lbl);
    lbl->style()->polish(lbl);
    lbl->update();
}

static constexpr int kKCalFrames = 20;
static constexpr int kKCalMinLevel = 24;
static constexpr int kKCalMinPixels = 1000;

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
    {"K",   "Channel balance (K#)"},
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

    probeCameras();
    initUI();
    initCommands();
    loadSettings();
    refreshPresetList();
    updateDiffReadout(true);

    m_resMonitor = new ResourceMonitor([this]() {
        ResourceMonitor::FrameCounters c;
        if (m_worker) {
            c.processed = m_worker->processedFrames();
            c.dropped = m_worker->droppedFrames();
            c.workerThreadId = m_worker->threadId();
        }
        c.displayed = m_displayedFrames;
        return c;
    }, this);
    connect(m_resMonitor, &ResourceMonitor::updated, this, &MainWindow::updateResourceDisplay);
    m_resMonitor->start(1000);

    {
        QSettings s(settingsPath(), QSettings::IniFormat);
        if (!s.value("tourShown", false).toBool()) {
            QTimer::singleShot(600, this, [this]() { showQuickTour(); });
        }
    }
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

        QMainWindow, QWidget#root { background: %1; color: %3; }
        QDialog { background: %2; color: %3; }
        QDialog QLabel { color: %3; background: transparent; }
        QWidget { color: %3; font-family: 'Inter','Segoe UI Variable','Segoe UI',sans-serif; font-size: 12px; }
        QToolTip {
            background: %6; color: %3; border: 1px solid %9; padding: 4px 8px;
            border-radius: 6px; font-size: 11px; font-weight: 500;
        }

        QMenu {
            background: %2; color: %3; border: 1px solid %9;
            border-radius: 8px; padding: 5px 4px; font-size: 12px;
        }
        QMenu::item {
            background: transparent; color: %3;
            padding: 6px 22px 6px 14px; border-radius: 5px; margin: 1px 2px;
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

        QLabel { color: %3; }
        QLabel[role="dim"]     { color: %4;  font-size: 11px; }
        QLabel[role="faint"]   { color: %10; font-size: 11px; }
        QLabel[role="title"]   { color: %3;  font-size: 13px; font-weight: 600; }
        QLabel[role="section"] { color: %4;  font-size: 10px; font-weight: 600; letter-spacing: 0.10em; }
        QLabel[role="hero"]    { color: %3;  font-family: 'Space Mono','JetBrains Mono',monospace; font-size: 22px; font-weight: 700; }
        QLabel[role="mono"]    { color: %3;  font-family: 'Space Mono','JetBrains Mono',monospace; font-size: 11px; }

        QLabel[role="pill"] {
            background: %6; color: %4; border: 1px solid %7;
            border-radius: 9px; padding: 2px 10px;
            font-family: 'Space Mono','JetBrains Mono',monospace;
            font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
        }
        QLabel[role="pill"][pillState="ok"]   { color: #8fe0a4; background: #10291a; border-color: #2a6b40; }
        QLabel[role="pill"][pillState="warn"] { color: #f5c98a; background: #2a2310; border-color: #7a5f2c; }
        QLabel[role="pill"][pillState="err"]  { color: #f5a89e; background: #2e1310; border-color: %5; }
        QLabel[role="pill"][pillState="info"] { color: %3;      background: %6;      border-color: %5; }
        QLabel[role="pill"][pillState="idle"] { color: %4;      background: %6;      border-color: %7; }

        QFrame[role="card"]  { background: %6; border: 1px solid %7; border-radius: 8px; }
        QFrame[role="panel"] { background: %2; border: 1px solid %7; border-radius: 8px; }
        QFrame[role="hud"]   { background: rgba(10,10,10,215); border: 1px solid %9; border-radius: 8px; }

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

        QPushButton {
            background: %6; color: %3; border: 1px solid %7; border-radius: 6px;
            padding: 5px 12px; min-height: 22px;
            font-size: 12px; font-weight: 500;
        }
        QPushButton:hover    { background: %8; border-color: %9; }
        QPushButton:pressed  { background: %7; }
        QPushButton:disabled { color: %10; background: %2; border-color: %7; }
        QPushButton:checked  { background: %11; border-color: %5; color: %3; }

        QToolButton {
            background: %6; color: %3; border: 1px solid %7; border-radius: 6px;
            padding: 5px 12px; min-height: 22px;
            font-size: 12px; font-weight: 500;
        }
        QToolButton:hover    { background: %8; border-color: %9; color: %3; }
        QToolButton:pressed  { background: %7; color: %3; }
        QToolButton:checked  { background: %11; border-color: %5; color: %3; }
        QToolButton:disabled { color: %10; background: %2; border-color: %7; }
        QToolButton::menu-indicator { image: none; }

        QPushButton[kind="primary"] {
            background: %5; color: %12; border: 1px solid %5;
            font-weight: 600; padding: 6px 18px; min-height: 24px;
        }
        QPushButton[kind="primary"]:hover    { background: #e0695d; border-color: #e0695d; }
        QPushButton[kind="primary"]:pressed  { background: %11; color: %3; border-color: %11; }
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

        QPushButton#sheetHandle {
            background: %2; border: none;
            border-top: 1px solid %7; border-bottom: 1px solid %7;
            color: %4; padding: 0; min-height: 22px; max-height: 22px;
        }
        QPushButton#sheetHandle:hover { background: %6; color: %3; }

        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background: %8; color: %3; border: 1px solid %7; border-radius: 6px;
            padding: 4px 8px; min-height: 22px;
            selection-background-color: %5; selection-color: %12;
            font-family: 'Space Mono','JetBrains Mono',monospace; font-size: 11px;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: %5; }
        QComboBox::drop-down { border: none; width: 18px; }
        QComboBox QAbstractItemView {
            background: %2; color: %3; border: 1px solid %7;
            selection-background-color: %11; selection-color: %3;
            font-family: 'Space Mono','JetBrains Mono',monospace; font-size: 11px;
        }

        QAbstractItemView, QTreeView, QListView, QTableView, QColumnView {
            background: %2; color: %3; border: 1px solid %7;
            selection-background-color: %11; selection-color: %3;
            alternate-background-color: %6; outline: 0;
        }
        QAbstractItemView::item:hover    { background: %6; color: %3; }
        QAbstractItemView::item:selected { background: %11; color: %3; }
        QHeaderView { background: %2; }
        QHeaderView::section {
            background: %6; color: %4; border: none;
            border-right: 1px solid %7; border-bottom: 1px solid %7;
            padding: 4px 8px;
        }

        QCheckBox { color: %3; spacing: 6px; font-size: 12px; }
        QCheckBox::indicator {
            width: 15px; height: 15px; border-radius: 4px;
            background: %8; border: 1px solid %9;
        }
        QCheckBox::indicator:hover    { border-color: %5; }
        QCheckBox::indicator:checked  { background: %5; border-color: %5; }
        QCheckBox::indicator:disabled { background: %2; border-color: %7; }

        QSlider::horizontal { min-height: 22px; }
        QSlider::groove:horizontal { background: %8; height: 4px; border: none; border-radius: 2px; }
        QSlider::sub-page:horizontal { background: %5; border-radius: 2px; }
        QSlider::add-page:horizontal { background: %8; border-radius: 2px; }
        QSlider::handle:horizontal {
            background: %3; width: 14px; height: 14px; margin: -5px 0;
            border: 2px solid %5; border-radius: 8px;
        }
        QSlider::handle:horizontal:hover    { background: %5; border-color: %3; }
        QSlider::handle:horizontal:pressed  { background: %11; border-color: %3; }
        QSlider::handle:horizontal:disabled { background: %8; border-color: %7; }

        QSlider::vertical { min-width: 22px; }
        QSlider::groove:vertical { background: %8; width: 4px; border: none; border-radius: 2px; }
        QSlider::sub-page:vertical { background: %8; border-radius: 2px; }
        QSlider::add-page:vertical { background: %5; border-radius: 2px; }
        QSlider::handle:vertical {
            background: %3; height: 14px; width: 14px; margin: 0 -5px;
            border: 2px solid %5; border-radius: 8px;
        }
        QSlider::handle:vertical:hover { background: %5; border-color: %3; }

        QListWidget {
            background: %2; color: %3; border: 1px solid %7; border-radius: 6px;
            outline: 0; font-size: 12px;
        }
        QListWidget::item {
            padding: 6px 10px; border-bottom: 1px solid %7;
        }
        QListWidget::item:selected { background: %6; color: %3; border-left: 2px solid %5; padding-left: 8px; }
        QListWidget::item:hover    { background: %6; }

        QStatusBar {
            background: %2; color: %4;
            border-top: 1px solid %7;
            font-family: 'Space Mono','JetBrains Mono',monospace; font-size: 10px;
            min-height: 22px;
        }
        QStatusBar::item { border: none; }

        QSplitter::handle           { background: %1; }
        QSplitter::handle:horizontal { width: 2px; }
        QSplitter::handle:vertical   { height: 2px; }
        QSplitter::handle:hover      { background: %5; }

        QScrollBar:vertical   { background: %2; width: 8px; }
        QScrollBar:horizontal { background: %2; height: 8px; }
        QScrollBar::handle:vertical   { background: %9; min-height: 24px; border-radius: 4px; }
        QScrollBar::handle:horizontal { background: %9; min-width:  24px; border-radius: 4px; }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: %5; }
        QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
        QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

        QGroupBox { border: 1px solid %7; border-radius: 8px; margin-top: 14px; padding: 10px; font-size: 11px; }
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
        .arg(T::accentDim)
        .arg(T::onAccent);
}

void MainWindow::initUI()
{
    setWindowTitle("DualCam Analysis Tool");
    setMinimumSize(560, 420);

    qApp->setStyleSheet(styleSheetText());

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
            const QSize native = sensorNativeSize();
            const int maxW = native.isValid() ? native.width()  : 3840;
            const int maxH = native.isValid() ? native.height() : 2160;
            const int maxF = native.isValid() ? std::max(1, int(sensorMaxFps(1, 1))) : 240;
            QSpinBox* spinW = new QSpinBox(&dlg); spinW->setRange(std::min(320, maxW), maxW); spinW->setValue(std::min(1920, maxW));
            QSpinBox* spinH = new QSpinBox(&dlg); spinH->setRange(std::min(240, maxH), maxH); spinH->setValue(std::min(1080, maxH));
            QSpinBox* spinF = new QSpinBox(&dlg); spinF->setRange(1, maxF); spinF->setValue(std::min(30, maxF));
            QFormLayout* form = new QFormLayout();
            form->addRow("Width:", spinW);
            form->addRow("Height:", spinH);
            form->addRow("FPS:", spinF);
            lay->addLayout(form);
            QDialogButtonBox* box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
            connect(box, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
            connect(box, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
            lay->addWidget(box);

            if (execDialog(&dlg, m_comboCamSet) == QDialog::Accepted) {
                int w = spinW->value(), h = spinH->value(), f = spinF->value();
                fitToSensor(w, h, f);
                QString label = QString("%1x%2 @ %3 FPS \u2605").arg(w).arg(h).arg(f);
                m_comboCamSet->setItemText(index, label);
                m_comboCamSet->setItemData(index, QVariantList{w, h, f});
            }
        }
    });

    m_fpsPill = new QLabel("IDLE", this);
    m_fpsPill->setProperty("role", "pill");
    m_fpsPill->setProperty("pillState", "idle");
    m_eccPill = new QLabel("NO ALIGN", this);
    m_eccPill->setProperty("role", "pill");
    m_eccPill->setProperty("pillState", "err");

    m_resPill = new QLabel("CPU  --", this);
    m_resPill->setProperty("role", "pill");
    m_resPill->setProperty("pillState", "idle");
    m_resPill->setCursor(Qt::PointingHandCursor);
    m_resPill->setToolTip("System resources \u2014 click for details");
    m_resPill->installEventFilter(this);

    m_btnGallery = new QPushButton("Snapshots", this);
    m_btnGallery->setCursor(Qt::PointingHandCursor);
    m_btnGallery->setToolTip("Open snapshots folder in OS");
    connect(m_btnGallery, &QPushButton::clicked, this, [this]() {
        QString dir = QCoreApplication::applicationDirPath() + "/metrics";
        QDir().mkpath(dir);
        QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
        m_statusBar->showMessage("Snapshots folder: " + dir, 4000);
    });

    m_btnHelp = new QPushButton("HK's", this);
    m_btnHelp->setProperty("kind", "ghost");
    m_btnHelp->setCursor(Qt::PointingHandCursor);
    m_btnHelp->setToolTip("Actions & Hotkeys — run any action or bind a key");
    connect(m_btnHelp, &QPushButton::clicked, this, &MainWindow::showActionsMenu);

    m_btnHelpDocs = new QPushButton("?", this);
    m_btnHelpDocs->setProperty("kind", "ghost");
    m_btnHelpDocs->setFixedWidth(28);
    m_btnHelpDocs->setStyleSheet("QPushButton { padding-left: 0; padding-right: 0; }");
    m_btnHelpDocs->setCursor(Qt::PointingHandCursor);
    m_btnHelpDocs->setToolTip("Help & Tips — overview of the app and the interactive tour");
    connect(m_btnHelpDocs, &QPushButton::clicked, this, &MainWindow::showHelpDialog);

    topLay->addWidget(m_comboCamSet);

    m_btnExpToggle = new QPushButton("AUTO", this);
    m_btnExpToggle->setCheckable(true);
    m_btnExpToggle->setCursor(Qt::PointingHandCursor);
    m_btnExpToggle->setToolTip("Toggle automatic / manual exposure");
    m_btnExpToggle->setFixedWidth(56);
    m_btnExpToggle->setStyleSheet(QString(
        "QPushButton { background: %1; color: %2; border: 1px solid %3; border-radius: 6px;"
        " padding: 3px 8px; font-weight: 700; font-size: 10px; letter-spacing: 0.10em; font-family: 'Space Mono',monospace; }"
        "QPushButton:checked { background: %4; color: %5; border-color: %4; }")
        .arg(T::bg2).arg(T::text).arg(T::border).arg(T::accent).arg(T::onAccent));

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
        "QLabel { background: %1; color: %2; border: 1px solid %3; border-radius: 6px;"
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
    topLay->addWidget(m_resPill);
    topLay->addWidget(m_fpsPill);
    topLay->addWidget(m_eccPill);
    topLay->addSpacing(4);
    topLay->addWidget(m_btnGallery);
    topLay->addWidget(m_btnHelp);
    topLay->addWidget(m_btnHelpDocs);

    rootLayout->addWidget(topStrip);

    m_videoArea = new QWidget(this);
    m_videoArea->setStyleSheet(QString("background:%1;").arg(T::bg0));
    QVBoxLayout* vidLay = new QVBoxLayout(m_videoArea);
    vidLay->setContentsMargins(2, 2, 2, 2);
    vidLay->setSpacing(2);

    m_splitter = new QSplitter(Qt::Horizontal, m_videoArea);

    m_splitter->setHandleWidth(2);

    auto makeView = [this](const QString& placeholder) {
        ImageView* v = new ImageView(this);
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
        "QPushButton { background: #0c0c0c; color: %1; border: 1px solid %2; border-radius: 8px; font-weight: bold; font-family: 'Space Mono',monospace; }"
        "QPushButton:hover { background: #1a1a1a; color: %3; border-color: %4; }")
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
        "QPushButton { background: #161616; border: 1px solid %1; border-radius: 20px; }"
        "QPushButton:hover { background: #2e0e0b; border: 1px solid %2; }")
        .arg(T::borderStrong).arg(T::accent));
    auto addFabIcon =[](QPushButton* btn, const QString& text, const QString& style) -> QLabel* {
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
        cameraTransition([this]() {
            if (m_camerasOpen) closeCameras();
            else openCameras();
        });
    });
    m_btnToggleCameras = m_btnFabStream;

    m_btnModeToggle = new QPushButton("Dual", m_videoArea);
    m_btnModeToggle->setFixedHeight(26);
    m_btnModeToggle->setCursor(Qt::PointingHandCursor);
    m_btnModeToggle->setStyleSheet(QString(
        "QPushButton { background: %1; color: %2; border: 1px solid %3;"
        " border-radius: 13px; padding: 3px 18px; font-weight: 700; font-size: 10px;"
        " letter-spacing: 0.12em; min-width: 68px; font-family: 'Space Mono',monospace; }"
        "QPushButton:hover { background: %4; color: %5; border-color: %4; }")
        .arg(T::accentDim).arg(T::text).arg(T::accent).arg(T::accent).arg(T::onAccent));
    connect(m_btnModeToggle, &QPushButton::clicked, this, [this]() {
        setDiffMode(!m_isDiffMode);
    });

    m_btnFabSnapshot = new QPushButton(m_videoArea);
    m_btnFabSnapshot->setText("");
    m_btnFabSnapshot->setFixedSize(40, 40);
    m_btnFabSnapshot->setCursor(Qt::PointingHandCursor);
    m_btnFabSnapshot->setToolTip("Save snapshot");
    m_btnFabSnapshot->setStyleSheet(QString(
        "QPushButton { background: %1; border: 1px solid %1; border-radius: 20px; }"
        "QPushButton:hover { background: #e0695d; border-color: #e0695d; }"
        "QPushButton:pressed { background: %2; border-color: %2; }")
        .arg(T::accent).arg(T::accentDim));
    m_fabSnapIcon = addFabIcon(m_btnFabSnapshot, "SNAP", QString(
        "QLabel { background: transparent; color: %1; border: none;"
        " font-family: 'Space Mono','JetBrains Mono',monospace;"
        " font-size: 10px; font-weight: 800; letter-spacing: 0.08em; }").arg(T::onAccent));
    connect(m_btnFabSnapshot, &QPushButton::clicked, this, [this]() {
        saveSnapshot();
        refreshSnapshotPreview();
    });

    m_btnFabStream->hide();
    m_btnModeToggle->hide();
    m_btnFabSnapshot->hide();

    m_resHud = new QFrame(m_videoArea);
    m_resHud->setProperty("role", "hud");
    m_resHud->setAttribute(Qt::WA_TransparentForMouseEvents);
    QVBoxLayout* hudLay = new QVBoxLayout(m_resHud);
    hudLay->setContentsMargins(10, 8, 10, 8);
    m_resHudText = new QLabel("Collecting\u2026", m_resHud);
    m_resHudText->setProperty("role", "mono");
    m_resHudText->setTextFormat(Qt::PlainText);
    hudLay->addWidget(m_resHudText);
    m_resHud->hide();

    m_videoArea->installEventFilter(this);

    m_mainSplit = new QSplitter(Qt::Horizontal, this);
    m_mainSplit->setHandleWidth(2);
    m_mainSplit->setChildrenCollapsible(false);

    m_sideNav = buildSideNav();
    m_mainSplit->addWidget(m_sideNav);

    m_workStack = new QStackedWidget(this);

    if (m_btnFloatingPreviewToggle) m_btnFloatingPreviewToggle->setParent(m_workStack);

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

    auto fadeInPage = [this](QWidget* w, int durationMs) {
        if (!w) return;
        if (!m_animTabsEnabled || w->findChild<ImageView*>()) {
            if (w->graphicsEffect()) w->setGraphicsEffect(nullptr);
            return;
        }
        QGraphicsOpacityEffect* effect = qobject_cast<QGraphicsOpacityEffect*>(w->graphicsEffect());
        if (!effect) {
            effect = new QGraphicsOpacityEffect(w);
            w->setGraphicsEffect(effect);
        }
        QPropertyAnimation* anim = new QPropertyAnimation(effect, "opacity", w);
        anim->setDuration(durationMs);
        anim->setStartValue(0.0);
        anim->setEndValue(1.0);
        anim->setEasingCurve(QEasingCurve::OutCubic);
        QObject::connect(anim, &QPropertyAnimation::finished, w, [w]() {
            w->setGraphicsEffect(nullptr);
        });
        anim->start(QAbstractAnimation::DeleteWhenStopped);
    };

    connect(m_bottomStack, &QStackedWidget::currentChanged, this, [this, fadeInPage](int index) {
        fadeInPage(m_bottomStack->widget(index), m_animSpeedMs * 8 / 10);
    });

    connect(m_workStack, &QStackedWidget::currentChanged, this, [this, fadeInPage](int index) {
        fadeInPage(m_workStack->widget(index), m_animSpeedMs);
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
    m_chkFusion = new QCheckBox("Sum (CAM1 + k·CAM2)/2", this);
    m_chkFusion->setToolTip("DUAL view: shows Σ/2 = (CAM1 + k·CAM2)/2 with the aligned CAM2 (needs a solved\n"
                            "alignment): the image of the total flux S0 with the polarization removed, in CAM1\n"
                            "units — for an unpolarized scene it equals CAM1. Uses the same k as the difference.\n"
                            "The DIFF view always uses the camera frames themselves.");
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

    QHBoxLayout* eccOpts = new QHBoxLayout();
    eccOpts->setSpacing(6);
    m_chkEccPolish = new QCheckBox("ECC polish", this);
    m_chkEccPolish->setChecked(true);
    m_chkEccPolish->setToolTip(
        "After the feature stage, refine the matrix with pyramidal ECC on gradient maps.\n"
        "Gives sub-pixel accuracy. Turn off if the two channels differ so strongly\n"
        "that the polish degrades the result — compare the diff view both ways.");
    m_chkPersistAlign = new QCheckBox("Remember", this);
    m_chkPersistAlign->setChecked(true);
    m_chkPersistAlign->setToolTip(
        "Save the solved warp matrix and restore it on the next launch.\n"
        "Automatically discarded if the frame size changes.");
    connect(m_chkPersistAlign, &QCheckBox::toggled, this, [this](bool on) {
        if (on) saveAlignmentMatrix();
        else {
            QSettings s(settingsPath(), QSettings::IniFormat);
            s.beginGroup("Alignment");
            s.remove("");
            s.endGroup();
            s.sync();
        }
    });
    eccOpts->addWidget(m_chkEccPolish, 1);
    eccOpts->addWidget(m_chkPersistAlign, 1);
    eccOpts->addStretch(1);
    eccLay->addLayout(eccOpts);

    m_alignDetail = new QLabel("No solution yet.", this);
    m_alignDetail->setWordWrap(true);
    m_alignDetail->setProperty("role", "faint");
    eccLay->addWidget(m_alignDetail);

    QLabel* eccHint = new QLabel(
        "Calibrate solves the warp matrix once: features (SIFT / ORB / AKAZE + RANSAC) "
        "find it globally, then sub-pixel correlation and ECC refine it on gradient maps. "
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
    m_noiseFloorSlider->setToolTip("|D| at or below this level (DN) is shown as 0");
    connect(m_noiseFloorSlider, &QSlider::valueChanged, this, [this](int v) {
        m_noiseFloor = v;
        m_noiseFloorLabel->setText(QString::number(v));
        pushWorkerParams();
        updateDiffReadout(true);
    });
    nfLay->addWidget(m_noiseFloorSlider, 1);
    nfLay->addWidget(m_noiseFloorLabel);
    grid->addWidget(nfBox, 1, 0);

    grid->addWidget(sectionLabel("STRETCH"), 0, 1);
    m_chkStretch = new QCheckBox("Stretch intensity range", this);
    m_chkStretch->setToolTip("Multiply |D| so that its largest value reaches the top of the colour scale.\n"
                             "The factor and the resulting scale are shown below and on the image.");
    connect(m_chkStretch, &QCheckBox::toggled, this, [this]() {
        updateView();
        updateDiffReadout(true);
    });
    grid->addWidget(m_chkStretch, 1, 1);

    grid->addWidget(sectionLabel("PEAKS"), 0, 2);
    m_btnPeakIntensities = new QPushButton("Track peaks", this);
    m_btnPeakIntensities->setCheckable(true);
    connect(m_btnPeakIntensities, &QPushButton::toggled, this, [this](bool checked) {
        if (!checked) m_lblPeakInfo->clear();
        updateView();
    });
    grid->addWidget(m_btnPeakIntensities, 1, 2);

    grid->addWidget(sectionLabel("CHANNEL BALANCE k"), 2, 0);
    QWidget* kBox = new QWidget(this);
    QVBoxLayout* kCol = new QVBoxLayout(kBox);
    kCol->setContentsMargins(0, 0, 0, 0);
    kCol->setSpacing(4);
    QHBoxLayout* kLay = new QHBoxLayout();
    kLay->setContentsMargins(0, 0, 0, 0);
    kLay->setSpacing(6);
    m_spnChannelK = new QDoubleSpinBox(this);
    m_spnChannelK->setRange(0.1, 10.0);
    m_spnChannelK->setDecimals(4);
    m_spnChannelK->setSingleStep(0.001);
    m_spnChannelK->setValue(m_channelK);
    m_spnChannelK->setKeyboardTracking(false);
    m_spnChannelK->setFocusPolicy(Qt::StrongFocus);
    m_spnChannelK->installEventFilter(this);
    m_spnChannelK->setToolTip("Weight of CAM2 in the difference D = CAM1 − k·CAM2: the radiometric balance of\n"
                              "the two channels. A calibration constant — set it once with Calibrate k,\n"
                              "with exposure on MAN, and keep it while the rig and exposure stay the same.\n"
                              "It is saved at once and restored on the next start; presets do not change it.");
    connect(m_spnChannelK, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
        m_channelK = v;
        if (!m_kProgrammatic) {
            KCalibration manual;
            manual.source = "manual";
            manual.when = QDateTime::currentDateTime();
            m_kCal = manual;
            saveKCalibration();
            updateKInfo();
        }
        updateView();
        updateDiffReadout(true);
    });
    m_btnCalibrateK = new QPushButton("Calibrate k", this);
    m_btnCalibrateK->setToolTip(QString("Measures k over %1 frames so that CAM1 − k·CAM2 = 0 (median of CAM1 / CAM2)\n"
                                        "and saves it. Point the rig at an unpolarized, evenly lit scene with the\n"
                                        "laser off and exposure set as for the measurement. Click again to cancel.")
                                    .arg(kKCalFrames));
    connect(m_btnCalibrateK, &QPushButton::clicked, this, &MainWindow::calibrateChannelBalance);
    kLay->addWidget(m_spnChannelK, 1);
    kLay->addWidget(m_btnCalibrateK);
    kCol->addLayout(kLay);
    m_lblKInfo = new QLabel(this);
    m_lblKInfo->setWordWrap(true);
    m_lblKInfo->setTextInteractionFlags(Qt::TextSelectableByMouse);
    kCol->addWidget(m_lblKInfo);
    grid->addWidget(kBox, 3, 0);

    grid->addWidget(sectionLabel("CLIP LEVEL"), 2, 1);
    m_spnClipLevel = new QSpinBox(this);
    m_spnClipLevel->setRange(128, 255);
    m_spnClipLevel->setSuffix(" DN");
    m_spnClipLevel->setValue(m_clipLevel);
    m_spnClipLevel->setToolTip("Pixels at or above this level count as clipped. Clipped in both\n"
                               "cameras they are equal by definition, so D = 0 there whatever k is.");
    connect(m_spnClipLevel, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        m_clipLevel = v;
        updateView();
        updateDiffReadout(true);
    });
    grid->addWidget(m_spnClipLevel, 3, 1, Qt::AlignTop);

    m_lblPeakInfo = new QLabel("", this);
    m_lblPeakInfo->setStyleSheet(QString("color:%1; font-family:'Space Mono',monospace;").arg(T::warn));
    grid->addWidget(m_lblPeakInfo, 4, 0, 1, 3);

    m_lblDiffInfo = new QLabel(this);
    m_lblDiffInfo->setProperty("role", "mono");
    m_lblDiffInfo->setWordWrap(true);
    m_lblDiffInfo->setTextInteractionFlags(Qt::TextSelectableByMouse);
    grid->addWidget(m_lblDiffInfo, 5, 0, 1, 3);

    grid->setColumnStretch(0, 1);
    grid->setColumnStretch(1, 1);
    grid->setColumnStretch(2, 1);
    grid->setRowStretch(6, 1);
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

    m_chkSeparateFiles = new QCheckBox("Separate files", this);
    m_chkSeparateFiles->setToolTip("Dual mode: save CAM1 and CAM2 as two files (_cam1 / _cam2)\n"
                                   "instead of one side-by-side image");
    saveLay->addWidget(m_chkSeparateFiles);

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
        "QListWidget { background: %1; border: 1px solid %2; border-radius: 6px; }"
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
        showSnapshotMenu(m_snapshotRecent, pos);
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
        name.replace('/', '-').replace('\\', '-');
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
            " border-left: 2px solid transparent; border-radius: 6px;"
            " text-align: left; padding: 0 10px; font-size: 12px; font-weight: 500;"
            " min-height: 28px; }"
            "QPushButton:hover { background: %2; color: %3; }"
            "QPushButton:checked { background: rgba(212, 86, 74, 0.14); color: %3;"
            " border-left: 2px solid %4; font-weight: 600; }")
            .arg(T::textDim)
            .arg(T::bg2)
            .arg(T::text)
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

    Q_UNUSED(startW);
    m_mainSplit->setSizes({ endW, std::max(300, total - endW) });
    applyNavLabels();
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

        Q_UNUSED(startH);
        m_bottomStack->setVisible(show);
        m_workSplit->setSizes(show ? QList<int>{ totalH - targetH, targetH }
                                   : QList<int>{ totalH, 0 });
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
        if (m_sideNav) m_sideNav->hide();
        if (m_bottomStack) m_bottomStack->hide();
        if (m_statusBar) {
            if (m_statusBar->graphicsEffect()) m_statusBar->setGraphicsEffect(nullptr);
            m_statusBar->hide();
        }
    } else {
        if (m_sideNav) {
            m_sideNav->setMinimumWidth(56);
            m_sideNav->show();
            if (m_mainSplit) {
                QList<int> sizes = m_mainSplit->sizes();
                if (sizes.size() >= 2) {
                    const int totalW = sizes[0] + sizes[1];
                    const int targetW = m_navExpanded ? 200 : 56;
                    m_mainSplit->setSizes({ targetW, totalW - targetW });
                }
            }
        }
        if (m_bottomStack && m_currentNav != NavItem::None) {
            m_bottomStack->show();
            if (m_workSplit) {
                QList<int> sizes = m_workSplit->sizes();
                if (sizes.size() >= 2) {
                    const int totalH = sizes[0] + sizes[1];
                    const int targetH = qMin(260, totalH / 2);
                    m_workSplit->setSizes({ totalH - targetH, targetH });
                }
            }
        }
        if (m_statusBar) {
            if (m_statusBar->graphicsEffect()) m_statusBar->setGraphicsEffect(nullptr);
            m_statusBar->show();
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
        showSnapshotMenu(m_snapshotPreview, pos);
    });

    connect(m_snapshotPreview, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* item) {
        if (!item) return;
        QString fileBase = item->text();
        QString imagePath = QCoreApplication::applicationDirPath() + "/metrics/" + fileBase;
        openPreviewWindow(imagePath, fileBase);
    });

    return page;
}

void MainWindow::showSnapshotMenu(QListWidget* list, const QPoint& pos)
{
    QListWidgetItem* item = list->itemAt(pos);
    if (!item) return;
    QMenu menu(this);
    menu.setStyleSheet(styleSheetText());
    QAction* openAct = menu.addAction("Preview");
    QAction* renameAct = menu.addAction("Rename");
    QAction* dltAct = menu.addAction("Delete");
    QAction* res = menu.exec(list->mapToGlobal(pos));
    if (!res) return;

    const QString dir = QCoreApplication::applicationDirPath() + "/metrics/";
    const QFileInfo oldInfo(dir + item->text());
    if (res == openAct) {
        emit list->itemDoubleClicked(item);
    } else if (res == renameAct) {
        bool ok;
        QString newName = QInputDialog::getText(this, "Rename", "New name:", QLineEdit::Normal, item->text(), &ok).trimmed();
        if (!ok || newName.isEmpty() || newName == item->text()) return;
        if (QFileInfo(newName).suffix().compare(oldInfo.suffix(), Qt::CaseInsensitive) != 0)
            newName += "." + oldInfo.suffix();
        if (QFile::exists(dir + newName) || !QFile::rename(oldInfo.filePath(), dir + newName)) {
            m_statusBar->showMessage("Rename failed: " + newName + " already exists or is not a valid name.", 4000);
            return;
        }
        refreshSnapshotPreview();
    } else if (res == dltAct) {
        if (QMessageBox::question(this, "Delete", "Delete " + item->text() + "?") == QMessageBox::Yes) {
            QFile::remove(oldInfo.filePath());
            refreshSnapshotPreview();
        }
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

void MainWindow::openPreviewWindow(const QString& imagePath, const QString& fileBase)
{
    QDialog* dlg = new QDialog(nullptr, Qt::Window);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle("Preview: " + fileBase);
    dlg->setStyleSheet(styleSheetText());
    dlg->resize(this->width() * 2 / 3, this->height() * 2 / 3);

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

    std::vector<uint8_t> jpegBytes;
    QFile jf(imagePath);
    if (jf.open(QIODevice::ReadOnly)) {
        QByteArray ba = jf.readAll();
        jpegBytes.assign(reinterpret_cast<const uint8_t*>(ba.constData()),
                         reinterpret_cast<const uint8_t*>(ba.constData()) + ba.size());
    }
    cv::Mat raw = jpegBytes.empty() ? cv::Mat() : cv::imdecode(jpegBytes, cv::IMREAD_UNCHANGED);
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
                    else if (v.isArray()) {
                        const QJsonArray arr = v.toArray();
                        bool numeric = arr.size() <= 64;
                        QStringList parts;
                        for (const QJsonValue& e : arr) {
                            if (!e.isDouble()) { numeric = false; break; }
                            parts << QString::number(e.toDouble(), 'g', 6);
                        }
                        line = full + ": " + (numeric ? "[" + parts.join(", ") + "]" : QString("[array]"));
                    }
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

    QSize targetSize = dlg->testAttribute(Qt::WA_Resized) ? dlg->size()
                                                          : dlg->sizeHint();
    if (targetSize.width() < 100 || targetSize.height() < 100) {
        targetSize = targetSize.expandedTo(dlg->sizeHint());
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

    QPoint dir = startPos - targetGeo.center();
    const int len = qMax(1, int(std::sqrt(double(dir.x() * dir.x() + dir.y() * dir.y()))));
    QPoint offset(dir.x() * 18 / len, dir.y() * 18 / len);

    dlg->setGeometry(targetGeo.translated(offset));
    dlg->setWindowOpacity(0.0);
    dlg->show();
    dlg->raise();
    dlg->activateWindow();

    QPropertyAnimation* posAnim = new QPropertyAnimation(dlg, "pos", dlg);
    posAnim->setDuration(durationMs);
    posAnim->setStartValue(targetGeo.topLeft() + offset);
    posAnim->setEndValue(targetGeo.topLeft());
    posAnim->setEasingCurve(QEasingCurve::OutCubic);

    QPropertyAnimation* fadeAnim = new QPropertyAnimation(dlg, "windowOpacity", dlg);
    fadeAnim->setDuration(durationMs);
    fadeAnim->setStartValue(0.0);
    fadeAnim->setEndValue(1.0);
    fadeAnim->setEasingCurve(QEasingCurve::OutCubic);

    QParallelAnimationGroup* group = new QParallelAnimationGroup(dlg);
    group->addAnimation(posAnim);
    group->addAnimation(fadeAnim);
    QObject::connect(group, &QParallelAnimationGroup::finished, dlg, [dlg]() {
        dlg->setWindowOpacity(1.0);
    });
    group->start(QAbstractAnimation::DeleteWhenStopped);
}

int MainWindow::execDialog(QDialog* dlg, QWidget* triggerWidget)
{

    dlg->setWindowModality(Qt::ApplicationModal);
    if (m_animDialogsEnabled) animateDialogEntry(dlg, triggerWidget, m_animSpeedMs);
    return dlg->exec();
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

void MainWindow::positionFloatingButtons(bool animate)
{
    if (!m_videoArea) return;
    const int w = m_videoArea->width();
    const int h = m_videoArea->height();
    const int pad = 10;
    const bool camerasHidden = (m_currentNav == NavItem::Gallery);

    auto animateButton = [this, animate](QWidget* btn, bool show, const QPoint& targetPos) {
        if (!btn) return;

        if (btn->graphicsEffect()) btn->setGraphicsEffect(nullptr);

        const auto running = btn->findChildren<QPropertyAnimation*>();
        for (QPropertyAnimation* a : running) a->stop();

        const bool currentlyVisible = btn->isVisible();

        if (!m_animFABsEnabled || !animate) {
            btn->move(targetPos);
            btn->setVisible(show);
            if (show) btn->raise();
            return;
        }

        if (show) {
            if (currentlyVisible) {
                if (btn->pos() != targetPos) {
                    QPropertyAnimation* posAnim = new QPropertyAnimation(btn, "pos", btn);
                    posAnim->setDuration(m_animSpeedMs);
                    posAnim->setStartValue(btn->pos());
                    posAnim->setEndValue(targetPos);
                    posAnim->setEasingCurve(QEasingCurve::OutCubic);
                    posAnim->start(QAbstractAnimation::DeleteWhenStopped);
                }
            } else {
                btn->move(targetPos + QPoint(0, 25));
                btn->show();
                btn->raise();

                QPropertyAnimation* posAnim = new QPropertyAnimation(btn, "pos", btn);
                posAnim->setDuration(m_animSpeedMs);
                posAnim->setStartValue(btn->pos());
                posAnim->setEndValue(targetPos);
                posAnim->setEasingCurve(QEasingCurve::OutCubic);
                posAnim->start(QAbstractAnimation::DeleteWhenStopped);
            }
        } else {
            if (currentlyVisible) {
                QPropertyAnimation* posAnim = new QPropertyAnimation(btn, "pos", btn);
                posAnim->setDuration(m_animSpeedMs);
                posAnim->setStartValue(btn->pos());
                posAnim->setEndValue(btn->pos() + QPoint(0, 25));
                posAnim->setEasingCurve(QEasingCurve::InCubic);
                QObject::connect(posAnim, &QPropertyAnimation::finished, btn, [btn]() {
                    btn->hide();
                });
                posAnim->start(QAbstractAnimation::DeleteWhenStopped);
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

        animateButton(m_btnFloatingPreviewToggle, m_previewMode, QPoint(pad, pad));
    }
    if (m_resHud) {

        const bool show = m_resHudVisible && !camerasHidden;
        if (show) {
            m_resHud->adjustSize();
            m_resHud->move(w - pad - m_resHud->width(), 40);
            m_resHud->raise();
        }
        m_resHud->setVisible(show);
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
    updateDiffReadout(true);
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    saveSettings();

    for (QWidget* w : QApplication::topLevelWidgets()) {
        if (w != this && w->isVisible()) w->close();
    }
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

    if (m_resPill) m_resPill->setVisible(!ultraCompact);
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
        m_thumbnails.clear();
        return;
    }

    QStringList filters = {"*.png", "*.jpg", "*.jpeg", "*.bmp"};
    QFileInfoList files = metricsDir.entryInfoList(filters, QDir::Files, QDir::Time);

    const QSize thumbMax(180, 130);
    QHash<QString, Thumbnail> current;
    auto thumbnailFor = [this, &current, thumbMax](const QFileInfo& fi) -> QPixmap {
        const QString path = fi.absoluteFilePath();
        const auto seen = current.constFind(path);
        if (seen != current.constEnd()) return seen->pixmap;
        Thumbnail t = m_thumbnails.value(path);
        if (t.pixmap.isNull() || t.modified != fi.lastModified() || t.size != fi.size()) {
            QImageReader reader(path);
            const QSize full = reader.size();
            if (full.isValid()) reader.setScaledSize(full.scaled(thumbMax, Qt::KeepAspectRatio));
            const QImage img = reader.read();
            t = Thumbnail{ fi.lastModified(), fi.size(), img.isNull() ? QPixmap() : QPixmap::fromImage(img) };
        }
        current.insert(path, t);
        return t.pixmap;
    };

    auto fillList = [&thumbnailFor](QListWidget* lw, const QFileInfoList& src, int maxItems,
                                    const QSize& thumbSize) {
        if (!lw) return;
        lw->clear();
        const QSize grid = lw->gridSize();
        int n = 0;
        for (const QFileInfo& fi : src) {
            if (maxItems > 0 && n >= maxItems) break;
            const QPixmap pix = thumbnailFor(fi);
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
    m_thumbnails.swap(current);
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
        m_fpsPill->setToolTip(QString());
    }
}

void MainWindow::toggleResourceHud()
{
    m_resHudVisible = !m_resHudVisible;
    if (m_resHudVisible && m_resMonitor) updateResourceDisplay(m_resMonitor->last());
    positionFloatingButtons(false);
}

QString MainWindow::formatResourceStats(const ResourceStats& st) const
{
    auto pct = [](double v) { return QString("%1%").arg(qRound(v), 3); };
    QStringList lines;
    if (st.cpuTotal >= 0) {
        QString l = "CPU    " + pct(st.cpuTotal);
        if (!st.cpuCores.isEmpty()) {
            QStringList cores;
            for (double c : st.cpuCores) cores << QString("%1").arg(qRound(c), 3);
            l += "   cores" + cores.join(QString());
        }
        lines << l;
    }
    if (st.appCpu >= 0) {
        QString l = "APP    " + pct(st.appCpu);
        if (st.guiThreadCpu >= 0)    l += QString("   GUI %1%").arg(qRound(st.guiThreadCpu));
        if (st.workerThreadCpu >= 0) l += QString("  CAM %1%").arg(qRound(st.workerThreadCpu));
        lines << l;
    }
    if (st.tempC >= 0 || st.armMHz > 0) {
        QString l = "TEMP  ";
        l += st.tempC >= 0 ? QString("%1\u00B0C").arg(st.tempC, 5, 'f', 1) : QString("  n/a");
        if (st.armMHz > 0) {
            l += QString("   ARM %1").arg(st.armMHz);
            if (st.armMaxMHz > 0) l += QString("/%1").arg(st.armMaxMHz);
            l += " MHz";
        }
        if (st.fanRpm >= 0) l += QString("   FAN %1 rpm").arg(st.fanRpm);
        lines << l;
    }
    if (st.throttleFlags >= 0) {
        lines << "THROT  now: " + ResourceMonitor::throttleText(st.throttleFlags, false)
                 + "   since boot: " + ResourceMonitor::throttleText(st.throttleFlags, true);
    }
    if (st.memTotalMB > 0) {
        QString l = QString("RAM   %1/%2 GB").arg(st.memUsedMB / 1024.0, 0, 'f', 1)
                                          .arg(st.memTotalMB / 1024.0, 0, 'f', 1);
        if (st.appRssMB >= 0) l += QString("   APP %1 MB").arg(st.appRssMB);
        lines << l;
    }
    if (st.processedFps >= 0) {
        lines << QString("FPS   camera %1   shown %2   dropped %3")
                     .arg(st.processedFps, 0, 'f', 1)
                     .arg(st.displayedFps, 0, 'f', 1)
                     .arg(st.droppedFps, 0, 'f', 1);
    } else {
        lines << "FPS   cameras stopped";
    }
    return lines.join('\n');
}

void MainWindow::updateResourceDisplay(const ResourceStats& st)
{
    if (m_resPill) {
        const unsigned now = st.throttleFlags >= 0
            ? unsigned(st.throttleFlags) & PiThrottle::NowMask : 0u;
        const char* state = "ok";
        if ((now & (PiThrottle::ThrottledNow | PiThrottle::UnderVoltageNow)) || st.tempC >= 80.0)
            state = "err";
        else if (now || st.tempC >= 70.0 || st.cpuTotal >= 90.0)
            state = "warn";

        QString text = st.cpuTotal >= 0 ? QString("CPU %1%").arg(qRound(st.cpuTotal), 3)
                                        : QString("CPU  --");
        if (st.tempC >= 0) text += QString(" \u00B7 %1\u00B0C").arg(qRound(st.tempC));
        if (now & (PiThrottle::ThrottledNow | PiThrottle::FreqCappedNow)) text += " \u00B7 THR";
        else if (now & PiThrottle::UnderVoltageNow) text += " \u00B7 LOW V";
        setPillState(m_resPill, state, text);
        m_resPill->setToolTip(formatResourceStats(st)
                              + "\n\nAPP/GUI/CAM are % of one core (4 cores = 400%)."
                                "\nClick to toggle the overlay.");
    }

    if (m_fpsPill && m_camerasOpen && st.processedFps >= 0) {
        const bool behind = m_requestedFps > 0 && st.processedFps < 0.9 * m_requestedFps;
        setPillState(m_fpsPill, behind ? "warn" : "ok",
                     QString("STREAMING %1 FPS").arg(qRound(st.processedFps), 2));
        m_fpsPill->setToolTip(QString("Camera %1 fps (requested %2), shown %3 fps, dropped %4 fps")
                                  .arg(st.processedFps, 0, 'f', 1).arg(m_requestedFps)
                                  .arg(st.displayedFps, 0, 'f', 1).arg(st.droppedFps, 0, 'f', 1));
    }

    if (m_resHud && m_resHudVisible && m_resHudText) {
        const QString text = formatResourceStats(st);
        if (m_resHudText->text() != text) {
            m_resHudText->setText(text);
            positionFloatingButtons(false);
        }
    }
}

void MainWindow::refreshCameraModes()
{
    m_comboCamSet->clear();

    struct Preset { int w, h, fps; };
    static const Preset kPresets[] = {
        {640, 480, 30}, {640, 480, 60}, {1280, 720, 30}, {1280, 720, 60},
        {1920, 1080, 30}, {1920, 1080, 60},
    };
    const QSize native = sensorNativeSize();
    for (const Preset& p : kPresets) {
        if (native.isValid() && (p.w > native.width() || p.h > native.height()
                                 || p.fps > sensorMaxFps(p.w, p.h)))
            continue;
        m_comboCamSet->addItem(QString("%1x%2 @ %3 FPS").arg(p.w).arg(p.h).arg(p.fps),
                               QVariantList{p.w, p.h, p.fps});
    }
    if (native.isValid()) {
        const int maxFps = static_cast<int>(sensorMaxFps(native.width(), native.height()));
        QList<int> rates;
        if (maxFps > 30) rates << 30;
        if (maxFps > 0) rates << maxFps;
        for (int fps : rates) {
            m_comboCamSet->addItem(QString("%1x%2 @ %3 FPS (native)")
                                       .arg(native.width()).arg(native.height()).arg(fps),
                                   QVariantList{native.width(), native.height(), fps});
        }
    }
    m_comboCamSet->addItem("Custom...", QVariantList{0, 0, 0});
}

void MainWindow::probeCameras()
{
    m_libCameraIds.clear();
    m_sensorModes.clear();

    QProcess process;
    process.start("rpicam-hello", QStringList() << "--list-cameras");
    if (!process.waitForFinished(5000) || process.exitCode() != 0) {
        process.start("libcamera-hello", QStringList() << "--list-cameras");
        process.waitForFinished(5000);
    }
    const QString output = QString::fromLocal8Bit(process.readAllStandardOutput());

    static const QRegularExpression headerRe(R"(^\s*\d+\s*:.*\((/base/[^)]+)\))");
    static const QRegularExpression modeRe(R"((\d+)x(\d+)\s*\[([\d.]+)\s*fps)");
    for (const QString& line : output.split('\n')) {
        const QRegularExpressionMatch h = headerRe.match(line);
        if (h.hasMatch()) {
            m_libCameraIds << h.captured(1);
            m_sensorModes.emplace_back();
            continue;
        }
        if (m_sensorModes.empty()) continue;
        QRegularExpressionMatchIterator it = modeRe.globalMatch(line);
        while (it.hasNext()) {
            const QRegularExpressionMatch m = it.next();
            m_sensorModes.back().push_back({m.captured(1).toInt(), m.captured(2).toInt(),
                                            m.captured(3).toDouble()});
        }
    }
}

QStringList MainWindow::getLibCameraIds()
{
    if (m_libCameraIds.size() < 2) probeCameras();
    return m_libCameraIds;
}

QSize MainWindow::sensorNativeSize() const
{
    QSize common;
    const size_t n = std::min<size_t>(2, m_sensorModes.size());
    for (size_t c = 0; c < n; ++c) {
        QSize best;
        for (const SensorMode& m : m_sensorModes[c])
            if (m.width * m.height > best.width() * best.height()) best = QSize(m.width, m.height);
        if (!best.isValid()) return QSize();
        common = common.isValid() ? common.boundedTo(best) : best;
    }
    return common;
}

double MainWindow::sensorMaxFps(int width, int height) const
{
    double overall = -1.0;
    const size_t n = std::min<size_t>(2, m_sensorModes.size());
    for (size_t c = 0; c < n; ++c) {
        double best = 0.0;
        for (const SensorMode& m : m_sensorModes[c])
            if (m.width >= width && m.height >= height) best = std::max(best, m.fps);
        overall = overall < 0 ? best : std::min(overall, best);
    }
    return std::max(0.0, overall);
}

bool MainWindow::fitToSensor(int& width, int& height, int& fps) const
{
    const QSize native = sensorNativeSize();
    if (!native.isValid() || width <= 0 || height <= 0) return false;
    bool changed = false;
    if (width > native.width() || height > native.height()) {
        const double k = std::min(double(native.width()) / width, double(native.height()) / height);
        width  = std::max(2, int(width * k) & ~1);
        height = std::max(2, int(height * k) & ~1);
        changed = true;
    }
    const int maxFps = static_cast<int>(sensorMaxFps(width, height));
    if (maxFps > 0 && fps > maxFps) {
        fps = maxFps;
        changed = true;
    }
    return changed;
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
        "video/x-raw, width=%2, height=%3, framerate=%4/1, format=BGR ! "
        "appsink drop=1 max-buffers=1 sync=false")
        .arg(cameraId)
        .arg(width)
        .arg(height)
        .arg(fps)
        .arg(controls)
        .toStdString();
}

void MainWindow::applyExposureControls()
{

    if (!m_camerasOpen) return;
    if (!m_libcameraSession) {
        m_statusBar->showMessage("Manual gain / shutter apply to libcamera (CSI) cameras only; "
                                 "these cameras keep their own exposure.", 5000);
        return;
    }
    if (!m_exposureRestartTimer) {
        m_exposureRestartTimer = new QTimer(this);
        m_exposureRestartTimer->setSingleShot(true);
        m_exposureRestartTimer->setInterval(400);
        connect(m_exposureRestartTimer, &QTimer::timeout, this, [this]() {
            if (!m_camerasOpen) return;
            if (m_cameraTransition) {
                m_exposureRestartTimer->start();
                return;
            }
            m_statusBar->showMessage("Restarting cameras to apply exposure...", 2000);
            cameraTransition([this]() {
                closeCameras();
                openCameras();
            });
        });
    }
    m_exposureRestartTimer->start();
}

void MainWindow::cameraTransition(const std::function<void()>& fn)
{

    if (m_cameraTransition) return;
    m_cameraTransition = true;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    fn();
    QApplication::restoreOverrideCursor();
    QApplication::processEvents();
    m_cameraTransition = false;
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

    {
        const int askW = reqW, askH = reqH, askFps = reqFps;
        if (fitToSensor(reqW, reqH, reqFps)) {
            m_statusBar->showMessage(QString("%1x%2 @ %3 exceeds the sensor; using %4x%5 @ %6.")
                .arg(askW).arg(askH).arg(askFps).arg(reqW).arg(reqH).arg(reqFps), 5000);
        }
    }

    bool started = false;
    QStringList camPaths = getLibCameraIds();
    const bool libcamera = camPaths.size() >= 2;
    if (!libcamera) {
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
            started = m_worker->startCamerasV4L2(c1, c2, reqW, reqH, reqFps);
        } else {
            m_statusBar->showMessage("Error: Could not find two fallback cameras.", 3000);
            return;
        }
    } else {
        std::string p1 = makeGStreamerPipeline(camPaths[0], reqW, reqH, reqFps, 0);
        std::string p2 = makeGStreamerPipeline(camPaths[1], reqW, reqH, reqFps, 1);
        started = m_worker->startCameras(p1, p2, reqW, reqH, reqFps);
        if (!started) m_libCameraIds.clear();
    }
    if (!started) {
        return;
    }

    m_camerasOpen = true;
    m_requestedFps = reqFps;
    m_libcameraSession = libcamera;

    if (m_comboCamSet) m_comboCamSet->setEnabled(false);

    m_btnFabStream->setToolTip("Stop streaming");
    m_btnFabStream->setStyleSheet(QString(
        "QPushButton { background: %1; border: 1px solid %1; border-radius: 20px; }"
        "QPushButton:hover { background: #e0695d; border-color: #e0695d; }")
        .arg(T::accent));
    if (m_fabStreamIcon) {
        m_fabStreamIcon->setText(QString::fromUtf8("\u25A0"));
        m_fabStreamIcon->setStyleSheet(QString(
            "QLabel { background: transparent; color: %1; border: none;"
            " font-family: 'Segoe UI Symbol','Segoe UI Emoji','Arial Unicode MS';"
            " font-size: 14px; font-weight: 800; }").arg(T::onAccent));
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
    updateDiffReadout(true);

    m_statusBar->showMessage("Cameras OK. Noise suppression active.", 4000);
}

void MainWindow::closeCameras()
{
    m_worker->stopCameras();
    m_camerasOpen = false;
    if (m_kCalTarget > 0) stopKCalibration("Calibrate k: stopped, cameras closed. k unchanged.");

    if (m_comboCamSet) m_comboCamSet->setEnabled(true);

    if (m_btnFabStream) {

        m_btnFabStream->setToolTip("Start streaming");
        m_btnFabStream->setStyleSheet(QString(
            "QPushButton { background: #161616; border: 1px solid %1; border-radius: 20px; }"
            "QPushButton:hover { background: #2e0e0b; border: 1px solid %2; }")
            .arg(T::borderStrong).arg(T::accent));
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
    updateDiffReadout(true);

    m_statusBar->showMessage("Cameras closed.", 2000);
}

cv::Mat MainWindow::fuseCameras(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat out;
    cv::addWeighted(a, 0.5, b, 0.5 * m_channelK, 0.0, out);
    return out;
}

static const std::array<cv::Vec3b, 256>& jetLut()
{
    static const std::array<cv::Vec3b, 256> lut = [] {
        cv::Mat index(1, 256, CV_8U);
        for (int i = 0; i < 256; ++i) index.at<uchar>(0, i) = static_cast<uchar>(i);
        cv::Mat bgr;
        cv::applyColorMap(index, bgr, cv::COLORMAP_JET);
        std::array<cv::Vec3b, 256> table;
        for (int i = 0; i < 256; ++i) table[i] = bgr.at<cv::Vec3b>(0, i);
        return table;
    }();
    return lut;
}

static QImage diffScaleRamp()
{
    const auto& lut = jetLut();
    QImage ramp(1, 256, QImage::Format_RGB32);
    for (int i = 0; i < 256; ++i) {
        const cv::Vec3b c = lut[255 - i];
        ramp.setPixel(0, i, qRgb(c[2], c[1], c[0]));
    }
    return ramp;
}

cv::Mat MainWindow::applyDiffView(const cv::Mat& d1, const cv::Mat& d2,
                                  const cv::Mat& valid1, const cv::Mat& valid2)
{
    cv::Mat a = d1, b = d2, vb = valid2;
    if (b.size() != a.size()) {
        cv::resize(b, b, a.size());
        if (!vb.empty()) cv::resize(vb, vb, a.size(), 0, 0, cv::INTER_NEAREST);
    }

    cv::Mat ga, gb;
    if (a.channels() == 3) cv::cvtColor(a, ga, cv::COLOR_BGR2GRAY);
    else ga = a;
    if (b.channels() == 3) cv::cvtColor(b, gb, cv::COLOR_BGR2GRAY);
    else gb = b;

    DiffStats st;
    st.valid = true;
    st.k = m_channelK;
    st.noiseFloor = m_noiseFloor;
    st.clipLevel = m_clipLevel;

    const double f1 = m_lastFocus1;
    const double f2 = m_lastFocus2 * m_channelK * m_channelK;
    if (f1 > 1.0 && f2 > 1.0) {
        st.sharpRatio = f1 / f2;
        const double ratio = (f1 > f2) ? (f1 / f2) : (f2 / f1);
        if (ratio > 1.15) {
            const double sigma = std::min(6.0, 0.6 * std::sqrt(ratio - 1.0));
            int k = static_cast<int>(std::ceil(sigma * 3.0)) | 1;
            if (k < 3) k = 3;
            if (k > 31) k = 31;
            if (f1 > f2) cv::GaussianBlur(ga, ga, cv::Size(k, k), sigma);
            else         cv::GaussianBlur(gb, gb, cv::Size(k, k), sigma);
            st.psfChannel = (f1 > f2) ? 1 : 2;
            st.psfSigma = sigma;
        }
    }

    cv::Mat valid;
    if (!valid1.empty() && !vb.empty()) cv::bitwise_and(valid1, vb, valid);
    else if (!valid1.empty()) valid = valid1;
    else if (!vb.empty()) valid = vb;

    const int rows = ga.rows, cols = ga.cols;
    cv::Mat d(ga.size(), CV_16S);
    std::vector<int> rowMax(rows, 0), rowBoth(rows, 0), rowOne(rows, 0), rowInside(rows, 0);
    const float k = static_cast<float>(m_channelK);
    const int noiseFloor = m_noiseFloor, clip = m_clipLevel;
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const uchar* pa = ga.ptr<uchar>(y);
            const uchar* pb = gb.ptr<uchar>(y);
            const uchar* pv = valid.empty() ? nullptr : valid.ptr<uchar>(y);
            short* pd = d.ptr<short>(y);
            int maxAbs = 0, both = 0, one = 0, inside = 0;
            for (int x = 0; x < cols; ++x) {
                if (pv && !pv[x]) { pd[x] = 0; continue; }
                ++inside;
                const bool c1 = pa[x] >= clip, c2 = pb[x] >= clip;
                int v = cvRound(pa[x] - k * pb[x]);
                if (std::abs(v) <= noiseFloor || (c1 && c2)) v = 0;
                pd[x] = static_cast<short>(v);
                if (c1 && c2) ++both;
                else if (c1 || c2) ++one;
                maxAbs = std::max(maxAbs, std::abs(v));
            }
            rowMax[y] = maxAbs;
            rowBoth[y] = both;
            rowOne[y] = one;
            rowInside[y] = inside;
        }
    });
    long long both = 0, one = 0, inside = 0;
    for (int y = 0; y < rows; ++y) {
        st.maxValue = std::max(st.maxValue, static_cast<double>(rowMax[y]));
        both += rowBoth[y];
        one += rowOne[y];
        inside += rowInside[y];
    }
    const double total = static_cast<double>(rows) * cols;
    st.noData = total > 0.0 ? 1.0 - inside / total : 0.0;
    if (inside > 0) {
        st.clippedBoth = static_cast<double>(both) / inside;
        st.clippedOne = static_cast<double>(one) / inside;
    }

    const bool stretch = m_chkStretch && m_chkStretch->isChecked();
    st.gain = (stretch && st.maxValue > 0.0) ? 255.0 / st.maxValue : 1.0;
    st.fullScale = 255.0 / st.gain;
    const auto& lut = jetLut();
    const float gain = static_cast<float>(st.gain);
    cv::Mat color(ga.size(), CV_8UC3);
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const short* pd = d.ptr<short>(y);
            cv::Vec3b* pc = color.ptr<cv::Vec3b>(y);
            for (int x = 0; x < cols; ++x)
                pc[x] = lut[std::min(255, cvRound(std::abs(pd[x]) * gain))];
        }
    });

    m_diffStats = st;
    m_lastDiffA = ga;
    m_lastDiffB = gb;
    m_lastDiffValid = valid;
    return color;
}

struct ChannelRatio {
    double k = 0.0;
    double spread = 0.0;
    int pixels = 0;
};

static ChannelRatio channelRatio(const cv::Mat& a, const cv::Mat& b, const cv::Mat& valid, int clip)
{
    std::vector<float> ratios;
    ratios.reserve(a.total() / 4 + 1);
    for (int y = 0; y < a.rows; y += 2) {
        const uchar* pa = a.ptr<uchar>(y);
        const uchar* pb = b.ptr<uchar>(y);
        const uchar* pv = valid.empty() ? nullptr : valid.ptr<uchar>(y);
        for (int x = 0; x < a.cols; x += 2) {
            if (pv && !pv[x]) continue;
            if (pa[x] < kKCalMinLevel || pb[x] < kKCalMinLevel) continue;
            if (pa[x] >= clip || pb[x] >= clip) continue;
            ratios.push_back(static_cast<float>(pa[x]) / static_cast<float>(pb[x]));
        }
    }
    ChannelRatio r;
    r.pixels = static_cast<int>(ratios.size());
    if (r.pixels < kKCalMinPixels) return r;
    auto quantile = [&ratios](double q) {
        const size_t i = std::min(ratios.size() - 1, static_cast<size_t>(q * static_cast<double>(ratios.size())));
        std::nth_element(ratios.begin(), ratios.begin() + static_cast<std::ptrdiff_t>(i), ratios.end());
        return static_cast<double>(ratios[i]);
    };
    r.k = quantile(0.5);
    const double q1 = quantile(0.25), q3 = quantile(0.75);
    r.spread = (q3 - q1) / (2.0 * r.k);
    return r;
}

template <typename V>
static V medianOf(std::vector<V> v)
{
    const size_t i = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i), v.end());
    return v[i];
}

void MainWindow::calibrateChannelBalance()
{
    if (m_kCalTarget > 0) {
        stopKCalibration("Calibrate k: cancelled, k unchanged.");
        return;
    }
    if (!m_camerasOpen || m_frame1.empty() || m_frame2.empty()) {
        m_statusBar->showMessage("Calibrate k: open the cameras first.", 3000);
        return;
    }
    if (!m_isDiffMode) setDiffMode(true);
    m_kCalSamples.clear();
    m_kCalPixels.clear();
    m_kCalSpreads.clear();
    m_kCalTarget = kKCalFrames;
    m_btnCalibrateK->setText(QString("Cancel 0/%1").arg(kKCalFrames));
    m_statusBar->showMessage("Calibrate k: measuring, keep the scene and the rig still.");
}

void MainWindow::collectKSample()
{
    if (m_kCalTarget <= 0) return;
    if (!m_isDiffMode || m_lastDiffA.empty() || m_lastDiffB.empty()) {
        stopKCalibration("Calibrate k: stopped, the Diff view was left. k unchanged.");
        return;
    }
    const ChannelRatio r = channelRatio(m_lastDiffA, m_lastDiffB, m_lastDiffValid, m_clipLevel);
    if (r.pixels < kKCalMinPixels) {
        stopKCalibration(QString("Calibrate k: too few well-exposed pixels (%1, need %2 at %3..%4 DN in both "
                                 "cameras). Light the scene evenly or adjust the exposure. k unchanged.")
                             .arg(r.pixels).arg(kKCalMinPixels).arg(kKCalMinLevel).arg(m_clipLevel - 1));
        return;
    }
    m_kCalSamples.push_back(r.k);
    m_kCalPixels.push_back(r.pixels);
    m_kCalSpreads.push_back(r.spread);
    const int n = static_cast<int>(m_kCalSamples.size());
    if (n < m_kCalTarget) {
        m_btnCalibrateK->setText(QString("Cancel %1/%2").arg(n).arg(m_kCalTarget));
        return;
    }

    double mean = 0.0;
    for (double v : m_kCalSamples) mean += v;
    mean /= n;
    double var = 0.0;
    for (double v : m_kCalSamples) var += (v - mean) * (v - mean);

    KCalibration rec;
    rec.source = "calibrated";
    rec.when = QDateTime::currentDateTime();
    rec.frames = n;
    rec.pixels = medianOf(m_kCalPixels);
    rec.sigma = n > 1 ? std::sqrt(var / (n - 1)) : 0.0;
    rec.spread = medianOf(m_kCalSpreads);
    rec.exposure = exposureSignature();
    rec.frameSize = QSize(m_lastDiffA.cols, m_lastDiffA.rows);
    rec.perFrame = m_kCalSamples;
    const double k = std::clamp(medianOf(m_kCalSamples), m_spnChannelK->minimum(), m_spnChannelK->maximum());

    m_kCalTarget = 0;
    m_btnCalibrateK->setText("Calibrate k");
    setChannelK(k, rec);
    saveKCalibration();

    QString msg = QString("k = %1 ± %2 (%3 frames, %4 px each), saved.")
                      .arg(m_channelK, 0, 'f', 4).arg(rec.sigma, 0, 'f', 4).arg(n).arg(rec.pixels);
    if (rec.spread > 0.05)
        msg += QString(" CAM1/CAM2 varies by ±%1 % across the frame: the scene may be polarized or unevenly lit.")
                   .arg(rec.spread * 100.0, 0, 'f', 1);
    if (!(m_manualExposure && m_libcameraSession))
        msg += " Exposure is not locked: each camera adapts on its own, so k will drift.";
    m_statusBar->showMessage(msg, 15000);
}

void MainWindow::stopKCalibration(const QString& message)
{
    m_kCalTarget = 0;
    m_kCalSamples.clear();
    m_kCalPixels.clear();
    m_kCalSpreads.clear();
    if (m_btnCalibrateK) m_btnCalibrateK->setText("Calibrate k");
    if (!message.isEmpty()) m_statusBar->showMessage(message, 8000);
}

void MainWindow::setChannelK(double k, const KCalibration& record)
{
    m_kCal = record;
    m_kProgrammatic = true;
    m_spnChannelK->setValue(k);
    m_kProgrammatic = false;
    m_channelK = m_spnChannelK->value();
    updateKInfo();
    updateDiffReadout(true);
}

void MainWindow::saveKCalibration()
{
    QSettings s(settingsPath(), QSettings::IniFormat);
    s.setValue("channelK", m_channelK);
    s.beginGroup("ChannelBalance");
    s.remove("");
    s.setValue("k", m_channelK);
    s.setValue("source", m_kCal.source);
    if (m_kCal.when.isValid()) s.setValue("when", m_kCal.when.toString(Qt::ISODate));
    if (m_kCal.source == "calibrated") {
        s.setValue("frames", m_kCal.frames);
        s.setValue("pixelsPerFrame", m_kCal.pixels);
        s.setValue("sigma", m_kCal.sigma);
        s.setValue("pixelSpread", m_kCal.spread);
        s.setValue("exposure", m_kCal.exposure);
        s.setValue("frameWidth", m_kCal.frameSize.width());
        s.setValue("frameHeight", m_kCal.frameSize.height());
        QStringList perFrame;
        for (double v : m_kCal.perFrame) perFrame << QString::number(v, 'f', 6);
        s.setValue("perFrame", perFrame);
    }
    s.endGroup();
}

void MainWindow::loadKCalibration(QSettings& s)
{
    double k = s.value("channelK", 1.0).toDouble();
    KCalibration rec;
    s.beginGroup("ChannelBalance");
    if (s.contains("k")) k = s.value("k").toDouble();
    rec.source = s.value("source").toString();
    rec.when = QDateTime::fromString(s.value("when").toString(), Qt::ISODate);
    rec.frames = s.value("frames", 0).toInt();
    rec.pixels = s.value("pixelsPerFrame", 0).toInt();
    rec.sigma = s.value("sigma", 0.0).toDouble();
    rec.spread = s.value("pixelSpread", 0.0).toDouble();
    rec.exposure = s.value("exposure").toString();
    rec.frameSize = QSize(s.value("frameWidth", 0).toInt(), s.value("frameHeight", 0).toInt());
    for (const QString& v : s.value("perFrame").toStringList()) rec.perFrame.push_back(v.toDouble());
    s.endGroup();
    setChannelK(k, rec);
}

QString MainWindow::exposureSignature() const
{
    if (!(m_manualExposure && m_libcameraSession)) return "auto";
    if (m_perCameraExposure)
        return QString("manual, CAM1 gain %1 shutter %2 us, CAM2 gain %3 shutter %4 us")
            .arg(m_gainQ8 / 256.0, 0, 'f', 2).arg(m_shutterUs)
            .arg(m_gain2Q8 / 256.0, 0, 'f', 2).arg(m_shutter2Us);
    return QString("manual, gain %1 shutter %2 us").arg(m_gainQ8 / 256.0, 0, 'f', 2).arg(m_shutterUs);
}

void MainWindow::updateKInfo()
{
    if (!m_lblKInfo) return;
    const KCalibration& c = m_kCal;
    const QString when = c.when.isValid() ? c.when.toString("dd.MM.yyyy HH:mm") : QString("?");
    QString text, warning;
    if (c.source == "calibrated") {
        text = QString("Calibrated %1  ·  %2 frames\nσ %3  ·  pixel spread ±%4 %")
                   .arg(when).arg(c.frames).arg(c.sigma, 0, 'f', 4).arg(c.spread * 100.0, 0, 'f', 1);
        if (!c.exposure.isEmpty() && c.exposure != exposureSignature())
            warning = "Exposure changed since calibration: recalibrate k.";
    } else if (c.source == "manual") {
        text = QString("Entered by hand %1").arg(when);
    } else {
        text = std::abs(m_channelK - 1.0) < 1e-9 ? QString("Not calibrated (k = 1)") : QString("No calibration record");
    }
    QString html = QString("<span style='color:%1'>%2</span>")
                       .arg(T::textFaint, text.toHtmlEscaped().replace('\n', "<br>"));
    if (!warning.isEmpty())
        html += QString("<br><span style='color:%1'>%2</span>").arg(T::warn, warning.toHtmlEscaped());
    if (m_lblKInfo->text() != html) m_lblKInfo->setText(html);
}

void MainWindow::updateDiffReadout(bool force)
{
    if (!m_lblDiffInfo) return;
    if (!force) {
        if (!m_diffReadoutTimer) {
            m_diffReadoutTimer = new QTimer(this);
            m_diffReadoutTimer->setSingleShot(true);
            m_diffReadoutTimer->setInterval(250);
            connect(m_diffReadoutTimer, &QTimer::timeout, this, [this]() { updateDiffReadout(true); });
        }
        if (!m_diffReadoutTimer->isActive()) m_diffReadoutTimer->start();
        return;
    }
    if (m_diffReadoutTimer) m_diffReadoutTimer->stop();
    updateKInfo();

    const DiffStats& st = m_diffStats;
    const bool stretch = m_chkStretch && m_chkStretch->isChecked();
    QStringList lines;
    lines << QString("D = CAM1 − k·CAM2, k = %1  ·  |D| ≤ %2 DN shown as 0  ·  clipped ≥ %3 DN")
                 .arg(m_channelK, 0, 'f', 4).arg(m_noiseFloor).arg(m_clipLevel);
    if (m_isDiffMode && m_camerasOpen && st.valid) {
        lines << QString("Colour = |D| × %1  (%2: top of scale = %3 DN)  ·  max |D| %4 DN")
                     .arg(st.gain, 0, 'f', st.gain < 10.0 ? 2 : 1)
                     .arg(stretch ? "stretch" : "no stretch")
                     .arg(st.fullScale, 0, 'f', st.fullScale < 10.0 ? 1 : 0)
                     .arg(st.maxValue, 0, 'f', 0);
        QString l3 = QString("Clipped: both %1 %, one %2 %")
                         .arg(st.clippedBoth * 100.0, 0, 'f', 1).arg(st.clippedOne * 100.0, 0, 'f', 1);
        if (st.noData > 0.0005)
            l3 += QString("  ·  no overlap %1 %").arg(st.noData * 100.0, 0, 'f', 1);
        l3 += st.sharpRatio > 0.0
            ? QString("  ·  sharpness F1/(k²F2) %1").arg(st.sharpRatio, 0, 'f', 2)
            : QString("  ·  sharpness match");
        l3 += st.psfChannel
            ? QString(" → CAM%1 blurred, σ %2 px").arg(st.psfChannel).arg(st.psfSigma, 0, 'f', 2)
            : QString(" → no blur");
        lines << l3;
    }
    if (m_camerasOpen && !(m_manualExposure && m_libcameraSession))
        lines << "Exposure is not locked (AUTO or camera-controlled): each camera adapts on its own, so k drifts.";
    const QString text = lines.join('\n');
    if (m_lblDiffInfo->text() != text) m_lblDiffInfo->setText(text);
}

bool MainWindow::eventFilter(QObject* obj, QEvent* event)
{
    if (obj == m_spnChannelK && event->type() == QEvent::Wheel && !m_spnChannelK->hasFocus()) {
        event->ignore();
        return true;
    }
    if (obj == m_resPill && event->type() == QEvent::MouseButtonRelease) {
        toggleResourceHud();
        return true;
    }
    if (obj == m_videoArea
        && (event->type() == QEvent::Resize || event->type() == QEvent::Show)) {

        positionFloatingButtons(false);
    }
    return QMainWindow::eventFilter(obj, event);
}

void MainWindow::onFramesProcessed(cv::Mat f1, cv::Mat f2, double focus1, double focus2, bool motionDetected, qint64 frameCount)
{
    if (m_worker) m_worker->m_pendingFrames.fetch_sub(1);

    if (!m_camerasOpen) return;
    ++m_displayedFrames;

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
    if (m_kCalTarget > 0) collectKSample();
}

void MainWindow::updateView()
{
    updateEccPill();

    if (m_frame1.empty() || m_frame2.empty()) return;

    cv::Mat f1 = m_frame1.clone();
    cv::Mat f2 = m_frame2.clone();

    const bool diffMode = m_isDiffMode;
    cv::Mat valid1, valid2;
    auto warpValid = [](cv::Mat& valid, const cv::Size& srcSize, const cv::Mat& H, const cv::Size& dstSize) {
        if (valid.empty()) valid = cv::Mat(srcSize, CV_8U, cv::Scalar(255));
        cv::Mat warped;
        cv::warpPerspective(valid, warped, H, dstSize, cv::INTER_LINEAR);
        cv::compare(warped, 255.0, valid, cv::CMP_EQ);
    };

    if (!m_manualAdj1.isIdentity()) {
        cv::Mat H = buildManualHomography(m_manualAdj1, f1.size());
        cv::Mat warped;
        cv::warpPerspective(f1, warped, H, f1.size(), cv::INTER_LINEAR);
        if (diffMode) warpValid(valid1, f1.size(), H, f1.size());
        f1 = warped;
    }
    if (!m_manualAdj2.isIdentity()) {
        cv::Mat H = buildManualHomography(m_manualAdj2, f2.size());
        cv::Mat warped;
        cv::warpPerspective(f2, warped, H, f2.size(), cv::INTER_LINEAR);
        if (diffMode) warpValid(valid2, f2.size(), H, f2.size());
        f2 = warped;
    }

    if (m_isAligned && m_alignCalibSize.width > 0 && m_alignCalibSize != f1.size()) {
        invalidateAlignment(QString("frame size changed (%1x%2 -> %3x%4)")
                                .arg(m_alignCalibSize.width).arg(m_alignCalibSize.height)
                                .arg(f1.cols).arg(f1.rows));
    }

    const bool eccReady = m_isAligned && !m_eccWarpMatrix.empty();
    const bool wantAlign = m_chkAlign && m_chkAlign->isChecked() && eccReady;
    const bool wantFusion = m_chkFusion && m_chkFusion->isChecked() && eccReady;

    cv::Mat warpedF2 = f2;
    bool warped = false;
    if (wantAlign || wantFusion) {
        try {
            cv::warpPerspective(f2, warpedF2, m_eccWarpMatrix, f1.size(), cv::INTER_LINEAR);
            warped = true;
        }
        catch (const cv::Exception& e) {
            std::cerr << "Warp error: " << e.what() << std::endl;
            warpedF2 = f2;
        }
    }

    cv::Mat alignedF2 = wantAlign ? warpedF2 : f2;
    if (diffMode && wantAlign && warped) warpValid(valid2, f2.size(), m_eccWarpMatrix, f1.size());
    if (wantFusion && !diffMode) {
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
        m_view1->setOverlayText(wantFusion
                                    ? QString("Σ/2 = (CAM1 + %1·CAM2)/2").arg(m_channelK, 0, 'f', 4)
                                    : QString("CAM1  Focus %1").arg(static_cast<int>(m_lastFocus1)), false);
        m_view2->setOverlayColor(QColor(0xce, 0x91, 0x78));
        m_view2->setOverlayText(QString("CAM2  Focus %1").arg(static_cast<int>(m_lastFocus2)), true);

        displayMat(m_view1, f1);
        displayMat(m_view2, alignedF2);
    }
    else
    {
        m_splitter->hide();
        m_resultView->show();

        cv::Mat diff = applyDiffView(f1, alignedF2, valid1, valid2);

        if (showPeaks) {
            drawTarget(diff, maxLoc1, cv::Scalar(0, 255, 255), "P1");
            drawTarget(diff, maxLoc2, cv::Scalar(255, 0, 255), "P2");
        }

        const DiffStats& st = m_diffStats;
        m_resultView->setOverlayColor(QColor(0xff, 0xff, 0xff));
        m_resultView->setOverlayText(QString("DIFF  |CAM1 − %1·CAM2|  ×%2")
                                         .arg(st.k, 0, 'f', 4)
                                         .arg(st.gain, 0, 'f', st.gain < 10.0 ? 2 : 1), false);
        if (m_diffScaleRamp.isNull()) m_diffScaleRamp = diffScaleRamp();
        m_resultView->setColorScale(m_diffScaleRamp, st.fullScale, "|D|, DN");

        m_lastDiffResult = diff.clone();
        displayMat(m_resultView, diff);
        updateDiffReadout();
    }
}

cv::Mat MainWindow::alignmentSignal(const cv::Mat& gray)
{
    cv::Mat g;
    if (gray.channels() == 3) cv::cvtColor(gray, g, cv::COLOR_BGR2GRAY);
    else g = gray;

    cv::Mat eq;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(g, eq);

    cv::Mat blurred;
    cv::GaussianBlur(eq, blurred, cv::Size(0, 0), 1.0);

    cv::Mat gx, gy, mag;
    cv::Sobel(blurred, gx, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);

    cv::normalize(mag, mag, 0.0, 1.0, cv::NORM_MINMAX);
    return mag;
}

double MainWindow::warpDisagreement(const cv::Mat& A, const cv::Mat& B, const cv::Size& sz)
{
    if (A.empty() || B.empty()) return std::numeric_limits<double>::infinity();
    cv::Mat Ad, Bd;
    A.convertTo(Ad, CV_64F);
    B.convertTo(Bd, CV_64F);

    std::vector<cv::Point2f> corners = {
        {0.f, 0.f},
        {static_cast<float>(sz.width - 1), 0.f},
        {static_cast<float>(sz.width - 1), static_cast<float>(sz.height - 1)},
        {0.f, static_cast<float>(sz.height - 1)}
    };
    std::vector<cv::Point2f> wa, wb;
    cv::perspectiveTransform(corners, wa, Ad);
    cv::perspectiveTransform(corners, wb, Bd);

    double worst = 0.0;
    for (size_t i = 0; i < corners.size(); ++i) {
        if (!std::isfinite(wa[i].x) || !std::isfinite(wa[i].y) ||
            !std::isfinite(wb[i].x) || !std::isfinite(wb[i].y)) {
            return std::numeric_limits<double>::infinity();
        }
        const double d = std::hypot(static_cast<double>(wa[i].x) - wb[i].x,
                                    static_cast<double>(wa[i].y) - wb[i].y);
        worst = std::max(worst, d);
    }
    return worst;
}

bool MainWindow::refineCorrespondences(const cv::Mat& sig1, const cv::Mat& sig2,
                                       std::vector<cv::Point2f>& ptsSrc,
                                       std::vector<cv::Point2f>& ptsDst,
                                       const cv::Mat& H, int halfWin, int searchRad)
{
    if (ptsSrc.size() != ptsDst.size() || ptsSrc.empty() || H.empty()) return false;

    std::vector<cv::Point2f> outSrc, outDst;
    outSrc.reserve(ptsSrc.size());
    outDst.reserve(ptsDst.size());

    const int win = 2 * halfWin + 1;

    for (size_t i = 0; i < ptsSrc.size(); ++i) {
        const cv::Point2f& ps = ptsSrc[i];
        const cv::Point2f& pd = ptsDst[i];

        int sx = cvRound(ps.x) - halfWin;
        int sy = cvRound(ps.y) - halfWin;
        if (sx < 0 || sy < 0 || sx + win > sig2.cols || sy + win > sig2.rows) {
            outSrc.push_back(ps); outDst.push_back(pd);
            continue;
        }
        cv::Mat patch = sig2(cv::Rect(sx, sy, win, win));

        int tx = cvRound(pd.x) - halfWin - searchRad;
        int ty = cvRound(pd.y) - halfWin - searchRad;
        int tw = win + 2 * searchRad;
        if (tx < 0 || ty < 0 || tx + tw > sig1.cols || ty + tw > sig1.rows) {
            outSrc.push_back(ps); outDst.push_back(pd);
            continue;
        }
        cv::Mat region = sig1(cv::Rect(tx, ty, tw, tw));

        cv::Scalar mu, sd;
        cv::meanStdDev(patch, mu, sd);
        if (sd[0] < 1e-3) {
            outSrc.push_back(ps); outDst.push_back(pd);
            continue;
        }

        cv::Mat resp;
        try {
            cv::matchTemplate(region, patch, resp, cv::TM_CCOEFF_NORMED);
        } catch (const cv::Exception&) {
            outSrc.push_back(ps); outDst.push_back(pd);
            continue;
        }

        double maxVal = 0.0;
        cv::Point maxLoc;
        cv::minMaxLoc(resp, nullptr, &maxVal, nullptr, &maxLoc);
        if (maxVal < 0.35) {
            outSrc.push_back(ps); outDst.push_back(pd);
            continue;
        }

        double dx = 0.0, dy = 0.0;
        if (maxLoc.x > 0 && maxLoc.x < resp.cols - 1) {
            double l = resp.at<float>(maxLoc.y, maxLoc.x - 1);
            double c = resp.at<float>(maxLoc.y, maxLoc.x);
            double r = resp.at<float>(maxLoc.y, maxLoc.x + 1);
            double den = (l - 2.0 * c + r);
            if (std::abs(den) > 1e-9) dx = 0.5 * (l - r) / den;
        }
        if (maxLoc.y > 0 && maxLoc.y < resp.rows - 1) {
            double u = resp.at<float>(maxLoc.y - 1, maxLoc.x);
            double c = resp.at<float>(maxLoc.y, maxLoc.x);
            double d = resp.at<float>(maxLoc.y + 1, maxLoc.x);
            double den = (u - 2.0 * c + d);
            if (std::abs(den) > 1e-9) dy = 0.5 * (u - d) / den;
        }
        if (!std::isfinite(dx) || std::abs(dx) > 1.0) dx = 0.0;
        if (!std::isfinite(dy) || std::abs(dy) > 1.0) dy = 0.0;

        cv::Point2f refined(static_cast<float>(tx + maxLoc.x + halfWin + dx),
                            static_cast<float>(ty + maxLoc.y + halfWin + dy));

        outSrc.push_back(ps);
        outDst.push_back(refined);
    }

    ptsSrc.swap(outSrc);
    ptsDst.swap(outDst);
    return true;
}

static cv::Mat invertWarp(const cv::Mat& m)
{
    cv::Mat d, inv;
    m.convertTo(d, CV_64F);
    if (d.rows == 2) {
        cv::invertAffineTransform(d, inv);
    } else {
        if (std::abs(cv::determinant(d)) < 1e-12) return cv::Mat();
        inv = d.inv();
        const double h22 = inv.at<double>(2, 2);
        if (std::abs(h22) < 1e-12) return cv::Mat();
        inv /= h22;
    }
    cv::Mat out;
    inv.convertTo(out, CV_32F);
    return out;
}

bool MainWindow::eccPolish(const cv::Mat& sig1, const cv::Mat& sig2,
                           cv::Mat& warp, int motionType, int levels,
                           int iters, double eps, double& outRho)
{
    if (warp.empty() || sig1.empty() || sig2.empty()) return false;

    std::vector<cv::Mat> pyr1, pyr2;
    pyr1.push_back(sig1);
    pyr2.push_back(sig2);
    for (int i = 1; i < levels; ++i) {
        cv::Mat d1, d2;
        if (pyr1.back().cols < 64 || pyr1.back().rows < 64) break;
        cv::pyrDown(pyr1.back(), d1);
        cv::pyrDown(pyr2.back(), d2);
        pyr1.push_back(d1);
        pyr2.push_back(d2);
    }
    const int used = static_cast<int>(pyr1.size());

    cv::Mat w = invertWarp(warp);
    if (w.empty()) return false;

    const double s0 = 1.0 / std::pow(2.0, used - 1);
    auto scaleWarp = [motionType](cv::Mat& m, double s) {
        if (motionType == cv::MOTION_HOMOGRAPHY) {
            m.at<float>(0, 2) *= static_cast<float>(s);
            m.at<float>(1, 2) *= static_cast<float>(s);
            m.at<float>(2, 0) /= static_cast<float>(s);
            m.at<float>(2, 1) /= static_cast<float>(s);
        } else {
            m.at<float>(0, 2) *= static_cast<float>(s);
            m.at<float>(1, 2) *= static_cast<float>(s);
        }
    };
    scaleWarp(w, s0);

    bool any = false;
    outRho = 0.0;

    for (int lvl = used - 1; lvl >= 0; --lvl) {
        cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iters, eps);
        cv::Mat attempt = w.clone();
        try {
            double rho = cv::findTransformECC(pyr1[lvl], pyr2[lvl], attempt,
                                              motionType, crit, cv::noArray(), 5);
            if (std::isfinite(rho)) {
                outRho = rho;
                any = true;
                w = attempt;
            }
        } catch (const cv::Exception&) {
            if (!any) return false;
            scaleWarp(w, std::pow(2.0, lvl));
            break;
        }
        if (lvl > 0) scaleWarp(w, 2.0);
    }

    if (!any) return false;
    for (int r = 0; r < w.rows; ++r)
        for (int c = 0; c < w.cols; ++c)
            if (!std::isfinite(w.at<float>(r, c))) return false;

    const cv::Mat result = invertWarp(w);
    if (result.empty()) return false;
    warp = result;
    return true;
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
    m_statusBar->showMessage("Calibrating (features + refinement)... please wait.");

    const bool eccEnabled = !m_chkEccPolish || m_chkEccPolish->isChecked();

    if (m_calibThread.joinable()) m_calibThread.join();

    m_calibThread = std::thread([this, eccEnabled,
                                 gray1 = std::move(gray1), gray2 = std::move(gray2)]() {
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

        cv::Mat sig1, sig2;
        try {
            sig1 = alignmentSignal(gray1);
            sig2 = alignmentSignal(gray2);
        } catch (const cv::Exception& e) {
            errorMsg = QString("Signal prep failed: ") + QString::fromStdString(e.what());
        }

        double finalRho = 0.0;
        QString refineNote;

        auto runWithDetector = [&](cv::Ptr<cv::Feature2D> detector,
                                   cv::Ptr<cv::DescriptorMatcher> matcher,
                                   float loweRatio,
                                   const QString& tag,
                                   QString& outErr,
                                   QString& outModel,
                                   cv::Mat& outWarp) -> bool {
            std::vector<cv::KeyPoint> kp1, kp2;
            cv::Mat des1, des2;
            cv::Mat det1, det2;
            sig1.convertTo(det1, CV_8U, 255.0);
            sig2.convertTo(det2, CV_8U, 255.0);
            detector->detectAndCompute(det1, cv::noArray(), kp1, des1);
            detector->detectAndCompute(det2, cv::noArray(), kp2, des2);

            if (kp1.size() < 16 || kp2.size() < 16) {
                std::vector<cv::KeyPoint> a1, a2;
                cv::Mat d1, d2;
                detector->detectAndCompute(gray1, cv::noArray(), a1, d1);
                detector->detectAndCompute(gray2, cv::noArray(), a2, d2);
                if (a1.size() > kp1.size() && a2.size() > kp2.size()) {
                    kp1 = a1; kp2 = a2; des1 = d1; des2 = d2;
                }
            }

            if (des1.empty() || des2.empty() || kp1.size() < 8 || kp2.size() < 8) {
                outErr = QString("%1: insufficient keypoints (%2/%3)")
                           .arg(tag).arg(static_cast<int>(kp1.size())).arg(static_cast<int>(kp2.size()));
                return false;
            }

            std::vector<std::vector<cv::DMatch>> knn, knnRev;
            matcher->knnMatch(des2, des1, knn, 2);
            matcher->knnMatch(des1, des2, knnRev, 2);

            std::vector<int> bestRev(kp1.size(), -1);
            for (const auto& pr : knnRev) {
                if (pr.empty()) continue;
                if (pr[0].queryIdx >= 0 && pr[0].queryIdx < static_cast<int>(bestRev.size()))
                    bestRev[pr[0].queryIdx] = pr[0].trainIdx;
            }

            auto collect = [&](float ratio, bool crossCheck,
                               std::vector<cv::Point2f>& ps,
                               std::vector<cv::Point2f>& pd) {
                ps.clear(); pd.clear();
                for (const auto& pair : knn) {
                    if (pair.empty()) continue;
                    bool ok = (pair.size() < 2) ? true
                                                : (pair[0].distance < ratio * pair[1].distance);
                    if (!ok) continue;
                    if (crossCheck) {
                        int t = pair[0].trainIdx;
                        if (t < 0 || t >= static_cast<int>(bestRev.size())) continue;
                        if (bestRev[t] != pair[0].queryIdx) continue;
                    }
                    ps.push_back(kp2[pair[0].queryIdx].pt);
                    pd.push_back(kp1[pair[0].trainIdx].pt);
                }
            };

            std::vector<cv::Point2f> ptsSrc, ptsDst;
            collect(loweRatio, true, ptsSrc, ptsDst);
            if (ptsSrc.size() < 8) collect(0.85f, true, ptsSrc, ptsDst);
            if (ptsSrc.size() < 8) collect(0.9f, false, ptsSrc, ptsDst);

            if (ptsSrc.size() < 6) {
                outErr = QString("%1: only %2 usable matches").arg(tag).arg(static_cast<int>(ptsSrc.size()));
                return false;
            }

            const int minInliers = (ptsSrc.size() < 20) ? 5 : 10;
            const double minRatio = (ptsSrc.size() < 20) ? 0.40 : 0.30;

            cv::Mat inliers;
            cv::Mat H = cv::findHomography(ptsSrc, ptsDst, cv::RANSAC, 3.0, inliers, 8000, 0.999);
            int inlierCount = inliers.empty() ? 0 : cv::countNonZero(inliers);
            double inlierRatio = ptsSrc.empty() ? 0.0
                               : static_cast<double>(inlierCount) / static_cast<double>(ptsSrc.size());

            if (H.empty() || inlierCount < minInliers || inlierRatio < minRatio) {
                outErr = QString("%1: %2/%3 inliers (%4%)")
                           .arg(tag).arg(inlierCount).arg(static_cast<int>(ptsSrc.size()))
                           .arg(static_cast<int>(inlierRatio * 100));
                return false;
            }

            if (!isHomographySane(H, imgSz)) {
                outErr = QString("%1: homography rejected (geometry sanity check failed — likely repetitive pattern)").arg(tag);
                return false;
            }

            std::vector<cv::Point2f> inSrc, inDst;
            for (int i = 0; i < static_cast<int>(ptsSrc.size()); ++i) {
                if (!inliers.empty() && inliers.at<uchar>(i)) {
                    inSrc.push_back(ptsSrc[i]);
                    inDst.push_back(ptsDst[i]);
                }
            }
            if (inSrc.size() >= 6 && !sig1.empty() && !sig2.empty()) {
                std::vector<cv::Point2f> rs = inSrc, rd = inDst;
                if (refineCorrespondences(sig1, sig2, rs, rd, H, 10, 4)) {
                    cv::Mat rIn;
                    cv::Mat H2 = cv::findHomography(rs, rd, cv::RANSAC, 2.0, rIn, 8000, 0.999);
                    int rc = rIn.empty() ? 0 : cv::countNonZero(rIn);
                    if (!H2.empty() && rc >= minInliers && isHomographySane(H2, imgSz) &&
                        warpDisagreement(H, H2, imgSz) < 12.0) {
                        H = H2;
                        refineNote = "+subpx";
                    }
                }
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
            auto sift = cv::SIFT::create(0, 3, 0.008, 12.0, 1.6);
            cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(5);
            cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(64);
            cv::Ptr<cv::DescriptorMatcher> flann = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);

            success = runWithDetector(sift, flann, 0.75f, "SIFT", errorMsg, modelUsed, warpMatrix);

            if (!success) {
                auto orb = cv::ORB::create(8000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 5);
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

#if CV_VERSION_MAJOR < 5
            if (!success) {
                auto akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.0005f, 4, 4);
                cv::Ptr<cv::DescriptorMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING, false);
                QString akErr, akModel;
                cv::Mat akWarp;
                if (runWithDetector(akaze, bf, 0.8f, "AKAZE", akErr, akModel, akWarp)) {
                    success = true;
                    warpMatrix = akWarp;
                    modelUsed = akModel;
                } else {
                    errorMsg += " | " + akErr;
                }
            }
#endif

            if (success && eccEnabled && !sig1.empty() && !sig2.empty()) {
                cv::Mat polished = warpMatrix.clone();
                double rho = 0.0;
                if (eccPolish(sig1, sig2, polished, cv::MOTION_HOMOGRAPHY, 3, 200, 1e-6, rho)) {
                    double shift = warpDisagreement(warpMatrix, polished, imgSz);
                    if (isHomographySane(polished, imgSz) && shift < 25.0) {
                        warpMatrix = polished;
                        finalRho = rho;
                        refineNote += QString("+ECC ρ%1").arg(rho, 0, 'f', 3);
                        std::cerr << "[align] ECC polish accepted, rho=" << rho
                                  << ", corner shift=" << shift << " px" << std::endl;
                    } else {
                        std::cerr << "[align] ECC polish rejected (shift=" << shift
                                  << " px), keeping feature solution" << std::endl;
                    }
                } else {
                    std::cerr << "[align] ECC did not converge, keeping feature solution" << std::endl;
                }
            }
        }
        catch (const cv::Exception& e) {
            errorMsg = QString("Feature stage failed: ") + QString::fromStdString(e.what());
            success = false;
        }

        if (!refineNote.isEmpty()) modelUsed += " " + refineNote.trimmed();

        cv::Mat resultMatrix = success ? warpMatrix : cv::Mat();
        QMetaObject::invokeMethod(this, [this, success, resultMatrix, errorMsg, modelUsed, imgSz, finalRho]() {
            if (success) {
                m_eccWarpMatrix = resultMatrix;
                m_isAligned = true;
                m_alignCalibSize = imgSz;
                m_alignModelInfo = modelUsed;
                m_alignRho = finalRho;
                setPillState(m_eccIndicator, "ok", "MATRIX READY");
                m_statusBar->showMessage("Alignment OK [" + modelUsed + "]", 3000);
                if (!m_chkPersistAlign || m_chkPersistAlign->isChecked()) saveAlignmentMatrix();
            }
            else {
                m_eccWarpMatrix.release();
                m_isAligned = false;
                m_alignCalibSize = cv::Size(0, 0);
                m_alignModelInfo.clear();
                m_alignRho = 0.0;
                setPillState(m_eccIndicator, "err", "NOT CALIBRATED");
                m_statusBar->showMessage("Alignment failed (low texture / too different views): " + errorMsg, 5000);
            }
            if (m_alignDetail) {
                m_alignDetail->setText(success ? ("Solution: " + modelUsed) : QString("No solution."));
            }
            m_btnCalibrateAlign->setEnabled(true);
            m_calibrating = false;
            updateEccPill();
        });
    });
}

void MainWindow::saveAlignmentMatrix()
{
    QSettings s(settingsPath(), QSettings::IniFormat);
    s.beginGroup("Alignment");
    if (m_isAligned && !m_eccWarpMatrix.empty() &&
        m_eccWarpMatrix.rows == 3 && m_eccWarpMatrix.cols == 3) {
        cv::Mat m;
        m_eccWarpMatrix.convertTo(m, CV_64F);
        QStringList vals;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                vals << QString::number(m.at<double>(r, c), 'g', 17);
        s.setValue("warp", vals.join(','));
        s.setValue("calibW", m_alignCalibSize.width);
        s.setValue("calibH", m_alignCalibSize.height);
        s.setValue("model", m_alignModelInfo);
        s.setValue("rho", m_alignRho);
        s.setValue("savedAt", QDateTime::currentDateTime().toString(Qt::ISODate));
    } else {
        s.remove("");
    }
    s.endGroup();
    s.sync();
}

void MainWindow::loadAlignmentMatrix()
{
    QSettings s(settingsPath(), QSettings::IniFormat);
    s.beginGroup("Alignment");
    const QString packed = s.value("warp").toString();
    const int w = s.value("calibW", 0).toInt();
    const int h = s.value("calibH", 0).toInt();
    const QString model = s.value("model").toString();
    const double rho = s.value("rho", 0.0).toDouble();
    const QString savedAt = s.value("savedAt").toString();
    s.endGroup();

    if (packed.isEmpty()) return;

    const QStringList vals = packed.split(',', Qt::SkipEmptyParts);
    if (vals.size() != 9) return;

    cv::Mat m(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
        bool ok = false;
        const double v = vals[i].toDouble(&ok);
        if (!ok || !std::isfinite(v)) return;
        m.at<double>(i / 3, i % 3) = v;
    }
    if (std::abs(cv::determinant(m)) < 1e-9) return;

    m.convertTo(m_eccWarpMatrix, CV_32F);
    m_isAligned = true;
    m_alignCalibSize = cv::Size(w, h);
    m_alignModelInfo = model;
    m_alignRho = rho;

    if (m_alignDetail) {
        QString txt = "Restored: " + (model.isEmpty() ? QString("saved matrix") : model);
        if (w > 0 && h > 0) txt += QString(" @ %1x%2").arg(w).arg(h);
        if (!savedAt.isEmpty()) txt += " (" + savedAt.left(10) + ")";
        m_alignDetail->setText(txt);
    }
    if (m_eccIndicator) setPillState(m_eccIndicator, "info", "RESTORED");
    updateEccPill();
}

void MainWindow::invalidateAlignment(const QString& reason)
{
    if (!m_isAligned && m_eccWarpMatrix.empty()) return;
    m_eccWarpMatrix.release();
    m_isAligned = false;
    m_alignCalibSize = cv::Size(0, 0);
    m_alignModelInfo.clear();
    m_alignRho = 0.0;
    if (m_eccIndicator) setPillState(m_eccIndicator, "err", "NOT CALIBRATED");
    if (m_alignDetail) m_alignDetail->setText("Invalidated: " + reason);
    if (m_statusBar) m_statusBar->showMessage("Alignment invalidated: " + reason, 4000);
    updateEccPill();
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

void MainWindow::displayMat(ImageView* view, const cv::Mat& mat)
{
    if (mat.empty() || view == nullptr) return;
    if (mat.channels() != 1 && mat.channels() != 3) return;

    cv::Mat src = mat;
    const QSize target = view->targetSize(QSize(mat.cols, mat.rows));
    if (target.isValid() && target.width() < mat.cols && target.height() < mat.rows) {
        cv::resize(mat, src, cv::Size(target.width(), target.height()), 0, 0, cv::INTER_AREA);
    }

    auto* bgra = new cv::Mat();
    cv::cvtColor(src, *bgra, src.channels() == 1 ? cv::COLOR_GRAY2BGRA : cv::COLOR_BGR2BGRA);
    QImage img(bgra->data, bgra->cols, bgra->rows, static_cast<qsizetype>(bgra->step),
               QImage::Format_RGB32,
               [](void* m) { delete static_cast<cv::Mat*>(m); }, bgra);
    img.setDevicePixelRatio(view->devicePixelRatioF());
    view->setImage(img);
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
        if (enabled("K") && std::abs(m_channelK - 1.0) > 1e-9) parts << QString("K%1").arg(m_channelK, 0, 'f', 4);
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

    if (execDialog(&dlg, m_btnConfigParams) != QDialog::Accepted) return;

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
        "QPushButton { background: %1; color: %2; border: 1px solid %3; border-radius: 6px;"
        " padding: 4px 14px; font-weight: 600; font-size: 11px;"
        " font-family: 'Space Mono','JetBrains Mono',monospace; letter-spacing: 0.06em; }"
        "QPushButton:checked { background: %4; color: %5; border-color: %4; }")
        .arg(T::bg2).arg(T::text).arg(T::border).arg(T::accent).arg(T::onAccent);
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
        QLabel* lbl = new QLabel(title, row);
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

    QFrame* rowG = buildRow("GAIN (0.10x – 10.00x)",   spnG, sldG);
    QFrame* rowS = buildRow("SHUTTER (50 – 30000 µs)", spnS, sldS);
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

    execDialog(&dlg, m_btnExpChange);
}

void MainWindow::saveSnapshot()
{
    if (m_isDiffMode) {
        saveDiffSnapshot();
    } else {
        saveDualSnapshot(!(m_chkSeparateFiles && m_chkSeparateFiles->isChecked()));
    }
}

static bool saveSnapshotFile(const QString& path, const cv::Mat& mat, const ExifParams& params) {
    std::vector<uint8_t> buf;
    if (!cv::imencode(".jpg", mat, buf)) return false;

    std::vector<uint8_t> finalData = insertExif(buf, params);

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) return false;
    const qint64 size = static_cast<qint64>(finalData.size());
    if (f.write(reinterpret_cast<const char*>(finalData.data()), size) != size) return false;
    f.close();
    return true;
}

static QString uniqueBaseName(const QString& dir, const QString& base, const QStringList& suffixes)
{
    QString candidate = base;
    for (int n = 2;; ++n) {
        bool taken = false;
        for (const QString& s : suffixes) taken = taken || QFile::exists(dir + "/" + candidate + s);
        if (!taken) return candidate;
        candidate = QString("%1_%2").arg(base).arg(n);
    }
}

void MainWindow::saveDiffSnapshot()
{
    if (m_lastDiffResult.empty()) {
        m_statusBar->showMessage("No difference image available to save.", 3000);
        return;
    }

    QString metricsDir = QCoreApplication::applicationDirPath() + "/metrics";
    QDir().mkpath(metricsDir);

    QString filename = uniqueBaseName(metricsDir, buildSnapshotBaseName("diff"), {".jpg"});
    QString filePath = metricsDir + "/" + filename + ".jpg";

    ExifParams p = buildExifParams("diff");
    bool success = saveSnapshotFile(filePath, m_lastDiffResult, p);

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

    const QString baseName = uniqueBaseName(metricsDir, buildSnapshotBaseName("dual"),
        combined ? QStringList{".jpg"} : QStringList{"_cam1.jpg", "_cam2.jpg"});

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
        bool ok = saveSnapshotFile(path, combo, p);
        m_statusBar->showMessage(ok ? "Saved: " + path : "Error saving: " + path, 4000);
    } else {
        QString filename1 = baseName + "_cam1";
        QString filename2 = baseName + "_cam2";
        QString path1 = metricsDir + "/" + filename1 + ".jpg";
        QString path2 = metricsDir + "/" + filename2 + ".jpg";

        ExifParams p1 = buildExifParams("dual_cam1");
        ExifParams p2 = buildExifParams("dual_cam2");

        bool ok1 = saveSnapshotFile(path1, f1, p1);
        bool ok2 = saveSnapshotFile(path2, f2, p2);
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
    if (m_chkFusion && m_chkFusion->isChecked())
        obj["fusionFormula"] = QString("(CAM1 + k*CAM2)/2, k = %1").arg(m_channelK, 0, 'f', 4);
    obj["bilateralFilter"] = m_chkBilateral && m_chkBilateral->isChecked();
    obj["bilateralStrength"] = m_bilateralSlider ? m_bilateralSlider->value() : m_bilateralStrength;
    obj["noiseFloor"]      = m_noiseFloorSlider ? m_noiseFloorSlider->value() : m_noiseFloor;
    obj["intensityStretch"] = m_chkStretch && m_chkStretch->isChecked();
    obj["trackPeaks"]      = m_btnPeakIntensities && m_btnPeakIntensities->isChecked();
    obj["channelBalanceK"] = m_channelK;
    {
        QJsonObject kc;
        kc["k"] = m_channelK;
        kc["source"] = m_kCal.source.isEmpty() ? QString("none") : m_kCal.source;
        if (m_kCal.when.isValid()) kc["at"] = m_kCal.when.toString(Qt::ISODate);
        if (m_kCal.source == "calibrated") {
            kc["method"] = QString("k = median over frames of the per-frame median CAM1/CAM2; pixels %1..clip-1 DN "
                                   "in both cameras, every 2nd row and column, inside the overlap")
                               .arg(kKCalMinLevel);
            kc["frames"] = m_kCal.frames;
            QJsonArray perFrame;
            for (double v : m_kCal.perFrame) perFrame.append(std::round(v * 1e6) / 1e6);
            kc["perFrame"] = perFrame;
            kc["pixelsPerFrame"] = m_kCal.pixels;
            kc["sigma"] = m_kCal.sigma;
            kc["pixelSpread"] = m_kCal.spread;
            kc["exposureAtCalibration"] = m_kCal.exposure;
            kc["exposureNow"] = exposureSignature();
            kc["frameSize"] = QJsonArray{ m_kCal.frameSize.width(), m_kCal.frameSize.height() };
        }
        obj["channelBalance"] = kc;
    }
    obj["clipLevel"]       = m_clipLevel;
    if (m_isDiffMode && m_diffStats.valid) {
        const DiffStats& st = m_diffStats;
        QJsonObject d;
        d["formula"]       = "colour index = |CAM1 - k*CAM2| * displayGain (COLORMAP_JET)";
        d["k"]             = st.k;
        d["noiseFloorDN"]  = st.noiseFloor;
        d["clipLevelDN"]   = st.clipLevel;
        d["displayGain"]   = st.gain;
        d["fullScaleDN"]   = st.fullScale;
        d["maxAbsDiffDN"]  = st.maxValue;
        d["clippedBoth"]   = st.clippedBoth;
        d["clippedOne"]    = st.clippedOne;
        d["noOverlap"]     = st.noData;
        d["sharpnessMatch"] = st.psfChannel
            ? QJsonObject{ { "blurredCamera", st.psfChannel }, { "sigmaPx", st.psfSigma } }
            : QJsonObject();
        obj["diff"] = d;
    }

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
    if (m_isAligned && m_eccWarpMatrix.rows == 3 && m_eccWarpMatrix.cols == 3) {
        cv::Mat H;
        m_eccWarpMatrix.convertTo(H, CV_64F);
        QJsonArray warp;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) warp.append(H.at<double>(r, c));
        align["warp"] = warp;
        align["warpModel"] = m_alignModelInfo;
        align["warpSize"] = QJsonArray{ m_alignCalibSize.width, m_alignCalibSize.height };
    }
    obj["alignment"] = align;

    const bool exposureApplied = m_manualExposure && m_libcameraSession;

    QJsonObject exposure;
    exposure["mode"]       = m_manualExposure ? "manual" : "auto";
    exposure["applied"]    = exposureApplied;
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

    const QString json = QString::fromUtf8(QJsonDocument(obj).toJson(QJsonDocument::Compact));
    std::string ascii;
    ascii.reserve(static_cast<size_t>(json.size()));
    for (const QChar c : json) {
        if (c.unicode() < 0x80) ascii += static_cast<char>(c.unicode());
        else ascii += QString::asprintf("\\u%04x", c.unicode()).toStdString();
    }
    p.description = ascii;

    if (exposureApplied) {
        const bool cam2 = m_perCameraExposure && mode == "dual_cam2";
        p.exposureTimeNum = static_cast<uint32_t>(cam2 ? m_shutter2Us : m_shutterUs);
        p.exposureTimeDen = 1000000;
        const int gainQ8 = std::max(256, cam2 ? m_gain2Q8 : m_gainQ8);
        p.isoSpeed = static_cast<uint16_t>(gainQ8 * 100 / 256);
    }

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
    s.setValue("channelK", m_channelK);
    s.setValue("clipLevel", m_clipLevel);
    s.setValue("appendParams", m_chkAppendParams ? m_chkAppendParams->isChecked() : false);
    s.setValue("separateFiles", m_chkSeparateFiles ? m_chkSeparateFiles->isChecked() : false);
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

    if (m_comboCamSet) {
        QVariantList v = m_comboCamSet->currentData().toList();
        if (v.size() == 3 && v[0].toInt() > 0) {
            s.setValue("camW", v[0].toInt());
            s.setValue("camH", v[1].toInt());
            s.setValue("camFps", v[2].toInt());
        }
    }

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
    s.setValue("resourceHud", m_resHudVisible);
    s.setValue("eccPolish", m_chkEccPolish ? m_chkEccPolish->isChecked() : true);
    s.setValue("persistAlign", m_chkPersistAlign ? m_chkPersistAlign->isChecked() : true);
    s.sync();

    if (!m_chkPersistAlign || m_chkPersistAlign->isChecked()) saveAlignmentMatrix();
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
    m_resHudVisible          = s.value("resourceHud", false).toBool();

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
    if (m_spnChannelK) loadKCalibration(s);
    if (m_spnClipLevel) m_spnClipLevel->setValue(s.value("clipLevel", 254).toInt());
    if (m_chkAppendParams) m_chkAppendParams->setChecked(s.value("appendParams", false).toBool());
    if (m_chkSeparateFiles) m_chkSeparateFiles->setChecked(s.value("separateFiles", false).toBool());
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

    if (m_chkEccPolish) m_chkEccPolish->setChecked(s.value("eccPolish", true).toBool());
    if (m_chkPersistAlign) {
        QSignalBlocker b(m_chkPersistAlign);
        m_chkPersistAlign->setChecked(s.value("persistAlign", true).toBool());
    }
    if (!m_chkPersistAlign || m_chkPersistAlign->isChecked()) loadAlignmentMatrix();

    if (m_comboCamSet) {
        int cw = s.value("camW", 0).toInt();
        int ch = s.value("camH", 0).toInt();
        int cf = s.value("camFps", 0).toInt();
        if (cw > 0 && ch > 0 && cf > 0) {
            fitToSensor(cw, ch, cf);
            int found = -1;
            for (int i = 0; i < m_comboCamSet->count(); ++i) {
                QVariantList v = m_comboCamSet->itemData(i).toList();
                if (v.size() == 3 && v[0].toInt() == cw
                    && v[1].toInt() == ch && v[2].toInt() == cf) {
                    found = i;
                    break;
                }
            }
            if (found < 0 && m_comboCamSet->count() > 0) {
                const int customIdx = m_comboCamSet->count() - 1;
                m_comboCamSet->setItemText(customIdx,
                    QString("%1x%2 @ %3 FPS ★").arg(cw).arg(ch).arg(cf));
                m_comboCamSet->setItemData(customIdx, QVariantList{cw, ch, cf});
                found = customIdx;
            }
            if (found >= 0) m_comboCamSet->setCurrentIndex(found);
        }
    }

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
    s.setValue("clipLevel", m_clipLevel);
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
    if (m_spnClipLevel) m_spnClipLevel->setValue(s.value("clipLevel", m_clipLevel).toInt());
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
            "QDialog { background: %1; border: 1px solid %2; border-radius: 8px; }")
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
        {"cmd_stream", "Start / Stop Streams", "Capture", CmdType::Action, [this](){
            cameraTransition([this]() { if (m_camerasOpen) closeCameras(); else openCameras(); });
        }, {}},
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
        {"cmd_calibrate_k", "Calibrate Channel Balance k", "Pipeline", CmdType::Action, [this](){ calibrateChannelBalance(); }, {}},
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

        {"cmd_history", "Chart History Limit", "Options", CmdType::Parameter, {}, [this, createPopup](QWidget*){ createPopup("Chart History Limit", m_historySlider); }},
        {"cmd_resources", "Toggle Resource Monitor", "Options", CmdType::Toggle, [this](){ toggleResourceHud(); }, {}},

        {"cmd_tour", "Show Quick Tour", "Help", CmdType::Action, [this](){ showQuickTour(); }, {}},
        {"cmd_help", "Open Help & Tips", "Help", CmdType::Action, [this](){ showHelpDialog(); }, {}}
    };

    QSettings s(settingsPath(), QSettings::IniFormat);
    s.beginGroup("Hotkeys");
    for (const auto& cmd : m_commands) {
        QString keyStr = s.value(cmd.id, "").toString();
        if (!keyStr.isEmpty()) {
            applyShortcut(cmd.id, QKeySequence(keyStr));
        }
    }
    const bool minimizeConfigured = s.contains("cmd_minimize_dialogs");
    s.endGroup();

    if (!minimizeConfigured) {
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
    s.setValue(id, seq.toString());
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

        setStyleSheet(QString(
            "QPushButton { color: %1; }"
            "QPushButton:checked { background: %2; color: %3; border: 1px solid %2; font-weight: 600; }")
            .arg(T::text).arg(T::accent).arg(T::onAccent));

        connect(this, &QPushButton::toggled, this, [this](bool on) {
            if (on) setText("Press a key...");
            else setText(m_seq.isEmpty() ? "None"
                                         : m_seq.toString(QKeySequence::NativeText));
        });
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
            setText("Press a key...");
        } else {
            QPushButton::keyReleaseEvent(e);
        }
    }

private:
    QKeySequence m_seq;
};

class TourOverlay : public QWidget {
public:
    struct Step { QWidget* target; QString title; QString text; };

    TourOverlay(QWidget* host, QVector<Step> steps, std::function<void()> onDone)
        : QWidget(host), m_steps(std::move(steps)), m_onDone(std::move(onDone))
    {
        setAttribute(Qt::WA_DeleteOnClose);
        setFocusPolicy(Qt::StrongFocus);

        m_card = new QFrame(this);
        m_card->setStyleSheet(QString(
            "QFrame { background: %1; border: 1px solid %2; border-radius: 10px; }"
            "QLabel { border: none; background: transparent; }")
            .arg(T::bg2).arg(T::borderStrong));
        QVBoxLayout* cl = new QVBoxLayout(m_card);
        cl->setContentsMargins(16, 14, 16, 12);
        cl->setSpacing(6);

        m_stepLbl = new QLabel(m_card);
        m_stepLbl->setStyleSheet(QString(
            "color:%1; font-size:10px; font-weight:700; letter-spacing:0.10em;"
            " font-family:'Space Mono',monospace;").arg(T::accent));
        m_title = new QLabel(m_card);
        m_title->setStyleSheet(QString("color:%1; font-size:14px; font-weight:600;").arg(T::text));
        m_text = new QLabel(m_card);
        m_text->setWordWrap(true);
        m_text->setStyleSheet(QString("color:%1; font-size:12px;").arg(T::textDim));
        cl->addWidget(m_stepLbl);
        cl->addWidget(m_title);
        cl->addWidget(m_text);

        QHBoxLayout* row = new QHBoxLayout();
        row->setSpacing(8);
        m_btnSkip = new QPushButton("Skip tour", m_card);
        m_btnSkip->setStyleSheet(QString(
            "QPushButton { background: transparent; border: none; color: %1; padding: 4px 2px; }"
            "QPushButton:hover { color: %2; }").arg(T::textFaint).arg(T::text));
        m_btnBack = new QPushButton("Back", m_card);
        m_btnBack->setStyleSheet(QString(
            "QPushButton { background: %1; color: %2; border: 1px solid %3;"
            " border-radius: 6px; padding: 5px 14px; }")
            .arg(T::bg3).arg(T::text).arg(T::borderStrong));
        m_btnNext = new QPushButton("Next", m_card);
        m_btnNext->setStyleSheet(QString(
            "QPushButton { background: %1; color: %2; border: 1px solid %1;"
            " border-radius: 6px; padding: 5px 16px; font-weight: 600; }"
            "QPushButton:hover { background: #e0695d; border-color: #e0695d; }")
            .arg(T::accent).arg(T::onAccent));
        row->addWidget(m_btnSkip);
        row->addStretch();
        row->addWidget(m_btnBack);
        row->addWidget(m_btnNext);
        cl->addLayout(row);

        connect(m_btnSkip, &QPushButton::clicked, this, [this]() { finish(); });
        connect(m_btnBack, &QPushButton::clicked, this, [this]() { setStep(m_idx - 1); });
        connect(m_btnNext, &QPushButton::clicked, this, [this]() { advance(); });

        if (host) {
            host->installEventFilter(this);
            setGeometry(host->rect());
        }
        setStep(0);
        show();
        raise();
        setFocus();
    }

protected:
    bool eventFilter(QObject* o, QEvent* e) override {
        if (o == parentWidget()
            && (e->type() == QEvent::Resize || e->type() == QEvent::LayoutRequest)) {
            setGeometry(parentWidget()->rect());
            layoutCard();
            update();
        }
        return QWidget::eventFilter(o, e);
    }
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        QPainterPath dim;
        dim.setFillRule(Qt::OddEvenFill);
        dim.addRect(rect());
        const QRect hole = targetRect();
        if (!hole.isNull()) dim.addRoundedRect(hole, 10, 10);
        p.fillPath(dim, QColor(5, 5, 5, 178));
        if (!hole.isNull()) {
            QPen pen{ QColor(T::accent) };
            pen.setWidth(2);
            p.setPen(pen);
            p.setBrush(Qt::NoBrush);
            p.drawRoundedRect(hole, 10, 10);
        }
    }
    void mousePressEvent(QMouseEvent*) override { advance(); }
    void keyPressEvent(QKeyEvent* e) override {
        switch (e->key()) {
        case Qt::Key_Escape: finish(); break;
        case Qt::Key_Left:   setStep(m_idx - 1); break;
        case Qt::Key_Right:
        case Qt::Key_Space:
        case Qt::Key_Return:
        case Qt::Key_Enter:  advance(); break;
        default: QWidget::keyPressEvent(e);
        }
    }

private:
    QRect targetRect() const {
        if (m_idx < 0 || m_idx >= m_steps.size()) return QRect();
        QWidget* t = m_steps[m_idx].target;
        if (!t || !t->isVisible() || !parentWidget()) return QRect();
        return QRect(t->mapTo(parentWidget(), QPoint(0, 0)), t->size())
               .adjusted(-4, -4, 4, 4);
    }
    void advance() {
        if (m_idx + 1 >= m_steps.size()) finish();
        else setStep(m_idx + 1);
    }
    void setStep(int i) {
        m_idx = qBound(0, i, int(m_steps.size()) - 1);
        const Step& s = m_steps[m_idx];
        m_stepLbl->setText(QString("STEP %1 / %2").arg(m_idx + 1).arg(m_steps.size()));
        m_title->setText(s.title);
        m_text->setText(s.text);
        m_btnBack->setVisible(m_idx > 0);
        m_btnNext->setText(m_idx + 1 >= m_steps.size() ? "Done" : "Next");
        layoutCard();
        update();
    }
    void layoutCard() {
        if (!m_card) return;
        m_card->setFixedWidth(qMin(380, width() - 32));
        m_card->adjustSize();
        const QRect hole = targetRect();
        QPoint pos;
        if (hole.isNull()) {
            pos = QPoint((width() - m_card->width()) / 2,
                         (height() - m_card->height()) / 2);
        } else {
            int x = qBound(16, hole.center().x() - m_card->width() / 2,
                           qMax(16, width() - m_card->width() - 16));
            int y = hole.bottom() + 14;
            if (y + m_card->height() > height() - 16)
                y = hole.top() - m_card->height() - 14;
            y = qMax(16, y);
            pos = QPoint(x, y);
        }
        m_card->move(pos);
    }
    void finish() {
        if (m_onDone) m_onDone();
        close();
    }

    QVector<Step> m_steps;
    std::function<void()> m_onDone;
    QFrame* m_card = nullptr;
    QLabel* m_stepLbl = nullptr;
    QLabel* m_title = nullptr;
    QLabel* m_text = nullptr;
    QPushButton* m_btnSkip = nullptr;
    QPushButton* m_btnBack = nullptr;
    QPushButton* m_btnNext = nullptr;
    int m_idx = 0;
};

struct HelpTopic { QString title; QString text; };

static QVector<HelpTopic> helpTopics()
{
    return {
        { "Workspaces & navigation",
          "The left rail switches workspaces: Capture, Pipeline, Diff, Focus, "
          "Snapshot and Settings. Each workspace opens its own control panel at "
          "the bottom; clicking the active item again hides the panel. \"G\" "
          "opens the snapshot gallery, \"<<\" collapses the rail to icons, "
          "\"[]\" hides all panels for a cameras-only view (a small button in "
          "the top-left corner brings everything back)." },
        { "Streaming & preview",
          "▶ starts and stops both camera streams; the resolution / FPS preset "
          "is chosen in the top-left combo box (including custom modes). The "
          "previews show CAM 1 and CAM 2 side by side — drag the divider "
          "between them to resize. DUAL / DIFF switches between side-by-side "
          "view and the difference image." },
        { "Exposure",
          "AUTO / MAN toggles automatic and manual exposure. In manual mode the "
          "current gain × shutter is shown next to the toggle, and "
          "\"Detail...\" opens per-camera gain and shutter controls with "
          "sliders and exact values. Manual exposure drives libcamera (CSI) "
          "cameras; USB / DirectShow cameras keep their own exposure." },
        { "Capture workspace",
          "Color mode (grayscale / color), flipping of CAM 2 (horizontal / "
          "vertical), bilateral noise filter with strength slider, and adaptive "
          "view options." },
        { "Pipeline & Diff",
          "Pipeline contains the processing chain: the T-buffer (temporal "
          "averaging over N frames), the motion threshold with its indicator "
          "(motion restarts the averaging), the sum (CAM1 + k·CAM2)/2 and the "
          "alignment controls. Diff shows D = CAM1 − k·CAM2 as a colour "
          "map with a scale in DN. k balances the two channels: Calibrate k on "
          "an unpolarized, evenly lit scene with the laser off and the exposure "
          "you will measure with; it takes the median of CAM1 / CAM2 over 20 "
          "frames, saves k at once with the date, spread and exposure, and "
          "keeps it until you calibrate again (presets do not touch it). Then "
          "point the rig at the laser. The noise floor zeroes small "
          "|D|, stretch multiplies |D| to fill the scale (the factor is shown), "
          "and pixels clipped in both cameras count as 0. The line under the "
          "controls lists every step applied. Peak tracking marks the "
          "brightest spot of each camera." },
        { "Focus view",
          "Focus shows a live sharpness metric for both cameras as a chart plus "
          "large per-camera focus values — useful for fine lens adjustment. "
          "The history length is configurable." },
        { "Alignment",
          "The NO ALIGN pill in the top bar shows the alignment state. "
          "Calibrate (Pipeline) solves the CAM2 -> CAM1 warp from image "
          "features, refines it to sub-pixel accuracy and, with ECC polish "
          "on, by ECC; Apply to CAM2 uses it on every frame and Remember keeps "
          "it between sessions. Manual 6-DOF alignment (shift, scale, "
          "rotation) is available in its own dialog and is saved too." },
        { "Snapshots & gallery",
          "SNAP saves a snapshot from both cameras (side by side, or as two "
          "files with \"Separate files\"); in DIFF mode it saves the difference "
          "map. All settings are stored inside the image file, in its EXIF "
          "metadata.The filename is built from a template with parameters "
          "(exposure, filters, etc.) — configure them in the Snapshot "
          "workspace. The gallery (\"G\") lists recent snapshots; "
          "\"Snapshots\" in the top bar opens the folder in the OS." },
        { "Analysis viewers",
          "From a snapshot you can open analysis views: an intensity profile "
          "along a line (2D chart) and a 3D surface of a region's intensity "
          "with rotation, zoom, point picking and CSV export." },
        { "Resource monitor",
          "The CPU pill in the top bar shows machine CPU load and SoC "
          "temperature; it turns amber at 70 \u00B0C, high CPU or a frequency "
          "cap and red at 80 \u00B0C, throttling or under-voltage. Click it (or "
          "bind \"Toggle Resource Monitor\") for an overlay with per-core load, "
          "app / GUI / camera-thread CPU, ARM clock, fan, throttle flags, RAM "
          "and camera / shown / dropped FPS. While streaming, the STREAMING "
          "pill shows the camera FPS and turns amber when it falls behind the "
          "requested rate." },
        { "Presets & settings",
          "Presets save the whole processing configuration and can be loaded "
          "or deleted in one click. Settings contains animation speed and "
          "toggles. Camera mode, exposure and all options persist between "
          "sessions automatically." },
        { "Hotkeys",
          "Every action is listed in the HK's menu — run it directly with "
          "\"Exec\" or click the key box to bind a keyboard shortcut. "
          "Conflicts are detected automatically." },
    };
}

void MainWindow::showHelpDialog()
{
    QDialog dlg(this);
    dlg.setWindowTitle("Help & Tips");
    dlg.resize(560, 660);

    QVBoxLayout* root = new QVBoxLayout(&dlg);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    QLabel* title = new QLabel("Help & Tips", &dlg);
    title->setProperty("role", "title");
    title->setStyleSheet("padding: 16px 16px 4px 16px;");
    root->addWidget(title);

    QLabel* tip = new QLabel(
        "Overview of every part of DualCam. For a guided walkthrough of the "
        "interface, start the interactive tour.", &dlg);
    tip->setProperty("role", "faint");
    tip->setWordWrap(true);
    tip->setStyleSheet("padding: 0px 16px 12px 16px;");
    root->addWidget(tip);

    QScrollArea* scroll = new QScrollArea(&dlg);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setStyleSheet("QScrollArea { background: transparent; }"
                          "QWidget#helpScrollContent { background: transparent; }");

    QWidget* content = new QWidget();
    content->setObjectName("helpScrollContent");
    QVBoxLayout* list = new QVBoxLayout(content);
    list->setContentsMargins(16, 0, 16, 16);
    list->setSpacing(8);

    for (const HelpTopic& t : helpTopics()) {
        QFrame* card = new QFrame();
        card->setProperty("role", "panel");
        QVBoxLayout* cl = new QVBoxLayout(card);
        cl->setContentsMargins(14, 10, 14, 12);
        cl->setSpacing(4);
        QLabel* h = new QLabel(t.title, card);
        h->setStyleSheet(QString(
            "color:%1; font-size:12px; font-weight:600; border:none; background:transparent;")
            .arg(T::accent));
        QLabel* b = new QLabel(t.text, card);
        b->setWordWrap(true);
        b->setStyleSheet(QString(
            "color:%1; font-size:12px; border:none; background:transparent;")
            .arg(T::textDim));
        cl->addWidget(h);
        cl->addWidget(b);
        list->addWidget(card);
    }
    list->addStretch();

    scroll->setWidget(content);
    root->addWidget(scroll, 1);

    QFrame* bottomBar = new QFrame(&dlg);
    bottomBar->setObjectName("helpBottomBar");
    bottomBar->setStyleSheet(QString(
        "QFrame#helpBottomBar { background: %1; border-top: 1px solid %2; }")
        .arg(T::bg2).arg(T::border));
    QHBoxLayout* btnLay = new QHBoxLayout(bottomBar);
    btnLay->setContentsMargins(16, 10, 16, 10);

    QPushButton* tourBtn = new QPushButton("Start interactive tour", bottomBar);
    tourBtn->setProperty("kind", "ghost");
    tourBtn->setCursor(Qt::PointingHandCursor);
    connect(tourBtn, &QPushButton::clicked, &dlg, [this, &dlg]() {
        dlg.accept();
        QTimer::singleShot(0, this, [this]() { showQuickTour(); });
    });
    btnLay->addWidget(tourBtn);
    btnLay->addStretch();

    QPushButton* closeBtn = new QPushButton("Close", bottomBar);
    closeBtn->setProperty("kind", "primary");
    closeBtn->setCursor(Qt::PointingHandCursor);
    connect(closeBtn, &QPushButton::clicked, &dlg, &QDialog::accept);
    btnLay->addWidget(closeBtn);

    root->addWidget(bottomBar);

    execDialog(&dlg, m_btnHelpDocs);
}

void MainWindow::showQuickTour()
{
    QWidget* host = centralWidget();
    if (!host) return;

    QVector<TourOverlay::Step> steps;
    auto add = [&steps](QWidget* t, const QString& title, const QString& text) {
        if (t && t->isVisible()) steps.append({ t, title, text });
    };

    QWidget* topStrip = m_fpsPill ? m_fpsPill->parentWidget() : nullptr;
    add(m_sideNav, "Workspaces",
        "Capture, Pipeline, Diff, Focus, Snapshot and Settings — each opens its "
        "own control panel at the bottom. \"G\" is the snapshot gallery. "
        "\"<<\" collapses this rail, \"[]\" hides all panels (cameras-only view).");
    add(topStrip, "Camera & status strip",
        "Left side: resolution / FPS, AUTO-MAN exposure toggle and detailed "
        "exposure settings. Right side: streaming status, alignment state and "
        "recent snapshots.");
    add(m_videoArea, "Live preview",
        "Both cameras side by side. ▶ starts and stops the streams, "
        "DUAL / DIFF switches to the difference view, SNAP saves a snapshot "
        "from both cameras.");
    add(m_bottomStack, "Control panel",
        "Settings for the selected workspace: color pipeline, filters, motion "
        "detection, focus metrics and more.");
    add(m_btnHelp, "HK's — Actions & Hotkeys",
        "Every action in the app is listed here — run it directly or assign a "
        "keyboard shortcut of your choice.");
    add(m_btnHelpDocs, "Help & Tips",
        "Detailed notes on every part of the app, and a way to restart this "
        "tour at any time.");

    if (steps.isEmpty()) return;
    new TourOverlay(host, steps, []() {
        QSettings s(settingsPath(), QSettings::IniFormat);
        s.setValue("tourShown", true);
    });
}

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
    bottomBar->setObjectName("hotkeysBottomBar");

    bottomBar->setStyleSheet(QString(
        "QFrame#hotkeysBottomBar { background: %1; border-top: 1px solid %2; }")
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

    execDialog(&dlg, m_btnHelp);
}

#include "mainwindow.moc"
