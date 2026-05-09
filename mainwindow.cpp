#include "mainwindow.h"

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
#include <QCloseEvent>
#include <QJsonObject>
#include <QJsonDocument>
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

#include <iostream>
#include <vector>
#include <cmath>

#include <QPainter>
#include <QPaintEvent>

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

void GpuImageView::resizeGL(int /*w*/, int /*h*/) {}

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
            p.setPen(QColor("#8a94a3"));
            p.drawText(rect(), Qt::AlignCenter, m_placeholder);
        }
        return;
    }

    // Aspect-fit
    QSize is = m_image.size();
    QSize ws = size();
    double k = std::min(static_cast<double>(ws.width()) / is.width(),
                        static_cast<double>(ws.height()) / is.height());
    int dw = static_cast<int>(is.width() * k);
    int dh = static_cast<int>(is.height() * k);
    int dx = (ws.width() - dw) / 2;
    int dy = (ws.height() - dh) / 2;
    p.setRenderHint(QPainter::SmoothPixmapTransform, false);
    p.drawImage(QRect(dx, dy, dw, dh), m_image);

    if (!m_overlayText.isEmpty()) {
        QFont f = p.font();
        f.setFamily("Inter");
        f.setPointSize(10);
        f.setBold(true);
        p.setFont(f);
        QFontMetrics fm(f);
        QRect tr = fm.boundingRect(m_overlayText).adjusted(-8, -4, 8, 4);
        int x = m_overlayRight ? (width() - 10 - tr.width()) : 10;
        int y = 10;
        QRect bg(x, y, tr.width(), tr.height());
        p.fillRect(bg, QColor(14, 16, 19, 210));
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
    if (m_cap1.isOpened()) {
        m_cap1.set(cv::CAP_PROP_FRAME_WIDTH, w);
        m_cap1.set(cv::CAP_PROP_FRAME_HEIGHT, h);
        m_cap1.set(cv::CAP_PROP_FPS, fps);
    }
    if (m_cap2.isOpened()) {
        m_cap2.set(cv::CAP_PROP_FRAME_WIDTH, w);
        m_cap2.set(cv::CAP_PROP_FRAME_HEIGHT, h);
        m_cap2.set(cv::CAP_PROP_FPS, fps);
    }
    m_running = true;
    start();
}
void CameraWorker::stopCameras() {
    m_running = false;
    if (!wait(2000)) {
        requestInterruption();
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
double CameraWorker::detectMotion(const cv::Mat& frame, double thr) {
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
            // UI is overwhelmed — drop this frame to avoid event-queue blowup.
            continue;
        }
        m_pendingFrames.fetch_add(1);
        emit framesProcessed(f1.clone(), f2.clone(), focus1, focus2, motionDetected, m_frameCount);
    }
}

namespace T {
    static const char* bg0 = "#0e1013";
    static const char* bg1 = "#14171c";
    static const char* bg2 = "#1a1e24";
    static const char* bg3 = "#232830";
    static const char* bgHover = "#2c323c";
    static const char* border = "#272c34";
    static const char* borderStrong = "#353c47";
    static const char* text = "#d8dde4";
    static const char* textDim = "#8a94a3";
    static const char* textFaint = "#5c6572";
    static const char* accent = "#6aa9ff";
    static const char* accentDim = "#2c4a7a";
    static const char* ok = "#4fd18b";
    static const char* warn = "#f0b429";
    static const char* err = "#ff6a6a";
    static const char* cam1 = "#4ec9b0";
    static const char* cam2 = "#ce9178";
}

static QString pillStyle(const char* fg, const char* bg, const char* border)
{
    return QString(
        "QLabel { color: %1; background: %2; border: 1px solid %3;"
        " border-radius: 10px; padding: 3px 10px; font-family: 'Inter','Segoe UI',sans-serif;"
        " font-size: 10px; font-weight: 600; letter-spacing: 0.5px; }")
        .arg(fg).arg(bg).arg(border);
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
    {"ECC", "ECC Align (ECC)"},
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
        QMainWindow, QWidget#root, QDialog { background: %1; color: %3; }
        QDialog QLabel { color: %3; background: transparent; }
        QWidget { color: %3; font-family: 'Inter','Segoe UI',sans-serif; font-size: 13px; }
        QLabel { color: %3; }
        QLabel[role="dim"] { color: %4; font-size: 11px; }
        QLabel[role="title"] { color: %3; font-family: 'Inter','Segoe UI',sans-serif; font-size: 12px; }

        QFrame[role="card"] { background: %6; border: 1px solid %7; border-radius: 5px; }
        QFrame[role="panel"] { background: %2; border: 1px solid %7; border-radius: 4px; }

        QTabWidget::pane { border: none; background: %2; }
        QTabBar { background: %2; qproperty-drawBase: 0; }
        QTabBar::tab {
            background: transparent; color: %4; padding: 10px 0px;
            font-family: 'Inter','Segoe UI',sans-serif; font-size: 12px; font-weight: 600;
            letter-spacing: 0.3px; text-transform: uppercase;
            border-bottom: 2px solid transparent;
        }
        QTabBar::tab:selected { color: %5; border-bottom: 2px solid %5; }
        QTabBar::tab:hover:!selected { color: %3; }

        QPushButton {
            background: %6; color: %3; border: 1px solid %7; border-radius: 4px;
            padding: 7px 14px; font-size: 12px;
        }
        QPushButton:hover { background: %8; border-color: %9; }
        QPushButton:pressed { background: %7; }
        QPushButton:disabled { color: %10; background: %2; }
        QPushButton:checked { background: %11; border-color: %5; color: %5; }

        QPushButton[kind="primary"] {
            background: %5; color: #0a1220; border: 1px solid %5;
            font-weight: 600; padding: 9px 18px;
        }
        QPushButton[kind="primary"]:hover { background: #86bcff; }
        QPushButton[kind="primary"]:pressed { background: #4d8ce0; }
        QPushButton[kind="primary"]:disabled { background: %11; color: %4; }
        QPushButton[kind="ghost"] {
            background: transparent; border: 1px solid transparent; color: %4;
        }
        QPushButton[kind="ghost"]:hover { color: %3; background: %6; }

        QPushButton#sheetHandle {
            background: %2; border: none; border-top: 1px solid %7; border-bottom: 1px solid %7;
            color: %4; padding: 0; min-height: 36px; max-height: 36px;
        }
        QPushButton#sheetHandle:hover { background: %6; color: %3; }

        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background: %8; color: %3; border: 1px solid %7; border-radius: 4px;
            padding: 6px 8px; selection-background-color: %11;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: %5; }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background: %2; color: %3; border: 1px solid %7; selection-background-color: %11;
        }

        QCheckBox { color: %3; spacing: 8px; }
        QCheckBox::indicator {
            width: 18px; height: 18px; border-radius: 4px; background: %8; border: 1px solid %7;
        }
        QCheckBox::indicator:hover { border-color: %9; }
        QCheckBox::indicator:checked { background: %5; border-color: %5; }

        QSlider::groove:horizontal {
            background: %8; height: 4px; border-radius: 2px;
        }
        QSlider::sub-page:horizontal { background: %5; border-radius: 2px; }
        QSlider::handle:horizontal {
            background: %3; width: 14px; margin: -6px 0; border-radius: 7px;
        }
        QSlider::handle:horizontal:hover { background: %5; }

        QListWidget {
            background: %2; color: %3; border: 1px solid %7; border-radius: 4px;
            outline: 0;
        }
        QListWidget::item {
            padding: 8px 10px; border-bottom: 1px solid %7;
            font-family: 'Inter','Segoe UI',sans-serif; font-size: 12px;
        }
        QListWidget::item:selected { background: %6; color: %5; }
        QListWidget::item:hover { background: %6; }

        QStatusBar { background: %2; color: %4; border-top: 1px solid %7; }
        QStatusBar::item { border: none; }

        QScrollBar:vertical { background: %2; width: 10px; }
        QScrollBar::handle:vertical { background: %7; border-radius: 5px; min-height: 30px; }
        QScrollBar::handle:vertical:hover { background: %9; }
        QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
        QScrollBar:horizontal { background: %2; height: 10px; }
        QScrollBar::handle:horizontal { background: %7; border-radius: 5px; min-width: 30px; }

        QToolTip {
            background: %6; color: %3; border: 1px solid %7; padding: 4px 8px;
            border-radius: 3px; font-size: 12px; font-weight: 600;
        }
    )")
        .arg(T::bg0)          // 1
        .arg(T::bg1)          // 2
        .arg(T::text)         // 3
        .arg(T::textDim)      // 4
        .arg(T::accent)       // 5
        .arg(T::bg2)          // 6
        .arg(T::border)       // 7
        .arg(T::bg3)          // 8
        .arg(T::borderStrong) // 9
        .arg(T::textFaint)    // 10
        .arg(T::accentDim)    // 11
        .arg(T::err);         // 12
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

    // ---------- Top strip ----------
    QWidget* topStrip = new QWidget(this);
    topStrip->setFixedHeight(48);
    topStrip->setStyleSheet(QString("background:%1; border-bottom:1px solid %2;")
                                .arg(T::bg1).arg(T::border));
    QHBoxLayout* topLay = new QHBoxLayout(topStrip);
    topLay->setContentsMargins(12, 0, 10, 0);
    topLay->setSpacing(10);

    m_comboCamSet = new QComboBox(this);
    m_comboCamSet->setToolTip("Camera Resolution & FPS (Fetches from system)");
    m_comboCamSet->setStyleSheet(QString(
        "QComboBox { background: %1; color: %2; border: 1px solid %3; border-radius: 4px; padding: 4px 10px; font-weight: 700; font-size: 13px; }"
        "QComboBox:disabled { background: %5; color: %6; border-color: %5; }"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView { background: %1; color: %2; selection-background-color: %4; }")
        .arg(T::bg2).arg(T::text).arg(T::border).arg(T::accentDim)
        .arg(T::bg0).arg(T::textFaint));
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
            
            if (dlg.exec() == QDialog::Accepted) {
                QString label = QString("%1x%2 @ %3 FPS \u2605").arg(spinW->value()).arg(spinH->value()).arg(spinF->value());
                m_comboCamSet->setItemText(index, label);
                m_comboCamSet->setItemData(index, QVariantList{spinW->value(), spinH->value(), spinF->value()});
            }
        }
    });

    m_fpsPill = new QLabel("IDLE", this);
    m_fpsPill->setStyleSheet(pillStyle(T::textDim, T::bg2, T::border));
    m_eccPill = new QLabel("ECC NONE", this);
    m_eccPill->setStyleSheet(pillStyle(T::err, "#2a1414", T::border));

    QString topBtnSty = QString(
        "QPushButton { background: %1; color: %2; border: 1px solid %3;"
        " border-radius: 4px; padding: 6px 12px; font-size: 12px; font-weight: 500; }"
        "QPushButton:hover { background: %4; border-color: %5; color: %6; }")
        .arg(T::bg2).arg(T::text).arg(T::border)
        .arg(T::bg3).arg(T::accent).arg(T::accent);

    m_btnGallery = new QPushButton("\u25A6  Snapshots", this);
    m_btnGallery->setStyleSheet(topBtnSty);
    m_btnGallery->setCursor(Qt::PointingHandCursor);
    m_btnGallery->setToolTip("Open snapshots folder");
    connect(m_btnGallery, &QPushButton::clicked, this, [this]() {
        QString dir = QCoreApplication::applicationDirPath() + "/metrics";
        QDir().mkpath(dir);
        QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
        m_statusBar->showMessage("Snapshots folder: " + dir, 4000);
    });

    m_btnHelp = new QPushButton("?", this);
    m_btnHelp->setStyleSheet(topBtnSty);
    m_btnHelp->setFixedWidth(34);
    m_btnHelp->setCursor(Qt::PointingHandCursor);
    m_btnHelp->setToolTip("Actions & Hotkeys Menu");
    connect(m_btnHelp, &QPushButton::clicked, this, &MainWindow::showActionsMenu);

    topLay->addWidget(m_comboCamSet);

    // ---------- Exposure controls (Auto/Manual + Gain + Shutter) ----------
    m_btnExpToggle = new QPushButton("AUTO", this);
    m_btnExpToggle->setCheckable(true);
    m_btnExpToggle->setCursor(Qt::PointingHandCursor);
    m_btnExpToggle->setToolTip("Toggle automatic / manual exposure");
    m_btnExpToggle->setFixedWidth(70);
    m_btnExpToggle->setStyleSheet(QString(
        "QPushButton { background: %1; color: %2; border: 1px solid %3; border-radius: 4px;"
        " padding: 4px 8px; font-weight: 700; font-size: 11px; letter-spacing: 0.5px; }"
        "QPushButton:checked { background: %4; color: #0a1220; border-color: %4; }")
        .arg(T::bg2).arg(T::text).arg(T::border).arg(T::accent));

    // Hidden controls (live in the popup dialog opened by Change). Parented to
    // MainWindow so they survive popup close/reopen and remain hotkey targets.
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

    // Inline values pill — shows current gain/shutter
    m_lblExposureVals = new QLabel(this);
    m_lblExposureVals->setStyleSheet(QString(
        "QLabel { background: %1; color: %2; border: 1px solid %3; border-radius: 4px;"
        " padding: 4px 10px; font-family: 'Inter','Segoe UI',monospace; font-size: 11px; }")
        .arg(T::bg2).arg(T::text).arg(T::border));
    m_lblExposureVals->setToolTip("Current gain * shutter (manual mode)");
    m_lblExposureVals->setMinimumWidth(220);
    m_lblExposureVals->setAlignment(Qt::AlignCenter);

    m_btnExpChange = new QPushButton("Change", this);
    m_btnExpChange->setStyleSheet(topBtnSty);
    m_btnExpChange->setCursor(Qt::PointingHandCursor);
    m_btnExpChange->setToolTip("Adjust gain and shutter (manual mode only)");
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

    // Wire each pair (spin <-> slider) and store updated value into the right backing field.
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

    topLay->addSpacing(6);
    topLay->addWidget(m_btnExpToggle);
    topLay->addWidget(m_lblExposureVals);
    topLay->addWidget(m_btnExpChange);

    topLay->addStretch();
    topLay->addWidget(m_fpsPill);
    topLay->addWidget(m_eccPill);
    topLay->addSpacing(6);
    topLay->addWidget(m_btnGallery);
    topLay->addWidget(m_btnHelp);

    rootLayout->addWidget(topStrip);

    // ---------- Video area ----------
    m_videoArea = new QWidget(this);
    m_videoArea->setStyleSheet(QString("background:%1;").arg(T::bg0));
    QVBoxLayout* vidLay = new QVBoxLayout(m_videoArea);
    vidLay->setContentsMargins(6, 6, 6, 6);
    vidLay->setSpacing(4);

    // -- Focus toggle (floating) --
    m_btnFocusToggle = new QPushButton("Focus", m_videoArea);
    m_btnFocusToggle->setCheckable(true);
    m_btnFocusToggle->setCursor(Qt::PointingHandCursor);
    m_btnFocusToggle->setFixedSize(80, 28);
    m_btnFocusToggle->setStyleSheet(QString(
        "QPushButton { background: %1; color: %2; border: 1px solid %3;"
        " border-radius: 14px; font-weight: 600; font-size: 12px; }"
        "QPushButton:hover { background: %4; color: %5; }"
        "QPushButton:checked { background: %6; color: #0a1220; }")
        .arg(T::bg3).arg(T::textDim).arg(T::border)
        .arg(T::bg2).arg(T::text).arg(T::accent));
    connect(m_btnFocusToggle, &QPushButton::clicked, this, &MainWindow::toggleFocusView);
    m_btnFocusToggle->hide(); // Will be shown and positioned by positionFloatingButtons

    m_splitter = new QSplitter(Qt::Horizontal, m_videoArea);
    m_splitter->setHandleWidth(4);

    auto makeView = [this](const QString& placeholder) {
        GpuImageView* v = new GpuImageView(this);
        v->setMinimumSize(160, 120);
        v->setPlaceholder(placeholder);
        return v;
    };
    m_view1 = makeView("Camera 1");
    m_view2 = makeView("Camera 2");
    m_osd1 = nullptr;
    m_osd2 = nullptr;
    m_splitter->addWidget(m_view1);
    m_splitter->addWidget(m_view2);

    m_resultView = makeView("Result View");
    m_osdResult = nullptr;
    m_resultView->hide();

    vidLay->addWidget(m_splitter, 1);
    vidLay->addWidget(m_resultView, 1);

    // -- Focus chart (built separately, shown on toggle) --
    buildFocusChart();
    m_chartView->setParent(m_videoArea);
    m_chartView->hide();
    vidLay->addWidget(m_chartView, 1);

    // -- Start/Stop, Mode Toggle, Snapshot (floating over video) --

    m_btnFabStream = new QPushButton(m_videoArea);
    m_btnFabStream->setText("");
    m_btnFabStream->setFixedSize(48, 48);
    m_btnFabStream->setCursor(Qt::PointingHandCursor);
    m_btnFabStream->setToolTip("Start streaming");
    m_btnFabStream->setStyleSheet(QString(
        "QPushButton { background: %1; border: 2px solid %2; border-radius: 24px; }"
        "QPushButton:hover { background: #3a5a8a; }")
        .arg(T::accentDim).arg(T::accent));
    {
        m_fabStreamIcon = new QLabel(QString::fromUtf8("\u25B6"), m_btnFabStream);
        m_fabStreamIcon->setAttribute(Qt::WA_TransparentForMouseEvents);
        m_fabStreamIcon->setAlignment(Qt::AlignCenter);
        m_fabStreamIcon->setStyleSheet(QString(
            "QLabel { background: transparent; color: %1; border: none;"
            " font-family: 'Segoe UI Symbol','Segoe UI Emoji','Arial Unicode MS';"
            " font-size: 20px; font-weight: 700; }").arg(T::accent));
        QVBoxLayout* fabLay = new QVBoxLayout(m_btnFabStream);
        fabLay->setContentsMargins(0, 0, 0, 0);
        fabLay->addWidget(m_fabStreamIcon, 0, Qt::AlignCenter);
    }
    connect(m_btnFabStream, &QPushButton::clicked, this, [this]() {
        if (m_camerasOpen) closeCameras();
        else openCameras();
    });
    m_btnToggleCameras = m_btnFabStream;

    m_btnModeToggle = new QPushButton("Dual", m_videoArea);
    m_btnModeToggle->setFixedHeight(32);
    m_btnModeToggle->setCursor(Qt::PointingHandCursor);
    m_btnModeToggle->setStyleSheet(QString(
        "QPushButton { background: %1; color: #0a1220; border: none;"
        " border-radius: 16px; padding: 4px 28px; font-weight: 600; font-size: 12px;"
        " min-width: 70px; }")
        .arg(T::accent));
    connect(m_btnModeToggle, &QPushButton::clicked, this, [this]() {
        setDiffMode(!m_isDiffMode);
    });

    m_btnFabSnapshot = new QPushButton(m_videoArea);
    m_btnFabSnapshot->setText("");
    m_btnFabSnapshot->setFixedSize(48, 48);
    m_btnFabSnapshot->setCursor(Qt::PointingHandCursor);
    m_btnFabSnapshot->setToolTip("Save snapshot");
    m_btnFabSnapshot->setStyleSheet(QString(
        "QPushButton { background: %1; border: 2px solid #0a1220; border-radius: 24px; }"
        "QPushButton:hover { background: #86bcff; }"
        "QPushButton:pressed { background: #4d8ce0; }")
        .arg(T::accent));
    {
        m_fabSnapIcon = new QLabel("snap", m_btnFabSnapshot);
        m_fabSnapIcon->setAttribute(Qt::WA_TransparentForMouseEvents);
        m_fabSnapIcon->setAlignment(Qt::AlignCenter);
        m_fabSnapIcon->setStyleSheet(
            "QLabel { background: transparent; color: #000000; border: none;"
            " font-family: 'Inter','Segoe UI',sans-serif;"
            " font-size: 12px; font-weight: 900; letter-spacing: 0.5px; }");
        QVBoxLayout* snapLay = new QVBoxLayout(m_btnFabSnapshot);
        snapLay->setContentsMargins(0, 0, 0, 0);
        snapLay->addWidget(m_fabSnapIcon, 0, Qt::AlignCenter);
    }
    connect(m_btnFabSnapshot, &QPushButton::clicked, this, [this]() {
        saveSnapshot();
        refreshSnapshotPreview();
    });

    m_btnFabStream->hide();
    m_btnModeToggle->hide();
    m_btnFabSnapshot->hide();
    
    m_videoArea->installEventFilter(this);
    // They will be shown and positioned via resize events on m_videoArea

    rootLayout->addWidget(m_videoArea, 3);

    // ---------- Wide collapse handle ----------
    m_btnSheetHandle = new QPushButton(this);
    m_btnSheetHandle->setObjectName("sheetHandle");
    m_btnSheetHandle->setCursor(Qt::PointingHandCursor);
    m_btnSheetHandle->setToolTip("Collapse / expand controls");

    QHBoxLayout* handleInner = new QHBoxLayout(m_btnSheetHandle);
    handleInner->setContentsMargins(0, 0, 0, 0);
    handleInner->setSpacing(8);
    QLabel* grip = new QLabel(m_btnSheetHandle);
    grip->setFixedSize(56, 4);
    grip->setStyleSheet(QString("background:%1; border-radius:2px;").arg(T::borderStrong));
    QLabel* chev = new QLabel(QString::fromUtf8("\u25BC"), m_btnSheetHandle);
    chev->setStyleSheet(QString("color:%1; font-size:10px;").arg(T::textDim));
    chev->setObjectName("sheetChevron");
    handleInner->addStretch();
    handleInner->addWidget(grip, 0, Qt::AlignCenter);
    handleInner->addWidget(chev, 0, Qt::AlignCenter);
    handleInner->addStretch();
    m_btnSheetHandle->installEventFilter(this);

    rootLayout->addWidget(m_btnSheetHandle);

    // ---------- Bottom sheet (tabs) ----------
    m_sheetWidget = new QWidget(this);
    m_sheetWidget->setMinimumHeight(140);
    m_sheetWidget->setMaximumHeight(140);
    m_sheetWidget->setStyleSheet(QString("background:%1;").arg(T::bg1));

    QVBoxLayout* sheetLay = new QVBoxLayout(m_sheetWidget);
    sheetLay->setContentsMargins(0, 0, 0, 0);
    sheetLay->setSpacing(0);

    m_sheetStack = new QStackedWidget(this);

    // Page 0: Normal tabs
    m_tabWidget = new QTabWidget(this);
    m_tabWidget->addTab(buildCaptureTab(),  "Capture");
    m_tabWidget->addTab(buildPipelineTab(), "Pipeline");
    m_tabWidget->addTab(buildDiffTab(),     "Diff");
    m_tabWidget->addTab(buildSnapshotTab(), "Snapshot");
    m_tabWidget->addTab(buildPresetsTab(),  "Presets");
    m_tabWidget->setUsesScrollButtons(false);
    m_tabWidget->setDocumentMode(true);
    m_tabWidget->tabBar()->setExpanding(true);
    m_tabWidget->tabBar()->setDocumentMode(true);
    m_tabWidget->tabBar()->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    QTimer::singleShot(0, this, [this]() {
        m_tabWidget->tabBar()->updateGeometry();
        m_tabWidget->update();
    });

    // Page 1: Focus data panel
    m_focusDataWidget = buildFocusDataPanel();

    m_sheetStack->addWidget(m_tabWidget);
    m_sheetStack->addWidget(m_focusDataWidget);
    m_sheetStack->setCurrentIndex(0);

    sheetLay->addWidget(m_sheetStack);
    rootLayout->addWidget(m_sheetWidget, 0);

    // Dialog for manual 6-DOF align
    buildAlignDialog();

    m_statusBar = new QStatusBar(this);
    setStatusBar(m_statusBar);
    m_statusBar->showMessage("Ready. Press \u25B6 to open cameras.");

    QTimer::singleShot(0, this, [this]() {
        positionFloatingButtons();
    });
}

// ---------- Capture tab ----------
QWidget* MainWindow::buildCaptureTab()
{
    QWidget* page = new QWidget(this);
    QGridLayout* grid = new QGridLayout(page);
    grid->setContentsMargins(14, 14, 14, 14);
    grid->setHorizontalSpacing(18);
    grid->setVerticalSpacing(12);

    auto sectionLabel = [this](const QString& t) {
        QLabel* l = new QLabel(t, this);
        l->setStyleSheet(QString(
            "color:%1; font-family:'Inter','Segoe UI',sans-serif;"
            "font-size:10px; letter-spacing:0.5px;").arg(T::textDim));
        return l;
    };

    auto updateWorkerParams = [this]() {
        WorkerParams p;
        p.colorMode = m_colorMode;
        p.flipHor2 = m_chkFlipHor2->isChecked();
        p.flipVer2 = m_chkFlipVer2->isChecked();
        p.motionThr = m_motionThreshold;
        p.bufferSize = m_bufferSize;
        p.applyBilateral = m_chkBilateral->isChecked();
        p.bilateralStrength = m_bilateralStrength;
        p.noiseFloor = m_noiseFloor;
        m_worker->setParams(p);
    };

    // Color mode
    grid->addWidget(sectionLabel("COLOR"), 0, 0);
    m_comboColorMode = new QComboBox(this);
    m_comboColorMode->addItems({ "Gray Native", "Gray CV", "Color" });
    m_comboColorMode->setCurrentIndex(1);
    connect(m_comboColorMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, updateWorkerParams](int i) {
        m_colorMode = static_cast<ColorMode>(i);
        updateWorkerParams();
    });
    grid->addWidget(m_comboColorMode, 1, 0);

    // Flip CAM2
    grid->addWidget(sectionLabel("FLIP CAM2"), 0, 1);
    QWidget* flipBox = new QWidget(this);
    QHBoxLayout* flipLay = new QHBoxLayout(flipBox);
    flipLay->setContentsMargins(0, 0, 0, 0);
    m_chkFlipHor2 = new QCheckBox("Horizontal", this);
    m_chkFlipVer2 = new QCheckBox("Vertical", this);
    connect(m_chkFlipHor2, &QCheckBox::stateChanged, updateWorkerParams);
    connect(m_chkFlipVer2, &QCheckBox::stateChanged, updateWorkerParams);
    flipLay->addWidget(m_chkFlipHor2);
    flipLay->addWidget(m_chkFlipVer2);
    flipLay->addStretch();
    grid->addWidget(flipBox, 1, 1);

    // Bilateral
    grid->addWidget(sectionLabel("BILATERAL FILTER"), 0, 2);
    QWidget* bfBox = new QWidget(this);
    QHBoxLayout* bfLay = new QHBoxLayout(bfBox);
    bfLay->setContentsMargins(0, 0, 0, 0);
    bfLay->setSpacing(6);
    m_chkBilateral = new QCheckBox("On", this);
    connect(m_chkBilateral, &QCheckBox::stateChanged, updateWorkerParams);
    m_bilateralSlider = new QSlider(Qt::Horizontal, this);
    m_bilateralSlider->setRange(1, 20);
    m_bilateralSlider->setValue(m_bilateralStrength);
    m_bilateralSlider->setToolTip("Filter strength (radius & sigma)");
    m_bilateralLabel = new QLabel(QString::number(m_bilateralStrength), this);
    m_bilateralLabel->setFixedWidth(24);
    m_bilateralLabel->setAlignment(Qt::AlignCenter);
    connect(m_bilateralSlider, &QSlider::valueChanged, this, [this, updateWorkerParams](int v) {
        m_bilateralStrength = v;
        m_bilateralLabel->setText(QString::number(v));
        updateWorkerParams();
    });
    bfLay->addWidget(m_chkBilateral);
    bfLay->addWidget(m_bilateralSlider, 1);
    bfLay->addWidget(m_bilateralLabel);
    grid->addWidget(bfBox, 1, 2);

    grid->setColumnStretch(0, 1);
    grid->setColumnStretch(1, 1);
    grid->setColumnStretch(2, 1);
    grid->setRowStretch(4, 1);
    return page;
}

// ---------- Pipeline tab ----------
QWidget* MainWindow::buildPipelineTab()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* root = new QVBoxLayout(page);
    root->setContentsMargins(14, 14, 14, 14);
    root->setSpacing(12);

    auto updateWorkerParams = [this]() {
        WorkerParams p;
        p.colorMode = m_colorMode;
        p.flipHor2 = m_chkFlipHor2->isChecked();
        p.flipVer2 = m_chkFlipVer2->isChecked();
        p.motionThr = m_motionThreshold;
        p.bufferSize = m_bufferSize;
        p.applyBilateral = m_chkBilateral->isChecked();
        p.bilateralStrength = m_bilateralStrength;
        p.noiseFloor = m_noiseFloor;
        m_worker->setParams(p);
    };

    // Row 1: T-buffer, Motion threshold, Fusion
    QGridLayout* row1 = new QGridLayout();
    row1->setHorizontalSpacing(18);
    row1->setVerticalSpacing(6);

    auto sectionLabel = [this](const QString& t) {
        QLabel* l = new QLabel(t, this);
        l->setStyleSheet(QString(
            "color:%1; font-family:'Inter','Segoe UI',sans-serif;"
            "font-size:10px; letter-spacing:0.5px;").arg(T::textDim));
        return l;
    };

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
    connect(m_bufferSlider, &QSlider::valueChanged, this, [this, updateWorkerParams](int v) {
        m_bufferSize = v;
        m_bufferLabel->setText(QString::number(v));
        updateWorkerParams();
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
    connect(m_motionThresholdSlider, &QSlider::valueChanged, this, [this, updateWorkerParams](int v) {
        m_motionThreshold = static_cast<double>(v) / 100.0;
        m_motionThresholdLabel->setText(QString("%1%").arg(v));
        updateWorkerParams();
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

    // ECC card — two separate actions
    QFrame* eccCard = new QFrame(this);
    eccCard->setProperty("role", "card");
    QVBoxLayout* eccLay = new QVBoxLayout(eccCard);
    eccLay->setContentsMargins(12, 10, 12, 12);
    eccLay->setSpacing(10);

    QHBoxLayout* eccHeader = new QHBoxLayout();
    QLabel* eccTitle = new QLabel("ECC ALIGNMENT", this);
    eccTitle->setStyleSheet(QString(
        "color:%1; font-family:'Inter','Segoe UI',sans-serif;"
        "font-size:11px; letter-spacing:0.5px; font-weight:600;").arg(T::text));
    m_eccIndicator = new QLabel("NOT CALIBRATED", this);
    m_eccIndicator->setStyleSheet(pillStyle(T::err, "#2a1414", T::border));
    eccHeader->addWidget(eccTitle);
    eccHeader->addWidget(m_eccIndicator);
    eccHeader->addStretch();
    eccLay->addLayout(eccHeader);

    QHBoxLayout* eccButtons = new QHBoxLayout();
    eccButtons->setSpacing(8);
    m_btnCalibrateAlign = new QPushButton("Calibrate ECC", this);
    m_btnCalibrateAlign->setMinimumHeight(36);
    connect(m_btnCalibrateAlign, &QPushButton::clicked, this, &MainWindow::calibrateAlignment);

    m_chkAlign = new QCheckBox("Apply ECC to CAM2", this);
    m_chkAlign->setMinimumHeight(36);
    connect(m_chkAlign, &QCheckBox::stateChanged, this, &MainWindow::updateView);

    m_btnOpenManualAlign = new QPushButton("Manual 6-DOF…", this);
    m_btnOpenManualAlign->setMinimumHeight(36);
    connect(m_btnOpenManualAlign, &QPushButton::clicked, this, [this]() {
        if (m_alignDialog) {
            applyAdjustToWidgets();
            m_alignDialog->show();
            m_alignDialog->raise();
        }
    });

    eccButtons->addWidget(m_btnCalibrateAlign, 1);
    eccButtons->addWidget(m_chkAlign, 1);
    eccButtons->addWidget(m_btnOpenManualAlign, 1);
    eccLay->addLayout(eccButtons);

    QLabel* eccHint = new QLabel(
        "Calibrate runs findTransformECC once to solve the warp matrix. "
        "Apply reuses that matrix on every frame — toggle it off without re-calibrating.",
        this);
    eccHint->setWordWrap(true);
    eccHint->setStyleSheet(QString("color:%1; font-size:11px;").arg(T::textFaint));
    eccLay->addWidget(eccHint);

    root->addWidget(eccCard);
    root->addStretch();
    return page;
}

// ---------- Diff tab ----------
QWidget* MainWindow::buildDiffTab()
{
    QWidget* page = new QWidget(this);
    QGridLayout* grid = new QGridLayout(page);
    grid->setContentsMargins(14, 14, 14, 14);
    grid->setHorizontalSpacing(18);
    grid->setVerticalSpacing(12);

    auto updateWorkerParams = [this]() {
        WorkerParams p;
        p.colorMode = m_colorMode;
        p.flipHor2 = m_chkFlipHor2->isChecked();
        p.flipVer2 = m_chkFlipVer2->isChecked();
        p.motionThr = m_motionThreshold;
        p.bufferSize = m_bufferSize;
        p.applyBilateral = m_chkBilateral->isChecked();
        p.bilateralStrength = m_bilateralStrength;
        p.noiseFloor = m_noiseFloor;
        m_worker->setParams(p);
    };

    auto sectionLabel = [this](const QString& t) {
        QLabel* l = new QLabel(t, this);
        l->setStyleSheet(QString(
            "color:%1; font-family:'Inter','Segoe UI',sans-serif;"
            "font-size:10px; letter-spacing:0.5px;").arg(T::textDim));
        return l;
    };

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
    connect(m_noiseFloorSlider, &QSlider::valueChanged, this, [this, updateWorkerParams](int v) {
        m_noiseFloor = v;
        m_noiseFloorLabel->setText(QString::number(v));
        updateWorkerParams();
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
    m_lblPeakInfo->setStyleSheet(QString("color:%1; font-family:'Inter','Segoe UI',sans-serif;").arg(T::warn));
    grid->addWidget(m_lblPeakInfo, 2, 0, 1, 3);

    grid->setColumnStretch(0, 1);
    grid->setColumnStretch(1, 1);
    grid->setColumnStretch(2, 1);
    grid->setRowStretch(3, 1);
    return page;
}

// ---------- Focus chart (built in initUI, shown when Focus toggled) ----------
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
    axisY->setRange(0, 5000);
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

// ---------- Focus data panel (bottom sheet when Focus active) ----------
QWidget* MainWindow::buildFocusDataPanel()
{
    QWidget* panel = new QWidget(this);
    panel->setStyleSheet(QString("background: %1;").arg(T::bg1));
    QHBoxLayout* lay = new QHBoxLayout(panel);
    lay->setContentsMargins(14, 10, 14, 10);
    lay->setSpacing(8);

    auto makeCard = [this](const QString& title, const char* titleColor, QLabel*& valueLabel) {
        QFrame* card = new QFrame(this);
        card->setProperty("role", "card");
        QVBoxLayout* cardLay = new QVBoxLayout(card);
        cardLay->setContentsMargins(10, 8, 10, 8);
        cardLay->setSpacing(4);
        QLabel* t = new QLabel(title, card);
        t->setStyleSheet(QString(
            "color:%1; font-family:'Inter','Segoe UI',sans-serif;"
            "font-size:10px; letter-spacing:0.5px;").arg(titleColor));
        valueLabel = new QLabel("0", card);
        valueLabel->setStyleSheet(QString(
            "color:%1; font-family:'Inter','Segoe UI',sans-serif; font-size:22px;").arg(T::text));
        cardLay->addWidget(t);
        cardLay->addWidget(valueLabel);
        return card;
    };

    lay->addWidget(makeCard("CAM1 FOCUS", T::cam1, m_lblFocus1Big), 1);
    lay->addWidget(makeCard("CAM2 FOCUS", T::cam2, m_lblFocus2Big), 1);

    // Graph history card
    QFrame* histCard = new QFrame(this);
    histCard->setProperty("role", "card");
    QVBoxLayout* histLay = new QVBoxLayout(histCard);
    histLay->setContentsMargins(10, 8, 10, 8);
    histLay->setSpacing(6);
    QLabel* histTitle = new QLabel("GRAPH HISTORY", histCard);
    histTitle->setStyleSheet(QString(
        "color:%1; font-family:'Inter','Segoe UI',sans-serif;"
        "font-size:10px; letter-spacing:0.5px;").arg(T::textDim));
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
    return panel;
}

// ---------- Snapshot tab ----------
QWidget* MainWindow::buildSnapshotTab()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* root = new QVBoxLayout(page);
    root->setContentsMargins(14, 14, 14, 14);
    root->setSpacing(10);

    // Top row: name, save
    QFrame* saveCard = new QFrame(this);
    saveCard->setProperty("role", "card");
    QHBoxLayout* saveLay = new QHBoxLayout(saveCard);
    saveLay->setContentsMargins(10, 10, 10, 10);
    saveLay->setSpacing(10);

    QLabel* nameLabel = new QLabel("NAME", this);
    nameLabel->setStyleSheet(QString(
        "color:%1; font-family:'Inter','Segoe UI',sans-serif;"
        "font-size:10px; letter-spacing:0.5px;").arg(T::textDim));
    saveLay->addWidget(nameLabel);

    m_snapshotNameEdit = new QLineEdit(this);
    m_snapshotNameEdit->setPlaceholderText("Auto-generated...");
    saveLay->addWidget(m_snapshotNameEdit, 1);

    m_chkAppendParams = new QCheckBox("Append params", this);
    m_chkAppendParams->setToolTip("Add active filter values to filename (e.g. _BF5_NF15)");
    saveLay->addWidget(m_chkAppendParams);

    m_btnConfigParams = new QPushButton("Configure…", this);
    m_btnConfigParams->setProperty("kind", "ghost");
    m_btnConfigParams->setToolTip("Choose which parameters are written into the filename");
    m_btnConfigParams->setCursor(Qt::PointingHandCursor);
    connect(m_btnConfigParams, &QPushButton::clicked, this, &MainWindow::showFilenameParamsDialog);
    saveLay->addWidget(m_btnConfigParams);

    m_btnSaveSnapshot = new QPushButton("Save", this);
    m_btnSaveSnapshot->setProperty("kind", "primary");
    m_btnSaveSnapshot->setMinimumHeight(34);
    connect(m_btnSaveSnapshot, &QPushButton::clicked, this, [this]() {
        saveSnapshot();
        refreshSnapshotPreview();
    });
    saveLay->addWidget(m_btnSaveSnapshot);

    root->addWidget(saveCard);

    // Preview area - thumbnails of saved snapshots
    m_snapshotPreview = new QListWidget(this);
    m_snapshotPreview->setViewMode(QListWidget::IconMode);
    m_snapshotPreview->setIconSize(QSize(160, 120));
    m_snapshotPreview->setGridSize(QSize(180, 200));
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
                QString oldJson = oldPath; oldJson.replace(".png", ".json").replace(".jpg", ".json");
                QString newJson = newPath; newJson.replace(".png", ".json").replace(".jpg", ".json");
                QFile::rename(oldPath, newPath);
                QFile::rename(oldJson, newJson);
                refreshSnapshotPreview();
            }
        } else if (res == dltAct) {
            if (QMessageBox::question(this, "Delete", "Delete " + item->text() + "?") == QMessageBox::Yes) {
                QString oldJson = oldPath; oldJson.replace(".png", ".json").replace(".jpg", ".json");
                QFile::remove(oldPath);
                QFile::remove(oldJson);
                refreshSnapshotPreview();
            }
        }
    });

    connect(m_snapshotPreview, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* item) {
        if (!item) return;
        QString fileBase = item->text();
        QString imagePath = QCoreApplication::applicationDirPath() + "/metrics/" + fileBase;
        QString jsonPath = imagePath;
        jsonPath.replace(".png", ".json").replace(".jpg", ".json");

        QDialog dlg(this);
        dlg.setWindowTitle("Preview: " + fileBase);
        dlg.setStyleSheet(styleSheetText());
        dlg.resize(this->width() * 2 / 3, this->height() * 2 / 3);
        
        QHBoxLayout* lay = new QHBoxLayout(&dlg);
        QLabel* imgLabel = new QLabel(&dlg);
        QPixmap pix(imagePath);
        if (!pix.isNull()) {
            imgLabel->setPixmap(pix.scaled(dlg.width() * 2 / 3, dlg.height(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
        lay->addWidget(imgLabel, 3);
        
        QFrame* metaFrame = new QFrame(&dlg);
        metaFrame->setProperty("role", "card");
        QVBoxLayout* metaLay = new QVBoxLayout(metaFrame);
        QLabel* lblTitle = new QLabel("<b>METADATA</b>", &dlg);
        lblTitle->setStyleSheet("font-size: 14px; color: #6aa9ff;");
        metaLay->addWidget(lblTitle);
        
        QFile jf(jsonPath);
        if (jf.open(QIODevice::ReadOnly)) {
            QJsonDocument jdoc = QJsonDocument::fromJson(jf.readAll());
            QJsonObject jobj = jdoc.object();
            for (auto key : jobj.keys()) {
                QString val = jobj[key].isString() ? jobj[key].toString() : QString::number(jobj[key].toDouble());
                metaLay->addWidget(new QLabel(key + ": " + val, &dlg));
            }
        } else {
            metaLay->addWidget(new QLabel("No metadata found for this snapshot."));
        }
        metaLay->addStretch();
        lay->addWidget(metaFrame, 1);
        
        dlg.exec();
    });

    refreshSnapshotPreview();
    return page;
}

// ---------- Presets tab ----------
QWidget* MainWindow::buildPresetsTab()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* root = new QVBoxLayout(page);
    root->setContentsMargins(14, 14, 14, 14);
    root->setSpacing(10);

    // Save row
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
    root->addWidget(saveCard);

    // List with load/delete
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
    root->addLayout(listRow, 1);

    return page;
}

// ---------- Manual Align dialog (drawer-style) ----------
void MainWindow::buildAlignDialog()
{
    m_alignDialog = new QDialog(this);
    m_alignDialog->setWindowTitle("Manual Align (6 DOF)");
    m_alignDialog->setModal(false);
    m_alignDialog->resize(620, 460);
    m_alignDialog->setStyleSheet(styleSheetText() + QString(R"(
        QDialog QLabel { font-size: 15px; font-weight: 600; }
        QDialog QComboBox { font-size: 14px; font-weight: 600; padding: 6px 10px; min-height: 22px; }
        QDialog QDoubleSpinBox { font-size: 14px; font-weight: 600; padding: 6px 8px; min-height: 22px; }
        QDialog QPushButton { font-size: 14px; font-weight: 700; padding: 8px 18px; }
        QDialog QSlider::groove:horizontal { height: 8px; }
        QDialog QSlider::handle:horizontal { width: 18px; margin: -6px 0; }
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

// ---------- Sheet toggle / FAB positioning ----------
void MainWindow::positionFloatingButtons()
{
    if (!m_videoArea) return;
    const int w = m_videoArea->width();
    const int h = m_videoArea->height();

    // Mode Toggle (Dual/Diff) - bottom center
    if (m_btnModeToggle) {
        if (!m_focusViewActive) {
            m_btnModeToggle->show();
            m_btnModeToggle->move(w / 2 - m_btnModeToggle->width() / 2, h - 14 - m_btnModeToggle->height());
            m_btnModeToggle->raise();
        } else {
            m_btnModeToggle->hide();
        }
    }
    // Stream button - bottom left
    if (m_btnFabStream) {
        if (!m_focusViewActive) {
            m_btnFabStream->show();
            m_btnFabStream->move(14, h - 14 - m_btnFabStream->height());
            m_btnFabStream->raise();
        } else {
            m_btnFabStream->hide();
        }
    }
    // Snapshot button - bottom right
    if (m_btnFabSnapshot) {
        if (!m_focusViewActive) {
            m_btnFabSnapshot->show();
            m_btnFabSnapshot->move(w - 14 - m_btnFabSnapshot->width(), h - 14 - m_btnFabSnapshot->height());
            m_btnFabSnapshot->raise();
        } else {
            m_btnFabSnapshot->hide();
        }
    }
    // Focus Toggle - top center
    if (m_btnFocusToggle) {
        m_btnFocusToggle->show();
        m_btnFocusToggle->move(w / 2 - m_btnFocusToggle->width() / 2, 14);
        m_btnFocusToggle->raise();
    }
}

void MainWindow::toggleSheet()
{
    m_sheetOpen = !m_sheetOpen;
    m_sheetWidget->setVisible(m_sheetOpen);
    QLabel* chev = m_btnSheetHandle->findChild<QLabel*>("sheetChevron");
    if (chev) chev->setText(m_sheetOpen ? QString::fromUtf8("\u25BC")
                                         : QString::fromUtf8("\u25B2"));
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

void MainWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);

    const int w = width();
    const bool compact = w < 720;
    const bool ultraCompact = w < 560;

    if (m_fpsPill) m_fpsPill->setVisible(!ultraCompact);
    if (m_eccPill) m_eccPill->setVisible(!ultraCompact);
    if (m_btnGallery) m_btnGallery->setText(compact ? QStringLiteral("▦") : QStringLiteral("▦  Snapshots"));

    int sheetMax = std::max(140, height() / 2);
    if (m_sheetWidget) m_sheetWidget->setMaximumHeight(sheetMax);
}

void MainWindow::toggleFocusView()
{
    m_focusViewActive = !m_focusViewActive;
    if (m_focusViewActive) {
        m_splitter->hide();
        m_resultView->hide();
        m_chartView->show();
        if (m_sheetStack) m_sheetStack->setCurrentIndex(1);
        if (m_btnFocusToggle) m_btnFocusToggle->setChecked(true);
    } else {
        m_chartView->hide();
        if (m_isDiffMode) {
            m_splitter->hide();
            m_resultView->show();
        } else {
            m_splitter->show();
            m_resultView->hide();
        }
        if (m_sheetStack) m_sheetStack->setCurrentIndex(0);
        if (m_btnFocusToggle) m_btnFocusToggle->setChecked(false);
    }
    positionFloatingButtons();
}

void MainWindow::refreshSnapshotPreview()
{
    if (!m_snapshotPreview) return;
    m_snapshotPreview->clear();

    QString dir = QCoreApplication::applicationDirPath() + "/metrics";
    QDir metricsDir(dir);
    if (!metricsDir.exists()) return;

    QStringList filters = {"*.png", "*.jpg", "*.jpeg", "*.bmp"};
    QFileInfoList files = metricsDir.entryInfoList(filters, QDir::Files, QDir::Time);

    const QSize grid = m_snapshotPreview->gridSize();
    for (const QFileInfo& fi : files) {
        QPixmap pix(fi.absoluteFilePath());
        if (pix.isNull()) continue;
        QListWidgetItem* item = new QListWidgetItem(
            QIcon(pix.scaled(160, 120, Qt::KeepAspectRatio, Qt::SmoothTransformation)),
            fi.fileName());
        item->setToolTip(fi.fileName());
        item->setTextAlignment(Qt::AlignHCenter | Qt::AlignTop);
        if (grid.isValid()) item->setSizeHint(grid);
        m_snapshotPreview->addItem(item);
    }
}

void MainWindow::updateEccPill()
{
    if (!m_eccPill) return;
    if (m_isAligned && !m_eccWarpMatrix.empty()) {
        if (m_chkAlign && m_chkAlign->isChecked()) {
            m_eccPill->setText("ECC ON");
            m_eccPill->setStyleSheet(pillStyle(T::ok, "#143022", T::border));
        } else {
            m_eccPill->setText("ECC READY");
            m_eccPill->setStyleSheet(pillStyle(T::accent, "#142030", T::border));
        }
    } else if (m_calibrating) {
        m_eccPill->setText("CALIBRATING\u2026");
        m_eccPill->setStyleSheet(pillStyle(T::warn, "#2a2310", T::border));
    } else {
        m_eccPill->setText("ECC NONE");
        m_eccPill->setStyleSheet(pillStyle(T::err, "#2a1414", T::border));
    }
}

void MainWindow::updateFpsPill()
{
    if (!m_fpsPill) return;
    if (m_camerasOpen) {
        m_fpsPill->setText("STREAMING");
        m_fpsPill->setStyleSheet(pillStyle(T::ok, "#143022", T::border));
    } else {
        m_fpsPill->setText("IDLE");
        m_fpsPill->setStyleSheet(pillStyle(T::textDim, T::bg2, T::border));
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

// ---------- libcamera detection ----------
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
    // Pipeline-side controls are baked at start; for live updates we'd need
    // libcamera-controls bus messages. Restart cameras to apply.
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
        "QPushButton { background: #3a1e1e; border: 2px solid %1; border-radius: 24px; }"
        "QPushButton:hover { background: #5a2a2a; }").arg(T::err));
    if (m_fabStreamIcon) {
        m_fabStreamIcon->setText(QString::fromUtf8("\u25A0"));
        m_fabStreamIcon->setStyleSheet(QString(
            "QLabel { background: transparent; color: %1; border: none;"
            " font-family: 'Segoe UI Symbol','Segoe UI Emoji','Arial Unicode MS';"
            " font-size: 22px; font-weight: 800; }").arg(T::err));
    }

    updateEccPill();
    updateFpsPill();

    m_motionActive = false;

    m_frameCount = 0;
    m_seriesCam1->clear();
    m_seriesCam2->clear();

    m_chart->axes(Qt::Horizontal).first()->setRange(0, m_maxHistory);
    m_chart->axes(Qt::Vertical).first()->setRange(0, 1000);

    m_btnFabStream->setObjectName("fabStreamStop");
    m_btnFabStream->setText(QString::fromUtf8("\u25A0"));
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
            "QPushButton { background: %1; border: 2px solid %2; border-radius: 24px; }"
            "QPushButton:hover { background: #3a5a8a; }")
            .arg(T::accentDim).arg(T::accent));
        if (m_fabStreamIcon) {
            m_fabStreamIcon->setText(QString::fromUtf8("\u25B6"));
            m_fabStreamIcon->setStyleSheet(QString(
                "QLabel { background: transparent; color: %1; border: none;"
                " font-family: 'Segoe UI Symbol','Segoe UI Emoji','Arial Unicode MS';"
                " font-size: 22px; font-weight: 800; }").arg(T::accent));
        }
    }
    if (m_view1) { m_view1->setOverlayText("", false); m_view1->setPlaceholder("Camera 1"); }
    if (m_view2) { m_view2->setOverlayText("", false); m_view2->setPlaceholder("Camera 2"); }

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
    if (obj == m_btnSheetHandle) {
        if (event->type() == QEvent::MouseButtonPress) {
            QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton) {
                m_dragStartPos = mouseEvent->globalPosition().toPoint().y();
                m_dragStartHeight = m_sheetWidget->height();
                m_isDraggingSheet = true;
                return true;
            }
        } else if (event->type() == QEvent::MouseMove) {
            if (m_isDraggingSheet) {
                QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
                int delta = m_dragStartPos - mouseEvent->globalPosition().toPoint().y();
                int newHeight = m_dragStartHeight + delta;
                int maxH = this->height() / 2;
                if (newHeight < 140) newHeight = 140;
                if (newHeight > maxH) newHeight = maxH;
                
                m_sheetWidget->setFixedHeight(newHeight);
                return true;
            }
        } else if (event->type() == QEvent::MouseButtonRelease) {
            if (m_isDraggingSheet) {
                QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
                int delta = m_dragStartPos - mouseEvent->globalPosition().toPoint().y();
                m_isDraggingSheet = false;
                if (std::abs(delta) < 5) {
                    toggleSheet();
                }
                return true;
            }
        }
    } else if (obj == m_videoArea) {
        if (event->type() == QEvent::Resize) {
            positionFloatingButtons();
        }
    }
    return QMainWindow::eventFilter(obj, event);
}


void MainWindow::onFramesProcessed(cv::Mat f1, cv::Mat f2, double focus1, double focus2, bool motionDetected, qint64 frameCount)
{
    if (m_worker) m_worker->m_pendingFrames.fetch_sub(1);

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

    if (m_focusViewActive) return;  // chart handles its own display

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
            cv::warpAffine(f2, warpedF2, m_eccWarpMatrix, f1.size(),
                cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
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
    if (m_eccIndicator) {
        m_eccIndicator->setText("CALIBRATING\u2026");
        m_eccIndicator->setStyleSheet(pillStyle(T::warn, "#2a2310", T::border));
    }
    updateEccPill();
    m_statusBar->showMessage("Calibrating (ECC)... please wait.");

    if (m_calibThread.joinable()) m_calibThread.join();

    m_calibThread = std::thread([this, gray1 = std::move(gray1), gray2 = std::move(gray2)]() {
        cv::Mat g1, g2;
        cv::equalizeHist(gray1, g1);
        cv::equalizeHist(gray2, g2);

        cv::Mat g1_small, g2_small;
        cv::resize(g1, g1_small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        cv::resize(g2, g2_small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);

        cv::TermCriteria criteria(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
            500, 1e-6);

        QString errorMsg;

        auto tryPyramid = [&](int motionType, cv::Mat& outMat) -> bool {
            cv::Mat coarse = cv::Mat::eye(2, 3, CV_32F);
            try {
                cv::findTransformECC(g1_small, g2_small, coarse,
                    motionType, criteria, cv::noArray(), 9);
            }
            catch (const cv::Exception& e) {
                errorMsg = QString::fromStdString(e.what());
                return false;
            }

            cv::Mat fine = coarse.clone();
            fine.at<float>(0, 2) *= 4.0f;
            fine.at<float>(1, 2) *= 4.0f;
            cv::Mat fineBackup = fine.clone();

            try {
                cv::findTransformECC(g1, g2, fine,
                    motionType, criteria, cv::noArray(), 9);
                outMat = fine;
            }
            catch (const cv::Exception&) {
                outMat = fineBackup;
            }
            return true;
        };

        cv::Mat warpMatrix;
        bool success = tryPyramid(cv::MOTION_AFFINE, warpMatrix);
        QString modelUsed = "AFFINE";
        if (!success) {
            success = tryPyramid(cv::MOTION_EUCLIDEAN, warpMatrix);
            if (success) modelUsed = "EUCLIDEAN";
        }

        cv::Mat resultMatrix = success ? warpMatrix : cv::Mat();
        QMetaObject::invokeMethod(this, [this, success, resultMatrix, errorMsg, modelUsed]() {
            if (success) {
                m_eccWarpMatrix = resultMatrix;
                m_isAligned = true;
                if (m_eccIndicator) {
                    m_eccIndicator->setText("MATRIX READY");
                    m_eccIndicator->setStyleSheet(pillStyle(T::ok, "#143022", T::border));
                }
                m_statusBar->showMessage("Alignment OK (ECC, " + modelUsed + ")!", 3000);
            }
            else {
                m_eccWarpMatrix.release();
                m_isAligned = false;
                if (m_eccIndicator) {
                    m_eccIndicator->setText("NOT CALIBRATED");
                    m_eccIndicator->setStyleSheet(pillStyle(T::err, "#2a1414", T::border));
                }
                m_statusBar->showMessage("ECC failed (low texture / too different views): " + errorMsg, 5000);
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
    dlg.setStyleSheet(styleSheetText() + "QDialog { background: #1a1e24; }");
    dlg.resize(440, 520);

    QVBoxLayout* root = new QVBoxLayout(&dlg);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    QLabel* title = new QLabel("<b>Filename parameters</b>", &dlg);
    title->setStyleSheet("padding: 20px 20px 5px 20px; font-size: 16px; color: #d8dde4;");
    root->addWidget(title);

    QLabel* hint = new QLabel("Only checked parameters appear in the filename, and only when their effect is active.", &dlg);
    hint->setStyleSheet("padding: 0px 20px 14px 20px; color: #8a94a3; font-size: 11px;");
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
    bottomBar->setStyleSheet("background: #14171c; border-top: 1px solid #272c34;");
    QHBoxLayout* btns = new QHBoxLayout(bottomBar);
    btns->setContentsMargins(20, 14, 20, 14);

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
    dlg.setStyleSheet(styleSheetText() + "QDialog { background: #1a1e24; }");
    dlg.resize(480, 320);

    QVBoxLayout* root = new QVBoxLayout(&dlg);
    root->setContentsMargins(20, 20, 20, 20);
    root->setSpacing(14);

    QCheckBox* chkPerCam = new QCheckBox("Per-camera exposure (separate Cam1 / Cam2)", &dlg);
    chkPerCam->setChecked(m_perCameraExposure);
    chkPerCam->setStyleSheet("font-size: 12px; color: #d8dde4;");
    root->addWidget(chkPerCam);

    // Cam1 / Cam2 toggle row (shown only in per-camera mode)
    QFrame* camToggleRow = new QFrame(&dlg);
    camToggleRow->setProperty("role", "panel");
    QHBoxLayout* camToggleLay = new QHBoxLayout(camToggleRow);
    camToggleLay->setContentsMargins(14, 8, 14, 8);
    camToggleLay->setSpacing(8);
    QLabel* camLbl = new QLabel("Camera:", camToggleRow);
    camLbl->setStyleSheet("font-weight: 600; font-size: 12px; color: #d8dde4;");
    camToggleLay->addWidget(camLbl);
    QPushButton* btnCam1 = new QPushButton("Cam1", camToggleRow);
    QPushButton* btnCam2 = new QPushButton("Cam2", camToggleRow);
    btnCam1->setCheckable(true);
    btnCam2->setCheckable(true);
    btnCam1->setCursor(Qt::PointingHandCursor);
    btnCam2->setCursor(Qt::PointingHandCursor);
    QString camBtnSty = QString(
        "QPushButton { background: %1; color: %2; border: 1px solid %3; border-radius: 4px;"
        " padding: 6px 16px; font-weight: 600; }"
        "QPushButton:checked { background: %4; color: #0a1220; border-color: %4; }")
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
        QLabel* lbl = new QLabel(title, row);
        lbl->setStyleSheet("font-weight: 600; font-size: 12px; color: #d8dde4;");
        head->addWidget(lbl);
        head->addStretch();
        head->addWidget(spin);
        lay->addLayout(head);
        lay->addWidget(slider);
        return row;
    };

    // Single set of editor widgets in the dialog. They proxy to whichever
    // backing pair (Cam1 or Cam2) is currently selected.
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

    // activeCam2 = false → editing Cam1 / shared, true → editing Cam2.
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
            // Mirror onto cam2 backing widgets so hotkeys/labels stay in sync.
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
        // Pill in the top bar updates via cam1/cam2 backing widgets' valueChanged → refreshExposureLabel,
        // but we set them with QSignalBlocker, so trigger a rebuild manually:
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

    // Wire spin↔slider within the dialog and push to backing on each edit.
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

    // Cam1/Cam2 toggle behaviour.
    auto selectCam = [activeCam2, btnCam1, btnCam2, loadValuesIntoEditors](bool cam2) {
        *activeCam2 = cam2;
        QSignalBlocker b1(btnCam1), b2(btnCam2);
        btnCam1->setChecked(!cam2);
        btnCam2->setChecked(cam2);
        loadValuesIntoEditors();
    };
    connect(btnCam1, &QPushButton::clicked, &dlg, [selectCam]() { selectCam(false); });
    connect(btnCam2, &QPushButton::clicked, &dlg, [selectCam]() { selectCam(true);  });

    // Per-camera checkbox: toggles the cam toggle row + reverts to Cam1 when off.
    connect(chkPerCam, &QCheckBox::toggled, &dlg, [this, camToggleRow, selectCam](bool on) {
        m_perCameraExposure = on;
        camToggleRow->setVisible(on);
        if (!on) selectCam(false);
        else      selectCam(false); // start at Cam1 in per-camera mode too
        applyExposureControls();
    });

    QHBoxLayout* btns = new QHBoxLayout();
    btns->addStretch();
    QPushButton* btnClose = new QPushButton("Close", &dlg);
    btnClose->setProperty("kind", "primary");
    btns->addWidget(btnClose);
    root->addLayout(btns);
    connect(btnClose, &QPushButton::clicked, &dlg, &QDialog::accept);

    dlg.exec();
}

void MainWindow::saveSnapshot()
{
    if (m_isDiffMode) {
        saveDiffSnapshot();
    } else {
        saveDualSnapshot(true);
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

    QString filename = buildSnapshotBaseName("diff");
    QString filePath = metricsDir + "/" + filename + ".png";
    QString jsonPath = metricsDir + "/" + filename + ".json";

    bool success = cv::imwrite(filePath.toStdString(), m_lastDiffResult);

    if (success) {
        writeSnapshotMeta(jsonPath, "diff");
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
        QString path = metricsDir + "/" + filename + ".png";
        bool ok = cv::imwrite(path.toStdString(), combo);
        if (ok) writeSnapshotMeta(metricsDir + "/" + filename + ".json", "dual_combined");
        m_statusBar->showMessage(ok ? "Saved: " + path : "Error saving: " + path, 4000);
    } else {
        QString filename1 = baseName + "_cam1";
        QString filename2 = baseName + "_cam2";
        QString path1 = metricsDir + "/" + filename1 + ".png";
        QString path2 = metricsDir + "/" + filename2 + ".png";
        bool ok1 = cv::imwrite(path1.toStdString(), f1);
        bool ok2 = cv::imwrite(path2.toStdString(), f2);
        if (ok1) writeSnapshotMeta(metricsDir + "/" + filename1 + ".json", "dual_cam1");
        if (ok2) writeSnapshotMeta(metricsDir + "/" + filename2 + ".json", "dual_cam2");
        if (ok1 && ok2)
            m_statusBar->showMessage("Saved cam1 and cam2 snapshots.", 4000);
        else
            m_statusBar->showMessage("Error saving dual snapshots.", 4000);
    }
}

void MainWindow::writeSnapshotMeta(const QString& jsonPath, const QString& mode) const
{
    QJsonObject obj;

    obj["mode"]      = mode;
    obj["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    obj["userName"]  = m_snapshotNameEdit ? m_snapshotNameEdit->text().trimmed() : QString();

    // Color / flip
    QString colorStr;
    switch (m_colorMode) {
        case ColorMode::COLOR:       colorStr = "RGB";        break;
        case ColorMode::GRAY_CV:     colorStr = "GRAY_CV";    break;
        case ColorMode::GRAY_NATIVE: colorStr = "GRAY_NATIVE"; break;
    }
    obj["colorMode"] = colorStr;
    obj["flipHorizontal2"] = m_chkFlipHor2 && m_chkFlipHor2->isChecked();
    obj["flipVertical2"]   = m_chkFlipVer2 && m_chkFlipVer2->isChecked();

    // Pipeline
    obj["timeBuffer"]      = m_bufferSlider ? m_bufferSlider->value() : 0;
    obj["motionThreshold"] = m_motionThresholdSlider ? m_motionThresholdSlider->value() : 0;
    obj["fusion"]          = m_chkFusion && m_chkFusion->isChecked();
    obj["bilateralFilter"] = m_chkBilateral && m_chkBilateral->isChecked();
    obj["bilateralStrength"] = m_bilateralSlider ? m_bilateralSlider->value() : m_bilateralStrength;
    obj["noiseFloor"]      = m_noiseFloorSlider ? m_noiseFloorSlider->value() : m_noiseFloor;
    obj["intensityStretch"] = m_chkStretch && m_chkStretch->isChecked();
    obj["trackPeaks"]      = m_btnPeakIntensities && m_btnPeakIntensities->isChecked();

    // Alignment
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

    // Exposure
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

    // Focus / motion (last reading)
    QJsonObject focus;
    focus["cam1"] = m_lastFocus1;
    focus["cam2"] = m_lastFocus2;
    obj["focus"] = focus;
    obj["motionActive"] = m_motionActive;

    // Frame mode
    obj["diffMode"]   = m_isDiffMode;
    obj["frameCount"] = m_frameCount;

    QJsonDocument doc(obj);
    QFile f(jsonPath);
    if (f.open(QIODevice::WriteOnly)) {
        f.write(doc.toJson());
        f.close();
    }
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
    s.setValue("sheetOpen", m_sheetOpen);

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
    s.sync();
}

void MainWindow::loadSettings()
{
    QSettings s(settingsPath(), QSettings::IniFormat);
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
    if (sheetOpen != m_sheetOpen) toggleSheet();

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
        dlg->setStyleSheet(styleSheetText() + "QDialog { background: #1a1e24; border: 1px solid #6aa9ff; border-radius: 6px; }");
        
        QVBoxLayout* lay = new QVBoxLayout(dlg);
        lay->setContentsMargins(20, 20, 20, 20);
        
        QLabel* lbl = new QLabel(title, dlg);
        lbl->setAlignment(Qt::AlignCenter);
        lbl->setStyleSheet("font-weight: bold; color: #d8dde4; font-size: 14px;");
        lay->addWidget(lbl);
        
        QSlider* sl = new QSlider(Qt::Horizontal, dlg);
        sl->setRange(slider->minimum(), slider->maximum());
        sl->setValue(slider->value());
        sl->setMinimumWidth(250);
        lay->addWidget(sl);
        
        QLabel* valLbl = new QLabel(QString::number(sl->value()), dlg);
        valLbl->setAlignment(Qt::AlignCenter);
        valLbl->setStyleSheet("color: #8a94a3; font-size: 12px;");
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
        {"cmd_sheet", "Toggle Bottom Sheet", "Capture", CmdType::Action, [this](){ toggleSheet(); }, {}},
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

        {"cmd_ecc", "Toggle ECC Alignment", "Pipeline", CmdType::Toggle, [this](){ if (m_chkAlign) m_chkAlign->setChecked(!m_chkAlign->isChecked()); }, {}},
        {"cmd_fusion", "Toggle Fusion", "Pipeline", CmdType::Toggle, [this](){ if (m_chkFusion) m_chkFusion->setChecked(!m_chkFusion->isChecked()); }, {}},
        {"cmd_bilateral", "Toggle Bilateral Filter", "Pipeline", CmdType::Toggle, [this](){ if (m_chkBilateral) m_chkBilateral->setChecked(!m_chkBilateral->isChecked()); }, {}},
        {"cmd_stretch", "Toggle Intensity Stretch", "Pipeline", CmdType::Toggle, [this](){ if (m_chkStretch) m_chkStretch->setChecked(!m_chkStretch->isChecked()); }, {}},
        {"cmd_peaks", "Toggle Tracking Peaks", "Pipeline", CmdType::Toggle, [this](){ if (m_btnPeakIntensities) m_btnPeakIntensities->setChecked(!m_btnPeakIntensities->isChecked()); }, {}},
        {"cmd_calibrate", "Calibrate ECC Alignment", "Pipeline", CmdType::Action, [this](){ calibrateAlignment(); }, {}},
        {"cmd_open_align", "Open Manual Align Dialog", "Pipeline", CmdType::Action, [this](){ if (m_btnOpenManualAlign) m_btnOpenManualAlign->click(); }, {}},

        {"cmd_tbuffer", "Time Buffer Size", "Pipeline Parameters", CmdType::Parameter, {}, [this, createPopup](QWidget*){ createPopup("Time Buffer Size", m_bufferSlider); }},
        {"cmd_motionthr", "Motion Threshold", "Pipeline Parameters", CmdType::Parameter, {}, [this, createPopup](QWidget*){ createPopup("Motion Threshold", m_motionThresholdSlider); }},
        {"cmd_noisefloor", "Noise Floor", "Pipeline Parameters", CmdType::Parameter, {}, [this, createPopup](QWidget*){ createPopup("Noise Floor", m_noiseFloorSlider); }},
        {"cmd_exposure_dialog", "Open Exposure Dialog", "Exposure", CmdType::Action, [this](){ if (m_manualExposure) showExposureDialog(); }, {}},

        {"cmd_rename_snapshot", "Edit Snapshot Name", "Snapshot", CmdType::Action, [this](){
            if (m_tabWidget) m_tabWidget->setCurrentIndex(3);
            if (!m_sheetOpen) toggleSheet();
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
        setStyleSheet("QPushButton:checked { background: #2c4a7a; color: #6aa9ff; border: 1px solid #6aa9ff; }");
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
    dlg.setStyleSheet(styleSheetText() + "QDialog { background: #1a1e24; }");
    dlg.resize(500, 600);
    
    QVBoxLayout* root = new QVBoxLayout(&dlg);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);
    
    QLabel* title = new QLabel("<b>Actions & Keybinds</b>", &dlg);
    title->setStyleSheet("padding: 20px 20px 5px 20px; font-size: 16px; color: #d8dde4;");
    root->addWidget(title);

    QLabel* tip = new QLabel("Click a shortcut box to assign a key. Click 'Execute' to run immediately.", &dlg);
    tip->setStyleSheet("padding: 0px 20px 20px 20px; color: #8a94a3; font-size: 11px;");
    root->addWidget(tip);
    
    QScrollArea* scroll = new QScrollArea(&dlg);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setStyleSheet("QScrollArea { background: transparent; } QWidget#scrollContent { background: transparent; }");
    
    QWidget* scrollContent = new QWidget();
    scrollContent->setObjectName("scrollContent");
    QVBoxLayout* listLay = new QVBoxLayout(scrollContent);
    listLay->setContentsMargins(20, 0, 20, 20);
    listLay->setSpacing(12);
    
    QString lastSection = "";
    
    QMap<QString, HotkeyButton*> editorMap;
    
    for (const auto& cmd : m_commands) {
        if (cmd.section != lastSection) {
            QLabel* sec = new QLabel(cmd.section.toUpper());
            sec->setStyleSheet("color: #6aa9ff; font-weight: bold; font-size: 11px; margin-top: 10px; margin-bottom: 2px; letter-spacing: 0.5px;");
            listLay->addWidget(sec);
            lastSection = cmd.section;
        }
        
        QFrame* row = new QFrame();
        row->setProperty("role", "panel");
        QHBoxLayout* rowLay = new QHBoxLayout(row);
        rowLay->setContentsMargins(14, 8, 14, 8);
        
        QLabel* name = new QLabel(cmd.name);
        name->setStyleSheet("font-weight: 600; font-size: 13px;");
        rowLay->addWidget(name, 1);
        
        QHBoxLayout* editorLay = new QHBoxLayout();
        editorLay->setSpacing(4);
        HotkeyButton* editor = new HotkeyButton(m_hotkeys.value(cmd.id));
        editor->setMinimumHeight(26);
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
        
        QPushButton* btnClear = new QPushButton(QString::fromUtf8("\u2715")); // X
        btnClear->setFixedSize(26, 26);
        btnClear->setStyleSheet("background: transparent; color: #ff6a6a; border: none; font-weight: bold; font-size: 14px; padding: 0;");
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
    bottomBar->setStyleSheet("background: #14171c; border-top: 1px solid #272c34;");
    QHBoxLayout* btnLay = new QHBoxLayout(bottomBar);
    btnLay->setContentsMargins(20, 14, 20, 14);
    
    btnLay->addStretch();
    
    QPushButton* resetBtn = new QPushButton("Reset", bottomBar);
    resetBtn->setProperty("kind", "primary");
    resetBtn->setStyleSheet("QPushButton { color: #FFFFFF; }");
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
    
    dlg.exec();
}
