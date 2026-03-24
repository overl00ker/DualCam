#include "mainwindow.h"

#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
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
#include <QDateTime>
#include <QGroupBox>
#include <QStackedWidget>
#include <QtCharts/QValueAxis>
#include <QtCharts/QChart>
#include <QtCharts/QLegend>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>

#include <iostream>
#include <vector>
#include <cmath>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    cv::setUseOptimized(true);

    m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16.0, false);

    m_timer = new QTimer(this);
    connect(m_timer, &QTimer::timeout, this, &MainWindow::updateFrames);

    initUI();
}

MainWindow::~MainWindow()
{
    closeCameras();
}

void MainWindow::initUI()
{
    setWindowTitle("DualCam Analysis Tool");
    setGeometry(100, 100, 1380, 900);

    m_centralWidget = new QWidget(this);
    setCentralWidget(m_centralWidget);

    QVBoxLayout* mainLayout = new QVBoxLayout(m_centralWidget);
    mainLayout->setContentsMargins(5, 5, 5, 5);

    QHBoxLayout* navBarLayout = new QHBoxLayout();

    m_btnToggleCameras = new QPushButton("Open Cameras", this);
    connect(m_btnToggleCameras, &QPushButton::clicked, this, [this]()
        {
            if (m_camerasOpen) closeCameras();
            else openCameras();
        });

    m_navCalibrate = new QPushButton("Calibrate", this);
    m_navFocus = new QPushButton("Focus", this);
    m_navLiveView = new QPushButton("Live View", this);
    m_navMode = new QPushButton("Dual View", this);
    m_navMode->setCheckable(true);

    QString navStyle = "QPushButton { font-weight: bold; font-size: 14px; padding: 8px 4px; }";
    m_btnToggleCameras->setStyleSheet(navStyle);
    m_navCalibrate->setStyleSheet(navStyle);
    m_navFocus->setStyleSheet(navStyle);
    m_navLiveView->setStyleSheet(navStyle);
    m_navMode->setStyleSheet(navStyle + " QPushButton:checked { background-color: #2196F3; color: white; border: none; }");

    m_btnToggleCameras->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_navCalibrate->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_navFocus->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_navLiveView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_navMode->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

    navBarLayout->addWidget(m_btnToggleCameras);
    navBarLayout->addWidget(m_navCalibrate);
    navBarLayout->addWidget(m_navFocus);
    navBarLayout->addWidget(m_navLiveView);
    navBarLayout->addWidget(m_navMode);

    mainLayout->addLayout(navBarLayout);

    m_stackedWidget = new QStackedWidget(this);
    mainLayout->addWidget(m_stackedWidget, 1);

    QWidget* videoPage = new QWidget();
    QVBoxLayout* videoPageLayout = new QVBoxLayout(videoPage);
    videoPageLayout->setContentsMargins(0, 0, 0, 0);

    m_controlsWidget = new QWidget();
    QHBoxLayout* panelsLayout = new QHBoxLayout(m_controlsWidget);
    panelsLayout->setContentsMargins(0, 5, 0, 5);

    QGroupBox* camGroup = new QGroupBox("Camera Setup", this);
    QVBoxLayout* camLayout = new QVBoxLayout(camGroup);

    QHBoxLayout* camMid = new QHBoxLayout();
    m_comboColorMode = new QComboBox(this);
    m_comboColorMode->addItems({ "Gray Native", "Gray CV", "Color" });
    m_comboColorMode->setCurrentIndex(1);
    connect(m_comboColorMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        m_colorMode = static_cast<ColorMode>(index);
        m_frameBuffer1.clear();
        m_frameBuffer2.clear();
        });
    camMid->addWidget(new QLabel("Color:"));
    camMid->addWidget(m_comboColorMode);

    m_chkFlipVer2 = new QCheckBox("Flip 2 V", this);
    m_chkFlipHor2 = new QCheckBox("Flip 2 H", this);
    camMid->addWidget(m_chkFlipVer2);
    camMid->addWidget(m_chkFlipHor2);
    camLayout->addLayout(camMid);

    QHBoxLayout* camBot = new QHBoxLayout();
    m_chkAlign = new QCheckBox("Align Frames", this);
    connect(m_chkAlign, &QCheckBox::stateChanged, this, &MainWindow::updateView);
    m_btnCalibrateAlign = new QPushButton("Calibrate ECC", this);
    connect(m_btnCalibrateAlign, &QPushButton::clicked, this, &MainWindow::calibrateAlignment);
    camBot->addWidget(m_chkAlign);
    camBot->addWidget(m_btnCalibrateAlign);
    camLayout->addLayout(camBot);

    panelsLayout->addWidget(camGroup);

    QGroupBox* noiseGroup = new QGroupBox("Denoise & Fusion", this);
    QVBoxLayout* noiseLayout = new QVBoxLayout(noiseGroup);

    QHBoxLayout* bufLayout = new QHBoxLayout();
    m_bufferSlider = new QSlider(Qt::Horizontal, this);
    m_bufferSlider->setRange(1, 30);
    m_bufferSlider->setValue(m_bufferSize);
    m_bufferLabel = new QLabel(QString::number(m_bufferSize), this);
    m_bufferLabel->setFixedWidth(25);
    connect(m_bufferSlider, &QSlider::valueChanged, this, [this](int value) {
        m_bufferSize = value;
        m_bufferLabel->setText(QString::number(value));
        });
    bufLayout->addWidget(new QLabel("T-Buffer:"));
    bufLayout->addWidget(m_bufferSlider);
    bufLayout->addWidget(m_bufferLabel);
    noiseLayout->addLayout(bufLayout);

    QHBoxLayout* motLayout = new QHBoxLayout();
    m_motionThresholdSlider = new QSlider(Qt::Horizontal, this);
    m_motionThresholdSlider->setRange(1, 50);
    m_motionThresholdSlider->setValue(static_cast<int>(m_motionThreshold * 100));
    m_motionThresholdLabel = new QLabel(QString("%1%").arg(m_motionThreshold * 100, 0, 'f', 0), this);
    m_motionThresholdLabel->setFixedWidth(35);
    connect(m_motionThresholdSlider, &QSlider::valueChanged, this, [this](int value) {
        m_motionThreshold = static_cast<double>(value) / 100.0;
        m_motionThresholdLabel->setText(QString("%1%").arg(value));
        });
    motLayout->addWidget(new QLabel("Motion Thr:"));
    motLayout->addWidget(m_motionThresholdSlider);
    motLayout->addWidget(m_motionThresholdLabel);
    noiseLayout->addLayout(motLayout);

    QHBoxLayout* filtLayout = new QHBoxLayout();
    m_chkFusion = new QCheckBox("Camera Fusion", this);
    m_chkBilateral = new QCheckBox("Bilateral Filter", this);
    filtLayout->addWidget(m_chkFusion);
    filtLayout->addWidget(m_chkBilateral);
    noiseLayout->addLayout(filtLayout);

    m_motionIndicator = new QLabel(QString::fromUtf8("\u25CF Idle"), this);
    m_motionIndicator->setStyleSheet("font-weight: bold; color: gray;");
    noiseLayout->addWidget(m_motionIndicator);

    panelsLayout->addWidget(noiseGroup);

    QGroupBox* diffGroup = new QGroupBox("Difference Analysis", this);
    QVBoxLayout* diffLayout = new QVBoxLayout(diffGroup);

    QHBoxLayout* noiseFloorLayout = new QHBoxLayout();
    m_noiseFloorSlider = new QSlider(Qt::Horizontal, this);
    m_noiseFloorSlider->setRange(0, 50);
    m_noiseFloorSlider->setValue(m_noiseFloor);
    m_noiseFloorLabel = new QLabel(QString::number(m_noiseFloor), this);
    m_noiseFloorLabel->setFixedWidth(25);
    connect(m_noiseFloorSlider, &QSlider::valueChanged, this, [this](int value) {
        m_noiseFloor = value;
        m_noiseFloorLabel->setText(QString::number(value));
        });
    noiseFloorLayout->addWidget(new QLabel("Noise Floor:"));
    noiseFloorLayout->addWidget(m_noiseFloorSlider);
    noiseFloorLayout->addWidget(m_noiseFloorLabel);
    diffLayout->addLayout(noiseFloorLayout);

    m_chkStretch = new QCheckBox("Stretch Intensity Range", this);
    diffLayout->addWidget(m_chkStretch);

    QHBoxLayout* diffBtnsLayout = new QHBoxLayout();
    m_btnPeakIntensities = new QPushButton("Track Peaks", this);
    m_btnPeakIntensities->setCheckable(true);
    connect(m_btnPeakIntensities, &QPushButton::toggled, this, [this](bool checked) {
        if (!checked) m_lblPeakInfo->clear();
        updateView();
        });
    m_btnSaveDiff = new QPushButton("Save Snapshot", this);
    connect(m_btnSaveDiff, &QPushButton::clicked, this, &MainWindow::saveDiffSnapshot);
    diffBtnsLayout->addWidget(m_btnPeakIntensities);
    diffBtnsLayout->addWidget(m_btnSaveDiff);
    diffLayout->addLayout(diffBtnsLayout);

    m_lblPeakInfo = new QLabel("", this);
    m_lblPeakInfo->setStyleSheet("font-weight: bold; color: #ffaa00;");
    diffLayout->addWidget(m_lblPeakInfo);

    panelsLayout->addWidget(diffGroup);
    panelsLayout->addStretch();

    videoPageLayout->addWidget(m_controlsWidget);

    QWidget* videoAreaWidget = new QWidget();
    QVBoxLayout* videoAreaLayout = new QVBoxLayout(videoAreaWidget);
    videoAreaLayout->setContentsMargins(0, 0, 0, 0);

    m_splitter = new QSplitter(Qt::Horizontal, this);

    m_view1 = new QLabel("Camera 1", this);
    m_view1->setScaledContents(true);
    m_view1->setAlignment(Qt::AlignCenter);
    m_view1->setMinimumSize(320, 240);
    m_view1->setStyleSheet("background-color: black; color: white;");
    m_splitter->addWidget(m_view1);

    m_view2 = new QLabel("Camera 2", this);
    m_view2->setScaledContents(true);
    m_view2->setAlignment(Qt::AlignCenter);
    m_view2->setMinimumSize(320, 240);
    m_view2->setStyleSheet("background-color: black; color: white;");
    m_splitter->addWidget(m_view2);

    videoAreaLayout->addWidget(m_splitter, 1);

    m_resultView = new QLabel("Result View", this);
    m_resultView->setScaledContents(true);
    m_resultView->setAlignment(Qt::AlignCenter);
    m_resultView->setMinimumSize(640, 480);
    m_resultView->setStyleSheet("background-color: black; color: white;");
    videoAreaLayout->addWidget(m_resultView, 1);
    m_resultView->hide();

    videoPageLayout->addWidget(videoAreaWidget, 1);

    m_stackedWidget->addWidget(videoPage);

    QWidget* focusPage = new QWidget();
    QVBoxLayout* focusLayout = new QVBoxLayout(focusPage);

    m_chart = new QChart();
    m_chart->setTitle("Live Focus Analysis");
    m_chartView = new QChartView(m_chart);
    m_chartView->setRenderHint(QPainter::Antialiasing);

    m_seriesCam1 = new QLineSeries();
    m_seriesCam1->setName("Camera 1 Focus");
    m_seriesCam2 = new QLineSeries();
    m_seriesCam2->setName("Camera 2 Focus");

    m_chart->addSeries(m_seriesCam1);
    m_chart->addSeries(m_seriesCam2);

    QValueAxis* axisX = new QValueAxis;
    axisX->setTitleText("Time (Frames)");
    axisX->setRange(0, m_maxHistory);
    axisX->setTickCount(10);
    m_chart->addAxis(axisX, Qt::AlignBottom);

    QValueAxis* axisY = new QValueAxis;
    axisY->setTitleText("Focus Score");
    axisY->setRange(0, 5000);
    m_chart->addAxis(axisY, Qt::AlignLeft);

    m_seriesCam1->attachAxis(axisX);
    m_seriesCam1->attachAxis(axisY);
    m_seriesCam2->attachAxis(axisX);
    m_seriesCam2->attachAxis(axisY);

    m_chart->legend()->setVisible(true);
    m_chart->legend()->setAlignment(Qt::AlignBottom);

    focusLayout->addWidget(m_chartView);

    QHBoxLayout* historyLayout = new QHBoxLayout();
    m_historySpinBox = new QSpinBox(this);
    m_historySpinBox->setRange(50, 2000);
    m_historySpinBox->setValue(m_maxHistory);
    m_historySpinBox->setSingleStep(50);
    connect(m_historySpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value)
        {
            m_maxHistory = value;
            if (m_camerasOpen) {
                auto ax = m_chart->axes(Qt::Horizontal).first();
                ax->setRange(std::max(0LL, m_frameCount - m_maxHistory), m_frameCount);
            }
            else {
                m_chart->axes(Qt::Horizontal).first()->setRange(0, m_maxHistory);
            }
        });

    historyLayout->addStretch();
    historyLayout->addWidget(new QLabel("Graph History:", this));
    historyLayout->addWidget(m_historySpinBox);
    historyLayout->addStretch();
    focusLayout->addLayout(historyLayout);

    m_stackedWidget->addWidget(focusPage);

    connect(m_navCalibrate, &QPushButton::clicked, this, [this]() {
        m_stackedWidget->setCurrentIndex(0);
        m_controlsWidget->show();
        });

    connect(m_navLiveView, &QPushButton::clicked, this, [this]() {
        m_stackedWidget->setCurrentIndex(0);
        m_controlsWidget->hide();
        });

    connect(m_navFocus, &QPushButton::clicked, this, [this]() {
        m_stackedWidget->setCurrentIndex(1);
        });

    diffGroup->hide();
    connect(m_navMode, &QPushButton::toggled, this, [this, diffGroup](bool checked) {
        m_isDiffMode = checked;
        m_navMode->setText(checked ? "Diff View" : "Dual View");
        diffGroup->setVisible(checked);
        if (!checked) m_lblPeakInfo->clear();
        updateView();
        });

    m_statusBar = new QStatusBar(this);
    setStatusBar(m_statusBar);
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

std::string MainWindow::makeGStreamerPipeline(const QString& cameraId, int width, int height, int fps)
{
    return QString("libcamerasrc camera-name=%1 ! "
        "video/x-raw, width=%2, height=%3, format=NV12 ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=1 sync=false")
        .arg(cameraId)
        .arg(width)
        .arg(height)
        .toStdString();
}

void MainWindow::openCameras()
{
    m_statusBar->showMessage("Detecting cameras...");
    QApplication::processEvents();

    QStringList camPaths = getLibCameraIds();

    if (camPaths.size() < 2) {
        m_statusBar->showMessage("Warning: < 2 libcameras found. Attempting V4L2/DSHOW fallback.", 3000);
#ifdef Q_OS_LINUX
        m_cap1.open(0, cv::CAP_V4L2);
        m_cap2.open(1, cv::CAP_V4L2);
#else
        m_cap1.open(0, cv::CAP_DSHOW);
        m_cap2.open(1, cv::CAP_DSHOW);
#endif
    }
    else {
        std::string pipe1 = makeGStreamerPipeline(camPaths[0], 640, 480, 30);
        std::string pipe2 = makeGStreamerPipeline(camPaths[1], 640, 480, 30);

        m_cap1.open(pipe1, cv::CAP_GSTREAMER);
        m_cap2.open(pipe2, cv::CAP_GSTREAMER);
    }

    if (!m_cap1.isOpened() || !m_cap2.isOpened())
    {
        m_statusBar->showMessage("Error: Camera open failed.", 5000);
        if (m_cap1.isOpened()) m_cap1.release();
        if (m_cap2.isOpened()) m_cap2.release();
        return;
    }

    m_camerasOpen = true;
    m_isAligned = false;
    m_eccWarpMatrix.release();
    m_btnToggleCameras->setText("Close Cameras");

    m_frameBuffer1.clear();
    m_frameBuffer2.clear();
    m_motionActive = false;
    m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16.0, false);

    m_frameCount = 0;
    m_seriesCam1->clear();
    m_seriesCam2->clear();

    m_chart->axes(Qt::Horizontal).first()->setRange(0, m_maxHistory);
    m_chart->axes(Qt::Vertical).first()->setRange(0, 1000);

    m_timer->start(33);
    m_statusBar->showMessage("Cameras OK. Noise suppression active.", 4000);
}

void MainWindow::closeCameras()
{
    m_timer->stop();
    if (m_cap1.isOpened()) m_cap1.release();
    if (m_cap2.isOpened()) m_cap2.release();
    m_camerasOpen = false;
    m_btnToggleCameras->setText("Open Cameras");
    m_view1->clear();
    m_view2->clear();
    m_view1->setText("Camera 1");
    m_view2->setText("Camera 2");

    m_frameBuffer1.clear();
    m_frameBuffer2.clear();

    m_motionIndicator->setText(QString::fromUtf8("\u25CF Idle"));
    m_motionIndicator->setStyleSheet("font-weight: bold; color: gray;");

    m_statusBar->showMessage("Cameras closed.", 2000);
}

double MainWindow::detectMotion(const cv::Mat& frame)
{
    if (frame.empty()) return 0.0;

    cv::Mat gray;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = frame;

    cv::Mat smallGray;
    cv::resize(gray, smallGray, cv::Size(160, 120), 0, 0, cv::INTER_LINEAR);

    cv::Mat fgMask;
    m_bgSubtractor->apply(smallGray, fgMask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);

    int nonZero = cv::countNonZero(fgMask);
    return static_cast<double>(nonZero) / (smallGray.rows * smallGray.cols);
}

cv::Mat MainWindow::applyTemporalDenoise()
{
    if (m_frameBuffer1.empty()) return cv::Mat();

    if (m_frameBuffer1.size() == 1) {
        return m_frameBuffer1.front().clone();
    }

    int cvType = (m_frameBuffer1.front().channels() == 3) ? CV_32FC3 : CV_32F;

    cv::Mat accumulator;
    m_frameBuffer1.front().convertTo(accumulator, cvType);

    for (size_t i = 1; i < m_frameBuffer1.size(); ++i) {
        cv::Mat tmp;
        m_frameBuffer1[i].convertTo(tmp, cvType);
        accumulator += tmp;
    }

    accumulator /= static_cast<double>(m_frameBuffer1.size());

    cv::Mat result;
    accumulator.convertTo(result, m_frameBuffer1.front().type());
    return result;
}

cv::Mat MainWindow::fuseCameras(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat out;
    cv::addWeighted(a, 0.5, b, 0.5, 0.0, out);
    return out;
}

cv::Mat MainWindow::applyBilateral(const cv::Mat& src)
{
    cv::Mat out;
    cv::bilateralFilter(src, out, 9, 75.0, 75.0);
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

    if (m_diffStretch || (m_chkStretch && m_chkStretch->isChecked())) {
        cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX);
    }

    cv::applyColorMap(diff, diff, cv::COLORMAP_JET);

    return diff;
}

void MainWindow::updateFrames()
{
    if (!m_cap1.isOpened() || !m_cap2.isOpened()) return;

    m_cap1.read(m_frame1);
    m_cap2.read(m_frame2);

    if (m_frame1.empty() || m_frame2.empty()) return;

    if (m_chkFlipHor2 && m_chkFlipHor2->isChecked()) {
        cv::flip(m_frame2, m_frame2, 0);
    }
    if (m_chkFlipVer2 && m_chkFlipVer2->isChecked()) {
        cv::flip(m_frame2, m_frame2, 1);
    }

    auto toWorkingFormat = [&](cv::Mat& frame) {
        switch (m_colorMode) {
        case ColorMode::GRAY_NATIVE:
            break;
        case ColorMode::GRAY_CV:
            if (frame.channels() == 3)
                cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
            break;
        case ColorMode::COLOR:
            break;
        }
        };
    toWorkingFormat(m_frame1);
    toWorkingFormat(m_frame2);

    double motionRatio = detectMotion(m_frame1);
    m_motionActive = (motionRatio >= m_motionThreshold);

    if (m_motionActive) {
        m_motionIndicator->setText(QString::fromUtf8("\u25CF Motion %1%")
            .arg(static_cast<int>(motionRatio * 100)));
        m_motionIndicator->setStyleSheet("font-weight: bold; color: #ff4444;");
    }
    else {
        m_motionIndicator->setText(QString::fromUtf8("\u25CF Static %1%")
            .arg(static_cast<int>(motionRatio * 100)));
        m_motionIndicator->setStyleSheet("font-weight: bold; color: #44ff44;");
    }

    if (m_motionActive) {
        while (m_frameBuffer1.size() > 2) m_frameBuffer1.pop_front();
        while (m_frameBuffer2.size() > 2) m_frameBuffer2.pop_front();
    }

    m_frameBuffer1.push_back(m_frame1.clone());
    m_frameBuffer2.push_back(m_frame2.clone());

    while (static_cast<int>(m_frameBuffer1.size()) > m_bufferSize)
        m_frameBuffer1.pop_front();
    while (static_cast<int>(m_frameBuffer2.size()) > m_bufferSize)
        m_frameBuffer2.pop_front();

    cv::Mat denoised1 = applyTemporalDenoise();

    cv::Mat denoised2;
    if (m_frameBuffer2.size() == 1) {
        denoised2 = m_frameBuffer2.front().clone();
    }
    else {
        int cvType = (m_frameBuffer2.front().channels() == 3) ? CV_32FC3 : CV_32F;
        cv::Mat acc2;
        m_frameBuffer2.front().convertTo(acc2, cvType);
        for (size_t i = 1; i < m_frameBuffer2.size(); ++i) {
            cv::Mat tmp;
            m_frameBuffer2[i].convertTo(tmp, cvType);
            acc2 += tmp;
        }
        acc2 /= static_cast<double>(m_frameBuffer2.size());
        acc2.convertTo(denoised2, m_frameBuffer2.front().type());
    }

    m_frame1 = denoised1;
    m_frame2 = denoised2;

    if (m_chkFusion->isChecked() && m_isAligned && !m_eccWarpMatrix.empty()) {
        try {
            cv::Mat aligned2;
            cv::warpAffine(m_frame2, aligned2, m_eccWarpMatrix, m_frame1.size(),
                cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
            m_frame1 = fuseCameras(m_frame1, aligned2);
        }
        catch (const cv::Exception& e) {
            std::cerr << "Fusion warp error: " << e.what() << std::endl;
        }
    }

    if (m_chkBilateral->isChecked()) {
        m_frame1 = applyBilateral(m_frame1);
        m_frame2 = applyBilateral(m_frame2);
    }

    m_lastFocus1 = calculateFocus(m_frame1);
    m_lastFocus2 = calculateFocus(m_frame2);
    m_frameCount++;

    m_seriesCam1->append(m_frameCount, m_lastFocus1);
    m_seriesCam2->append(m_frameCount, m_lastFocus2);

    if (m_seriesCam1->count() > m_maxHistory) m_seriesCam1->remove(0);
    if (m_seriesCam2->count() > m_maxHistory) m_seriesCam2->remove(0);

    auto axisX = m_chart->axes(Qt::Horizontal).first();
    axisX->setRange(std::max(0LL, m_frameCount - m_maxHistory), m_frameCount);

    double maxFocus = 0.0;
    for (const auto& p : m_seriesCam1->points()) maxFocus = std::max(maxFocus, p.y());
    for (const auto& p : m_seriesCam2->points()) maxFocus = std::max(maxFocus, p.y());

    auto axisY = m_chart->axes(Qt::Vertical).first();
    if (maxFocus > 10.0) {
        axisY->setRange(0, maxFocus * 1.1);
    }

    m_statusBar->showMessage(QString("Frames processed: %1 | Buffer: %2/%3")
        .arg(m_frameCount)
        .arg(m_frameBuffer1.size())
        .arg(m_bufferSize));

    updateView();
}

void MainWindow::updateView()
{
    if (m_frame1.empty() || m_frame2.empty()) return;

    cv::Mat f1 = m_frame1.clone();
    cv::Mat f2 = m_frame2.clone();
    cv::Mat alignedF2 = f2;

    if (m_chkAlign->isChecked() && m_isAligned && !m_eccWarpMatrix.empty()) {
        try {
            cv::warpAffine(f2, alignedF2, m_eccWarpMatrix, f1.size(),
                cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
        }
        catch (const cv::Exception& e) {
            std::cerr << "Warp error: " << e.what() << std::endl;
            alignedF2 = f2;
        }
    }

    bool showPeaks = m_btnPeakIntensities->isChecked();
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

    auto drawOSD = [](cv::Mat& img, const std::string& camName, double focusScore) {
        if (img.channels() == 1) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        std::string text = camName + " Focus: " + std::to_string(static_cast<int>(focusScore));
        cv::putText(img, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        };

    if (!m_isDiffMode)
    {
        m_resultView->hide();
        m_splitter->show();

        if (f1.channels() == 1) cv::cvtColor(f1, f1, cv::COLOR_GRAY2BGR);
        if (alignedF2.channels() == 1) cv::cvtColor(alignedF2, alignedF2, cv::COLOR_GRAY2BGR);

        drawOSD(f1, "CAM1", m_lastFocus1);
        drawOSD(alignedF2, "CAM2", m_lastFocus2);

        if (showPeaks) {
            drawTarget(f1, maxLoc1, cv::Scalar(0, 255, 255), "Max 1");
            drawTarget(alignedF2, maxLoc2, cv::Scalar(0, 255, 255), "Max 2");
        }

        displayMat(m_view1, f1);
        displayMat(m_view2, alignedF2);
    }
    else
    {
        m_splitter->hide();
        m_resultView->show();

        cv::Mat diff = applyDiffView(f1, alignedF2);

        cv::putText(diff, "Difference View", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        if (showPeaks) {
            drawTarget(diff, maxLoc1, cv::Scalar(0, 255, 255), "P1");
            drawTarget(diff, maxLoc2, cv::Scalar(255, 0, 255), "P2");
        }

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

    cv::Mat f1, f2;
    for (int i = 0; i < 5; ++i) {
        m_cap1.read(f1);
        m_cap2.read(f2);
        QApplication::processEvents();
    }

    if (f1.empty() || f2.empty()) {
        m_statusBar->showMessage("Calibration failed: Empty frames.", 3000);
        return;
    }

    cv::Mat gray1, gray2;
    if (f1.channels() == 3) cv::cvtColor(f1, gray1, cv::COLOR_BGR2GRAY);
    else gray1 = f1.clone();
    if (f2.channels() == 3) cv::cvtColor(f2, gray2, cv::COLOR_BGR2GRAY);
    else gray2 = f2.clone();

    if (calculateFocus(gray1) < 5.0) {
        m_statusBar->showMessage("Error: Too dark for calibration!", 4000);
        return;
    }

    m_statusBar->showMessage("Calibrating (ECC)... please wait.");
    QApplication::processEvents();

    cv::Mat warpMatrix = cv::Mat::eye(2, 3, CV_32F);
    cv::TermCriteria criteria(
        cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
        100, 1e-4);

    try {
        cv::findTransformECC(gray1, gray2, warpMatrix,
            cv::MOTION_AFFINE, criteria);
        m_eccWarpMatrix = warpMatrix;
        m_isAligned = true;
        m_statusBar->showMessage("Alignment OK (ECC)!", 3000);
    }
    catch (const cv::Exception& e) {
        m_eccWarpMatrix.release();
        m_isAligned = false;
        m_statusBar->showMessage(
            "ECC failed: " + QString::fromStdString(e.what()), 4000);
    }
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

void MainWindow::displayMat(QLabel* label, const cv::Mat& mat)
{
    if (mat.empty() || label == nullptr) return;

    QImage::Format format = QImage::Format_RGB888;
    cv::Mat tempMat;

    if (mat.channels() == 1) {
        format = QImage::Format_Grayscale8;
        tempMat = mat.clone();
    }
    else if (mat.channels() == 3) {
        cv::cvtColor(mat, tempMat, cv::COLOR_BGR2RGB);
    }
    else {
        return;
    }

    QImage img(tempMat.data, tempMat.cols, tempMat.rows, static_cast<int>(tempMat.step), format);
    label->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::showPeakIntensities()
{}

void MainWindow::saveDiffSnapshot()
{
    if (m_lastDiffResult.empty()) {
        m_statusBar->showMessage("No difference image available to save.", 3000);
        return;
    }

    QString metricsDir = QCoreApplication::applicationDirPath() + "/metrics";
    QDir().mkpath(metricsDir);

    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    QString filePath = metricsDir + "/diff_snapshot_" + timestamp + ".png";

    bool success = cv::imwrite(filePath.toStdString(), m_lastDiffResult);

    if (success) {
        m_statusBar->showMessage("Saved snapshot: " + filePath, 4000);
    }
    else {
        m_statusBar->showMessage("Error saving snapshot to: " + filePath, 4000);
    }
}