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
#include <QTabWidget>
#include <QSpinBox>
#include <QSlider>
#include <QProcess>
#include <QRegularExpression>
#include <QDebug>
#include <QtCharts/QValueAxis>
#include <QtCharts/QChart>
#include <QtCharts/QLegend>

#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include <iostream>
#include <vector>
#include <cmath>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
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
    setWindowTitle("DualCam");
    setGeometry(100, 100, 1280, 850);

    m_centralWidget = new QWidget(this);
    setCentralWidget(m_centralWidget);

    QVBoxLayout* mainLayout = new QVBoxLayout(m_centralWidget);
    m_tabWidget = new QTabWidget(this);
    mainLayout->addWidget(m_tabWidget);

    QWidget* liveTab = new QWidget();
    QVBoxLayout* liveLayout = new QVBoxLayout(liveTab);

    QHBoxLayout* controlsLayout = new QHBoxLayout();
    m_btnToggleCameras = new QPushButton("Open Cameras", this);
    connect(m_btnToggleCameras, &QPushButton::clicked, this, [this]()
        {
            if (m_camerasOpen) closeCameras();
            else openCameras();
        });
    controlsLayout->addWidget(m_btnToggleCameras);

    m_chkFlipVer2 = new QCheckBox("Flip 2ndV", this);
    controlsLayout->addWidget(m_chkFlipVer2);
    m_chkFlipHor2 = new QCheckBox("Flip 2ndH", this);
    controlsLayout->addWidget(m_chkFlipHor2);

    m_comboViewMode = new QComboBox(this);
    m_comboViewMode->addItems({ "Dual View", "Difference View" });
    connect(m_comboViewMode, &QComboBox::currentTextChanged, this, &MainWindow::updateView);
    controlsLayout->addWidget(m_comboViewMode);

    QLabel* colorLabel = new QLabel("Color:", this);
    m_comboColorMode = new QComboBox(this);
    m_comboColorMode->addItems({ "Gray Native", "Gray CV", "Color" });
    m_comboColorMode->setCurrentIndex(1);
    m_comboColorMode->setToolTip("Gray Native: expect GRAY8 from camera\n"
                                  "Gray CV: capture BGR, convert to grayscale\n"
                                  "Color: keep BGR as-is");
    connect(m_comboColorMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        m_colorMode = static_cast<ColorMode>(index);
        m_frameBuffer1.clear();
        m_frameBuffer2.clear();
        if (m_colorMode == ColorMode::COLOR) {
            m_statusBar->showMessage("Color mode: ECC calibration uses grayscale internally.", 4000);
        }
    });
    controlsLayout->addWidget(colorLabel);
    controlsLayout->addWidget(m_comboColorMode);

    m_chkAlign = new QCheckBox("Align Frames", this);
    connect(m_chkAlign, &QCheckBox::stateChanged, this, &MainWindow::updateView);
    controlsLayout->addWidget(m_chkAlign);

    m_btnCalibrateAlign = new QPushButton("Calibrate Alignment", this);
    connect(m_btnCalibrateAlign, &QPushButton::clicked, this, &MainWindow::calibrateAlignment);
    controlsLayout->addWidget(m_btnCalibrateAlign);

    controlsLayout->addStretch();
    liveLayout->addLayout(controlsLayout);

    QHBoxLayout* denoiseLayout = new QHBoxLayout();

    QLabel* bufferTitle = new QLabel("Buffer:", this);
    m_bufferSlider = new QSlider(Qt::Horizontal, this);
    m_bufferSlider->setRange(1, 30);
    m_bufferSlider->setValue(m_bufferSize);
    m_bufferSlider->setTickPosition(QSlider::TicksBelow);
    m_bufferSlider->setTickInterval(5);
    m_bufferSlider->setFixedWidth(160);

    m_bufferLabel = new QLabel(QString::number(m_bufferSize), this);
    m_bufferLabel->setFixedWidth(30);
    m_bufferLabel->setStyleSheet("font-weight: bold; font-size: 14px; color: cyan;");

    connect(m_bufferSlider, &QSlider::valueChanged, this, [this](int value) {
        m_bufferSize = value;
        m_bufferLabel->setText(QString::number(value));
    });

    denoiseLayout->addWidget(bufferTitle);
    denoiseLayout->addWidget(m_bufferSlider);
    denoiseLayout->addWidget(m_bufferLabel);

    QLabel* motionTitle = new QLabel("Motion Threshold:", this);
    m_motionThresholdSlider = new QSlider(Qt::Horizontal, this);
    m_motionThresholdSlider->setRange(1, 50);
    m_motionThresholdSlider->setValue(static_cast<int>(m_motionThreshold * 100));
    m_motionThresholdSlider->setTickPosition(QSlider::TicksBelow);
    m_motionThresholdSlider->setTickInterval(5);
    m_motionThresholdSlider->setFixedWidth(160);

    m_motionThresholdLabel = new QLabel(QString("%1%").arg(m_motionThreshold * 100, 0, 'f', 0), this);
    m_motionThresholdLabel->setFixedWidth(40);
    m_motionThresholdLabel->setStyleSheet("font-weight: bold; font-size: 14px; color: cyan;");

    connect(m_motionThresholdSlider, &QSlider::valueChanged, this, [this](int value) {
        m_motionThreshold = static_cast<double>(value) / 100.0;
        m_motionThresholdLabel->setText(QString("%1%").arg(value));
    });

    denoiseLayout->addWidget(motionTitle);
    denoiseLayout->addWidget(m_motionThresholdSlider);
    denoiseLayout->addWidget(m_motionThresholdLabel);

    m_chkFusion = new QCheckBox("Camera Fusion", this);
    m_chkFusion->setToolTip("Blend both cameras for +3 dB SNR (requires calibrated alignment)");
    denoiseLayout->addWidget(m_chkFusion);

    m_chkBilateral = new QCheckBox("Bilateral Filter", this);
    m_chkBilateral->setToolTip("Apply bilateral filter on top for edge-preserving smoothing");
    denoiseLayout->addWidget(m_chkBilateral);

    m_motionIndicator = new QLabel(QString::fromUtf8("\u25CF Idle"), this);
    m_motionIndicator->setStyleSheet("font-weight: bold; font-size: 13px; color: gray;");
    m_motionIndicator->setFixedWidth(110);
    denoiseLayout->addWidget(m_motionIndicator);

    denoiseLayout->addStretch();
    liveLayout->addLayout(denoiseLayout);

    QHBoxLayout* diffLayout = new QHBoxLayout();

    QLabel* noiseFloorTitle = new QLabel("Noise Floor:", this);
    m_noiseFloorSlider = new QSlider(Qt::Horizontal, this);
    m_noiseFloorSlider->setRange(0, 50);
    m_noiseFloorSlider->setValue(m_noiseFloor);
    m_noiseFloorSlider->setTickPosition(QSlider::TicksBelow);
    m_noiseFloorSlider->setTickInterval(5);
    m_noiseFloorSlider->setFixedWidth(160);

    m_noiseFloorLabel = new QLabel(QString::number(m_noiseFloor), this);
    m_noiseFloorLabel->setFixedWidth(30);
    m_noiseFloorLabel->setStyleSheet("font-weight: bold; font-size: 14px; color: cyan;");

    connect(m_noiseFloorSlider, &QSlider::valueChanged, this, [this](int value) {
        m_noiseFloor = value;
        m_noiseFloorLabel->setText(QString::number(value));
    });

    m_chkStretch = new QCheckBox("Stretch Range", this);
    m_chkStretch->setToolTip("Normalize remaining diff to full 0-255 range for visibility");

    diffLayout->addWidget(noiseFloorTitle);
    diffLayout->addWidget(m_noiseFloorSlider);
    diffLayout->addWidget(m_noiseFloorLabel);
    diffLayout->addWidget(m_chkStretch);
    diffLayout->addStretch();
    liveLayout->addLayout(diffLayout);

    noiseFloorTitle->setObjectName("diffCtrl");
    m_noiseFloorSlider->setObjectName("diffCtrl");
    m_noiseFloorLabel->setObjectName("diffCtrl");
    m_chkStretch->setObjectName("diffCtrl");
    noiseFloorTitle->hide();
    m_noiseFloorSlider->hide();
    m_noiseFloorLabel->hide();
    m_chkStretch->hide();

    connect(m_comboViewMode, &QComboBox::currentTextChanged, this, [this](const QString& text) {
        bool isDiff = (text == "Difference View");
        m_noiseFloorSlider->setVisible(isDiff);
        m_noiseFloorLabel->setVisible(isDiff);
        m_chkStretch->setVisible(isDiff);
        for (auto* child : m_centralWidget->findChildren<QLabel*>("diffCtrl")) {
            child->setVisible(isDiff);
        }
    });

    m_splitter = new QSplitter(Qt::Horizontal, this);

    m_view1 = new QLabel("Camera 1", this);
    m_view1->setScaledContents(true);
    m_view1->setAlignment(Qt::AlignCenter);
    m_view1->setMinimumSize(320, 240);
    m_view1->setFrameShape(QFrame::Box);
    m_splitter->addWidget(m_view1);

    m_view2 = new QLabel("Camera 2", this);
    m_view2->setScaledContents(true);
    m_view2->setAlignment(Qt::AlignCenter);
    m_view2->setMinimumSize(320, 240);
    m_view2->setFrameShape(QFrame::Box);
    m_splitter->addWidget(m_view2);

    liveLayout->addWidget(m_splitter, 1);

    m_resultView = new QLabel("Result View", this);
    m_resultView->setScaledContents(true);
    m_resultView->setAlignment(Qt::AlignCenter);
    m_resultView->setMinimumSize(640, 480);
    m_resultView->setFrameShape(QFrame::Box);
    liveLayout->addWidget(m_resultView, 1);
    m_resultView->hide();

    m_tabWidget->addTab(liveTab, "Live View");

    QWidget* focusTab = new QWidget();
    QVBoxLayout* focusLayout = new QVBoxLayout(focusTab);

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

    m_tabWidget->addTab(focusTab, "Live Focus Analysis");

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
        m_statusBar->showMessage("Warning: < 2 libcameras found.", 3000);
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
    m_motionIndicator->setStyleSheet("font-weight: bold; font-size: 13px; color: gray;");

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

    cv::Mat fgMask;
    m_bgSubtractor->apply(gray, fgMask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);

    int nonZero = cv::countNonZero(fgMask);
    return static_cast<double>(nonZero) / (frame.rows * frame.cols);
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
        m_motionIndicator->setStyleSheet("font-weight: bold; font-size: 13px; color: #ff4444;");
    }
    else {
        m_motionIndicator->setText(QString::fromUtf8("\u25CF Static %1%")
            .arg(static_cast<int>(motionRatio * 100)));
        m_motionIndicator->setStyleSheet("font-weight: bold; font-size: 13px; color: #44ff44;");
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

    double focus1 = calculateFocus(m_frame1);
    double focus2 = calculateFocus(m_frame2);
    m_frameCount++;

    m_seriesCam1->append(m_frameCount, focus1);
    m_seriesCam2->append(m_frameCount, focus2);

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

    m_statusBar->showMessage(QString("F1: %1 | F2: %2 | Buf: %3/%4 | Motion: %5%")
        .arg(static_cast<int>(focus1))
        .arg(static_cast<int>(focus2))
        .arg(m_frameBuffer1.size())
        .arg(m_bufferSize)
        .arg(static_cast<int>(motionRatio * 100)));

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

    if (m_comboViewMode->currentText() == "Dual View")
    {
        m_resultView->hide();
        m_splitter->show();
        displayMat(m_view1, f1);
        displayMat(m_view2, alignedF2);
    }
    else
    {
        m_splitter->hide();
        m_resultView->show();

        cv::Mat diff = applyDiffView(f1, alignedF2);
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