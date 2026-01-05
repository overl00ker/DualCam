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
#include <QProcess>
#include <QRegularExpression>
#include <QDebug>
#include <QtCharts/QValueAxis>
#include <QtCharts/QChart>
#include <QtCharts/QLegend>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <vector>
#include <cmath>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    m_orb = cv::ORB::create(1000);
    m_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
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
    setWindowTitle("Stereo Camera Calibrate");
    setGeometry(100, 100, 1280, 720);

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

    m_comboViewMode = new QComboBox(this);
    m_comboViewMode->addItems({ "Dual View", "Difference View" });
    connect(m_comboViewMode, &QComboBox::currentTextChanged, this, &MainWindow::updateView);
    controlsLayout->addWidget(m_comboViewMode);

    m_chkAlign = new QCheckBox("Align Frames", this);
    connect(m_chkAlign, &QCheckBox::stateChanged, this, &MainWindow::updateView);
    controlsLayout->addWidget(m_chkAlign);

    m_btnCalibrateAlign = new QPushButton("Calibrate Alignment", this);
    connect(m_btnCalibrateAlign, &QPushButton::clicked, this, &MainWindow::calibrateAlignment);
    controlsLayout->addWidget(m_btnCalibrateAlign);

    controlsLayout->addStretch();
    liveLayout->addLayout(controlsLayout);

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

    // --- Tab 2: Focus Analysis ---
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
    axisY->setTitleText("Focus Score (Lap Var)");
    axisY->setRange(0, 10000);
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
    historyLayout->addWidget(new QLabel("Graph History Length:", this));
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
        "video/x-raw, width=%2, height=%3, framerate=%4/1 ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1")
        .arg(cameraId)
        .arg(width)
        .arg(height)
        .arg(fps)
        .toStdString();
}

void MainWindow::openCameras()
{
    m_statusBar->showMessage("Detecting cameras...");
    QApplication::processEvents();

    QStringList camPaths = getLibCameraIds();

    if (camPaths.size() < 2) {
        // Fallback to V4L2/DSHOW if not enough libcameras found (e.g., testing on PC)
        m_statusBar->showMessage("Warning: < 2 libcameras found. Trying default indices.", 3000);

#ifdef Q_OS_LINUX
        // Try V4L2 fallback
        m_cap1.open(0, cv::CAP_V4L2);
        m_cap2.open(1, cv::CAP_V4L2);
#else
        m_cap1.open(0, cv::CAP_DSHOW);
        m_cap2.open(1, cv::CAP_DSHOW);
#endif
    }
    else {
        // Use GStreamer pipelines with specific camera paths
        std::string pipe1 = makeGStreamerPipeline(camPaths[0], 640, 480, 30);
        std::string pipe2 = makeGStreamerPipeline(camPaths[1], 640, 480, 30);

        qDebug() << "Opening Pipeline 1:" << QString::fromStdString(pipe1);
        qDebug() << "Opening Pipeline 2:" << QString::fromStdString(pipe2);

        m_cap1.open(pipe1, cv::CAP_GSTREAMER);
        m_cap2.open(pipe2, cv::CAP_GSTREAMER);
    }

    if (!m_cap1.isOpened() || !m_cap2.isOpened())
    {
        m_statusBar->showMessage("Error: Could not open one or both cameras.", 5000);
        if (m_cap1.isOpened()) m_cap1.release();
        if (m_cap2.isOpened()) m_cap2.release();
        return;
    }

    m_camerasOpen = true;
    m_isAligned = false;
    m_homography.release();
    m_btnToggleCameras->setText("Close Cameras");

    m_frameCount = 0;
    m_seriesCam1->clear();
    m_seriesCam2->clear();
    m_chart->axes(Qt::Horizontal).first()->setRange(0, m_maxHistory);
    m_chart->axes(Qt::Vertical).first()->setRange(0, 10000);

    m_timer->start(33); // ~30 FPS
    m_statusBar->showMessage(QString("Cameras opened. Mode: %1").arg(camPaths.size() >= 2 ? "Libcamera/GStreamer" : "V4L2/Legacy"), 3000);
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
    m_statusBar->showMessage("Cameras closed.", 2000);
}

void MainWindow::updateFrames()
{
    if (!m_cap1.isOpened() || !m_cap2.isOpened()) return;

    m_cap1.read(m_frame1);
    m_cap2.read(m_frame2);

    if (m_frame1.empty() || m_frame2.empty())
    {
        m_statusBar->showMessage("Error: Dropped frame or stream ended.", 1000);
        return;
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
    axisY->setRange(0, std::max(100.0, maxFocus * 1.1));

    m_statusBar->showMessage(QString("Focus Cam1: %1 | Focus Cam2: %2")
        .arg(static_cast<int>(focus1))
        .arg(static_cast<int>(focus2)));

    updateView();
}

void MainWindow::updateView()
{
    if (m_frame1.empty() || m_frame2.empty()) return;

    cv::Mat f1 = m_frame1.clone();
    cv::Mat f2 = m_frame2.clone();
    cv::Mat alignedF2 = f2;

    if (m_chkAlign->isChecked() && m_isAligned && !m_homography.empty()) {
        try {
            cv::warpPerspective(f2, alignedF2, m_homography, f1.size());
        }
        catch (const cv::Exception& e) {
            std::cerr << "Warp failed: " << e.what() << std::endl;
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
        cv::Mat diff;

        if (alignedF2.size() != f1.size()) cv::resize(alignedF2, alignedF2, f1.size());

        cv::absdiff(f1, alignedF2, diff);
        cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);
        cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX);

        // Apply heatmap for better visualization of differences
        cv::applyColorMap(diff, diff, cv::COLORMAP_JET);

        displayMat(m_resultView, diff);
    }
}

void MainWindow::calibrateAlignment()
{
    if (!m_camerasOpen) {
        m_statusBar->showMessage("Cameras are closed. Cannot calibrate.", 3000);
        return;
    }

    // Capture fresh frames for calibration to ensure sync
    cv::Mat f1, f2;
    m_cap1.read(f1);
    m_cap2.read(f2);

    if (f1.empty() || f2.empty()) {
        m_statusBar->showMessage("Calibration failed: Empty frames.", 3000);
        return;
    }

    m_statusBar->showMessage("Calibrating alignment... please wait.");
    QApplication::processEvents();

    cv::Mat h;
    if (computeHomography(f1, f2, h)) {
        m_homography = h;
        m_isAligned = true;
        m_statusBar->showMessage("Alignment calibrated successfully.", 3000);
    }
    else {
        m_homography.release();
        m_isAligned = false;
        m_statusBar->showMessage("Calibration failed: Not enough matches found.", 5000);
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

bool MainWindow::computeHomography(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& outHomography)
{
    if (img1.empty() || img2.empty()) return false;

    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    m_orb->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    m_orb->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);

    if (descriptors1.empty() || descriptors2.empty() || descriptors1.rows < 4 || descriptors2.rows < 4) {
        return false;
    }

    if (descriptors1.type() != CV_8U) descriptors1.convertTo(descriptors1, CV_8U);
    if (descriptors2.type() != CV_8U) descriptors2.convertTo(descriptors2, CV_8U);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    m_matcher->knnMatch(descriptors2, descriptors1, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() > 1 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    if (good_matches.size() < 10) return false;

    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < good_matches.size(); i++) {
        points2.push_back(keypoints2[good_matches[i].queryIdx].pt);
        points1.push_back(keypoints1[good_matches[i].trainIdx].pt);
    }

    cv::Mat h = cv::findHomography(points2, points1, cv::RANSAC, 5.0);
    if (h.empty()) return false;

    outHomography = h;
    return true;
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