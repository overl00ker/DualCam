#include "MainWindow.h"
#include <algorithm>
#include <cmath>
#include <iostream>

#include <QMessageBox>
#include <QApplication> 
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QProcess>
#include <opencv2/opencv.hpp>

MainWindow::MainWindow(const CaptureParams& p0,
                       const CaptureParams& p1,
                       QWidget* parent)
    : QMainWindow(parent),
      p0_(p0), p1_(p1)
{
    std::cout << "MainWindow constructor started" << std::endl;

    CAP_W_ = p0_.width;
    CAP_H_ = p0_.height;
    DISP_W_ = CAP_W_ * 2;
    DISP_H_ = CAP_H_ * 2;

    // central widget
    QWidget* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);
    viewLabel_ = new QLabel;
    viewLabel_->setFixedSize(DISP_W_, DISP_H_);
    layout->addWidget(viewLabel_);
    setCentralWidget(central);
    setFixedSize(DISP_W_, DISP_H_);

    // gauss mask for sharpness weight
    cv::Mat ky = cv::getGaussianKernel(CAP_H_, CAP_H_ / 6.0, CV_64F);
    cv::Mat kx = cv::getGaussianKernel(CAP_W_, CAP_W_ / 6.0, CV_64F);
    gaussMask_ = ky * kx.t();
    gaussMask_ /= cv::sum(gaussMask_)[0];

    createMenus();
    openCams();

    std::cout << "cam0 opened: " << cam0_.isOpened() << std::endl;
    std::cout << "cam1 opened: " << cam1_.isOpened() << std::endl;

    if (!cam0_.isOpened() || !cam1_.isOpened()) {
        std::cerr << "\u274C Cameras failed to open. Exiting." << std::endl;
        QMessageBox::critical(this, "Camera Error", "Failed to open one or both cameras.");
        return;
    }

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::updateFrame);
    timer_->start(1000 / std::max(1, p0_.fps));

    std::cout << "MainWindow initialized successfully" << std::endl;
}

MainWindow::~MainWindow()
{
    closeCams();
}

/* ------------------ menu ------------------ */
void MainWindow::createMenus()
{
    menuBar_ = new QMenuBar(this);
    setMenuBar(menuBar_);

    // Camera
    cameraMenu_ = menuBar_->addMenu(tr("&Camera"));
    actOneCam_ = cameraMenu_->addAction(tr("1 camera"), this, &MainWindow::setOneCamera);
    actOneCam_->setCheckable(true);
    actTwoCam_ = cameraMenu_->addAction(tr("2 cameras"), this, &MainWindow::setTwoCameras);
    actTwoCam_->setCheckable(true);
    actTwoCam_->setChecked(true);

    cameraMenu_->addSeparator();
    cameraMenu_->addAction(tr("640x480"), this, &MainWindow::setResolution640x480);
    cameraMenu_->addAction(tr("1280x720"), this, &MainWindow::setResolution1280x720);
    cameraMenu_->addSeparator();
    cameraMenu_->addAction(tr("30 FPS"), this, &MainWindow::setFps30);
    cameraMenu_->addAction(tr("60 FPS"), this, &MainWindow::setFps60);

    // Advance
    advanceMenu_ = menuBar_->addMenu(tr("&Advance"));
    actDiff_ = advanceMenu_->addAction(tr("Show difference"));
    actDiff_->setCheckable(true);
    actDiff_->setChecked(showDiff_);
    connect(actDiff_, &QAction::toggled, this, &MainWindow::toggleDifference);

    actGraph_ = advanceMenu_->addAction(tr("Show graph"));
    actGraph_->setCheckable(true);
    actGraph_->setChecked(showGraph_);
    connect(actGraph_, &QAction::toggled, this, &MainWindow::toggleGraph);

    // Window
    windowMenu_ = menuBar_->addMenu(tr("&Window"));
    windowMenu_->addAction(tr("Fullscreen"),      QKeySequence("Ctrl+F"),
                       this, &MainWindow::enterFullScreen);
    windowMenu_->addAction(tr("Exit fullscreen"), QKeySequence("Esc"),
                       this, &MainWindow::exitFullScreen);
    windowMenu_->addSeparator();
	windowMenu_->addAction(tr("Refresh"),         QKeySequence("Ctrl+R"),
                       this, &MainWindow::refreshApp);
    windowMenu_->addAction(tr("Restart"), this, &MainWindow::restartApp);
}

/* ------------------ camera open/close ------------------ */
void MainWindow::openCams()
{
    std::string gst0 = "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@88000/imx296@1a ! video/x-raw,width="
                        + std::to_string(CAP_W_) + ",height=" + std::to_string(CAP_H_) +
                        ",format=YUY2 ! videoconvert ! appsink";
    std::string gst1 = "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@80000/imx296@1a ! video/x-raw,width="
                        + std::to_string(CAP_W_) + ",height=" + std::to_string(CAP_H_) +
                        ",format=YUY2 ! videoconvert ! appsink";

    cam0_.open(gst0, cv::CAP_GSTREAMER);
    cam1_.open(gst1, cv::CAP_GSTREAMER);
}

void MainWindow::closeCams()
{
    cam0_.release();
    cam1_.release();
}

/* ------------------ utility ------------------ */
cv::Mat MainWindow::ensureGray(const cv::Mat& src)
{
    if (src.channels() == 1) return src;
    cv::Mat g;
    cv::cvtColor(src, g, cv::COLOR_BGR2GRAY);
    return g;
}

double MainWindow::computeSharp(const cv::Mat& gray)
{
    cv::Mat lap, absL, weighted;
    cv::Laplacian(gray, lap, CV_64F);
    cv::absdiff(lap, cv::Scalar::all(0), absL);
    cv::multiply(absL, gaussMask_, weighted, 1.0, CV_64F);
    return cv::sum(weighted)[0];
}

void MainWindow::drawOverlay(cv::Mat& img, const std::string& text)
{
    cv::putText(img, text, cv::Point(5,20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0,0,255), 2);
}

cv::Mat MainWindow::makeGraphImage(int w, int h)
{
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(0,0,0));
    if (history_.size() < 2) return img;
    double dx = double(w) / (MAX_HISTORY_ - 1);
    for (int i = 1; i < (int)history_.size(); ++i)
    {
        int x0 = int((i-1)*dx), y0 = h - history_[i-1].first  * h / 100;
        int x1 = int(i*dx),     y1 = h - history_[i].first    * h / 100;
        cv::line(img, {x0,y0}, {x1,y1}, {0,255,0}, 2);
        int y0b = h - history_[i-1].second * h / 100;
        int y1b = h - history_[i].second   * h / 100;
        cv::line(img, {x0,y0b}, {x1,y1b}, {0,0,255}, 2);
    }
    return img;
}

/* ------------------ frame update ------------------ */
void MainWindow::updateFrame()
{
    cv::Mat f0, f1;
    bool g0 = cam0_.grab();
    bool g1 = cam1_.grab();
    if (g0) cam0_.retrieve(f0); else cam0_.read(f0);
    if (g1) cam1_.retrieve(f1); else cam1_.read(f1);
    if (f0.empty() || f1.empty()) return;

    cv::Mat gray0 = ensureGray(f0);
    cv::Mat gray1 = ensureGray(f1);

    if (gray0.cols != CAP_W_ || gray0.rows != CAP_H_)
        cv::resize(gray0, gray0, cv::Size(CAP_W_, CAP_H_));
    if (gray1.cols != CAP_W_ || gray1.rows != CAP_H_)
        cv::resize(gray1, gray1, cv::Size(CAP_W_, CAP_H_));

    double s0 = computeSharp(gray0);
    double s1 = computeSharp(gray1);
    double maxSharp = std::max(s0, s1);
    int pct0 = maxSharp > 0 ? int(s0 / maxSharp * 100) : 0;
    int pct1 = maxSharp > 0 ? int(s1 / maxSharp * 100) : 0;

    if (!freezeHistory_)
    {
        if (history_.empty() ||
            std::abs(pct0 - history_.back().first)  >= 1 ||
            std::abs(pct1 - history_.back().second) >= 1)
        {
            history_.emplace_back(pct0, pct1);
            if ((int)history_.size() > MAX_HISTORY_) history_.pop_front();
        }
    }

    cv::Mat disp0, disp1;
    cv::cvtColor(gray0, disp0, cv::COLOR_GRAY2BGR);
    cv::cvtColor(gray1, disp1, cv::COLOR_GRAY2BGR);
    drawOverlay(disp0, std::to_string(pct0) + "%");
    drawOverlay(disp1, std::to_string(pct1) + "%");

    cv::Mat diff, diffC;
    cv::absdiff(gray0, gray1, diff);
    cv::cvtColor(diff, diffC, cv::COLOR_GRAY2BGR);

    cv::Mat graph = (showGraph_ ? makeGraphImage(CAP_W_, CAP_H_)
                                : cv::Mat::zeros(CAP_H_, CAP_W_, CV_8UC3));

    cv::Mat canvas;
    if (mode_ == ViewMode::Analytics4Q)
    {
        canvas = cv::Mat(DISP_H_, DISP_W_, CV_8UC3, cv::Scalar(0,0,0));
        disp0.copyTo(canvas(cv::Rect(0,          0,      CAP_W_, CAP_H_)));
        disp1.copyTo(canvas(cv::Rect(CAP_W_,     0,      CAP_W_, CAP_H_)));
        if (showDiff_)
            diffC.copyTo(canvas(cv::Rect(0,      CAP_H_, CAP_W_, CAP_H_)));
        graph.copyTo(canvas(cv::Rect(CAP_W_,     CAP_H_, CAP_W_, CAP_H_)));
    }
    else // SideBySide
    {
        canvas = cv::Mat(CAP_H_, CAP_W_*2, CV_8UC3);
        disp0.copyTo(canvas(cv::Rect(0,0,CAP_W_,CAP_H_)));
        disp1.copyTo(canvas(cv::Rect(CAP_W_,0,CAP_W_,CAP_H_)));
    }

    QImage img(canvas.data, canvas.cols, canvas.rows, int(canvas.step), QImage::Format_BGR888);
    viewLabel_->setPixmap(QPixmap::fromImage(img));
}

/* ------------------ key events ------------------ */
void MainWindow::keyPressEvent(QKeyEvent* event)
{
    switch (event->key())
    {
    case Qt::Key_F: freezeHistory_ = !freezeHistory_; break;
    case Qt::Key_T: showGraph_    = !showGraph_;     actGraph_->setChecked(showGraph_); break;
    case Qt::Key_S:
        mode_ = (mode_ == ViewMode::Analytics4Q ? ViewMode::SideBySide
                                                : ViewMode::Analytics4Q);
        break;
    case Qt::Key_Escape: close(); break;
    default: QMainWindow::keyPressEvent(event);
    }
}

/* ------------------ slots ------------------ */
// Camera
void MainWindow::setOneCamera()
{
    desiredCamCount_ = 1;
    actOneCam_->setChecked(true);
    actTwoCam_->setChecked(false);
    // not implemented: actually closing second camera feed yet
}

void MainWindow::setTwoCameras()
{
    desiredCamCount_ = 2;
    actOneCam_->setChecked(false);
    actTwoCam_->setChecked(true);
    // not implemented: open second camera if closed
}

void MainWindow::setResolution640x480()
{
    p0_.width = p1_.width = CAP_W_ = 640;
    p0_.height = p1_.height = CAP_H_ = 480;
    closeCams(); openCams();
}

void MainWindow::setResolution1280x720()
{
    p0_.width = p1_.width = CAP_W_ = 1280;
    p0_.height = p1_.height = CAP_H_ = 720;
    closeCams(); openCams();
}

void MainWindow::setFps30()
{
    p0_.fps = p1_.fps = 30;
    timer_->setInterval(1000/30);
}

void MainWindow::setFps60()
{
    p0_.fps = p1_.fps = 60;
    timer_->setInterval(1000/60);
}

// Advance
void MainWindow::toggleDifference(bool checked)
{
    showDiff_ = checked;
}

void MainWindow::toggleGraph(bool checked)
{
    showGraph_ = checked;
}

// Window
void MainWindow::enterFullScreen()
{
    showFullScreen();
}

void MainWindow::exitFullScreen()
{
    showNormal();
}

void MainWindow::refreshApp()
{
    history_.clear();
}

void MainWindow::restartApp()
{
    QProcess::startDetached(QCoreApplication::applicationFilePath(),
                            QCoreApplication::arguments());
    QCoreApplication::quit();
}
