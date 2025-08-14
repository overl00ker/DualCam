#include "MainWindow.h"

#include <QVBoxLayout>
#include <QImage>
#include <QPixmap>
#include <QApplication>
#include <QProcess>
#include <QMessageBox>
#include <QCoreApplication>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

MainWindow::MainWindow(const CaptureParams& cam0,
    const CaptureParams& cam1,
    QWidget* parent)
    : QMainWindow(parent),
    p0_(cam0), p1_(cam1),
    CAP_W_{ std::max(p0_.width,  p1_.width) },
    CAP_H_{ std::max(p0_.height, p1_.height) },
    DISP_W_{ CAP_W_ * 2 },
    DISP_H_{ CAP_H_ },
    gaussMask_(cv::getGaussianKernel(5, -1, CV_64F)*
        cv::getGaussianKernel(5, -1, CV_64F).t())
{
    setWindowTitle("DualCam");

    grabber0_ = new FrameGrabber();
    grabber1_ = new FrameGrabber();

    buildMenus();

    viewLabel_ = new QLabel(this);
    viewLabel_->setAlignment(Qt::AlignCenter);
    updateDisplayGeometry();

    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(viewLabel_);
    setCentralWidget(central);

    QTimer::singleShot(0, this, [this]
    {
    openCams();
    });

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::updateFrame);
    timer_->start(1000 / std::max(1, std::max(p0_.fps, p1_.fps)));
}

MainWindow::~MainWindow()
{
    if (grabber0_) grabber0_->stop();
    if (grabber1_) grabber1_->stop();
    closeCams();
    delete grabber0_;
    delete grabber1_;
}

void MainWindow::buildMenus()
{
    auto camMenu = menuBar()->addMenu(tr("&Camera"));
    camMenu->addAction(tr("1 camera"), this, &MainWindow::setOneCam);
    camMenu->addAction(tr("2 cameras"), this, &MainWindow::setTwoCams);
    camMenu->addSeparator();

    auto resMenu = camMenu->addMenu(tr("Resolution"));
    resMenu->addAction(tr("640x480"), this, &MainWindow::setRes640);
    resMenu->addAction(tr("1280x960"), this, &MainWindow::setRes1280);

    auto fpsMenu = camMenu->addMenu(tr("FPS"));
    fpsMenu->addAction(tr("30"), this, &MainWindow::setFps30);
    fpsMenu->addAction(tr("60"), this, &MainWindow::setFps60);

    auto advMenu = menuBar()->addMenu(tr("&Advanced"));
    actDiff_ = advMenu->addAction(tr("Show &difference"), this, &MainWindow::toggleDiff);
    actDiff_->setCheckable(true);  actDiff_->setChecked(showDiff_);
    actGraph_ = advMenu->addAction(tr("Show &graph"), this, &MainWindow::toggleGraph);
    actGraph_->setCheckable(true); actGraph_->setChecked(showGraph_);
    actFreeze_ = advMenu->addAction(tr("&Freeze history"), [this]() 
        {
        freezeHistory_ = !freezeHistory_;
        actFreeze_->setChecked(freezeHistory_);
        });
    actFreeze_->setCheckable(true);

    auto winMenu = menuBar()->addMenu(tr("&Window"));
    actFull_ = winMenu->addAction(tr("&Fullscreen"), this, &MainWindow::toggleFullscreen);
    actFull_->setCheckable(true);
    winMenu->addAction(tr("Restart"), this, &MainWindow::restartApp);
    winMenu->addAction(tr("Check updates"), this, &MainWindow::checkUpdates);
}

void MainWindow::showEvent(QShowEvent* e) 
{
    QMainWindow::showEvent(e);
    qInfo() << "MainWindow shown";  
}

static bool isVEYEFixedGRAY(const CaptureParams& p)
{
    return (p.v4l2PixelFmt == "GRAY8" &&
        p.width == 1440 &&
        p.height == 1088);
}

void MainWindow::openCams()
{
    cam0_ok_ = cam0_.open(p0_);
    cam1_ok_ = cam1_.open(p1_);

    if (cam0_ok_ && grabber0_) grabber0_->start(&cam0_);
    if (cam1_ok_ && grabber1_) grabber1_->start(&cam1_);

    if (!cam0_ok_ && !cam1_ok_) 
    {
        QMessageBox::warning(this, tr("Camera error"),
            tr("No cameras opened. Running in demo mode."));
    }
}

void MainWindow::closeCams()
{
    if (grabber0_) grabber0_->stop();
    if (grabber1_) grabber1_->stop();
    cam0_.release();
    cam1_.release();
    cam0_ok_ = cam1_ok_ = false;
}

void MainWindow::setOneCam()
{
    closeCams();
    cam1_ok_ = false;
    cam0_ok_ = cam0_.open(p0_);
    if (cam0_ok_ && grabber0_) grabber0_->start(&cam0_);
}

void MainWindow::setTwoCams()
{
    closeCams();
    openCams();
}

void MainWindow::setRes640()
{
    if (!isVEYEFixedGRAY(p0_)) { p0_.width = 640;  p0_.height = 480; }
    if (!isVEYEFixedGRAY(p1_)) { p1_.width = 640;  p1_.height = 480; }

    updateDisplayGeometry();
    closeCams();
    openCams();
}

void MainWindow::setRes1280()
{
    if (!isVEYEFixedGRAY(p0_)) { p0_.width = 1280; p0_.height = 960; }
    if (!isVEYEFixedGRAY(p1_)) { p1_.width = 1280; p1_.height = 960; }

    updateDisplayGeometry();
    closeCams();
    openCams();
}

void MainWindow::setFps30()
{
    p0_.fps = 30; p1_.fps = 30;
    timer_->setInterval(1000 / 30);
    closeCams();
    openCams();
}

void MainWindow::setFps60()
{
    p0_.fps = 60; p1_.fps = 60;
    timer_->setInterval(1000 / 60);
    closeCams();
    openCams();
}

void MainWindow::toggleDiff()
{
    showDiff_ = !showDiff_;
    actDiff_->setChecked(showDiff_);
    updateDisplayGeometry();
}

void MainWindow::toggleGraph()
{
    showGraph_ = !showGraph_;
    actGraph_->setChecked(showGraph_);
    updateDisplayGeometry();
}

void MainWindow::toggleFullscreen()
{
    fullscreen_ = !fullscreen_;
    actFull_->setChecked(fullscreen_);
    if (fullscreen_) showFullScreen();
    else             showNormal();
}

void MainWindow::restartApp()
{
    const QString program = QCoreApplication::applicationFilePath();
    const QStringList args = QCoreApplication::arguments();
    QProcess::startDetached(program, args);
    qApp->quit();
}

void MainWindow::checkUpdates()
{
    QMessageBox::information(this, tr("Updates"), tr("No updates available."));
}

void MainWindow::updateFrame()
{
    cv::Mat f0, f1;
    if (cam0_ok_ && grabber0_) f0 = grabber0_->getFrame();
    if (cam1_ok_ && grabber1_) f1 = grabber1_->getFrame();

    if (f0.empty() && f1.empty()) return;
    if (f0.empty()) f0 = f1.clone();
    if (f1.empty()) f1 = f0.clone();

    cv::Mat f0b = ensureBGR(f0);
    cv::Mat f1b = ensureBGR(f1);

    cv::Mat diff;
    if (showDiff_) 
    {
        cv::absdiff(f0b, f1b, diff);
        cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);
        cv::applyColorMap(diff, diff, cv::COLORMAP_JET);
    }

    cv::Mat g0 = ensureGray(f0b);
    cv::Mat g1 = ensureGray(f1b);
    const double s0 = computeSharp(g0);
    const double s1 = computeSharp(g1);

    history_.push_back({ s0, s1 });
    if (!freezeHistory_ && history_.size() > MAX_HISTORY_) history_.pop_front();

    const int graphH = showGraph_ ? 120 : 0;
    int rows = DISP_H_ + graphH;
    if (showDiff_) rows += CAP_H_;

    cv::Mat canvas(rows, DISP_W_, CV_8UC3, cv::Scalar::all(0));

    cv::resize(f0b, canvas(cv::Rect(0, 0, CAP_W_, CAP_H_)), cv::Size(CAP_W_, CAP_H_));
    cv::resize(f1b, canvas(cv::Rect(CAP_W_, 0, CAP_W_, CAP_H_)), cv::Size(CAP_W_, CAP_H_));

    cv::Mat roiLeft = canvas(cv::Rect(0, 0, CAP_W_, CAP_H_));
    cv::Mat roiRight = canvas(cv::Rect(CAP_W_, 0, CAP_W_, CAP_H_));
    drawOverlay(roiLeft, QString("S0: %1").arg(s0, 0, 'f', 1).toStdString());
    drawOverlay(roiRight, QString("S1: %1").arg(s1, 0, 'f', 1).toStdString());

    int y = CAP_H_;
    if (showDiff_) 
    {
        cv::Mat roiDiff = canvas(cv::Rect(0, y, DISP_W_, CAP_H_));
        if (!diff.empty()) cv::resize(diff, roiDiff, roiDiff.size());
        y += CAP_H_;
    }

    if (showGraph_) 
    {
        cv::Mat graph = makeGraphImage(DISP_W_, graphH);
        cv::Mat roiGraph = canvas(cv::Rect(0, y, DISP_W_, graphH));
        graph.copyTo(roiGraph);
    }

    QImage img(canvas.data, canvas.cols, canvas.rows,
        static_cast<int>(canvas.step), QImage::Format_BGR888);
    viewLabel_->setPixmap(QPixmap::fromImage(img.copy()));
}

void MainWindow::updateDisplayGeometry()
{
    CAP_W_ = std::max(p0_.width, p1_.width);
    CAP_H_ = std::max(p0_.height, p1_.height);
    DISP_W_ = CAP_W_ * 2;
    DISP_H_ = CAP_H_;

    if (viewLabel_) 
    {
        int totalH = DISP_H_;
        if (showDiff_)  totalH += CAP_H_;
        if (showGraph_) totalH += 120;
        viewLabel_->setMinimumSize(DISP_W_, totalH);
    }
}

cv::Mat MainWindow::ensureGray(const cv::Mat& src)
{
    if (src.empty()) return src;
    if (src.channels() == 1) return src;
    cv::Mat gray;
    if (src.channels() == 3)       cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4)  cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else                           cv::cvtColor(ensureBGR(src), gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat MainWindow::ensureBGR(const cv::Mat& src)
{
    if (src.empty()) return src;
    if (src.channels() == 3) return src;
    cv::Mat bgr;
    if (src.channels() == 1)       cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    else if (src.channels() == 4)  cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    else                           cv::cvtColor(src, bgr, cv::COLOR_YUV2BGR_YUYV); 
    return bgr;
}


double MainWindow::computeSharp(const cv::Mat& gray)
{
    cv::Mat bl, lap;
    cv::GaussianBlur(gray, bl, cv::Size(3, 3), 0.0);
    cv::Laplacian(bl, lap, CV_64F);
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    return sigma[0] * sigma[0];
}

void MainWindow::drawOverlay(cv::Mat& img, const std::string& text)
{
    cv::putText(img, text, { 10, 30 }, cv::FONT_HERSHEY_SIMPLEX, 1.0,
        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

cv::Mat MainWindow::makeGraphImage(int w, int h)
{
    cv::Mat graph(h, w, CV_8UC3, cv::Scalar::all(30));
    if (history_.size() < 2) return graph;

    auto drawSeries = [&](int idx, const cv::Scalar& col)
        {
            for (size_t i = 1; i < history_.size(); ++i)
            {
                double vPrev = (idx == 0) ? history_[i - 1].first : history_[i - 1].second;
                double vCurr = (idx == 0) ? history_[i].first : history_[i].second;
                int x0 = static_cast<int>((i - 1) * w / MAX_HISTORY_);
                int x1 = static_cast<int>(i * w / MAX_HISTORY_);
                int y0 = h - std::clamp(static_cast<int>(vPrev / 10.0), 0, h);
                int y1 = h - std::clamp(static_cast<int>(vCurr / 10.0), 0, h);
                cv::line(graph, { x0,y0 }, { x1,y1 }, col, 1, cv::LINE_AA);
            }
        };

    drawSeries(0, { 0,255,0 });
    drawSeries(1, { 0,0,255 });
    return graph;

}

