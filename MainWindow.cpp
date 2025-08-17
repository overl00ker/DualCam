#include "MainWindow.h"

#include <QApplication>
#include <QCoreApplication>
#include <QImage>
#include <QKeyEvent>
#include <QMessageBox>
#include <QPixmap>
#include <QScreen>
#include <QShowEvent>
#include <QtGlobal>
#include <algorithm>
#include <numeric>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace 
{
    static QImage matToQImage(const cv::Mat& m)
    {
        if (m.empty()) return {};
        if (m.type() == CV_8UC3)
        {
            QImage img(m.data, m.cols, m.rows, static_cast<int>(m.step), QImage::Format_BGR888);
            return img.copy(); 
        }
        if (m.type() == CV_8UC1)
        {
            QImage img(m.data, m.cols, m.rows, static_cast<int>(m.step), QImage::Format_Grayscale8);
            return img.copy();
        }
        cv::Mat bgr;
        if (m.channels() == 4)
        {
            cv::cvtColor(m, bgr, cv::COLOR_BGRA2BGR);
        }
        else
        {
            cv::cvtColor(m, bgr, cv::COLOR_RGBA2BGR);
        }
        QImage img(bgr.data, bgr.cols, bgr.rows, static_cast<int>(bgr.step), QImage::Format_BGR888);
        return img.copy();
    }

    static double varianceOfLaplacian(const cv::Mat& gray)
    {
        cv::Mat lap;
        cv::Laplacian(gray, lap, CV_16S, 3);
        cv::Mat absLap;
        cv::convertScaleAbs(lap, absLap);
        cv::Scalar mean, stddev;
        cv::meanStdDev(absLap, mean, stddev);
        return stddev[0] * stddev[0];
    }

    static void drawText(cv::Mat& img, const std::string& text, const cv::Point& org)
    {
        cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, { 255,255,255 }, 1, cv::LINE_AA);
        cv::putText(img, text, org + cv::Point(1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, { 0,0,0 }, 1, cv::LINE_AA);
    }
}

MainWindow::MainWindow(const CaptureParams& cam0, const CaptureParams& cam1, QWidget* parent)
    : QMainWindow(parent)
    , p0_(cam0)
    , p1_(cam1)
{
    setWindowTitle("DualCam");
    resize(1280, 720);

    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);
    layout->setContentsMargins(0, 0, 0, 0);

    viewLabel_ = new QLabel(central);
    viewLabel_->setAlignment(Qt::AlignCenter);
    viewLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    layout->addWidget(viewLabel_);
    setCentralWidget(central);

    buildMenus();

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::updateFrame);
    timer_->start(33); 
    QTimer::singleShot(0, this, [this] { openCams(); });
}

MainWindow::~MainWindow()
{
    if (timer_) timer_->stop();
    closeCams();
    if (openThread_.joinable()) openThread_.join();
}

void MainWindow::showEvent(QShowEvent* e)
{
    QMainWindow::showEvent(e);
    updateDisplayGeometry();
}

void MainWindow::buildMenus()
{
    auto* modeMenu = menuBar()->addMenu(tr("&Mode"));
    modeMenu->addAction(tr("One camera"), this, &MainWindow::setOneCam);
    modeMenu->addAction(tr("Two cameras"), this, &MainWindow::setTwoCams);

    auto* resMenu = menuBar()->addMenu(tr("&Resolution"));
    resMenu->addAction(tr("640x480"), this, &MainWindow::setRes640);
    resMenu->addAction(tr("1280x960"), this, &MainWindow::setRes1280);

    auto* fpsMenu = menuBar()->addMenu(tr("FP&S"));
    fpsMenu->addAction(tr("30"), this, &MainWindow::setFps30);
    fpsMenu->addAction(tr("60"), this, &MainWindow::setFps60);

    auto* advMenu = menuBar()->addMenu(tr("&Advanced"));
    actDiff_ = advMenu->addAction(tr("Show &diff"), this, &MainWindow::toggleDiff);
    actDiff_->setCheckable(true); actDiff_->setChecked(showDiff_);
    actGraph_ = advMenu->addAction(tr("Show &graph"), this, &MainWindow::toggleGraph);
    actGraph_->setCheckable(true); actGraph_->setChecked(showGraph_);

    auto* winMenu = menuBar()->addMenu(tr("&Window"));
    actFull_ = winMenu->addAction(tr("&Fullscreen"), this, &MainWindow::toggleFullscreen);
    actFull_->setCheckable(true);
    winMenu->addAction(tr("Restart"), this, &MainWindow::restartApp);
    winMenu->addAction(tr("Check updates"), this, &MainWindow::checkUpdates);
}

void MainWindow::openCams()
{
    if (openThread_.joinable()) openThread_.join();

    openThread_ = std::thread([this] 
        {
        bool ok0 = cam0_.open(p0_);
        bool ok1 = cam1_.open(p1_);
        QMetaObject::invokeMethod(this, [this, ok0, ok1] 
            {
            cam0_ok_ = ok0; cam1_ok_ = ok1;
            if (!ok0 && !ok1)
            {
                QMessageBox::warning(this, tr("Camera error"), tr("No cameras opened. Running without input."));
            }
            }, Qt::QueuedConnection);
        });
}

void MainWindow::closeCams()
{
    cam0_.release();
    cam1_.release();
}

void MainWindow::updateDisplayGeometry()
{
    const int w = viewLabel_->width();
    const int h = viewLabel_->height();
    DISP_W_ = std::max(1, w);
    DISP_H_ = std::max(1, h);
}

cv::Mat MainWindow::ensureGray(const cv::Mat& src)
{
    if (src.empty()) return {};
    if (src.type() == CV_8UC1) return src;
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else src.convertTo(gray, CV_8U);
    return gray;
}

cv::Mat MainWindow::ensureBGR(const cv::Mat& src)
{
    if (src.empty()) return {};
    if (src.type() == CV_8UC3) return src;
    cv::Mat bgr;
    if (src.type() == CV_8UC1) cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    else if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    else src.convertTo(bgr, CV_8UC3);
    return bgr;
}

cv::Mat MainWindow::makeGraphImage(int w, int h)
{
    cv::Mat graph(h, w, CV_8UC3, cv::Scalar(30, 30, 30));
    if (history_.empty()) return graph;

    auto draw = [&](int idx, cv::Scalar col) 
        {
        double prev = history_.front().first;
        for (int i = 1; i < static_cast<int>(history_.size()); ++i)
        {
            double curr = (idx == 0) ? history_[i].first : history_[i].second;
            int x0 = (i - 1) * w / MAX_HISTORY_;
            int x1 = i * w / MAX_HISTORY_;
            int y0 = h - std::clamp(static_cast<int>(prev / 10.0), 0, h);
            int y1 = h - std::clamp(static_cast<int>(curr / 10.0), 0, h);
            cv::line(graph, { x0, y0 }, { x1, y1 }, col, 1, cv::LINE_AA);
            prev = curr;
        }
        };

    draw(0, { 0,255,0 });
    draw(1, { 0,0,255 });
    return graph;
}

void MainWindow::updateFrame()
{
    updateDisplayGeometry();

    cv::Mat f0, f1;
    bool has0 = false, has1 = false;
    try { has0 = cam0_.read(f0); }
    catch (...) { has0 = false; }
    try { has1 = cam1_.read(f1); }
    catch (...) { has1 = false; }

    if (!has0 && !has1)
    {
        cv::Mat blank(DISP_H_, DISP_W_, CV_8UC3, cv::Scalar(0, 0, 0));
        QImage qi = matToQImage(blank);
        viewLabel_->setPixmap(QPixmap::fromImage(qi).scaled(viewLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        return;
    }

    cv::Mat g0 = ensureGray(f0);
    cv::Mat g1 = ensureGray(f1);

    double sharp0 = g0.empty() ? 0.0 : varianceOfLaplacian(g0);
    double sharp1 = g1.empty() ? 0.0 : varianceOfLaplacian(g1);

    if (!freezeHistory_)
    {
        history_.emplace_back(sharp0, sharp1);
        if (history_.size() > MAX_HISTORY_) history_.pop_front();
    }

    cv::Mat disp;

    if (mode_ == ViewMode::Analytics4Q && has0 && has1)
    {
        int halfW = std::max(2, DISP_W_ / 2);
        int halfH = std::max(2, DISP_H_ / 2);

        cv::Mat c0bgr = ensureBGR(g0.empty() ? f0 : f0);
        cv::Mat c1bgr = ensureBGR(g1.empty() ? f1 : f1);
        cv::resize(c0bgr, c0bgr, { halfW, halfH });
        cv::resize(c1bgr, c1bgr, { halfW, halfH });

        cv::Mat diffBGR(halfH, halfW, CV_8UC3, cv::Scalar(0, 0, 0));
        if (showDiff_ && !g0.empty() && !g1.empty())
        {
            cv::Mat diff;
            cv::absdiff(g0, g1, diff);
            cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX);
            cv::cvtColor(diff, diffBGR, cv::COLOR_GRAY2BGR);
            cv::resize(diffBGR, diffBGR, { halfW, halfH });
        }

        cv::Mat graph = showGraph_ ? makeGraphImage(halfW, halfH) : cv::Mat(halfH, halfW, CV_8UC3, cv::Scalar(20, 20, 20));

        disp = cv::Mat(2 * halfH, 2 * halfW, CV_8UC3);
        c0bgr.copyTo(disp(cv::Rect(0, 0, halfW, halfH)));
        c1bgr.copyTo(disp(cv::Rect(halfW, 0, halfW, halfH)));
        diffBGR.copyTo(disp(cv::Rect(0, halfH, halfW, halfH)));
        graph.copyTo(disp(cv::Rect(halfW, halfH, halfW, halfH)));

        drawText(disp, "Sharp0: " + std::to_string(static_cast<int>(sharp0)), { 10, 20 });
        drawText(disp, "Sharp1: " + std::to_string(static_cast<int>(sharp1)), { halfW + 10, 20 });
    }
    else
    {
        cv::Mat bgr = has0 ? ensureBGR(f0) : ensureBGR(f1);
        cv::resize(bgr, bgr, { DISP_W_, DISP_H_ });
        drawText(bgr, has0 ? "CAM0" : "CAM1", { 10, 20 });
        disp = std::move(bgr);
    }

    QImage qi = matToQImage(disp);
    viewLabel_->setPixmap(QPixmap::fromImage(qi).scaled(viewLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::setOneCam()
{
    mode_ = ViewMode::SideBySide; 
}

void MainWindow::setTwoCams()
{
    mode_ = ViewMode::Analytics4Q;
}

void MainWindow::setRes640()
{
    p0_.width = 640; p0_.height = 480;
    p1_.width = 640; p1_.height = 480;
    closeCams();
    openCams();
}

void MainWindow::setRes1280()
{
    p0_.width = 1280; p0_.height = 960;
    p1_.width = 1280; p1_.height = 960;
    closeCams();
    openCams();
}

void MainWindow::setFps30()
{
    p0_.fps = 30; p1_.fps = 30;
    closeCams();
    openCams();
}

void MainWindow::setFps60()
{
    p0_.fps = 60; p1_.fps = 60;
    closeCams();
    openCams();
}

void MainWindow::toggleDiff()
{
    showDiff_ = !showDiff_;
    if (actDiff_) actDiff_->setChecked(showDiff_);
}

void MainWindow::toggleGraph()
{
    showGraph_ = !showGraph_;
    if (actGraph_) actGraph_->setChecked(showGraph_);
}

void MainWindow::toggleFullscreen()
{
    if (isFullScreen()) 
    {
        showNormal();
        if (actFull_) actFull_->setChecked(false);
    }
    else {
        showFullScreen();
        if (actFull_) actFull_->setChecked(true);
    }
}

void MainWindow::restartApp()
{
    QString program = QCoreApplication::applicationFilePath();
    QStringList args = QCoreApplication::arguments();
    QProcess::startDetached(program, args);
    qApp->quit();
}

void MainWindow::checkUpdates()
{
    QMessageBox::information(this, tr("Updates"), tr("No update mechanism wired yet."));
}

