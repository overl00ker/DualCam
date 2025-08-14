#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <QAction>
#include <QMenuBar>
#include <QKeyEvent>
#include <QVBoxLayout>
#include <QProcess>
#include <deque>
#include <memory>
#include <opencv2/opencv.hpp>
#include "capture_backend.h"

class FrameGrabber;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(const CaptureParams& cam0,
                        const CaptureParams& cam1,
                        QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void setOneCam();
    void setTwoCams();
    void setRes640();
    void setRes1280();
    void setFps30();
    void setFps60();

    void toggleDiff();
    void toggleGraph();

    void toggleFullscreen();
    void restartApp();
    void checkUpdates();

private:
    void buildMenus();
    void openCams();
    void closeCams();
    void updateFrame();

    void updateDisplayGeometry();

    [[nodiscard]] cv::Mat ensureGray(const cv::Mat& src);
    [[nodiscard]] cv::Mat ensureBGR(const cv::Mat& src);
    double  computeSharp(const cv::Mat& gray);
    void    drawOverlay(cv::Mat& img, const std::string& text);
    cv::Mat makeGraphImage(int w, int h);

private:
    CaptureParams p0_, p1_;

    int CAP_W_{ 0 }, CAP_H_{ 0 }, DISP_W_{ 0 }, DISP_H_{ 0 };

    LibcameraCapture cam0_, cam1_;
    bool cam0_ok_{ false };
    bool cam1_ok_{ false };

    QLabel* viewLabel_{ nullptr };
    QTimer* timer_{ nullptr };

    QAction* actDiff_{ nullptr };
    QAction* actGraph_{ nullptr };
    QAction* actFull_{ nullptr };
    QAction* actFreeze_{ nullptr };

    bool showDiff_{ true };
    bool showGraph_{ true };
    bool fullscreen_{ false };
    bool freezeHistory_{ false };

    cv::Mat gaussMask_;

    std::deque<std::pair<double, double>> history_;
    static constexpr int MAX_HISTORY_ = 100;

    enum class ViewMode { SideBySide, Analytics4Q };
    ViewMode mode_{ ViewMode::Analytics4Q };

    FrameGrabber* grabber0_{ nullptr };
    FrameGrabber* grabber1_{ nullptr };

protected:
    void showEvent(QShowEvent* e) override;                        
};

