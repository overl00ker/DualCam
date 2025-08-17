#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <QAction>
#include <QMenuBar>
#include <QVBoxLayout>
#include <QProcess>
#include <deque>
#include <utility>
#include <memory>
#include <thread>
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

protected:
    void showEvent(QShowEvent* e) override;

private:
    void buildMenus();

    void openCams();
    void closeCams();

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

    void updateFrame();
    void updateDisplayGeometry();

    cv::Mat ensureGray(const cv::Mat& src);
    cv::Mat ensureBGR(const cv::Mat& src);
    cv::Mat makeGraphImage(int w, int h);

private:
    CaptureParams p0_;
    CaptureParams p1_;

    LibcameraCapture cam0_;
    LibcameraCapture cam1_;
    bool cam0_ok_{ false };
    bool cam1_ok_{ false };

    QLabel* viewLabel_{ nullptr };
    QTimer* timer_{ nullptr };

    QAction* actDiff_{ nullptr };
    QAction* actGraph_{ nullptr };
    QAction* actFreeze_{ nullptr };
    QAction* actFull_{ nullptr };

    bool showDiff_{ false };
    bool showGraph_{ false };
    bool freezeHistory_{ false };

    int CAP_W_{ 0 }, CAP_H_{ 0 };  
    int DISP_W_{ 0 }, DISP_H_{ 0 }; 

    cv::Mat gaussMask_;

    std::deque<std::pair<double, double>> history_;
    static constexpr int MAX_HISTORY_ = 100;

    enum class ViewMode { SideBySide, Analytics4Q };
    ViewMode mode_{ ViewMode::Analytics4Q };

    FrameGrabber* grabber0_{ nullptr };
    FrameGrabber* grabber1_{ nullptr };

    std::thread openThread_;
};
