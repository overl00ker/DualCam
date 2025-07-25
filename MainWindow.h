#pragma once

#include <QMainWindow>
#include <QTimer>
#include <QProcess>
#include <QMenuBar>
#include <QAction>
#include <QLabel>
#include <deque>
#include <opencv2/opencv.hpp>
#include "capture_backend.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(const CaptureParams& p0, const CaptureParams& p1, QWidget* parent = nullptr);
    ~MainWindow();

protected:
    void keyPressEvent(QKeyEvent* event) override;

private slots:
    // Camera
    void setOneCamera();
    void setTwoCameras();
    void setResolution640x480();
    void setResolution1280x720();
    void setFps30();
    void setFps60();
    // Advance
    void toggleDifference(bool);
    void toggleGraph(bool);
    // Window
    void enterFullScreen();
    void exitFullScreen();
    void refreshApp();
    void restartApp();

private:
    void openCams();
    void closeCams();
    void updateFrame();
    void createMenus();

    cv::Mat ensureGray(const cv::Mat& src);
    double computeSharp(const cv::Mat& gray);
    void drawOverlay(cv::Mat& img, const std::string& text);
    cv::Mat makeGraphImage(int w, int h);

    CaptureParams p0_, p1_;
    int CAP_W_ = 640, CAP_H_ = 480;
    int DISP_W_ = 640, DISP_H_ = 480;

    enum class ViewMode { SideBySide, Analytics4Q };
    ViewMode mode_ = ViewMode::Analytics4Q;

    // UI
    QLabel* viewLabel_ = nullptr;
    QTimer* timer_     = nullptr;

    // Menu
    QMenuBar* menuBar_ = nullptr;
    QMenu* cameraMenu_ = nullptr;
    QMenu* advanceMenu_ = nullptr;
    QMenu* windowMenu_ = nullptr;
    QAction* actOneCam_ = nullptr;
    QAction* actTwoCam_ = nullptr;
    QAction* actDiff_   = nullptr;
    QAction* actGraph_  = nullptr;

    // Capture
    cv::VideoCapture cam0_, cam1_;
    std::deque<std::pair<int,int>> history_;
    cv::Mat gaussMask_;
    bool freezeHistory_ = false;
    bool showGraph_ = true;
    bool showDiff_ = true;
    int desiredCamCount_ = 2;

    static constexpr int MAX_HISTORY_ = 120;
};
