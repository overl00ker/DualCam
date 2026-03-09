#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QChart>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/background_segm.hpp>

#include <deque>

class QWidget;
class QVBoxLayout;
class QHBoxLayout;
class QLabel;
class QPushButton;
class QTimer;
class QStatusBar;
class QComboBox;
class QCheckBox;
class QSplitter;
class QTabWidget;
class QSpinBox;
class QSlider;

enum class ColorMode { GRAY_NATIVE, GRAY_CV, COLOR };

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void openCameras();
    void closeCameras();
    void updateFrames();
    void updateView();
    void calibrateAlignment();

private:
    void initUI();
    void displayMat(QLabel* label, const cv::Mat& mat);
    double calculateFocus(const cv::Mat& frame);

    double detectMotion(const cv::Mat& frame);
    cv::Mat applyTemporalDenoise();
    cv::Mat fuseCameras(const cv::Mat& a, const cv::Mat& b);
    cv::Mat applyBilateral(const cv::Mat& src);
    cv::Mat applyDiffView(const cv::Mat& d1, const cv::Mat& d2);

    QStringList getLibCameraIds();
    std::string makeGStreamerPipeline(const QString& cameraId, int width, int height, int fps);

    cv::VideoCapture m_cap1;
    cv::VideoCapture m_cap2;
    QTimer* m_timer;

    cv::Mat m_frame1;
    cv::Mat m_frame2;
    cv::Mat m_eccWarpMatrix;

    bool m_camerasOpen = false;
    bool m_isAligned = false;
    qint64 m_frameCount = 0;
    int m_maxHistory = 200;

    QCheckBox* m_chkFlipVer2 = nullptr;
    QCheckBox* m_chkFlipHor2 = nullptr;

    cv::Ptr<cv::BackgroundSubtractorMOG2> m_bgSubtractor;
    std::deque<cv::Mat> m_frameBuffer1;
    std::deque<cv::Mat> m_frameBuffer2;
    int m_bufferSize = 8;
    double m_motionThreshold = 0.05;
    bool m_motionActive = false;
    int m_noiseFloor = 15;
    bool m_diffStretch = false;

    ColorMode m_colorMode = ColorMode::GRAY_CV;

    QWidget* m_centralWidget;
    QTabWidget* m_tabWidget;

    QLabel* m_view1;
    QLabel* m_view2;
    QLabel* m_resultView;
    QSplitter* m_splitter;
    QPushButton* m_btnToggleCameras;
    QComboBox* m_comboViewMode;
    QComboBox* m_comboColorMode;
    QCheckBox* m_chkAlign;
    QPushButton* m_btnCalibrateAlign;

    QSlider* m_bufferSlider;
    QLabel* m_bufferLabel;
    QSlider* m_motionThresholdSlider;
    QLabel* m_motionThresholdLabel;
    QCheckBox* m_chkFusion;
    QCheckBox* m_chkBilateral;
    QLabel* m_motionIndicator;

    QSlider* m_noiseFloorSlider;
    QLabel* m_noiseFloorLabel;
    QCheckBox* m_chkStretch;

    QChartView* m_chartView;
    QChart* m_chart;
    QLineSeries* m_seriesCam1;
    QLineSeries* m_seriesCam2;
    QSpinBox* m_historySpinBox;

    QStatusBar* m_statusBar;
};

#endif