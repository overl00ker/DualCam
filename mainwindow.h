#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QChart>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>

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
    bool computeHomography(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& outHomography);

    // --- New Helper Methods for GStreamer/Libcamera ---
    QStringList getLibCameraIds();
    std::string makeGStreamerPipeline(const QString& cameraId, int width, int height, int fps);

    cv::VideoCapture m_cap1;
    cv::VideoCapture m_cap2;
    QTimer* m_timer;

    cv::Mat m_frame1;
    cv::Mat m_frame2;
    cv::Ptr<cv::ORB> m_orb;
    cv::Ptr<cv::BFMatcher> m_matcher;
    cv::Mat m_homography;

    bool m_camerasOpen = false;
    bool m_isAligned = false;
    qint64 m_frameCount = 0;
    int m_maxHistory = 200;

    QWidget* m_centralWidget;
    QTabWidget* m_tabWidget;

    QLabel* m_view1;
    QLabel* m_view2;
    QLabel* m_resultView;
    QSplitter* m_splitter;
    QPushButton* m_btnToggleCameras;
    QComboBox* m_comboViewMode;
    QCheckBox* m_chkAlign;
    QPushButton* m_btnCalibrateAlign;

    QChartView* m_chartView;
    QChart* m_chart;
    QLineSeries* m_seriesCam1;
    QLineSeries* m_seriesCam2;
    QSpinBox* m_historySpinBox;

    QStatusBar* m_statusBar;
};

#endif