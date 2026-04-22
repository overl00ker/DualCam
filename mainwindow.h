#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QChart>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#include <deque>
#include <thread>
#include <atomic>
#include <QString>
#include <QKeySequence>
#include <QShortcut>
#include <functional>
#include <QMap>
#include <QThread>
#include <QMutex>

enum class CmdType { Action, Toggle, Parameter };

struct AppCommand {
    QString id;
    QString name;
    QString section;
    CmdType type;
    std::function<void()> trigger;
    std::function<void(QWidget*)> showParam;
};

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
class QStackedWidget;
class QSpinBox;
class QSlider;
class QGroupBox;
class QTabWidget;
class QLineEdit;
class QDoubleSpinBox;
class QDialog;
class QListWidget;

enum class ColorMode { GRAY_NATIVE, GRAY_CV, COLOR };

struct ManualAdjust {
    double tx = 0.0;
    double ty = 0.0;
    double scale = 1.0;
    double rx = 0.0;
    double ry = 0.0;
    double rz = 0.0;
    bool isIdentity() const {
        return tx == 0.0 && ty == 0.0 && scale == 1.0 &&
               rx == 0.0 && ry == 0.0 && rz == 0.0;
    }
};

struct WorkerParams {
    ColorMode colorMode = ColorMode::GRAY_CV;
    bool flipHor2 = false;
    bool flipVer2 = false;
    double motionThr = 0.05;
    int bufferSize = 8;
    bool applyBilateral = false;
    int noiseFloor = 15;
};

class CameraWorker : public QThread {
    Q_OBJECT
public:
    CameraWorker(QObject* parent = nullptr);
    ~CameraWorker();

    void startCameras(const std::string& pipe1, const std::string& pipe2, int w, int h, int fps);
    void startCamerasV4L2(int id1, int id2, int w, int h, int fps);
    void stopCameras();
    void setParams(const WorkerParams& p);

signals:
    void framesProcessed(cv::Mat f1, cv::Mat f2, double focus1, double focus2, bool motion, qint64 frameCount);
    void cameraError(QString msg);

protected:
    void run() override;

private:
    cv::Mat applyTemporalDenoise(cv::Mat& frame, cv::Mat& ema, int bufferSize);
    double detectMotion(const cv::Mat& frame, double thr);
    double calculateFocus(const cv::Mat& frame);
    cv::Mat toWorkingFormat(const cv::Mat& frame, ColorMode mode);

    cv::VideoCapture m_cap1;
    cv::VideoCapture m_cap2;
    std::atomic<bool> m_running{false};
    
    QMutex m_paramMutex;
    WorkerParams m_params;

    cv::Ptr<cv::BackgroundSubtractorMOG2> m_bgSubtractor;
    cv::Mat m_ema1;
    cv::Mat m_ema2;
    qint64 m_frameCount = 0;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

    void initCommands();
    void showActionsMenu();
    void applyShortcut(const QString& id, const QKeySequence& seq);
    void executeCommand(const QString& id);

    QList<AppCommand> m_commands;
    QMap<QString, QShortcut*> m_shortcuts;
    QMap<QString, QKeySequence> m_hotkeys;

protected:
    void resizeEvent(QResizeEvent* event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;

private slots:
    void openCameras();
    void closeCameras();
    void updateView();
    void calibrateAlignment();
    void saveSnapshot();
    void toggleSheet();
    void setDiffMode(bool on);

public slots:
    void onFramesProcessed(cv::Mat f1, cv::Mat f2, double focus1, double focus2, bool motionDetected, qint64 frameCount);

private:
    void initUI();
    QString styleSheetText() const;
    QWidget* buildCaptureTab();
    QWidget* buildPipelineTab();
    QWidget* buildDiffTab();
    QWidget* buildSnapshotTab();
    QWidget* buildPresetsTab();
    void buildAlignDialog();
    void positionFloatingButtons();
    void buildFocusChart();
    QWidget* buildFocusDataPanel();
    void toggleFocusView();
    void refreshSnapshotPreview();
    void updateEccPill();
    void updateFpsPill();

    void saveDiffSnapshot();
    void saveDualSnapshot(bool combined);

    void displayMat(QLabel* label, const cv::Mat& mat);
    double calculateFocus(const cv::Mat& frame);

    cv::Mat fuseCameras(const cv::Mat& a, const cv::Mat& b);
    cv::Mat applyDiffView(const cv::Mat& d1, const cv::Mat& d2);

    QStringList getLibCameraIds();
    std::string makeGStreamerPipeline(const QString& cameraId, int width, int height, int fps);

    void saveSettings();
    void loadSettings();
    void savePreset(const QString& name);
    void loadPreset(const QString& name);
    void deletePreset(const QString& name);
    void refreshPresetList();
    void refreshCameraModes();

    cv::Mat buildManualHomography(const ManualAdjust& a, const cv::Size& sz) const;
    ManualAdjust& activeManualAdjust();
    void applyAdjustToWidgets();
    void applyWidgetsToAdjust();
    void resetActiveAdjust();

    CameraWorker* m_worker = nullptr;

    cv::Mat m_frame1;
    cv::Mat m_frame2;
    cv::Mat m_eccWarpMatrix;
    cv::Mat m_lastDiffResult;

    bool m_camerasOpen = false;
    bool m_isAligned = false;
    bool m_isDiffMode = false;
    bool m_sheetOpen = true;
    bool m_focusViewActive = false;
    qint64 m_frameCount = 0;
    int m_maxHistory = 200;

    int m_dragStartPos = 0;
    int m_dragStartHeight = 0;
    bool m_isDraggingSheet = false;

    double m_lastFocus1 = 0.0;
    double m_lastFocus2 = 0.0;

    int m_bufferSize = 8;
    double m_motionThreshold = 0.05;
    bool m_motionActive = false;
    int m_noiseFloor = 15;

    ColorMode m_colorMode = ColorMode::GRAY_CV;

    ManualAdjust m_manualAdj1;
    ManualAdjust m_manualAdj2;
    int m_activeAdjCam = 2;
    bool m_updatingAdjUI = false;

    QWidget* m_centralWidget;
    QWidget* m_videoArea;
    QWidget* m_sheetWidget;
    QTabWidget* m_tabWidget;
    QStackedWidget* m_sheetStack;

    QLabel* m_fpsPill;
    QLabel* m_eccPill;
    QPushButton* m_btnGallery;
    QPushButton* m_btnHelp;

    QLabel* m_view1;
    QLabel* m_view2;
    QLabel* m_resultView;
    QLabel* m_osd1;
    QLabel* m_osd2;
    QLabel* m_osdResult;
    QSplitter* m_splitter;
    QPushButton* m_btnModeToggle;
    QPushButton* m_btnFocusToggle;
    QPushButton* m_btnFabSnapshot;
    QPushButton* m_btnFabStream;
    QComboBox* m_comboCamSet;
    QLabel* m_fabStreamIcon = nullptr;
    QLabel* m_fabSnapIcon = nullptr;

    QPushButton* m_btnSheetHandle;

    QPushButton* m_btnToggleCameras;
    QComboBox* m_comboColorMode;
    QCheckBox* m_chkFlipVer2;
    QCheckBox* m_chkFlipHor2;
    QCheckBox* m_chkBilateral;
    QComboBox* m_comboSnapshotMode;
    QPushButton* m_btnSaveSnapshot;

    QSlider* m_bufferSlider;
    QLabel* m_bufferLabel;
    QSlider* m_motionThresholdSlider;
    QLabel* m_motionThresholdLabel;
    QCheckBox* m_chkFusion;
    QLabel* m_motionIndicator;
    QCheckBox* m_chkAlign;
    QPushButton* m_btnCalibrateAlign;
    QLabel* m_eccIndicator;
    QPushButton* m_btnOpenManualAlign;

    QSlider* m_noiseFloorSlider;
    QLabel* m_noiseFloorLabel;
    QCheckBox* m_chkStretch;
    QPushButton* m_btnPeakIntensities;
    QLabel* m_lblPeakInfo;

    QLabel* m_lblFocus1Big;
    QLabel* m_lblFocus2Big;
    QSpinBox* m_historySpinBox;
    QSlider* m_historySlider;
    QChartView* m_chartView;
    QChart* m_chart;
    QLineSeries* m_seriesCam1;
    QLineSeries* m_seriesCam2;

    QWidget* m_focusDataWidget;
    QListWidget* m_snapshotPreview;
    QLineEdit* m_snapshotNameEdit;

    QListWidget* m_presetList;
    QLineEdit* m_presetNameEdit;
    QPushButton* m_btnSavePreset;
    QPushButton* m_btnLoadPreset;
    QPushButton* m_btnDeletePreset;

    QDialog* m_alignDialog;
    QComboBox* m_comboAdjCam;
    QSlider* m_sldAdjTx; QDoubleSpinBox* m_spnAdjTx;
    QSlider* m_sldAdjTy; QDoubleSpinBox* m_spnAdjTy;
    QSlider* m_sldAdjScale; QDoubleSpinBox* m_spnAdjScale;
    QSlider* m_sldAdjRx; QDoubleSpinBox* m_spnAdjRx;
    QSlider* m_sldAdjRy; QDoubleSpinBox* m_spnAdjRy;
    QSlider* m_sldAdjRz; QDoubleSpinBox* m_spnAdjRz;
    QPushButton* m_btnResetAdj;

    QStatusBar* m_statusBar;

    std::thread m_calibThread;
    std::atomic<bool> m_calibrating{false};
};

#endif
