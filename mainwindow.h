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

#include "exif_writer.h"

#include <deque>
#include <vector>
#include <thread>
#include <atomic>
#include <QString>
#include <QKeySequence>
#include <QShortcut>
#include <functional>
#include <QMap>
#include <QThread>
#include <QMutex>
#include <QImage>
#include <QSize>
#include <QHash>
#include <QDateTime>
#include <QPixmap>
#include <QList>
#include <QPair>
#include <QColor>

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
class QSettings;
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
class QFrame;

enum class ColorMode { GRAY_NATIVE, GRAY_CV, COLOR };

enum class NavItem {
    None,
    Capture,
    Pipeline,
    Diff,
    Focus,
    Snapshot,
    Presets,
    Gallery
};

class ImageView : public QWidget {
    Q_OBJECT
public:
    explicit ImageView(QWidget* parent = nullptr);
    void setImage(const QImage& img);
    void setPlaceholder(const QString& text);
    void setOverlayText(const QString& text, bool rightAlign);
    void setOverlayColor(const QColor& c);
    void setColorScale(const QImage& ramp, double maxValue, const QString& title);
    void setStretch(bool stretch) { m_stretch = stretch; update(); }
    QSize targetSize(const QSize& imageSize) const;

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QImage m_image;
    QString m_placeholder;
    QString m_overlayText;
    bool m_overlayRight = false;
    bool m_stretch = false;
    QColor m_overlayColor{0xe5, 0xe2, 0xe1};
    QImage m_scaleRamp;
    double m_scaleMax = 0.0;
    QString m_scaleTitle;
};

struct DiffStats {
    bool valid = false;
    double k = 1.0;
    int noiseFloor = 0;
    int clipLevel = 255;
    double gain = 1.0;
    double fullScale = 255.0;
    double maxValue = 0.0;
    double clippedBoth = 0.0;
    double clippedOne = 0.0;
    double noData = 0.0;
    double sharpRatio = 0.0;
    int psfChannel = 0;
    double psfSigma = 0.0;
};

struct KCalibration {
    QString source;
    QDateTime when;
    int frames = 0;
    int pixels = 0;
    double sigma = 0.0;
    double spread = 0.0;
    QString exposure;
    QSize frameSize;
    std::vector<double> perFrame;
};

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
    int bilateralStrength = 5;
    int noiseFloor = 15;
};

class CameraWorker : public QThread {
    Q_OBJECT
public:
    CameraWorker(QObject* parent = nullptr);
    ~CameraWorker();

    bool startCameras(const std::string& pipe1, const std::string& pipe2, int w, int h, int fps);
    bool startCamerasV4L2(int id1, int id2, int w, int h, int fps);
    void stopCameras();
    void setParams(const WorkerParams& p);

    qint64 processedFrames() const { return m_statProcessed.load(); }
    qint64 droppedFrames() const { return m_statDropped.load(); }
    int threadId() const { return m_threadId.load(); }

signals:
    void framesProcessed(cv::Mat f1, cv::Mat f2, double focus1, double focus2, bool motion, qint64 frameCount);
    void cameraError(QString msg);

protected:
    void run() override;

private:
    cv::Mat applyTemporalDenoise(cv::Mat& frame, cv::Mat& ema, int bufferSize);
    double detectMotion(const cv::Mat& frame);
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

    std::atomic<qint64> m_statProcessed{0};
    std::atomic<qint64> m_statDropped{0};
    std::atomic<int> m_threadId{0};

public:
    std::atomic<int> m_pendingFrames{0};
};

class ResourceMonitor;
struct ResourceStats;

struct SensorMode {
    int width = 0;
    int height = 0;
    double fps = 0.0;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

    void initCommands();
    void showQuickTour();
    void showHelpDialog();
    void showActionsMenu();
    void applyShortcut(const QString& id, const QKeySequence& seq);
    void executeCommand(const QString& id);

    QList<AppCommand> m_commands;
    QMap<QString, QShortcut*> m_shortcuts;
    QMap<QString, QKeySequence> m_hotkeys;

protected:
    void resizeEvent(QResizeEvent* event) override;
    void closeEvent(QCloseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
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
    QWidget* buildSettingsTab();
    QWidget* buildGalleryView();
    QWidget* buildSideNav();
    void buildAlignDialog();
    void positionFloatingButtons(bool animate = true);
    void buildFocusChart();
    QWidget* buildFocusDataPanel();
    void toggleFocusView();
    void refreshSnapshotPreview();
    void openPreviewWindow(const QString& imagePath, const QString& fileBase);
    void launchProfileViewer(const cv::Mat& img, QPoint a, QPoint b,
                             const QString& title, bool darkTheme,
                             QWidget* parent = nullptr);
    void launchSurfaceViewer(const cv::Mat& img, QRect roi,
                             const QString& title, bool darkTheme,
                             QWidget* parent = nullptr);
    void minimizeAllDialogs();
    void animateDialogEntry(QDialog* dlg, QWidget* triggerWidget, int durationMs);
    int execDialog(QDialog* dlg, QWidget* triggerWidget);
    void cameraTransition(const std::function<void()>& fn);
    void updateEccPill();
    void updateFpsPill();
    void toggleResourceHud();
    void updateResourceDisplay(const ResourceStats& st);
    QString formatResourceStats(const ResourceStats& st) const;
    void setNavItem(NavItem item);
    void toggleNavItem(NavItem item);
    void setNavExpanded(bool expanded);
    void togglePreviewMode();
    void applyNavLabels();

    void saveDiffSnapshot();
    void saveDualSnapshot(bool combined);
    void showSnapshotMenu(QListWidget* list, const QPoint& pos);
    QString buildSnapshotBaseName(const QString& prefix) const;
    ExifParams buildExifParams(const QString& mode) const;

    void displayMat(ImageView* view, const cv::Mat& mat);
    double calculateFocus(const cv::Mat& frame);
    void pushWorkerParams();

    cv::Mat fuseCameras(const cv::Mat& a, const cv::Mat& b);
    cv::Mat applyDiffView(const cv::Mat& d1, const cv::Mat& d2,
                          const cv::Mat& valid1, const cv::Mat& valid2);
    void calibrateChannelBalance();
    void collectKSample();
    void stopKCalibration(const QString& message);
    void setChannelK(double k, const KCalibration& record);
    void saveKCalibration();
    void loadKCalibration(QSettings& s);
    void updateKInfo();
    QString exposureSignature() const;
    void updateDiffReadout(bool force = false);

    QStringList getLibCameraIds();
    void probeCameras();
    QSize sensorNativeSize() const;
    double sensorMaxFps(int width, int height) const;
    bool fitToSensor(int& width, int& height, int& fps) const;
    std::string makeGStreamerPipeline(const QString& cameraId, int width, int height, int fps, int camIndex = 0);
    void applyExposureControls();

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

    static cv::Mat alignmentSignal(const cv::Mat& gray);
    static bool refineCorrespondences(const cv::Mat& sig1, const cv::Mat& sig2,
                                      std::vector<cv::Point2f>& ptsSrc,
                                      std::vector<cv::Point2f>& ptsDst,
                                      const cv::Mat& H, int halfWin, int searchRad);
    static bool eccPolish(const cv::Mat& sig1, const cv::Mat& sig2,
                          cv::Mat& warp, int motionType, int levels,
                          int iters, double eps, double& outRho);
    static double warpDisagreement(const cv::Mat& A, const cv::Mat& B, const cv::Size& sz);

    void saveAlignmentMatrix();
    void loadAlignmentMatrix();
    void invalidateAlignment(const QString& reason);

    CameraWorker* m_worker = nullptr;

    cv::Mat m_frame1;
    cv::Mat m_frame2;
    cv::Mat m_eccWarpMatrix;
    cv::Mat m_lastDiffResult;

    cv::Size m_alignCalibSize{0, 0};
    QString m_alignModelInfo;
    double m_alignRho = 0.0;

    bool m_camerasOpen = false;
    bool m_isAligned = false;
    bool m_isDiffMode = false;
    bool m_focusViewActive = false;
    bool m_navExpanded = true;
    bool m_previewMode = false;
    NavItem m_currentNav = NavItem::Capture;
    NavItem m_lastNonGalleryNav = NavItem::Capture;
    qint64 m_frameCount = 0;
    int m_maxHistory = 200;

    double m_lastFocus1 = 0.0;
    double m_lastFocus2 = 0.0;

    int m_bufferSize = 8;
    double m_motionThreshold = 0.05;
    bool m_motionActive = false;
    int m_noiseFloor = 15;
    double m_channelK = 1.0;
    KCalibration m_kCal;
    bool m_kProgrammatic = false;
    int m_kCalTarget = 0;
    std::vector<double> m_kCalSamples;
    std::vector<int> m_kCalPixels;
    std::vector<double> m_kCalSpreads;
    int m_clipLevel = 254;
    DiffStats m_diffStats;
    cv::Mat m_lastDiffA, m_lastDiffB, m_lastDiffValid;
    QImage m_diffScaleRamp;
    QTimer* m_diffReadoutTimer = nullptr;
    int m_bilateralStrength = 5;

    ColorMode m_colorMode = ColorMode::GRAY_CV;

    ManualAdjust m_manualAdj1;
    ManualAdjust m_manualAdj2;
    int m_activeAdjCam = 2;
    bool m_updatingAdjUI = false;

    QWidget* m_centralWidget;
    QWidget* m_videoArea;
    QStackedWidget* m_sheetStack = nullptr;
    QSplitter* m_mainSplit = nullptr;
    QSplitter* m_workSplit = nullptr;
    QStackedWidget* m_workStack = nullptr;
    QStackedWidget* m_bottomStack = nullptr;
    QWidget* m_galleryView = nullptr;
    QWidget* m_sideNav = nullptr;
    QWidget* m_navInner = nullptr;
    QPushButton* m_btnNavCollapse = nullptr;
    QPushButton* m_btnPreviewMode = nullptr;
    QPushButton* m_btnFloatingPreviewToggle = nullptr;
    QMap<NavItem, QPushButton*> m_navButtons;
    QListWidget* m_snapshotRecent = nullptr;

    QLabel* m_fpsPill;
    QLabel* m_eccPill;
    QPushButton* m_btnGallery;
    QPushButton* m_btnHelp;
    QPushButton* m_btnHelpDocs = nullptr;

    QPushButton* m_btnExpToggle = nullptr;
    QPushButton* m_btnExpChange = nullptr;
    QLabel* m_lblExposureVals = nullptr;
    QDoubleSpinBox* m_spnGain = nullptr;
    QSpinBox* m_spnShutter = nullptr;
    QSlider* m_sldGain = nullptr;
    QSlider* m_sldShutter = nullptr;
    QDoubleSpinBox* m_spnGain2 = nullptr;
    QSpinBox* m_spnShutter2 = nullptr;
    QSlider* m_sldGain2 = nullptr;
    QSlider* m_sldShutter2 = nullptr;
    bool m_manualExposure = false;
    bool m_perCameraExposure = false;
    int m_gainQ8 = 256;
    int m_shutterUs = 10000;
    int m_gain2Q8 = 256;
    int m_shutter2Us = 10000;
    void showExposureDialog();

    QPushButton* m_btnConfigParams = nullptr;
    QMap<QString, bool> m_paramInName;
    void showFilenameParamsDialog();

    ImageView* m_view1;
    ImageView* m_view2;
    ImageView* m_resultView;
    QSplitter* m_splitter;
    QPushButton* m_btnModeToggle;
    QPushButton* m_btnFocusToggle;
    QPushButton* m_btnFabSnapshot;
    QPushButton* m_btnFabStream;
    QComboBox* m_comboCamSet;
    QLabel* m_fabStreamIcon = nullptr;
    QLabel* m_fabSnapIcon = nullptr;

    QPushButton* m_btnToggleCameras;
    QComboBox* m_comboColorMode;
    QCheckBox* m_chkFlipVer2;
    QCheckBox* m_chkFlipHor2;
    QCheckBox* m_chkStretchView;
    QCheckBox* m_chkBilateral;
    QSlider* m_bilateralSlider;
    QLabel* m_bilateralLabel;
    QCheckBox* m_chkAppendParams;
    QCheckBox* m_chkSeparateFiles = nullptr;
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
    QCheckBox* m_chkEccPolish = nullptr;
    QCheckBox* m_chkPersistAlign = nullptr;
    QLabel* m_alignDetail = nullptr;

    QSlider* m_noiseFloorSlider;
    QLabel* m_noiseFloorLabel;
    QCheckBox* m_chkStretch;
    QPushButton* m_btnPeakIntensities;
    QLabel* m_lblPeakInfo;
    QDoubleSpinBox* m_spnChannelK = nullptr;
    QPushButton* m_btnCalibrateK = nullptr;
    QLabel* m_lblKInfo = nullptr;
    QSpinBox* m_spnClipLevel = nullptr;
    QLabel* m_lblDiffInfo = nullptr;

    QLabel* m_lblFocus1Big;
    QLabel* m_lblFocus2Big;
    QSpinBox* m_historySpinBox;
    QSlider* m_historySlider;
    QChartView* m_chartView;
    QChart* m_chart;
    QLineSeries* m_seriesCam1;
    QLineSeries* m_seriesCam2;

    QWidget* m_focusDataWidget;
    QListWidget* m_snapshotPreview = nullptr;
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

    int m_animSpeedMs = 220;
    bool m_animSidebarEnabled = true;
    bool m_animFABsEnabled = true;
    bool m_animBottomPanelEnabled = true;
    bool m_animTabsEnabled = true;
    bool m_animDialogsEnabled = true;

    QStatusBar* m_statusBar;

    std::thread m_calibThread;
    std::atomic<bool> m_calibrating{false};

    QStringList m_libCameraIds;
    std::vector<std::vector<SensorMode>> m_sensorModes;
    int m_requestedFps = 0;
    QTimer* m_exposureRestartTimer = nullptr;
    bool m_cameraTransition = false;
    bool m_libcameraSession = false;

    struct Thumbnail { QDateTime modified; qint64 size = 0; QPixmap pixmap; };
    QHash<QString, Thumbnail> m_thumbnails;

    ResourceMonitor* m_resMonitor = nullptr;
    QLabel* m_resPill = nullptr;
    QFrame* m_resHud = nullptr;
    QLabel* m_resHudText = nullptr;
    bool m_resHudVisible = false;
    qint64 m_displayedFrames = 0;
};

#endif
