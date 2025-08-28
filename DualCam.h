#pragma once

#include <QWidget>
#include <QKeyEvent>
#include <deque>
#include <opencv2/opencv.hpp>

class QLabel;
class QTimer;

// DualCam displays two camera feeds side by side (or in an analytics 4-quadrant layout),
// computes a simple focus/sharpness metric for each frame, plots a history graph,
// and shows an absolute difference heatmap.  The implementation builds a list of
// possible capture sources for each camera using environment variables and
// reasonable fallbacks.  It works on both Windows and Linux (including
// Raspberry Pi) and can capture from libcamera via GStreamer pipelines or
// directly via V4L2/DirectShow.
class DualCam : public QWidget {
    Q_OBJECT
public:
    explicit DualCam(QWidget* parent = nullptr);
    ~DualCam();

protected:
    void keyPressEvent(QKeyEvent* event) override;

private slots:
    void tick();

private:
    // Build a list of candidate capture specifications for a given camera index.
    static std::vector<std::tuple<std::string, bool, bool>> buildCandidates(int index);
    // Attempt to open the first working candidate from the list.  The returned
    // VideoCapture will be opened and warmed up, or an empty optional if none succeed.
    static std::optional<cv::VideoCapture> openWithCandidates(const std::vector<std::tuple<std::string, bool, bool>>& cands,
                                                              int width, int height, double fps);
    // Warm up the capture by reading frames for a short time to flush buffers.
    static void warmup(cv::VideoCapture& cap, int ms);
    // Read a single frame with timeout.  Returns true if a non-empty frame was read.
    static bool readOnce(cv::VideoCapture& cap, cv::Mat& out);
    // Compute a variance of Laplacian focus metric on a BGR image.
    static double focusMeasure(const cv::Mat& bgr);
    // Draw a history plot of two sequences into the given image.
    static void drawFocusPlot(cv::Mat& plotImg,
                              const std::deque<double>& a,
                              const std::deque<double>& b,
                              int margin,
                              int thickness,
                              double eqIdx);
    // Estimate the index (fractional) where the two sequences last crossed.
    static double findEqualIndex(const std::deque<double>& a, const std::deque<double>& b);
    // Resize src to fit inside (W,H) preserving aspect ratio and put it into a black canvas.
    static cv::Mat resizeFit(const cv::Mat& src, int W, int H);
    // Convert BGR (or grayscale) image to QImage (RGB888).
    static QImage toQImageRGB(const cv::Mat& bgr);

    QLabel* view_;            // label used to display the combined image
    QTimer* timer_;           // periodic timer for grabbing frames
    cv::VideoCapture cap0_;   // first camera
    cv::VideoCapture cap1_;   // second camera
    bool ok0_ = false;        // whether the first camera is opened
    bool ok1_ = false;        // whether the second camera is opened
    int tileW_ = 960;         // width of each tile in the 2x2 grid
    int tileH_ = 540;         // height of each tile in the 2x2 grid
    double fps_ = 30.0;       // target frames per second
    int histCap_ = 120;       // maximum number of history points for plots
    bool freezeHistory_ = false; // toggle for freezing the history graph
    bool showGraph_ = true;       // toggle for showing the history graph
    enum class ViewMode { Analytics4Q, SideBySide };
    ViewMode mode_ = ViewMode::Analytics4Q;
    std::deque<double> h0_;   // history of focus metrics for camera 0
    std::deque<double> h1_;   // history of focus metrics for camera 1
    cv::Mat grid_;            // combined 2x2 image
    cv::Mat plotImg_;         // plot image
    cv::Mat black_;           // black placeholder image
};