#pragma once

#include <QWidget>
#include <QKeyEvent>
#include <deque>
#include <opencv2/opencv.hpp>

class QLabel;
class QTimer;

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
    static std::vector<std::tuple<std::string, bool, bool>> buildCandidates(int index);
    static std::optional<cv::VideoCapture> openWithCandidates(const std::vector<std::tuple<std::string, bool, bool>>& cands,
                                                              int width, int height, double fps);
    static void warmup(cv::VideoCapture& cap, int ms);
    static bool readOnce(cv::VideoCapture& cap, cv::Mat& out);
    static double focusMeasure(const cv::Mat& bgr);
    static void drawFocusPlot(cv::Mat& plotImg,
                              const std::deque<double>& a,
                              const std::deque<double>& b,
                              int margin,
                              int thickness,
                              double eqIdx);
    static double findEqualIndex(const std::deque<double>& a, const std::deque<double>& b);
    static cv::Mat resizeFit(const cv::Mat& src, int W, int H);
    static QImage toQImageRGB(const cv::Mat& bgr);

    QLabel* view_;           
    QTimer* timer_;           
    cv::VideoCapture cap0_;   
    cv::VideoCapture cap1_;   
    bool ok0_ = false;        
    bool ok1_ = false;       
    int tileW_ = 960;         
    int tileH_ = 540;       
    double fps_ = 30.0;   
    int histCap_ = 120;       
    bool freezeHistory_ = false; 
    bool showGraph_ = true;      
    enum class ViewMode { Analytics4Q, SideBySide };
    ViewMode mode_ = ViewMode::Analytics4Q;
    std::deque<double> h0_;   
    std::deque<double> h1_;  
    cv::Mat grid_;            
    cv::Mat plotImg_;        
    cv::Mat black_;          
};
