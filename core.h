#ifndef DUALCAM_CORE_H
#define DUALCAM_CORE_H

#include <QString>
#include <QSize>
#include <QSettings>
#include <QWidget>
#include <QImage>
#include <opencv2/opencv.hpp>
#include <array>
#include <string>
#include <memory>

struct CaptureParams 
{
    std::string pipeline;
    int width{0};
    int height{0};
    int fps{0};
    bool gray{false};
    bool valid{false};
};

struct AppConfig 
{
    std::array<CaptureParams,2> cam;
    bool simpleMode{true};
    QSize resolution{1280,960};
    int fps{60};
    bool showDiff{false};
    bool showGraph{false};
};

QSettings* openSettings();
bool loadConfig(AppConfig&, QSettings&);
bool saveConfig(const AppConfig&, QSettings&);
void parseCliOverrides(int argc, char** argv, AppConfig&);
void parseEnvOverrides(AppConfig&);
std::string makeLibcameraPipe(int camId,int w,int h,int fps,bool gray);
std::string makeV4L2Pipe(const std::string& dev,const std::string& fmt,int w,int h,int fps,bool gray);
bool tryOpenLibcamera(int camId,CaptureParams&,const AppConfig&);
bool tryOpenV4L2(const std::string& dev,const std::string& fmt,CaptureParams&,const AppConfig&);
bool tryOpenIndex(int index,CaptureParams&,const AppConfig&);
bool autodetect(AppConfig&);

class CameraCapture 
{
public:
    ~CameraCapture();
    bool open(const CaptureParams&);
    bool isOpen() const;
    bool read(cv::Mat& dest);
    void close();
private:
    std::unique_ptr<cv::VideoCapture> m_cap;
};

class CameraWidget : public QWidget 
{
    Q_OBJECT
public:
    explicit CameraWidget(QWidget* parent=nullptr);
    void setFrame(const QImage&);
    void clearFrame();
protected:
    void paintEvent(QPaintEvent*) override;
private:
    QImage m_frame;
};

#endif