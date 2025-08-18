#include "core.h"
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <algorithm>

QSettings* openSettings()
{
    QString appDir=QCoreApplication::applicationDirPath();
    QString portableIni=appDir+QDir::separator()+QStringLiteral("config.ini");
    if(QFile::exists(portableIni))return new QSettings(portableIni,QSettings::IniFormat);
    return new QSettings(QSettings::IniFormat,QSettings::UserScope,QStringLiteral("DualCam"),QStringLiteral("DualCam"));
}

bool loadConfig(AppConfig& cfg,QSettings& settings)
{
    settings.beginGroup(QStringLiteral("App"));
    if(settings.contains(QStringLiteral("simpleMode")))cfg.simpleMode=settings.value(QStringLiteral("simpleMode")).toBool();
    if(settings.contains(QStringLiteral("resolutionWidth"))&&settings.contains(QStringLiteral("resolutionHeight"))){
        int w=settings.value(QStringLiteral("resolutionWidth")).toInt();
        int h=settings.value(QStringLiteral("resolutionHeight")).toInt();
        if(w>0&&h>0)cfg.resolution=QSize(w,h);
    }
    if(settings.contains(QStringLiteral("fps"))) 
    {
        int f=settings.value(QStringLiteral("fps")).toInt();
        if(f>0)cfg.fps=f;
    }
    if(settings.contains(QStringLiteral("showDiff")))cfg.showDiff=settings.value(QStringLiteral("showDiff")).toBool();
    if(settings.contains(QStringLiteral("showGraph")))cfg.showGraph=settings.value(QStringLiteral("showGraph")).toBool();
    settings.endGroup();
    for(int i=0;i<2;++i){
        QString group=QStringLiteral("CAM%1").arg(i);
        settings.beginGroup(group);
        CaptureParams& cp=cfg.cam[i];
        QString pipeline=settings.value(QStringLiteral("pipeline")).toString();
        if(!pipeline.isEmpty()){cp.pipeline=pipeline.toStdString();cp.valid=true;}
        int w=settings.value(QStringLiteral("width")).toInt();
        int h=settings.value(QStringLiteral("height")).toInt();
        int f=settings.value(QStringLiteral("fps")).toInt();
        bool gray=settings.value(QStringLiteral("gray")).toBool();
        if(w>0)cp.width=w;
        if(h>0)cp.height=h;
        if(f>0)cp.fps=f;
        cp.gray=gray;
        settings.endGroup();
    }
    return true;
}

bool saveConfig(const AppConfig& cfg,QSettings& settings)
{
    settings.beginGroup(QStringLiteral("App"));
    settings.setValue(QStringLiteral("simpleMode"),cfg.simpleMode);
    settings.setValue(QStringLiteral("resolutionWidth"),cfg.resolution.width());
    settings.setValue(QStringLiteral("resolutionHeight"),cfg.resolution.height());
    settings.setValue(QStringLiteral("fps"),cfg.fps);
    settings.setValue(QStringLiteral("showDiff"),cfg.showDiff);
    settings.setValue(QStringLiteral("showGraph"),cfg.showGraph);
    settings.endGroup();
    for(int i=0;i<2;++i)
    {
        QString group=QStringLiteral("CAM%1").arg(i);
        if(cfg.cam[i].valid){
            settings.beginGroup(group);
            settings.setValue(QStringLiteral("pipeline"),QString::fromStdString(cfg.cam[i].pipeline));
            settings.setValue(QStringLiteral("width"),cfg.cam[i].width);
            settings.setValue(QStringLiteral("height"),cfg.cam[i].height);
            settings.setValue(QStringLiteral("fps"),cfg.cam[i].fps);
            settings.setValue(QStringLiteral("gray"),cfg.cam[i].gray);
            settings.endGroup();
        }else{
            settings.remove(group);
        }
    }
    settings.sync();
    return !settings.status();
}

static std::string cstr_to_std(const char* cstr){return cstr?std::string(cstr):std::string();}
void parseCliOverrides(int argc,char** argv,AppConfig& cfg)
{
    for(int i=1;i<argc;++i)
    {
        const char* arg=argv[i];
        if(strcmp(arg,"--cam0")==0&&i+1<argc){cfg.cam[0].pipeline=cstr_to_std(argv[++i]);cfg.cam[0].valid=true;}
        else if(strcmp(arg,"--cam1")==0&&i+1<argc){cfg.cam[1].pipeline=cstr_to_std(argv[++i]);cfg.cam[1].valid=true;}
        else if((strcmp(arg,"--width")==0||strcmp(arg,"-w")==0)&&i+1<argc){int w=atoi(argv[++i]);if(w>0)cfg.resolution.setWidth(w);} 
        else if((strcmp(arg,"--height")==0||strcmp(arg,"-h")==0)&&i+1<argc){int h=atoi(argv[++i]);if(h>0)cfg.resolution.setHeight(h);} 
        else if(strcmp(arg,"--fps")==0&&i+1<argc){int f=atoi(argv[++i]);if(f>0)cfg.fps=f;} 
        else if(strcmp(arg,"--simpleMode")==0){cfg.simpleMode=true;} 
        else if(strcmp(arg,"--no-simpleMode")==0){cfg.simpleMode=false;} 
        else if(strcmp(arg,"--showDiff")==0){cfg.showDiff=true;} 
        else if(strcmp(arg,"--no-showDiff")==0){cfg.showDiff=false;} 
        else if(strcmp(arg,"--showGraph")==0){cfg.showGraph=true;} 
        else if(strcmp(arg,"--no-showGraph")==0){cfg.showGraph=false;} 
    }
}
void parseEnvOverrides(AppConfig& cfg)
{
    auto env=[&](const char* n){return std::getenv(n);} ;
    if(const char* p0=env("DUALCAM_CAM0")){cfg.cam[0].pipeline=p0;cfg.cam[0].valid=true;}
    if(const char* p1=env("DUALCAM_CAM1")){cfg.cam[1].pipeline=p1;cfg.cam[1].valid=true;}
    if(const char* w=env("DUALCAM_WIDTH")){int iw=atoi(w);if(iw>0)cfg.resolution.setWidth(iw);} 
    if(const char* h=env("DUALCAM_HEIGHT")){int ih=atoi(h);if(ih>0)cfg.resolution.setHeight(ih);} 
    if(const char* r=env("DUALCAM_RESOLUTION")){
        QString res=QString::fromUtf8(r);
        auto parts=res.split('x');
        if(parts.size()==2){int iw=parts[0].toInt();int ih=parts[1].toInt();if(iw>0&&ih>0){cfg.resolution.setWidth(iw);cfg.resolution.setHeight(ih);}}
    }
    if(const char* f=env("DUALCAM_FPS")){int ifps=atoi(f);if(ifps>0)cfg.fps=ifps;} 
    if(const char* sm=env("DUALCAM_SIMPLEMODE")){QString v=QString::fromUtf8(sm).toLower();if(v=="1"||v=="true"||v=="yes")cfg.simpleMode=true;else if(v=="0"||v=="false"||v=="no")cfg.simpleMode=false;}
    if(const char* sd=env("DUALCAM_SHOWDIFF")){QString v=QString::fromUtf8(sd).toLower();if(v=="1"||v=="true"||v=="yes")cfg.showDiff=true;else if(v=="0"||v=="false"||v=="no")cfg.showDiff=false;}
    if(const char* sg=env("DUALCAM_SHOWGRAPH")){QString v=QString::fromUtf8(sg).toLower();if(v=="1"||v=="true"||v=="yes")cfg.showGraph=true;else if(v=="0"||v=="false"||v=="no")cfg.showGraph=false;}
}

static bool tryOpenPipeline(const std::string& pipeline,CaptureParams& out,bool grayExpected,const AppConfig& cfg){
    cv::VideoCapture cap;
#ifdef __linux__
    cap.open(pipeline,cv::CAP_GSTREAMER);
#else
    cap.open(pipeline);
#endif
    if(!cap.isOpened())return false;
    cv::Mat frame;
    for(int i=0;i<5;++i){if(!cap.read(frame))continue; if(!frame.empty())break;}
    if(frame.empty()){cap.release();return false;}
    int w=frame.cols;int h=frame.rows;int minDim=std::min(w,h);
    if(minDim<480){cap.release();return false;}
    out.pipeline=pipeline;out.width=w;out.height=h;double fps=cap.get(cv::CAP_PROP_FPS);out.fps=fps>0?static_cast<int>(fps):cfg.fps;out.gray=grayExpected;out.valid=true;cap.release();return true;
}

std::string makeLibcameraPipe(int camId,int w,int h,int fps,bool gray)
{
    std::string caps="video/x-raw";
    if(gray)caps+=", format=GRAY8";
    caps+=", width="+std::to_string(w)+", height="+std::to_string(h);
    caps+=", framerate="+std::to_string(fps)+"/1";
    return std::string("libcamerasrc camera-id=")+std::to_string(camId)+" ! "+caps+" ! videoconvert ! appsink drop=true max-buffers=1 sync=false";
}
std::string makeV4L2Pipe(const std::string& dev,const std::string& fmt,int w,int h,int fps,bool gray)
{
    std::string caps="video/x-raw,format="+fmt+",width="+std::to_string(w)+",height="+std::to_string(h)+",framerate="+std::to_string(fps)+"/1";
    return std::string("v4l2src device=")+dev+" ! "+caps+" ! videoconvert ! appsink drop=true max-buffers=1 sync=false";
}
bool tryOpenLibcamera(int camId,CaptureParams& out,const AppConfig& cfg){
    std::vector<int> fpsList={cfg.fps}; if(cfg.fps>15)fpsList.push_back(cfg.fps/2);
    for(int f:fpsList){for(bool gray:{false,true}){
        std::string pipeline=makeLibcameraPipe(camId,cfg.resolution.width(),cfg.resolution.height(),f,gray);
        if(tryOpenPipeline(pipeline,out,gray,cfg))return true;
    }}return false;
}
bool tryOpenV4L2(const std::string& dev,const std::string& fmt,CaptureParams& out,const AppConfig& cfg){
    std::vector<int> fpsList={cfg.fps}; if(cfg.fps>15)fpsList.push_back(cfg.fps/2);
    bool gray=(fmt=="GRAY8");
    for(int f:fpsList){std::string pipeline=makeV4L2Pipe(dev,fmt,cfg.resolution.width(),cfg.resolution.height(),f,gray); if(tryOpenPipeline(pipeline,out,gray,cfg))return true;}
    return false;
}
bool tryOpenIndex(int index,CaptureParams& out,const AppConfig& cfg){
    cv::VideoCapture cap(index);
    if(!cap.isOpened())return false;
    cap.set(cv::CAP_PROP_FRAME_WIDTH,cfg.resolution.width());
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,cfg.resolution.height());
    cap.set(cv::CAP_PROP_FPS,cfg.fps);
    cv::Mat frame;
    for(int i=0;i<5;++i){if(!cap.read(frame))continue; if(!frame.empty())break;}
    if(frame.empty()){cap.release();return false;}
    int w=frame.cols;int h=frame.rows; if(std::min(w,h)<480){cap.release();return false;}
    out.pipeline=std::to_string(index); out.width=w; out.height=h; double fps=cap.get(cv::CAP_PROP_FPS); out.fps=fps>0?static_cast<int>(fps):cfg.fps; out.gray=false; out.valid=true; cap.release(); return true;
}
bool autodetect(AppConfig& cfg){
    bool any=false;
    for(int camIdx=0;camIdx<2;++camIdx){CaptureParams candidate;
        if(cfg.cam[camIdx].valid && !cfg.cam[camIdx].pipeline.empty()){any=true;continue;}
        if(tryOpenLibcamera(camIdx,candidate,cfg)){cfg.cam[camIdx]=candidate;any=true;continue;}
        for(const char* dev:{"/dev/video0","/dev/video1"}){
            for(const char* fmt:{"YUY2","GRAY8"}){
                if(tryOpenV4L2(dev,fmt,candidate,cfg)){cfg.cam[camIdx]=candidate;any=true;goto done_v4l;}
            }
        }
        done_v4l: if(cfg.cam[camIdx].valid)continue;
        for(int idx:{camIdx,0,1}){if(tryOpenIndex(idx,candidate,cfg)){cfg.cam[camIdx]=candidate;any=true;break;}}
    }return any;
}

CameraCapture::~CameraCapture(){close();}
bool CameraCapture::open(const CaptureParams& params){
    close();
    m_cap=std::make_unique<cv::VideoCapture>();
    const std::string& p=params.pipeline;
    bool numeric=!p.empty()&&std::all_of(p.begin(),p.end(),[](char c){return std::isdigit(c);});
    bool ok=false;
    if(numeric){int index=std::stoi(p); ok=m_cap->open(index); if(ok){m_cap->set(cv::CAP_PROP_FRAME_WIDTH,params.width);m_cap->set(cv::CAP_PROP_FRAME_HEIGHT,params.height);m_cap->set(cv::CAP_PROP_FPS,params.fps);}}
    else{
#ifdef __linux__
        ok=m_cap->open(p,cv::CAP_GSTREAMER);
#else
        ok=m_cap->open(p);
#endif
    }
    if(!ok){m_cap.reset();return false;}
    return true;
}
bool CameraCapture::isOpen() const{return m_cap&&m_cap->isOpened();}
bool CameraCapture::read(cv::Mat& dest){if(!isOpen())return false;bool ok=m_cap->read(dest);return ok&&!dest.empty();}
void CameraCapture::close(){if(m_cap){m_cap->release();m_cap.reset();}}

CameraWidget::CameraWidget(QWidget* parent):QWidget(parent){setMinimumSize(320,240);setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);} 
void CameraWidget::setFrame(const QImage& frame){m_frame=frame;update();}
void CameraWidget::clearFrame(){m_frame=QImage();update();}
void CameraWidget::paintEvent(QPaintEvent*){
    QPainter painter(this);
    painter.setRenderHint(QPainter::SmoothPixmapTransform,true);
    if(!m_frame.isNull()){
        QImage scaled=m_frame.scaled(size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
        QPoint pos((width()-scaled.width())/2,(height()-scaled.height())/2);
        painter.drawImage(pos,scaled);
    }else{
        painter.fillRect(rect(),QColor(80,80,80)); painter.setPen(Qt::white); painter.setFont(QFont("Sans",14,QFont::Bold)); QString text=tr("No input"); QRect textRect=painter.fontMetrics().boundingRect(text); QPoint pos((width()-textRect.width())/2,(height()-textRect.height())/2); painter.drawText(pos,text);
    }
}