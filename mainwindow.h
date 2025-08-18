#ifndef DUALCAM_MAINWINDOW_H
#define DUALCAM_MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <array>
#include <vector>
#include "core.h"

class SettingsDialog;
class GraphWidget;

class MainWindow:public QMainWindow{
    Q_OBJECT
public:
    MainWindow(int argc,char** argv,QWidget* parent=nullptr);
    ~MainWindow() override;
private slots:
    void updateFrames();
    void openSettingsDialog();
private:
    void initConfig(int argc,char** argv);
    void setupLayout();
    QImage matToQImage(const cv::Mat& mat) const;
    void persistConfig();
    AppConfig m_cfg;
    std::unique_ptr<QSettings> m_settings;
    std::array<CameraCapture,2> m_captures;
    std::array<CameraWidget*,2> m_camWidgets;
    CameraWidget* m_diffWidget{nullptr};
    GraphWidget* m_graphWidget{nullptr};
    QTimer m_updateTimer;
    SettingsDialog* m_settingsDialog{nullptr};
    std::array<cv::Mat,2> m_lastFrames;
    std::vector<float> m_diffHistory;
    size_t m_historyMax{100};
};

class GraphWidget:public QWidget{
    Q_OBJECT
public:
    explicit GraphWidget(QWidget* parent=nullptr):QWidget(parent){setMinimumHeight(100);setMinimumWidth(200);} 
    void setSamples(const std::vector<float>& samples){m_samples=samples;update();}
protected:
    void paintEvent(QPaintEvent*) override{
        QPainter painter(this); painter.fillRect(rect(),QColor(40,40,40)); if(m_samples.empty()) return; float maxVal=0.f; for(float v:m_samples) maxVal=std::max(maxVal,v); if(maxVal<=0.f) maxVal=1.f; int n=static_cast<int>(m_samples.size()); float xStep=width()/static_cast<float>(n-1); QPolygonF poly; poly.reserve(n); for(int i=0;i<n;++i){float norm=m_samples[i]/maxVal; float x=i*xStep; float y=height()-norm*height(); poly<<QPointF(x,y);} painter.setRenderHint(QPainter::Antialiasing,true); painter.setPen(QPen(Qt::green,2)); painter.drawPolyline(poly); painter.setPen(Qt::white); painter.drawText(10,20,QString("Max diff: %1").arg(maxVal,0,'f',2));
    }
private:
    std::vector<float> m_samples;
};

class SettingsDialog:public QDialog{
    Q_OBJECT
public:
    explicit SettingsDialog(AppConfig& cfg,QWidget* parent=nullptr);
signals:
    void configChanged();
private slots:
    void applyChanges();
    void resetToAutoDetect();
private:
    void buildUi();
    void loadFromConfig();
    void storeToConfig();
    AppConfig& m_config;
    QTabWidget* m_tabs{nullptr};
    QCheckBox* m_simpleModeCheck{nullptr};
    QSpinBox* m_widthSpin{nullptr};
    QSpinBox* m_heightSpin{nullptr};
    QSpinBox* m_fpsSpin{nullptr};
    QCheckBox* m_diffCheck{nullptr};
    QCheckBox* m_graphCheck{nullptr};
    struct CamControls{QLineEdit* pipelineEdit;QSpinBox* widthSpin;QSpinBox* heightSpin;QSpinBox* fpsSpin;QCheckBox* grayCheck;};
    std::array<CamControls,2> m_camControls;
    QPushButton* m_applyButton{nullptr};
    QPushButton* m_resetButton{nullptr};
    QPushButton* m_closeButton{nullptr};
};

#endif