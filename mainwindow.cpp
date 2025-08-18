#include "mainwindow.h"
#include <QApplication>
#include <QMenuBar>
#include <QStatusBar>
#include <QGridLayout>
#include <QWidget>
#include <QAction>
#include <QDateTime>
#include <QDebug>
#include <QTabWidget>
#include <QFormLayout>
#include <QCheckBox>
#include <QSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPainter>

MainWindow::MainWindow(int argc,char** argv,QWidget* parent):QMainWindow(parent),m_updateTimer(this){initConfig(argc,argv);setupLayout();connect(&m_updateTimer,&QTimer::timeout,this,&MainWindow::updateFrames);m_updateTimer.start(33);statusBar()->showMessage(tr("Ready"));}
MainWindow::~MainWindow(){if(m_settings)persistConfig();}
void MainWindow::initConfig(int argc,char** argv){m_settings.reset(openSettings());loadConfig(m_cfg,*m_settings);parseEnvOverrides(m_cfg);parseCliOverrides(argc,argv,m_cfg);bool needDetect=false;for(int i=0;i<2;++i){if(!m_cfg.cam[i].valid)needDetect=true;} if(needDetect)autodetect(m_cfg);persistConfig();}
void MainWindow::persistConfig(){if(m_settings)saveConfig(m_cfg,*m_settings);}
void MainWindow::setupLayout(){QWidget* old=centralWidget();if(old)old->deleteLater();QWidget* central=new QWidget;setCentralWidget(central);QGridLayout* grid=new QGridLayout;central->setLayout(grid);
    for(int i=0;i<2;++i)m_camWidgets[i]=new CameraWidget;
    m_diffWidget=new CameraWidget;
    m_graphWidget=new GraphWidget;
    if(m_cfg.simpleMode){grid->addWidget(m_camWidgets[0],0,0);grid->addWidget(m_camWidgets[1],0,1);m_diffWidget->hide();m_graphWidget->hide();}else{grid->addWidget(m_camWidgets[0],0,0);grid->addWidget(m_camWidgets[1],0,1);grid->addWidget(m_diffWidget,1,0);grid->addWidget(m_graphWidget,1,1);m_diffWidget->show();m_graphWidget->show();}
    grid->setRowStretch(0,1);grid->setRowStretch(1,1);grid->setColumnStretch(0,1);grid->setColumnStretch(1,1);
    QMenu* fileMenu=menuBar()->addMenu(tr("File"));QAction* settingsAct=fileMenu->addAction(tr("Settings…"));connect(settingsAct,&QAction::triggered,this,&MainWindow::openSettingsDialog);QAction* quitAct=fileMenu->addAction(tr("Quit"));connect(quitAct,&QAction::triggered,qApp,&QApplication::quit);
}
QImage MainWindow::matToQImage(const cv::Mat& mat) const{if(mat.empty())return QImage(); if(mat.type()==CV_8UC3){cv::Mat rgb;cv::cvtColor(mat,rgb,cv::COLOR_BGR2RGB);return QImage(rgb.data,rgb.cols,rgb.rows,(int)rgb.step,QImage::Format_RGB888).copy();}else if(mat.type()==CV_8UC1){return QImage(mat.data,mat.cols,mat.rows,(int)mat.step,QImage::Format_Grayscale8).copy();}else{cv::Mat rgb;if(mat.channels()==4){cv::cvtColor(mat,rgb,cv::COLOR_BGRA2RGB);return QImage(rgb.data,rgb.cols,rgb.rows,(int)rgb.step,QImage::Format_RGB888).copy();}return QImage();}}
void MainWindow::updateFrames(){bool newFrames=false;for(int i=0;i<2;++i){CaptureParams& cp=m_cfg.cam[i];if(!cp.valid){m_camWidgets[i]->clearFrame();continue;} if(!m_captures[i].isOpen())m_captures[i].open(cp);cv::Mat frame;if(m_captures[i].read(frame)){m_lastFrames[i]=frame;QImage img=matToQImage(frame);m_camWidgets[i]->setFrame(img);newFrames=true;}}
    if(!m_cfg.simpleMode&&m_cfg.showDiff){if(!m_lastFrames[0].empty()&&!m_lastFrames[1].empty()){cv::Mat diff;if(m_lastFrames[0].size()!=m_lastFrames[1].size()){cv::Mat r1,r2;cv::resize(m_lastFrames[0],r1,cv::Size(),0.5,0.5);cv::resize(m_lastFrames[1],r2,cv::Size(),0.5,0.5);cv::absdiff(r1,r2,diff);}else cv::absdiff(m_lastFrames[0],m_lastFrames[1],diff);
            cv::Scalar meanS=cv::mean(diff);float meanVal=diff.channels()==3?float((meanS[0]+meanS[1]+meanS[2])/3.0):float(meanS[0]);m_diffHistory.push_back(meanVal);if(m_diffHistory.size()>m_historyMax)m_diffHistory.erase(m_diffHistory.begin());cv::Mat diffAmp;diff.convertTo(diffAmp,CV_8UC3,4.0);QImage diffImg=matToQImage(diffAmp);m_diffWidget->setFrame(diffImg);if(m_cfg.showGraph){m_graphWidget->setSamples(m_diffHistory);} }}
void MainWindow::openSettingsDialog(){if(!m_settingsDialog){m_settingsDialog=new SettingsDialog(m_cfg,this);connect(m_settingsDialog,&SettingsDialog::configChanged,this,[this](){persistConfig();setupLayout();});}m_settingsDialog->show();m_settingsDialog->raise();m_settingsDialog->activateWindow();}

SettingsDialog::SettingsDialog(AppConfig& cfg,QWidget* parent):QDialog(parent),m_config(cfg){setWindowTitle(tr("Settings"));buildUi();loadFromConfig();connect(m_applyButton,&QPushButton::clicked,this,&SettingsDialog::applyChanges);connect(m_resetButton,&QPushButton::clicked,this,&SettingsDialog::resetToAutoDetect);connect(m_closeButton,&QPushButton::clicked,this,&QDialog::accept);}
void SettingsDialog::buildUi(){QVBoxLayout* mainLayout=new QVBoxLayout(this);m_tabs=new QTabWidget;mainLayout->addWidget(m_tabs);QWidget* appTab=new QWidget;QFormLayout* appForm=new QFormLayout(appTab);m_simpleModeCheck=new QCheckBox(tr("Simple mode (2× view)"));appForm->addRow(m_simpleModeCheck);m_widthSpin=new QSpinBox;m_widthSpin->setRange(160,3840);m_widthSpin->setSingleStep(10);appForm->addRow(tr("Resolution width"),m_widthSpin);m_heightSpin=new QSpinBox;m_heightSpin->setRange(120,2160);m_heightSpin->setSingleStep(10);appForm->addRow(tr("Resolution height"),m_heightSpin);m_fpsSpin=new QSpinBox;m_fpsSpin->setRange(1,120);m_fpsSpin->setValue(30);appForm->addRow(tr("FPS"),m_fpsSpin);m_diffCheck=new QCheckBox(tr("Show difference"));appForm->addRow(m_diffCheck);m_graphCheck=new QCheckBox(tr("Show graph"));appForm->addRow(m_graphCheck);m_tabs->addTab(appTab,tr("App"));for(int i=0;i<2;++i){QWidget* camTab=new QWidget;QFormLayout* camForm=new QFormLayout(camTab);CamControls& ctrls=m_camControls[i];ctrls.pipelineEdit=new QLineEdit;camForm->addRow(tr("Pipeline"),ctrls.pipelineEdit);ctrls.widthSpin=new QSpinBox;ctrls.widthSpin->setRange(160,3840);camForm->addRow(tr("Width"),ctrls.widthSpin);ctrls.heightSpin=new QSpinBox;ctrls.heightSpin->setRange(120,2160);camForm->addRow(tr("Height"),ctrls.heightSpin);ctrls.fpsSpin=new QSpinBox;ctrls.fpsSpin->setRange(1,120);camForm->addRow(tr("FPS"),ctrls.fpsSpin);ctrls.grayCheck=new QCheckBox(tr("Grayscale"));camForm->addRow(ctrls.grayCheck);m_tabs->addTab(camTab,tr("Cam%1").arg(i));}
    QHBoxLayout* buttonRow=new QHBoxLayout;m_applyButton=new QPushButton(tr("Apply"));m_resetButton=new QPushButton(tr("Reset"));m_closeButton=new QPushButton(tr("Close"));buttonRow->addStretch(1);buttonRow->addWidget(m_applyButton);buttonRow->addWidget(m_resetButton);buttonRow->addWidget(m_closeButton);mainLayout->addLayout(buttonRow);
}
void SettingsDialog::loadFromConfig(){m_simpleModeCheck->setChecked(m_config.simpleMode);m_widthSpin->setValue(m_config.resolution.width());m_heightSpin->setValue(m_config.resolution.height());m_fpsSpin->setValue(m_config.fps);m_diffCheck->setChecked(m_config.showDiff);m_graphCheck->setChecked(m_config.showGraph);for(int i=0;i<2;++i){const CaptureParams& cp=m_config.cam[i];CamControls& ctrls=m_camControls[i];ctrls.pipelineEdit->setText(QString::fromStdString(cp.pipeline));ctrls.widthSpin->setValue(cp.width>0?cp.width:m_config.resolution.width());ctrls.heightSpin->setValue(cp.height>0?cp.height:m_config.resolution.height());ctrls.fpsSpin->setValue(cp.fps>0?cp.fps:m_config.fps);ctrls.grayCheck->setChecked(cp.gray);} }
void SettingsDialog::storeToConfig(){m_config.simpleMode=m_simpleModeCheck->isChecked();m_config.resolution=QSize(m_widthSpin->value(),m_heightSpin->value());m_config.fps=m_fpsSpin->value();m_config.showDiff=m_diffCheck->isChecked();m_config.showGraph=m_graphCheck->isChecked();for(int i=0;i<2;++i){CaptureParams& cp=m_config.cam[i];const CamControls& ctrls=m_camControls[i];QString pipeline=ctrls.pipelineEdit->text();cp.pipeline=pipeline.toStdString();cp.width=ctrls.widthSpin->value();cp.height=ctrls.heightSpin->value();cp.fps=ctrls.fpsSpin->value();cp.gray=ctrls.grayCheck->isChecked();cp.valid=!pipeline.isEmpty();}}
void SettingsDialog::applyChanges(){storeToConfig();emit configChanged();}
void SettingsDialog::resetToAutoDetect(){for(int i=0;i<2;++i){m_config.cam[i].pipeline.clear();m_config.cam[i].valid=false;}autodetect(m_config);loadFromConfig();emit configChanged();}