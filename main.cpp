#include <QApplication>
#include <QScreen>
#include "mainwindow.h"

static const char* DARK_STYLE = R"(
    QWidget {
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-size: 11px;
    }
    QGroupBox {
        border: 1px solid #3c3c3c;
        border-radius: 4px;
        margin-top: 6px;
        padding: 6px 4px 4px 4px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 6px;
        padding: 0 3px;
        color: #80bfff;
    }
    QPushButton {
        background-color: #2d2d2d;
        border: 1px solid #3c3c3c;
        border-radius: 3px;
        padding: 3px 6px;
        color: #d4d4d4;
    }
    QPushButton:hover { background-color: #3e3e3e; }
    QPushButton:pressed { background-color: #094771; }
    QPushButton:checked { background-color: #2196F3; color: white; border: none; }
    QPushButton:disabled { color: #5a5a5a; }
    QComboBox {
        background-color: #2d2d2d;
        border: 1px solid #3c3c3c;
        border-radius: 3px;
        padding: 2px 4px;
        color: #d4d4d4;
    }
    QComboBox::drop-down { border: none; }
    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        color: #d4d4d4;
        selection-background-color: #094771;
    }
    QCheckBox { color: #d4d4d4; spacing: 4px; }
    QCheckBox::indicator {
        width: 14px; height: 14px;
        border: 1px solid #5a5a5a;
        border-radius: 3px;
        background-color: #2d2d2d;
    }
    QCheckBox::indicator:hover { border-color: #0078d4; }
    QCheckBox::indicator:checked {
        background-color: #0078d4;
        border-color: #0078d4;
    }
    QSlider::groove:horizontal { background: #3c3c3c; height: 4px; border-radius: 2px; }
    QSlider::handle:horizontal { background: #0078d4; width: 14px; margin: -5px 0; border-radius: 7px; }
    QSpinBox 
    {
        background-color: #2d2d2d;
        border: 1px solid #3c3c3c;
        border-radius: 3px;
        padding: 1px 3px;
        color: #d4d4d4;
    }
    QStatusBar { background-color: #252525; color: #808080; border-top: 1px solid #3c3c3c; }
    QSplitter::handle { background-color: #3c3c3c; width: 2px; }
    QScrollArea { border: none; background-color: transparent; }
    QScrollBar:horizontal {
        background: #1e1e1e; height: 8px;
    }
    QScrollBar::handle:horizontal {
        background: #3c3c3c; border-radius: 4px; min-width: 20px;
    }
)";

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setStyleSheet(DARK_STYLE);

    MainWindow w;
    w.setWindowTitle("DualCam");

    QScreen* screen = QApplication::primaryScreen();
    QSize screenSize = screen->availableSize();
    if (screenSize.width() <= 1024) {
        w.showMaximized();
    } else {
        w.resize(std::min(1280, screenSize.width()), std::min(800, screenSize.height()));
        w.show();
    }

    return app.exec();
}
