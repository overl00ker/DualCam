#include <QApplication>
#include <QScreen>
#include <QIcon>
#include "mainwindow.h"


int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setWindowIcon(QIcon(":/icon.ico"));

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
