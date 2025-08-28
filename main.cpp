#include <QApplication>
#include "DualCam.h"

int main(int argc, char* argv[]) 
{
    QApplication app(argc, argv);
    DualCam w;
    w.setWindowTitle("DualCam");
    w.resize(1920, 1080);
    w.show();
    return app.exec();
}
