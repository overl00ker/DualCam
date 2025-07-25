#include "MainWindow.h"
#include <QApplication>
#include <iostream>

int main(int argc, char *argv[])
{
    std::cout << "main.cpp entered" << std::endl;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <fps>\n";
        return 1;
    }

    int w = std::stoi(argv[1]);
    int h = std::stoi(argv[2]);
    int fps = std::stoi(argv[3]);

    QApplication app(argc, argv);

    CaptureParams p0{0, w, h, fps};
    CaptureParams p1{1, w, h, fps};

    MainWindow win(p0, p1);
    win.show();

    return app.exec();
}
