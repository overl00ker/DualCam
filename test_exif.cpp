#include <QImage>
#include <QImageWriter>
#include <QCoreApplication>
#include <QDebug>
#include <iostream>

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    QImage img(100, 100, QImage::Format_RGB32);
    img.fill(Qt::red);
    img.setText("Exif/Image/Make", "DualCam");
    img.setText("Exif/Image/Model", "SuperCam");
    img.setText("Exif/Photo/FNumber", "1.8");
    img.setText("Exif/Photo/ExposureTime", "1/320");
    img.setText("Exif/Photo/ISOSpeedRatings", "100");
    bool res = img.save("test.jpg", "JPG");
    std::cout << "Saved: " << res << std::endl;
    return 0;
}
