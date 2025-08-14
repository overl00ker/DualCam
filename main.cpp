#include <QApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QStringList>
#include <QFileInfo>
#include <iostream>
#include "MainWindow.h"
#include "capture_backend.h"

static bool parseSize(const QString& s, int& w, int& h) 
{
    const auto parts = s.split('x');
    if (parts.size() != 2) return false;
    bool ok1 = false, ok2 = false;
    int tw = parts[0].toInt(&ok1);
    int th = parts[1].toInt(&ok2);
    if (!ok1 || !ok2 || tw <= 0 || th <= 0) return false;
    w = tw; h = th; return true;
}

static Backend parseBackend(const QString& s) 
{
    const auto v = s.toLower();
    if (v == "libcamera" || v == "lib") return Backend::LIBCAMERA;
    if (v == "v4l2" || v == "v4l") return Backend::V4L2;
    return Backend::AUTO;
}

static void applyCamOpts(CaptureParams& p,
    const QCommandLineParser& parser,
    const QString& prefix)
{
    const auto bk = parser.value(QStringLiteral("%1-backend").arg(prefix));
    if (!bk.isEmpty()) p.backend = parseBackend(bk);

    const auto camId = parser.value(QStringLiteral("%1-id").arg(prefix));
    if (!camId.isEmpty()) p.cameraId = camId.toInt();

    const auto dev = parser.value(QStringLiteral("%1-device").arg(prefix));
    if (!dev.isEmpty()) p.device = dev.toStdString();

    const auto size = parser.value(QStringLiteral("%1-size").arg(prefix));
    if (!size.isEmpty()) 
    {
        int w, h;
        if (parseSize(size, w, h)) { p.width = w; p.height = h; }
        else std::cerr << "[args] Invalid " << prefix.toStdString() << "-size: " << size.toStdString() << "\n";
    }

    const auto fps = parser.value(QStringLiteral("%1-fps").arg(prefix));
    if (!fps.isEmpty()) p.fps = std::max(1, fps.toInt());

    if (parser.isSet(QStringLiteral("%1-gray").arg(prefix))) {
        const auto v = parser.value(QStringLiteral("%1-gray").arg(prefix));
        if (!v.isEmpty()) p.forceGray = (v != "0");
        else p.forceGray = true;
    }

    const auto fmt = parser.value(QStringLiteral("%1-v4l2-fmt").arg(prefix));
    if (!fmt.isEmpty()) p.v4l2PixelFmt = fmt.toStdString();

    const auto pipe = parser.value(QStringLiteral("%1-pipeline").arg(prefix));
    if (!pipe.isEmpty()) p.pipelineOverride = pipe.toStdString();

    const auto wms = parser.value(QStringLiteral("%1-warmup-ms").arg(prefix));
    if (!wms.isEmpty()) p.warmupMs = std::max(0, wms.toInt());

    if (parser.isSet(QStringLiteral("%1-verbose").arg(prefix))) {
        const auto v = parser.value(QStringLiteral("%1-verbose").arg(prefix));
        if (!v.isEmpty()) p.verbose = (v != "0");
        else p.verbose = true;
    }
}

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName("DualCam");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription(
        "Dual camera viewer (libcamera + V4L2). Any combination: 2×libcamera, 2×V4L2 или 1+1.\n"
        "Примеры:\n"
        "  ./DualCam                                   (2×libcamera — AUTO)\n"
        "  ./DualCam --cam0-backend=libcamera --cam1-backend=v4l2 --cam1-device=/dev/video1 --cam1-size=1440x1088 --cam1-v4l2-fmt=GRAY8\n"
        "  ./DualCam --cam0-backend=v4l2 --cam0-device=/dev/video0 --cam0-size=1440x1088 --cam0-v4l2-fmt=GRAY8 --cam1-backend=v4l2 --cam1-device=/dev/video1\n"
    );
    parser.addHelpOption();
    parser.addVersionOption();

    const QStringList cams = { "cam0", "cam1" };
    for (const auto& c : cams) 
    {
        parser.addOption({ QStringLiteral("%1-backend").arg(c),
                           "backend: auto|libcamera|v4l2.", "name" });
        parser.addOption({ QStringLiteral("%1-id").arg(c),
                           "libcamera camera-id.", "N" });
        parser.addOption({ QStringLiteral("%1-device").arg(c),
                           "V4L2 device (/dev/videoN).", "path" });
        parser.addOption({ QStringLiteral("%1-size").arg(c),
                           "Size of capture WxH.", "WxH" });
        parser.addOption({ QStringLiteral("%1-fps").arg(c),
                           "FPS.", "N" });
        parser.addOption({ QStringLiteral("%1-gray").arg(c),
                           "Mono (1|0).", "0|1" });
        parser.addOption({ QStringLiteral("%1-v4l2-fmt").arg(c),
                           "GStreamer caps format для v4l2src (напр. GRAY8,BGR,YUY2).", "FMT" });
        parser.addOption({ QStringLiteral("%1-pipeline").arg(c),
                           "Full override GStreamer-pipeline.", "GST" });
        parser.addOption({ QStringLiteral("%1-warmup-ms").arg(c),
                           "Time to warmup camerа.", "ms" });
        parser.addOption({ QStringLiteral("%1-verbose").arg(c),
                           "Verbose (1|0).", "0|1" });
    }

    parser.process(app);

    CaptureParams p0;
    p0.backend = Backend::AUTO;
    p0.cameraId = 0;
    p0.device = "/dev/video0";
    p0.width = 1280;
    p0.height = 960;
    p0.fps = 30;
    p0.forceGray = false;          
    p0.v4l2PixelFmt = "GRAY8";     
    p0.verbose = true;

    CaptureParams p1 = p0;
    p1.cameraId = 1;
    p1.device = "/dev/video1";

    applyCamOpts(p0, parser, "cam0");
    applyCamOpts(p1, parser, "cam1");

    MainWindow w(p0, p1);
    qInfo() << "Before show()";
    w.resize(1280, 720);
    w.move(50, 50);
    w.show();
    qInfo() << "After show()";
    return app.exec();

}
