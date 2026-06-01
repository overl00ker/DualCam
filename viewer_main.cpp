#include <QApplication>
#include <QCommandLineParser>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include "viewer_dialogs.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    QCommandLineParser parser;
    parser.setApplicationDescription("DualCam graph viewer");
    parser.addHelpOption();
    QCommandLineOption modeOpt({"m", "mode"},
        "Mode: profile | surface", "mode");
    QCommandLineOption payloadOpt({"p", "payload"},
        "Path to JSON payload file (deleted on read)", "payload");
    QCommandLineOption keepOpt("keep", "Keep payload after read");
    parser.addOption(modeOpt);
    parser.addOption(payloadOpt);
    parser.addOption(keepOpt);
    parser.process(app);

    const QString mode = parser.value(modeOpt);
    const QString path = parser.value(payloadOpt);
    if (mode.isEmpty() || path.isEmpty()) {
        QMessageBox::critical(nullptr, "DualCamViewer",
            "Missing --mode or --payload argument.");
        return 1;
    }

    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(nullptr, "DualCamViewer",
            "Cannot open payload:\n" + path);
        return 1;
    }
    QByteArray bytes = f.readAll();
    f.close();
    if (!parser.isSet(keepOpt)) QFile::remove(path);

    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(bytes, &err);
    if (doc.isNull() || !doc.isObject()) {
        QMessageBox::critical(nullptr, "DualCamViewer",
            "Invalid payload JSON: " + err.errorString());
        return 1;
    }
    QJsonObject obj = doc.object();

    QDialog* dlg = nullptr;
    if (mode == "profile") {
        dlg = makeProfileDialog(obj);
    } else if (mode == "surface") {
        dlg = makeSurfaceDialog(obj);
    } else {
        QMessageBox::critical(nullptr, "DualCamViewer", "Unknown mode: " + mode);
        return 1;
    }
    if (!dlg) return 1;
    dlg->show();
    return app.exec();
}
