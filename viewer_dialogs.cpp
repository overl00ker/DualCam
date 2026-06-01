#include "viewer_dialogs.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QByteArray>
#include <QString>
#include <QStringList>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QColorDialog>
#include <QPainter>
#include <QPen>
#include <QBrush>
#include <QColor>
#include <QLinearGradient>
#include <QDir>
#include <QDateTime>
#include <QTextStream>
#include <QPointer>
#include <QSurfaceFormat>
#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QtDataVisualization/Q3DSurface>
#include <QtDataVisualization/QSurface3DSeries>
#include <QtDataVisualization/QSurfaceDataProxy>
#include <QtDataVisualization/QValue3DAxis>
#include <QtDataVisualization/QAbstract3DInputHandler>
#include <QtDataVisualization/Q3DInputHandler>
#include <QtDataVisualization/Q3DCamera>
#include <QtDataVisualization/Q3DScene>
#include <QtDataVisualization/Q3DTheme>
#include <QMouseEvent>
#include <QWheelEvent>
#include <vector>
#include <algorithm>
#include <cmath>

class SurfaceInputHandler : public QAbstract3DInputHandler {
    Q_OBJECT
public:
    explicit SurfaceInputHandler(QObject* parent = nullptr) : QAbstract3DInputHandler(parent) {}
    void onClick(std::function<void(QPoint)> cb) { m_clickCb = std::move(cb); }
protected:
    void mousePressEvent(QMouseEvent* ev, const QPoint& pos) override {
        QAbstract3DInputHandler::mousePressEvent(ev, pos);
        m_last = pos;
        if (ev->button() == Qt::LeftButton) {
            m_lmbDown = true;
            m_pressPos = pos;
            m_rotating = false;
        }
        if (ev->button() == Qt::RightButton) m_panning = true;
    }
    void mouseReleaseEvent(QMouseEvent* ev, const QPoint& pos) override {
        QAbstract3DInputHandler::mouseReleaseEvent(ev, pos);
        if (ev->button() == Qt::LeftButton) {
            if (m_lmbDown && !m_rotating && m_clickCb) m_clickCb(m_pressPos);
            m_lmbDown = false;
            m_rotating = false;
        }
        if (ev->button() == Qt::RightButton) m_panning = false;
    }
    void mouseMoveEvent(QMouseEvent* ev, const QPoint& pos) override {
        QAbstract3DInputHandler::mouseMoveEvent(ev, pos);
        if (!scene() || !scene()->activeCamera()) return;
        Q3DCamera* cam = scene()->activeCamera();
        const QPoint d = pos - m_last;
        m_last = pos;
        if (m_lmbDown && !m_rotating) {
            const QPoint t = pos - m_pressPos;
            if (std::abs(t.x()) + std::abs(t.y()) > 4) m_rotating = true;
        }
        if (m_rotating) {
            cam->setXRotation(cam->xRotation() + d.x() * 0.5f);
            cam->setYRotation(cam->yRotation() + d.y() * 0.5f);
        } else if (m_panning) {
            QVector3D tg = cam->target();
            const float lo = -1.0f, hi = 1.0f;
            float nx = std::clamp(tg.x() - d.x() * 0.004f, lo, hi);
            float ny = std::clamp(tg.y() + d.y() * 0.004f, lo, hi);
            tg.setX(nx);
            tg.setY(ny);
            cam->setTarget(tg);
        }
    }
    void wheelEvent(QWheelEvent* ev) override {
        QAbstract3DInputHandler::wheelEvent(ev);
        if (!scene() || !scene()->activeCamera()) return;
        Q3DCamera* cam = scene()->activeCamera();
        float zl = cam->zoomLevel();
        zl += ev->angleDelta().y() * 0.05f;
        cam->setZoomLevel(std::clamp(zl, 30.0f, 500.0f));
    }
private:
    bool m_lmbDown = false;
    bool m_rotating = false;
    bool m_panning  = false;
    QPoint m_last;
    QPoint m_pressPos;
    std::function<void(QPoint)> m_clickCb;
};

QDialog* makeProfileDialog(const QJsonObject& obj, QWidget* parent) {
    const QString title  = obj.value("title").toString();
    const bool darkTheme = obj.value("dark").toBool(true);
    const int ax = obj.value("ax").toInt();
    const int ay = obj.value("ay").toInt();
    const int bx = obj.value("bx").toInt();
    const int by = obj.value("by").toInt();
    const double length = obj.value("length").toDouble();

    QJsonArray samplesArr = obj.value("samples").toArray();
    struct ProfileSample { double dist; double xPix; double yPix; double intensity; };
    std::vector<ProfileSample> samples;
    samples.reserve(samplesArr.size());
    for (const QJsonValue& v : samplesArr) {
        QJsonArray row = v.toArray();
        if (row.size() < 4) continue;
        samples.push_back({ row[0].toDouble(), row[1].toDouble(), row[2].toDouble(), row[3].toDouble() });
    }

    QDialog* w = new QDialog(parent, Qt::Window);
    w->setAttribute(Qt::WA_DeleteOnClose);
    w->setWindowTitle(QString("Profile: %1").arg(title));
    w->resize(720, 420);

    const QColor bgCol     = darkTheme ? QColor("#0e0e0e") : QColor("#fafafa");
    const QColor textCol   = darkTheme ? QColor("#e5e2e1") : QColor("#101010");
    const QColor axisCol   = darkTheme ? QColor("#c8c6c5") : QColor("#404040");
    const QColor seriesCol = darkTheme ? QColor("#e5e2e1") : QColor("#101010");
    const QColor viewBg    = darkTheme ? QColor("#050505") : QColor("#ffffff");

    QVBoxLayout* lay = new QVBoxLayout(w);
    lay->setContentsMargins(8, 8, 8, 8);
    lay->setSpacing(8);

    QFrame* toolbar = new QFrame(w);
    QHBoxLayout* tlay = new QHBoxLayout(toolbar);
    tlay->setContentsMargins(10, 6, 10, 6);
    tlay->setSpacing(10);
    QPushButton* exportBtn = new QPushButton(QStringLiteral("Export CSV…"), toolbar);
    tlay->addWidget(exportBtn);
    tlay->addStretch();
    lay->addWidget(toolbar);

    QChart* chart = new QChart();
    chart->setTitle(QString("Intensity along (%1,%2) → (%3,%4)").arg(ax).arg(ay).arg(bx).arg(by));
    chart->setBackgroundBrush(QBrush(bgCol));
    chart->setTitleBrush(QBrush(textCol));

    QLineSeries* s = new QLineSeries();
    s->setName("Intensity");
    QPen pen(seriesCol); pen.setWidth(2); s->setPen(pen);
    for (const auto& sm : samples) s->append(sm.dist, sm.intensity);
    chart->addSeries(s);
    chart->legend()->setVisible(false);

    QValueAxis* axX = new QValueAxis();
    axX->setTitleText("distance, px");
    axX->setRange(0, length);
    axX->setLabelsBrush(QBrush(axisCol));
    axX->setTitleBrush(QBrush(axisCol));
    QValueAxis* axY = new QValueAxis();
    axY->setTitleText("intensity");
    axY->setRange(0, 255);
    axY->setLabelsBrush(QBrush(axisCol));
    axY->setTitleBrush(QBrush(axisCol));
    chart->addAxis(axX, Qt::AlignBottom);
    chart->addAxis(axY, Qt::AlignLeft);
    s->attachAxis(axX);
    s->attachAxis(axY);

    QChartView* view = new QChartView(chart, w);
    view->setRenderHint(QPainter::Antialiasing);
    view->setBackgroundBrush(QBrush(viewBg));
    lay->addWidget(view, 1);

    QObject::connect(exportBtn, &QPushButton::clicked, w, [w, samples, title, ax, ay, bx, by]() {
        const QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        const QString suggested = QString("profile_%1_%2.csv")
            .arg(title.isEmpty() ? "view" : title).arg(stamp);
        const QString dir = QDir::homePath();
        QString fileName = QFileDialog::getSaveFileName(w, "Export profile to CSV",
            dir + "/" + suggested,
            "CSV file (*.csv);;Tab-separated (*.tsv);;Text file (*.txt)");
        if (fileName.isEmpty()) return;
        const bool tabSep = fileName.endsWith(".tsv", Qt::CaseInsensitive);
        const QChar sep = tabSep ? QChar('\t') : QChar(',');
        if (!fileName.contains('.')) fileName += tabSep ? ".tsv" : ".csv";

        QFile f(fileName);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QMessageBox::warning(w, "Export CSV", "Cannot open file for writing:\n" + fileName);
            return;
        }
        QTextStream out(&f);
        out.setRealNumberPrecision(6);
        out << "# Source: " << title << "\n";
        out << "# Endpoints (px): (" << ax << ", " << ay << ") -> (" << bx << ", " << by << ")\n";
        out << "# Samples: " << samples.size() << "\n";
        out << "distance_px" << sep << "x_pix" << sep << "y_pix" << sep << "intensity\n";
        for (const auto& sm : samples) {
            out << sm.dist << sep << sm.xPix << sep << sm.yPix << sep << sm.intensity << "\n";
        }
        f.close();
    });

    return w;
}

QDialog* makeSurfaceDialog(const QJsonObject& obj, QWidget* parent) {
    const QString title  = obj.value("title").toString();
    const bool darkTheme = obj.value("dark").toBool(true);
    const int roiX0 = obj.value("roiX").toInt();
    const int roiY0 = obj.value("roiY").toInt();
    const double kBackX = obj.value("kx").toDouble(1.0);
    const double kBackY = obj.value("ky").toDouble(1.0);
    const int cols = obj.value("cols").toInt();
    const int rows = obj.value("rows").toInt();
    QByteArray gridBytes = QByteArray::fromBase64(obj.value("grid").toString().toLatin1());

    QDialog* w = new QDialog(parent, Qt::Window);
    w->setAttribute(Qt::WA_DeleteOnClose);
    w->setWindowTitle(QString("Surface: %1").arg(title));
    w->resize(800, 640);

    QVBoxLayout* lay = new QVBoxLayout(w);
    lay->setContentsMargins(8, 8, 8, 8);
    lay->setSpacing(8);

    if (cols < 2 || rows < 2 || gridBytes.size() < cols * rows) {
        lay->addWidget(new QLabel("ROI too small or payload invalid."));
        return w;
    }

    QFrame* toolbar = new QFrame(w);
    QHBoxLayout* tlay = new QHBoxLayout(toolbar);
    tlay->setContentsMargins(10, 6, 10, 6);
    tlay->setSpacing(10);

    QPushButton* lowColorBtn = new QPushButton("Low…", toolbar);
    QFrame* lowSwatch = new QFrame(toolbar);
    lowSwatch->setFixedSize(28, 20);
    QPushButton* highColorBtn = new QPushButton("High…", toolbar);
    QFrame* highSwatch = new QFrame(toolbar);
    highSwatch->setFixedSize(28, 20);

    QPushButton* saveSnapBtn = new QPushButton("Save snapshot…", toolbar);
    QPushButton* exportDataBtn = new QPushButton("Export CSV…", toolbar);

    tlay->addWidget(lowColorBtn);
    tlay->addWidget(lowSwatch);
    tlay->addSpacing(6);
    tlay->addWidget(highColorBtn);
    tlay->addWidget(highSwatch);
    tlay->addSpacing(10);
    tlay->addWidget(saveSnapBtn);
    tlay->addWidget(exportDataBtn);
    QLabel* coordLbl = new QLabel(toolbar);
    coordLbl->setMinimumWidth(220);
    coordLbl->setText("Click a point to read X, Y, intensity");
    tlay->addSpacing(12);
    tlay->addWidget(coordLbl);
    QLabel* hint = new QLabel("LMB click=pick / drag=rotate · RMB pan · wheel zoom", toolbar);
    tlay->addSpacing(12);
    tlay->addWidget(hint, 1);
    lay->addWidget(toolbar);

    Q3DSurface* surface = new Q3DSurface();
    surface->setShadowQuality(QAbstract3DGraph::ShadowQualityNone);
    surface->setSelectionMode(QAbstract3DGraph::SelectionItem);
    SurfaceInputHandler* inputHandler = new SurfaceInputHandler(surface);
    inputHandler->onClick([surface](QPoint p) {
        if (surface->scene()) surface->scene()->setSelectionQueryPosition(p);
    });
    surface->setActiveInputHandler(inputHandler);

    Q3DTheme* theme = new Q3DTheme(darkTheme ? Q3DTheme::ThemeQt : Q3DTheme::ThemePrimaryColors);
    if (darkTheme) {
        theme->setBackgroundColor(QColor("#050505"));
        theme->setWindowColor(QColor("#050505"));
        theme->setLabelTextColor(QColor("#e5e2e1"));
        theme->setLabelBackgroundColor(QColor(20, 20, 20, 200));
        theme->setLabelBorderEnabled(false);
        theme->setGridLineColor(QColor(255, 255, 255, 40));
    } else {
        theme->setBackgroundColor(QColor("#ffffff"));
        theme->setWindowColor(QColor("#fafafa"));
        theme->setLabelTextColor(QColor("#101010"));
        theme->setLabelBackgroundColor(QColor(245, 245, 245, 220));
        theme->setGridLineColor(QColor(0, 0, 0, 60));
    }
    surface->setActiveTheme(theme);

    QWidget* container = QWidget::createWindowContainer(surface, w);
    container->setMinimumSize(400, 300);
    container->setFocusPolicy(Qt::StrongFocus);
    lay->addWidget(container, 1);

    QSurfaceDataArray* data = new QSurfaceDataArray();
    data->reserve(rows);
    const uchar* grid = reinterpret_cast<const uchar*>(gridBytes.constData());
    for (int y = 0; y < rows; ++y) {
        QSurfaceDataRow* row = new QSurfaceDataRow(cols);
        for (int x = 0; x < cols; ++x) {
            const double z = grid[y * cols + x];
            (*row)[x].setPosition(QVector3D(float(x), float(z), float(y)));
        }
        data->append(row);
    }
    QSurface3DSeries* series = new QSurface3DSeries();
    series->setDrawMode(QSurface3DSeries::DrawSurfaceAndWireframe);
    series->setFlatShadingEnabled(false);
    QSurfaceDataProxy* proxy = new QSurfaceDataProxy();
    proxy->resetArray(data);
    series->setDataProxy(proxy);

    QObject::connect(series, &QSurface3DSeries::selectedPointChanged, w, [series, proxy, coordLbl](const QPoint& p) {
        if (p == QSurface3DSeries::invalidSelectionPosition() || !proxy->array()) {
            coordLbl->setText("Click a point to read X, Y, intensity");
            return;
        }
        const QSurfaceDataArray* arr = proxy->array();
        if (p.x() < 0 || p.x() >= arr->size()) return;
        const QSurfaceDataRow* row = arr->at(p.x());
        if (!row || p.y() < 0 || p.y() >= row->size()) return;
        const QVector3D v = row->at(p.y()).position();
        coordLbl->setText(QString("X=%1  Y=%2  intensity=%3")
                          .arg(v.x(), 0, 'f', 0)
                          .arg(v.z(), 0, 'f', 0)
                          .arg(v.y(), 0, 'f', 0));
    });

    QColor* lowColor  = new QColor(darkTheme ? QColor(58, 58, 58) : QColor(232, 232, 232));
    QColor* highColor = new QColor(78, 201, 176);
    QObject::connect(w, &QObject::destroyed, [lowColor, highColor]{ delete lowColor; delete highColor; });

    auto applyGradient = [series, lowSwatch, highSwatch, lowColor, highColor]() {
        QLinearGradient gr;
        gr.setColorAt(0.0, *lowColor);
        gr.setColorAt(1.0, *highColor);
        series->setBaseGradient(gr);
        series->setColorStyle(Q3DTheme::ColorStyleRangeGradient);
        lowSwatch->setStyleSheet(QString("background:%1; border:1px solid rgba(255,255,255,0.15);").arg(lowColor->name()));
        highSwatch->setStyleSheet(QString("background:%1; border:1px solid rgba(255,255,255,0.15);").arg(highColor->name()));
    };
    applyGradient();

    auto openPicker = [w, applyGradient](QColor* slot, const QString& ttl) {
        QColorDialog* d = new QColorDialog(*slot, w);
        d->setWindowFlag(Qt::Window);
        d->setAttribute(Qt::WA_DeleteOnClose);
        d->setOption(QColorDialog::NoButtons, true);
        d->setWindowTitle(ttl);
        QObject::connect(d, &QColorDialog::currentColorChanged, w, [slot, applyGradient](const QColor& c) {
            if (c.isValid()) { *slot = c; applyGradient(); }
        });
        d->show();
    };
    QObject::connect(lowColorBtn,  &QPushButton::clicked, w, [openPicker, lowColor]()  { openPicker(lowColor,  "Low color");  });
    QObject::connect(highColorBtn, &QPushButton::clicked, w, [openPicker, highColor]() { openPicker(highColor, "High color"); });

    QByteArray gridCopy = gridBytes.left(rows * cols);
    QObject::connect(exportDataBtn, &QPushButton::clicked, w,
        [w, gridCopy, cols, rows, kBackX, kBackY, roiX0, roiY0, title]() {
        const QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        const QString suggested = QString("surface_%1_%2.csv")
            .arg(title.isEmpty() ? "view" : title).arg(stamp);
        const QString dir = QDir::homePath();
        QString fileName = QFileDialog::getSaveFileName(w, "Export surface to CSV",
            dir + "/" + suggested,
            "Long form CSV — x,y,intensity (*.csv);;"
            "Matrix CSV — rows=Y, cols=X (*_matrix.csv);;"
            "Tab-separated long form (*.tsv)",
            nullptr);
        if (fileName.isEmpty()) return;

        const bool matrixForm = fileName.contains("_matrix", Qt::CaseInsensitive)
                              || fileName.endsWith("_matrix.csv", Qt::CaseInsensitive);
        const bool tabSep     = fileName.endsWith(".tsv", Qt::CaseInsensitive);
        if (!fileName.contains('.')) fileName += matrixForm ? "_matrix.csv"
                                              : (tabSep ? ".tsv" : ".csv");
        const QChar sep = tabSep ? QChar('\t') : QChar(',');

        QFile f(fileName);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QMessageBox::warning(w, "Export CSV", "Cannot open file for writing:\n" + fileName);
            return;
        }
        QTextStream out(&f);
        out << "# Source: " << title << "\n";
        out << "# ROI offset in source image (px): x0=" << roiX0 << ", y0=" << roiY0 << "\n";
        out << "# Grid: " << cols << " cols x " << rows << " rows\n";
        out << "# Pixel scale (source px per sample): kx=" << kBackX << ", ky=" << kBackY << "\n";

        const uchar* g = reinterpret_cast<const uchar*>(gridCopy.constData());
        if (matrixForm) {
            out << "y\\x";
            for (int x = 0; x < cols; ++x) {
                const double sx = roiX0 + (x + 0.5) * kBackX;
                out << sep << sx;
            }
            out << "\n";
            for (int y = 0; y < rows; ++y) {
                const double sy = roiY0 + (y + 0.5) * kBackY;
                out << sy;
                for (int x = 0; x < cols; ++x) {
                    out << sep << int(g[y * cols + x]);
                }
                out << "\n";
            }
        } else {
            out << "x_pix" << sep << "y_pix" << sep << "intensity\n";
            for (int y = 0; y < rows; ++y) {
                const double sy = roiY0 + (y + 0.5) * kBackY;
                for (int x = 0; x < cols; ++x) {
                    const double sx = roiX0 + (x + 0.5) * kBackX;
                    out << sx << sep << sy << sep << int(g[y * cols + x]) << "\n";
                }
            }
        }
        f.close();
    });

    QObject::connect(saveSnapBtn, &QPushButton::clicked, w, [w, surface, title]() {
        QString suggested = QString("surface_%1_%2.png")
            .arg(title.isEmpty() ? "view" : title)
            .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
        QString dir = QDir::homePath();
        QString fileName = QFileDialog::getSaveFileName(w, "Save 3D snapshot",
            dir + "/" + suggested,
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg)");
        if (fileName.isEmpty()) return;
        if (!fileName.endsWith(".png", Qt::CaseInsensitive)
            && !fileName.endsWith(".jpg", Qt::CaseInsensitive)
            && !fileName.endsWith(".jpeg", Qt::CaseInsensitive)) {
            fileName += ".png";
        }
        QImage img = surface->renderToImage(8, surface->size());
        if (img.isNull()) {
            QMessageBox::warning(w, "Save 3D snapshot", "Failed to render image.");
            return;
        }
        if (!img.save(fileName)) {
            QMessageBox::warning(w, "Save 3D snapshot", "Failed to save:\n" + fileName);
        }
    });

    surface->axisX()->setTitle("X");
    surface->axisY()->setTitle("intensity");
    surface->axisZ()->setTitle("Y");
    surface->axisX()->setTitleVisible(true);
    surface->axisY()->setTitleVisible(true);
    surface->axisZ()->setTitleVisible(true);
    surface->axisY()->setRange(0.0f, 255.0f);
    surface->addSeries(series);
    return w;
}

#include "viewer_dialogs.moc"
