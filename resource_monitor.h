#ifndef RESOURCE_MONITOR_H
#define RESOURCE_MONITOR_H

#include <QObject>
#include <QElapsedTimer>
#include <QString>
#include <QVector>
#include <functional>

class QTimer;

namespace PiThrottle {
    constexpr unsigned UnderVoltageNow     = 0x1;
    constexpr unsigned FreqCappedNow       = 0x2;
    constexpr unsigned ThrottledNow        = 0x4;
    constexpr unsigned SoftTempLimitNow    = 0x8;
    constexpr unsigned NowMask             = 0xF;
    constexpr unsigned OccurredShift       = 16;
}

struct ResourceStats {
    double cpuTotal = -1;
    QVector<double> cpuCores;
    double appCpu = -1;
    double guiThreadCpu = -1;
    double workerThreadCpu = -1;
    double tempC = -1;
    int armMHz = -1;
    int armMaxMHz = -1;
    int fanRpm = -1;
    qint64 throttleFlags = -1;
    qint64 memUsedMB = -1;
    qint64 memTotalMB = -1;
    qint64 appRssMB = -1;
    double processedFps = -1;
    double displayedFps = -1;
    double droppedFps = -1;
};

class ResourceMonitor : public QObject {
    Q_OBJECT
public:
    struct FrameCounters {
        qint64 processed = 0;
        qint64 displayed = 0;
        qint64 dropped = 0;
        int workerThreadId = 0;
    };

    explicit ResourceMonitor(std::function<FrameCounters()> counters, QObject* parent = nullptr);
    ~ResourceMonitor() override;

    void start(int intervalMs = 1000);
    const ResourceStats& last() const { return m_last; }

    static QString throttleText(qint64 flags, bool sinceBoot);

signals:
    void updated(const ResourceStats& stats);

private:
    struct CpuTimes { quint64 busy = 0; quint64 total = 0; };

    void sample();
    bool readCpuTimes(QVector<CpuTimes>& out) const;
    qint64 readThrottleFlags();

    std::function<FrameCounters()> m_counters;
    QTimer* m_timer = nullptr;
    QElapsedTimer m_clock;
    qint64 m_lastWallMs = -1;

    QVector<CpuTimes> m_lastCpu;
    double m_lastAppSec = -1;
    double m_lastGuiSec = -1;
    double m_lastWorkerSec = -1;
    int m_lastWorkerTid = 0;
    FrameCounters m_lastFrames;

    QString m_fanPath;
    int m_vcioFd = -1;
    bool m_vcioTried = false;

    ResourceStats m_last;
};

#endif
