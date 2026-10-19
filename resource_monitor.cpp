#include "resource_monitor.h"

#include <QTimer>
#include <QFile>
#include <QDir>
#include <QStringList>
#include <QCoreApplication>

#ifdef Q_OS_LINUX
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#endif

#ifdef Q_OS_WIN
#include <windows.h>
#endif

namespace {

#ifdef Q_OS_LINUX
QByteArray readSmallFile(const QString& path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return QByteArray();
    return f.readAll();
}

qint64 readInt(const QString& path, qint64 fallback = -1)
{
    bool ok = false;
    const qint64 v = readSmallFile(path).trimmed().toLongLong(&ok);
    return ok ? v : fallback;
}

double statCpuSeconds(const QString& path)
{
    const QByteArray s = readSmallFile(path);
    const int rp = s.lastIndexOf(')');
    if (rp < 0) return -1;
    const QList<QByteArray> f = s.mid(rp + 2).split(' ');
    if (f.size() < 13) return -1;
    static const double tck = double(sysconf(_SC_CLK_TCK));
    return (f[11].toDouble() + f[12].toDouble()) / tck;
}
#endif

#ifdef Q_OS_WIN
quint64 fileTimeToU64(const FILETIME& ft)
{
    return (quint64(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
}
#endif

}

ResourceMonitor::ResourceMonitor(std::function<FrameCounters()> counters, QObject* parent)
    : QObject(parent), m_counters(std::move(counters))
{
#ifdef Q_OS_LINUX
    const QDir hwmon("/sys/class/hwmon");
    for (const QString& d : hwmon.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        const QString base = hwmon.filePath(d);
        if (readSmallFile(base + "/name").trimmed() == "pwmfan"
            && QFile::exists(base + "/fan1_input")) {
            m_fanPath = base + "/fan1_input";
            break;
        }
    }
#endif
}

ResourceMonitor::~ResourceMonitor()
{
#ifdef Q_OS_LINUX
    if (m_vcioFd >= 0) ::close(m_vcioFd);
#endif
}

void ResourceMonitor::start(int intervalMs)
{
    if (!m_timer) {
        m_timer = new QTimer(this);
        connect(m_timer, &QTimer::timeout, this, &ResourceMonitor::sample);
    }
    m_clock.start();
    sample();
    m_timer->start(intervalMs);
}

QString ResourceMonitor::throttleText(qint64 flags, bool sinceBoot)
{
    if (flags < 0) return QStringLiteral("n/a");
    const unsigned bits = unsigned(flags >> (sinceBoot ? PiThrottle::OccurredShift : 0));
    QStringList parts;
    if (bits & PiThrottle::ThrottledNow)     parts << QStringLiteral("throttled");
    if (bits & PiThrottle::FreqCappedNow)    parts << QStringLiteral("freq capped");
    if (bits & PiThrottle::SoftTempLimitNow) parts << QStringLiteral("soft temp limit");
    if (bits & PiThrottle::UnderVoltageNow)  parts << QStringLiteral("under-voltage");
    return parts.isEmpty() ? QStringLiteral("none") : parts.join(QStringLiteral(", "));
}

bool ResourceMonitor::readCpuTimes(QVector<CpuTimes>& out) const
{
    out.clear();
#ifdef Q_OS_LINUX
    const QList<QByteArray> lines = readSmallFile("/proc/stat").split('\n');
    for (const QByteArray& line : lines) {
        if (!line.startsWith("cpu")) break;
        const QList<QByteArray> f = line.simplified().split(' ');
        if (f.size() < 6) continue;
        CpuTimes t;
        for (int i = 1; i < f.size() && i <= 8; ++i) t.total += f[i].toULongLong();
        t.busy = t.total - f[4].toULongLong() - f[5].toULongLong();
        out.push_back(t);
    }
    return !out.isEmpty();
#elif defined(Q_OS_WIN)
    FILETIME idle, kernel, user;
    if (!GetSystemTimes(&idle, &kernel, &user)) return false;
    CpuTimes t;
    t.total = fileTimeToU64(kernel) + fileTimeToU64(user);
    t.busy  = t.total - fileTimeToU64(idle);
    out.push_back(t);
    return true;
#else
    return false;
#endif
}

qint64 ResourceMonitor::readThrottleFlags()
{
#ifdef Q_OS_LINUX
    if (!m_vcioTried) {
        m_vcioTried = true;
        m_vcioFd = ::open("/dev/vcio", O_RDWR | O_CLOEXEC);
    }
    if (m_vcioFd < 0) return -1;
    constexpr unsigned long kMboxProperty = _IOWR(100, 0, char*);
    constexpr quint32 kTagGetThrottled = 0x00030046;
    alignas(16) quint32 msg[8] = { sizeof(msg), 0, kTagGetThrottled, 4, 4, 0, 0, 0 };
    if (::ioctl(m_vcioFd, kMboxProperty, msg) < 0 || msg[1] != 0x80000000u) return -1;
    return qint64(msg[5]);
#else
    return -1;
#endif
}

void ResourceMonitor::sample()
{
    const qint64 nowMs = m_clock.elapsed();
    const double dt = m_lastWallMs >= 0 ? (nowMs - m_lastWallMs) / 1000.0 : 0.0;
    const bool haveDelta = dt > 0.05;

    ResourceStats st;
    const FrameCounters frames = m_counters ? m_counters() : FrameCounters();

    QVector<CpuTimes> cpu;
    if (readCpuTimes(cpu) && cpu.size() == m_lastCpu.size()) {
        auto pct = [](const CpuTimes& a, const CpuTimes& b) {
            const quint64 total = b.total - a.total;
            return total ? 100.0 * double(b.busy - a.busy) / double(total) : 0.0;
        };
        st.cpuTotal = pct(m_lastCpu[0], cpu[0]);
        for (int i = 1; i < cpu.size(); ++i) st.cpuCores.push_back(pct(m_lastCpu[i], cpu[i]));
    }
    m_lastCpu = cpu;

#ifdef Q_OS_LINUX
    const double appSec = statCpuSeconds("/proc/self/stat");
    const double guiSec = statCpuSeconds(QString("/proc/self/task/%1/stat").arg(::getpid()));
    const double workerSec = frames.workerThreadId > 0
        ? statCpuSeconds(QString("/proc/self/task/%1/stat").arg(frames.workerThreadId)) : -1;

    const qint64 milliC = readInt("/sys/class/thermal/thermal_zone0/temp");
    if (milliC >= 0) st.tempC = milliC / 1000.0;
    const qint64 curKHz = readInt("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    const qint64 maxKHz = readInt("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (curKHz > 0) st.armMHz = int(curKHz / 1000);
    if (maxKHz > 0) st.armMaxMHz = int(maxKHz / 1000);
    if (!m_fanPath.isEmpty()) st.fanRpm = int(readInt(m_fanPath));
    st.throttleFlags = readThrottleFlags();

    qint64 memTotalKB = -1, memAvailKB = -1;
    for (const QByteArray& line : readSmallFile("/proc/meminfo").split('\n')) {
        const QList<QByteArray> f = line.simplified().split(' ');
        if (f.size() < 2) continue;
        if (f[0] == "MemTotal:") memTotalKB = f[1].toLongLong();
        else if (f[0] == "MemAvailable:") memAvailKB = f[1].toLongLong();
        if (memTotalKB >= 0 && memAvailKB >= 0) break;
    }
    if (memTotalKB > 0 && memAvailKB >= 0) {
        st.memTotalMB = memTotalKB / 1024;
        st.memUsedMB = (memTotalKB - memAvailKB) / 1024;
    }
    const QList<QByteArray> statm = readSmallFile("/proc/self/statm").split(' ');
    if (statm.size() > 1)
        st.appRssMB = statm[1].toLongLong() * sysconf(_SC_PAGESIZE) / (1024 * 1024);
#elif defined(Q_OS_WIN)
    double appSec = -1, guiSec = -1, workerSec = -1;
    FILETIME c, e, k, u;
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u))
        appSec = double(fileTimeToU64(k) + fileTimeToU64(u)) / 1e7;
    if (GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u))
        guiSec = double(fileTimeToU64(k) + fileTimeToU64(u)) / 1e7;
    if (frames.workerThreadId > 0) {
        HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, DWORD(frames.workerThreadId));
        if (h) {
            if (GetThreadTimes(h, &c, &e, &k, &u))
                workerSec = double(fileTimeToU64(k) + fileTimeToU64(u)) / 1e7;
            CloseHandle(h);
        }
    }
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        st.memTotalMB = qint64(ms.ullTotalPhys / (1024 * 1024));
        st.memUsedMB = qint64((ms.ullTotalPhys - ms.ullAvailPhys) / (1024 * 1024));
    }
#else
    const double appSec = -1, guiSec = -1, workerSec = -1;
#endif

    if (haveDelta) {
        auto rate = [dt](double now, double before) {
            return (now >= 0 && before >= 0 && now >= before) ? 100.0 * (now - before) / dt : -1.0;
        };
        st.appCpu = rate(appSec, m_lastAppSec);
        st.guiThreadCpu = rate(guiSec, m_lastGuiSec);
        if (frames.workerThreadId == m_lastWorkerTid)
            st.workerThreadCpu = rate(workerSec, m_lastWorkerSec);

        if (frames.workerThreadId > 0) {
            st.processedFps = (frames.processed - m_lastFrames.processed) / dt;
            st.displayedFps = (frames.displayed - m_lastFrames.displayed) / dt;
            st.droppedFps = (frames.dropped - m_lastFrames.dropped) / dt;
        }
    }
    m_lastAppSec = appSec;
    m_lastGuiSec = guiSec;
    m_lastWorkerSec = workerSec;
    m_lastWorkerTid = frames.workerThreadId;
    m_lastFrames = frames;
    m_lastWallMs = nowMs;

    if (!haveDelta) return;
    m_last = st;
    emit updated(m_last);
}
