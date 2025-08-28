#include "DualCam.h"
#include "CaptureBackend.h"
#include <QLabel>
#include <QTimer>
#include <QVBoxLayout>
#include <QImage>
#include <QPixmap>
#include <filesystem>
#include <cstdlib>
#include <cctype>
#include <algorithm>
#include <memory>



using namespace cv;
using namespace std;


static std::string get_priority()
{
	const char* v = getenv("CAP_PRIORITY");
	if (!v) return "v4l2";
	std::string s(v);
	for (char& c : s) c = (char)tolower((unsigned char)c);
	if (s == "libcamera" || s == "gst" || s == "gstreamer") return "libcamera";
	return "v4l2";
}

static QImage mat_to_qimage_rgb(const Mat& bgr)
{
	Mat rgb; cv::cvtColor(bgr, rgb, COLOR_BGR2RGB);
	return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
}


bool DualCam::is_number(const string& s)
{
	if (s.empty()) return false;
	size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0;
	for (; i < s.size(); ++i)
		if (!isdigit((unsigned char)s[i])) return false; return true;
}


optional<DualCam::Source> DualCam::env_source(const char* key)
{
	const char* v = getenv(key);
	if (!v) return nullopt;
	string s(v);
	bool gst = s.find('!') != string::npos || s.rfind("gst:", 0) == 0;
	if (s.rfind("gst:", 0) == 0) s = s.substr(4);
	return Source{ s,gst };
}


vector<string> DualCam::list_video_nodes()
{
	vector<string> nodes; for (int i = 0; i < 64; ++i)
	{
		string p = "/dev/video" + to_string(i); if (std::filesystem::exists(p)) nodes.push_back(p);
	} return nodes;
}


static bool read_once(ICapture* c, Mat& f)
{
	if (!c) return false;
	if (!c->isOpened()) return false; return c->read(f);
}


optional<unique_ptr<ICapture>> DualCam::try_open(const Source& src, int w, int h, double fps)
{
	auto p = create_capture(src.spec, src.is_gst, w, h, fps);
	if (!p) return nullopt; Mat f;
	if (!read_once(p.get(), f)) return nullopt; return optional<unique_ptr<ICapture>>(std::move(p));
}


vector<DualCam::Source> DualCam::autodetect_sources()
{
	vector<Source> out;
#if defined(_WIN32)
	out.push_back(Source{ "0",false });
	out.push_back(Source{ "1",false });
	return out;
#else
	string pr = get_priority();

	const char* e0 = getenv("CAM0");
	const char* e1 = getenv("CAM1");
	if (e0) out.push_back(Source{ std::string(e0), std::string(e0).find('!') != std::string::npos || std::string(e0).rfind("gst:",0) == 0 });
	if (e1) out.push_back(Source{ std::string(e1), std::string(e1).find('!') != std::string::npos || std::string(e1).rfind("gst:",0) == 0 });
	if (out.size() >= 2) return out;

	if (pr == "v4l2")
	{
		auto nodes = list_video_nodes();
		for (auto& n : nodes) out.push_back(Source{ n,false });
		if (out.size() >= 2) return out;
		out.clear();
	}

	out.push_back(Source
		{
		"libcamerasrc camera-id=0 ! video/x-raw,width=1280,height=720,format=BGRx ! "
		"videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true", true
		});
	out.push_back(Source
		{
		"libcamerasrc camera-id=1 ! video/x-raw,width=1280,height=720,format=BGRx ! "
		"videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true", true
		});
	return out;
#endif
}



double DualCam::focus_measure(const Mat& img)
{
	Mat g, lap;
	if (img.channels() == 3) cv::cvtColor(img, g, COLOR_BGR2GRAY);
	else g = img;
	Laplacian(g, lap, CV_64F, 3);
	Scalar mu, sigma; meanStdDev(lap, mu, sigma);
	return sigma[0] * sigma[0];
}


double DualCam::find_equal_index(const deque<double>& a, const deque<double>& b)
{
	if (a.empty() || b.empty()) return -1.0; size_t n = min(a.size(), b.size());
	if (n < 2) return -1.0;
	size_t sa = a.size() - n, sb = b.size() - n;
	for (size_t i = 1; i < n; ++i)
	{
		double p = a[sa + i - 1] - b[sb + i - 1];
		double c = a[sa + i] - b[sb + i];
		if ((p <= 0 && c >= 0) || (p >= 0 && c <= 0))
		{
			double t = p / (p - c + 1e-12);
			return (sa + i - 1) + t;
		}
	}
	double best = 1e300; size_t idx = sa;
	for (size_t i = 0; i < n; ++i)
	{
		double d = fabs(a[sa + i] - b[sb + i]);
		if (d < best)
		{
			best = d; idx = sa + i;
		}
	}
	return (double)idx;
}


void DualCam::draw_focus_plot(Mat& plot, const deque<double>& a, const deque<double>& b, int margin, int thickness, double eq_idx)
{
	plot.setTo(Scalar(10, 10, 10));
	if (a.empty() || b.empty()) return; int W = plot.cols, H = plot.rows;

	int L = margin, R = W - margin, T = margin, B = H - margin;
	rectangle(plot, Rect(L, T, R - L, B - T), Scalar(40, 40, 40), 1, LINE_AA);
	size_t n = min(a.size(), b.size());
	size_t sa = a.size() - n, sb = b.size() - n;
	double mn = 1e300, mx = -1e300;
	for (size_t i = 0; i < n; ++i)
	{
		mn = min(mn, a[sa + i]); mx = max(mx, a[sa + i]);
	}
	for (size_t i = 0; i < n; ++i)
	{
		mn = min(mn, b[sb + i]);
		mx = max(mx, b[sb + i]);
	}
	if (!(mx > mn)) mx = mn + 1.0; auto mapx = [&](double i)->int
		{
			double t = (i - (double)sa) / max(1.0, (double)(n - 1)); return L + (int)(t * (R - L));
		};
	auto mapy = [&](double v)->int
		{
			double t = (v - mn) / (mx - mn); return B - (int)(t * (B - T));
		};
	for (size_t i = 1; i < n; ++i)
	{
		size_t ia = sa + i, ib = sb + i;
		line(plot, Point(mapx((double)ia - 1), mapy(a[ia - 1])), Point(mapx((double)ia), mapy(a[ia])), Scalar(50, 220, 50), thickness, LINE_AA);
		line(plot, Point(mapx((double)ib - 1), mapy(b[ib - 1])), Point(mapx((double)ib), mapy(b[ib])), Scalar(60, 120, 255), thickness, LINE_AA);
	}
	if (eq_idx >= 0.0)
	{
		int x = mapx(eq_idx);
		line(plot, Point(x, T), Point(x, B), Scalar(0, 0, 255), 1, LINE_AA);
		circle(plot, Point(x, (mapy(a[min((size_t)floor(eq_idx), a.size() - 1)]) + mapy(b[min((size_t)floor(eq_idx), b.size() - 1)])) / 2), 4, Scalar(0, 0, 255), FILLED, LINE_AA);
	}
}


DualCam::DualCam(QWidget* parent) : QWidget(parent), view(new QLabel(this)), timer(new QTimer(this)), tileW(960), tileH(540), fps(60.0), ok0(false), ok1(false), histCap((size_t)tileW)
{
	vector<Source> candidates; auto e0 = env_source("GST_PIPELINE_CAM0"); auto e1 = env_source("GST_PIPELINE_CAM1");
	if (e0 && e1)
	{
		candidates.push_back(*e0); candidates.push_back(*e1);
	}
	else
	{
		auto nodes = autodetect_sources(); for (auto& n : nodes) candidates.push_back(n);
	}
	if (candidates.size() >= 2)
	{
		auto c0 = try_open(candidates[0], tileW, tileH, fps); if (c0)
		{
			cap0 = std::move(*c0); ok0 = true;
		}
		auto c1 = try_open(candidates[1], tileW, tileH, fps); if (c1)
		{
			cap1 = std::move(*c1); ok1 = true;
		}
	}

	if (!(ok0 && ok1))
	{
		if (!ok0)
		{
			for (size_t i = 0; i < candidates.size(); ++i)
			{
				auto c = try_open(candidates[i], tileW, tileH, fps); if (c)
				{
					cap0 = std::move(*c); ok0 = true; break;
				}
			}
		}
		if (!ok1)
		{
			for (size_t i = 0; i < candidates.size(); ++i)
			{
				if (ok0 && candidates.size() > 1 && candidates[i].spec == candidates[0].spec) continue; auto c = try_open(candidates[i], tileW, tileH, fps);
				if (c)
				{
					cap1 = std::move(*c); ok1 = true; break;
				}
			}
		}
	}

	black = Mat(tileH, tileW, CV_8UC3, Scalar(0, 0, 0));
	grid = Mat(tileH * 2, tileW * 2, CV_8UC3, Scalar(0, 0, 0));
	plot = Mat(tileH, tileW, CV_8UC3, Scalar(0, 0, 0));
	auto layout = new QVBoxLayout(this); layout->setContentsMargins(0, 0, 0, 0); view->setAlignment(Qt::AlignCenter); layout->addWidget(view); setLayout(layout); resize(tileW * 2, tileH * 2); connect(timer, &QTimer::timeout, this, [this] { tick(); }); timer->start(1);
}


DualCam::~DualCam() {}


void DualCam::keyPressEvent(QKeyEvent* e)
{
	if (e->key() == Qt::Key_Escape || e->key() == Qt::Key_Q) close(); QWidget::keyPressEvent(e);
}


void DualCam::tick()
{
	Mat f0_, f1_; bool got0 = ok0 && cap0 && cap0->read(f0_);
	bool got1 = ok1 && cap1 && cap1->read(f1_);
	if (got0) cv::resize(f0_, r0, cv::Size(tileW, tileH));
	else r0 = black;
	if (got1) cv::resize(f1_, r1, cv::Size(tileW, tileH));
	else r1 = black; if (r0.channels() == 1) cv::cvtColor(r0, r0, COLOR_GRAY2BGR);
	if (r1.channels() == 1) cv::cvtColor(r1, r1, COLOR_GRAY2BGR);

	if (got0 && got1)
	{
		cv::absdiff(r0, r1, diff);
		if (diff.channels() == 1) cv::cvtColor(diff, diff3, COLOR_GRAY2BGR);
		else diff3 = diff;
	}
	else diff3 = black;
	double s0 = got0 ? focus_measure(r0) : 0.0; double s1 = got1 ? focus_measure(r1) : 0.0; h0.push_back(s0);
	if (h0.size() > histCap) h0.pop_front(); h1.push_back(s1);
	if (h1.size() > histCap) h1.pop_front();

	double eq_idx = find_equal_index(h0, h1);
	draw_focus_plot(plot, h0, h1, 24, 2, eq_idx);
	r0.copyTo(grid(Rect(0, 0, tileW, tileH)));
	r1.copyTo(grid(Rect(tileW, 0, tileW, tileH)));
	diff3.copyTo(grid(Rect(0, tileH, tileW, tileH)));
	plot.copyTo(grid(Rect(tileW, tileH, tileW, tileH)));
	string info = string("S0:") + to_string((int)s0) + " S1:" + to_string((int)s1) + " D:" + to_string((int)llabs((long long)(s0 - s1)));
	putText(grid, info, Point(16, 32), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA); QImage img = mat_to_qimage_rgb(grid); view->setPixmap(QPixmap::fromImage(img));
}