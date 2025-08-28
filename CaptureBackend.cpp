#include "CaptureBackend.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <cctype>


using namespace cv;
using namespace std;


static bool is_number(const string& s) { if (s.empty())return false; size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0; for (; i < s.size(); ++i) if (!isdigit((unsigned char)s[i])) return false; return true; }


class OpenCVCapture : public ICapture
{
	VideoCapture cap;
public:
	OpenCVCapture(const string& spec, bool is_gst, int w, int h, double fps) {
		if (is_gst) cap.open(spec, CAP_GSTREAMER);
		else if (spec.rfind("/dev/video", 0) == 0) { cap.open(spec, CAP_V4L2); }
		else if (is_number(spec)) { int idx = stoi(spec); if (!cap.open(idx, CAP_V4L2)) cap.open(idx, CAP_ANY); }
		else { if (!cap.open(spec, CAP_V4L2)) cap.open(spec, CAP_ANY); }
		if (cap.isOpened()) {
			cap.set(CAP_PROP_FRAME_WIDTH, w);
			cap.set(CAP_PROP_FRAME_HEIGHT, h);
			cap.set(CAP_PROP_FPS, fps);
			cap.set(CAP_PROP_BUFFERSIZE, 1);
			cap.set(CAP_PROP_CONVERT_RGB, 1);
		}
	}
	bool isOpened() const override { return cap.isOpened(); }
	bool read(Mat& out) override { return cap.read(out); }
};


#if defined(__linux__)
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <errno.h>


static int xioctl(int fd, unsigned long req, void* arg) { int r; do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR); return r; }


class V4L2Capture : public ICapture {
	int fd = -1; int W = 0, H = 0; uint32_t fmt = 0; struct Buf { void* p; size_t len; }; vector<Buf> bufs; bool streaming = false;
public:
	V4L2Capture(const string& dev, int w, int h, double fps) {
		fd = open(dev.c_str(), O_RDWR | O_NONBLOCK);
		if (fd < 0) return;
		vector<uint32_t> order = { V4L2_PIX_FMT_MJPEG, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_GREY, V4L2_PIX_FMT_Y16, V4L2_PIX_FMT_BGR24 };
		for (uint32_t f : order) { struct v4l2_format s; memset(&s, 0, sizeof(s)); s.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; s.fmt.pix.width = w; s.fmt.pix.height = h; s.fmt.pix.pixelformat = f; s.fmt.pix.field = V4L2_FIELD_ANY; if (xioctl(fd, VIDIOC_S_FMT, &s) == 0) { fmt = s.fmt.pix.pixelformat; W = s.fmt.pix.width; H = s.fmt.pix.height; break; } }
		if (W == 0 || H == 0) { close(fd); fd = -1; return; }
		struct v4l2_streamparm sp; memset(&sp, 0, sizeof(sp)); sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; if (xioctl(fd, VIDIOC_G_PARM, &sp) == 0) { if (sp.parm.capture.capability & V4L2_CAP_TIMEPERFRAME) { sp.parm.capture.timeperframe.numerator = 1; sp.parm.capture.timeperframe.denominator = (fps > 0 ? (unsigned int)fps : 30); xioctl(fd, VIDIOC_S_PARM, &sp); } }
		struct v4l2_requestbuffers rb; memset(&rb, 0, sizeof(rb)); rb.count = 4; rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; rb.memory = V4L2_MEMORY_MMAP; if (xioctl(fd, VIDIOC_REQBUFS, &rb) != 0 || rb.count < 2) { close(fd); fd = -1; return; }
		bufs.resize(rb.count);
		for (unsigned i = 0; i < rb.count; ++i) { struct v4l2_buffer b; memset(&b, 0, sizeof(b)); b.type = rb.type; b.memory = rb.memory; b.index = i; if (xioctl(fd, VIDIOC_QUERYBUF, &b) != 0) { close(fd); fd = -1; return; } bufs[i].len = b.length; bufs[i].p = mmap(NULL, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, b.m.offset); if (bufs[i].p == MAP_FAILED) { close(fd); fd = -1; return; } }
		for (unsigned i = 0; i < rb.count; ++i) { struct v4l2_buffer b; memset(&b, 0, sizeof(b)); b.type = rb.type; b.memory = rb.memory; b.index = i; xioctl(fd, VIDIOC_QBUF, &b); }
		enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE; if (xioctl(fd, VIDIOC_STREAMON, &type) == 0) streaming = true; else { close(fd); fd = -1; }
	}
	bool isOpened() const override { return fd >= 0 && streaming; }
	bool read(Mat& out) override {
		if (!isOpened()) return false;
		struct pollfd pfd; pfd.fd = fd; pfd.events = POLLIN; int pr = poll(&pfd, 1, 1000);
		if (pr <= 0) return false;
		struct v4l2_buffer b; memset(&b, 0, sizeof(b)); b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; if (xioctl(fd, VIDIOC_DQBUF, &b) != 0) return false;
		void* data = bufs[b.index].p; size_t used = b.bytesused; bool ok = false;
		if (fmt == V4L2_PIX_FMT_MJPEG) { Mat enc(1, (int)used, CV_8UC1, (unsigned char*)data); out = imdecode(enc, IMREAD_COLOR); ok = !out.empty(); }
		else if (fmt == V4L2_PIX_FMT_YUYV) { Mat yuyv(H, W, CV_8UC2, data); cvtColor(yuyv, out, COLOR_YUV2BGR_YUY2); ok = !out.empty(); }
		else if (fmt == V4L2_PIX_FMT_GREY) { Mat g(H, W, CV_8UC1, data); out = g.clone(); ok = !out.empty(); }
		else if (fmt == V4L2_PIX_FMT_Y16) { Mat g16(H, W, CV_16UC1, data); Mat g8; g16.convertTo(g8, CV_8U, 1.0 / 256.0); out = g8; ok = !out.empty(); }
		else if (fmt == V4L2_PIX_FMT_BGR24) { Mat bgr(H, W, CV_8UC3, data); out = bgr.clone(); ok = !out.empty(); }
		xioctl(fd, VIDIOC_QBUF, &b);
		return ok;
	}
	~V4L2Capture() { if (fd >= 0) { if (streaming) { enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE; xioctl(fd, VIDIOC_STREAMOFF, &t); } for (auto& b : bufs) { if (b.p && b.p != MAP_FAILED) munmap(b.p, b.len); } close(fd); } }
};
#endif


static bool wants_v4l2(const string& s) { if (s.rfind("v4l2:", 0) == 0) return true; if (s.rfind("/dev/video", 0) == 0) return true; return false; }


std::unique_ptr<ICapture> create_capture(const string& spec, bool is_gst, int w, int h, double fps)
{
#if defined(__linux__)
	if (wants_v4l2(spec) && !is_gst)
	{
		string dev = spec;
		if (dev.rfind("v4l2:", 0) == 0) dev = dev.substr(5);
		unique_ptr<ICapture> p(new V4L2Capture(dev, w, h, fps));
		if (p->isOpened()) return p;
	}
#endif
	unique_ptr<ICapture> q(new OpenCVCapture(spec, is_gst, w, h, fps));
	if (q->isOpened()) return q;
	return nullptr;
}