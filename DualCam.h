#pragma once
#include <QWidget>
#include <QKeyEvent>
#include <deque>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>


class QLabel;
class QTimer;
class ICapture;


class DualCam : public QWidget {
public:
	explicit DualCam(QWidget* parent = nullptr);
	~DualCam();
protected:
	void keyPressEvent(QKeyEvent* e) override;
private:
	void tick();
	static double focus_measure(const cv::Mat& img);
	static void draw_focus_plot(cv::Mat& plot, const std::deque<double>& a, const std::deque<double>& b, int margin, int thickness, double eq_idx);
	static double find_equal_index(const std::deque<double>& a, const std::deque<double>& b);
	static bool is_number(const std::string& s);
	struct Source { std::string spec; bool is_gst; };
	static std::vector<std::string> list_video_nodes();
	static std::optional<Source> env_source(const char* key);
	static std::optional<std::unique_ptr<ICapture>> try_open(const Source& src, int w, int h, double fps);
	static std::vector<Source> autodetect_sources();


	QLabel* view;
	QTimer* timer;
	int tileW;
	int tileH;
	double fps;
	std::unique_ptr<ICapture> cap0;
	std::unique_ptr<ICapture> cap1;
	bool ok0;
	bool ok1;
	cv::Mat black;
	cv::Mat grid;
	cv::Mat r0, r1, diff, diff3, plot;
	std::deque<double> h0, h1;
	size_t histCap;
};