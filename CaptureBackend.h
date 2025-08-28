#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>


class ICapture {
public:
	virtual ~ICapture() = default;
	virtual bool isOpened() const = 0;
	virtual bool read(cv::Mat& out) = 0;
};


std::unique_ptr<ICapture> create_capture(const std::string& spec, bool is_gst, int w, int h, double fps);