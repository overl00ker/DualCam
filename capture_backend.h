#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <iostream>

struct CaptureParams
{
    int cameraId{0};
    int width{640};
    int height{480};
    int fps{30};
    bool forceGray{true};      // attempt GRAY8 output from pipeline
};

inline std::string make_libcamera_pipeline(const CaptureParams& p)
{
    std::ostringstream ss;
    ss << "libcamerasrc camera-id=" << p.cameraId
       << " ! video/x-raw,format=YUV420,width=" << p.width
       << ",height=" << p.height
       << ",framerate=" << p.fps << "/1"
       << " ! videoconvert";
    if (p.forceGray)
        ss << " ! video/x-raw,format=GRAY8";
    else
        ss << " ! video/x-raw,format=BGR";
    ss << " ! appsink drop=true max-buffers=1 sync=false";
    return ss.str();
}

class LibcameraCapture
{
public:
    LibcameraCapture() = default;
    ~LibcameraCapture() { release(); }

    bool open(const CaptureParams& params)
    {
        params_ = params;
        std::string pipeline = make_libcamera_pipeline(params);
        cap_.open(pipeline, cv::CAP_GSTREAMER);
        if (!cap_.isOpened())
        {
            std::cerr << "[LibcameraCapture] Failed to open GStreamer pipeline, falling back to index "
                      << params.cameraId << std::endl;
            cap_.open(params.cameraId, cv::CAP_ANY);
            if (cap_.isOpened())
            {
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, params.width);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, params.height);
                cap_.set(cv::CAP_PROP_FPS, params.fps);
            }
        }
        if (!cap_.isOpened())
            return false;

        warmup();
        return true;
    }

    bool isOpened() const { return cap_.isOpened(); }

    void release()
    {
        if (cap_.isOpened())
            cap_.release();
    }

    bool read(cv::Mat& out)
    {
        if (!cap_.isOpened())
            return false;
        return cap_.read(out);
    }

    bool grab() { return cap_.grab(); }
    bool retrieve(cv::Mat& out) { return cap_.retrieve(out); }

    const CaptureParams& params() const { return params_; }

private:
    void warmup()
    {
        if (!cap_.isOpened()) return;
        int drops = std::max(1, params_.fps * 2);
        cv::Mat tmp;
        for (int i = 0; i < drops; ++i)
            cap_.read(tmp);
    }

    cv::VideoCapture cap_;
    CaptureParams params_;
};
