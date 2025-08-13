#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <optional>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>

enum class Backend { AUTO, LIBCAMERA, V4L2 };

struct CaptureParams 
{
    int  cameraId{ 0 };           
    int  width{ 640 };
    int  height{ 480 };
    int  fps{ 30 };
    bool forceGray{ true };       

    Backend backend{ Backend::AUTO };
    std::optional<std::string> pipelineOverride{};  

    std::string device{ "/dev/video0" };
    std::string v4l2PixelFmt{ "GRAY8" };           

    bool convertBGR{ true }; 

    int  warmupFrames{ -1 };     
    int  warmupMs{ 2000 };       

    bool verbose{ true };
};

inline std::string make_libcamera_pipeline(const CaptureParams& p)
{
    std::ostringstream ss;
    ss << "libcamerasrc camera-id=" << p.cameraId
        << " ! video/x-raw,width=" << p.width
        << ",height=" << p.height
        << ",framerate=" << p.fps << "/1"
        << " ! videoconvert";
    if (p.forceGray) 
    {
        ss << " ! video/x-raw,format=GRAY8";
    }
    else 
    {
        ss << " ! video/x-raw,format=BGR";
    }
    ss << " ! appsink drop=true max-buffers=1 sync=false";
    return ss.str();
}

inline std::string make_v4l2_pipeline(const CaptureParams& p)
{
    std::ostringstream ss;
    ss << "v4l2src device=" << p.device
        << " ! video/x-raw,format=" << p.v4l2PixelFmt
        << ",width=" << p.width
        << ",height=" << p.height
        << ",framerate=" << p.fps << "/1"
        << " ! appsink drop=true max-buffers=1 sync=false";
    return ss.str();
}

class LibcameraCapture
{
public:
    LibcameraCapture() = default;
    ~LibcameraCapture() { release(); }

    bool open(const CaptureParams& params)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        params_ = params;
        lastError_.clear();

        if (params_.pipelineOverride && !params_.pipelineOverride->empty()) 
        {
            if (params_.verbose) {
                std::cerr << "[Capture] Opening custom pipeline: "
                    << *params_.pipelineOverride << std::endl;
            }
            if (cap_.open(*params_.pipelineOverride, cv::CAP_GSTREAMER)) {
                warmup_locked();
                return true;
            }
            lastError_ = "Failed to open custom GStreamer pipeline.";
            if (params_.verbose) std::cerr << "[Capture] " << lastError_ << std::endl;
            return false;
        }

        if (params_.backend == Backend::LIBCAMERA) 
        {
            const std::string pipe = make_libcamera_pipeline(params_);
            if (params_.verbose) std::cerr << "[Capture] LIBCAMERA: " << pipe << std::endl;
            if (cap_.open(pipe, cv::CAP_GSTREAMER)) { warmup_locked(); return true; }
            lastError_ = "Failed to open libcamera pipeline.";
            return false;
        }
        if (params_.backend == Backend::V4L2) 
        {
            const std::string pipe = make_v4l2_pipeline(params_);
            if (params_.verbose) std::cerr << "[Capture] V4L2: " << pipe << std::endl;
            if (cap_.open(pipe, cv::CAP_GSTREAMER)) { warmup_locked(); return true; }
            lastError_ = "Failed to open v4l2 pipeline.";
            return false;
        }

        {
            const std::string pipe = make_libcamera_pipeline(params_);
            if (params_.verbose) std::cerr << "[Capture] AUTO(libcamera): " << pipe << std::endl;
            if (cap_.open(pipe, cv::CAP_GSTREAMER)) { warmup_locked(); return true; }
        }
        if (!params_.device.empty()) 
        {
            const std::string pipe = make_v4l2_pipeline(params_);
            if (params_.verbose) std::cerr << "[Capture] AUTO(v4l2): " << pipe << std::endl;
            if (cap_.open(pipe, cv::CAP_GSTREAMER)) { warmup_locked(); return true; }
        }

        if (params_.verbose) 
        {
            std::cerr << "[Capture] Fallback: index " << params_.cameraId << std::endl;
        }
        if (cap_.open(params_.cameraId, cv::CAP_ANY)) {
            if (params_.width > 0) cap_.set(cv::CAP_PROP_FRAME_WIDTH, params_.width);
            if (params_.height > 0) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, params_.height);
            if (params_.fps > 0) cap_.set(cv::CAP_PROP_FPS, params_.fps);
            warmup_locked();
            return true;
        }

        lastError_ = "Failed to open camera (pipelines and index).";
        return false;
    }

    bool isOpened() const { return cap_.isOpened(); }

    void release()
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (cap_.isOpened()) cap_.release();
    }

    bool read(cv::Mat& out)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!cap_.isOpened()) 
        {
            lastError_ = "read() on closed capture";
            return false;
        }
        if (!cap_.read(out) || out.empty()) 
        {
            lastError_ = "VideoCapture::read() failed";
            return false;
        }
        postprocess_locked(out);
        return true;
    }

    bool grab()
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return cap_.grab();
    }

    bool retrieve(cv::Mat& out)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!cap_.isOpened()) {
            lastError_ = "retrieve() on closed capture";
            return false;
        }
        if (!cap_.retrieve(out) || out.empty()) {
            lastError_ = "retrieve() failed";
            return false;
        }
        postprocess_locked(out);
        return true;
    }

    const CaptureParams& params() const { return params_; }

    std::string lastError() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return lastError_;
    }

private:
    void warmup_locked()
    {
        if (!cap_.isOpened()) return;

        int drops = params_.warmupFrames;
        if (drops < 0) 
        {
            drops = (params_.fps > 0)
                ? static_cast<int>((params_.fps * params_.warmupMs) / 1000)
                : 1;
            if (drops < 1) drops = 1;
        }
        if (params_.verbose) 
        {
            std::cerr << "[Capture] Warmup: dropping " << drops << " frame(s)\n";
        }
        cv::Mat tmp;
        for (int i = 0; i < drops; ++i) 
        {
            if (!cap_.read(tmp)) break;
        }
    }

    void postprocess_locked(cv::Mat& frame)
    {
        if (!params_.convertBGR) return;

        if (params_.forceGray) {
            if (frame.channels() == 3) 
            {
                cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
            }
            else if (frame.channels() == 4) 
            {
                cv::cvtColor(frame, frame, cv::COLOR_BGRA2GRAY);
            }
        }
        else {
            if (frame.channels() == 1) 
            {
                cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
            }
            else if (frame.type() == CV_8UC2) 
            {
                cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR_YUYV);
            }
        }
    }

private:
    mutable std::mutex mtx_{};
    cv::VideoCapture cap_{};
    CaptureParams params_{};
    std::string lastError_{};
};

class FrameGrabber 
{
public:
    FrameGrabber() = default;
    ~FrameGrabber() { stop(); }

    void start(LibcameraCapture* camera) 
    {
        stop();
        camera_ = camera;
        running_.store(true, std::memory_order_release);
        thread_ = std::thread(&FrameGrabber::run, this);
    }

    void stop() 
    {
        bool expected = true;
        if (running_.compare_exchange_strong(expected, false)) {
            if (thread_.joinable()) thread_.join();
        }
    }

    cv::Mat getFrame()
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return frame_.empty() ? cv::Mat() : frame_.clone();
    }

    bool isRunning() const { return running_.load(std::memory_order_acquire); }

private:
    void run() 
    {
        if (!camera_ || !camera_->isOpened()) 
        {
            std::cerr << "[FrameGrabber] Camera not opened.\n";
            return;
        }
        cv::Mat local;
        while (running_.load(std::memory_order_acquire)) 
        {
            if (camera_ && camera_->isOpened()) 
            {
                if (camera_->read(local)) 
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    frame_ = local.clone();
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    LibcameraCapture* camera_{ nullptr };
    std::thread thread_{};
    std::mutex mtx_{};
    cv::Mat frame_{};
    std::atomic<bool> running_{ false };
};
