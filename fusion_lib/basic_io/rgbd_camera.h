#ifndef RGBD_CAMERA_H
#define RGBD_CAMERA_H

#include <openni2/OpenNI.h>
#include <opencv2/opencv.hpp>

namespace fusion
{

class RgbdCamera
{
public:
    RgbdCamera();
    RgbdCamera(size_t cols, size_t rows, int fps);
    // stop and destroy streams
    ~RgbdCamera();

    bool get_image();
    RgbdCamera(const RgbdCamera &) = delete;
    RgbdCamera &operator=(const RgbdCamera &) = delete;

    cv::Mat image, depth;

private:
    int fps;
    size_t width;
    size_t height;

    openni::Device device;
    openni::VideoStream depth_stream;
    openni::VideoStream color_stream;
    openni::VideoFrameRef depth_ref;
    openni::VideoFrameRef color_ref;
};

} // namespace fusion

#endif