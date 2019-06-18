#ifndef RGBD_CAMERA_H
#define RGBD_CAMERA_H

#include <openni2/OpenNI.h>
#include <opencv2/opencv.hpp>

namespace fusion
{

class RgbdCamera
{
public:
    ~RgbdCamera();
    RgbdCamera(int cols = 640, int rows = 480, int fps = 30);

    bool get_image();
    RgbdCamera(const RgbdCamera &) = delete;
    RgbdCamera &operator=(const RgbdCamera &) = delete;
    cv::Mat image, depth;

private:
    int fps;
    int width;
    int height;

    openni::Device device;
    openni::VideoStream depthStream;
    openni::VideoStream rgbStream;
    openni::VideoFrameRef depthFrame;
    openni::VideoFrameRef rgbFrame;
};

} // namespace fusion

#endif