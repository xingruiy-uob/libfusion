#include "rgbd_camera.h"

namespace fusion
{

RgbdCamera::RgbdCamera(int cols, int rows, int fps)
    : width(cols), height(rows), fps(fps)
{
    // openni context initialization
    if (openni::OpenNI::initialize() != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // openni camera open
    if (device.open(openni::ANY_DEVICE) != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // create depth stream
    if (depthStream.create(device, openni::SENSOR_DEPTH) != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // create colour stream
    if (rgbStream.create(device, openni::SENSOR_COLOR) != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    auto videoModeDepth = openni::VideoMode();
    videoModeDepth.setResolution(width, height);
    videoModeDepth.setFps(fps);
    videoModeDepth.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);
    depthStream.setVideoMode(videoModeDepth);

    auto videoModeRGB = openni::VideoMode();
    videoModeRGB.setResolution(width, height);
    videoModeRGB.setFps(fps);
    videoModeRGB.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
    rgbStream.setVideoMode(videoModeRGB);

    if (device.isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR))
        device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

    rgbStream.setMirroringEnabled(false);
    depthStream.setMirroringEnabled(false);

    if (depthStream.start() != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    if (rgbStream.start() != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    std::cout << "camera ready" << std::endl;
}

RgbdCamera::~RgbdCamera()
{
    rgbStream.stop();
    rgbStream.destroy();
    depthStream.stop();
    depthStream.destroy();
    device.close();
    openni::OpenNI::shutdown();

    std::cout << "camera released" << std::endl;
    return;
}

bool RgbdCamera::get_image()
{
    openni::VideoStream *streams[] = {&depthStream, &rgbStream};

    int stream_ready = -1;
    auto last_state = openni::STATUS_OK;

    while (last_state == openni::STATUS_OK)
    {
        last_state = openni::OpenNI::waitForAnyStream(streams, 2, &stream_ready, 0);

        if (last_state == openni::STATUS_OK)
        {
            switch (stream_ready)
            {
            case 0: //depth ready
                if (depthStream.readFrame(&depthFrame) == openni::STATUS_OK)
                    depth = cv::Mat(height, width, CV_16UC1, const_cast<void *>(depthFrame.getData()));
                break;

            case 1: // color ready
                if (rgbStream.readFrame(&rgbFrame) == openni::STATUS_OK)
                {
                    image = cv::Mat(height, width, CV_8UC3, const_cast<void *>(rgbFrame.getData()));
                    // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                }
                break;

            default: // unexpected stream
                return false;
            }
        }
    }

    if (!depthFrame.isValid() || !rgbFrame.isValid())
        return false;

    return true;
}

} // namespace fusion