#include "rgbd_camera.h"

namespace fusion
{

RgbdCamera::RgbdCamera() : RgbdCamera(640, 480, 30)
{
}

RgbdCamera::RgbdCamera(size_t cols, size_t rows, int fps)
    : width(cols), height(rows), fps(fps)
{
    // openni context initialization
    if (openni::OpenNI::initialize() != openni::STATUS_OK)
        return;

    // openni camera open
    if (device.open(openni::ANY_DEVICE) != openni::STATUS_OK)
        return;

    // create depth stream
    if (depth_stream.create(device, openni::SENSOR_DEPTH) != openni::STATUS_OK)
    {
        std::cout << "failed openning depth stream" << std::endl;
        return;
    }

    // create colour stream
    if (color_stream.create(device, openni::SENSOR_COLOR) != openni::STATUS_OK)
    {
        std::cout << "failed openning color stream" << std::endl;
        return;
    }

    // NOTE: not recommended
    auto vm_depth = depth_stream.getVideoMode();
    vm_depth.setResolution(width, height);
    vm_depth.setFps(fps);
    vm_depth.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

    // NOTE: not recommended
    auto vm_colour = color_stream.getVideoMode();
    vm_colour.setResolution(width, height);
    vm_colour.setFps(fps);
    vm_colour.setPixelFormat(openni::PIXEL_FORMAT_RGB888);

    depth_stream.setVideoMode(vm_depth);
    color_stream.setVideoMode(vm_colour);

    if (device.isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR))
        device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

    depth_stream.setMirroringEnabled(false);
    color_stream.setMirroringEnabled(false);

    if (depth_stream.start() != openni::STATUS_OK)
    {
        std::cout << "failed starting depth stream" << std::endl;
        return;
    }

    if (color_stream.start() != openni::STATUS_OK)
    {
        std::cout << "failed starting color stream" << std::endl;
        return;
    }

    std::cout << "camera ready" << std::endl;
}

RgbdCamera::~RgbdCamera()
{
    color_stream.stop();
    color_stream.destroy();
    depth_stream.stop();
    depth_stream.destroy();
    device.close();
    openni::OpenNI::shutdown();

    std::cout << "camera released" << std::endl;
    return;
}

bool RgbdCamera::get_image()
{
    openni::VideoStream *streams[] = {&depth_stream, &color_stream};

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
                if (depth_stream.readFrame(&depth_ref) == openni::STATUS_OK)
                    depth = cv::Mat(height, width, CV_16UC1, const_cast<void *>(depth_ref.getData()));
                break;

            case 1: // color ready
                if (color_stream.readFrame(&color_ref) == openni::STATUS_OK)
                {
                    image = cv::Mat(height, width, CV_8UC3, const_cast<void *>(color_ref.getData()));
                    // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                }
                break;

            default: // unexpected stream
                return false;
            }
        }
    }

    if (!depth_ref.isValid() || !color_ref.isValid())
        return false;

    return true;
}

} // namespace fusion