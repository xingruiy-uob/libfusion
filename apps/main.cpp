#include <iostream>
#include "system.h"
#include "window.h"
#include "rgbd_camera.h"
#include "intrinsic_matrix.h"

int main(int argc, char **argv)
{
    fusion::RgbdCamera camera(640, 480, 30);
    fusion::IntrinsicMatrix K(640, 480, 570, 570, 319.5, 239.5);
    fusion::System slam(K, 5);

    WindowManager window(&slam);
    window.initialize_gl_context(1920, 960);

    while (!window.should_quit())
    {
        if (camera.get_image())
        {
            window.process_images(camera.depth, camera.image);
            window.render_scene();
        }
    }
}