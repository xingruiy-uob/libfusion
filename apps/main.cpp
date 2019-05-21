#include <iostream>
#include "system.h"
#include "rgbd_camera.h"
#include "intrinsic_matrix.h"
#include "window_manager.h"

int main(int argc, char **argv)
{
    fusion::RgbdCamera camera(640, 480, 30);
    fusion::IntrinsicMatrix K(640, 480, 580, 580, 319.5, 239.5);
    fusion::System slam(K, 5);

    WindowManager wm;
    wm.initialize_gl_context(1920, 960);

    while (!wm.should_quit())
    {
        if (camera.get_image())
        {
            switch (WindowManager::run_mode)
            {
            case 1:
                slam.process_images(camera.depth, camera.image);
                wm.set_rendered_scene(slam.get_rendered_scene());
            }

            wm.set_source_image(camera.image.clone());
            wm.set_input_depth(camera.depth.clone());
        }

        wm.render_scene();
    }
}