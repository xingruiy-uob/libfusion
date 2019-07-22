#include "system.h"
#include "system_new.h"
#include "input/oni_camera.h"
#include "tracking/cuda_imgproc.h"
#include "visualization/main_window.h"

int main(int argc, char **argv)
{
    fusion::ONICamera camera(640, 480, 30);
    fusion::IntrinsicMatrix K(640, 480, 580, 580, 319.5, 239.5);
    fusion::SystemNew slam(K, 5);

    fusion::MainWindow window("SLAM", 1920, 920);
    window.set_system(&slam);
    cv::Mat depth, image;

    while (!pangolin::ShouldQuit())
    {
        if (camera.get_next_images(depth, image))
        {
            window.set_image_src(image, fusion::MainWindow::RGB);

            if (!window.is_paused())
            {
                slam.spawn_work(depth, image);

                cv::Mat scene, current;
                if (slam.get_rendered_scene(scene))
                {
                    window.set_image_src(scene, fusion::MainWindow::SCENE);
                }

                if (slam.get_rendered_depth(current))
                {
                    window.set_image_src(current, fusion::MainWindow::DEPTH);
                }
            }
        }

        window.render();
    }
}