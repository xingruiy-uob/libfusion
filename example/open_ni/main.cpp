#include "system.h"
#include "visualization/main_window.h"
#include "input/oni_camera.h"

int main(int argc, char **argv)
{
    fusion::ONICamera camera(640, 480, 30);
    fusion::IntrinsicMatrix K(640, 480, 580, 580, 319.5, 239.5);
    fusion::System slam(K, 5);

    MainWindow window("Untitled", 1920, 920);
    window.SetSystem(&slam);
    cv::Mat image, depth;

    while (!pangolin::ShouldQuit())
    {
        if (camera.get_next_images(depth, image))
        {
            window.SetRGBSource(image);
            if (!window.IsPaused())
            {
                slam.process_images(depth, image);

                // window.SetDepthSource(slam.get_shaded_depth());
                window.SetRenderScene(slam.get_rendered_scene());
                window.SetCurrentCamera(slam.get_camera_pose());
                window.mbFlagUpdateMesh = true;
            }

            if (window.IsPaused() && window.mbFlagUpdateMesh)
            {
                auto *vertex = window.GetMappedVertexBuffer();
                auto *normal = window.GetMappedNormalBuffer();
                window.VERTEX_COUNT = slam.fetch_mesh_with_normal(vertex, normal);
                window.mbFlagUpdateMesh = false;
            }
        }

        window.Render();
    }
}