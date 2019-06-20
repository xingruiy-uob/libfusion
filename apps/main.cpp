#include "system.h"
#include "window.h"
#include "cuda_runtime.h"
#include <xutils/IOWrapper/rgbd_camera.h>
#include <xutils/DataStruct/stop_watch.h>

int main(int argc, char **argv)
{
    xutils::RgbdCamera camera(640, 480, 30);
    fusion::IntrinsicMatrix K(640, 480, 570, 570, 319.5, 239.5);
    fusion::System slam(K, 5);

    MainWindow window("Untitled", 1920, 920);
    window.SetSystem(&slam);

    while (!pangolin::ShouldQuit())
    {
        xutils::StopWatch swatch(true);

        if (camera.get_image())
        {
            window.SetRGBSource(camera.image);

            if (window.mbFlagRestart)
                slam.restart();

            if (!window.IsPaused())
            {
                slam.process_images(camera.depth, camera.image);
                // window.SetDepthSource(slam.get_shaded_depth());
                window.SetRenderScene(slam.get_rendered_scene());
                window.SetCurrentCamera(slam.get_camera_pose());
                window.mbFlagUpdateMesh = true;
            }

            if (window.mbFlagUpdateMesh)
            {
                auto *vertex = window.GetMappedVertexBuffer();
                auto *normal = window.GetMappedNormalBuffer();
                window.VERTEX_COUNT = slam.fetch_mesh_with_normal(vertex, normal);
                window.mbFlagUpdateMesh = false;
            }
        }

        window.Render();

        std::cout << swatch << std::endl;
    }
}