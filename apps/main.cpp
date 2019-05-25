#include <iostream>
#include "system.h"
#include "window.h"
#include "rgbd_camera.h"
#include "intrinsic_matrix.h"

int main(int argc, char **argv)
{
    fusion::RgbdCamera camera(640, 480, 30);
    fusion::IntrinsicMatrix K(640, 480, 580, 580, 319.5, 239.5);
    fusion::System slam(K, 5);

    WindowManager wm;
    wm.initialize_gl_context(1920, 960);
    wm.set_system(&slam);

    while (!wm.should_quit())
    {
        if (camera.get_image())
        {
            if (WindowManager::should_reset)
            {
                slam.restart();
                WindowManager::should_reset = false;
            }

            switch (WindowManager::run_mode)
            {
            case 1:
            {
                // processing current images
                slam.process_images(camera.depth, camera.image);

                switch (WindowManager::colour_mode)
                {
                case 0:
                    wm.set_rendered_scene(slam.get_rendered_scene());
                    break;

                case 1:
                    wm.set_rendered_scene(slam.get_rendered_scene_textured());
                    break;
                }
                break;
            }
            default:
                // TODO: only update when tracking was succeeded.
                // float3 *vertex_ptr = wm.get_cuda_mapped_ptr_vertex(0);
                // slam.create_mesh_gl(vertex_ptr, wm.num_mesh_triangles);
                // wm.cuda_unmap_resources(0);
                // wm.view_matrix = slam.get_current_camera_pose();
                float3 *vertex_ptr = wm.get_cuda_mapped_ptr(0);
                float3 *normal_ptr = wm.get_cuda_mapped_ptr(1);
                slam.fetch_mesh_with_normal(vertex_ptr, normal_ptr, wm.num_mesh_triangles);
                wm.cuda_unmap_resources(0);
                wm.cuda_unmap_resources(1);
                wm.view_matrix = slam.get_current_camera_pose();
            }

            if (WindowManager::should_save_file)
            {
                WindowManager::should_save_file = false;
                slam.save_mesh_to_file("mesh.stl");
            }

            wm.set_source_image(camera.image.clone());
            wm.set_input_depth(camera.depth.clone());
        }

        wm.render_scene();
    }
}