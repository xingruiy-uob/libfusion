#ifndef WINDOW_MANAGER_H
#define WINDOW_MANAGER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "system.h"

class WindowManager
{
public:
    WindowManager();
    ~WindowManager();

    // init opengl context and create a window
    // OPTIONAL: compile glsl shader programs
    bool initialize_gl_context(const size_t width, const int height);

    void set_system(fusion::System *system);

    // if window is closed
    bool should_quit() const;

    // Main loop
    void render_scene();

    void set_rendered_scene(cv::Mat scene);
    void set_source_image(cv::Mat image_src);
    void set_input_depth(cv::Mat depth);

    // get mapped resources
    float3 *get_cuda_mapped_ptr(int id);
    void cuda_unmap_resources(int id);

    // system control
    static int run_mode;
    static int colour_mode;
    static bool should_save_file;
    static bool should_reset;

    uint num_mesh_triangles;
    Eigen::Matrix4f view_matrix;

public:
    // textures used in our code
    // scene, depth, colour respectively
    GLuint textures[3];

    // glsl programs
    // phong shading, normal map, colour map
    GLuint program[3];

    // vertex buffer, normal buffer and colour buffer
    GLuint buffers[3];
    GLuint gl_array[3];
    cudaGraphicsResource_t buffer_res[3];

    // shaders temporary variables
    GLuint shaders[4];

    GLfloat position[3];
    GLfloat lookat[3];

    double prev_mouse_pos[2];
    glm::mat4 model_matrix;

    // drawing functions
    void draw_source_image();
    void draw_rendered_scene();
    void draw_input_depth();

    fusion::System *system;

    // window control
    static void toggle_full_screen();

    glm::mat4 get_view_projection_matrix(Eigen::Matrix4f eigen_view_matrix);
};

#endif