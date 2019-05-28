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
    WindowManager(fusion::System *system);
    ~WindowManager();

    // init opengl context and create a window
    // OPTIONAL: compile glsl shader programs
    bool initialize_gl_context(const size_t width, const int height);

    void set_system(fusion::System *system);

    // if window is closed
    bool should_quit() const;

    // process data
    void process_images(cv::Mat depth, cv::Mat image);

    // Main loop
    void render_screen();

    // set display images
    // TODO: only initiate textures once
    // **potential performance overhead**
    void set_rendered_scene(cv::Mat scene);
    void set_source_image(cv::Mat image_src);
    void set_input_depth(cv::Mat depth);

    // get mapped resources
    void *get_cuda_mapped_ptr(int id);
    void cuda_unmap_resources(int id);

    // system control
    static int run_mode;
    static int colour_mode;
    static bool should_save_file;
    static bool should_reset;

    // triangle count
    uint num_mesh_triangles;
    size_t num_key_points;
    float *keypoint3d;

public:
    // textures used in our code
    // scene, depth, colour respectively
    GLuint textures[3];

    // glsl programs
    // phong shading, normal map, colour map
    GLuint program[3];

    // vertex buffer, normal buffer and colour buffer
    GLuint buffers[4];
    GLuint gl_array[4];
    cudaGraphicsResource_t buffer_res[4]; // map buffer to CUDA

    // shaders temporary variables
    GLuint shaders[4];

    // camera control
    // TODO: move this to a separate struct
    // probably called "Camera"
    glm::mat4 get_view_projection_matrix();
    double prev_mouse_pos[2];
    glm::mat4 model_matrix;
    glm::mat4 view_matrix;
    glm::vec3 cam_position;
    glm::vec3 lookat_vec;
    glm::vec3 camera_up_vec;

    // drawing functions
    void draw_source_image();
    void draw_rendered_scene();
    void draw_input_depth();
    void draw_mesh();
    void draw_keypoints();

    // system control
    fusion::System *system;
    bool need_update;
    bool keypoints_need_update;

    // window control
    static void toggle_full_screen();
};

#endif