#ifndef WINDOW_MANAGER_H
#define WINDOW_MANAGER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
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

    // system control
    static int run_mode;
    static int colour_mode;
    static bool should_save_file;
    static bool should_reset;

public:
    GLuint textures[3]; // scene, depth and source textures
    GLuint shading_program[3];
    GLuint array_buffers[3];

    // drawing functions
    void draw_source_image();
    void draw_rendered_scene();
    void draw_input_depth();

    fusion::System *system;

    // window control
    static void toggle_full_screen();
};

#endif