#ifndef __WINDOW_MANAGER__
#define __WINDOW_MANAGER__

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

class WindowManager
{
public:
    WindowManager();
    ~WindowManager();
    bool initialize_gl_context(const size_t width, const int height);
    void initialize_textures(const size_t width, const int height);
    bool should_quit() const;
    void update_texures();
    void render_scene();

    void set_rendered_scene(cv::Mat scene);
    void set_source_image(cv::Mat image_src);
    void set_input_depth(cv::Mat depth);

    // system control
    static int run_mode;

private:
    GLuint textures[3]; // scene, depth and source textures
    GLuint shaders[3];  // vertex, geometry and fragment shaders
    GLuint program;

    // drawing functions
    void draw_source_image();
    void draw_rendered_scene();
    void draw_input_depth();

    // window control
    static void toggle_full_screen();

    // callbacks
    static void error_callback(int error, const char *description);
    static void window_size_callback(GLFWwindow *window, int width, int height);
    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
};

#endif