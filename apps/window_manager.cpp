#include "window_manager.h"
#include <iostream>
#include <thread>
#include <chrono>

int window_width = 0;
int window_height = 0;
bool full_screen = false;
GLFWwindow *window = NULL;
int WindowManager::run_mode = 0;
bool WindowManager::should_save_file = false;

void WindowManager::error_callback(int error, const char *description)
{
    std::cerr << description << std::endl;
}

void WindowManager::key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    if (key == GLFW_KEY_TAB && action == GLFW_RELEASE)
        toggle_full_screen();

    if (key == GLFW_KEY_S && action == GLFW_RELEASE)
        run_mode = (run_mode == 1) ? 0 : 1;

    if (key == GLFW_KEY_O && action == GLFW_PRESS)
        should_save_file = true;
}

void WindowManager::window_size_callback(GLFWwindow *window, int width, int height)
{
    window_width = width;
    window_height = height;
}

WindowManager::WindowManager()
{
    full_screen = false;
}

WindowManager::~WindowManager()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

void WindowManager::toggle_full_screen()
{
    if (!full_screen)
    {
        glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, 3840, 2160, 60);
        full_screen = true;
    }
    else
    {
        glfwSetWindowMonitor(window, NULL, 0, 0, 1920, 960, 60);
        full_screen = false;
    }
}

bool WindowManager::initialize_gl_context(const size_t width, const int height)
{
    window_width = width;
    window_height = height;

    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        return false;

    window = glfwCreateWindow(width, height, "RelocFusion", NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);

    // initialize textures
    glGenTextures(3, &textures[0]);

    if (glGetError() != GLEW_OK)
        return false;

    return true;
}

void WindowManager::set_rendered_scene(cv::Mat scene)
{
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        scene.cols,
        scene.rows,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        scene.ptr());
}

void WindowManager::set_source_image(cv::Mat image_src)
{
    glBindTexture(GL_TEXTURE_2D, textures[2]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        image_src.cols,
        image_src.rows,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        image_src.ptr());
}

void WindowManager::set_input_depth(cv::Mat depth)
{
    depth.convertTo(depth, CV_8UC3);
    glBindTexture(GL_TEXTURE_2D, textures[1]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_R8,
        depth.cols,
        depth.rows,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        depth.ptr());
}

void draw_quads()
{
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex3f(-1, 1, 0);
    glTexCoord2f(0, 1);
    glVertex3f(-1, -1, 0);
    glTexCoord2f(1, 1);
    glVertex3f(1, -1, 0);
    glTexCoord2f(1, 0);
    glVertex3f(1, 1, 0);
    glEnd();
}

void WindowManager::draw_source_image()
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textures[2]);

    // Draw a textured quad
    draw_quads();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void WindowManager::draw_rendered_scene()
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textures[0]);

    // Draw a textured quad
    draw_quads();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void WindowManager::draw_input_depth()
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textures[1]);

    // Draw a textured quad
    draw_quads();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

bool WindowManager::should_quit() const
{
    return glfwWindowShouldClose(window);
}

void WindowManager::render_scene()
{

    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(120.f / 255.f, 120.f / 255.f, 236.f / 255.f, 255.f);
    glMatrixMode(GL_MODELVIEW);

    int separate_x = (int)((float)window_width / 3);

    glViewport(separate_x * 2, 0, separate_x, window_height / 2);
    draw_source_image();

    glViewport(0, 0, separate_x * 2, window_height);
    draw_rendered_scene();

    glViewport(separate_x * 2, window_height / 2, separate_x, window_height / 2);
    draw_input_depth();

    glfwSwapBuffers(window);
    glfwPollEvents();
}