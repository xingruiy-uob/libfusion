#include "window.h"
#include <iostream>
#include <thread>
#include <chrono>

int window_width = 0;
int window_height = 0;
bool full_screen = false;
GLFWwindow *window = NULL;
int WindowManager::run_mode = 0;
int WindowManager::colour_mode = 0;
bool WindowManager::should_save_file = false;
bool WindowManager::should_reset = false;

void error_callback(int error, const char *description)
{
    std::cerr << description << std::endl;
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    WindowManager *wm;
    fusion::System *sys;
    void *data = glfwGetWindowUserPointer(window);
    if (data != NULL)
    {
        wm = static_cast<WindowManager *>(data);
        sys = wm->system;
    }

    // Quit
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    if (sys == NULL)
        return;

    // Toggle full screen mode
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS)
        wm->toggle_full_screen();

    // Start and pause the sytem
    if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
        wm->run_mode = (wm->run_mode == 1) ? 0 : 1;

    // Download mesh to disk
    if (key == GLFW_KEY_S && action == GLFW_PRESS)
        wm->should_save_file = true;

    // Restart the system
    if (key == GLFW_KEY_R && action == GLFW_PRESS)
        wm->should_reset = true;

    // Switch colour
    if (key == GLFW_KEY_C && action == GLFW_PRESS)
        wm->colour_mode = (wm->colour_mode + 1) % 2;
}

static void cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
        std::cout << "cursor pos: X " << xpos << " Y " << ypos << std::endl;
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
}

void window_size_callback(GLFWwindow *window, int width, int height)
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

    window = glfwCreateWindow(width, height, "BasicFusion", NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        return false;
    }

    glfwSetWindowUserPointer(window, this);
    glfwMakeContextCurrent(window);

    // must be after setting opengl context
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }

    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    // glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSwapInterval(1);

 
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
    glBindTexture(GL_TEXTURE_2D, textures[1]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_LUMINANCE16,
        depth.cols,
        depth.rows,
        0,
        GL_LUMINANCE,
        GL_UNSIGNED_SHORT,
        depth.ptr());
}

void draw_quads()
{
    // glColor3f(0.0f, 0.0f, 0.0f);
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

void WindowManager::set_system(fusion::System *system)
{
    this->system = system;
}

bool WindowManager::should_quit() const
{
    return glfwWindowShouldClose(window);
}

void WindowManager::render_scene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1, 1, 1, 1);
    glMatrixMode(GL_MODELVIEW);

    int separate_x = (int)((float)window_width / 3);

    glViewport(separate_x * 2, 0, separate_x, window_height / 2);
    draw_source_image();

    glViewport(0, 0, separate_x * 2, window_height);
    if (run_mode == 1)
        draw_rendered_scene();

    glViewport(separate_x * 2, window_height / 2, separate_x, window_height / 2);
    draw_input_depth();

    // glViewport(0, 0, window_width, window_height);
    // glColor3f(1, 0, 1);
    // glLineWidth(30);
    // glBegin(GL_POLYGON);
    // glVertex2f(-0.1, 0.1);
    // glVertex2f(-0.1, -0.1);
    // glVertex2f(0.1, -0.1);
    // glVertex2f(0.1, 0.1);
    // glEnd();

    glfwSwapBuffers(window);
    glfwPollEvents();
}
