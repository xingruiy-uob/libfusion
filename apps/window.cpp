#include "window.h"
#include "cuda_utils.h"
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
    bool left_pressed = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_RELEASE);
    bool right_pressed = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_RELEASE);

    if (!left_pressed && !right_pressed)
    {
        return;
    }

    void *data = glfwGetWindowUserPointer(window);
    WindowManager *wm = static_cast<WindowManager *>(data);

    // handle translation
    if (right_pressed && xpos - wm->prev_mouse_pos[0] < -5)
    {
        wm->position[0] += 0.01f;
        wm->lookat[0] += 0.01f;
    }

    if (right_pressed && xpos - wm->prev_mouse_pos[0] > 5)
    {
        wm->position[0] -= 0.01f;
        wm->lookat[0] -= 0.01f;
    }

    if (right_pressed && ypos - wm->prev_mouse_pos[1] < -5)
    {
        wm->position[1] += 0.01f;
        wm->lookat[1] += 0.01f;
    }

    if (right_pressed && ypos - wm->prev_mouse_pos[1] > 5)
    {
        wm->position[1] -= 0.01f;
        wm->lookat[1] -= 0.01f;
    }

    // handle rotation
    if (left_pressed && xpos - wm->prev_mouse_pos[0] < -2)
    {
        wm->model_matrix = glm::rotate_slow(wm->model_matrix, glm::radians(3.f), glm::vec3(0, 1, 0));
    }

    if (left_pressed && xpos - wm->prev_mouse_pos[0] > 2)
    {
        wm->model_matrix = glm::rotate_slow(wm->model_matrix, glm::radians(3.f), glm::vec3(0, -1, 0));
    }

    if (left_pressed && ypos - wm->prev_mouse_pos[1] < -2)
    {
        wm->model_matrix = glm::rotate_slow(wm->model_matrix, glm::radians(3.f), glm::vec3(1, 0, 0));
    }

    if (left_pressed && ypos - wm->prev_mouse_pos[1] > 2)
    {
        wm->model_matrix = glm::rotate_slow(wm->model_matrix, glm::radians(3.f), glm::vec3(-1, 0, 0));
    }

    wm->prev_mouse_pos[0] = xpos;
    wm->prev_mouse_pos[1] = ypos;
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    void *data = glfwGetWindowUserPointer(window);
    WindowManager *wm = static_cast<WindowManager *>(data);

    if ((button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) && action == GLFW_PRESS)
    {
        glfwGetCursorPos(window, &wm->prev_mouse_pos[0], &wm->prev_mouse_pos[1]);
    }
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    void *data = glfwGetWindowUserPointer(window);
    WindowManager *wm = static_cast<WindowManager *>(data);

    if (yoffset > 0)
    {
        wm->position[2] += 0.1f;
        wm->lookat[2] += 0.f;
    }
    else
    {
        wm->position[2] += -0.1f;
        wm->lookat[2] += -0.f;
    }
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

GLuint load_shader_from_file(const char *file_name, int type)
{
    std::ifstream file(file_name, std::ifstream::in);

    if (file.is_open())
    {
        GLuint shader = glCreateShader(type);

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string source_str = buffer.str();
        const GLchar *source = source_str.c_str();
        GLint source_size = source_str.size();

        glShaderSource(shader, 1, &source, &source_size);
        glCompileShader(shader);

        // error handling
        GLint code;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &code);
        if (code == GL_TRUE)
            return shader;
        else
        {
            GLint log_size;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_size);
            GLchar *log = new GLchar[log_size];
            glGetShaderInfoLog(shader, log_size, NULL, log);
            std::cout << "shaders failed loading with message: \n"
                      << log << std::endl;
        }
    }

    return 0;
}

GLuint create_program_from_shaders(GLuint *shaders, GLint size)
{
    GLuint program = glCreateProgram();
    for (int i = 0; i < size; ++i)
        glAttachShader(program, shaders[i]);
    glLinkProgram(program);

    GLint code = GL_TRUE;
    glGetProgramiv(program, GL_LINK_STATUS, &code);
    if (code == GL_TRUE)
    {
        std::cout << "program linked!" << std::endl;
        return program;
    }
    else
    {
        GLint log_size = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_size);
        GLchar *log = new GLchar[log_size];
        glGetProgramInfoLog(program, log_size, NULL, log);
        std::cout << "link program failed with error: \n"
                  << log << std::endl;
        return 0;
    }
}

GLuint create_buffer_and_bind(size_t size)
{
    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_STATIC_DRAW);
    return buffer;
}

bool WindowManager::initialize_gl_context(const size_t width, const int height)
{
    window_width = width;
    window_height = height;
    num_mesh_triangles = 0;
    view_matrix = Eigen::Matrix4f::Identity();
    model_matrix = glm::mat4(1.f); // default to identity matrix
    position[0] = 0;
    position[1] = 0;
    position[2] = -1;
    lookat[0] = 0;
    lookat[1] = 0;
    lookat[2] = 0;

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
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSwapInterval(1);

    // initialize textures
    glGenTextures(3, &textures[0]);

    shaders[0] = load_shader_from_file("./shaders/phong_vertex.shader", GL_VERTEX_SHADER);
    shaders[1] = load_shader_from_file("./shaders/fragment.shader", GL_FRAGMENT_SHADER);
    program[0] = create_program_from_shaders(&shaders[0], 2);

    // create array object
    glGenVertexArrays(1, &gl_array[0]);
    glBindVertexArray(gl_array[0]);

    // create buffer object
    size_t size = sizeof(GLfloat) * 50000000 * 3;
    buffers[0] = create_buffer_and_bind(size);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    buffers[1] = create_buffer_and_bind(size);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // bind buffer object to CUDA
    safe_call(cudaGraphicsGLRegisterBuffer(&buffer_res[0], buffers[0], cudaGraphicsMapFlagsWriteDiscard));
    safe_call(cudaGraphicsGLRegisterBuffer(&buffer_res[1], buffers[1], cudaGraphicsMapFlagsWriteDiscard));

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

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

float3 *WindowManager::get_cuda_mapped_ptr(int id)
{
    float3 *dev_ptr;
    safe_call(cudaGraphicsMapResources(1, &buffer_res[id]));
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&dev_ptr, &num_bytes, buffer_res[id]);
    return dev_ptr;
}

void WindowManager::cuda_unmap_resources(int id)
{
    safe_call(cudaGraphicsUnmapResources(1, &buffer_res[id]));
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

glm::mat4 WindowManager::get_view_projection_matrix(Eigen::Matrix4f eigen_view_matrix)
{
    glm::mat4 view_matrix = glm::lookAt(
        glm::vec3(position[0], position[1], position[2]),
        glm::vec3(lookat[0], lookat[1], lookat[2]),
        glm::vec3(0, -1, 0));

    glm::mat4 projection_matrix = glm::perspective(glm::radians(45.f), 4.0f / 3.0f, 0.1f, 1000.f);
    return projection_matrix * view_matrix * model_matrix;
}

void WindowManager::render_scene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.f, 0.f, 0.f, 1.f);

    int separate_x = (int)((float)window_width / 3);
    glViewport(separate_x * 2, 0, separate_x, window_height / 2);
    draw_source_image();

    glViewport(0, 0, separate_x * 2, window_height);
    if (run_mode == 1)
        draw_rendered_scene();
    else if (num_mesh_triangles != 0)
    {
        glUseProgram(program[0]);
        glBindVertexArray(gl_array[0]);
        glm::mat4 mvp_mat = get_view_projection_matrix(view_matrix);

        GLint loc = glGetUniformLocation(program[0], "mvp_matrix");
        glUniformMatrix4fv(loc, 1, GL_FALSE, &mvp_mat[0][0]);
        glDrawArrays(GL_TRIANGLES, 0, num_mesh_triangles * 3);
        glUseProgram(0);
    }

    glViewport(separate_x * 2, window_height / 2, separate_x, window_height / 2);
    draw_input_depth();

    glfwSwapBuffers(window);
    glfwPollEvents();
}
