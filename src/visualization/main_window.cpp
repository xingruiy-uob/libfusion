#include "visualization/main_window.h"

#define ENTER_KEY 13

namespace fusion
{

MainWindow::~MainWindow()
{
    pangolin::DestroyWindow(window_title);
    std::cout << "GUI Released." << std::endl;
}

MainWindow::MainWindow(
    const char *name,
    const size_t width,
    const size_t height)
    : window_title(name),
      VERTEX_COUNT(0),
      MAX_VERTEX_COUNT(50000000)
{
    pangolin::CreateWindowAndBind(window_title, width, height);

    setup_gl_flags();
    setup_displays();
    register_key_callbacks();
    init_textures();
    init_mesh_buffers();
    init_glsl_programs();

    std::cout << "GUI Ready." << std::endl;
}

void MainWindow::setup_gl_flags()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void MainWindow::init_textures()
{
    image_tex.Reinitialise(
        640, 480,
        GL_RGB,
        true,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        NULL);

    depth_tex.Reinitialise(
        640, 480,
        GL_RGBA,
        true,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        NULL);

    scene_tex.Reinitialise(
        640, 480,
        GL_RGBA,
        true,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        NULL);
}

void MainWindow::init_mesh_buffers()
{
    auto size = sizeof(float) * 3 * MAX_VERTEX_COUNT;

    vertex_buffer.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    normal_buffer.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    colour_buffer.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    vertex_mapped = std::make_shared<pangolin::CudaScopedMappedPtr>(vertex_buffer);
    normal_mapped = std::make_shared<pangolin::CudaScopedMappedPtr>(normal_buffer);
    colour_mapped = std::make_shared<pangolin::CudaScopedMappedPtr>(colour_buffer);

    glGenVertexArrays(1, &vao_shaded);
    glBindVertexArray(vao_shaded);

    vertex_buffer.Bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    normal_buffer.Bind();
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    glGenVertexArrays(1, &vao_colour);
    glBindVertexArray(vao_colour);

    vertex_buffer.Bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    colour_buffer.Bind();
    glVertexAttribPointer(2, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
    glEnableVertexAttribArray(2);

    colour_buffer.Unbind();
    glBindVertexArray(0);
}

bool MainWindow::is_paused() const
{
    return *BoxPaused;
}

void MainWindow::init_glsl_programs()
{
    phong_shader.AddShaderFromFile(pangolin::GlSlShaderType::GlSlVertexShader, "./glsl_shader/phong.vert");
    phong_shader.AddShaderFromFile(pangolin::GlSlShaderType::GlSlFragmentShader, "./glsl_shader/direct_output.frag");
    phong_shader.Link();
}

void MainWindow::setup_displays()
{
    camera_view = std::make_shared<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAtRDF(0, 0, 0, 0, 0, -1, 0, 1, 0));

    auto MenuDividerLeft = pangolin::Attach::Pix(200);
    float RightSideBarDividerLeft = 0.7f;

    pangolin::CreatePanel("Menu").SetBounds(0, 1, 0, MenuDividerLeft);

    BtnReset = std::make_shared<pangolin::Var<bool>>("Menu.RESET", false, false);
    BtnSaveMap = std::make_shared<pangolin::Var<bool>>("Menu.Save Map", false, false);
    BtnSetLost = std::make_shared<pangolin::Var<bool>>("Menu.Set Lost", false, false);
    BtnReadMap = std::make_shared<pangolin::Var<bool>>("Menu.Read Map", false, false);
    BoxPaused = std::make_shared<pangolin::Var<bool>>("Menu.PAUSE", true, true);
    BoxDisplayImage = std::make_shared<pangolin::Var<bool>>("Menu.Display Image", true, true);
    BoxDisplayDepth = std::make_shared<pangolin::Var<bool>>("Menu.Display Depth", true, true);
    BoxDisplayScene = std::make_shared<pangolin::Var<bool>>("Menu.Display Scene", true, true);
    BoxDisplayMesh = std::make_shared<pangolin::Var<bool>>("Menu.Display Mesh", true, true);
    BoxDisplayCamera = std::make_shared<pangolin::Var<bool>>("Menu.Display Camera", false, true);
    BoxDisplayKeyCameras = std::make_shared<pangolin::Var<bool>>("Menu.Display KeyFrame", false, true);
    BoxDisplayKeyPoint = std::make_shared<pangolin::Var<bool>>("Menu.Display KeyPoint", false, true);

    right_side_view = &pangolin::Display("Right Side Bar");
    right_side_view->SetBounds(0, 1, RightSideBarDividerLeft, 1);
    image_view = &pangolin::Display("RGB");
    image_view->SetBounds(0, 0.5, 0, 1);
    depth_view = &pangolin::Display("Depth");
    depth_view->SetBounds(0.5, 1, 0, 1);
    scene_view = &pangolin::Display("Scene");
    scene_view->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft);
    mesh_view = &pangolin::Display("Mesh");
    mesh_view->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft).SetHandler(new pangolin::Handler3D(*camera_view));

    right_side_view->AddDisplay(*image_view);
    right_side_view->AddDisplay(*depth_view);
}

void MainWindow::register_key_callbacks()
{
    //! Retart the system
    pangolin::RegisterKeyPressCallback('r', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    pangolin::RegisterKeyPressCallback('R', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    //! Pause / Resume the system
    pangolin::RegisterKeyPressCallback(ENTER_KEY, pangolin::ToggleVarFunctor("Menu.PAUSE"));
    //! Display keyframes
    pangolin::RegisterKeyPressCallback('c', pangolin::ToggleVarFunctor("Menu.Display KeyFrame"));
    pangolin::RegisterKeyPressCallback('C', pangolin::ToggleVarFunctor("Menu.Display KeyFrame"));
    //! Save Maps
    pangolin::RegisterKeyPressCallback('s', pangolin::SetVarFunctor<bool>("Menu.Save Map", true));
    pangolin::RegisterKeyPressCallback('S', pangolin::SetVarFunctor<bool>("Menu.Save Map", true));
    //! Load Maps
    pangolin::RegisterKeyPressCallback('l', pangolin::SetVarFunctor<bool>("Menu.Read Map", true));
    pangolin::RegisterKeyPressCallback('L', pangolin::SetVarFunctor<bool>("Menu.Read Map", true));
}

void MainWindow::set_image_src(const cv::Mat image, const ImageType type)
{
    switch (type)
    {
    case RGB:
        image_tex.Upload(image.data, GL_RGB, GL_UNSIGNED_BYTE);
        break;
    case DEPTH:
        depth_tex.Upload(image.data, GL_RGBA, GL_UNSIGNED_BYTE);
        break;
    case SCENE:
        scene_tex.Upload(image.data, GL_RGBA, GL_UNSIGNED_BYTE);
        break;
    }
}

void MainWindow::render()
{
    // Clear frame buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.f, 0.f, 0.f, 1.f);

    check_buttons();
    draw_image();
    draw_depth();
    // draw_scene();
    draw_camera();
    draw_mesh_phong_shaded();

    // Swap frame buffers
    pangolin::FinishFrame();
}

void MainWindow::draw_image()
{
    // Draw Rgb image to screen
    if (*BoxDisplayImage)
    {
        image_view->Activate();
        image_tex.RenderToViewportFlipY();
    }
}

void MainWindow::draw_depth()
{
    // Draw depth to screen
    if (*BoxDisplayDepth)
    {
        depth_view->Activate();
        depth_tex.RenderToViewportFlipY();
    }
}

void MainWindow::draw_scene()
{
    if (!is_paused())
    {
        scene_view->Activate();
        scene_tex.RenderToViewportFlipY();
    }
}

void MainWindow::check_buttons()
{
    if (pangolin::Pushed(*BtnReset))
    {
        slam->reset();
        VERTEX_COUNT = 0;
    }
}

void MainWindow::draw_camera()
{
    if (*BoxDisplayCamera)
    {
        mesh_view->Activate();
        pangolin::glDrawFrustum(slam->get_intrinsics(), 640, 480, slam->get_current_pose(), 0.05f);
    }
}

void MainWindow::update_vertex_and_normal()
{
    auto *vertex = get_vertex_buffer_mapped();
    auto *normal = get_normal_buffer_mapped();
    VERTEX_COUNT = slam->fetch_mesh_with_normal(vertex, normal);
}

void MainWindow::draw_mesh_phong_shaded()
{
    if (VERTEX_COUNT == 0)
        return;

    mesh_view->Activate();

    phong_shader.Bind();
    glBindVertexArray(vao_shaded);

    phong_shader.SetUniform("mvp_matrix", camera_view->GetProjectionModelViewMatrix());

    glDrawArrays(GL_TRIANGLES, 0, VERTEX_COUNT * 3);

    glBindVertexArray(0);
    phong_shader.Unbind();
}

void MainWindow::draw_mesh_colour_mapped()
{
}

void MainWindow::draw_mesh_normal_mapped()
{
}

void MainWindow::set_current_camera(const Eigen::Matrix4f &T)
{
    curren_pose = T;
}

void MainWindow::set_system(fusion::SystemNew *const sys)
{
    slam = sys;
}

float *MainWindow::get_vertex_buffer_mapped()
{
    return (float *)**vertex_mapped;
}

float *MainWindow::get_normal_buffer_mapped()
{
    return (float *)**normal_mapped;
}

unsigned char *MainWindow::get_colour_buffer_mapped()
{
    return (unsigned char *)**colour_mapped;
}

} // namespace fusion