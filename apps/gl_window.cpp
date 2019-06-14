#include "gl_window.h"

#define ENTER_KEY 13

MainWindow::~MainWindow()
{
    pangolin::DestroyWindow(WindowName);
}

MainWindow::MainWindow(const char *name, size_t width, size_t height)
    : mbFlagRestart(false), WindowName(name), mbFlagUpdateMesh(false),
      VERTEX_COUNT(0), MAX_VERTEX_COUNT(20000000)
{
    ResetAllFlags();

    pangolin::CreateWindowAndBind(WindowName, width, height);

    SetupGLFlags();
    SetupDisplays();
    RegisterKeyCallback();
    InitTextures();
    InitMeshBuffers();
}

void MainWindow::SetupGLFlags()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void MainWindow::InitTextures()
{
    TextureRGB.Reinitialise(
        640, 480,
        GL_RGB,
        true,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        NULL);

    TextureDepth.Reinitialise(
        640, 480,
        GL_RGBA,
        true,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        NULL);

    TextureScene.Reinitialise(
        640, 480,
        GL_RGBA,
        true,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        NULL);
}

void MainWindow::InitMeshBuffers()
{
    auto size = sizeof(float) * MAX_VERTEX_COUNT;

    BufferVertex.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    BufferNormal.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    BufferColour.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    MappedVertex = std::make_shared<pangolin::CudaScopedMappedPtr>(BufferVertex);
    MappedNormal = std::make_shared<pangolin::CudaScopedMappedPtr>(BufferNormal);
    MappedColour = std::make_shared<pangolin::CudaScopedMappedPtr>(BufferColour);

    glGenVertexArrays(1, &VAOShade);
    glBindVertexArray(VAOShade);

    BufferVertex.Bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    BufferNormal.Bind();
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    glGenVertexArrays(1, &VAOColour);
    glBindVertexArray(VAOColour);

    BufferVertex.Bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    BufferColour.Bind();
    glVertexAttribPointer(2, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
    glEnableVertexAttribArray(2);

    BufferColour.Unbind();
    glBindVertexArray(0);
}

bool MainWindow::IsPaused()
{
    return *BoxPaused;
}

void MainWindow::InitGlSlPrograms()
{
    ShadingProg.AddShaderFromFile(
        pangolin::GlSlShaderType::GlSlVertexShader,
        "./shaders/phong_vertex.shader");

    ShadingProg.AddShaderFromFile(
        pangolin::GlSlShaderType::GlSlFragmentShader,
        "./shaders/fragment.shader");

    ShadingProg.Link();
}

void MainWindow::SetupDisplays()
{
    CameraView = std::make_shared<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAt(1, 0.5, -2, 0, 0, 0, pangolin::AxisY));

    auto MenuDividerLeft = pangolin::Attach::Pix(200);
    float RightSideBarDividerLeft = 0.7f;

    pangolin::CreatePanel("Menu").SetBounds(0, 1, 0, MenuDividerLeft);

    BtnReset = std::make_shared<pangolin::Var<bool>>("Menu.RESET", false, false);
    BoxPaused = std::make_shared<pangolin::Var<bool>>("Menu.PAUSE", true, true);
    BoxDisplayImage = std::make_shared<pangolin::Var<bool>>("Menu.Display Image", true, true);
    BoxDisplayDepth = std::make_shared<pangolin::Var<bool>>("Menu.Display Depth", true, true);
    BoxDisplayScene = std::make_shared<pangolin::Var<bool>>("Menu.Display Scene", true, true);
    BoxDisplayMesh = std::make_shared<pangolin::Var<bool>>("Menu.Display Mesh", false, true);
    BoxDisplayCamera = std::make_shared<pangolin::Var<bool>>("Menu.Display Camera", false, true);

    mpViewSideBar = &pangolin::Display("Right Side Bar");
    mpViewSideBar->SetBounds(0, 1, RightSideBarDividerLeft, 1);
    mpViewRGB = &pangolin::Display("RGB");
    mpViewRGB->SetBounds(0, 0.5, 0, 1);
    mpViewDepth = &pangolin::Display("Depth");
    mpViewDepth->SetBounds(0.5, 1, 0, 1);
    mpViewScene = &pangolin::Display("Scene");
    mpViewScene->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft);
    mpViewMesh = &pangolin::Display("Mesh");
    mpViewMesh->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft).SetHandler(new pangolin::Handler3D(*CameraView));

    mpViewSideBar->AddDisplay(*mpViewRGB);
    mpViewSideBar->AddDisplay(*mpViewDepth);
}

void MainWindow::RegisterKeyCallback()
{
    //! r: retart the system
    pangolin::RegisterKeyPressCallback('r', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    pangolin::RegisterKeyPressCallback('R', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    //! Enter Key: pause / unpause the system
    pangolin::RegisterKeyPressCallback(ENTER_KEY, pangolin::ToggleVarFunctor("Menu.PAUSE"));
}

void MainWindow::ResetAllFlags()
{
    mbFlagRestart = false;
    mbFlagUpdateMesh = false;
}

void MainWindow::SetRGBSource(cv::Mat RgbImage)
{
    TextureRGB.Upload(RgbImage.data, GL_RGB, GL_UNSIGNED_BYTE);
}

void MainWindow::SetDepthSource(cv::Mat DepthImage)
{
    TextureDepth.Upload(DepthImage.data, GL_RGBA, GL_UNSIGNED_BYTE);
}

void MainWindow::SetRenderScene(cv::Mat SceneImage)
{
    TextureScene.Upload(SceneImage.data, GL_RGBA, GL_UNSIGNED_BYTE);
}

void MainWindow::Render()
{
    // ResetAllFlags();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.f, 0.f, 0.f, 1.f);

    if (pangolin::Pushed(*BtnReset))
        mbFlagRestart = true;

    if (*BoxDisplayImage)
    {
        mpViewRGB->Activate();
        TextureRGB.RenderToViewportFlipY();
    }

    if (*BoxDisplayDepth)
    {
        mpViewDepth->Activate();
        TextureDepth.RenderToViewportFlipY();
    }

    if (!IsPaused())
    {
        mpViewScene->Activate();
        TextureScene.RenderToViewportFlipY();
    }
    else
    {
        mpViewMesh->Activate(*CameraView);
        DrawMeshShaded();

        Eigen::Matrix3f K;
        K << 580, 0, 320, 0, 580, 240, 0, 0, 1;

        if (*BoxDisplayCamera)
            pangolin::glDrawFrustum(K.inverse().eval(), 640, 480, CameraPose, 0.1f);
    }

    pangolin::FinishFrame();
}

void MainWindow::DrawMeshShaded()
{
    if (VERTEX_COUNT == 0)
        return;
    std::cout << VERTEX_COUNT << std::endl;

    ShadingProg.Bind();
    glBindVertexArray(VAOShade);

    glDrawArrays(GL_TRIANGLES, 0, VERTEX_COUNT);

    glBindVertexArray(0);
    ShadingProg.Unbind();
}

void MainWindow::SetCurrentCamera(Eigen::Matrix4f T)
{
    CameraPose = T;
}

void MainWindow::DrawMeshColoured()
{
}

void MainWindow::DrawMeshNormalMapped()
{
}

float3 *MainWindow::GetMappedVertexBuffer()
{
    return (float3 *)**MappedVertex;
}

float3 *MainWindow::GetMappedNormalBuffer()
{
    return (float3 *)**MappedVertex;
}

uchar3 *MainWindow::GetMappedColourBuffer()
{
    return (uchar3 *)**MappedVertex;
}
